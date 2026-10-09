"""Conditional normalising flow used as a neural likelihood for accumulators without one.

The flow is amortised over *parameters*, not over datasets: it learns the first-passage
density of a **single** accumulator conditioned on that accumulator's parameters, and the
race is assembled around it by `eamax.race` exactly as it would be around a closed form.

The generative direction is

    Z ~ N(0, 1)  --rational-quadratic spline-->  --exp-->  T

so the support is `(0, inf)` by construction and the transform is strictly increasing. That
monotonicity is what makes the race likelihood possible: `{T > t}` and `{Z > z}` are the
same event, so the survival function is `S(t) = sf_{N(0,1)}(z)` with
`z = bijector.inverse(t)` -- exact *given* the flow, not a second approximation stacked on
top of it. A race can therefore multiply a flow density by a flow survival without
compounding error.

`t0` is deliberately not a conditioning variable. Training data are generated with `t0 = 0`,
so the flow models decision times only and the race evaluates it at `rt - t0`. That drops a
dimension from the conditioning set and makes `t0` exactly a location shift -- which is what
`eamax.race` assumes of it anyway.

`FlowAccumulator` is generic over the conditioning set rather than tied to one model: the
Wald flow conditions on `(v, s, b)` and the pulsed flow on `(v, amp, tau, s, b)`, and the
only difference is `context_names`. Making that explicit and required also closes a real
hazard -- nothing in an Orbax checkpoint records the context layout, so a *reordered*
context of the same width would not raise, it would silently produce wrong densities.

**Affine stage.** The plain spline is active on a fixed `[-5, 5]` window of log decision
time shared by every context, but one context occupies only a few units of it while the
medians of different contexts spread over most of it. The spline then spends its bins
reaching the corners -- on a trained pulsed-Wald conditioner two of twelve bins carried
98.6% of the base mass, visible as kinks at the knots. With ``affine=True`` the conditioner
emits a per-context location and scale as well, and the flow becomes

    Z ~ N(0, 1)  --spline-->  Y  --loc(ctx) + scale(ctx) * Y-->  log T  --exp-->  T

so the spline models only the *shape* of a standardised log decision time. Every stage is
strictly increasing (``scale > 0``), so the survival function stays exact. The two extra
outputs are appended *after* the spline parameters, giving width ``3K + 3`` against the plain
``3K + 1``; the widths never coincide, so the layout is read off the weights.
``scale = softplus(raw + c) + MIN_SCALE`` with ``c`` chosen so ``raw = 0`` gives exactly 1:
zeroed affine outputs reproduce the plain flow. An affine flow may also narrow the spline's
range or learn its boundary slopes, since the range is then relative to each context; for a
plain flow the range is absolute log time, where narrowing it forbids fast decision times.

**Log-scaled inputs** (affine only). The conditioner can see each input as
``(log(x + eps) - loc) / scale`` instead of ``x``. A diffusion's time and space rescaling
symmetries are multiplicative on the raw scale and become shifts in log space, where the
affine location is linear along them. ``loc`` and ``scale`` are the mean and SD of
``log(x + eps)`` under the uniform training box, in closed form (:func:`log_input_scaling`),
so they are fixed by the box rather than learned.

**Depth.** ``num_hidden > 1`` builds a :class:`DeepMLP` with further GELU layers between
``linear1`` and ``linear2``; one hidden layer is :class:`MLP` itself, so existing checkpoints
keep their structure.

**Training box.** A conditioner can record the box it was trained on (``context_bounds``),
and :class:`FlowAccumulator` then clips its inputs to that box. Outside the box the flow
extrapolates, and its log-density develops gradient spikes and flat plateaus that collapse
step-size adaptation and freeze an SMC cloud. Clipped, a stray particle sees the density at
the box edge -- bounded, continuous, with zero gradient along the clipped direction, so the
prior alone pulls it back. Inside the box the likelihood is bit-identical.

The spline settings, input scaling, depth and box are plain attributes on the conditioner --
part of its graph, not its weights -- so the checkpoint sidecar records them and
:func:`eamax.flows.load_conditioner` checks them: the same weights under a different setting
are a different density.
"""

import math

import distrax
import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy import stats

# The spline's active range in the *latent* Gaussian coordinate; outside it the bijector is
# the identity. Five standard deviations covers the base distribution to ~6e-7 per tail, so
# the identity region is only reached by extreme decision times. Narrowing it truncates the
# left tail and biases `t0` upward.
RANGE_MIN = -5.0
RANGE_MAX = 5.0

DEFAULT_BOUNDARY_SLOPES = "identity"
_BOUNDARY_SLOPES = ("identity", "unconstrained", "lower_identity", "upper_identity")

#: Floor on the affine stage's per-context scale, so it can never collapse to a point mass.
MIN_SCALE = 1e-3

# log(expm1(1 - MIN_SCALE)) rounded to float32, so softplus(_SCALE_OFFSET) + MIN_SCALE is 1 to
# ~3e-8 and a zero raw output is the identity to that precision. Fixed as a literal because it
# is part of what a trained affine checkpoint means: the conditioners trained so far used this
# float32-rounded value, and the exact float64 one would shift their densities by ~1e-7.
_SCALE_OFFSET = 0.5397424697875977


class MLP(nnx.Module):
    """Two-layer GELU perceptron mapping a context vector to spline knots.

    The class name and the `linear1` / `linear2` attribute names are part of the on-disk
    checkpoint format -- Orbax restores by structure -- so they must not be renamed.
    """

    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        return self.linear2(nnx.gelu(self.linear1(x)))


class DeepMLP(MLP):
    """:class:`MLP` with ``num_hidden - 1`` extra ``dmid -> dmid`` GELU layers.

    ``linear1`` and ``linear2`` keep their roles and names -- input and output layer -- and the
    extra layers are ``linear_mid1``, ``linear_mid2``, ... in order.
    """

    def __init__(self, din: int, dmid: int, dout: int, num_hidden: int, *, rngs: nnx.Rngs):
        super().__init__(din, dmid, dout, rngs=rngs)
        self.num_hidden = int(num_hidden)
        for index in range(1, self.num_hidden):
            setattr(self, f"linear_mid{index}", nnx.Linear(dmid, dmid, rngs=rngs))

    def __call__(self, x):
        x = nnx.gelu(self.linear1(x))
        for index in range(1, self.num_hidden):
            x = nnx.gelu(getattr(self, f"linear_mid{index}")(x))
        return self.linear2(x)


def _validate_bounds(bounds, num_in):
    bounds = tuple((float(low), float(high)) for low, high in bounds)
    if len(bounds) != num_in:
        raise ValueError(f"context_bounds needs {num_in} (low, high) pairs, got {len(bounds)}.")
    for low, high in bounds:
        if not low < high:
            raise ValueError(f"context_bounds interval ({low}, {high}) is empty.")
    return bounds


def make_mlp_conditioner(
    rngs, num_in, num_mid=64, num_bins=4, *, affine=False, spline_range=RANGE_MAX,
    boundary_slopes=DEFAULT_BOUNDARY_SLOPES, input_scaling=None, num_hidden=1,
    context_bounds=None,
):
    """Build the conditioner MLP for a `num_bins`-knot spline.

    The output width is ``3 * num_bins + 1``: one width, one height and one derivative per
    bin, plus the final knot derivative -- the parameterisation
    ``distrax.RationalQuadraticSpline`` expects. An affine conditioner appends a location and
    a raw scale, for ``3 * num_bins + 3``.

    Parameters
    ----------
    rngs : flax.nnx.Rngs
        Supplies the ``default`` stream for initialisation.
    num_in : int
        Number of conditioning variables.
    num_mid : int, optional
        Hidden width.
    num_bins : int, optional
        Number of spline bins.
    affine : bool, optional
        Add the per-context affine stage on log decision time; see the module docstring.
    spline_range : float, optional
        The spline is active on ``[-spline_range, spline_range]``. Affine only.
    boundary_slopes : str, optional
        `distrax` boundary condition; ``"unconstrained"`` learns both end slopes. Affine
        only.
    input_scaling : dict, optional
        ``{"eps", "loc", "scale"}``, each a length-``num_in`` sequence in context order, as
        returned by :func:`log_input_scaling`. Affine only.
    num_hidden : int, optional
        Hidden layers. ``1`` is :class:`MLP`; more gives a :class:`DeepMLP`.
    context_bounds : sequence of (float, float), optional
        The box the conditioner is trained on, one ``(low, high)`` per input in context
        order. Recorded, not enforced here: :class:`FlowAccumulator` clips to it.

    Returns
    -------
    MLP
    """
    spline_range = float(spline_range)
    boundary_slopes = str(boundary_slopes)
    num_hidden = int(num_hidden)
    if num_hidden < 1:
        raise ValueError(f"num_hidden must be at least 1, got {num_hidden}.")
    if boundary_slopes not in _BOUNDARY_SLOPES:
        raise ValueError(f"boundary_slopes must be one of {_BOUNDARY_SLOPES}, got {boundary_slopes!r}.")
    if not spline_range > 0.0:
        raise ValueError(f"spline_range must be positive, got {spline_range}.")
    if not affine:
        if (spline_range, boundary_slopes) != (RANGE_MAX, DEFAULT_BOUNDARY_SLOPES):
            raise ValueError(
                "spline_range and boundary_slopes can only be changed for an affine flow: without "
                "the per-context location and scale the range is absolute log time, and "
                "narrowing it forbids fast decision times."
            )
        if input_scaling is not None:
            raise ValueError("Log-scaled inputs are only implemented for an affine flow.")

    dout = 3 * num_bins + (3 if affine else 1)
    if num_hidden == 1:
        conditioner = MLP(din=num_in, dmid=num_mid, dout=dout, rngs=rngs)
    else:
        conditioner = DeepMLP(din=num_in, dmid=num_mid, dout=dout, num_hidden=num_hidden, rngs=rngs)

    if affine:
        conditioner.spline_range = spline_range
        conditioner.boundary_slopes = boundary_slopes
    if input_scaling is not None:
        columns = {key: tuple(float(x) for x in input_scaling[key]) for key in ("eps", "loc", "scale")}
        if any(len(column) != num_in for column in columns.values()):
            raise ValueError(f"input_scaling needs {num_in} entries per key, got {columns}.")
        if not all(sd > 0.0 for sd in columns["scale"]):
            raise ValueError(f"input_scaling scales must be positive, got {columns['scale']}.")
        conditioner.input_log_eps = columns["eps"]
        conditioner.input_loc = columns["loc"]
        conditioner.input_scale = columns["scale"]
    if context_bounds is not None:
        conditioner.context_bounds = _validate_bounds(context_bounds, num_in)
    return conditioner


def num_hidden_layers(conditioner):
    """Hidden layers in a conditioner: 1 for :class:`MLP`."""
    return int(getattr(conditioner, "num_hidden", 1))


def spline_settings(conditioner):
    """``(spline_range, boundary_slopes)`` of a conditioner; the defaults for one without them."""
    return (
        float(getattr(conditioner, "spline_range", RANGE_MAX)),
        str(getattr(conditioner, "boundary_slopes", DEFAULT_BOUNDARY_SLOPES)),
    )


def input_scaling(conditioner):
    """The conditioner's log-input scaling as ``{"eps", "loc", "scale"}`` tuples, or ``None``."""
    if getattr(conditioner, "input_log_eps", None) is None:
        return None
    return {
        "eps": tuple(conditioner.input_log_eps),
        "loc": tuple(conditioner.input_loc),
        "scale": tuple(conditioner.input_scale),
    }


def context_bounds(conditioner):
    """The recorded training box, one ``(low, high)`` per input in context order, or ``None``."""
    bounds = getattr(conditioner, "context_bounds", None)
    return None if bounds is None else tuple(tuple(pair) for pair in bounds)


def _layout_from_width(width):
    if width % 3 == 1:
        return (width - 1) // 3, False
    if width % 3 == 0:
        return (width - 3) // 3, True
    raise ValueError(f"Conditioner output width {width} is neither 3K+1 (plain) nor 3K+3 (affine).")


def conditioner_layout(conditioner):
    """``(num_bins, affine)`` of a conditioner, read off its output layer."""
    return _layout_from_width(int(conditioner.linear2.out_features))


def _log_uniform_moments(low, high, eps):
    """Mean and SD of ``log(x + eps)`` for ``x ~ Uniform(low, high)``, in closed form.

    With ``y = x + eps`` uniform on ``[a, b]``: ``E[log y] = [F]_a^b / (b - a)`` with
    ``F(y) = y log y - y``, and ``E[log^2 y] = [G]_a^b / (b - a)`` with
    ``G(y) = y (log^2 y - 2 log y + 2)``.
    """
    a, b = low + eps, high + eps
    if not (a > 0.0 and b > a):
        raise ValueError(
            f"log input scaling needs 0 < low + eps < high + eps, got low={low}, high={high}, eps={eps}."
        )

    def F(y):
        return y * math.log(y) - y

    def G(y):
        return y * (math.log(y) ** 2 - 2.0 * math.log(y) + 2.0)

    mean = (F(b) - F(a)) / (b - a)
    var = (G(b) - G(a)) / (b - a) - mean**2
    return mean, math.sqrt(max(var, 0.0))


def log_input_scaling(context_names, eps, bounds):
    """Fixed log-input scaling for a conditioner trained on a uniform box.

    Parameters
    ----------
    context_names : sequence of str
        The flow's inputs, in training order.
    eps : mapping of str to float
        Offset added before the log, per input. Must name exactly ``context_names``.
    bounds : mapping of str to (low, high)
        The uniform training box per input.

    Returns
    -------
    dict
        ``{"eps", "loc", "scale"}``, tuples in ``context_names`` order -- the
        ``input_scaling`` argument of :func:`make_mlp_conditioner`.
    """
    names = [str(name) for name in context_names]
    eps = {str(k): float(v) for k, v in eps.items()}
    if set(eps) != set(names):
        raise ValueError(f"eps must name exactly {names}, got {sorted(eps)}.")
    columns = {"eps": [], "loc": [], "scale": []}
    for name in names:
        low, high = (float(limit) for limit in bounds[name])
        mean, sd = _log_uniform_moments(low, high, eps[name])
        columns["eps"].append(eps[name])
        columns["loc"].append(mean)
        columns["scale"].append(sd)
    return {key: tuple(value) for key, value in columns.items()}


def _conditioner_outputs(conditioner, context):
    """The conditioner applied to `context`, through its log-input scaling if it has one.

    Every path from a context to spline parameters goes through here, so training and
    inference cannot see differently scaled inputs.
    """
    scaling = input_scaling(conditioner)
    if scaling is None:
        return conditioner(context)
    context = jnp.asarray(context)
    eps, loc, scale = (jnp.asarray(scaling[key], dtype=context.dtype) for key in ("eps", "loc", "scale"))
    return conditioner((jnp.log(context + eps) - loc) / scale)


def _spline(spline_params, spline_range=RANGE_MAX, boundary_slopes=DEFAULT_BOUNDARY_SLOPES):
    return distrax.RationalQuadraticSpline(
        spline_params,
        range_min=-spline_range,
        range_max=spline_range,
        boundary_slopes=boundary_slopes,
        min_bin_size=1e-4,
    )


def _exponential():
    return distrax.Lambda(
        forward=jnp.exp,
        inverse=jnp.log,
        forward_log_det_jacobian=lambda z: z,
        inverse_log_det_jacobian=lambda x: -jnp.log(x),
        event_ndims_in=0,
        event_ndims_out=0,
    )


def _loc_scale(params):
    return params[..., -2], jax.nn.softplus(params[..., -1] + _SCALE_OFFSET) + MIN_SCALE


def spline_flow(data, context, conditioner):
    """Build the conditional flow for `context` and score `data` under it.

    Parameters
    ----------
    data : array
        Decision times. Must broadcast against the batch shape ``context`` implies.
    context : array
        Shape ``(num_in,)`` for one parameter vector shared across all of ``data``, or
        ``(N, num_in)`` for one per observation.
    conditioner : MLP
        From :func:`make_mlp_conditioner`. Plain or affine, read off its output width.

    Returns
    -------
    log_prob : array
        Log density of ``data`` under the flow.
    flow : distrax.Transformed
        Returned so callers can reach the bijector for the survival function; ``log_prob``
        is often discarded and eliminated by jit. Its batch shape is the context's, so
        ``flow.sample`` draws one independent decision time per context row.
    """
    params = _conditioner_outputs(conditioner, context)
    num_bins, affine = _layout_from_width(params.shape[-1])
    if affine:
        loc, scale = _loc_scale(params)
        spline = _spline(params[..., : 3 * num_bins + 1], *spline_settings(conditioner))
        bijector = distrax.Chain([_exponential(), distrax.ScalarAffine(shift=loc, scale=scale), spline])
    else:
        bijector = distrax.Chain([_exponential(), _spline(params)])
    # The base carries the batch shape the context implies. A scalar base would make
    # `flow.sample` draw a single `z` and push it through every context in the batch, so
    # every draw sharing a context would be the same number.
    batch_shape = params.shape[:-1]
    base = distrax.Normal(
        loc=jnp.zeros(batch_shape, dtype=params.dtype), scale=jnp.ones(batch_shape, dtype=params.dtype)
    )
    flow = distrax.Transformed(base, bijector)
    return flow.log_prob(data), flow


def spline_knots(conditioner, context):
    """The spline's knots for one context: base ``z`` and the log decision time they map to.

    The knots are where the transform's curvature, and so the density's slope, can jump. A
    plain flow's knots sit at fixed log times; an affine flow's move with the context
    (``log t = loc + scale * y``). Bin sizes include the spline's ``min_bin_size``, as
    `distrax` builds them.

    Parameters
    ----------
    conditioner : MLP
    context : array
        Shape ``(num_in,)``.

    Returns
    -------
    z_knots, log_t_knots : array
        Shape ``(num_bins + 1,)`` each.
    """
    params = _conditioner_outputs(conditioner, jnp.asarray(context))
    num_bins, affine = _layout_from_width(params.shape[-1])
    spline = _spline(params[..., : 3 * num_bins + 1], *spline_settings(conditioner))
    log_t = spline.y_pos
    if affine:
        loc, scale = _loc_scale(params)
        log_t = loc + scale * log_t
    return spline.x_pos, log_t


def evaluate_pdf_sf(conditioner, data, context):
    """The trained flow's log density and log survival at `data`.

    The inference-time entry point. The survival is the standard-normal one evaluated at the
    flow inverse, which is exact given the flow -- see the module docstring.

    Parameters
    ----------
    conditioner : MLP
        A trained conditioner.
    data : array
        Decision times ``rt - t0``, already floored away from zero by the caller. Values at
        or below zero leave the flow's support.
    context : array
        Shape ``(num_in,)`` or ``(N, num_in)``, in natural (non-log) space.

    Returns
    -------
    log_pdf, log_sf : array
        Broadcast to the shape of ``data``.
    """
    conditioner.eval()
    _, flow = spline_flow(jnp.squeeze(data), context, conditioner)
    return flow.log_prob(data), stats.norm.logsf(flow.bijector.inverse(data))


def _bounds_by_name(context_bounds, context_names):
    """`context_bounds` as ``{name: (low, high)}`` floats, checked against `context_names`.

    Accepts any mapping of name to a two-element sequence, so a config object works as-is. A
    name the flow is not conditioned on is an error rather than ignored: it is almost always
    a typo, and a silently absent clamp is exactly the failure the clamp exists to prevent.
    """
    bounds = {}
    for key, limits in context_bounds.items():
        name = str(key)
        if name not in context_names:
            raise ValueError(
                f"context_bounds names {name!r}, which is not a flow input; "
                f"expected a subset of {list(context_names)}."
            )
        low, high = (float(limit) for limit in limits)
        if not low < high:
            raise ValueError(f"context_bounds[{name!r}] = ({low}, {high}) is empty.")
        bounds[name] = (low, high)
    return bounds


class FlowAccumulator:
    """An `eamax` accumulator whose density comes from a trained conditional flow.

    Density only: pair it with a simulator for the same model -- typically
    `SimulatedPulsedWald` -- and note that doing so gives up the "one object, one
    distribution" guarantee `eamax.simulate` otherwise provides, since draws and densities
    then come from different implementations.

    Parameters
    ----------
    conditioner : MLP
        A trained conditioner.
    context_names : sequence of str
        The parameters, **in the order the flow was trained on**, that make up its
        conditioning vector. Required rather than inferred: an Orbax checkpoint records
        neither the names nor the order, so a mismatched ordering of the right width fails
        silently rather than loudly.
    transform : dict of str to callable, optional
        Applied to a parameter before it enters the context. The pulsed flow trains on
        ``|amp|`` because the pulse's sign selects *which* accumulator carries it rather
        than changing its shape, so ``{"amp": jnp.abs}`` is the usual value.
    dtype : dtype, optional
        Conditioner compute dtype. Flows train in float32; the surrounding likelihood
        usually runs in float64, so the context is cast down and the result cast back.
    remat : bool, optional
        Wrap the forward pass in ``jax.checkpoint``, trading recomputation for activation
        memory. Worth it inside a hierarchical likelihood over many subjects.
    context_bounds : mapping of str to (float, float), optional
        The training box, for some or all inputs; each named input is clipped to it after
        `transform`. Defaults to the box the conditioner records, if any. Given both, they
        must agree. With neither, inputs are not clipped.

    Notes
    -----
    The conditioner is an MLP evaluated once per context row, and its activations are what
    a gradient keeps -- per particle and per subject inside a hierarchical sampler. So the
    accumulator sets ``broadcasts_params``: :func:`eamax.design.build_params_fn` then hands
    it trial-invariant quantities as ``(N, 1)`` rather than ``(N, T)``, the MLP runs ``N``
    times instead of ``N * T``, and the spline broadcasts over the ``(T,)`` decision times.

    A quantity is trial-invariant when the design makes it so: a covariate that is the same
    on every trial -- ``target`` in an accuracy-coded dataset, say -- should be passed to
    :class:`~eamax.design.TrialDesign` as a scalar. As a ``(T,)`` column of equal values it
    still produces ``(N, T)`` parameters, and the MLP still runs per trial.

    Clipping to the training box keeps a sampler out of the region where the flow
    extrapolates (see the module docstring). Inside the box it changes nothing; outside, the
    density is the one at the box edge and its gradient along the clipped direction is zero,
    so the prior alone pulls a stray particle back. Only the flow's inputs are clipped.
    """

    #: Read by :func:`eamax.design.build_params_fn`; see Notes.
    broadcasts_params = True

    def __init__(self, conditioner, context_names, transform=None, dtype=jnp.float32, remat=False,
                 context_bounds=None):
        self.conditioner = conditioner
        self.context_names = tuple(context_names)
        self.transform = dict(transform or {})
        self.dtype = dtype
        self.context_bounds = self._resolve_bounds(context_bounds)

        def forward(data, context):
            return evaluate_pdf_sf(self.conditioner, data, context)

        self._forward = jax.checkpoint(forward) if remat else forward

    def _resolve_bounds(self, given):
        recorded = context_bounds(self.conditioner)
        if recorded is not None:
            if len(recorded) != len(self.context_names):
                raise ValueError(
                    f"The conditioner records {len(recorded)} context bounds but is used with "
                    f"{len(self.context_names)} context names."
                )
            recorded = dict(zip(self.context_names, recorded, strict=True))
        if given is None:
            return recorded or {}
        given = _bounds_by_name(given, self.context_names)
        if recorded is not None and any(recorded[name] != limits for name, limits in given.items()):
            raise ValueError(
                f"context_bounds {given} disagree with the box the conditioner was trained on, "
                f"{recorded}."
            )
        return given

    @property
    def param_names(self):
        return self.context_names

    def build_context(self, params):
        """Stack ``params`` into the flow's context, in training order.

        Each column is passed through its `transform`, then clipped to its `context_bounds`.

        Parameters
        ----------
        params : dict of array
            Keyed by ``context_names``.

        Returns
        -------
        array
            The parameters' broadcast shape plus a trailing ``num_in`` axis -- ``(N, T,
            num_in)``, or ``(N, 1, num_in)`` when every parameter is trial-invariant.
        """
        columns = []
        for name in self.context_names:
            value = jnp.asarray(params[name])
            transform = self.transform.get(name)
            if transform:
                value = transform(value)
            if name in self.context_bounds:
                value = jnp.clip(value, *self.context_bounds[name])
            columns.append(value)
        return jnp.stack(jnp.broadcast_arrays(*columns), axis=-1).astype(self.dtype)

    def log_pdf_sf(self, t, params):
        t = jnp.asarray(t)
        log_pdf, log_sf = self._forward(t.astype(self.dtype), self.build_context(params))
        return log_pdf.astype(t.dtype), log_sf.astype(t.dtype)

    def sample(self, key, params):
        context = self.build_context(params)
        _, flow = spline_flow(
            jnp.ones(jnp.shape(context)[:-1], dtype=self.dtype), context, self.conditioner
        )
        return flow.sample(seed=key)

    def __repr__(self):
        return f"FlowAccumulator(context_names={list(self.context_names)})"
