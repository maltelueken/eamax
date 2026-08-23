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
"""

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


def make_mlp_conditioner(rngs, num_in, num_mid=64, num_bins=4):
    """Build the conditioner MLP for a `num_bins`-knot spline.

    The output width is ``3 * num_bins + 1``: one width, one height and one derivative per
    bin, plus the final knot derivative -- the parameterisation
    ``distrax.RationalQuadraticSpline`` expects.

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

    Returns
    -------
    MLP
    """
    return MLP(din=num_in, dmid=num_mid, dout=num_bins * 3 + 1, rngs=rngs)


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
        From :func:`make_mlp_conditioner`.

    Returns
    -------
    log_prob : array
        Log density of ``data`` under the flow.
    flow : distrax.Transformed
        Returned so callers can reach the bijector for the survival function; ``log_prob``
        is often discarded and eliminated by jit.
    """
    spline = distrax.RationalQuadraticSpline(
        conditioner(context),
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
        boundary_slopes="identity",
        min_bin_size=1e-4,
    )
    exponential = distrax.Lambda(
        forward=jnp.exp,
        inverse=jnp.log,
        forward_log_det_jacobian=lambda z: z,
        inverse_log_det_jacobian=lambda x: -jnp.log(x),
        event_ndims_in=0,
        event_ndims_out=0,
    )
    flow = distrax.Transformed(
        distrax.Normal(loc=0.0, scale=1.0), distrax.Chain([exponential, spline])
    )
    return flow.log_prob(data), flow


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


class FlowAccumulator:
    """An `eamax` accumulator whose density comes from a trained conditional flow.

    Density only: pair it with a simulator for the same model -- typically
    `EulerMaruyamaPulsedWald` -- and note that doing so gives up the "one object, one
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
    """

    def __init__(self, conditioner, context_names, transform=None, dtype=jnp.float32, remat=False):
        self.conditioner = conditioner
        self.context_names = tuple(context_names)
        self.transform = dict(transform or {})
        self.dtype = dtype

        def forward(data, context):
            return evaluate_pdf_sf(self.conditioner, data, context)

        self._forward = jax.checkpoint(forward) if remat else forward

    @property
    def param_names(self):
        return self.context_names

    def build_context(self, params):
        """Stack ``params`` into the flow's context, in training order.

        Parameters
        ----------
        params : dict of array
            Keyed by ``context_names``.

        Returns
        -------
        array
            Shape ``(T, num_in)``.
        """
        columns = []
        for name in self.context_names:
            value = jnp.asarray(params[name])
            transform = self.transform.get(name)
            columns.append(transform(value) if transform else value)
        return jnp.stack(columns, axis=-1).astype(self.dtype)

    def log_pdf_sf(self, t, params):
        t = jnp.asarray(t)
        log_pdf, log_sf = self._forward(t.astype(self.dtype), self.build_context(params))
        return log_pdf.astype(t.dtype), log_sf.astype(t.dtype)

    def sample(self, key, params):
        _, flow = spline_flow(
            jnp.ones(jnp.shape(next(iter(params.values()))), dtype=self.dtype),
            self.build_context(params),
            self.conditioner,
        )
        return flow.sample(seed=key)

    def __repr__(self):
        return f"FlowAccumulator(context_names={list(self.context_names)})"
