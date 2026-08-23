"""`ParamSpec`: the layout of a flat parameter vector, and how to read it.

A spec says which named quantities a model estimates, in what order, on which link, and
which of them are allowed to differ between conditions. It is the piece the three source
repositories each reinvented, and the piece that makes one engine serve all of them.

Two coding schemes for a condition effect, because the repositories genuinely disagree and
both are defensible:

* `"free"` -- two independent parameters, `q_con` and `q_inc`. Lets the data put them in
  either order.
* `"sum"` -- a base `q` and a non-negative increment `q_diff`, with `q_inc = q + q_diff`.
  Makes an ordering true by construction (`b_accuracy > b_speed` in a speed/accuracy
  manipulation) and keeps every parameter positive and log-transformable.

Two references for the within-trial noise, which is where the repositories differ in a way
that is *not* just naming (see `eamax.design.map.accumulator_noise`).

Links: `"log"` for strictly positive quantities, `"identity"` for signed ones. The link is
part of the parameterization -- it is inseparable from which quantities are differences --
so the spec owns `constrain` and `log_det_jacobian`. It does not own any prior: the three
consumers disagree scientifically about priors, and there is no shared implementation
underneath to extract.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp

from .effects import IDENTITY_QTYPES


@dataclass(frozen=True)
class ParamSpec:
    """Layout of one subject's flat parameter vector.

    Attributes
    ----------
    names : tuple of str
        Parameter names in vector order.
    links : tuple of str
        ``"log"`` or ``"identity"`` per entry.
    quantities : dict
        ``{quantity: (index_con, index_inc)}``. The two indices are equal when the quantity
        has no condition effect.
    derived : dict
        ``{quantity: (terms_con, terms_inc)}``, where each ``terms`` is a tuple of
        ``(parameter_name, coefficient)`` pairs summed to give the quantity. This is what
        lets a spec keep parameter names its consumers already depend on while still
        speaking the average/difference language internally -- an intercept/slope
        parameterization is exactly ``V = v_intercept + v_slope/2``, ``v_d = v_slope``. A
        quantity mentioned in neither ``derived``, ``fixed`` nor ``quantities`` is zero.
    coding : dict
        ``{quantity: "free" | "sum"}`` for quantities that have a condition effect.
    fixed : dict
        ``{quantity: value}`` for quantities held at a constant rather than estimated.
    response_offsets : tuple of int
        Indices of the effect-coded per-response threshold offsets. There are
        ``num_responses - 1`` of them; the last is derived so they sum to zero.
    num_responses : int
        Number of accumulators the spec is built for.
    noise_reference : {"average", "mismatch"}
        See :func:`eamax.design.map.accumulator_noise`.
    noise_scale : float
        The noise value held fixed under ``noise_reference="mismatch"``.
    threshold_is_gap : bool
        When True the threshold quantity is the *gap* above the start point, so the absolute
        boundary is ``A + gap``. This is the LBA convention, which keeps ``b > A`` true by
        construction.
    num_centered : int
        Number of trailing parameters a hierarchical prior should treat as centered. Carried
        through unchanged; :mod:`eamax.design` does not interpret it.
    hyperparameters : dict
        Optional per-parameter prior arrays, keyed by whatever the consumer's prior wants
        (``mu_loc``, ``mu_scale``, ...). Carried, never interpreted.

    Raises
    ------
    ValueError
        If ``names`` and ``links`` differ in length, a link is unrecognised, or
        ``noise_reference`` is not one of the two supported values.
    """

    names: tuple[str, ...]
    links: tuple[str, ...]
    quantities: dict[str, tuple[int, int]] = field(default_factory=dict)
    derived: dict[str, tuple] = field(default_factory=dict)
    coding: dict[str, str] = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)
    response_offsets: tuple[int, ...] = ()
    num_responses: int = 2
    noise_reference: str = "average"
    noise_scale: float = 1.0
    threshold_is_gap: bool = False
    num_centered: int = 0
    hyperparameters: dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.names) != len(self.links):
            raise ValueError(f"{len(self.names)} names but {len(self.links)} links")
        bad = set(self.links) - {"log", "identity"}
        if bad:
            raise ValueError(f"Unknown link(s) {sorted(bad)}; expected 'log' or 'identity'")
        if self.noise_reference not in {"average", "mismatch"}:
            raise ValueError(
                f"noise_reference must be 'average' or 'mismatch', got {self.noise_reference!r}"
            )

    @property
    def num_params(self):
        return len(self.names)

    @property
    def exp_mask(self):
        """Boolean array: which entries live on the log link."""
        return jnp.asarray([link == "log" for link in self.links])

    def index(self, name):
        return self.names.index(name)

    def constrain(self, theta):
        """Map an unconstrained vector to natural space, entry by entry on its own link."""
        theta = jnp.asarray(theta)
        return jnp.where(self.exp_mask, jnp.exp(theta), theta)

    def log_det_jacobian(self, theta):
        """Log determinant of `constrain`'s Jacobian.

        For the exp link this is ``sum(theta)`` over the log-linked entries; the identity
        link contributes nothing. Belongs with whichever term the consumer treats as the
        prior.

        Parameters
        ----------
        theta : array
            Unconstrained parameter vector.

        Returns
        -------
        array
            Scalar log determinant.
        """
        theta = jnp.asarray(theta)
        return jnp.sum(jnp.where(self.exp_mask, theta, 0.0))

    def hyperparameter_array(self, key):
        """One prior hyperparameter as an array aligned with `names`."""
        return jnp.asarray(self.hyperparameters[key])


class ParamSpecBuilder:
    """Incrementally assembles a `ParamSpec`.

    Only generic bookkeeping lives here -- appending names, tracking condition-effect pairs,
    tracking per-response offsets. The *ordering* of quantities is model-specific and
    deliberately left to the caller, because a hierarchical prior may need particular
    parameters to land in a trailing block (see `ParamSpec.num_centered`).
    """

    def __init__(self, identity_qtypes=IDENTITY_QTYPES):
        self.names = []
        self.qtype = []
        self.identity_qtypes = frozenset(identity_qtypes)
        self.quantities = {}
        self.coding = {}
        self.fixed = {}
        self.response_offsets = []

    def add(self, name, qtype):
        """Append one named parameter.

        Returns
        -------
        int
            The new parameter's index.
        """
        index = len(self.names)
        self.names.append(name)
        self.qtype.append(qtype)
        return index

    def add_quantity(self, q, effects=(), coding="free"):
        """Add a shared parameter, or a condition-varying pair if `q` is in `effects`.

        Under ``coding="sum"`` the second index points at an *increment*, not at the
        incongruent value itself -- :mod:`eamax.design.map` knows the difference from
        ``spec.coding``.

        Parameters
        ----------
        q : str
            Quantity name.
        effects : container of str, optional
            Quantities allowed to differ between conditions.
        coding : {"free", "sum"}, optional
            How a condition effect is parameterized.

        Returns
        -------
        tuple of int
            ``(index_con, index_inc)``, equal when ``q`` has no condition effect.

        Raises
        ------
        ValueError
            If ``coding`` is not one of the two supported values.
        """
        if coding not in {"free", "sum"}:
            raise ValueError(f"coding must be 'free' or 'sum', got {coding!r}")
        if q in effects:
            if coding == "sum":
                i_con = self.add(q, q)
                i_inc = self.add(f"{q}_diff", q)
            else:
                i_con = self.add(f"{q}_con", q)
                i_inc = self.add(f"{q}_inc", q)
            self.coding[q] = coding
        else:
            i_con = i_inc = self.add(q, q)
        self.quantities[q] = (i_con, i_inc)
        return i_con, i_inc

    def fix(self, q, value):
        """Hold a quantity at a constant instead of estimating it."""
        self.fixed[q] = float(value)
        self.quantities[q] = None

    def add_response_offsets(self, num_responses):
        """Add `num_responses - 1` effect-coded per-response threshold offsets.

        A condition-independent motor bias: each accumulator gets its own baseline
        threshold, constrained to average to the shared threshold. The last offset is
        derived rather than estimated, so the constraint holds exactly.

        Parameters
        ----------
        num_responses : int
            Number of accumulators.

        Returns
        -------
        list of int
            Indices of the free offsets.
        """
        self.response_offsets = [self.add(f"c_{r}", "c") for r in range(1, num_responses)]
        return self.response_offsets

    def finalize(self, num_responses, param_priors=None, num_centered=0, **kwargs):
        """Assemble the `ParamSpec`.

        Parameters
        ----------
        num_responses : int
            Number of accumulators.
        param_priors : dict, optional
            ``{qtype: tuple}`` of prior hyperparameters, expanded to per-parameter arrays
            and carried on the spec without interpretation.
        num_centered : int, optional
            Trailing parameters a hierarchical prior should centre.
        **kwargs
            Forwarded to :class:`ParamSpec` (``noise_reference``, ``threshold_is_gap``, ...).

        Returns
        -------
        ParamSpec
        """
        hyperparameters = {}
        if param_priors:
            width = len(next(iter(param_priors.values())))
            keys = kwargs.pop("hyperparameter_names", None) or [
                f"hyper_{i}" for i in range(width)
            ]
            for position, key in enumerate(keys):
                hyperparameters[key] = tuple(param_priors[q][position] for q in self.qtype)

        return ParamSpec(
            names=tuple(self.names),
            links=tuple(
                "identity" if q in self.identity_qtypes else "log" for q in self.qtype
            ),
            quantities=dict(self.quantities),
            coding=dict(self.coding),
            fixed=dict(self.fixed),
            response_offsets=tuple(self.response_offsets),
            num_responses=num_responses,
            num_centered=num_centered,
            hyperparameters=hyperparameters,
            **kwargs,
        )
