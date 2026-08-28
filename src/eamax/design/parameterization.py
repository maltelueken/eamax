"""`Parameterization`: the layout of a flat parameter vector and how to read it.

A parameterization says which coefficients a model estimates, in what order, on which
transformation (link), and how each per-accumulator quantity is assembled from them as a sum
of contrast-weighted terms (see :mod:`eamax.design.contrasts`). It is the piece the three
source repositories each reinvented, and the piece that makes one engine serve all of them --
and, now, models they could not express at all, because a quantity is an open-ended linear
combination rather than a fixed `average + sgn * difference / 2`.

Two ideas replace the previous `ParamSpec`'s dozen special-purpose fields:

* **Coefficients carry a bijector.** The link is inseparable from the parameterization --
  it is which quantities are strictly positive and which are signed differences -- so each
  `Free` coefficient owns a TensorFlow Probability bijector -- :func:`eamax.design.log` for a
  positive quantity, :func:`eamax.design.identity` for a signed one, or anything else (see
  :mod:`eamax.design.links`). `constrain` and
  `log_det_jacobian` compose over them, so the design layer owns the change of variables but
  no prior: the three consumers disagree scientifically about priors, and there is no shared
  implementation underneath to extract.

* **The vector order is stated explicitly.** ``names``, ``index`` and the trailing
  ``num_centered`` block a hierarchical prior centres are all read off ``order``, which is
  given independently of the quantities -- a single coefficient may appear in several
  quantities, and a hierarchical prior may need particular coefficients last, so neither the
  quantity order nor an alphabetical rule can be trusted to place them.

The object is static configuration used at trace time to build ``params_fn``; it holds
Python bijector objects and contrast closures and is never itself a traced argument. Only the
parameter vector ``theta`` is traced.
"""

from dataclasses import dataclass, field
from typing import Union

import jax.numpy as jnp

from .links import bijector_for_link, link_name


@dataclass(frozen=True)
class Free:
    """An estimated coefficient: one slot in the parameter vector.

    Attributes
    ----------
    name : str
        The coefficient's name, and its label in ``names``/``index``. These are external
        contracts (config keys, saved coordinate labels), so presets keep the names their
        consumers already depend on.
    bijector : tfp bijector
        The coefficient's link: maps the unconstrained real value to the natural scale. Use
        :func:`eamax.design.log` for a strictly positive quantity and
        :func:`eamax.design.identity` for a signed difference (or any TFP bijector directly).
    """

    name: str
    bijector: object


@dataclass(frozen=True)
class Const:
    """A compile-time constant coefficient. Never occupies a slot in the parameter vector."""

    value: float


@dataclass(frozen=True)
class Term:
    """One coefficient times a product of contrast columns.

    Attributes
    ----------
    coefficient : Free or Const
        The estimated or constant multiplier.
    contrasts : tuple of Contrast
        Multiplied together to form the term's design column; empty means a bare coefficient.
    """

    coefficient: Union[Free, Const]
    contrasts: tuple = ()


@dataclass(frozen=True)
class Quantity:
    """One accumulator parameter, assembled as a sum of terms.

    Attributes
    ----------
    name : str
        The accumulator ``param_names`` entry this produces (``"v"``, ``"s"``, ``"b"``,
        ``"A"``, ``"amp"``, ``"tau"``), or ``"t0"`` for the non-decision time.
    terms : tuple of Term
        Summed to give the quantity's per-trial, per-accumulator value.
    transform : callable, optional
        A function of the *other* quantities, ``transform(others) -> value``, that rewrites
        this quantity as an expression of them. ``others`` maps each quantity name to its
        assembled (pre-transform) value, so a quantity can be defined relative to another --
        the LBA threshold is ``transform=lambda q: q["A"] + q["b"]``, the gap ``b`` above the
        start point ``A``, which keeps ``b > A`` true by construction. When ``None`` the
        assembled sum of ``terms`` is used directly.
    """

    name: str
    terms: tuple = ()
    transform: object = None


def coef(name, bijector):
    """A free (estimated) coefficient named `name` on the link `bijector`."""
    return Free(name, bijector)


def constant(value):
    """A constant coefficient held at `value`, never estimated."""
    return Const(float(value))


def term(coefficient, *contrasts):
    """One term: `coefficient` times the product of `contrasts`."""
    return Term(coefficient, tuple(contrasts))


def quantity(name, *terms, transform=None):
    """One accumulator quantity `name`, the sum of `terms`.

    Pass ``transform`` to express the quantity as a function of the other quantities instead
    of using the assembled sum directly; see :class:`Quantity`.
    """
    return Quantity(name, tuple(terms), transform)


@dataclass(frozen=True)
class Parameterization:
    """Layout of one subject's flat parameter vector, and how to read it.

    Attributes
    ----------
    order : tuple of Free
        The parameter vector's coefficients, in vector order.
    quantities : tuple of Quantity
        One per accumulator parameter the bound accumulator needs, plus optionally ``t0``.
    num_responses : int
        Number of accumulators the parameterization is built for.
    num_centered : int
        Number of trailing coefficients a hierarchical prior should treat as centered.
        Carried through unchanged; :mod:`eamax.design` does not interpret it.
    first_response : int
        Index of the first accumulator in the response/target coding.
    hyperparameters : dict
        Optional per-coefficient prior arrays, keyed by whatever the consumer's prior wants
        (``mu_loc``, ``mu_scale``, ...). Carried, never interpreted.

    Raises
    ------
    ValueError
        If a coefficient name is repeated in ``order``, or a `Free` referenced by a term is
        absent from ``order``.
    """

    order: tuple
    quantities: tuple
    num_responses: int = 2
    num_centered: int = 0
    first_response: int = 1
    hyperparameters: dict = field(default_factory=dict)

    def __post_init__(self):
        names = [c.name for c in self.order]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Repeated coefficient name(s) in order: {dupes}")
        known = set(names)
        for q in self.quantities:
            for t in q.terms:
                if isinstance(t.coefficient, Free) and t.coefficient.name not in known:
                    raise ValueError(
                        f"Quantity {q.name!r} uses coefficient {t.coefficient.name!r}, "
                        f"which is not in order {names}"
                    )

    # -- layout -------------------------------------------------------------------------- #
    @property
    def names(self):
        """Coefficient names in vector order."""
        return tuple(c.name for c in self.order)

    @property
    def num_params(self):
        return len(self.order)

    def index(self, name):
        return self.names.index(name)

    # -- links / change of variables ----------------------------------------------------- #
    @property
    def links(self):
        """The coefficients' link names (``"log"``/``"identity"``), in vector order."""
        return tuple(link_name(c.bijector) for c in self.order)

    def constrain(self, theta):
        """Map an unconstrained vector to natural space, coefficient by coefficient.

        Batch-safe: the last axis is the parameter axis, so an input of shape ``(..., P)``
        keeps its leading axes.
        """
        theta = jnp.asarray(theta)
        cols = [c.bijector.forward(theta[..., i]) for i, c in enumerate(self.order)]
        return jnp.stack(cols, axis=-1)

    def to_natural(self, theta):
        """Alias for :meth:`constrain`, read as "report these on the natural scale"."""
        return self.constrain(theta)

    def log_det_jacobian(self, theta):
        """Log determinant of `constrain`'s Jacobian, summed over coefficients.

        Each coefficient is a scalar transform, so its ``forward_log_det_jacobian`` is called
        with ``event_ndims=0``; for ``Exp`` this contributes ``theta`` and for ``Identity``
        nothing, reproducing the old closed form. Belongs with whichever term the consumer
        treats as the prior.
        """
        theta = jnp.asarray(theta)
        total = jnp.zeros(jnp.shape(theta)[:-1])
        for i, c in enumerate(self.order):
            total = total + c.bijector.forward_log_det_jacobian(theta[..., i], event_ndims=0)
        return total

    def hyperparameter_array(self, key):
        """One prior hyperparameter as an array aligned with `names`."""
        return jnp.asarray(self.hyperparameters[key])

    @classmethod
    def of_names(cls, names, links):
        """A bare layout from parallel ``names`` and ``"log"``/``"identity"`` links.

        No quantities -- for consumers that only need ``names`` and ``constrain`` /
        ``to_natural`` (posterior reporting and back-transforms), not the accumulator map.
        """
        order = tuple(Free(name, bijector_for_link(link)) for name, link in zip(names, links))
        return cls(order=order, quantities=(), num_responses=1)


def parameterization(order, quantities, num_responses, *, num_centered=0, first_response=1,
                     hyperparameters=None):
    """Assemble a `Parameterization` from an explicit coefficient order and its quantities.

    Parameters
    ----------
    order : sequence of Free
        The parameter vector's layout, in vector order. Load-bearing: ``names``, ``index``
        and the trailing ``num_centered`` block are read from it, so the ``t0``-last
        convention several call sites depend on, and a hierarchical prior's centered block,
        are expressed here rather than inferred.
    quantities : sequence of Quantity
        One per accumulator parameter the bound accumulator will need, plus optionally
        ``t0``. Every `Free` a term references must appear exactly once in ``order``.
    num_responses : int
        Number of accumulators.
    num_centered : int, optional
        Trailing coefficients a hierarchical prior should centre.
    first_response : int, optional
        Index of the first accumulator in the response/target coding.
    hyperparameters : dict, optional
        Per-coefficient prior arrays, carried on the parameterization without interpretation.

    Returns
    -------
    Parameterization
    """
    return Parameterization(
        order=tuple(order),
        quantities=tuple(quantities),
        num_responses=num_responses,
        num_centered=num_centered,
        first_response=first_response,
        hyperparameters=dict(hyperparameters or {}),
    )
