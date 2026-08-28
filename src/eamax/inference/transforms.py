"""Unconstraining transforms for `[*bounded_block, *positive_block]` parameter vectors.

A sampler explores unconstrained real space; a model's parameters are positive, or bounded
on both sides. Something has to map between them, and something has to supply the matching
log-det-Jacobian so the density is a density in the coordinates being explored.
:class:`BlockTransform` is one object owning the forward map, its inverse, and the Jacobian,
so the three stay in step.

The layout is the one the hierarchical models use: prior hyperparameters bounded on both
sides come first and take a `Sigmoid`, the strictly positive subject-level parameters
follow and take `Exp`. An empty bounded block is the ordinary all-positive case.

This module needs no sampler; it imports nothing from `_blackjax`.
"""

import jax.numpy as jnp

from .._tfp import tfb


class BlockTransform:
    """Maps unconstrained coordinates to natural ones for a two-block parameter vector.

    Directions follow the TFP convention, the same one :func:`eamax.hierarchical.default_bijector`
    uses: :meth:`forward` goes unconstrained -> natural, :meth:`inverse` goes back.

    Every method is vectorized over the trailing axis, so a whole
    ``(dataset, sample, param)`` block transforms in one call -- which is what the posterior
    read-back path needs.

    Parameters
    ----------
    lower, upper : sequence of float
        Bounds of the leading block, one pair per bounded entry. Both empty gives the
        all-positive case.

    Attributes
    ----------
    num_bounded : int
        Length of the leading bounded block.

    Raises
    ------
    ValueError
        If ``lower`` and ``upper`` differ in length, or any bound is not strictly
        increasing.
    """

    def __init__(self, lower=(), upper=()):
        lower = [float(value) for value in lower]
        upper = [float(value) for value in upper]

        if len(lower) != len(upper):
            raise ValueError(f"{len(lower)} lower bounds but {len(upper)} upper bounds")

        bad = [(low, high) for low, high in zip(lower, upper) if not low < high]
        if bad:
            raise ValueError(f"Each lower bound must be below its upper bound; got {bad}")

        self.lower = tuple(lower)
        self.upper = tuple(upper)
        self.num_bounded = len(lower)
        self._bijectors = [tfb().Sigmoid(low=low, high=high) for low, high in zip(lower, upper)]

    def forward(self, y):
        """Unconstrained -> natural scale.

        Parameters
        ----------
        y : array
            Unconstrained values, shape ``(..., P)``.

        Returns
        -------
        array
            Natural-scale values, shape ``(..., P)``.
        """
        y = jnp.asarray(y)
        if self.num_bounded == 0:
            return jnp.exp(y)

        bounded = jnp.stack(
            [bijector.forward(y[..., i]) for i, bijector in enumerate(self._bijectors)], axis=-1
        )
        return jnp.concatenate([bounded, jnp.exp(y[..., self.num_bounded :])], axis=-1)

    def inverse(self, x):
        """Natural scale -> unconstrained. Exact inverse of :meth:`forward`.

        Parameters
        ----------
        x : array
            Natural-scale values, shape ``(..., P)``.

        Returns
        -------
        array
            Unconstrained values, shape ``(..., P)``.
        """
        x = jnp.asarray(x)
        if self.num_bounded == 0:
            return jnp.log(x)

        bounded = jnp.stack(
            [bijector.inverse(x[..., i]) for i, bijector in enumerate(self._bijectors)], axis=-1
        )
        return jnp.concatenate([bounded, jnp.log(x[..., self.num_bounded :])], axis=-1)

    def log_det_jacobian(self, y):
        """Log determinant of :meth:`forward`'s Jacobian, summed over the parameter axis.

        Parameters
        ----------
        y : array
            Unconstrained values, shape ``(..., P)``.

        Returns
        -------
        array
            Shape ``(...)``; scalar for a 1-D input.
        """
        y = jnp.asarray(y)
        positive = jnp.sum(y[..., self.num_bounded :], axis=-1)

        if self.num_bounded == 0:
            return positive

        bounded = sum(
            bijector.forward_log_det_jacobian(y[..., i], event_ndims=0)
            for i, bijector in enumerate(self._bijectors)
        )
        return bounded + positive

    def midpoint(self):
        """Midpoints of the bounded block, shape ``(num_bounded,)``.

        A starting value for a bounded hyperparameter has to lie strictly inside its
        support -- outside it, :meth:`inverse` returns NaN and the whole fit is wasted.
        Deriving the midpoint from the same bounds the transform uses keeps the two in step.
        """
        return jnp.asarray([0.5 * (low + high) for low, high in zip(self.lower, self.upper)])

    def __repr__(self):
        if self.num_bounded == 0:
            return "BlockTransform(all positive)"
        return f"BlockTransform(lower={self.lower}, upper={self.upper})"


#: The all-positive transform, shared because it carries no state.
SIMPLE = BlockTransform()


def simple_to_unconstrained(position):
    """Natural scale -> unconstrained, for an all-positive parameter vector."""
    return SIMPLE.inverse(position)


def simple_to_constrained(position):
    """Unconstrained -> natural scale, for an all-positive parameter vector."""
    return SIMPLE.forward(position)
