"""Contrasts: per-trial, per-accumulator design columns.

A `Contrast` is one column of a design matrix, evaluated lazily from the accumulator index
and a `TrialDesign`. The parameterization engine multiplies each column by an estimated (or
constant) coefficient and sums the terms, so *every* per-accumulator quantity -- drift,
noise, threshold, the conflict pulse -- is one linear model in disguise:

    q[n, t] = sum_k  ( product of that term's contrast columns )  *  coefficient_k

Writing quantities this way makes several modelling choices ordinary linear combinations.
The average/difference split (`v = V + sgn * v_d / 2`) is `intercept()*V + match(0.5)*v_d`.
Both within-trial noise conventions become linear: the "average" one is
`intercept()*S + match(0.5)*s_d`, and the "mismatch" one -- pinning the non-target
accumulator's noise to a constant -- is `constant(scale)*nontarget() + s_true*target()`.
The sum-to-zero per-accumulator threshold offsets are deviation coding over the accumulator
index rather than over a condition level.

Everything a contrast does is a JAX op on covariates -- selections are `jnp.where`, never a
Python `if` on a traced value -- so a column keeps a static shape under `jit` and `vmap`.
The *structure* (which contrasts, how many terms) is fixed in Python before tracing; only
the parameter vector is traced.

Each contrast receives ``accum``, the ``(N, 1)`` integer column of accumulator indices
(``design.first_response + arange(num_responses)``), and the ``design`` itself, and returns
something broadcastable to ``(N, T)``.
"""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class Contrast:
    """One design column, plus a static label for readable specs and errors.

    Attributes
    ----------
    fn : callable
        ``fn(accum, design) -> array``. ``accum`` is the ``(N, 1)`` accumulator-index
        column; the return is broadcastable to ``(N, T)``.
    label : str
        Human-readable name, e.g. ``"match(0.5)"``. Never used in arithmetic.
    """

    fn: Callable
    label: str

    def __mul__(self, other):
        """Interaction: the column-wise product of two contrasts.

        ``match(0.5) * condition(1)`` is the difference term active only in the reference
        condition -- exactly the interaction a per-condition difference needs.
        """
        if not isinstance(other, Contrast):
            return NotImplemented
        return Contrast(
            lambda accum, design: self.fn(accum, design) * other.fn(accum, design),
            f"{self.label}*{other.label}",
        )


def intercept(scale=1.0):
    """A constant column: the average over accumulators.

    Paired with `match`, ``intercept()*Q + match(0.5)*q_d`` is the average/difference
    decomposition, with ``Q`` the mean and ``q_d`` the target-minus-others difference.
    """
    return Contrast(lambda accum, design: jnp.asarray(scale), f"intercept({scale})")


def match(scale=1.0):
    """The signed target-match column: ``+scale`` on the target accumulator, ``-scale`` else.

    This is the ``sgn`` that splits every average into a match and a mismatch value. With
    ``scale=0.5`` a coefficient ``q_d`` adds ``+q_d/2`` to the matching accumulator and
    ``-q_d/2`` to the others, so ``q_d`` is the full match-minus-mismatch difference.
    """

    def fn(accum, design):
        return scale * jnp.where(accum == jnp.asarray(design.target)[None, :], 1.0, -1.0)

    return Contrast(fn, f"match({scale})")


def target(scale=1.0):
    """Indicator column: ``scale`` on the target accumulator, ``0`` elsewhere.

    With `nontarget`, ``constant(c)*nontarget() + q*target()`` pins the non-target
    accumulators' quantity to ``c`` and frees the target's to ``q`` -- the "mismatch"
    identification, where a parameter named ``s_true`` means the target accumulator's noise.
    """

    def fn(accum, design):
        return scale * jnp.where(accum == jnp.asarray(design.target)[None, :], 1.0, 0.0)

    return Contrast(fn, f"target({scale})")


def nontarget(scale=1.0):
    """Indicator column: ``scale`` on every non-target accumulator, ``0`` on the target."""

    def fn(accum, design):
        return scale * jnp.where(accum != jnp.asarray(design.target)[None, :], 1.0, 0.0)

    return Contrast(fn, f"nontarget({scale})")


def distractor(scale=1.0):
    """Indicator column: ``scale`` on the accumulator the distractor selects, ``0`` else.

    In a conflict task an irrelevant stimulus feature drives an early pulse on exactly one
    accumulator (see :mod:`eamax.accumulators.pulse`). ``term(amp, distractor())`` routes the
    pulse amplitude there and leaves it zero everywhere else.
    """

    def fn(accum, design):
        return scale * jnp.where(accum == jnp.asarray(design.distractor)[None, :], 1.0, 0.0)

    return Contrast(fn, f"distractor({scale})")


def condition(level, scale=1.0):
    """Indicator column for one condition level: ``scale`` where ``condition == level``.

    ``condition(1)`` is the reference level (congruent, or speed-instructed). A free
    condition effect is two indicators, ``term(Q_con, condition(1)) + term(Q_inc,
    condition(0))``, which selects ``Q_con`` on reference trials and ``Q_inc`` on the rest.
    """

    def fn(accum, design):
        return scale * jnp.where(jnp.asarray(design.condition)[None, :] == level, 1.0, 0.0)

    return Contrast(fn, f"condition({level},{scale})")


def contrast_column(weights_by_level):
    """A general condition contrast: gather one weight per condition *level*.

    ``design.condition`` is read as an integer level index ``0..L-1`` and used to gather from
    ``weights_by_level``. This is the escape hatch to arbitrary coding schemes -- dummy,
    deviation, Helmert, orthogonal polynomial -- one `contrast_column` (and one coefficient)
    per column of the contrast matrix, where the two-level `condition` indicators are only
    the simplest case.
    """
    weights = jnp.asarray(weights_by_level)

    def fn(accum, design):
        return weights[jnp.asarray(design.condition).astype(int)][None, :]

    return Contrast(fn, f"contrast_column({list(weights_by_level)})")


def response_offset(r, num_responses):
    """Deviation-coded column for the ``r``-th per-accumulator threshold offset.

    A condition-independent motor bias: each accumulator gets its own baseline threshold,
    constrained to average to the shared threshold. There are ``num_responses - 1`` free
    offsets (``r = 1 .. num_responses - 1``); this column is ``+1`` on accumulator ``r``,
    ``-1`` on the last accumulator, and ``0`` elsewhere. Summed over the free coefficients,
    the last accumulator carries minus their sum, so the per-accumulator offsets average to
    zero exactly -- a bias rather than a second threshold parameter.

    The comparison is against the accumulator *position* ``accum - design.first_response``,
    so the offsets follow the accumulators regardless of the response coding's origin.
    """

    def fn(accum, design):
        position = accum - design.first_response  # (N, 1), 0-based
        plus = jnp.where(position == (r - 1), 1.0, 0.0)
        minus = jnp.where(position == (num_responses - 1), -1.0, 0.0)
        return plus + minus

    return Contrast(fn, f"response_offset({r})")


def constant_column(value=1.0):
    """A bare constant column, ``value`` on every accumulator and trial.

    A synonym for `intercept` kept for symmetry with `constant` coefficients; use whichever
    reads better where a fixed quantity is assembled.
    """
    return intercept(value)
