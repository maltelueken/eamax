"""Numerical guards shared by every density and likelihood in `eamax`.

Three distinct jobs live here, and keeping them distinct is the point of the module:

1. **Parameter guards** (`guard_positive`) keep strictly-positive parameters away from
   zero so the closed forms stay finite under an unconstrained sampler.
2. **NaN containment** (`contain_nan`) replaces `NaN` and `+inf` intermediates with the
   log floor *without* clamping, so an underflowed-but-representable value survives
   intact.
3. **The trial floor** (`finalize_trial_logp`) applies once, to the assembled per-trial
   log-likelihood.

One thing this module deliberately does not do: **it never floors a density component.**
Flooring the winner's density and each loser's survival separately would make the
effective floor grow with the number of accumulators and inflate deep-tail survival
terms. The floor applies once, to the assembled trial total, so it is independent of the
number of accumulators.

There is no separate treatment of `rt <= t0`. Such a trial is evaluated at the clamped
decision time `MIN_RT`, where the winner's density underflows, and comes out at the floor
like any other hopeless trial. The floor is flat, so nothing in the likelihood pushes
`t0` back into the valid region -- that is the job of the support constraint at
initialisation, `eamax.inference.init.T0Support`.

`MIN_P` and `MIN_RT` share a default value but are different quantities -- a probability
floor and a time floor -- and are spelled separately so they can diverge independently.
"""

import jax.numpy as jnp

#: Floor on a per-trial likelihood, in probability units. `log(MIN_P)` is the log floor.
MIN_P = 1e-10

#: Floor on a decision time (`rt - t0`), in seconds. The race clamps decision times to it
#: before they reach an accumulator, so a closed form is never evaluated at or below zero.
MIN_RT = 1e-10


def log_floor(min_p=MIN_P):
    """Log of the per-trial likelihood floor."""
    return jnp.log(min_p)


def guard_positive(x, floor=MIN_P):
    """Clamp a strictly-positive parameter away from zero.

    Applied to drifts, diffusion coefficients and thresholds before they enter a closed
    form, so an unconstrained sampler that proposes a non-positive value gets a finite
    (very bad) log-density rather than a NaN that poisons the whole gradient.
    """
    return jnp.maximum(x, floor)


def contain_nan(x, min_p=MIN_P):
    """Replace `NaN` and `+inf` log-densities with the log floor, leaving finite values alone.

    Containment only -- this is *not* a clamp. A finite value far below the floor passes
    through unchanged, so it can still cancel correctly against other terms before
    `finalize_trial_logp` applies the floor once, at the end.

    `+inf` is contained rather than left to `jnp.nan_to_num`'s default, which maps it to the
    largest representable float. That value survives the floor and comes back as the trial's
    log-likelihood, making a breakdown look like an infinitely good fit -- the one direction
    a guard must never fail in. An infinite density carries no information, so it takes the
    floor alongside `NaN`. `-inf` keeps the default (the most negative float); it is already
    on the right side and the floor catches it at the end.

    `jnp.nan_to_num` is used rather than `jnp.clip` because `clip` propagates NaN
    (IEEE 754 says every comparison with NaN is false).
    """
    floor = log_floor(min_p)
    return jnp.nan_to_num(x, nan=floor, posinf=floor)


def finalize_trial_logp(total_logp, min_p=MIN_P):
    """Turn an assembled per-trial log-likelihood into a finished, guarded one.

    This is the single place a likelihood floor is applied.

    Parameters
    ----------
    total_logp : array
        The assembled log-likelihood for the trial: the winner's log-density plus every
        loser's log-survival, summed and *unfloored*.
    min_p : float, optional
        Likelihood floor in probability units.

    Returns
    -------
    array
        Per-trial log-likelihood, floored at ``log(min_p)``.
    """
    floor = log_floor(min_p)
    return jnp.maximum(jnp.nan_to_num(total_logp, nan=floor, posinf=floor), floor)
