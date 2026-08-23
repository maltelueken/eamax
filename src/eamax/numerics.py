"""Numerical guards shared by every density and likelihood in `eamax`.

Three distinct jobs live here, and keeping them distinct is the point of the module.
The three source repos conflated them, with consequences documented below.

1. **Parameter guards** (`guard_positive`) keep strictly-positive parameters away from
   zero so the closed forms stay finite under an unconstrained sampler.
2. **NaN containment** (`contain_nan`) replaces non-finite intermediates with the log
   floor *without* clamping, so an underflowed-but-representable value survives intact.
3. **The trial floor and the invalid-RT penalty** (`finalize_trial_logp`) apply once, to
   the assembled per-trial log-likelihood.

Two things this module deliberately does not do, both learned from the source repos:

* **It never floors a density component.** `racing-diffusion-conflict` and
  `cognitive-control-comparison` clamp `log_pdf` and `log_sf` separately at
  `log(1e-12)`. That makes the effective floor scale with the number of accumulators
  (a four-response race floors at `4 * log(1e-12)`) while simultaneously *inflating*
  every deep-tail survival term once per loser. For a likelihood used to compare models
  by marginal likelihood, that bias has the wrong sign: it flatters models that push
  losers into the tail. Flooring the assembled trial total once, as `eam-abi-robustness`
  does, is N-independent -- and it is the only variant validated against an outside
  implementation (EMC2, to 1e-9 on total dataset log-likelihood).
* **It never lets a caller re-clamp the penalty.** The `rt <= t0` penalty is by
  construction `<= log_floor`, so passing it through a `maximum(..., log_floor)` flattens
  it to a constant and destroys the gradient that pushes `t0` back into the valid region.
  `eamax` gives callers no intermediate to make that mistake with: `finalize_trial_logp`
  is the last step and returns a finished per-trial value.

`MIN_P` and `MIN_RT` share a default value but are different quantities -- a probability
floor and a time floor -- and are spelled separately so they can diverge without one
silently dragging the other.
"""

import jax.numpy as jnp

#: Floor on a per-trial likelihood, in probability units. `log(MIN_P)` is the log floor.
#: Matches EMC2's `log_likelihood_race`, which floors each trial at `min_ll = log(1e-10)`.
MIN_P = 1e-10

#: Floor on a decision time (`rt - t0`), in seconds. Below this a trial is *invalid*
#: rather than merely unlikely, and takes the penalty branch instead of the floor.
MIN_RT = 1e-10

#: Slope of the invalid-RT penalty, in nats per second of violation. Steep enough to
#: dominate any plausible likelihood gradient, so a sampler that proposes `t0 > min(rt)`
#: is pushed back rather than wandering across a flat barrier.
PENALTY_SLOPE = 1e3


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
    """Replace non-finite log-densities with the log floor, leaving finite values alone.

    Containment only -- this is *not* a clamp. A finite value far below the floor passes
    through unchanged, so it can still cancel correctly against other terms before
    `finalize_trial_logp` applies the floor once, at the end.

    `jnp.nan_to_num` is used rather than `jnp.clip` because `clip` propagates NaN
    (IEEE 754 says every comparison with NaN is false).
    """
    return jnp.nan_to_num(x, nan=log_floor(min_p))


def finalize_trial_logp(rt_shifted, total_logp, min_p=MIN_P, min_rt=MIN_RT):
    """Turn an assembled per-trial log-likelihood into a finished, guarded one.

    This is the single place a likelihood floor is applied, and the single place the
    invalid-RT penalty is introduced. Order matters: the floor is applied to the *valid*
    branch first, and only then is the penalty substituted in for invalid trials. Doing
    it the other way round -- flooring after substitution -- would clamp the penalty to a
    constant and remove its gradient.

    Parameters
    ----------
    rt_shifted : array
        Decision time ``rt - t0``, unclamped. Its sign is what distinguishes an invalid
        trial from an unlikely one, so it must be the raw difference.
    total_logp : array
        The assembled log-likelihood for the trial: the winner's log-density plus every
        loser's log-survival, summed and *unfloored*.
    min_p : float, optional
        Likelihood floor in probability units.
    min_rt : float, optional
        Decision times at or below this are invalid.

    Returns
    -------
    array
        Per-trial log-likelihood, floored at ``log(min_p)`` where the trial is feasible
        and replaced by a downward-sloping penalty where it is not.
    """
    floor = log_floor(min_p)
    valid = jnp.maximum(jnp.nan_to_num(total_logp, nan=floor), floor)
    penalty = floor + PENALTY_SLOPE * jnp.minimum(rt_shifted - min_rt, 0.0)
    return jnp.where(rt_shifted > min_rt, valid, penalty)
