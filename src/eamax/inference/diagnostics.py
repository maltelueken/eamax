"""Convergence and degeneracy diagnostics.

Three numbers, each answering a question the source repositories ask in different places or
not at all: has the chain converged, how much of the particle cloud is real, and how far
from uniform are the weights.
"""

import jax
import jax.numpy as jnp

from ._blackjax import RHAT_MIN_VERSION, blackjax


def rhat(positions, *, chain_axis=1, sample_axis=0):
    """Rank-normalized split-R-hat (Vehtari et al. 2021).

    A thin wrapper over ``blackjax.diagnostics.rhat``, here so that convergence checking
    costs no dependency beyond the sampler. It computes the same statistic as
    ``arviz.rhat(method="rank")`` -- verified to six decimals against ``arviz_stats.rhat``
    on both a converged and a deliberately non-converged set of 4 x 2000 chains -- so a
    consumer switching to it keeps the numbers it had.

    Defaults match :func:`eamax.inference.mcmc.inference_loop_multiple_chains`'s output
    layout, ``(draw, chain, param)``, rather than BlackJAX's own ``(chain, draw)``.

    Parameters
    ----------
    positions : array
        Samples with a chain axis and a sample axis; any remaining axes are treated
        elementwise.
    chain_axis, sample_axis : int, optional

    Returns
    -------
    array
        R-hat per remaining element, shaped exactly like ``positions`` with the chain and
        sample axes removed. BlackJAX returns this ``.squeeze()``d, which drops *every*
        singleton axis and not merely the two it consumed -- so a single-dataset file and
        a single-parameter file come back with the same shape and the caller cannot tell
        which axis went missing. Restoring the shape keeps a downstream
        ``all(axis=-1)`` reducing over parameters rather than over datasets.

    Raises
    ------
    ImportError
        If the installed BlackJAX predates ``diagnostics.rhat``. It is deliberately not
        backfilled with the older ``potential_scale_reduction``: that is plain
        Gelman-Rubin (1992), a *different statistic*, and silently substituting it would
        change which fits a downstream threshold accepts.
    """
    diagnostics = blackjax().diagnostics

    if not hasattr(diagnostics, "rhat"):
        raise ImportError(
            f"This BlackJAX has no `diagnostics.rhat`; it arrived in ~{RHAT_MIN_VERSION}. "
            f"Upgrade with `pip install 'blackjax>={RHAT_MIN_VERSION}'`.\n\n"
            "eamax does not fall back to `potential_scale_reduction`: that is the plain "
            "Gelman-Rubin statistic rather than the rank-normalized split-R-hat, so "
            "substituting it would quietly move any convergence threshold built on it."
        )

    positions = jnp.asarray(positions)
    consumed = {chain_axis % positions.ndim, sample_axis % positions.ndim}
    remaining = tuple(
        size for axis, size in enumerate(positions.shape) if axis not in consumed
    )

    values = diagnostics.rhat(positions, chain_axis=chain_axis, sample_axis=sample_axis)

    return jnp.reshape(values, remaining)


def count_unique_particles(particles):
    """Count distinct particles in a cloud -- the SMC degeneracy diagnostic.

    Each SMC step resamples and then mutates. Resampling duplicates particles outright, and
    a mutation step only undoes that if its proposals are accepted: a rejected HMC move
    leaves the particle *bit-identical* to its parent. With ``num_mcmc_steps = 1`` there is
    one move per resample to rediversify a cloud whose ESS has just been driven to
    ``target_ess``, so duplicates can compound across tempering iterations.

    Nothing else detects this. The weight ESS measures how far the *weights* are from
    uniform within one step; a cloud of 1000 particles collapsed onto 40 distinct values has
    perfectly uniform weights and an ESS of 1.0. Only a count of distinct particles
    separates "1000 draws" from "40 draws, each stored 25 times", and R-hat and ESS computed
    downstream take the stored draws at face value.

    Method: project each particle onto a fixed random vector, sort by that projection, then
    compare *adjacent full rows* for exact equality. Comparing all pairs of full particles
    would be ``O(N^2 D)`` and would allocate ~1 GB at ``N = 1000, D = 120``; sorting first is
    ``O(N log N + N D)``.

    The full-row comparison is not redundant with the projection, and the version this was
    ported from -- which counted changes in the sorted projections alone -- **overcounts**
    without it. XLA accumulates the projection differently depending on a row's position in
    the batch, so two bit-identical particles can come out one ULP apart: measured on a cloud
    of 60 rows containing 20 exact duplicates, one duplicate pair projected to
    0.5320762401923617 and 0.5320762401923615 and was counted twice. That is the unsafe
    direction for this diagnostic -- it under-reports collapse, which is the only thing it
    exists to find. Sorting by a projection that is merely *close* for identical rows still
    places them adjacent, so the exact comparison recovers the true count.

    Parameters
    ----------
    particles : array
        Cloud of shape ``(num_particles, num_flat_params)``.

    Returns
    -------
    array
        Number of distinct particles, as a traced scalar. Compare it against
        ``num_particles``: equal means no collapse, and the ratio is the fraction of the
        stored draws that carry independent information.
    """
    particles = jnp.asarray(particles)

    # Fixed key, so the same cloud always yields the same count.
    projection = jax.random.normal(jax.random.key(0), (particles.shape[-1],))
    ordered = particles[jnp.argsort(particles @ projection)]

    changed = jnp.any(ordered[1:] != ordered[:-1], axis=-1)
    return jnp.sum(changed) + 1


def weight_ess(weights):
    """Effective sample size of a set of normalized weights, as a fraction.

    Answers only how far the weights are from uniform. It says nothing about whether the
    particles they weight are distinct -- see :func:`count_unique_particles`, which is the
    diagnostic that catches a collapsed cloud.

    Parameters
    ----------
    weights : array
        Normalized linear-space weights along the last axis, shape ``(..., N)``.

    Returns
    -------
    array
        Shape ``(...)``, in ``(0, 1]``. 1.0 means uniform.
    """
    weights = jnp.asarray(weights)
    num_particles = weights.shape[-1]
    return 1.0 / jnp.sum(weights**2, axis=-1) / num_particles
