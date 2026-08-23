"""Adaptive tempered SMC.

The target is a conjugate Gaussian, so the posterior mean, the posterior width **and the
log marginal likelihood** are all known in closed form. That last one is the point: the
evidence is the entire scientific output of one consumer repository and has never been
checked against a known answer anywhere, while the other consumer discards it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eamax.inference.smc import (
    DEFAULT_MAX_STEPS,
    SMCResult,
    tempered_smc,
)

NUM_DIMS = 2
PRIOR_SD = 5.0
NOISE_SD = 1.0
OBSERVED = jnp.asarray([1.0, -0.5])

NUM_CHAINS = 4
NUM_PARTICLES = 2000


def log_prior_fn(theta):
    return jnp.sum(jax.scipy.stats.norm.logpdf(theta, 0.0, PRIOR_SD))


def log_likelihood_fn(theta):
    return jnp.sum(jax.scipy.stats.norm.logpdf(OBSERVED, theta, NOISE_SD))


def analytic_posterior():
    variance = 1.0 / (1.0 / PRIOR_SD**2 + 1.0 / NOISE_SD**2)
    return variance * OBSERVED / NOISE_SD**2, np.sqrt(variance)


def analytic_log_evidence(prior_sd=PRIOR_SD):
    """`log p(x) = log N(x | 0, prior_sd^2 + noise_sd^2)`."""
    return float(
        jnp.sum(jax.scipy.stats.norm.logpdf(OBSERVED, 0.0, jnp.sqrt(prior_sd**2 + NOISE_SD**2)))
    )


def prior_cloud(key, num_chains=NUM_CHAINS, num_particles=NUM_PARTICLES, prior_sd=PRIOR_SD):
    return jax.random.normal(key, (num_chains, num_particles, NUM_DIMS)) * prior_sd


def tuning(num_chains=NUM_CHAINS, step_size=0.6):
    return {
        "step_size": jnp.full(num_chains, step_size),
        "inverse_mass_matrix": jnp.ones((num_chains, NUM_DIMS)),
    }


@pytest.fixture(scope="module")
def result():
    return tempered_smc(
        jax.random.key(0),
        log_prior_fn,
        log_likelihood_fn,
        prior_cloud(jax.random.key(1)),
        tuning(),
        num_integration_steps=20,
        num_mcmc_steps=5,
    )


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
def test_it_recovers_the_conjugate_posterior_mean(result):
    mean, _ = analytic_posterior()

    assert jnp.allclose(jnp.mean(result.particles, axis=(0, 1)), mean, atol=0.1)


def test_it_recovers_the_conjugate_posterior_width(result):
    _, sd = analytic_posterior()
    sampled = jnp.std(result.particles.reshape(-1, NUM_DIMS), axis=0)

    assert jnp.allclose(sampled, sd, rtol=0.15)


def test_the_log_marginal_likelihood_matches_the_analytic_evidence(result):
    """Never checked against a known answer in any source repository."""
    estimate = float(jnp.mean(result.log_marginal_likelihood))

    assert estimate == pytest.approx(analytic_log_evidence(), abs=0.2)


def test_a_bayes_factor_between_two_priors_is_recovered():
    """Better conditioned than the absolute evidence, and the quantity actually reported.

    A constant offset in the estimator would cancel here, which is why this runs *alongside*
    the absolute test rather than instead of it.
    """
    wide, narrow = 5.0, 2.0

    def run(prior_sd):
        def prior_fn(theta):
            return jnp.sum(jax.scipy.stats.norm.logpdf(theta, 0.0, prior_sd))

        outcome = tempered_smc(
            jax.random.key(0), prior_fn, log_likelihood_fn,
            prior_cloud(jax.random.key(1), prior_sd=prior_sd), tuning(),
            num_integration_steps=20, num_mcmc_steps=5,
        )
        return float(jnp.mean(outcome.log_marginal_likelihood))

    estimated = run(wide) - run(narrow)
    expected = analytic_log_evidence(wide) - analytic_log_evidence(narrow)

    assert estimated == pytest.approx(expected, abs=0.15)


# --------------------------------------------------------------------------- #
# The result object
# --------------------------------------------------------------------------- #
def test_every_field_carries_a_chain_axis(result):
    assert isinstance(result, SMCResult)
    assert result.particles.shape == (NUM_CHAINS, NUM_PARTICLES, NUM_DIMS)
    assert result.weights.shape == (NUM_CHAINS, NUM_PARTICLES)
    for field in ("log_marginal_likelihood", "num_iterations", "num_unique",
                  "num_unique_smc", "weight_ess"):
        assert getattr(result, field).shape == (NUM_CHAINS,)


def test_the_weights_are_normalized_linear_weights_not_log_weights(result):
    """The field is named for what BlackJAX returns.

    One consumer stores exactly this array under the name ``log_weights``; these would be a
    strange set of logarithms, being positive and summing to one.
    """
    assert jnp.all(result.weights >= 0.0)
    assert jnp.allclose(jnp.sum(result.weights, axis=-1), 1.0)


def test_the_temperature_reached_one(result):
    assert jnp.all(result.num_iterations < DEFAULT_MAX_STEPS)


def test_the_cloud_did_not_collapse(result):
    assert jnp.all(result.num_unique_smc > 0.9 * NUM_PARTICLES)


def test_the_weight_ess_is_a_fraction(result):
    assert jnp.all(result.weight_ess > 0.0)
    assert jnp.all(result.weight_ess <= 1.0)


# --------------------------------------------------------------------------- #
# The final resample
# --------------------------------------------------------------------------- #
def test_the_final_resample_trades_distinct_particles_for_equal_weights(result):
    """Resampling duplicates, so `num_unique` drops. That is the price of uniformity.

    Both counts are reported precisely so the trade is visible rather than assumed.
    """
    assert jnp.all(result.num_unique < result.num_unique_smc)


def test_skipping_the_final_resample_leaves_the_particles_unequally_weighted():
    common = dict(
        num_integration_steps=20,
        num_mcmc_steps=5,
    )
    cloud = prior_cloud(jax.random.key(1), num_chains=2)

    resampled = tempered_smc(
        jax.random.key(0), log_prior_fn, log_likelihood_fn, cloud, tuning(2), **common
    )
    raw = tempered_smc(
        jax.random.key(0), log_prior_fn, log_likelihood_fn, cloud, tuning(2),
        resample_final=False, **common,
    )

    assert not jnp.allclose(resampled.particles, raw.particles)
    # Untouched by the resample, the cloud keeps every distinct particle it had.
    assert jnp.all(raw.num_unique == raw.num_unique_smc)


# --------------------------------------------------------------------------- #
# Chains
# --------------------------------------------------------------------------- #
def test_per_chain_clouds_differ_from_each_other():
    cloud = prior_cloud(jax.random.key(1), num_chains=2, num_particles=200)

    assert not jnp.allclose(cloud[0], cloud[1])


def test_mapping_chains_with_vmap_agrees_with_the_sequential_default():
    cloud = prior_cloud(jax.random.key(1), num_chains=2, num_particles=500)
    common = dict(num_integration_steps=10, num_mcmc_steps=2)

    sequential = tempered_smc(
        jax.random.key(0), log_prior_fn, log_likelihood_fn, cloud, tuning(2), **common
    )
    vmapped = tempered_smc(
        jax.random.key(0), log_prior_fn, log_likelihood_fn, cloud, tuning(2),
        map_chains="vmap", **common,
    )

    assert jnp.allclose(
        sequential.log_marginal_likelihood, vmapped.log_marginal_likelihood, atol=1e-8
    )
    assert jnp.allclose(sequential.particles, vmapped.particles, atol=1e-8)


def test_the_source_repositorys_configuration_still_runs():
    """Shared tuning is gone; `resample_final=False` is the compatibility knob that remains."""
    cloud = prior_cloud(jax.random.key(1), num_chains=2, num_particles=500)

    outcome = tempered_smc(
        jax.random.key(0), log_prior_fn, log_likelihood_fn, cloud, tuning(2),
        num_integration_steps=20, num_mcmc_steps=1, resample_final=False,
    )

    assert outcome.particles.shape == (2, 500, NUM_DIMS)
    assert jnp.all(jnp.isfinite(outcome.log_marginal_likelihood))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_a_cloud_without_a_chain_axis_is_an_error():
    """A single shared cloud is not broadcast for you -- it is refused."""
    with pytest.raises(ValueError, match="init_particles_from_prior"):
        tempered_smc(
            jax.random.key(0), log_prior_fn, log_likelihood_fn,
            jnp.zeros((100, NUM_DIMS)), tuning(),
        )


def test_tuning_that_disagrees_about_the_chain_count_is_an_error():
    with pytest.raises(ValueError, match="leading axis of 4 chains"):
        tempered_smc(
            jax.random.key(0), log_prior_fn, log_likelihood_fn,
            prior_cloud(jax.random.key(1), num_particles=100), tuning(num_chains=3),
        )


def test_an_unknown_chain_mapping_is_an_error():
    with pytest.raises(ValueError, match="map_chains"):
        tempered_smc(
            jax.random.key(0), log_prior_fn, log_likelihood_fn,
            prior_cloud(jax.random.key(1), num_particles=100), tuning(),
            map_chains="parallel",
        )
