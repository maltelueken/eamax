"""Simulation and scoring must describe the same model.

The source repositories asserted this in prose -- "simulating from a set of parameters is
guaranteed to match what the likelihood actually scores, because both call the same
parameter map" -- and never tested it. The shared `params_fn` makes the *parameterization*
half structural; these tests check the other half, that the accumulator's sampler and its
density agree.

The instrument is the score test rather than parameter recovery. At the true parameters the
expected gradient of the log-likelihood is exactly zero, so averaging that gradient over
simulated datasets gives a sharp, cheap check that is sensitive to any mismatch between the
two sides. Recovery would be slower, flakier, and would only detect large errors.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eamax import race_loglik, simulate_race
from eamax.accumulators import LBA, Wald
from eamax.design import TrialDesign, build_params_fn, intercept_slope_spec
from eamax.design.legacy import lba_intercept_slope_spec


def _score_statistic(spec, accumulator, theta_true, num_datasets=64, num_trials=400, seed=0):
    """Mean and standard error of `grad_theta loglik(theta_true)` over simulated datasets."""
    params_fn = build_params_fn(spec, accumulator)
    base = TrialDesign(
        rt=jnp.zeros((num_trials,)),
        response=jnp.zeros((num_trials,), dtype=int),
        target=jnp.where(jnp.arange(num_trials) % 2 == 0, 1, 2),
    )

    def one_dataset(key):
        rt, response = simulate_race(key, params_fn, theta_true, base, accumulator)
        design = base.replace(rt=rt, response=response)

        def total(theta):
            params, t0 = params_fn(theta, design)
            return jnp.sum(
                race_loglik(
                    design.rt,
                    design.response,
                    t0,
                    lambda t: accumulator.log_pdf_sf(t, params),
                    first_response=design.first_response,
                )
            )

        return jax.grad(total)(theta_true)

    scores = jax.vmap(one_dataset)(jax.random.split(jax.random.key(seed), num_datasets))
    scores = np.array(scores)
    return scores.mean(axis=0), scores.std(axis=0, ddof=1) / np.sqrt(num_datasets)


def test_wald_race_score_vanishes_at_the_true_parameters():
    spec = intercept_slope_spec()
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3]))
    mean, stderr = _score_statistic(spec, Wald(), theta)
    # Every component within 3.5 standard errors of zero. A parameterization mismatch
    # between the simulator and the likelihood shows up as a bias many multiples of this.
    assert np.all(np.abs(mean) < 3.5 * stderr), dict(zip(spec.names, mean / stderr))


def test_lba_race_score_vanishes_at_the_true_parameters():
    spec = lba_intercept_slope_spec()
    theta = jnp.log(jnp.array([2.0, 1.0, 1.2, 0.6, 1.2, 0.3]))
    mean, stderr = _score_statistic(spec, LBA(), theta)
    assert np.all(np.abs(mean) < 3.5 * stderr), dict(zip(spec.names, mean / stderr))


def test_a_deliberately_mismatched_parameterization_is_detected():
    # Guards the guard: if the score test cannot see a wrong parameterization, it is not
    # doing its job. Scoring data simulated at one drift with the likelihood evaluated at
    # another must produce a score far from zero.
    spec = intercept_slope_spec()
    truth = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3]))
    wrong = truth.at[0].set(jnp.log(1.6))

    params_fn = build_params_fn(spec, Wald())
    base = TrialDesign(
        rt=jnp.zeros((400,)),
        response=jnp.zeros((400,), dtype=int),
        target=jnp.full((400,), 2),
    )

    def score(key):
        rt, response = simulate_race(key, params_fn, truth, base, Wald())
        design = base.replace(rt=rt, response=response)

        def total(theta):
            params, t0 = params_fn(theta, design)
            return jnp.sum(race_loglik(design.rt, design.response, t0,
                                       lambda t: Wald().log_pdf_sf(t, params),
                                       first_response=design.first_response))

        return jax.grad(total)(wrong)

    scores = np.array(jax.vmap(score)(jax.random.split(jax.random.key(1), 64)))
    mean = scores.mean(axis=0)
    stderr = scores.std(axis=0, ddof=1) / np.sqrt(64)
    assert np.abs(mean[0]) > 10 * stderr[0]


def test_simulated_response_times_all_exceed_the_non_decision_time():
    spec = intercept_slope_spec()
    params_fn = build_params_fn(spec, Wald())
    design = TrialDesign(
        rt=jnp.zeros((5000,)),
        response=jnp.zeros((5000,), dtype=int),
        target=jnp.full((5000,), 2),
    )
    rt, response = simulate_race(
        jax.random.key(2), params_fn, jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3])), design, Wald()
    )
    assert np.all(np.array(rt) > 0.3)
    assert set(np.unique(np.array(response))) <= {1, 2}
    # The target accumulator has the drift advantage, so it should usually win.
    assert float((np.array(response) == 2).mean()) > 0.5
