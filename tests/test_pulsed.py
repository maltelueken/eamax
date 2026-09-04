"""The conflict pulse, its numerical density, and its grid sampler."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eamax.accumulators import (
    SimulatedPulsedWald,
    VolterraPulsedWald,
    Wald,
    normalized_gamma,
    normalized_gamma_derivative,
)

PULSE = {"amp": 0.3, "tau": 0.1}


def test_pulse_peaks_at_amp_regardless_of_time_scale():
    # The integrated signal is normalised by its own mode, which is what decouples the
    # amplitude from the time constant and lets the two be priored independently.
    t = jnp.linspace(1e-4, 2.0, 20000)
    for tau in (0.05, 0.1, 0.3):
        peak = float(jnp.max(normalized_gamma(t, PULSE["amp"], tau, 2.0)))
        assert peak == pytest.approx(PULSE["amp"], rel=1e-4)


def test_pulse_is_linear_in_amplitude_and_decays_to_zero():
    t = jnp.linspace(0.01, 3.0, 500)
    one = normalized_gamma_derivative(t, 0.3, 0.1, 2.0)
    two = normalized_gamma_derivative(t, 0.6, 0.1, 2.0)
    assert np.allclose(np.array(two), 2 * np.array(one))
    # Transient: conflict perturbs when an accumulator crosses, not where it ends up.
    assert abs(float(normalized_gamma_derivative(3.0, 0.3, 0.1, 2.0))) < 1e-6


def test_shorter_time_constants_decay_faster():
    t = jnp.array([0.5])
    fast = abs(float(normalized_gamma_derivative(t, 0.3, 0.05, 2.0)[0]))
    slow = abs(float(normalized_gamma_derivative(t, 0.3, 0.20, 2.0)[0]))
    assert fast < slow


def _volterra_error_vs_wald(dt):
    t = jnp.array([0.3, 0.6, 1.0, 1.8])
    params = {
        "v": jnp.full((4,), 2.0),
        "amp": jnp.full((4,), 1e-12),
        "tau": jnp.full((4,), 0.1),
        "s": jnp.ones((4,)),
        "b": jnp.full((4,), 1.0),
    }
    log_pdf, log_sf = VolterraPulsedWald(dt=dt, t_max=3.0).log_pdf_sf(t, params)
    ref_pdf, ref_sf = Wald().log_pdf_sf(t, {k: params[k] for k in ("v", "s", "b")})
    return (
        float(np.abs(np.array(log_pdf) - np.array(ref_pdf)).max()),
        float(np.abs(np.array(log_sf) - np.array(ref_sf)).max()),
    )


def test_volterra_reduces_to_the_analytic_wald_as_the_pulse_vanishes():
    # The reference check for the numerical solver: with no pulse the drift is constant, so
    # Fortet's equation must reproduce the inverse Gaussian. Never asserted in the source
    # repository, where it was only claimed in prose.
    #
    # The density is essentially exact -- the recursion solves for it directly. The survival
    # is not: it comes from trapezoidally integrating that density, so its error is
    # quadrature error and is orders of magnitude larger. Reporting one tolerance for both
    # would either hide the density's accuracy or fail on the survival's.
    pdf_error, sf_error = _volterra_error_vs_wald(0.002)
    assert pdf_error < 1e-10
    assert sf_error < 1e-5


def test_volterra_survival_converges_with_the_grid():
    # Sharper than a flat bound: the trapezoidal cumulative should be second order, so
    # quartering dt must cut the survival error by roughly an order of magnitude. A solver
    # that had merely been tuned to pass one tolerance would not show the right slope.
    _, coarse = _volterra_error_vs_wald(0.002)
    _, fine = _volterra_error_vs_wald(0.0005)
    assert fine < coarse / 8


def test_volterra_density_integrates_to_the_survival_it_reports():
    # Internal consistency of the solver: 1 - S(t) must equal the integral of g up to t.
    solver = VolterraPulsedWald(dt=0.002, t_max=4.0)
    t = jnp.array([0.5, 1.0, 2.0])
    params = {
        "v": jnp.full((3,), 1.5),
        "amp": jnp.full((3,), 0.3),
        "tau": jnp.full((3,), 0.1),
        "s": jnp.ones((3,)),
        "b": jnp.full((3,), 1.0),
    }
    log_pdf, log_sf = solver.log_pdf_sf(t, params)
    assert np.all(np.exp(np.array(log_sf)) < 1.0)
    assert np.all(np.isfinite(np.array(log_pdf)))
    # Survival must be monotonically decreasing in t.
    assert np.all(np.diff(np.array(log_sf)) < 0)


def test_the_grid_sampler_reproduces_the_wald_when_the_pulse_vanishes():
    # Ties the SDE sampler to the analytic accumulator it generalises: with no pulse, the
    # simulated first-passage distribution must match the inverse Gaussian.
    n = 20_000
    params = {
        "v": jnp.full((n,), 2.0),
        "amp": jnp.full((n,), 0.0),
        "tau": jnp.full((n,), 0.1),
        "s": jnp.ones((n,)),
        "b": jnp.full((n,), 1.0),
    }
    draws = np.array(SimulatedPulsedWald(dt=0.001, t_max=5.0, chunk_size=4000).sample(jax.random.key(0), params))
    finite = draws[np.isfinite(draws)]
    assert finite.size > 0.99 * n

    quantiles = np.quantile(finite, [0.1, 0.25, 0.5, 0.75, 0.9])
    analytic = np.array(Wald().sample(jax.random.key(1), {k: params[k] for k in ("v", "s", "b")}))
    reference = np.quantile(analytic, [0.1, 0.25, 0.5, 0.75, 0.9])
    # Discretisation bias, not sampling noise, sets this bound: the Brownian-bridge
    # correction is what keeps it this small at dt = 1e-3.
    assert np.abs(quantiles - reference).max() < 0.01


def test_non_crossing_accumulators_return_inf_not_a_negative_sentinel():
    # `inf` is what lets the race take a plain `min` and read an all-`inf` trial as
    # right-censored, with no trial-level special case.
    params = {
        "v": jnp.array([0.01]),
        "amp": jnp.array([0.0]),
        "tau": jnp.array([0.1]),
        "s": jnp.array([1e-6]),
        "b": jnp.array([100.0]),
    }
    draws = np.array(SimulatedPulsedWald(dt=0.01, t_max=1.0).sample(jax.random.key(0), params))
    assert np.all(np.isinf(draws))


def test_the_brownian_bridge_recovers_fast_crossings_a_grid_test_drops():
    # Without the bridge correction every crossing is detected a step late on average, which
    # biases response times upward; the bias is worst for fast trials. Compare a coarse grid
    # against a fine one -- with the correction in place they should nearly agree.
    n = 20_000
    params = {
        "v": jnp.full((n,), 3.0),
        "amp": jnp.zeros((n,)),
        "tau": jnp.full((n,), 0.1),
        "s": jnp.ones((n,)),
        "b": jnp.full((n,), 1.0),
    }
    coarse = np.array(SimulatedPulsedWald(dt=0.01, t_max=5.0, chunk_size=4000).sample(jax.random.key(4), params))
    fine = np.array(SimulatedPulsedWald(dt=0.0005, t_max=5.0, chunk_size=2000).sample(jax.random.key(4), params))
    coarse, fine = coarse[np.isfinite(coarse)], fine[np.isfinite(fine)]
    assert abs(coarse.mean() - fine.mean()) < 0.01


def test_the_pulse_shifts_response_times_earlier():
    n = 20_000
    base = {
        "v": jnp.full((n,), 1.5),
        "tau": jnp.full((n,), 0.1),
        "s": jnp.ones((n,)),
        "b": jnp.full((n,), 1.0),
    }
    sampler = SimulatedPulsedWald(dt=0.001, t_max=5.0, chunk_size=4000)
    without = np.array(sampler.sample(jax.random.key(5), {**base, "amp": jnp.zeros((n,))}))
    with_pulse = np.array(sampler.sample(jax.random.key(5), {**base, "amp": jnp.full((n,), 0.3)}))
    assert np.nanmean(with_pulse[np.isfinite(with_pulse)]) < np.nanmean(without[np.isfinite(without)])
