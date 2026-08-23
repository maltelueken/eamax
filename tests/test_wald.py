"""The Wald primitives against scipy, and the sampler against the density."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from eamax.accumulators import Wald, inv_gauss_logpdf, inv_gauss_logsf

# scipy parameterizes the inverse Gaussian as invgauss(mu=m/l, scale=l); ours is (mean, shape).
GRID = [0.01, 1.0, 2.0, 10.0, 100.0]


def _relerr(ref, res):
    ref, res = np.asarray(ref), np.asarray(res)
    return np.abs(ref - res) / np.maximum(np.abs(ref), 1.0)


@pytest.mark.parametrize("mu", GRID)
@pytest.mark.parametrize("lam", GRID)
def test_inv_gauss_logpdf_matches_scipy(mu, lam):
    t = np.linspace(0.01, 10.0, 200)
    ours = np.array(inv_gauss_logpdf(jnp.array(t), mu, lam))
    ref = stats.invgauss.logpdf(t, mu=mu / lam, scale=lam)
    assert _relerr(ref, ours).max() < 1e-6


@pytest.mark.parametrize("mu", GRID)
@pytest.mark.parametrize("lam", GRID)
def test_inv_gauss_logsf_matches_scipy(mu, lam):
    t = np.linspace(0.01, 10.0, 200)
    ours = np.array(inv_gauss_logsf(jnp.array(t), mu, lam))
    ref = stats.invgauss.logsf(t, mu=mu / lam, scale=lam)
    assert _relerr(ref, ours).max() < 1e-6


def test_inv_gauss_logsf_stays_finite_and_differentiable_in_the_far_tail():
    # This is the whole reason the survival function is hand-rolled rather than delegated:
    # tfd.InverseGaussian.log_survival_function underflows to -inf here, which would zero out
    # the gradient of every loser's term -- i.e. most of a race likelihood.
    for mu, lam in [(0.5, 1.0), (1.0, 10.0), (2.0, 100.0), (0.1, 0.5)]:
        g = jax.grad(inv_gauss_logsf, argnums=(0, 1, 2))(50.0, mu, lam)
        assert np.all(np.isfinite(np.array(g)))
        assert np.isfinite(inv_gauss_logsf(50.0, mu, lam))


def test_wald_sampler_agrees_with_its_own_density():
    # Ties the simulator to the likelihood: the empirical CDF of 200k draws must match the
    # analytic CDF implied by `log_sf`. Nothing else in the package checks that the sampler
    # and the density describe the same distribution.
    v, s, b = 2.0, 1.1, 1.3
    params = {"v": jnp.full((200_000,), v), "s": jnp.full((200_000,), s), "b": jnp.full((200_000,), b)}
    draws = np.array(Wald().sample(jax.random.key(0), params))

    quantiles = np.quantile(draws, [0.1, 0.25, 0.5, 0.75, 0.9])
    empirical = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    analytic = 1.0 - np.exp(np.array(inv_gauss_logsf(jnp.array(quantiles), b / v, (b / s) ** 2)))
    assert np.abs(analytic - empirical).max() < 0.005


def test_wald_guards_non_positive_parameters():
    # An unconstrained sampler proposing v <= 0 must get a finite (very bad) log-density
    # rather than a NaN that poisons the whole gradient.
    p = {"v": jnp.array([-1.0, 0.0]), "s": jnp.array([0.0, 1.0]), "b": jnp.array([1.0, -2.0])}
    log_pdf, log_sf = Wald().log_pdf_sf(jnp.array([0.5, 0.5]), p)
    assert np.all(np.isfinite(np.array(log_pdf)))
    assert np.all(np.isfinite(np.array(log_sf)))
