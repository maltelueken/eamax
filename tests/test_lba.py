"""LBA densities by self-consistency, and the sampler against the likelihood.

scipy has no LBA closed form, so the internal references here are integration identities
and the simulator itself. `test_emc2_reference.py` supplies the outside check.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from eamax.accumulators import LBA
from eamax.accumulators.lba import lba_logpdf, lba_logsf
from eamax.numerics import MIN_P

PARAMS = {"v_win": 3.5, "v_lose": 2.0, "s_win": 1.2, "s_lose": 1.0, "A": 0.6, "b": 1.8}


def test_defective_densities_integrate_to_one():
    # The decisive internal-consistency check: summing "accumulator i wins at time t" over
    # both i and t must give 1. It pins pdf and sf against each other simultaneously, which
    # neither can do alone.
    p = PARAMS

    def defective(t, win_v, win_s, lose_v, lose_s):
        lp = float(lba_logpdf(t, win_v, win_s, p["A"], p["b"]))
        ls = float(lba_logsf(t, lose_v, lose_s, p["A"], p["b"]))
        return np.exp(lp + ls)

    total = 0.0
    for win, lose in [("win", "lose"), ("lose", "win")]:
        value, _ = integrate.quad(
            defective,
            1e-6,
            np.inf,
            args=(p[f"v_{win}"], p[f"s_{win}"], p[f"v_{lose}"], p[f"s_{lose}"]),
            limit=400,
        )
        total += value
    assert total == pytest.approx(1.0, abs=1e-5)


def test_logsf_is_the_integral_of_logpdf():
    p = PARAMS
    for t in [0.3, 0.8, 2.0]:
        cdf, _ = integrate.quad(
            lambda x: float(np.exp(lba_logpdf(x, p["v_win"], p["s_win"], p["A"], p["b"]))),
            1e-6,
            t,
            limit=400,
        )
        assert float(np.exp(lba_logsf(t, p["v_win"], p["s_win"], p["A"], p["b"]))) == pytest.approx(
            1.0 - cdf, abs=1e-6
        )


def test_survival_stays_finite_in_the_far_tail():
    # Unlike the Wald, the LBA's truncated-normal drift leaves a heavy tail, so the survival
    # function must remain well above the floor even at absurd decision times.
    value = float(np.exp(lba_logsf(1e4, PARAMS["v_win"], PARAMS["s_win"], PARAMS["A"], PARAMS["b"])))
    assert value > 1e-8


def test_density_floor_engages_rather_than_returning_neg_inf():
    # At small decision times the density is a difference of nearly equal terms and underflows
    # to exactly zero. The floor must substitute, not `log(0)`.
    value = float(lba_logpdf(0.05, 3.5, 1.2, 0.6, 1.8))
    assert value == pytest.approx(float(np.log(MIN_P)))


def test_choice_probability_matches_the_simulator():
    # Ties the analytic likelihood to the sampler: P(accumulator 1 wins), computed by
    # integrating the defective density, must match the simulated win rate.
    p = PARAMS
    analytic, _ = integrate.quad(
        lambda t: float(
            np.exp(
                lba_logpdf(t, p["v_win"], p["s_win"], p["A"], p["b"])
                + lba_logsf(t, p["v_lose"], p["s_lose"], p["A"], p["b"])
            )
        ),
        1e-6,
        np.inf,
        limit=400,
    )

    n = 200_000
    ones = jnp.ones((2, n))
    params = {
        "v": jnp.stack([jnp.full((n,), p["v_win"]), jnp.full((n,), p["v_lose"])]),
        "s": jnp.stack([jnp.full((n,), p["s_win"]), jnp.full((n,), p["s_lose"])]),
        "A": ones * p["A"],
        "b": ones * p["b"],
    }
    fpt = np.array(LBA().sample(jax.random.key(1), params))
    simulated = float((np.argmin(fpt, axis=0) == 0).mean())
    assert simulated == pytest.approx(analytic, abs=0.005)


def test_threshold_stays_above_start_point_across_unconstrained_space():
    # `_lba_guard` caps A just below b, so the closed forms stay inside their support even for
    # the degenerate b < A an unconstrained sampler can propose.
    for log_gap in np.linspace(-30, 10, 40):
        gap = float(np.exp(log_gap))
        value = float(lba_logpdf(0.5, 3.0, 1.0, 0.6, 0.6 + gap))
        grad = float(jax.grad(lba_logpdf, argnums=4)(0.5, 3.0, 1.0, 0.6, 0.6 + gap))
        assert np.isfinite(value)
        assert np.isfinite(grad)
