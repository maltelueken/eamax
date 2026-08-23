"""The race likelihood: its guards, its censoring, and its N-independence.

Several of these tests exist because the source repositories got the corresponding thing
wrong or left it to convention. Each one names which.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eamax import gather_by_mask, overlay_by_mask, race_from_arrays, race_loglik, winner_mask
from eamax.accumulators import Wald
from eamax.numerics import MIN_P, log_floor

FLOOR = float(np.log(MIN_P))


def _wald_pdf_sf(v, s, b):
    return lambda t: Wald().log_pdf_sf(t, {"v": v, "s": s, "b": b})


def test_matches_the_hand_written_two_accumulator_expression():
    # The regression that makes deleting the six duplicated race implementations safe: the
    # N-way form must reproduce the unrolled two-accumulator expression the source repos
    # wrote by hand. Reduction order differs (a length-2 `sum` versus a scalar `+`), so this
    # holds to a ULP rather than bit-for-bit.
    rng = np.random.default_rng(0)
    n = 2000
    rt = rng.uniform(0.4, 3.0, n)
    t0 = 0.2
    v = jnp.array(rng.uniform(0.5, 4.0, (2, n)))
    s = jnp.array(rng.uniform(0.5, 1.5, (2, n)))
    b = jnp.array(rng.uniform(0.5, 2.0, (2, n)))
    response = rng.integers(0, 2, n)

    log_pdf, log_sf = Wald().log_pdf_sf(jnp.maximum(jnp.array(rt) - t0, 1e-10), {"v": v, "s": s, "b": b})
    expected = np.where(
        response == 1,
        np.array(log_pdf[1] + log_sf[0]),
        np.array(log_pdf[0] + log_sf[1]),
    )
    expected = np.maximum(expected, FLOOR)

    ours = np.array(race_loglik(jnp.array(rt), jnp.array(response), t0, _wald_pdf_sf(v, s, b)))
    assert np.abs(ours - expected).max() < 1e-12


def test_penalty_slopes_when_rt_precedes_t0():
    # THE regression `cognitive-control-comparison` violates today: it re-clamps the value
    # returned by the per-accumulator function, flattening this penalty to a constant and
    # removing the gradient that pushes t0 back into the valid region. `eamax` hands callers
    # no intermediate to make that mistake with, so this must hold by construction.
    v = jnp.array([[2.0], [1.0]])
    s = jnp.ones((2, 1))
    b = jnp.ones((2, 1))

    def total(t0):
        return jnp.sum(race_loglik(jnp.array([0.3]), jnp.array([0]), t0, _wald_pdf_sf(v, s, b)))

    # t0 = 0.5 > rt = 0.3, so the trial is infeasible.
    assert float(total(0.5)) < FLOOR
    assert float(jax.grad(total)(0.5)) < -1.0

    # A feasible trial stays at or above the floor and keeps an ordinary gradient.
    assert float(total(0.1)) >= FLOOR


def test_the_floor_does_not_scale_with_the_number_of_accumulators():
    # Encodes the reason `eamax` floors the assembled trial total once rather than flooring
    # each density component. Flooring components makes the effective per-trial floor
    # N * log(min_p) -- so a four-accumulator race would sit far below a two-accumulator one
    # on the same hopeless trial, biasing any comparison across models with different N.
    def hopeless(n):
        # Every accumulator is impossibly slow, so the whole trial underflows.
        v = jnp.full((n, 1), 1e-3)
        s = jnp.full((n, 1), 1e-3)
        b = jnp.full((n, 1), 50.0)
        return float(race_loglik(jnp.array([0.5]), jnp.array([0]), 0.1, _wald_pdf_sf(v, s, b))[0])

    values = [hopeless(n) for n in (2, 3, 4, 8)]
    assert all(value == pytest.approx(FLOOR) for value in values)


def test_masked_trials_contribute_exactly_zero_whatever_they_hold():
    # Padding a ragged multi-subject dataset to a rectangle is only safe if the padding value
    # cannot influence the result. In the source repos this held by convention (_PAD_RT = 1.0)
    # rather than by construction.
    v = jnp.full((2, 4), 2.0)
    s = jnp.ones((2, 4))
    b = jnp.ones((2, 4))
    mask = jnp.array([True, True, False, False])
    response = jnp.array([0, 1, 0, 1])

    a = race_loglik(jnp.array([0.5, 0.7, 1.0, 1.0]), response, 0.2, _wald_pdf_sf(v, s, b), mask=mask)
    b_ = race_loglik(jnp.array([0.5, 0.7, -99.0, 1e9]), response, 0.2, _wald_pdf_sf(v, s, b), mask=mask)

    assert np.array_equal(np.array(a), np.array(b_))
    assert np.array_equal(np.array(a)[2:], np.zeros(2))


def test_censored_trials_are_scored_by_the_summed_survival():
    # A non-crossing trial is not missing data: it is the observation `T > t_max`, whose
    # likelihood is the probability every accumulator was still running. That is `sum(log_sf)`
    # for any N, which is why censoring needs no per-model routing.
    n_acc, t_max, t0 = 3, 4.0, 0.2
    v = jnp.full((n_acc, 1), 1.5)
    s = jnp.ones((n_acc, 1))
    b = jnp.full((n_acc, 1), 1.2)

    _, log_sf = Wald().log_pdf_sf(jnp.array([t_max]), {"v": v, "s": s, "b": b})
    expected = float(jnp.sum(log_sf))

    ours = float(
        race_loglik(jnp.array([-1.0]), jnp.array([0]), t0, _wald_pdf_sf(v, s, b), t_max=t_max)[0]
    )
    assert ours == pytest.approx(expected)


def test_censored_trials_carry_no_t0_gradient():
    # The censored contribution is evaluated at `t_max + t0`, so its decision time is exactly
    # t_max regardless of t0. A censored trial must therefore say nothing about t0.
    v = jnp.full((2, 1), 1.5)
    s = jnp.ones((2, 1))
    b = jnp.full((2, 1), 1.2)

    def total(t0, t_max):
        return jnp.sum(race_loglik(jnp.array([-1.0]), jnp.array([0]), t0, _wald_pdf_sf(v, s, b), t_max=t_max))

    assert float(jax.grad(total)(0.2, 4.0)) == pytest.approx(0.0, abs=1e-12)

    # Without a censoring model the same negative RT is simply an invalid trial, and does
    # carry a (steep, penalising) gradient.
    def uncensored(t0):
        return jnp.sum(race_loglik(jnp.array([-1.0]), jnp.array([0]), t0, _wald_pdf_sf(v, s, b)))

    assert float(jax.grad(uncensored)(0.2)) < -1.0


def test_winner_mask_handles_both_response_codings():
    # Two-accumulator repos label choices 0/1; the empirical datasets label responses 1..R.
    assert np.array_equal(
        np.array(winner_mask(jnp.array([0, 1, 1]), 2)),
        np.array([[True, False, False], [False, True, True]]),
    )
    assert np.array_equal(
        np.array(winner_mask(jnp.array([1, 2, 2]), 2, first_response=1)),
        np.array([[True, False, False], [False, True, True]]),
    )


def test_gather_and_overlay_round_trip():
    # The hybrid-race seam: pick out the one special accumulator per trial, evaluate it
    # separately (once per trial, not once per accumulator), and put the result back.
    base = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    special = jnp.array([1, 0, 2])
    mask = winner_mask(special, 3)

    (gathered,) = gather_by_mask(mask, base)
    assert np.array_equal(np.array(gathered), np.array([4.0, 2.0, 9.0]))
    assert np.array_equal(np.array(overlay_by_mask(mask, gathered, base)), np.array(base))

    replaced = np.array(overlay_by_mask(mask, jnp.array([-1.0, -2.0, -3.0]), base))
    assert np.array_equal(replaced[:, 0], np.array([1.0, -1.0, 7.0]))


def test_race_is_vmappable_over_subjects():
    # `eamax` deliberately handles one dataset's worth of trials and leaves every outer axis
    # to the caller. This is the shape contract the hierarchical consumers rely on.
    n_subj, n_trial = 5, 20
    rng = np.random.default_rng(3)
    rt = jnp.array(rng.uniform(0.4, 2.0, (n_subj, n_trial)))
    response = jnp.array(rng.integers(0, 2, (n_subj, n_trial)))
    t0 = jnp.full((n_subj,), 0.2)
    v = jnp.array(rng.uniform(1.0, 3.0, (n_subj, 2, n_trial)))

    def one(rt_s, resp_s, t0_s, v_s):
        return race_loglik(
            rt_s, resp_s, t0_s, _wald_pdf_sf(v_s, jnp.ones((2, n_trial)), jnp.ones((2, n_trial)))
        )

    out = jax.vmap(one)(rt, response, t0, v)
    assert out.shape == (n_subj, n_trial)
    assert np.all(np.isfinite(np.array(out)))
