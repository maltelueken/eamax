"""The hierarchical prior, and the reconstruction formula six call sites once shared."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eamax.hierarchical import (
    HierarchicalLKJMVNPrior,
    cholesky_factor,
    default_bijector,
    interval_to_mu_loc_scale,
    reconstruct_from_dict,
    reconstruct_semicentered,
)

HYPER = {
    "inverse_gamma_scale": jnp.full((5,), 1.6),
    "mu_loc": jnp.array([0.1, 0.3, 0.4, 0.4, -1.2]),
    "mu_scale": jnp.full((5,), 0.15),
}


def _prior(num_subjects=6, num_params=5, num_centered=2, **overrides):
    return HierarchicalLKJMVNPrior(
        num_subjects, num_params, num_centered=num_centered, **{**HYPER, **overrides}
    )


def test_prior_components_have_the_shapes_the_reconstruction_expects():
    draw = _prior().sample(jax.random.key(0))
    assert {k: tuple(v.shape) for k, v in draw.items()} == {
        "s": (5,),
        "mu": (5,),
        "psi_raw": (5, 5),
        "z": (6, 3),
        "theta_bt": (6, 2),
    }
    assert np.all(np.array(draw["s"]) > 0)
    assert np.isfinite(float(_prior().log_prob(draw)))


def test_psi_raw_is_a_valid_correlation_cholesky():
    draw = _prior().sample(jax.random.key(1))
    psi = np.array(draw["psi_raw"])
    assert np.allclose(psi, np.tril(psi))  # lower triangular
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0)  # unit row norms


def test_reconstruction_reproduces_the_formula_it_replaced():
    # The verbatim block that appeared in about six places across the source repositories.
    # This test is what makes deleting those copies safe.
    draw = _prior().sample(jax.random.key(2))
    num_ncp = 3

    factor = np.array(draw["s"])[:, None] * np.array(draw["psi_raw"])
    expected = np.concatenate(
        [
            np.array(draw["mu"])[:num_ncp]
            + np.einsum("nj,ij->ni", np.array(draw["z"]), factor[:num_ncp, :num_ncp]),
            np.array(draw["theta_bt"]),
        ],
        axis=-1,
    )
    assert np.allclose(np.array(reconstruct_from_dict(draw)), expected)


def test_the_centered_block_lands_where_num_centered_says():
    # Parameter order is load-bearing: whichever parameters should be centered must be last,
    # because the split is positional.
    draw = _prior(num_centered=2).sample(jax.random.key(3))
    log_theta = np.array(reconstruct_from_dict(draw))
    assert np.allclose(log_theta[:, -2:], np.array(draw["theta_bt"]))


def test_the_semicentered_joint_is_the_intended_multivariate_normal():
    # The centered block is drawn from the MVN conditional on the non-centered one, so
    # however the split is placed, the marginal covariance of the reconstructed parameters
    # must be `L @ L.T`. If it were drawn independently this would fail for any correlated
    # population.
    prior = _prior(num_subjects=4000)
    draw = prior.sample(jax.random.key(4))
    log_theta = np.array(reconstruct_from_dict(draw))

    factor = np.array(cholesky_factor(draw["s"], draw["psi_raw"]))
    expected = factor @ factor.T
    empirical = np.cov(log_theta, rowvar=False)
    assert np.abs(empirical - expected).max() < 0.15 * np.abs(expected).max()


def test_the_bijector_round_trips_every_component():
    bijector = default_bijector()
    draw = _prior().sample(jax.random.key(5))
    restored = bijector.forward(bijector.inverse(draw))
    for name in draw:
        assert np.allclose(np.array(restored[name]), np.array(draw[name]), atol=1e-10), name


def test_the_mode_is_a_valid_position_the_bijector_accepts():
    # An init position that the bijector's inverse turns into NaN is the classic way a
    # sampler fails on step one.
    prior = _prior()
    mode = prior.mode()
    unconstrained = default_bijector().inverse(mode)
    for name, value in unconstrained.items():
        assert np.all(np.isfinite(np.array(value))), name
    assert np.allclose(np.array(mode["psi_raw"]), np.eye(5))
    assert np.allclose(np.array(mode["mu"]), np.array(HYPER["mu_loc"]))


def test_mismatched_hyperparameter_lengths_are_refused():
    # A length mismatch means the hyperparameters came from a different parameter spec than
    # the model being fitted, which is otherwise a hard error to trace.
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        HierarchicalLKJMVNPrior(4, 7, num_centered=2, **HYPER)


def test_reconstruction_is_vmappable_over_particles():
    # SMC reconstructs one `log_theta` per particle per chain, so this has to survive
    # nested vmap.
    prior = _prior()
    draws = jax.vmap(prior.sample)(jax.random.split(jax.random.key(6), 12))
    log_theta = jax.vmap(reconstruct_from_dict)(draws)
    assert log_theta.shape == (12, 6, 5)
    assert np.all(np.isfinite(np.array(log_theta)))


def test_interval_helper_splits_variance_between_population_and_subjects():
    mu_loc, mu_scale = interval_to_mu_loc_scale((np.array([0.0]), np.array([2.0])), np.array([0.2]))
    assert mu_loc[0] == pytest.approx(1.0)
    total = ((2.0 - 0.0) / (2 * 1.959963984540054)) ** 2
    assert mu_scale[0] ** 2 + 0.2**2 == pytest.approx(total)

    with pytest.raises(ValueError, match="too large"):
        interval_to_mu_loc_scale((np.array([0.0]), np.array([1.0])), np.array([5.0]))


def test_a_hierarchical_likelihood_is_just_the_race_vmapped_over_subjects():
    # There is no separate hierarchical likelihood in `eamax`, and this is why: once the
    # race handles one subject's trials with a mask, the multi-subject case is a vmap. The
    # source repositories maintained four near-identical hierarchical likelihood functions.
    from eamax import race_loglik
    from eamax.accumulators import Wald
    from eamax.design import TrialDesign, build_params_fn, intercept_slope_spec

    spec = intercept_slope_spec()
    params_fn = build_params_fn(spec, Wald())
    num_subjects, num_trials = 4, 30
    rng = np.random.default_rng(0)

    log_theta = jnp.tile(jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3])), (num_subjects, 1))
    rt = jnp.array(rng.uniform(0.5, 2.0, (num_subjects, num_trials)))
    response = jnp.array(rng.integers(1, 3, (num_subjects, num_trials)))
    mask = jnp.ones((num_subjects, num_trials), dtype=bool).at[-1, -5:].set(False)

    def one_subject(theta, rt_s, response_s, mask_s):
        design = TrialDesign(
            rt=rt_s, response=response_s, target=jnp.full((num_trials,), 2), mask=mask_s
        )
        params, t0 = params_fn(theta, design)
        return jnp.sum(
            race_loglik(rt_s, response_s, t0, lambda t: Wald().log_pdf_sf(t, params), mask=mask_s)
        )

    per_subject = jax.vmap(one_subject)(log_theta, rt, response, mask)
    assert per_subject.shape == (num_subjects,)
    assert np.all(np.isfinite(np.array(per_subject)))
    # The masked subject contributes only its unmasked trials.
    unmasked = one_subject(log_theta[-1], rt[-1], response[-1], jnp.ones((num_trials,), dtype=bool))
    assert float(per_subject[-1]) != float(unmasked)
