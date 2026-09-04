"""Flat unconstrained coordinates for the hierarchical prior.

The tests that matter here check that the log-det-Jacobian is right, and that the generic
form agrees with a hand-written reference that works around a TFP broadcasting quirk.
"""

import jax
import jax.numpy as jnp
import pytest

from eamax._tfp import tfb
from eamax.hierarchical import (
    HierarchicalFlatSpace,
    HierarchicalLKJMVNPrior,
    default_bijector,
    joint_log_det_jacobian,
    reconstruct_from_dict,
)

NUM_PARAMS = 4
NUM_SUBJECTS = 3
NUM_CENTERED = 2


@pytest.fixture
def prior():
    return HierarchicalLKJMVNPrior(
        NUM_SUBJECTS,
        NUM_PARAMS,
        inverse_gamma_scale=jnp.full(NUM_PARAMS, 0.4),
        mu_loc=jnp.linspace(-1.0, 1.0, NUM_PARAMS),
        mu_scale=jnp.full(NUM_PARAMS, 0.25),
        num_centered=NUM_CENTERED,
    )


@pytest.fixture
def flat_space(prior):
    return prior.flat_space()


def _draws(flat_space, num=3):
    return [flat_space.sample(jax.random.key(seed)) for seed in range(num)]


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def test_the_flat_layout_covers_every_component_exactly_once(flat_space):
    slices = flat_space.component_slices

    assert set(slices) == {"mu", "psi_raw", "s", "theta_bt", "z"}

    covered = sorted(
        index for piece in slices.values() for index in range(piece.start, piece.stop)
    )
    assert covered == list(range(flat_space.num_flat_params))


def test_the_flat_layout_is_ravel_pytrees_sorted_key_order(flat_space):
    # Not the order the prior declares them in. Anything hard-coding offsets from the
    # declaration order would be silently wrong, which is why `component_slices` exists.
    starts = [flat_space.component_slices[name].start for name in sorted(flat_space.component_slices)]
    assert starts == sorted(starts)


def test_raveling_round_trips(flat_space):
    for flat in _draws(flat_space):
        assert jnp.allclose(flat_space.ravel(flat_space.unravel(flat)), flat)


def test_raveling_a_dict_missing_a_component_is_an_error(flat_space):
    """The layout is fixed at construction; `ravel` must hold the argument to it.

    Re-deriving a layout from whatever it is handed accepts a short dict and returns a
    short vector, which `unravel` then *misinterprets* -- reading each component out of the
    wrong offsets -- rather than rejecting.
    """
    components = flat_space.unravel(flat_space.mode())
    del components["z"]

    with pytest.raises(ValueError, match="do not match this flat space's layout"):
        flat_space.ravel(components)


def test_raveling_a_dict_with_an_extra_component_is_an_error(flat_space):
    components = dict(flat_space.unravel(flat_space.mode()), spurious=jnp.zeros(3))

    with pytest.raises(ValueError, match="do not match this flat space's layout"):
        flat_space.ravel(components)


def test_the_centered_block_is_the_trailing_columns_of_the_subject_parameters(flat_space):
    """What `init_particles_from_prior` reads instead of running the reconstruction."""
    num_ncp = flat_space.prior.num_params_ncp

    for flat in _draws(flat_space):
        assert jnp.allclose(
            flat_space.centered_block(flat), flat_space.subject_params(flat)[:, num_ncp:]
        )


def test_the_mode_is_the_priors_mode(prior, flat_space):
    expected = flat_space.ravel(flat_space.bijector.inverse(prior.mode()))
    assert jnp.allclose(flat_space.mode(), expected)


def test_subject_params_agree_with_the_shared_reconstruction(flat_space):
    for flat in _draws(flat_space):
        assert jnp.allclose(
            flat_space.subject_params(flat),
            reconstruct_from_dict(flat_space.forward(flat)),
        )


def test_subject_params_have_one_row_per_subject(flat_space):
    assert flat_space.subject_params(flat_space.mode()).shape == (NUM_SUBJECTS, NUM_PARAMS)


def test_population_returns_mu_and_s(flat_space):
    mu, s = flat_space.population(flat_space.mode())

    assert mu.shape == (NUM_PARAMS,)
    assert s.shape == (NUM_PARAMS,)
    assert jnp.all(s > 0.0)  # `s` is Exp-linked, so it comes back positive


def test_sample_particles_draws_independently(flat_space):
    particles = flat_space.sample_particles(jax.random.key(0), 8)

    assert particles.shape == (8, flat_space.num_flat_params)
    assert not jnp.allclose(particles[0], particles[1])


# --------------------------------------------------------------------------- #
# The Jacobian
# --------------------------------------------------------------------------- #
def test_the_jacobian_matches_the_hand_written_formula(flat_space):
    """`eamax`'s generic form reproduces the hand-written inlined one exactly.

    The hand-written version computes `sum(unconstrained["s"]) + CorrelationCholesky ljd`,
    naming the prior's components inline. `joint_log_det_jacobian` gets the same number
    without naming anything.
    """
    for flat in _draws(flat_space):
        unconstrained = flat_space.unravel(flat)
        hand_written = jnp.sum(unconstrained["s"]) + tfb().CorrelationCholesky(
        ).forward_log_det_jacobian(unconstrained["psi_raw"], event_ndims=1)

        assert jnp.allclose(flat_space.log_det_jacobian(flat), hand_written)


def test_the_jacobian_is_consistent_with_the_inverse_map(flat_space):
    """An independent check: the forward log-det is minus the inverse log-det.

    Independent of the hand-written formula, so together the two pin the value rather than
    just pinning agreement with one particular way of writing it down.
    """
    bijector = flat_space.bijector

    for flat in _draws(flat_space):
        constrained = bijector.forward(flat_space.unravel(flat))
        inverse = sum(
            component.inverse_log_det_jacobian(constrained[name], event_ndims=jnp.ndim(constrained[name]))
            for name, component in bijector.bijectors.items()
        )

        assert jnp.allclose(flat_space.log_det_jacobian(flat), -inverse)


def test_a_square_bijector_matches_an_autodiff_determinant():
    """For a JointMap with no manifold-valued component, check against `slogdet`.

    `default_bijector` cannot be checked this way: `CorrelationCholesky` maps
    `P(P-1)/2` unconstrained values onto a `(P, P)` matrix, so the raveled map is not
    square and has no determinant. Dropping that one component leaves a square map, which
    autodiff can verify end to end.
    """
    from jax.flatten_util import ravel_pytree

    bijector = tfb().JointMap({"s": tfb().Exp(), "mu": tfb().Identity()})
    unconstrained = {"s": jnp.array([0.3, -1.2, 0.7]), "mu": jnp.array([0.1, 0.4, -0.2])}

    def forward(flat):
        return ravel_pytree(bijector.forward(unravel(flat)))[0]

    flat, unravel = ravel_pytree(unconstrained)
    _, log_det = jnp.linalg.slogdet(jax.jacfwd(forward)(flat))

    assert jnp.allclose(joint_log_det_jacobian(bijector, unconstrained), log_det)


def test_omitting_event_ndims_overcounts_by_the_number_of_parameters(flat_space):
    """Canary for the TFP miscall this works around.

    Without `event_ndims` each component falls back to its bijector's *minimum* event rank
    -- 0 for `Exp` -- so that term is returned unreduced with shape `(P,)` and TFP
    broadcasts the scalar `CorrelationCholesky` term across it. Summing then adds that
    scalar `P` times instead of once.

    The discrepancy is `(P - 1)` times a state-dependent quantity, so it does not cancel
    between two positions: it would tilt the posterior over correlations rather than shift
    the log density by a constant. Asserted here so that the day TFP changes this
    behaviour, something says so.
    """
    bijector = flat_space.bijector

    for flat in _draws(flat_space):
        unconstrained = flat_space.unravel(flat)
        naive = sum(
            jnp.sum(leaf) for leaf in jax.tree.leaves(bijector.forward_log_det_jacobian(unconstrained))
        )
        correct = joint_log_det_jacobian(bijector, unconstrained)
        cholesky_term = tfb().CorrelationCholesky().forward_log_det_jacobian(
            unconstrained["psi_raw"], event_ndims=1
        )

        assert not jnp.allclose(naive, correct)
        assert jnp.allclose(naive - correct, (NUM_PARAMS - 1) * cholesky_term)


# --------------------------------------------------------------------------- #
# The density
# --------------------------------------------------------------------------- #
def test_log_prob_is_the_priors_density_plus_the_jacobian(prior, flat_space):
    for flat in _draws(flat_space):
        expected = prior.log_prob(flat_space.forward(flat)) + flat_space.log_det_jacobian(flat)

        assert jnp.allclose(flat_space.log_prob(flat), expected)


def test_log_prob_is_finite_at_the_mode_and_at_prior_draws(flat_space):
    assert jnp.isfinite(flat_space.log_prob(flat_space.mode()))

    for flat in _draws(flat_space, num=5):
        assert jnp.isfinite(flat_space.log_prob(flat))


def test_log_prob_has_a_finite_gradient(flat_space):
    """A sampler needs the gradient, not just the value."""
    gradient = jax.grad(flat_space.log_prob)(flat_space.mode())

    assert gradient.shape == (flat_space.num_flat_params,)
    assert jnp.all(jnp.isfinite(gradient))


def test_it_works_under_jit_and_vmap(flat_space):
    """The unravel closure is fixed at construction, so it is static under tracing."""
    particles = flat_space.sample_particles(jax.random.key(3), 6)
    values = jax.jit(jax.vmap(flat_space.log_prob))(particles)

    assert values.shape == (6,)
    assert jnp.all(jnp.isfinite(values))


def test_an_explicit_bijector_is_used_instead_of_the_default(prior):
    space = HierarchicalFlatSpace(prior, default_bijector())

    assert space.num_flat_params == prior.flat_space().num_flat_params
