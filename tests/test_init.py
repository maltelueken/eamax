"""Starting values.

The constraint these tests are about is not a nicety. A chain started at ``t0 >= min(rt)``
begins on the race likelihood's penalty region, where the gradient is order 1e3 and step
size adaptation has nothing useful to do. One of the four source routines enforces this by
assignment, two by rejection, and one not at all.
"""

import jax
import jax.numpy as jnp
import pytest

from eamax._tfp import tfb
from eamax.design import coef, intercept, rdm_intercept_slope_spec, parameterization, quantity, term
from eamax.hierarchical import HierarchicalLKJMVNPrior
from eamax.inference.init import (
    T0Support,
    init_particles_from_prior,
    init_position_from_mode,
    init_position_from_values,
    init_positions_from_prior,
    jitter_positions,
    min_valid_rt,
    rejection_sample,
)
from eamax.inference.transforms import BlockTransform

NUM_PARAMS = 4
NUM_SUBJECTS = 3
NUM_CENTERED = 2


def _t0_first_spec():
    """A spec whose `t0` is at index 0 rather than last."""
    t0 = coef("t0", tfb().Exp())
    v = coef("V", tfb().Exp())
    b = coef("B", tfb().Exp())
    return parameterization(
        order=[t0, v, b],
        quantities=[
            quantity("v", term(v, intercept())),
            quantity("b", term(b, intercept())),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=2,
    )


@pytest.fixture
def spec():
    return rdm_intercept_slope_spec()


@pytest.fixture
def prior():
    # `t0` sits last, so it lands in the trailing centered block -- the usual arrangement.
    return HierarchicalLKJMVNPrior(
        NUM_SUBJECTS,
        NUM_PARAMS,
        inverse_gamma_scale=jnp.full(NUM_PARAMS, 0.4),
        mu_loc=jnp.asarray([0.9, 0.4, 0.0, jnp.log(0.2)]),
        mu_scale=jnp.full(NUM_PARAMS, 0.25),
        num_centered=NUM_CENTERED,
    )


# --------------------------------------------------------------------------- #
# min_valid_rt
# --------------------------------------------------------------------------- #
def test_the_non_crossing_sentinel_is_not_a_fast_response():
    rt = jnp.asarray([0.9, -1.0, 0.5, -1.0])

    assert float(min_valid_rt(rt)) == pytest.approx(0.5)


def test_padding_is_excluded():
    rt = jnp.asarray([0.9, 0.2, 0.5])
    mask = jnp.asarray([True, False, True])

    assert float(min_valid_rt(rt, mask=mask)) == pytest.approx(0.5)


def test_it_returns_one_value_per_subject():
    rt = jnp.asarray([[0.9, 0.5], [0.4, -1.0]])

    assert jnp.allclose(min_valid_rt(rt), jnp.asarray([0.5, 0.4]))


def test_a_subject_with_no_valid_observation_is_an_error():
    """`inf` is the identity element of `min`, not an answer, and it propagates silently.

    A subject whose trials all timed out -- or a fully padded row of a ragged batch -- has
    no fastest response. Returning `inf` gives `T0Support.from_spec` an infinite cap, which
    disables the constraint for exactly the subject whose data cannot support it, and
    nothing downstream raises.
    """
    rt = jnp.asarray([[-1.0, -1.0], [0.5, 0.7]])

    with pytest.raises(ValueError, match="No valid response time"):
        min_valid_rt(rt)


def test_masking_every_trial_of_a_subject_is_the_same_error():
    rt = jnp.asarray([[0.9, 0.5], [0.4, 0.6]])
    mask = jnp.asarray([[False, False], [True, True]])

    with pytest.raises(ValueError, match="No valid response time"):
        min_valid_rt(rt, mask=mask)


# --------------------------------------------------------------------------- #
# T0Support
# --------------------------------------------------------------------------- #
def test_it_finds_t0_by_name(spec):
    support = T0Support.from_spec(spec, 0.5)

    assert support.index == spec.index("t0")
    assert float(support.log_t0_max) == pytest.approx(float(jnp.log(0.97 * 0.5)))


def test_a_spec_without_t0_is_an_error():
    v = coef("V", tfb().Exp())
    spec = parameterization(
        order=[v], quantities=[quantity("v", term(v, intercept()))], num_responses=2
    )

    with pytest.raises(ValueError, match="needs a parameter named 't0'"):
        T0Support.from_spec(spec, 0.5)


def test_a_t0_on_the_wrong_link_is_an_error():
    """The cap compares an unconstrained value, so it is only `log(t0)` under the log link."""
    t0 = coef("t0", tfb().Identity())
    spec = parameterization(
        order=[t0], quantities=[quantity("t0", term(t0, intercept()))], num_responses=2
    )

    with pytest.raises(ValueError, match="must be on the log"):
        T0Support.from_spec(spec, 0.5)


def test_the_constraint_fires_when_t0_is_not_last():
    """The positional assumption is gone: `t0` at index 0 is found and checked there."""
    support = T0Support.from_spec(_t0_first_spec(), 0.5)
    assert support.index == 0

    breaching = jnp.asarray([jnp.log(0.49), 0.0, 0.0])
    fine = jnp.asarray([jnp.log(0.05), 0.0, 0.0])

    assert bool(support.violates(breaching))
    assert not bool(support.violates(fine))
    assert not bool(support.violates(support.clip(breaching)))


def test_one_breaching_subject_spoils_the_draw(spec):
    support = T0Support.from_spec(spec, jnp.asarray([0.5, 0.5, 0.5]))
    theta = jnp.zeros((3, spec.num_params)).at[:, spec.index("t0")].set(jnp.log(0.1))

    assert not bool(support.violates(theta))
    assert bool(support.violates(theta.at[2, spec.index("t0")].set(jnp.log(0.9))))


def test_clipping_only_moves_the_breaching_entries(spec):
    support = T0Support.from_spec(spec, 0.5)
    theta = jnp.asarray([0.1, 0.2, 0.3, 0.4, jnp.log(0.9)])
    clipped = support.clip(theta)

    assert jnp.allclose(clipped[:-1], theta[:-1])
    assert not bool(support.violates(clipped))


def test_clipping_a_single_vector_against_a_per_subject_cap(spec):
    """`clip` takes every shape `violates` does, including the per-subject cap.

    A shape-(S,) cap against one shape-(P,) vector is what `init_positions_from_prior`
    passes when it is given the support `T0Support.from_spec` builds from a per-subject
    `min_valid_rt`. There is one `t0` for every bound and `violates` clears it only against
    all of them, so the clip has to respect the tightest.
    """
    support = T0Support.from_spec(spec, jnp.asarray([0.5, 0.4, 0.6]))
    theta = jnp.asarray([0.1, 0.2, 0.3, 0.4, jnp.log(0.9)])

    clipped = support.clip(theta)

    assert clipped.shape == theta.shape
    assert jnp.allclose(clipped[:-1], theta[:-1])
    assert not bool(support.violates(clipped))


def test_clipping_a_subject_block_moves_each_subject_to_its_own_cap(spec):
    support = T0Support.from_spec(spec, jnp.asarray([0.5, 0.4, 0.6]))
    index = spec.index("t0")
    theta = jnp.zeros((3, spec.num_params)).at[:, index].set(jnp.log(0.9))

    clipped = support.clip(theta)

    assert not bool(support.violates(clipped))
    assert jnp.allclose(clipped[:, index], jnp.log(0.97 * jnp.asarray([0.5, 0.4, 0.6])))


def test_a_per_subject_cap_survives_the_prior_rejection_path(spec):
    """The combination both `T0Support` and `init_positions_from_prior` document."""
    support = T0Support.from_spec(spec, jnp.asarray([0.5, 0.4, 0.6]))

    def sample_fn(key):
        return jax.random.normal(key, (spec.num_params,)) * 0.5 + jnp.asarray(
            [0.0, 0.7, 0.0, 0.0, jnp.log(0.2)]
        )

    positions, _ = init_positions_from_prior(
        sample_fn, 4, jax.random.key(0), support=support
    )

    assert positions.shape == (4, spec.num_params)
    assert not bool(jax.vmap(support.violates)(positions).any())


# --------------------------------------------------------------------------- #
# rejection_sample
# --------------------------------------------------------------------------- #
def test_every_accepted_draw_satisfies_the_constraint():
    def sample_fn(key):
        return jax.random.uniform(key, (), minval=-1.0, maxval=1.0)

    draws, exhausted = rejection_sample(sample_fn, lambda x: x > 0.0, 64, jax.random.key(0))

    assert jnp.all(draws <= 0.0)
    assert int(exhausted) == 0


def test_an_impossible_constraint_exhausts_and_repairs():
    def sample_fn(key):
        return jax.random.uniform(key, (), minval=1.0, maxval=2.0)

    draws, exhausted = rejection_sample(
        sample_fn, lambda x: x > 0.0, 8, jax.random.key(0),
        max_attempts=2, repair_fn=lambda x: jnp.minimum(x, 0.0),
    )

    assert int(exhausted) == 8
    assert jnp.all(draws <= 0.0)  # the fallback still satisfies the constraint


def test_a_repair_touches_only_the_draws_that_gave_up():
    """`repair_fn` is the exhaustion fallback, not a post-processing step.

    Applying it to every draw would push the accepted ones through the projection too, so
    what comes back is the source distribution *mapped*, not the source distribution
    conditioned on the constraint -- which is the one property the module promises. A
    repair that is not a no-op on satisfying draws makes the difference visible.
    """
    def sample_fn(key):
        return jax.random.uniform(key, (), minval=-1.0, maxval=1.0)

    draws, exhausted = rejection_sample(
        sample_fn, lambda x: x > 0.0, 256, jax.random.key(0),
        repair_fn=lambda x: jnp.full_like(x, -99.0),
    )

    assert int(exhausted) == 0
    assert jnp.all(draws <= 0.0)
    assert not jnp.any(draws == -99.0)


def test_a_repair_still_fires_on_the_draws_that_did_give_up():
    def sample_fn(key):
        return jax.random.uniform(key, (), minval=1.0, maxval=2.0)

    draws, exhausted = rejection_sample(
        sample_fn, lambda x: x > 0.0, 8, jax.random.key(0),
        max_attempts=2, repair_fn=lambda x: jnp.full_like(x, -99.0),
    )

    assert int(exhausted) == 8
    assert jnp.all(draws == -99.0)


def test_without_a_repair_an_exhausted_draw_comes_back_as_drawn():
    def sample_fn(key):
        return jax.random.uniform(key, (), minval=1.0, maxval=2.0)

    draws, exhausted = rejection_sample(
        sample_fn, lambda x: x > 0.0, 4, jax.random.key(0), max_attempts=2
    )

    assert int(exhausted) == 4
    assert jnp.all(draws > 0.0)


# --------------------------------------------------------------------------- #
# Single-subject starts
# --------------------------------------------------------------------------- #
def test_prior_starts_are_dispersed_and_in_support(spec):
    support = T0Support.from_spec(spec, 0.4)

    def sample_fn(key):
        return jax.random.normal(key, (spec.num_params,)) * 0.5 + jnp.asarray(
            [0.0, 0.7, 0.0, 0.0, jnp.log(0.3)]
        )

    positions, exhausted = init_positions_from_prior(sample_fn, 4, jax.random.key(0), support=support)

    assert positions.shape == (4, spec.num_params)
    assert int(exhausted) == 0
    assert not bool(support.violates(positions))
    assert float(jnp.std(positions[:, 0])) > 0.0  # genuinely dispersed, not replicated


def test_jittering_disperses_a_single_position(spec):
    support = T0Support.from_spec(spec, 0.4)
    position = jnp.asarray([0.0, 0.7, 0.0, 0.0, jnp.log(0.2)])

    positions, exhausted = jitter_positions(position, 4, jax.random.key(0), support=support)

    assert positions.shape == (4, spec.num_params)
    assert int(exhausted) == 0
    assert not bool(support.violates(positions))
    assert float(jnp.std(positions[:, 0])) > 0.0


def test_jittering_without_a_support_warns():
    with pytest.warns(UserWarning, match="without a support constraint"):
        jitter_positions(jnp.zeros(3), 2, jax.random.key(0))


def test_no_warning_when_nothing_is_jittered():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jitter_positions(jnp.zeros(3), 2, jax.random.key(0), scale=0.0)


# --------------------------------------------------------------------------- #
# Fixed values
# --------------------------------------------------------------------------- #
def test_t0_is_taken_from_the_data_by_name_not_by_position(spec):
    position, num_exhausted = init_position_from_values(
        [1.0, 2.0, 1.0, 1.0, 99.0], spec=spec, min_rt=0.4, t0_fraction=0.5
    )

    assert float(position[spec.index("t0")]) == pytest.approx(0.2)
    assert int(num_exhausted) == 0


def test_a_leading_bounded_block_is_offset(spec):
    transform = BlockTransform([0.5], [2.5])
    values = [*transform.midpoint(), 1.0, 2.0, 1.0, 1.0, 99.0]

    position, _ = init_position_from_values(values, spec=spec, offset=1, min_rt=0.4)

    assert float(position[1 + spec.index("t0")]) == pytest.approx(0.2)


def test_values_can_be_returned_unconstrained(spec):
    transform = BlockTransform()
    position, _ = init_position_from_values(
        [1.0, 2.0, 1.0, 1.0, 0.2], spec=spec, transform=transform
    )

    assert jnp.allclose(transform.forward(position), jnp.asarray([1.0, 2.0, 1.0, 1.0, 0.2]))


def test_deriving_t0_without_a_way_to_locate_it_is_an_error():
    with pytest.raises(ValueError, match="needs either `spec` or `t0_index`"):
        init_position_from_values([1.0, 2.0], min_rt=0.4)


# --------------------------------------------------------------------------- #
# Hierarchical particles
# --------------------------------------------------------------------------- #
def test_the_mode_alone_is_one_undispersed_point(prior):
    flat_space = prior.flat_space()
    position, exhausted = init_position_from_mode(flat_space)

    assert position.shape == (flat_space.num_flat_params,)
    assert int(exhausted) == 0
    assert jnp.allclose(position, flat_space.mode())


def test_the_mode_replicated_across_chains_has_no_dispersion(prior):
    flat_space = prior.flat_space()
    positions, _ = init_position_from_mode(flat_space, num_chains=4)

    assert positions.shape == (4, flat_space.num_flat_params)
    assert float(jnp.std(positions, axis=0).max()) == 0.0


def test_jittering_the_mode_needs_a_key(prior):
    with pytest.raises(ValueError, match="needs a PRNG key"):
        init_position_from_mode(prior.flat_space(), num_chains=4, jitter=0.1)


def test_every_particle_clears_the_cap_for_every_subject(prior, spec):
    flat_space = prior.flat_space()
    min_rt = jnp.asarray([0.35, 0.40, 0.30])
    support = T0Support(index=NUM_PARAMS - 1, log_t0_max=jnp.log(0.97 * min_rt))

    particles, exhausted = init_particles_from_prior(
        flat_space, 32, jax.random.key(0), support=support
    )

    assert particles.shape == (32, flat_space.num_flat_params)
    assert int(exhausted) == 0

    subject_t0 = jax.vmap(flat_space.subject_params)(particles)[..., support.index]
    assert jnp.all(subject_t0 <= support.log_t0_max)


def test_the_restricted_cloud_is_narrower_in_t0_than_the_prior(prior):
    """The initial cloud is not the prior, and storing it as one corrupts contraction.

    `adaptive_tempered_smc` weights increments by the likelihood alone, so it never
    corrects the initial distribution -- only mutation can. That makes the truncation an
    initialisation artefact worth keeping separate from the prior it was drawn from.
    """
    flat_space = prior.flat_space()
    min_rt = jnp.asarray([0.30, 0.30, 0.30])
    support = T0Support(index=NUM_PARAMS - 1, log_t0_max=jnp.log(0.97 * min_rt))

    unrestricted, _ = init_particles_from_prior(flat_space, 256, jax.random.key(1))
    restricted, _ = init_particles_from_prior(
        flat_space, 256, jax.random.key(1), support=support
    )

    def t0_spread(particles):
        return float(jnp.std(jax.vmap(flat_space.subject_params)(particles)[..., support.index]))

    assert t0_spread(restricted) < 0.95 * t0_spread(unrestricted)


def test_an_unrestricted_cloud_can_breach_the_cap(prior):
    """Which is the whole reason the constraint exists -- and what repo 3 does today."""
    flat_space = prior.flat_space()
    min_rt = jnp.asarray([0.25, 0.25, 0.25])
    support = T0Support(index=NUM_PARAMS - 1, log_t0_max=jnp.log(0.97 * min_rt))

    unrestricted, _ = init_particles_from_prior(flat_space, 256, jax.random.key(2))
    breaches = jax.vmap(lambda p: support.violates(flat_space.subject_params(p)))(unrestricted)

    assert int(jnp.sum(breaches)) > 0
