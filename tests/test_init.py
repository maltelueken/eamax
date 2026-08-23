"""Starting values.

The constraint these tests are about is not a nicety. A chain started at ``t0 >= min(rt)``
begins on the race likelihood's penalty region, where the gradient is order 1e3 and step
size adaptation has nothing useful to do. One of the four source routines enforces this by
assignment, two by rejection, and one not at all.
"""

import jax
import jax.numpy as jnp
import pytest

from eamax.design import ParamSpecBuilder
from eamax.design.legacy import intercept_slope_spec
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
    """A spec whose `t0` is at index 0, which no source repository can express."""
    builder = ParamSpecBuilder()
    builder.add("t0", "t0")
    builder.add_quantity("V")
    builder.add_quantity("B")
    return builder.finalize(num_responses=2)


@pytest.fixture
def spec():
    return intercept_slope_spec()


@pytest.fixture
def prior():
    # `t0` sits last, so it lands in the trailing centered block -- the arrangement the
    # source repositories assume throughout.
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


# --------------------------------------------------------------------------- #
# T0Support
# --------------------------------------------------------------------------- #
def test_it_finds_t0_by_name(spec):
    support = T0Support.from_spec(spec, 0.5)

    assert support.index == spec.index("t0")
    assert float(support.log_t0_max) == pytest.approx(float(jnp.log(0.97 * 0.5)))


def test_a_spec_without_t0_is_an_error():
    builder = ParamSpecBuilder()
    builder.add_quantity("V")

    with pytest.raises(ValueError, match="needs a parameter named 't0'"):
        T0Support.from_spec(builder.finalize(num_responses=2), 0.5)


def test_a_t0_on_the_wrong_link_is_an_error():
    """The cap compares an unconstrained value, so it is only `log(t0)` under the log link."""
    builder = ParamSpecBuilder(identity_qtypes={"t0"})
    builder.add("t0", "t0")

    with pytest.raises(ValueError, match="must be on the 'log' link"):
        T0Support.from_spec(builder.finalize(num_responses=2), 0.5)


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
    position = init_position_from_values(
        [1.0, 2.0, 1.0, 1.0, 99.0], spec=spec, min_rt=0.4, t0_fraction=0.5
    )

    assert float(position[spec.index("t0")]) == pytest.approx(0.2)


def test_a_leading_bounded_block_is_offset(spec):
    transform = BlockTransform([0.5], [2.5])
    values = [*transform.midpoint(), 1.0, 2.0, 1.0, 1.0, 99.0]

    position = init_position_from_values(values, spec=spec, offset=1, min_rt=0.4)

    assert float(position[1 + spec.index("t0")]) == pytest.approx(0.2)


def test_values_can_be_returned_unconstrained(spec):
    transform = BlockTransform()
    position = init_position_from_values(
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
