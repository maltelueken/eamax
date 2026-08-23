"""Unconstraining transforms.

Two things are worth pinning: that the transform is bit-identical to the one it replaces,
and that its log-det-Jacobian agrees with its own forward map. The second could not be
tested in the source repository, because the map and the Jacobian lived in different files
and only a comment held them together.
"""

import jax
import jax.numpy as jnp
import pytest

from eamax._tfp import tfb
from eamax.inference.transforms import (
    BlockTransform,
    simple_to_constrained,
    simple_to_unconstrained,
)

# `[drift_slope_loc, threshold_scale]` from the study-2 models, then the subject block.
META_BOUNDS = ([0.5, 0.05], [2.5, 0.25])
SUBJECT = [1.0, 2.0, 1.0, 1.0, 0.2]


def _reference_to_unconstrained(lower, upper):
    """`eam-abi-robustness/src/mcmc.py:55-74`, verbatim, as the migration reference."""
    bijectors = [tfb().Sigmoid(low=low, high=high) for low, high in zip(lower, upper, strict=True)]

    def to_unconstrained(position):
        position = jnp.asarray(position)
        y_bounded = jnp.stack(
            [bij.inverse(position[..., i]) for i, bij in enumerate(bijectors)], axis=-1
        )
        y_rest = jnp.log(position[..., len(bijectors) :])
        return jnp.concatenate([y_bounded, y_rest], axis=-1)

    return to_unconstrained


def _reference_to_constrained(lower, upper):
    """`eam-abi-robustness/src/mcmc.py:77-89`, verbatim."""
    bijectors = [tfb().Sigmoid(low=low, high=high) for low, high in zip(lower, upper, strict=True)]

    def to_constrained(position):
        position = jnp.asarray(position)
        x_bounded = jnp.stack(
            [bij.forward(position[..., i]) for i, bij in enumerate(bijectors)], axis=-1
        )
        x_rest = jnp.exp(position[..., len(bijectors) :])
        return jnp.concatenate([x_bounded, x_rest], axis=-1)

    return to_constrained


@pytest.fixture
def meta():
    return BlockTransform(*META_BOUNDS)


@pytest.fixture
def natural():
    return jnp.asarray([1.0, 0.1, *SUBJECT])


# --------------------------------------------------------------------------- #
# Migration equivalence
# --------------------------------------------------------------------------- #
def test_it_is_bit_identical_to_the_transform_it_replaces(meta, natural):
    lower, upper = META_BOUNDS

    unconstrained = meta.inverse(natural)
    assert jnp.array_equal(unconstrained, _reference_to_unconstrained(lower, upper)(natural))
    assert jnp.array_equal(
        meta.forward(unconstrained), _reference_to_constrained(lower, upper)(unconstrained)
    )


def test_the_all_positive_case_is_bit_identical_to_log_and_exp(natural):
    assert jnp.array_equal(simple_to_unconstrained(natural), jnp.log(natural))
    assert jnp.array_equal(simple_to_constrained(natural), jnp.exp(natural))


def test_a_one_bound_transform_generalizes_the_two_bound_one():
    # The study-3 models have a single bounded hyperparameter, the study-2 models two.
    # One class, not two factories.
    one = BlockTransform([0.02], [0.18])
    position = jnp.asarray([0.1, *SUBJECT])

    assert one.num_bounded == 1
    assert jnp.allclose(one.forward(one.inverse(position)), position)


# --------------------------------------------------------------------------- #
# Round trips and bounds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bounds", [((), ()), (([0.02],), ([0.18],)), (META_BOUNDS[0], META_BOUNDS[1])])
def test_forward_and_inverse_are_inverses(bounds):
    lower, upper = bounds
    lower = list(lower[0]) if lower and isinstance(lower[0], list) else list(lower)
    upper = list(upper[0]) if upper and isinstance(upper[0], list) else list(upper)

    transform = BlockTransform(lower, upper)
    position = jnp.asarray([*[0.5 * (a + b) for a, b in zip(lower, upper)], *SUBJECT])

    assert jnp.allclose(transform.forward(transform.inverse(position)), position, atol=1e-9)


def test_extreme_unconstrained_values_stay_within_the_bounds(meta):
    """Whatever the sampler proposes, a bounded hyperparameter lands in its support.

    Closed interval, not open: at ``|y| = 40`` the sigmoid saturates to the bound exactly
    in float64. The source repository's own version of this test asserts the same closed
    interval for the same reason.
    """
    lower, upper = META_BOUNDS
    natural = meta.forward(jnp.asarray([-40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert lower[0] <= float(natural[0]) <= upper[0]
    assert lower[1] <= float(natural[1]) <= upper[1]


def test_moderate_unconstrained_values_stay_strictly_inside_the_bounds(meta):
    lower, upper = META_BOUNDS
    natural = meta.forward(jnp.asarray([-4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert lower[0] < float(natural[0]) < upper[0]
    assert lower[1] < float(natural[1]) < upper[1]


def test_it_vectorizes_over_leading_axes(meta, natural):
    # The posterior read-back path transforms a whole (dataset, sample, param) block.
    block = jnp.broadcast_to(natural, (4, 3, natural.size))

    assert jnp.allclose(meta.forward(meta.inverse(block)), block, atol=1e-9)


# --------------------------------------------------------------------------- #
# The Jacobian
# --------------------------------------------------------------------------- #
def test_the_jacobian_matches_an_autodiff_determinant(meta, natural):
    """The check the source repository could not make.

    Its forward map lives in `mcmc.py` and its Jacobian inside each of eight log-density
    factories; nothing compared them, and a comment warned that they must stay in step.
    """
    unconstrained = meta.inverse(natural)
    _, log_det = jnp.linalg.slogdet(jax.jacfwd(meta.forward)(unconstrained))

    assert jnp.allclose(meta.log_det_jacobian(unconstrained), log_det, rtol=1e-6)


def test_the_all_positive_jacobian_is_the_sum_of_the_position(natural):
    transform = BlockTransform()
    unconstrained = transform.inverse(natural)

    assert jnp.allclose(transform.log_det_jacobian(unconstrained), jnp.sum(unconstrained))


def test_the_jacobian_vectorizes_over_leading_axes(meta, natural):
    block = jnp.broadcast_to(meta.inverse(natural), (4, 3, natural.size))

    assert meta.log_det_jacobian(block).shape == (4, 3)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_the_midpoint_comes_from_the_bounds_the_transform_uses(meta):
    assert jnp.allclose(meta.midpoint(), jnp.asarray([1.5, 0.15]))
    assert jnp.all(jnp.isfinite(meta.inverse(jnp.asarray([*meta.midpoint(), *SUBJECT]))))


def test_mismatched_bounds_are_an_error():
    with pytest.raises(ValueError, match="lower bounds but"):
        BlockTransform([0.5], [2.5, 0.25])


def test_a_non_increasing_bound_is_an_error():
    with pytest.raises(ValueError, match="below its upper bound"):
        BlockTransform([2.5], [0.5])
