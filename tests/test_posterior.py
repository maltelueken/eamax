"""Tests for the array-level posterior helpers.

The load-bearing one is :func:`back_transform_then_select`. The two steps do not commute;
the test below builds a case where the wrong order returns finite, plausible, wrong numbers
rather than failing.
"""

import numpy as np
import pytest

from eamax.inference.posterior import (
    back_transform_then_select,
    pool_chains,
    select_params,
    thin,
    to_arviz_layout,
)
from eamax.inference.transforms import BlockTransform


# ---------------------------------------------------------------------------
# pool_chains
# ---------------------------------------------------------------------------
def test_pool_chains_merges_the_two_sampling_axes():
    samples = np.zeros((4, 200, 5, 3))

    assert pool_chains(samples).shape == (5, 800, 3)


def test_pool_chains_is_chain_major():
    """Every draw of chain 0, then every draw of chain 1 -- xarray's own stack order.

    Load-bearing: a pooled array and a pooled `Dataset` must index the same way, or draws
    matched per-index against anything else are matched against the wrong draw.
    """
    num_chains, num_draws = 2, 8
    # (chain, draw, dataset, param), value = chain * num_draws + draw.
    samples = (
        np.arange(num_chains * num_draws)
        .reshape(num_chains, num_draws, 1, 1)
        .astype(float)
    )

    assert pool_chains(samples)[0, :, 0].tolist() == [float(v) for v in range(16)]


def test_pool_chains_preserves_every_element():
    samples = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(float)

    assert sorted(pool_chains(samples).ravel()) == sorted(samples.ravel())


@pytest.mark.parametrize("shape", [(4, 200, 3), (4, 200, 5, 3, 2)])
def test_pool_chains_rejects_the_wrong_rank(shape):
    with pytest.raises(ValueError, match="rank"):
        pool_chains(np.zeros(shape))


# ---------------------------------------------------------------------------
# to_arviz_layout
# ---------------------------------------------------------------------------


def test_to_arviz_layout_moves_each_axis_where_documented():
    positions = np.arange(2 * 7 * 3 * 5).reshape(2, 7, 3, 5)

    out = to_arviz_layout(positions)

    assert out.shape == (3, 7, 2, 5)


def test_to_arviz_layout_preserves_every_element():
    positions = np.arange(2 * 7 * 3 * 5).reshape(2, 7, 3, 5)

    out = to_arviz_layout(positions)

    for dataset in range(2):
        for draw in range(7):
            for chain in range(3):
                assert np.array_equal(
                    out[chain, draw, dataset], positions[dataset, draw, chain]
                )


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 3, 4, 5, 6), (7,)])
def test_to_arviz_layout_rejects_the_wrong_rank(shape):
    with pytest.raises(ValueError, match="rank"):
        to_arviz_layout(np.zeros(shape))


# ---------------------------------------------------------------------------
# thin
# ---------------------------------------------------------------------------


def test_thin_returns_a_short_axis_whole():
    samples = np.arange(30).reshape(1, 30, 1)

    assert thin(samples, 200).shape == (1, 30, 1)


def test_thin_caps_at_the_target():
    for length in (399, 400, 401, 4000):
        out = thin(np.arange(length).reshape(1, length, 1), 200)
        assert out.shape[1] <= 200, length


def test_thin_strides_rather_than_truncating():
    samples = np.arange(1000).reshape(1, 1000, 1)

    out = thin(samples, 100)[0, :, 0]

    # A truncation would end at 99; a stride of 10 spans the whole chain.
    assert out[0] == 0
    assert out[-1] >= 900


def test_thin_uses_an_integer_stride():
    out = thin(np.arange(1000).reshape(1, 1000, 1), 100)[0, :, 0]

    assert np.array_equal(np.diff(out), np.full(len(out) - 1, 10))


def test_thin_honours_the_axis_argument():
    samples = np.arange(1000).reshape(1000, 1, 1)

    assert thin(samples, 100, axis=0).shape == (100, 1, 1)


def test_thin_honours_a_negative_axis():
    samples = np.arange(1000).reshape(1, 1, 1000)

    assert thin(samples, 100, axis=-1).shape == (1, 1, 100)


@pytest.mark.parametrize("num_target", [0, -1])
def test_thin_rejects_a_nonpositive_target(num_target):
    with pytest.raises(ValueError, match="positive"):
        thin(np.zeros((1, 10, 1)), num_target)


# ---------------------------------------------------------------------------
# select_params
# ---------------------------------------------------------------------------


def test_select_params_reorders_by_name():
    samples = np.arange(6).reshape(2, 3)

    out = select_params(samples, ["a", "b", "c"], ["c", "a"])

    assert np.array_equal(out, np.array([[2, 0], [5, 3]]))


def test_select_params_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="not among the stored"):
        select_params(np.zeros((2, 3)), ["a", "b", "c"], ["d"])


def test_select_params_rejects_a_name_list_of_the_wrong_length():
    with pytest.raises(ValueError, match="names were given"):
        select_params(np.zeros((2, 3)), ["a", "b"], ["a"])


# ---------------------------------------------------------------------------
# back_transform_then_select -- the order that matters
# ---------------------------------------------------------------------------


def _hierarchical_case():
    """A layout where the two orders differ: two bounded hyperparameters, then three positive.

    A hierarchical layout where the subset wanted is the subject-level block only, which is
    what an NPE comparison asks for.
    """
    transform = BlockTransform(lower=[0.0, 0.0], upper=[1.0, 1.0])
    stored = ["p_1", "p_2", "v", "b", "t0"]
    wanted = ["v", "b", "t0"]
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(4, 6, len(stored)))
    return transform, stored, wanted, samples


def test_back_transform_then_select_matches_transforming_the_full_vector():
    transform, stored, wanted, samples = _hierarchical_case()

    out = back_transform_then_select(samples, transform.forward, stored, wanted)

    expected = np.asarray(transform.forward(samples))[..., 2:]
    assert np.allclose(out, expected)


def test_the_wrong_order_is_wrong_but_not_an_error():
    """Selecting first hands the transform a vector whose block boundary has moved."""
    transform, stored, wanted, samples = _hierarchical_case()

    correct = back_transform_then_select(samples, transform.forward, stored, wanted)

    selected_first = np.asarray(transform.forward(samples[..., 2:]))

    # No exception, no NaN, right shape -- and different numbers.
    assert selected_first.shape == correct.shape
    assert np.all(np.isfinite(selected_first))
    assert not np.allclose(selected_first, correct)


def test_the_wrong_order_applies_sigmoid_to_positive_parameters():
    """Naming the specific corruption, not just that the two differ."""
    transform, stored, wanted, samples = _hierarchical_case()

    selected_first = np.asarray(transform.forward(samples[..., 2:]))

    # `v` and `b` are strictly positive and unbounded above, but the shifted boundary
    # squashes them into the hyperparameters' (0, 1) support.
    assert np.all(selected_first[..., :2] < 1.0)

    correct = back_transform_then_select(samples, transform.forward, stored, wanted)
    assert np.any(correct[..., :2] > 1.0)


def test_back_transform_then_select_is_a_no_op_order_for_the_all_positive_case():
    """With no bounded block the two orders agree -- which is why the bug hides."""
    transform = BlockTransform()
    stored = ["v", "b", "t0"]
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(3, 5, 3))

    correct = back_transform_then_select(samples, transform.forward, stored, ["b", "t0"])
    selected_first = np.asarray(transform.forward(samples[..., 1:]))

    assert np.allclose(correct, selected_first)


def test_back_transform_then_select_propagates_the_name_check():
    transform, stored, _, samples = _hierarchical_case()

    with pytest.raises(ValueError, match="not among the stored"):
        back_transform_then_select(samples, transform.forward, stored, ["nope"])


def test_back_transform_then_select_accepts_a_param_spec_constrain():
    """The other transform a consumer might pass: `Parameterization.constrain`."""
    from eamax.design import Parameterization

    spec = Parameterization.of_names(("v", "b", "t0"), ("log", "log", "log"))
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(2, 4, 3))

    out = back_transform_then_select(samples, spec.constrain, spec.names, ["t0", "v"])

    assert np.allclose(out[..., 0], np.exp(samples[..., 2]))
    assert np.allclose(out[..., 1], np.exp(samples[..., 0]))
