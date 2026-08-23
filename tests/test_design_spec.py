"""The parameterization layer: both naming conventions, and where they genuinely differ."""

import jax.numpy as jnp
import numpy as np
import pytest

from eamax.accumulators import LBA, Wald
from eamax.design import (
    ParamSpecBuilder,
    TrialDesign,
    build_params_fn,
    effects_label,
    intercept_slope_spec,
    parse_effects,
    sat_spec,
)
from eamax.design.legacy import lba_intercept_slope_spec


def _design(n=6, target=None, condition=None):
    return TrialDesign(
        rt=jnp.linspace(0.4, 1.5, n),
        response=jnp.ones((n,), dtype=int),
        target=jnp.full((n,), 2) if target is None else jnp.asarray(target),
        condition=jnp.ones((n,)) if condition is None else jnp.asarray(condition),
    )


def test_intercept_slope_spec_reproduces_the_two_accumulator_convention():
    # Accumulator 1 is the non-target: drift `v_intercept`, noise fixed to 1.
    # Accumulator 2 is the target: drift `v_intercept + v_slope`, noise `s_true`.
    spec = intercept_slope_spec()
    params, t0 = build_params_fn(spec, Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.3])), _design()
    )
    assert np.allclose(np.array(params["v"])[:, 0], [1.0, 2.5])
    assert np.allclose(np.array(params["s"])[:, 0], [1.0, 1.2])
    assert np.allclose(np.array(params["b"]), 1.3)
    assert float(t0) == pytest.approx(0.3)


def test_the_target_covariate_swaps_which_accumulator_is_matching():
    spec = intercept_slope_spec()
    params, _ = build_params_fn(spec, Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.3])), _design(n=2, target=[1, 2])
    )
    # Trial 0 targets accumulator 1, trial 1 targets accumulator 2.
    assert np.allclose(np.array(params["v"]), [[2.5, 1.0], [1.0, 2.5]])
    assert np.allclose(np.array(params["s"]), [[1.2, 1.0], [1.0, 1.2]])


def test_sat_spec_raises_the_threshold_only_in_the_non_reference_condition():
    # `"sum"` coding: accuracy-instructed trials get `b + b_diff`, so the ordering holds by
    # construction and both parameters stay positive and log-transformable.
    spec = sat_spec()
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.0, 0.4, 0.3]))
    params, _ = build_params_fn(spec, Wald())(theta, _design(n=2, condition=[1.0, 0.0]))
    assert np.allclose(np.array(params["b"])[:, 0], 1.0)
    assert np.allclose(np.array(params["b"])[:, 1], 1.4)


def test_sat_spec_reduces_to_the_simple_spec_when_every_trial_is_the_reference():
    simple = build_params_fn(intercept_slope_spec(), Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.0, 0.3])), _design()
    )[0]
    sat = build_params_fn(sat_spec(), Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.0, 0.4, 0.3])), _design()
    )[0]
    for name in simple:
        assert np.allclose(np.array(simple[name]), np.array(sat[name]))


def test_lba_spec_treats_the_threshold_as_a_gap_above_the_start_point():
    spec = lba_intercept_slope_spec()
    params, _ = build_params_fn(spec, LBA())(
        jnp.log(jnp.array([2.0, 1.0, 1.2, 0.6, 1.2, 0.3])), _design()
    )
    assert np.allclose(np.array(params["A"]), 0.6)
    assert np.allclose(np.array(params["b"]), 1.8)  # A + B, so b > A by construction
    assert np.all(np.array(params["b"]) > np.array(params["A"]))


def _effects_spec(effects, num_responses=2):
    """The average/difference layout, in the order the hierarchical prior wants."""
    builder = ParamSpecBuilder()
    builder.add_quantity("V", effects)
    builder.add_quantity("v_d", effects)
    builder.add_quantity("b_d", effects)
    builder.add_response_offsets(num_responses)
    builder.add_quantity("s_d", effects)
    if "S" in effects:
        builder.add_quantity("S", effects)
    else:
        builder.fix("S", 1.0)
    builder.add_quantity("B", effects)
    builder.add("t0", "t0")
    return builder.finalize(num_responses, num_centered=(2 if "B" in effects else 1) + 1)


def test_effects_codes_round_trip():
    assert parse_effects("VBs") == {"V", "B", "s_d"}
    assert effects_label({"V", "B", "s_d"}) == "VBs"
    assert effects_label(set()) == "baseline"
    with pytest.raises(ValueError, match="Unknown effect code"):
        parse_effects("VX")


def test_a_condition_effect_adds_exactly_one_parameter_pair():
    baseline = _effects_spec(set())
    with_v = _effects_spec({"V"})
    assert with_v.num_params == baseline.num_params + 1
    assert "V_con" in with_v.names and "V_inc" in with_v.names
    assert "V" in baseline.names


def test_difference_terms_are_signed_and_averages_are_positive():
    # The whole point of estimating a difference is to learn its sign, so the differences
    # take the identity link while the averages take the exp link.
    spec = _effects_spec({"V", "s_d"})
    links = dict(zip(spec.names, spec.links))
    assert links["V_con"] == "log"
    assert links["v_d"] == "identity"
    assert links["s_d_con"] == "identity"
    assert links["c_1"] == "identity"


def test_response_offsets_sum_to_zero_across_accumulators():
    # The per-accumulator baseline thresholds must average to the shared threshold exactly,
    # which is what makes the offsets a bias rather than a second threshold parameter.
    spec = _effects_spec(set(), num_responses=4)
    theta = jnp.zeros((spec.num_params,)).at[jnp.asarray(spec.response_offsets)].set(
        jnp.array([0.1, -0.05, 0.2])
    )
    from eamax.design.map import response_offsets

    offsets = np.array(response_offsets(spec.constrain(theta), spec))
    assert offsets.shape == (4,)
    assert offsets.sum() == pytest.approx(0.0)


def test_effects_spec_matches_intercept_slope_on_drift_and_threshold():
    # The two conventions are the same map under different names for the drift and the
    # threshold: V = v_intercept + v_slope/2, v_d = v_slope.
    v_intercept, v_slope, b, t0 = 1.0, 1.5, 1.3, 0.3

    legacy = build_params_fn(intercept_slope_spec(), Wald())(
        jnp.log(jnp.array([v_intercept, v_slope, 1.0, b, t0])), _design()
    )[0]

    spec = _effects_spec(set())
    theta = jnp.zeros((spec.num_params,))
    theta = theta.at[spec.index("V")].set(jnp.log(v_intercept + v_slope / 2))
    theta = theta.at[spec.index("v_d")].set(v_slope)
    theta = theta.at[spec.index("B")].set(jnp.log(b))
    theta = theta.at[spec.index("t0")].set(jnp.log(t0))
    effects = build_params_fn(spec, Wald())(theta, _design())[0]

    assert np.allclose(np.array(legacy["v"]), np.array(effects["v"]))
    assert np.allclose(np.array(legacy["b"]), np.array(effects["b"]))


def test_the_two_conventions_pin_a_different_noise():
    # Not a naming difference: `intercept_slope_spec` pins the *mismatching* accumulator's
    # noise to 1 and frees the matching one, while the average/difference layout pins the
    # *average* to 1 -- which leaves the mismatching accumulator at `1 - s_d/2`, not 1.
    # This is a modelling choice that was made implicitly in the source repositories; making
    # it explicit is the point of `noise_reference`.
    legacy = build_params_fn(intercept_slope_spec(), Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.3])), _design()
    )[0]
    assert np.allclose(np.array(legacy["s"])[:, 0], [1.0, 1.2])  # mismatch pinned

    spec = _effects_spec({"s_d"})
    theta = jnp.zeros((spec.num_params,))
    theta = theta.at[spec.index("s_d_con")].set(0.2)  # s_match - s_mismatch
    theta = theta.at[spec.index("V")].set(jnp.log(1.75))
    theta = theta.at[spec.index("B")].set(jnp.log(1.3))
    theta = theta.at[spec.index("t0")].set(jnp.log(0.3))
    effects = build_params_fn(spec, Wald())(theta, _design())[0]

    # Average pinned to 1, so the pair straddles 1 rather than starting at it.
    assert np.allclose(np.array(effects["s"])[:, 0], [0.9, 1.1])


def test_a_spec_missing_a_quantity_the_accumulator_needs_fails_at_bind_time():
    # Not at trace time, and not as a silently wrong density.
    with pytest.raises(KeyError, match="missing"):
        build_params_fn(intercept_slope_spec(), LBA())
