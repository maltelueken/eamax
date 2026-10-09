"""The parameterization layer: both naming conventions, and where they genuinely differ."""

import jax.numpy as jnp
import numpy as np
import pytest

from eamax.accumulators import LBA, Wald
from eamax.design import (
    TrialDesign,
    build_params_fn,
    effects_label,
    effects_spec,
    rdm_intercept_slope_spec,
    lba_intercept_slope_spec,
    lba_sat_spec,
    parse_effects,
    rdm_sat_spec,
)


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
    spec = rdm_intercept_slope_spec()
    params, t0 = build_params_fn(spec, Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.3])), _design()
    )
    assert np.allclose(np.array(params["v"])[:, 0], [1.0, 2.5])
    assert np.allclose(np.array(params["s"])[:, 0], [1.0, 1.2])
    assert np.allclose(np.array(params["b"]), 1.3)
    assert float(t0) == pytest.approx(0.3)


def test_the_target_covariate_swaps_which_accumulator_is_matching():
    spec = rdm_intercept_slope_spec()
    params, _ = build_params_fn(spec, Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.3])), _design(n=2, target=[1, 2])
    )
    # Trial 0 targets accumulator 1, trial 1 targets accumulator 2.
    assert np.allclose(np.array(params["v"]), [[2.5, 1.0], [1.0, 2.5]])
    assert np.allclose(np.array(params["s"]), [[1.2, 1.0], [1.0, 1.2]])


def test_sat_spec_raises_the_threshold_only_in_the_non_reference_condition():
    # `"sum"` coding: accuracy-instructed trials get `b + b_diff`, so the ordering holds by
    # construction and both parameters stay positive and log-transformable.
    spec = rdm_sat_spec()
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.0, 0.4, 0.3]))
    params, _ = build_params_fn(spec, Wald())(theta, _design(n=2, condition=[1.0, 0.0]))
    assert np.allclose(np.array(params["b"])[:, 0], 1.0)
    assert np.allclose(np.array(params["b"])[:, 1], 1.4)


def test_sat_spec_reduces_to_the_simple_spec_when_every_trial_is_the_reference():
    simple = build_params_fn(rdm_intercept_slope_spec(), Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.0, 0.3])), _design()
    )[0]
    sat = build_params_fn(rdm_sat_spec(), Wald())(
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


def test_lba_sat_spec_adds_the_speed_accuracy_gap_on_top_of_the_lba_boundary():
    # `[v_intercept, v_slope, s_true, A, B, b_diff, t0]`: the gap above A is B for speed
    # (condition 1) and B + b_diff for accuracy (condition 0), so b > A and b_acc > b_speed.
    spec = lba_sat_spec()
    assert spec.names == ("v_intercept", "v_slope", "s_true", "A", "B", "b_diff", "t0")
    theta = jnp.log(jnp.array([2.0, 1.0, 1.2, 0.6, 1.2, 0.5, 0.3]))
    params, _ = build_params_fn(spec, LBA())(theta, _design(n=2, condition=[1.0, 0.0]))
    assert np.allclose(np.array(params["A"]), 0.6)
    assert np.allclose(np.array(params["b"])[:, 0], 1.8)  # speed: A + B
    assert np.allclose(np.array(params["b"])[:, 1], 2.3)  # accuracy: A + B + b_diff
    assert np.all(np.array(params["b"]) > np.array(params["A"]))


def test_lba_sat_spec_reduces_to_the_lba_spec_when_every_trial_is_speed():
    lba = build_params_fn(lba_intercept_slope_spec(), LBA())(
        jnp.log(jnp.array([2.0, 1.0, 1.2, 0.6, 1.2, 0.3])), _design()
    )[0]
    sat = build_params_fn(lba_sat_spec(), LBA())(
        jnp.log(jnp.array([2.0, 1.0, 1.2, 0.6, 1.2, 0.5, 0.3])), _design()
    )[0]
    for name in lba:
        assert np.allclose(np.array(lba[name]), np.array(sat[name]))


def _effects_spec(effects, num_responses=2):
    """The average/difference layout, in the order the hierarchical prior wants."""
    return effects_spec(effects, num_responses)


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
    theta = jnp.zeros((spec.num_params,))
    for name, value in zip(("c_1", "c_2", "c_3"), (0.1, -0.05, 0.2)):
        theta = theta.at[spec.index(name)].set(value)
    theta = theta.at[spec.index("B")].set(jnp.log(1.3))

    design = TrialDesign(rt=jnp.linspace(0.4, 1.5, 5), target=jnp.full((5,), 1))
    params, _ = build_params_fn(spec, Wald())(theta, design)
    per_accumulator = np.array(params["b"])[:, 0]

    assert per_accumulator.shape == (4,)
    # b = B + offset with b_d = 0, so the offsets are b - mean(b) and must sum to zero.
    assert per_accumulator.mean() == pytest.approx(1.3)
    assert (per_accumulator - per_accumulator.mean()).sum() == pytest.approx(0.0)


def test_effects_spec_matches_intercept_slope_on_drift_and_threshold():
    # The two conventions are the same map under different names for the drift and the
    # threshold: V = v_intercept + v_slope/2, v_d = v_slope.
    v_intercept, v_slope, b, t0 = 1.0, 1.5, 1.3, 0.3

    legacy = build_params_fn(rdm_intercept_slope_spec(), Wald())(
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
    # Not a naming difference: `rdm_intercept_slope_spec` pins the *mismatching*
    # accumulator's noise to 1 and frees the matching one, while the average/difference
    # layout pins the *average* to 1 -- which leaves the mismatching accumulator at
    # `1 - s_d/2`, not 1. The two are genuinely different models, expressed as two choices of
    # contrast rather than a flag.
    legacy = build_params_fn(rdm_intercept_slope_spec(), Wald())(
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


def test_an_S_effect_frees_only_the_non_reference_noise():
    spec = effects_spec("S", fixed_noise=1.5)
    assert "S_inc" in spec.names
    assert "S_con" not in spec.names and "S" not in spec.names
    assert spec.num_params == effects_spec("").num_params + 1  # one parameter, like V or B

    theta = jnp.zeros((spec.num_params,)).at[spec.index("S_inc")].set(jnp.log(0.7))
    theta = theta.at[spec.index("s_d")].set(0.2)
    params = build_params_fn(spec, Wald())(theta, _design(n=2, condition=[1.0, 0.0]))[0]
    s = np.array(params["s"])
    assert s[:, 0].mean() == pytest.approx(1.5)  # congruent: the fixed reference
    assert s[:, 1].mean() == pytest.approx(0.7)  # incongruent: estimated
    assert s[1, 0] - s[0, 0] == pytest.approx(0.2)  # s_d still splits the pair


def test_an_S_effect_leaves_the_scale_identified():
    """Rescaling every drift, threshold and noise coefficient must change the likelihood.

    Wald first-passage times depend only on ``b / v`` and ``(b / s)^2``, so if every noise
    coefficient were free, multiplying all of them by a constant would leave the likelihood
    unchanged and the model would have a flat ridge.
    """
    spec = effects_spec("VBS")
    design = _design(n=4, target=[1, 2, 1, 2], condition=[1.0, 1.0, 0.0, 0.0])
    theta = jnp.zeros((spec.num_params,))
    for name, value in [("V_con", 2.0), ("V_inc", 1.8), ("B_con", 1.2), ("B_inc", 1.4), ("S_inc", 0.9)]:
        theta = theta.at[spec.index(name)].set(jnp.log(value))
    theta = theta.at[spec.index("v_d")].set(1.0).at[spec.index("t0")].set(jnp.log(0.3))

    scale = 1.7
    links = np.array(spec.links)
    scaled_names = [n for n in spec.names if n != "t0"]
    rescaled = theta
    for name in scaled_names:
        i = spec.index(name)
        rescaled = rescaled.at[i].set(theta[i] + jnp.log(scale) if links[i] == "log" else theta[i] * scale)

    def wald_shape(t):
        params = build_params_fn(spec, Wald())(t, design)[0]
        return np.array(params["b"] / params["v"]), np.array((params["b"] / params["s"]) ** 2)

    assert not np.allclose(wald_shape(theta)[1], wald_shape(rescaled)[1])


def test_a_spec_missing_a_quantity_the_accumulator_needs_fails_at_bind_time():
    # Not at trace time, and not as a silently wrong density.
    with pytest.raises(KeyError, match="missing"):
        build_params_fn(rdm_intercept_slope_spec(), LBA())


def test_pulsed_conflict_spec_routes_the_pulse_to_the_distractor_accumulator():
    # The conflict pulse rides exactly one accumulator per trial -- the one the distractor
    # selects -- which the previous engine could not express (it broadcast `amp` uniformly
    # and never read `distractor`).
    from eamax.accumulators import VolterraPulsedWald
    from eamax.design import pulsed_conflict_spec

    spec = pulsed_conflict_spec()
    design = TrialDesign(
        rt=jnp.linspace(0.4, 1.5, 3),
        response=jnp.ones((3,), dtype=int),
        target=jnp.full((3,), 2),
        distractor=jnp.array([1, 2, 1]),
    )
    theta = jnp.log(jnp.array([1.0, 1.0, 1.3, 0.3, 0.1, 0.3]))
    params, _ = build_params_fn(spec, VolterraPulsedWald(dt=1e-2, t_max=2.0))(theta, design)

    amp = np.array(params["amp"])
    # amp == 0.3 exactly where the accumulator index equals the trial's distractor, else 0.
    assert np.allclose(amp, [[0.3, 0.0, 0.3], [0.0, 0.3, 0.0]])


# --------------------------------------------------------------------------- #
# Links: the string view of a bijector, and what it can round-trip
# --------------------------------------------------------------------------- #
def test_importing_eamax_does_not_import_tfp():
    """`eamax._tfp` exists so that TFP is a peer, not a hard requirement of `import eamax`.

    Run in a subprocess with `tensorflow_probability` blocked: `eamax.design` is on the
    package's own import path, so a single module-level `tfb()` there -- the natural way to
    cache the `Exp` class for a link-name lookup -- makes plain `import eamax` fail outright
    for anyone who only wants the closed-form Wald and LBA densities, and costs everyone
    else TFP's multi-second import.
    """
    import subprocess
    import sys

    script = """
import sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "tensorflow_probability":
            raise ImportError("blocked")
        return None

sys.meta_path.insert(0, Block())

import eamax
import eamax.design
from eamax.accumulators import wald, lba

assert not any(k.startswith("tensorflow_probability") for k in sys.modules)
"""
    assert subprocess.run([sys.executable, "-c", script]).returncode == 0


def test_link_names_round_trip_through_of_names():
    from eamax.design import Parameterization

    spec = Parameterization.of_names(("v", "b", "shift"), ("log", "log", "identity"))

    assert spec.links == ("log", "log", "identity")


def test_a_bijector_with_no_link_name_raises_rather_than_reporting_identity():
    """Reporting a `Softplus` as `"identity"` would rebuild it as `Identity()`.

    `of_names(spec.names, spec.links)` is the documented way to reconstruct a reporting
    spec for `load_dataset_posterior`, so a wrong name there back-transforms stored samples
    on the wrong link -- finite, plausible, wrong numbers and no error.
    """
    from eamax._tfp import tfb
    from eamax.design import coef, intercept, link_name, parameterization, quantity, term

    with pytest.raises(ValueError, match="No link name"):
        link_name(tfb().Softplus())

    v = coef("v", tfb().Softplus())
    spec = parameterization(
        order=[v], quantities=[quantity("v", term(v, intercept()))], num_responses=2
    )

    with pytest.raises(ValueError, match="No link name"):
        spec.links


def test_a_bijector_with_no_link_name_still_constrains_through_its_own_forward():
    """Only the *name* is unavailable; the bijector itself is used directly."""
    from eamax._tfp import tfb
    from eamax.design import coef, intercept, parameterization, quantity, term

    v = coef("v", tfb().Softplus())
    spec = parameterization(
        order=[v], quantities=[quantity("v", term(v, intercept()))], num_responses=2
    )
    theta = jnp.asarray([0.3])

    assert np.allclose(spec.constrain(theta), tfb().Softplus().forward(theta))


def test_of_names_rejects_name_and_link_lists_of_different_lengths():
    """Zipping would truncate to the shorter and return a self-consistent wrong spec.

    Nothing downstream catches it: `select_params`' own length check compares against the
    truncated `names`, so the samples are back-transformed and selected on the links of
    other coefficients.
    """
    from eamax.design import Parameterization

    with pytest.raises(ValueError, match="3 names but 2 links"):
        Parameterization.of_names(("v", "b", "t0"), ("log", "log"))


def test_constrain_matches_the_per_coefficient_forward_on_a_mixed_spec():
    """The vectorized log/identity path must agree with applying each bijector in turn."""
    from eamax.design import Parameterization

    spec = Parameterization.of_names(("v", "b", "shift"), ("log", "log", "identity"))
    theta = jnp.asarray([[0.3, -1.2, 2.0], [-0.5, 0.1, -3.0]])

    expected = jnp.stack(
        [c.bijector.forward(theta[..., i]) for i, c in enumerate(spec.order)], axis=-1
    )

    assert np.allclose(spec.constrain(theta), expected)


def test_constrain_has_finite_gradients_for_a_large_identity_linked_value():
    """The vectorized path exponentiates under a mask; the discarded branch must not be inf.

    `constrain` is on the gradient path via `accumulator_params`, and a masked-out `inf`
    gives `0 * inf` in the backward pass -- a NaN gradient from a value that is perfectly
    ordinary on its own link.
    """
    import jax

    from eamax.design import Parameterization

    spec = Parameterization.of_names(("v", "shift"), ("log", "identity"))
    grad = jax.grad(lambda theta: jnp.sum(spec.constrain(theta)))(
        jnp.asarray([0.5, 800.0])
    )

    assert bool(jnp.all(jnp.isfinite(grad)))


def test_a_scalar_covariate_is_the_same_as_a_constant_column():
    # A scalar `target` says "the same on every trial"; broadcast parameters must not be able
    # to tell it from the equivalent `(T,)` column.
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.3]))
    params_fn = build_params_fn(rdm_intercept_slope_spec(), Wald())
    column, _ = params_fn(theta, _design())
    scalar, _ = params_fn(theta, _design().replace(target=2))
    for name in column:
        assert scalar[name].shape == column[name].shape == (2, 6)
        assert np.array_equal(np.array(scalar[name]), np.array(column[name]))


def test_unbroadcast_quantities_are_trial_invariant_only_by_shape():
    # `broadcast=False` keeps a length-1 trial axis only where nothing a quantity reads
    # varies by trial. A per-trial covariate must still produce `(N, T)`, or an accumulator
    # that trusts the shape would score every trial with the first trial's parameters.
    from eamax.design import accumulator_params

    spec = rdm_sat_spec()
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.4, 0.3]))
    design = _design(n=4, condition=[1.0, 0.0, 1.0, 0.0]).replace(target=2)

    unbroadcast, _ = accumulator_params(spec, theta, design, broadcast=False)
    broadcast, _ = accumulator_params(spec, theta, design)

    assert unbroadcast["v"].shape == unbroadcast["s"].shape == (2, 1)
    assert unbroadcast["b"].shape == (2, 4)  # reads `condition`, which varies by trial
    for name in broadcast:
        assert broadcast[name].shape == (2, 4)
        assert np.array_equal(
            np.array(jnp.broadcast_to(unbroadcast[name], (2, 4))), np.array(broadcast[name])
        )

    per_trial_target, _ = accumulator_params(
        spec, theta, design.replace(target=jnp.array([2, 1, 2, 1])), broadcast=False
    )
    assert per_trial_target["v"].shape == (2, 4)
