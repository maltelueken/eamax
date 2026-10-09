"""The neural density estimator, and the checkpoint metadata that keeps it honest."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("distrax")
pytest.importorskip("flax")

import optax  # noqa: E402
from flax import nnx  # noqa: E402

from eamax import race_loglik  # noqa: E402
from eamax.accumulators import Wald  # noqa: E402
from eamax.flows import (  # noqa: E402
    MIN_SCALE,
    MLP,
    DeepMLP,
    FlowAccumulator,
    Maximum,
    conditioner_layout,
    context_bounds,
    evaluate_pdf_sf,
    input_scaling,
    load_conditioner,
    log_input_scaling,
    loss_fn,
    make_mlp_conditioner,
    num_hidden_layers,
    read_metadata,
    save_conditioner,
    spline_flow,
    spline_knots,
    spline_settings,
    train_step,
    write_metadata,
)
from eamax.flows import model as flow_model  # noqa: E402

CONTEXT = ("v", "s", "b")


def _conditioner(num_in=3, num_mid=32, num_bins=8, seed=0):
    return make_mlp_conditioner(nnx.Rngs(seed), num_in=num_in, num_mid=num_mid, num_bins=num_bins)


def _params(n, v=2.0, s=1.0, b=1.0):
    return {"v": jnp.full((n,), v), "s": jnp.full((n,), s), "b": jnp.full((n,), b)}


def test_survival_is_exact_given_the_flow():
    # The property the whole design rests on: both the spline and `exp` are strictly
    # increasing, so {T > t} and {Z > z} are the same event and S(t) is the standard-normal
    # survival at the flow inverse -- not a second approximation stacked on the density.
    # Checking it against a numerical integral of the flow's own density pins that claim.
    conditioner = _conditioner()
    context = jnp.array([[2.0, 1.0, 1.0]])

    grid = jnp.linspace(1e-4, 40.0, 400_000)
    log_pdf, _ = spline_flow(grid, jnp.repeat(context, grid.size, axis=0), conditioner)
    cumulative = jnp.cumsum(jnp.exp(log_pdf)) * (grid[1] - grid[0])

    for target in (0.5, 1.0, 3.0):
        index = int(jnp.searchsorted(grid, target))
        reported = float(jnp.exp(evaluate_pdf_sf(conditioner, grid[index : index + 1], context)[1][0]))
        integrated = float(1.0 - cumulative[index])
        assert reported == pytest.approx(integrated, abs=2e-3)


def test_flow_accumulator_satisfies_the_accumulator_interface():
    accumulator = FlowAccumulator(_conditioner(), CONTEXT)
    assert accumulator.param_names == CONTEXT

    t = jnp.linspace(0.1, 2.0, 16)
    log_pdf, log_sf = accumulator.log_pdf_sf(t, _params(16))
    assert log_pdf.shape == t.shape and log_sf.shape == t.shape
    assert np.all(np.isfinite(np.array(log_pdf)))
    # The likelihood runs in float64 even though the flow computes in float32.
    assert log_pdf.dtype == t.dtype


def test_a_flow_accumulator_drops_straight_into_the_race():
    # The payoff of the shared protocol: no race code knows or cares that this density is
    # learned rather than closed-form.
    accumulator = FlowAccumulator(_conditioner(), CONTEXT)
    params = {k: jnp.stack([v, v]) for k, v in _params(20).items()}
    out = race_loglik(
        jnp.linspace(0.5, 2.0, 20),
        jnp.zeros((20,), dtype=int),
        0.2,
        lambda t: accumulator.log_pdf_sf(t, params),
    )
    assert out.shape == (20,)
    assert np.all(np.isfinite(np.array(out)))


def test_context_columns_follow_the_declared_order():
    # A reordered context of the same width is the failure mode a checkpoint cannot detect,
    # so the ordering must be driven by `context_names` and nothing else.
    params = {"v": jnp.array([1.0]), "s": jnp.array([2.0]), "b": jnp.array([3.0])}
    assert np.allclose(np.array(FlowAccumulator(_conditioner(), ("v", "s", "b")).build_context(params)), [[1.0, 2.0, 3.0]])
    assert np.allclose(np.array(FlowAccumulator(_conditioner(), ("b", "v", "s")).build_context(params)), [[3.0, 1.0, 2.0]])


def test_a_context_transform_is_applied_before_stacking():
    # The pulsed flow trains on |amp|, because the pulse's sign selects which accumulator
    # carries it rather than changing its shape.
    accumulator = FlowAccumulator(_conditioner(5), ("v", "amp", "tau", "s", "b"), transform={"amp": jnp.abs})
    context = accumulator.build_context(
        {"v": jnp.array([1.0]), "amp": jnp.array([-0.3]), "tau": jnp.array([0.1]),
         "s": jnp.array([1.0]), "b": jnp.array([1.0])}
    )
    assert float(np.array(context)[0, 1]) == pytest.approx(0.3)


def test_censored_and_uncensored_losses_agree_when_nothing_is_censored():
    conditioner = _conditioner()
    data = jnp.linspace(0.2, 2.0, 32)
    context = jnp.repeat(jnp.array([[2.0, 1.0, 1.0]]), 32, axis=0)
    assert float(loss_fn(conditioner, data, context, None)) == float(
        loss_fn(conditioner, data, context, 4.0)
    )


def test_censored_trials_are_scored_by_the_survival_not_dropped():
    # Dropping them fits p(t | T < t_max) instead of p(t), which biases the posterior in a
    # parameter-dependent way that does not cancel out of a likelihood ratio.
    conditioner = _conditioner()
    context = jnp.repeat(jnp.array([[2.0, 1.0, 1.0]]), 4, axis=0)
    clean = jnp.array([0.5, 0.8, 1.2, 1.5])
    censored = clean.at[3].set(-1.0)

    assert float(loss_fn(conditioner, censored, context, 4.0)) != float(
        loss_fn(conditioner, clean, context, 4.0)
    )
    # A censored trial must still contribute a finite, differentiable term.
    grad = jax.grad(lambda c: loss_fn(c, censored, context, 4.0))(conditioner)
    assert np.all(np.isfinite(np.array(grad.linear1.kernel[...])))


def test_a_checkpoint_round_trips_and_records_its_context(tmp_path):
    conditioner = _conditioner(num_mid=16, num_bins=4, seed=3)
    path = tmp_path / "conditioner"
    save_conditioner(conditioner, str(path), context_names=CONTEXT, num_mid=16, num_bins=4)

    fresh = _conditioner(num_mid=16, num_bins=4, seed=99)
    restored = load_conditioner(fresh, str(path), context_names=CONTEXT)

    t = jnp.linspace(0.2, 2.0, 8)
    context = jnp.repeat(jnp.array([[2.0, 1.0, 1.0]]), 8, axis=0)
    assert np.allclose(
        np.array(evaluate_pdf_sf(conditioner, t, context)[0]),
        np.array(evaluate_pdf_sf(restored, t, context)[0]),
    )


def test_loading_with_a_reordered_context_is_refused(tmp_path):
    # Without the sidecar this is the silent-wrong-density case: same width, different
    # meaning, no error from Orbax.
    conditioner = _conditioner(num_mid=16, num_bins=4)
    path = tmp_path / "conditioner"
    save_conditioner(conditioner, str(path), context_names=("v", "s", "b"))

    with pytest.raises(ValueError, match="silently wrong densities"):
        load_conditioner(_conditioner(num_mid=16, num_bins=4), str(path), context_names=("b", "v", "s"))


def _flow_race(accumulator, spec, theta, design):
    from eamax.design import build_params_fn

    params, t0 = build_params_fn(spec, accumulator)(theta, design)
    loglik = race_loglik(
        design.rt, design.response, t0, lambda t: accumulator.log_pdf_sf(t, params),
        first_response=spec.first_response,
    )
    return jnp.sum(loglik), params


def test_a_trial_invariant_flow_runs_its_conditioner_once_per_accumulator():
    # The memory regression this guards: with `(N, T)` parameters the MLP ran on every trial,
    # and inside a 1000-particle hierarchical SMC cloud its activations did not fit on a GPU.
    # A scalar target has to reach the conditioner as `(N, 1, num_in)` and still produce the
    # same log-likelihood and gradient as the per-trial path.
    from eamax.design import TrialDesign, rdm_intercept_slope_spec

    spec = rdm_intercept_slope_spec()
    accumulator = FlowAccumulator(_conditioner(), CONTEXT, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    rt = jnp.array(rng.uniform(0.5, 2.0, 40))
    response = jnp.array(rng.integers(1, 3, 40))
    per_trial = TrialDesign(rt=rt, response=response, target=jnp.full((40,), 2))
    invariant = per_trial.replace(target=2)
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.2]))

    _, params = _flow_race(accumulator, spec, theta, invariant)
    assert accumulator.build_context(params).shape == (2, 1, 3)
    _, params = _flow_race(accumulator, spec, theta, per_trial)
    assert accumulator.build_context(params).shape == (2, 40, 3)

    def value_and_grad(design):
        return jax.value_and_grad(lambda th: _flow_race(accumulator, spec, th, design)[0])(theta)

    value, grad = value_and_grad(invariant)
    expected_value, expected_grad = value_and_grad(per_trial)
    assert np.isfinite(float(value))
    assert float(value) == pytest.approx(float(expected_value), rel=1e-12)
    assert np.allclose(np.array(grad), np.array(expected_grad), rtol=1e-10, atol=1e-12)


def test_a_trial_invariant_flow_still_simulates_one_draw_per_trial():
    from eamax.design import TrialDesign, build_params_fn, rdm_intercept_slope_spec
    from eamax.simulate import simulate_race

    spec = rdm_intercept_slope_spec()
    accumulator = FlowAccumulator(_conditioner(), CONTEXT)
    design = TrialDesign(rt=jnp.zeros(25), target=2)
    rt, response = simulate_race(
        jax.random.key(0), build_params_fn(spec, accumulator),
        jnp.log(jnp.array([1.0, 1.5, 1.2, 1.3, 0.2])), design, accumulator,
    )
    assert rt.shape == response.shape == (25,)


@pytest.mark.parametrize("affine", [False, True])
def test_draws_are_independent_and_follow_the_flow_density(affine):
    """Probability integral transform: the flow's own survival at its draws is uniform.

    Every row shares one context, which is exactly the case a scalar base distribution got
    wrong: one base draw pushed through every row made all the draws the same number.
    """
    from scipy import stats

    conditioner = make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_mid=16, num_bins=4, affine=affine)
    accumulator = FlowAccumulator(conditioner, CONTEXT)
    params = _params(4000)
    draws = accumulator.sample(jax.random.key(0), params)

    assert draws.shape == (4000,)
    assert np.unique(np.asarray(draws)).size > 0.99 * draws.size  # float32 allows the odd tie
    _, log_sf = accumulator.log_pdf_sf(draws, params)
    assert stats.kstest(np.exp(np.asarray(log_sf, dtype=float)), "uniform").pvalue > 1e-3


def test_the_batched_base_leaves_the_density_unchanged():
    """The base's batch shape only changes sampling; scoring broadcasts exactly as before."""
    import distrax

    conditioner = _conditioner()
    contexts = jnp.array([[[2.0, 1.0, 1.0]], [[1.0, 1.5, 0.8]]])  # (N, 1, num_in)
    for context, data in [
        (contexts[0, 0], jnp.linspace(0.05, 3.0, 7)),  # one context for all data
        (contexts, jnp.linspace(0.05, 3.0, 7)),  # (N, 1, num_in) against (T,)
        (contexts[:, 0], jnp.array([0.3, 0.9])),  # one context per observation
    ]:
        log_prob, flow = spline_flow(data, context, conditioner)
        scalar_base = distrax.Transformed(distrax.Normal(0.0, 1.0), flow.bijector)
        np.testing.assert_allclose(log_prob, scalar_base.log_prob(data), rtol=1e-6)


# --------------------------------------------------------------------------------------- #
# Affine stage, spline settings, log inputs, depth
# --------------------------------------------------------------------------------------- #

NUM_BINS = 4
BATCH_CONTEXT = jnp.array([[[2.0, 1.0, 1.0]], [[1.0, 1.5, 0.8]]])  # (N, 1, num_in)
TIMES = jnp.tile(jnp.linspace(0.05, 3.0, 40), (2, 1))  # (N, T)


def _pair():
    """A plain conditioner and an affine one sharing its spline weights, affine rows zeroed."""
    plain = make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_mid=16, num_bins=NUM_BINS)
    affine = make_mlp_conditioner(nnx.Rngs(1), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True)
    width = 3 * NUM_BINS + 1
    affine.linear1.kernel[...] = plain.linear1.kernel[...]
    affine.linear1.bias[...] = plain.linear1.bias[...]
    affine.linear2.kernel[...] = jnp.zeros_like(affine.linear2.kernel[...]).at[:, :width].set(plain.linear2.kernel[...])
    affine.linear2.bias[...] = jnp.zeros(width + 2).at[:width].set(plain.linear2.bias[...])
    return plain, affine


def _set_affine_outputs(conditioner, loc, raw_scale):
    conditioner.linear2.bias[...] = conditioner.linear2.bias[...].at[-2].set(loc).at[-1].set(raw_scale)


def _assert_survival_matches_integrated_density(conditioner, context, upper, num, abs_tol):
    grid = jnp.linspace(1e-6, upper, num)
    log_pdf, _ = spline_flow(grid, context, conditioner)
    cumulative = jnp.cumsum(jnp.exp(log_pdf)) * (grid[1] - grid[0])
    assert float(cumulative[-1]) == pytest.approx(1.0, abs=abs_tol)
    for target in (0.1, 0.4, 1.0):
        index = int(jnp.searchsorted(grid, target))
        reported = float(jnp.exp(evaluate_pdf_sf(conditioner, grid[index : index + 1], context)[1][0]))
        assert reported == pytest.approx(1.0 - float(cumulative[index]), abs=abs_tol)


def test_layout_is_read_off_the_output_width():
    plain, affine = _pair()
    assert conditioner_layout(plain) == (NUM_BINS, False)
    assert conditioner_layout(affine) == (NUM_BINS, True)
    with pytest.raises(ValueError, match="neither 3K\\+1"):
        flow_model._layout_from_width(3 * NUM_BINS + 2)


def test_zeroed_affine_outputs_reproduce_the_plain_flow():
    # loc = 0 and scale = 1 must be the identity, so an affine conditioner starts training
    # from the same family the plain one does. The scale offset is float32-rounded, so the
    # identity holds to ~1e-7 rather than bit for bit.
    plain, affine = _pair()
    for a, b in zip(
        evaluate_pdf_sf(plain, TIMES, BATCH_CONTEXT), evaluate_pdf_sf(affine, TIMES, BATCH_CONTEXT), strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_the_affine_stage_is_a_location_and_scale_on_log_time():
    plain, affine = _pair()
    _set_affine_outputs(affine, loc=-1.3, raw_scale=-0.7)
    loc, raw_scale = (float(x) for x in affine.linear2.bias[...][-2:])  # as stored, in float32
    scale = float(jax.nn.softplus(raw_scale + flow_model._SCALE_OFFSET) + MIN_SCALE)

    z = jnp.tile(jnp.linspace(-6.0, 6.0, 9), (2, 1))  # beyond the spline range on both sides
    _, plain_flow = spline_flow(TIMES, BATCH_CONTEXT, plain)
    _, affine_flow = spline_flow(TIMES, BATCH_CONTEXT, affine)
    np.testing.assert_allclose(
        np.log(np.asarray(affine_flow.bijector.forward(z))),
        loc + scale * np.log(np.asarray(plain_flow.bijector.forward(z))),
        atol=1e-12,
    )


def test_survival_is_exact_given_the_affine_flow():
    # The race multiplies flow densities by flow survivals; that is only sound while every
    # stage is strictly increasing.
    _, affine = _pair()
    _set_affine_outputs(affine, loc=-1.0, raw_scale=-0.5)
    _assert_survival_matches_integrated_density(affine, jnp.array([2.0, 1.0, 1.0]), 20.0, 400_000, 2e-3)


def test_loss_gradient_reaches_the_affine_outputs():
    _, affine = _pair()
    context = jnp.tile(jnp.array([2.0, 1.0, 1.0]), (32, 1))
    data = jnp.linspace(0.2, 2.0, 32).at[-1].set(-1.0)  # one censored trial

    grads = jax.grad(lambda c: loss_fn(c, data, context, 4.0))(affine)
    kernel = np.asarray(grads.linear2.kernel[...])
    assert np.all(np.isfinite(kernel))
    assert np.any(kernel[:, -2:] != 0.0)


def test_an_affine_flow_drops_straight_into_the_race():
    _, affine = _pair()
    _set_affine_outputs(affine, loc=-0.5, raw_scale=0.2)
    accumulator = FlowAccumulator(affine, CONTEXT, dtype=jnp.float64)

    def total(scale):
        params = {k: jnp.stack([v, v]) * scale for k, v in _params(20).items()}
        return jnp.sum(race_loglik(
            jnp.linspace(0.5, 2.0, 20), jnp.zeros((20,), dtype=int), 0.2,
            lambda t: accumulator.log_pdf_sf(t, params),
        ))

    value, grad = jax.value_and_grad(total)(1.0)
    assert np.isfinite(float(value)) and np.isfinite(float(grad))


def test_an_affine_checkpoint_round_trips_and_records_its_layout(tmp_path):
    _, affine = _pair()
    _set_affine_outputs(affine, loc=-0.8, raw_scale=0.3)
    path = tmp_path / "conditioner"
    save_conditioner(affine, str(path), context_names=CONTEXT, num_mid=16, num_bins=NUM_BINS)
    assert read_metadata(str(path))["affine"] is True

    template = make_mlp_conditioner(nnx.Rngs(9), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True)
    restored = load_conditioner(template, str(path), context_names=CONTEXT)
    np.testing.assert_allclose(
        np.asarray(evaluate_pdf_sf(affine, TIMES, BATCH_CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(restored, TIMES, BATCH_CONTEXT)[0]),
        rtol=1e-6,
    )


def test_loading_with_the_wrong_layout_is_refused(tmp_path):
    plain, affine = _pair()
    for saved, wrong in ((affine, False), (plain, True)):
        path = tmp_path / f"conditioner_{wrong}"
        save_conditioner(saved, str(path), context_names=CONTEXT)
        template = make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=wrong)
        with pytest.raises(ValueError, match="affine="):
            load_conditioner(template, str(path), context_names=CONTEXT)


def test_a_sidecar_without_architecture_keys_loads_as_plain(tmp_path):
    # Checkpoints written before the sidecar recorded the architecture.
    plain, _ = _pair()
    path = tmp_path / "conditioner"
    save_conditioner(plain, str(path), context_names=CONTEXT)
    write_metadata(str(path), CONTEXT)
    assert "affine" not in read_metadata(str(path))

    template = make_mlp_conditioner(nnx.Rngs(5), num_in=3, num_mid=16, num_bins=NUM_BINS)
    restored = load_conditioner(template, str(path), context_names=CONTEXT)
    assert context_bounds(restored) is None
    np.testing.assert_allclose(
        np.asarray(evaluate_pdf_sf(plain, TIMES, BATCH_CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(restored, TIMES, BATCH_CONTEXT)[0]),
        rtol=1e-6,
    )


@pytest.mark.parametrize("layout", ["plain", "affine"])
def test_knots_lie_on_the_flow_transform(layout):
    # A knot maps to a knot, so pushing the z knots through the full flow bijector must land
    # exactly on the reported log-time knots.
    plain, affine = _pair()
    conditioner = plain if layout == "plain" else affine
    _set_affine_outputs(affine, loc=-1.2, raw_scale=-0.4)
    context = jnp.array([2.0, 1.0, 1.0])

    z_knots, log_t_knots = spline_knots(conditioner, context)
    _, flow = spline_flow(jnp.ones(()), context, conditioner)
    assert z_knots.shape == log_t_knots.shape == (NUM_BINS + 1,)
    np.testing.assert_allclose(
        np.log(np.asarray(flow.bijector.forward(z_knots))), np.asarray(log_t_knots), atol=1e-10,
    )


def _narrow_pair():
    """A default affine conditioner and a narrow/unconstrained one sharing all weights."""
    _, default = _pair()
    narrow = make_mlp_conditioner(
        nnx.Rngs(1), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True,
        spline_range=3.5, boundary_slopes="unconstrained",
    )
    nnx.update(narrow, nnx.state(default))
    return default, narrow


def test_spline_settings_are_refused_for_a_plain_flow():
    # Without the affine stage the range is absolute log time; narrowing it forbids fast
    # decision times.
    with pytest.raises(ValueError, match="only be changed for an affine flow"):
        make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_bins=NUM_BINS, spline_range=3.5)
    with pytest.raises(ValueError, match="boundary_slopes must be one of"):
        make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_bins=NUM_BINS, affine=True, boundary_slopes="nope")


def test_the_spline_settings_change_the_density():
    default, narrow = _narrow_pair()
    _set_affine_outputs(default, loc=-1.0, raw_scale=-0.3)
    _set_affine_outputs(narrow, loc=-1.0, raw_scale=-0.3)
    assert spline_settings(default) == (5.0, "identity")
    assert spline_settings(narrow) == (3.5, "unconstrained")
    assert not np.allclose(
        np.asarray(evaluate_pdf_sf(default, TIMES, BATCH_CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(narrow, TIMES, BATCH_CONTEXT)[0]),
    )


def test_survival_is_exact_with_unconstrained_boundary_slopes():
    # Beyond the range the spline extrapolates with learned slopes; it must still be
    # strictly increasing.
    _, narrow = _narrow_pair()
    _set_affine_outputs(narrow, loc=-1.0, raw_scale=-0.5)
    # Push the boundary slopes away from 1 so the extrapolation actually differs.
    width = 3 * NUM_BINS + 1
    narrow.linear2.bias[...] = narrow.linear2.bias[...].at[2 * NUM_BINS].set(1.5).at[width - 1].set(-1.5)
    context = jnp.array([2.0, 1.0, 1.0])
    _assert_survival_matches_integrated_density(narrow, context, 30.0, 600_000, 3e-3)

    z_knots, log_t_knots = spline_knots(narrow, context)
    assert float(z_knots[0]) == pytest.approx(-3.5) and float(z_knots[-1]) == pytest.approx(3.5)


def test_spline_settings_round_trip_and_mismatches_are_refused(tmp_path):
    _, narrow = _narrow_pair()
    path = tmp_path / "conditioner"
    save_conditioner(narrow, str(path), context_names=CONTEXT)
    metadata = read_metadata(str(path))
    assert (metadata["spline_range"], metadata["boundary_slopes"]) == (3.5, "unconstrained")

    template = make_mlp_conditioner(
        nnx.Rngs(4), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True,
        spline_range=3.5, boundary_slopes="unconstrained",
    )
    restored = load_conditioner(template, str(path), context_names=CONTEXT)
    assert spline_settings(restored) == (3.5, "unconstrained")
    np.testing.assert_allclose(
        np.asarray(evaluate_pdf_sf(narrow, TIMES, BATCH_CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(restored, TIMES, BATCH_CONTEXT)[0]),
        rtol=1e-6,
    )

    wrong = make_mlp_conditioner(nnx.Rngs(4), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True)
    with pytest.raises(ValueError, match="spline_range, boundary_slopes"):
        load_conditioner(wrong, str(path), context_names=CONTEXT)


PULSED_NAMES = ("v", "amp", "tau", "s", "b")
PULSED_BOX = {"v": (0.0, 8.0), "amp": (0.0, 1.0), "tau": (0.01, 0.5), "s": (0.25, 3.0), "b": (0.25, 3.0)}
PULSED_EPS = {"v": 0.05, "amp": 0.01, "tau": 0.005, "s": 0.05, "b": 0.05}


def _scaled_pair():
    """An affine conditioner and the same weights with log-scaled inputs."""
    raw = make_mlp_conditioner(nnx.Rngs(3), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=True)
    scaling = log_input_scaling(PULSED_NAMES, PULSED_EPS, PULSED_BOX)
    scaled = make_mlp_conditioner(
        nnx.Rngs(7), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=True, input_scaling=scaling,
    )
    nnx.update(scaled, nnx.state(raw))
    return raw, scaled, scaling


def test_log_uniform_moments_match_numerical_integration():
    # Brute-force quadrature, so a sign slip in the antiderivatives cannot hide.
    for low, high, eps in [(0.0, 0.5, 0.005), (0.0, 8.0, 0.05), (0.25, 3.0, 0.0)]:
        x = np.linspace(low, high, 2_000_001)
        y = np.log(x + eps)
        mean, sd = flow_model._log_uniform_moments(low, high, eps)
        assert mean == pytest.approx(np.trapezoid(y, x) / (high - low), abs=1e-5)
        assert sd == pytest.approx(np.sqrt(np.trapezoid((y - mean) ** 2, x) / (high - low)), abs=1e-5)


def test_log_input_scaling_is_ordered_by_context_and_validated():
    scaling = log_input_scaling(PULSED_NAMES, PULSED_EPS, PULSED_BOX)
    assert scaling["eps"] == tuple(PULSED_EPS[n] for n in PULSED_NAMES)
    assert scaling["loc"][2] == pytest.approx(flow_model._log_uniform_moments(0.01, 0.5, 0.005)[0])
    with pytest.raises(ValueError, match="must name exactly"):
        log_input_scaling(PULSED_NAMES, {**PULSED_EPS, "t0": 0.1}, PULSED_BOX)
    with pytest.raises(ValueError, match="0 < low \\+ eps"):
        log_input_scaling(PULSED_NAMES, {**PULSED_EPS, "v": 0.0}, PULSED_BOX)


def test_log_inputs_are_refused_for_a_plain_flow():
    with pytest.raises(ValueError, match="only implemented for an affine flow"):
        make_mlp_conditioner(
            nnx.Rngs(0), num_in=5, num_bins=NUM_BINS,
            input_scaling=log_input_scaling(PULSED_NAMES, PULSED_EPS, PULSED_BOX),
        )


def test_scaling_off_leaves_the_conditioner_input_unchanged():
    raw, _, _ = _scaled_pair()
    assert input_scaling(raw) is None
    context = jnp.array([[4.0, 0.3, 0.1, 0.8, 0.9]])
    np.testing.assert_array_equal(
        np.asarray(flow_model._conditioner_outputs(raw, context)), np.asarray(raw(context))
    )


def test_scaled_flow_is_the_raw_flow_on_transformed_inputs():
    raw, scaled, scaling = _scaled_pair()
    context = jnp.array([[[4.0, 0.3, 0.1, 0.8, 0.9]], [[1.2, 0.0, 0.05, 1.0, 1.4]]])
    times = jnp.tile(jnp.linspace(0.03, 1.5, 30), (2, 1))
    transformed = (
        jnp.log(context + jnp.asarray(scaling["eps"])) - jnp.asarray(scaling["loc"])
    ) / jnp.asarray(scaling["scale"])
    for a, b in zip(
        evaluate_pdf_sf(scaled, times, context), evaluate_pdf_sf(raw, times, transformed), strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


def test_knots_lie_on_the_scaled_flow_transform():
    _, scaled, _ = _scaled_pair()
    _set_affine_outputs(scaled, loc=-1.2, raw_scale=-0.4)
    context = jnp.array([4.0, 0.3, 0.1, 0.8, 0.9])
    z_knots, log_t_knots = spline_knots(scaled, context)
    _, flow = spline_flow(jnp.ones(()), context, scaled)
    np.testing.assert_allclose(
        np.log(np.asarray(flow.bijector.forward(z_knots))), np.asarray(log_t_knots), atol=1e-10,
    )


def test_scaling_round_trips_and_mismatches_are_refused(tmp_path):
    raw, scaled, scaling = _scaled_pair()
    scaled_path, raw_path = tmp_path / "scaled", tmp_path / "raw"
    save_conditioner(scaled, str(scaled_path), context_names=PULSED_NAMES)
    save_conditioner(raw, str(raw_path), context_names=PULSED_NAMES)
    assert read_metadata(str(scaled_path))["input_scaling"]["loc"] == list(scaling["loc"])
    assert read_metadata(str(raw_path))["input_scaling"] is None

    def template(with_scaling):
        return make_mlp_conditioner(
            nnx.Rngs(11), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=True,
            input_scaling=with_scaling,
        )

    restored = load_conditioner(template(scaling), str(scaled_path), context_names=PULSED_NAMES)
    assert input_scaling(restored) == scaling

    with pytest.raises(ValueError, match="input_scaling"):
        load_conditioner(template(None), str(scaled_path), context_names=PULSED_NAMES)
    with pytest.raises(ValueError, match="input_scaling"):
        load_conditioner(template(scaling), str(raw_path), context_names=PULSED_NAMES)
    shifted = {**scaling, "loc": tuple(x + 0.1 for x in scaling["loc"])}
    with pytest.raises(ValueError, match="input_scaling"):
        load_conditioner(template(shifted), str(scaled_path), context_names=PULSED_NAMES)


def test_one_hidden_layer_is_mlp():
    # The default must not change any existing checkpoint's structure.
    for affine in (False, True):
        conditioner = make_mlp_conditioner(nnx.Rngs(0), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=affine)
        assert type(conditioner) is MLP
        assert num_hidden_layers(conditioner) == 1
    with pytest.raises(ValueError, match="at least 1"):
        make_mlp_conditioner(nnx.Rngs(0), num_in=5, num_bins=NUM_BINS, affine=True, num_hidden=0)


@pytest.mark.parametrize("affine", [False, True])
def test_deep_conditioner_is_the_composed_layers(affine):
    conditioner = make_mlp_conditioner(
        nnx.Rngs(0), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=affine, num_hidden=3,
    )
    assert isinstance(conditioner, DeepMLP) and num_hidden_layers(conditioner) == 3
    assert conditioner_layout(conditioner) == (NUM_BINS, affine)

    x = jnp.array([[4.0, 0.3, 0.1, 0.8, 0.9]])
    h = nnx.gelu(conditioner.linear1(x))
    h = nnx.gelu(conditioner.linear_mid1(h))
    h = nnx.gelu(conditioner.linear_mid2(h))
    np.testing.assert_allclose(np.asarray(conditioner(x)), np.asarray(conditioner.linear2(h)), rtol=1e-12)


def test_deep_log_scaled_affine_flow_trains_and_logs_its_gradient_norm():
    # The full stack must give a finite loss, reach every layer with its gradient, and
    # survive a jitted train step that reports the raw gradient norm.
    conditioner = make_mlp_conditioner(
        nnx.Rngs(5), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=True, num_hidden=2,
        input_scaling=log_input_scaling(PULSED_NAMES, PULSED_EPS, PULSED_BOX),
    )
    context = jnp.tile(jnp.array([[[4.0, 0.3, 0.1, 0.8, 0.9]]]), (4, 1, 1))
    data = jnp.tile(jnp.linspace(0.05, 1.0, 16), (4, 1)).at[0, -1].set(jnp.inf)

    grads = jax.grad(lambda c: loss_fn(c, data, context, 4.0))(conditioner)
    for layer in ("linear1", "linear_mid1", "linear2"):
        kernel = np.asarray(getattr(grads, layer).kernel[...])
        assert np.all(np.isfinite(kernel)) and np.any(kernel != 0.0)

    optimizer = nnx.Optimizer(conditioner, optax.adam(1e-3), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
        grad_norm_max=Maximum("grad_norm"),
    )
    train_step(conditioner, optimizer, metrics, data, context, 4.0)
    train_step(conditioner, optimizer, metrics, data, context, 4.0)
    computed = metrics.compute()
    assert np.isfinite(float(computed["loss"]))
    assert 0.0 < float(computed["grad_norm"]) <= float(computed["grad_norm_max"]) < np.inf
    metrics.reset()
    assert float(metrics.compute()["grad_norm_max"]) == -np.inf


def test_a_loss_only_metric_ignores_the_gradient_norm():
    # Callers that only average the loss keep working.
    conditioner = _conditioner(num_mid=16, num_bins=NUM_BINS)
    optimizer = nnx.Optimizer(conditioner, optax.adam(1e-3), wrt=nnx.Param)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    train_step(conditioner, optimizer, metrics, jnp.linspace(0.2, 2.0, 8), jnp.ones((8, 3)), None)
    assert np.isfinite(float(metrics.compute()["loss"]))


def test_depth_round_trips_and_mismatches_are_refused(tmp_path):
    deep = make_mlp_conditioner(nnx.Rngs(1), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True, num_hidden=2)
    shallow = make_mlp_conditioner(nnx.Rngs(1), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True)
    deep_path, shallow_path = tmp_path / "deep", tmp_path / "shallow"
    save_conditioner(deep, str(deep_path), context_names=CONTEXT)
    save_conditioner(shallow, str(shallow_path), context_names=CONTEXT)
    assert read_metadata(str(deep_path))["num_hidden"] == 2
    assert read_metadata(str(shallow_path))["num_hidden"] == 1

    template = make_mlp_conditioner(nnx.Rngs(9), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True, num_hidden=2)
    restored = load_conditioner(template, str(deep_path), context_names=CONTEXT)
    np.testing.assert_allclose(
        np.asarray(evaluate_pdf_sf(deep, TIMES, BATCH_CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(restored, TIMES, BATCH_CONTEXT)[0]),
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="num_hidden"):
        load_conditioner(shallow, str(deep_path), context_names=CONTEXT)
    with pytest.raises(ValueError, match="num_hidden"):
        load_conditioner(template, str(shallow_path), context_names=CONTEXT)


# --------------------------------------------------------------------------------------- #
# Training box
# --------------------------------------------------------------------------------------- #

WALD_BOX = {"v": (0.0, 8.0), "s": (0.25, 3.0), "b": (0.25, 3.0)}


def _bounded(seed=0, affine=False):
    return make_mlp_conditioner(
        nnx.Rngs(seed), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=affine,
        context_bounds=[WALD_BOX[name] for name in CONTEXT],
    )


def _accumulator_total(accumulator, values, t=jnp.linspace(0.1, 2.0, 16)):
    params = {name: jnp.full(t.shape, values[i]) for i, name in enumerate(CONTEXT)}
    log_pdf, log_sf = accumulator.log_pdf_sf(t, params)
    return jnp.sum(log_pdf) + jnp.sum(log_sf)


def test_the_clamp_is_inert_inside_the_box():
    conditioner = _bounded()
    clamped = FlowAccumulator(conditioner, CONTEXT, dtype=jnp.float64)
    free = FlowAccumulator(conditioner, CONTEXT, dtype=jnp.float64, context_bounds={})
    assert clamped.context_bounds == WALD_BOX
    inside = jnp.array([2.0, 1.0, 1.2])
    assert float(_accumulator_total(clamped, inside)) == float(_accumulator_total(free, inside))
    np.testing.assert_array_equal(
        np.asarray(jax.grad(lambda x: _accumulator_total(clamped, x))(inside)),
        np.asarray(jax.grad(lambda x: _accumulator_total(free, x))(inside)),
    )


def test_outside_the_box_the_flow_is_evaluated_at_the_edge_with_no_gradient():
    accumulator = FlowAccumulator(_bounded(affine=True), CONTEXT, dtype=jnp.float64)
    outside = jnp.array([11.0, 1.0, 0.1])  # v above, b below
    edge = jnp.array([8.0, 1.0, 0.25])
    assert float(_accumulator_total(accumulator, outside)) == float(_accumulator_total(accumulator, edge))
    grad = np.asarray(jax.grad(lambda x: _accumulator_total(accumulator, x))(outside))
    assert grad[0] == 0.0 and grad[2] == 0.0 and grad[1] != 0.0


def test_the_clamp_follows_the_transform():
    # |amp| is clipped, not amp: a negative amplitude inside the box in magnitude is untouched.
    conditioner = make_mlp_conditioner(
        nnx.Rngs(0), num_in=5, num_mid=16, num_bins=NUM_BINS,
        context_bounds=[PULSED_BOX[name] for name in PULSED_NAMES],
    )
    accumulator = FlowAccumulator(conditioner, PULSED_NAMES, transform={"amp": jnp.abs})
    one = {"v": 1.0, "tau": 0.1, "s": 1.0, "b": 1.0}
    context = accumulator.build_context({**{k: jnp.array([v]) for k, v in one.items()}, "amp": jnp.array([-0.3])})
    assert float(np.asarray(context)[0, 1]) == pytest.approx(0.3)
    context = accumulator.build_context({**{k: jnp.array([v]) for k, v in one.items()}, "amp": jnp.array([-1.7])})
    assert float(np.asarray(context)[0, 1]) == pytest.approx(1.0)


def test_an_explicit_box_clamps_an_unbounded_conditioner():
    accumulator = FlowAccumulator(_conditioner(), CONTEXT, context_bounds={"v": (0.0, 8.0)})
    context = accumulator.build_context({"v": jnp.array([9.0]), "s": jnp.array([9.0]), "b": jnp.array([1.0])})
    np.testing.assert_allclose(np.asarray(context), [[8.0, 9.0, 1.0]])
    assert FlowAccumulator(_conditioner(), CONTEXT).context_bounds == {}


@pytest.mark.parametrize(
    "bounds, match",
    [({"t0": (0.0, 1.0)}, "not a flow input"), ({"v": (2.0, 1.0)}, "is empty"), ({"v": (0.0, 9.0)}, "disagree")],
)
def test_bad_context_bounds_raise(bounds, match):
    with pytest.raises(ValueError, match=match):
        FlowAccumulator(_bounded(), CONTEXT, context_bounds=bounds)


def test_bad_recorded_bounds_raise():
    with pytest.raises(ValueError, match="needs 3"):
        make_mlp_conditioner(nnx.Rngs(0), num_in=3, context_bounds=[(0.0, 1.0)])
    with pytest.raises(ValueError, match="is empty"):
        make_mlp_conditioner(nnx.Rngs(0), num_in=3, context_bounds=[(0.0, 1.0), (1.0, 1.0), (0.0, 1.0)])


def test_the_box_round_trips_and_an_unbounded_template_adopts_it(tmp_path):
    conditioner = _bounded()
    path = tmp_path / "conditioner"
    save_conditioner(conditioner, str(path), context_names=CONTEXT)
    assert read_metadata(str(path))["context_bounds"] == [list(WALD_BOX[n]) for n in CONTEXT]

    restored = load_conditioner(_bounded(seed=3), str(path), context_names=CONTEXT)
    assert context_bounds(restored) == tuple(WALD_BOX[n] for n in CONTEXT)
    adopted = load_conditioner(_conditioner(num_mid=16, num_bins=NUM_BINS), str(path), context_names=CONTEXT)
    assert context_bounds(adopted) == tuple(WALD_BOX[n] for n in CONTEXT)
    assert FlowAccumulator(adopted, CONTEXT).context_bounds == WALD_BOX


def test_a_template_with_a_different_box_is_refused(tmp_path):
    path = tmp_path / "conditioner"
    save_conditioner(_bounded(), str(path), context_names=CONTEXT)
    other = make_mlp_conditioner(
        nnx.Rngs(0), num_in=3, num_mid=16, num_bins=NUM_BINS,
        context_bounds=[(0.0, 14.0), (0.25, 3.0), (0.25, 3.0)],
    )
    with pytest.raises(ValueError, match="trained on the box"):
        load_conditioner(other, str(path), context_names=CONTEXT)
