"""The neural density estimator, and the checkpoint metadata that keeps it honest."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("distrax")
pytest.importorskip("flax")

from flax import nnx  # noqa: E402

from eamax import race_loglik  # noqa: E402
from eamax.accumulators import Wald  # noqa: E402
from eamax.flows import (  # noqa: E402
    FlowAccumulator,
    evaluate_pdf_sf,
    load_conditioner,
    loss_fn,
    make_mlp_conditioner,
    save_conditioner,
    spline_flow,
)

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
