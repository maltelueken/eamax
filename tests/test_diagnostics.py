"""Diagnostics, and the BlackJAX API surface they and the samplers depend on.

Two of these tests are canaries rather than tests of `eamax`: they fail the day a BlackJAX
release moves something, which is preferable to a fit failing on a cluster six weeks later.
"""

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax

from eamax.inference._blackjax import (
    RHAT_MIN_VERSION,
    extend_params,
    get_filter_adapt_info_fn,
    smc_resampling,
    tempering_param,
)
from eamax.inference.diagnostics import count_unique_particles, rhat, weight_ess


# --------------------------------------------------------------------------- #
# R-hat
# --------------------------------------------------------------------------- #
def test_converged_chains_report_one():
    draws = jax.random.normal(jax.random.key(0), (2000, 4, 3))

    assert float(jnp.max(rhat(draws))) < 1.01


def test_chains_stuck_in_different_places_report_more_than_one():
    draws = jax.random.normal(jax.random.key(0), (2000, 4, 3))
    separated = draws + jnp.arange(4)[None, :, None] * 3.0

    assert float(jnp.max(rhat(separated))) > 1.1


def test_it_matches_arviz_to_the_precision_a_threshold_cares_about():
    """So a caller switching off `arviz_stats` keeps the numbers it had.

    A convergence filter such as ``rhat < 1.01`` needs the same statistic, not merely a
    similar one.
    """
    arviz_stats = pytest.importorskip("arviz_stats")
    xarray = pytest.importorskip("xarray")
    if not hasattr(arviz_stats, "rhat"):
        pytest.skip("this arviz_stats exposes no top-level rhat")

    chains = np.asarray(jax.random.normal(jax.random.key(0), (4, 2000)))
    chains = chains + np.arange(4)[:, None] * 2.0  # deliberately not converged

    ours = float(jnp.ravel(rhat(chains.T[..., None]))[0])
    theirs = float(
        arviz_stats.rhat(
            xarray.Dataset({"x": xarray.DataArray(chains, dims=["chain", "draw"])}),
            var_names=["x"],
        )["x"]
    )

    assert ours == pytest.approx(theirs, abs=1e-5)


def test_the_default_axes_match_the_inference_loops_output_layout():
    # (draw, chain, param), not BlackJAX's own (chain, draw).
    draws = jax.random.normal(jax.random.key(0), (500, 4, 3))

    assert rhat(draws).shape == (3,)


def test_singleton_axes_survive():
    """BlackJAX squeezes its result; a squeezed axis is a silently misread one.

    With one dataset or one parameter, the raw BlackJAX shape is the same either way, so
    a caller reducing with ``all(axis=-1)`` cannot tell whether it is reducing over
    parameters or over datasets.
    """
    key = jax.random.key(0)

    # (chain, draw, dataset, param), the layout `eamax.io` filters on.
    assert rhat(
        jax.random.normal(key, (4, 500, 1, 3)), chain_axis=0, sample_axis=1
    ).shape == (1, 3)
    assert rhat(
        jax.random.normal(key, (4, 500, 5, 1)), chain_axis=0, sample_axis=1
    ).shape == (5, 1)
    assert rhat(
        jax.random.normal(key, (4, 500, 1, 1)), chain_axis=0, sample_axis=1
    ).shape == (1, 1)


def test_a_scalar_target_still_reduces_to_a_scalar():
    draws = jax.random.normal(jax.random.key(0), (500, 4))

    assert rhat(draws).shape == ()


# --------------------------------------------------------------------------- #
# Particle degeneracy
# --------------------------------------------------------------------------- #
def test_a_fully_distinct_cloud_counts_every_particle():
    particles = jax.random.normal(jax.random.key(0), (200, 7))

    assert int(count_unique_particles(particles)) == 200


def test_duplicates_are_not_counted_twice():
    particles = jax.random.normal(jax.random.key(0), (50, 7))
    collapsed = particles.at[10:].set(particles[0])  # rows 0..9 distinct; row 0 repeated

    assert int(count_unique_particles(collapsed)) == 10


def test_a_totally_collapsed_cloud_counts_one():
    particles = jnp.broadcast_to(jax.random.normal(jax.random.key(0), (7,)), (100, 7))

    assert int(count_unique_particles(particles)) == 1


def test_particles_differing_by_one_ulp_are_still_distinct():
    particles = jnp.ones((2, 4))
    particles = particles.at[1, 2].set(jnp.nextafter(1.0, 2.0))

    assert int(count_unique_particles(particles)) == 2


def test_bit_identical_particles_are_never_counted_twice():
    """XLA can project identical rows to values one ULP apart, depending on their position.

    Counting changes in the sorted projections alone therefore overcounts -- under-reporting
    collapse, which is the one thing this diagnostic exists to detect.
    """
    base = jax.random.normal(jax.random.key(0), (40, 5))
    duplicated = jnp.concatenate([base, base[:20]])

    assert int(count_unique_particles(duplicated)) == 40


def test_it_agrees_with_a_brute_force_count():
    base = jax.random.normal(jax.random.key(0), (40, 5))
    particles = jnp.concatenate([base, base[:20]])

    expected = np.unique(np.asarray(particles), axis=0).shape[0]
    assert int(count_unique_particles(particles)) == expected


def test_uniform_weights_hide_a_collapsed_cloud():
    """Which is exactly why the unique count exists alongside the weight ESS."""
    particles = jnp.broadcast_to(jax.random.normal(jax.random.key(0), (5,)), (1000, 5))
    weights = jnp.full(1000, 1.0 / 1000)

    assert float(weight_ess(weights)) == pytest.approx(1.0)
    assert int(count_unique_particles(particles)) == 1


# --------------------------------------------------------------------------- #
# Weight ESS
# --------------------------------------------------------------------------- #
def test_uniform_weights_have_full_effective_sample_size():
    assert float(weight_ess(jnp.full(100, 0.01))) == pytest.approx(1.0)


def test_one_dominant_weight_has_almost_none():
    weights = jnp.zeros(100).at[0].set(1.0)

    assert float(weight_ess(weights)) == pytest.approx(0.01)


def test_it_maps_over_leading_axes():
    weights = jnp.full((3, 2, 50), 1.0 / 50)

    assert weight_ess(weights).shape == (3, 2)


# --------------------------------------------------------------------------- #
# BlackJAX API canaries
# --------------------------------------------------------------------------- #
class TestBlackjaxSurface:
    def test_the_names_eamax_reaches_for_all_resolve(self):
        assert callable(extend_params())
        assert callable(get_filter_adapt_info_fn())
        assert callable(smc_resampling().systematic)

    def test_extend_params_takes_a_single_argument(self):
        import inspect

        assert len(inspect.signature(extend_params()).parameters) == 1

    def test_resampling_is_not_reexported_from_blackjax_smc(self):
        """So the shim must import the submodule explicitly, or break on a fresh install."""
        import blackjax.smc

        assert not hasattr(blackjax.smc, "systematic")

    def test_the_adaptive_tempered_signature_is_the_one_eamax_passes(self):
        import inspect

        from blackjax.smc import adaptive_tempered

        parameters = list(inspect.signature(adaptive_tempered.as_top_level_api).parameters)
        assert parameters[:7] == [
            "logprior_fn",
            "loglikelihood_fn",
            "mcmc_step_fn",
            "mcmc_init_fn",
            "mcmc_parameters",
            "resampling_fn",
            "target_ess",
        ]

    def test_the_tempering_field_is_read_under_whichever_name_this_version_uses(self):
        """It was renamed from `lmbda` to `tempering_param`, and it is the loop's only
        termination condition -- reading it under one name breaks silently on the other."""
        from blackjax.smc.tempered import TemperedSMCState

        state = TemperedSMCState(particles=jnp.zeros((2, 2)), weights=jnp.ones(2), **{
            field: jnp.asarray(0.25)
            for field in TemperedSMCState._fields
            if field not in ("particles", "weights")
        })

        assert float(tempering_param(state)) == pytest.approx(0.25)

    def test_an_unrecognised_state_says_which_field_to_add(self):
        from collections import namedtuple

        Unknown = namedtuple("Unknown", ["particles", "weights", "inverse_temperature"])

        with pytest.raises(AttributeError, match="_TEMPERING_FIELDS"):
            tempering_param(Unknown(None, None, 0.5))

    def test_the_installed_version_carries_rhat(self):
        if not hasattr(blackjax.diagnostics, "rhat"):
            pytest.skip(f"BlackJAX predates {RHAT_MIN_VERSION}")

        assert callable(blackjax.diagnostics.rhat)


# --------------------------------------------------------------------------- #
# Optional dependency
# --------------------------------------------------------------------------- #
def test_eamax_imports_and_transforms_work_without_blackjax():
    """A subprocess, because `blackjax` is already imported in this one.

    The contract is that no `eamax.inference` submodule imports BlackJAX at module scope,
    so everything that does not sample stays usable without it.
    """
    program = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_module(self, name, path=None):
                return self if name == "blackjax" or name.startswith("blackjax.") else None

            def load_module(self, name):
                raise ImportError(f"blocked: {name}")

            def find_spec(self, name, path=None, target=None):
                if name == "blackjax" or name.startswith("blackjax."):
                    raise ImportError(f"blocked: {name}")
                return None

        sys.meta_path.insert(0, Blocker())

        import eamax
        import eamax.inference
        from eamax.inference.transforms import BlockTransform
        from eamax.inference.init import min_valid_rt, rejection_sample
        from eamax.inference import posterior

        import jax.numpy as jnp
        transform = BlockTransform()
        assert jnp.allclose(transform.forward(transform.inverse(jnp.asarray([1.0, 2.0]))),
                            jnp.asarray([1.0, 2.0]))
        assert float(min_valid_rt(jnp.asarray([0.9, -1.0, 0.5]))) == 0.5

        from eamax.inference.warmup import window_adaptation
        try:
            window_adaptation(None, None, jnp.zeros((2, 2)), 1, None)
        except ImportError as exc:
            assert "eamax[inference]" in str(exc), exc
        else:
            raise AssertionError("expected an ImportError naming the extra")

        print("ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
