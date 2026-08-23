"""NetCDF round-trips for both posterior schemas.

The interesting tests are on the read-back path: that the back-transform is applied to the
full vector before parameters are selected, that the reader applies no diagnostic and no
thinning of its own, and that the recipe `eamax.io` documents in its place -- diagnose,
pool, thin, at the call site -- reproduces what the reader used to do.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("arviz_base")
pytest.importorskip("h5netcdf")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from eamax.design.spec import ParamSpec  # noqa: E402
from eamax.inference.diagnostics import rhat  # noqa: E402
from eamax.inference.posterior import pool_chains, thin  # noqa: E402
from eamax.hierarchical import HierarchicalLKJMVNPrior  # noqa: E402
from eamax.inference.smc import SMCResult  # noqa: E402
from eamax.inference.transforms import BlockTransform  # noqa: E402
from eamax.io import (  # noqa: E402
    load_dataset_posterior,
    open_dataset_posterior,
    save_dataset_posterior,
    save_hierarchical_posterior,
)

NUM_PARAMS = 3
NUM_SUBJECTS = 4


# --------------------------------------------------------------------------- #
# Dataset posteriors
# --------------------------------------------------------------------------- #
def _converged_samples(num_datasets, num_draws, num_chains, num_params, *, seed=0):
    """Chains that agree: identical distributions, independent noise."""
    key = jax.random.key(seed)
    return 0.1 * jax.random.normal(
        key, (num_datasets, num_draws, num_chains, num_params)
    )


def _split_dataset(samples, index):
    """Push one dataset's chains apart so its R-hat blows up."""
    samples = np.array(samples)
    offsets = np.linspace(-8.0, 8.0, samples.shape[2])
    samples[index] += offsets[None, :, None]
    return samples


def test_dataset_posterior_round_trips_shape_and_names(tmp_path):
    samples = _converged_samples(5, 200, 4, NUM_PARAMS)
    path = tmp_path / "posterior.nc"

    save_dataset_posterior(path, samples, ["v", "b", "t0"])
    stored = open_dataset_posterior(path)

    assert stored["theta"].dims == ("chain", "draw", "dataset", "param")
    assert stored["theta"].shape == (4, 200, 5, NUM_PARAMS)
    assert [str(n) for n in stored.coords["param"].to_numpy()] == ["v", "b", "t0"]


def test_dataset_posterior_stores_the_values_unchanged(tmp_path):
    samples = _converged_samples(3, 20, 2, NUM_PARAMS)
    path = tmp_path / "posterior.nc"

    save_dataset_posterior(path, samples, ["v", "b", "t0"])
    stored = open_dataset_posterior(path)["theta"].to_numpy()

    # (dataset, draw, chain, param) -> (chain, draw, dataset, param)
    assert np.allclose(stored, np.transpose(np.asarray(samples), (2, 1, 0, 3)))


def test_arviz_layout_is_stored_as_given(tmp_path):
    """Amortized draws arrive already in ArviZ order, with a single chain."""
    samples = np.arange(1 * 30 * 4 * NUM_PARAMS).reshape(1, 30, 4, NUM_PARAMS).astype(float)
    path = tmp_path / "npe.nc"

    save_dataset_posterior(path, samples, ["v", "b", "t0"], layout="arviz")

    assert np.allclose(open_dataset_posterior(path)["theta"].to_numpy(), samples)


def test_save_rejects_an_unknown_layout(tmp_path):
    with pytest.raises(ValueError, match="layout"):
        save_dataset_posterior(tmp_path / "x.nc", np.zeros((1, 2, 3, 4)), list("abcd"), layout="wat")


def test_save_rejects_a_name_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="names were given"):
        save_dataset_posterior(tmp_path / "x.nc", np.zeros((1, 2, 3, 4)), ["a", "b"])


def test_load_returns_every_chain_and_every_dataset(tmp_path):
    samples = _converged_samples(5, 400, 4, NUM_PARAMS)
    path = tmp_path / "posterior.nc"
    save_dataset_posterior(path, samples, ["v", "b", "t0"])

    posterior = load_dataset_posterior(
        path, to_constrained=jnp.exp, param_names=["v", "b", "t0"]
    )

    assert posterior.shape == (4, 400, 5, NUM_PARAMS)


def test_load_keeps_an_unconverged_dataset_and_says_nothing(tmp_path):
    """The reader applies no diagnostic, so a broken fit reads back like any other."""
    samples = _split_dataset(_converged_samples(4, 400, 4, NUM_PARAMS), index=2)
    path = tmp_path / "posterior.nc"
    save_dataset_posterior(path, samples, ["v", "b", "t0"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        posterior = load_dataset_posterior(
            path, to_constrained=jnp.exp, param_names=["v", "b", "t0"]
        )

    assert posterior.shape == (4, 400, 4, NUM_PARAMS)
    # The split dataset is still in there, still split.
    assert float(np.std(np.mean(posterior[:, :, 2, :], axis=1))) > 1.0


def test_the_documented_filter_recipe_reproduces_the_old_reader(tmp_path):
    """What `load_dataset_posterior` used to do, now assembled at the call site.

    This is the four-line recipe in `eamax.io`'s docstring. It stays a test because the
    composition -- diagnose, then pool, then thin -- is the part that has an order.
    """
    samples = _split_dataset(_converged_samples(4, 400, 4, NUM_PARAMS), index=2)
    path = tmp_path / "posterior.nc"
    save_dataset_posterior(path, samples, ["v", "b", "t0"])

    theta = load_dataset_posterior(
        path, to_constrained=jnp.exp, param_names=["v", "b", "t0"]
    )
    is_converged = np.all(
        np.asarray(rhat(theta, chain_axis=0, sample_axis=1)) < 1.05, axis=-1
    )
    posterior = thin(pool_chains(theta)[is_converged], 50, axis=1)

    assert is_converged.tolist() == [True, True, False, True]
    assert posterior.shape == (3, 50, NUM_PARAMS)


def test_the_recipe_yields_an_empty_result_when_nothing_converged(tmp_path):
    """Masking a pooled numpy array is what keeps total failure from raising."""
    samples = _converged_samples(3, 400, 4, NUM_PARAMS)
    for index in range(3):
        samples = _split_dataset(samples, index)
    path = tmp_path / "posterior.nc"
    save_dataset_posterior(path, samples, ["v", "b", "t0"])

    theta = load_dataset_posterior(
        path, to_constrained=jnp.exp, param_names=["v", "b", "t0"]
    )
    is_converged = np.all(
        np.asarray(rhat(theta, chain_axis=0, sample_axis=1)) < 1.05, axis=-1
    )
    posterior = thin(pool_chains(theta)[is_converged], 50, axis=1)

    assert not is_converged.any()
    assert posterior.shape == (0, 50, NUM_PARAMS)


def test_load_back_transforms_before_selecting(tmp_path):
    """The hierarchical layout: two bounded hyperparameters ahead of the subject block."""
    stored_names = ["p_1", "p_2", "v", "b", "t0"]
    samples = _converged_samples(2, 200, 4, len(stored_names), seed=3)
    path = tmp_path / "meta.nc"
    save_dataset_posterior(path, samples, stored_names)

    transform = BlockTransform(lower=[0.0, 0.0], upper=[1.0, 1.0])

    posterior = load_dataset_posterior(
        path, to_constrained=transform.forward, param_names=["v", "b", "t0"]
    )

    stored = np.transpose(np.asarray(samples), (2, 1, 0, 3))
    expected = np.asarray(transform.forward(stored))[..., 2:]

    assert np.allclose(posterior, expected)
    # And the wrong order would have squashed the positive block below 1.
    assert np.any(posterior[..., :2] > 1.0)


def test_load_rejects_an_unstored_parameter(tmp_path):
    samples = _converged_samples(2, 100, 4, NUM_PARAMS)
    path = tmp_path / "posterior.nc"
    save_dataset_posterior(path, samples, ["v", "b", "t0"])

    with pytest.raises(ValueError, match="not among the stored"):
        load_dataset_posterior(
            path, to_constrained=jnp.exp, param_names=["drift"]
        )


# --------------------------------------------------------------------------- #
# Hierarchical posteriors
# --------------------------------------------------------------------------- #
@pytest.fixture
def spec():
    return ParamSpec(names=("v", "b", "t0"), links=("log", "log", "log"))


@pytest.fixture
def flat_space(spec):
    prior = HierarchicalLKJMVNPrior(
        NUM_SUBJECTS,
        NUM_PARAMS,
        inverse_gamma_scale=jnp.full(NUM_PARAMS, 0.4),
        mu_loc=jnp.zeros(NUM_PARAMS),
        mu_scale=jnp.full(NUM_PARAMS, 0.25),
        num_centered=1,
    )
    return prior.flat_space()


def _result(flat_space, num_chains=2, num_particles=16, seed=0):
    keys = jax.random.split(jax.random.key(seed), num_chains)
    particles = jnp.stack(
        [flat_space.sample_particles(key, num_particles) for key in keys]
    )
    return SMCResult(
        particles=particles,
        weights=jnp.full((num_chains, num_particles), 1.0 / num_particles),
        log_marginal_likelihood=jnp.asarray([-100.0, -102.0][:num_chains]),
        num_iterations=jnp.asarray([12, 14][:num_chains]),
        num_unique=jnp.asarray([16, 15][:num_chains]),
        num_unique_smc=jnp.asarray([11, 12][:num_chains]),
        weight_ess=jnp.asarray([0.8, 0.75][:num_chains]),
    )


def test_hierarchical_posterior_has_the_expected_groups_and_dims(tmp_path, flat_space, spec):
    import xarray as xr

    subjects = [f"s{i}" for i in range(NUM_SUBJECTS)]
    path = tmp_path / "fit.nc"

    save_hierarchical_posterior(path, _result(flat_space), flat_space, spec, subjects)
    tree = xr.open_datatree(path)

    posterior = tree["posterior"].to_dataset()
    assert posterior["mu"].dims == ("chain", "draw", "param")
    assert posterior["sigma"].dims == ("chain", "draw", "param")
    for name in spec.names:
        assert posterior[name].dims == ("chain", "draw", "subject")
    assert [str(s) for s in posterior.coords["subject"].to_numpy()] == subjects


def test_hierarchical_posterior_reports_the_natural_scale(tmp_path, flat_space, spec):
    import xarray as xr

    result = _result(flat_space)
    path = tmp_path / "fit.nc"
    save_hierarchical_posterior(path, result, flat_space, spec, list(range(NUM_SUBJECTS)))

    posterior = xr.open_datatree(path)["posterior"].to_dataset()

    expected = np.exp(
        np.asarray(jax.vmap(jax.vmap(flat_space.subject_params))(result.particles))
    )
    assert np.allclose(posterior["v"].to_numpy(), expected[..., 0])
    assert np.all(posterior["b"].to_numpy() > 0.0)


def test_sigma_is_stored_on_the_log_scale(tmp_path, flat_space, spec):
    import xarray as xr

    result = _result(flat_space)
    path = tmp_path / "fit.nc"
    save_hierarchical_posterior(path, result, flat_space, spec, list(range(NUM_SUBJECTS)))

    stored = xr.open_datatree(path)["posterior"].to_dataset()["sigma"].to_numpy()
    expected = np.asarray(
        jax.vmap(jax.vmap(lambda p: flat_space.population(p)[1]))(result.particles)
    )

    assert np.allclose(stored, expected)


def test_identity_linked_parameters_are_not_exponentiated(tmp_path, flat_space):
    import xarray as xr

    signed = ParamSpec(names=("v", "b", "shift"), links=("log", "log", "identity"))
    result = _result(flat_space)
    path = tmp_path / "signed.nc"
    save_hierarchical_posterior(path, result, flat_space, signed, list(range(NUM_SUBJECTS)))

    posterior = xr.open_datatree(path)["posterior"].to_dataset()
    raw = np.asarray(jax.vmap(jax.vmap(flat_space.subject_params))(result.particles))

    assert np.allclose(posterior["shift"].to_numpy(), raw[..., 2])


def test_smc_summaries_land_in_the_attributes(tmp_path, flat_space, spec):
    import xarray as xr

    result = _result(flat_space)
    path = tmp_path / "fit.nc"
    save_hierarchical_posterior(path, result, flat_space, spec, list(range(NUM_SUBJECTS)))

    attrs = xr.open_datatree(path).attrs

    # NetCDF round-trips list-valued attributes as numpy arrays.
    assert np.allclose(attrs["log_marginal_likelihood"], [-100.0, -102.0])
    assert attrs["log_marginal_likelihood_mean"] == pytest.approx(-101.0)
    assert attrs["log_marginal_likelihood_sd"] == pytest.approx(np.sqrt(2.0))
    assert np.array_equal(attrs["smc_num_iterations"], [12, 14])
    assert attrs["num_subjects"] == NUM_SUBJECTS


def test_extra_attrs_are_merged(tmp_path, flat_space, spec):
    import xarray as xr

    path = tmp_path / "fit.nc"
    save_hierarchical_posterior(
        path,
        _result(flat_space),
        flat_space,
        spec,
        list(range(NUM_SUBJECTS)),
        attrs={"dataset": "flanker", "effects": ["congruency", "block"]},
    )

    attrs = xr.open_datatree(path).attrs
    assert attrs["dataset"] == "flanker"
    assert list(attrs["effects"]) == ["congruency", "block"]
    assert attrs["num_subjects"] == NUM_SUBJECTS


def test_observed_data_is_written_with_subject_and_trial_dims(tmp_path, flat_space, spec):
    import xarray as xr

    num_trials = 7
    observed = {
        "rt": np.linspace(0.3, 1.2, NUM_SUBJECTS * num_trials).reshape(NUM_SUBJECTS, num_trials),
        "response": np.zeros((NUM_SUBJECTS, num_trials)),
        "mask": np.ones((NUM_SUBJECTS, num_trials), dtype=bool),
    }
    path = tmp_path / "fit.nc"
    save_hierarchical_posterior(
        path, _result(flat_space), flat_space, spec, list(range(NUM_SUBJECTS)), observed=observed
    )

    stored = xr.open_datatree(path)["observed_data"].to_dataset()
    assert stored["rt"].dims == ("subject", "trial")
    assert np.allclose(stored["rt"].to_numpy(), observed["rt"])


def test_prior_particles_are_reconstructed_onto_the_posterior_scale(tmp_path, flat_space, spec):
    import xarray as xr

    particles = flat_space.sample_particles(jax.random.key(11), 32)
    path = tmp_path / "fit.nc"
    save_hierarchical_posterior(
        path,
        _result(flat_space),
        flat_space,
        spec,
        list(range(NUM_SUBJECTS)),
        prior_particles=particles,
    )

    prior = xr.open_datatree(path)["prior"].to_dataset()
    assert prior["v"].dims == ("draw", "subject")
    assert prior["mu"].dims == ("draw", "param")
    assert np.all(prior["v"].to_numpy() > 0.0)


def test_a_spec_of_the_wrong_width_is_rejected(tmp_path, flat_space):
    wrong = ParamSpec(names=("v", "b"), links=("log", "log"))

    with pytest.raises(ValueError, match="reconstructs"):
        save_hierarchical_posterior(
            tmp_path / "fit.nc", _result(flat_space), flat_space, wrong, list(range(NUM_SUBJECTS))
        )


def test_a_subject_label_count_mismatch_is_rejected(tmp_path, flat_space, spec):
    with pytest.raises(ValueError, match="subject labels"):
        save_hierarchical_posterior(
            tmp_path / "fit.nc", _result(flat_space), flat_space, spec, ["only-one"]
        )
