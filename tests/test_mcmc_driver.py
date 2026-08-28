"""The NUTS drivers.

These test the plumbing rather than the statistics: that the closures are rebuilt per
dataset under the outer map, that the axes come out in the documented order, and that a
model whose posterior is known is actually recovered.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax

from eamax.inference.init import jitter_positions
from eamax.inference.mcmc import fit_nuts, fit_nuts_batch, inference_loop_multiple_chains

NUM_CHAINS = 4
NUM_PARAMS = 2


def gaussian_logdensity_factory(data):
    """Posterior for the mean of a unit-variance Gaussian under a flat prior.

    Conjugate, so the posterior is `N(mean(data), 1 / n)` and recovery is checkable in
    closed form.
    """
    mean = jnp.mean(data, axis=0)
    count = data.shape[0]

    def logdensity_fn(position):
        return -0.5 * count * jnp.sum((position - mean) ** 2)

    return logdensity_fn


def dispersed_starts(key, data):
    del data
    return jax.random.normal(key, (NUM_CHAINS, NUM_PARAMS)) * 2.0


@pytest.fixture
def data():
    return jax.random.normal(jax.random.key(0), (200, NUM_PARAMS)) + jnp.asarray([1.0, -2.0])


class TestInferenceLoop:
    def _states_and_params(self, logdensity_fn):
        from eamax.inference.warmup import window_adaptation

        return window_adaptation(
            blackjax.nuts, logdensity_fn, jnp.zeros((NUM_CHAINS, NUM_PARAMS)), 200,
            jax.random.key(0),
        )

    def test_positions_come_out_as_draws_by_chains_by_parameters(self, data):
        logdensity_fn = gaussian_logdensity_factory(data)
        states, parameters = self._states_and_params(logdensity_fn)
        step = blackjax.nuts.build_kernel()

        positions, infos = inference_loop_multiple_chains(
            jax.random.key(1),
            lambda key, state, params: step(key, state, logdensity_fn, **params),
            states, 50, NUM_CHAINS, parameters,
        )

        assert positions.shape == (50, NUM_CHAINS, NUM_PARAMS)
        assert infos.is_divergent.shape == (50, NUM_CHAINS)

    def test_a_prebound_kernel_needs_no_parameters(self, data):
        logdensity_fn = gaussian_logdensity_factory(data)
        states, parameters = self._states_and_params(logdensity_fn)
        single = jax.tree.map(lambda leaf: leaf[0], parameters)
        kernel = blackjax.nuts(logdensity_fn, **single).step

        positions, _ = inference_loop_multiple_chains(
            jax.random.key(1), kernel, states, 20, NUM_CHAINS
        )

        assert positions.shape == (20, NUM_CHAINS, NUM_PARAMS)


class TestFitNuts:
    def test_it_recovers_a_conjugate_posterior(self, data):
        positions, infos = fit_nuts(
            jax.random.key(0), data, gaussian_logdensity_factory, dispersed_starts,
            NUM_CHAINS, 400, 600,
        )

        assert positions.shape == (600, NUM_CHAINS, NUM_PARAMS)
        assert jnp.allclose(jnp.mean(positions, axis=(0, 1)), jnp.mean(data, axis=0), atol=0.05)
        assert int(jnp.sum(infos.is_divergent)) == 0

    def test_the_posterior_width_matches_the_analytic_one(self, data):
        positions, _ = fit_nuts(
            jax.random.key(0), data, gaussian_logdensity_factory, dispersed_starts,
            NUM_CHAINS, 400, 2000,
        )
        expected = 1.0 / np.sqrt(data.shape[0])

        assert jnp.allclose(jnp.std(positions, axis=(0, 1)), expected, rtol=0.15)

    def test_there_is_no_shared_warmup_switch(self):
        """Removed, not deprecated: a broadcast warmup makes R-hat unable to fail."""
        with pytest.raises(TypeError, match="shared_warmup"):
            fit_nuts(
                jax.random.key(0), jnp.zeros((10, NUM_PARAMS)), gaussian_logdensity_factory,
                dispersed_starts, NUM_CHAINS, 10, 10, shared_warmup=True,
            )

    def test_starting_values_may_depend_on_the_data(self):
        """Starting values that depend on both the key and the dataset.

        The whole start is a function of both the PRNG key and the per-dataset data, not a
        fixed vector with one entry overwritten.
        """
        seen = {}

        def make_starts(key, data):
            seen["shape"] = data.shape
            centre = jnp.mean(data, axis=0)
            return centre + jax.random.normal(key, (NUM_CHAINS, NUM_PARAMS)) * 0.1

        data = jax.random.normal(jax.random.key(3), (100, NUM_PARAMS)) + 5.0
        positions, _ = fit_nuts(
            jax.random.key(0), data, gaussian_logdensity_factory, make_starts,
            NUM_CHAINS, 200, 200,
        )

        assert seen["shape"] == (100, NUM_PARAMS)
        assert jnp.allclose(jnp.mean(positions, axis=(0, 1)), jnp.mean(data, axis=0), atol=0.1)


class TestFitNutsBatch:
    def test_each_dataset_gets_its_own_posterior(self):
        """The log-density closure is rebuilt per traced slice, not baked in before the vmap."""
        means = jnp.asarray([[-3.0, 0.0], [0.0, 0.0], [3.0, 1.0]])
        key = jax.random.key(0)
        data = jax.random.normal(key, (3, 200, NUM_PARAMS)) + means[:, None, :]

        positions, infos = fit_nuts_batch(
            jax.random.key(1), data, gaussian_logdensity_factory, dispersed_starts,
            NUM_CHAINS, 300, 400,
        )

        assert positions.shape == (3, 400, NUM_CHAINS, NUM_PARAMS)
        assert infos.is_divergent.shape == (3, 400, NUM_CHAINS)

        recovered = jnp.mean(positions, axis=(1, 2))
        assert jnp.allclose(recovered, jnp.mean(data, axis=1), atol=0.05)

    def test_datasets_are_keyed_independently(self):
        data = jnp.broadcast_to(
            jax.random.normal(jax.random.key(0), (200, NUM_PARAMS)), (2, 200, NUM_PARAMS)
        )
        positions, _ = fit_nuts_batch(
            jax.random.key(1), data, gaussian_logdensity_factory, dispersed_starts,
            NUM_CHAINS, 200, 100,
        )

        # Identical data, different keys -> different draws.
        assert not jnp.allclose(positions[0], positions[1])
