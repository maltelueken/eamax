"""Window adaptation, and the diagnostic it either enables or destroys.

The headline here is the first test class: it exercises **both** initialisation strategies
against the same target. That comparison is the whole argument for per-chain adaptation being
the only warm-up `eamax` offers.

`eamax` offers no shared-start warm-up, so the shared-start arm is built here, in the test,
out of `jnp.broadcast_to`. That is deliberate: the measurement stays, the API does not.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax

from eamax.inference.diagnostics import rhat
from eamax.inference.mcmc import inference_loop_multiple_chains
from eamax.inference.warmup import window_adaptation

NUM_CHAINS = 4
SEPARATION = 6.0


def bimodal_logdensity(x):
    """Two well-separated Gaussians. NUTS cannot cross between them."""
    return jnp.logaddexp(
        -0.5 * jnp.sum((x - SEPARATION) ** 2), -0.5 * jnp.sum((x + SEPARATION) ** 2)
    )


def unimodal_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def _sample(logdensity_fn, states, parameters, num_samples=1000, seed=0):
    step = blackjax.nuts.build_kernel()

    def kernel(key, state, chain_params):
        return step(key, state, logdensity_fn, **chain_params)

    return inference_loop_multiple_chains(
        jax.random.key(seed), kernel, states, num_samples, NUM_CHAINS, parameters
    )


def _max_rhat(positions):
    return float(jnp.max(rhat(positions)))


def _fraction_in_positive_mode(positions):
    return float(jnp.mean(np.asarray(positions)[:, :, 0] > 0.0))


class TestSharedVersusPerChainStarts:
    """The regression that motivates the default.

    Replicating one warmed-up state across chains leaves between-chain variance at zero, so
    R-hat measures Monte-Carlo noise instead of convergence. On a target with two modes it
    reports success while sampling one of them.
    """

    def test_a_shared_warmed_start_hides_a_missed_mode(self):
        """Built by hand, because `eamax` deliberately provides no way to do this."""
        start = jnp.asarray([[SEPARATION, SEPARATION]])
        one_state, one_parameters = window_adaptation(
            blackjax.nuts, bimodal_logdensity, start, 400, jax.random.key(1)
        )
        repeat = lambda leaf: jnp.broadcast_to(leaf[:1], (NUM_CHAINS, *jnp.shape(leaf)[1:]))
        states = jax.tree.map(repeat, one_state)
        parameters = jax.tree.map(repeat, one_parameters)
        positions, _ = _sample(bimodal_logdensity, states, parameters)

        # Every chain sits in one mode ...
        assert _fraction_in_positive_mode(positions) == pytest.approx(1.0)
        # ... and the diagnostic reports convergence anyway.
        assert _max_rhat(positions) < 1.01

    def test_per_chain_warmup_from_dispersed_starts_detects_it(self):
        starts = jnp.asarray([[-SEPARATION, 0.0], [SEPARATION, 0.0]] * (NUM_CHAINS // 2))
        states, parameters = window_adaptation(
            blackjax.nuts, bimodal_logdensity, starts, 400, jax.random.key(1)
        )
        positions, _ = _sample(bimodal_logdensity, states, parameters)

        assert 0.2 < _fraction_in_positive_mode(positions) < 0.8
        assert _max_rhat(positions) > 1.1

    def test_a_unimodal_target_raises_no_false_alarm(self):
        starts = jax.random.normal(jax.random.key(2), (NUM_CHAINS, 2)) * 3.0
        states, parameters = window_adaptation(
            blackjax.nuts, unimodal_logdensity, starts, 400, jax.random.key(1)
        )
        positions, info = _sample(unimodal_logdensity, states, parameters)

        assert _max_rhat(positions) < 1.01
        assert int(jnp.sum(info.is_divergent)) == 0


class TestWindowAdaptation:
    def test_chains_are_tuned_independently(self):
        starts = jax.random.normal(jax.random.key(1), (NUM_CHAINS, 2)) * 3.0
        _, parameters = window_adaptation(
            blackjax.nuts, unimodal_logdensity, starts, 400, jax.random.key(0)
        )

        assert parameters["step_size"].shape == (NUM_CHAINS,)
        assert len(np.unique(np.asarray(parameters["step_size"]))) > 1

    def test_chains_end_warmup_in_different_states(self):
        starts = jax.random.normal(jax.random.key(1), (NUM_CHAINS, 2)) * 3.0
        states, _ = window_adaptation(
            blackjax.nuts, unimodal_logdensity, starts, 400, jax.random.key(0)
        )

        assert states.position.shape == (NUM_CHAINS, 2)
        assert float(jnp.std(states.position, axis=0).max()) > 0.0

    def test_a_position_without_a_chain_axis_is_an_error(self):
        with pytest.raises(ValueError, match="leading chain axis"):
            window_adaptation(
                blackjax.nuts, unimodal_logdensity, jnp.zeros(2), 50, jax.random.key(0)
            )

    def test_a_declared_chain_count_is_checked(self):
        starts = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="leading axis of 3"):
            window_adaptation(
                blackjax.nuts, unimodal_logdensity, starts, 50, jax.random.key(0),
                num_chains=4,
            )
