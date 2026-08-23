"""NUTS drivers: run chains in lockstep, and fit many datasets in one job.

The sampling loop is a `scan` over draws with a `vmap` over chains inside each step -- a
single XLA computation that runs efficiently on one GPU and, unlike `pmap`, composes with
an outer `vmap` over datasets. That outer map is what turns "one job per dataset" into one
job for a whole study.

Two contracts are worth stating because they are what make the nesting work:

* `make_logdensity_fn(data)` is called **inside** the driver, not by the caller. Under the
  outer `vmap`, `data` is a per-dataset traced slice, so building the closure in here gives
  each dataset its own; building it outside would bake one dataset in before the vmap.
* `make_init_positions(key, data)` is called inside for the same reason, and that is what
  lets starting values be per-dataset *and* per-chain *and* support-aware at once -- the
  combination none of the source repositories can express, since theirs is a single fixed
  vector with one entry overwritten from the data.

Positions are unconstrained throughout. The log-density owns its own transform and
Jacobian; nothing here transforms anything.

There is one warm-up path and no way around it. ``make_init_positions`` must return one
dispersed start *per chain* -- :func:`eamax.inference.init.init_positions_from_prior` is
what it is for -- and every chain is then adapted independently by
:func:`eamax.inference.warmup.window_adaptation`. The ``shared_warmup=True`` switch that
adapted one chain and replicated it has been **removed**, as has the post-warm-up tuning
repair; see :mod:`eamax.inference.warmup` for why neither should be reachable from a
consumer repository.
"""

import jax

from ._blackjax import blackjax
from .warmup import window_adaptation


def inference_loop_multiple_chains(key, kernel, initial_state, num_samples, num_chains,
                                   kernel_params=None):
    """Advance `num_chains` chains in lockstep, recording every draw.

    Parameters
    ----------
    key : jax.Array
        PRNG key; split once per step, then once per chain within the step.
    kernel : callable
        Called as ``kernel(key, state)`` when ``kernel_params`` is ``None`` -- i.e. the
        tuning constants are already bound -- and as ``kernel(key, state, params)``
        otherwise.
    initial_state : pytree
        Per-chain sampler states, leading axis ``num_chains``.
    num_samples : int
        Draws recorded per chain. Nothing is discarded as burn-in; warm-up already happened.
    num_chains : int
    kernel_params : pytree, optional
        Leaves carrying a leading axis of length ``num_chains``, so each chain uses the
        tuning its own adaptation produced. Required whenever chains were adapted
        independently, which is :func:`eamax.inference.warmup.window_adaptation`'s output.

    Returns
    -------
    positions : array
        Shape ``(num_samples, num_chains, ...)``.
    infos : pytree
        Sampler diagnostics with the same two leading axes -- ``is_divergent``,
        ``acceptance_rate``, ``num_integration_steps`` for NUTS.
    """

    @jax.jit
    def one_step(states, step_key):
        keys = jax.random.split(step_key, num_chains)
        if kernel_params is None:
            states, infos = jax.vmap(kernel)(keys, states)
        else:
            states, infos = jax.vmap(kernel)(keys, states, kernel_params)
        return states, (states.position, infos)

    keys = jax.random.split(key, num_samples)
    _, (positions, infos) = jax.lax.scan(one_step, initial_state, keys)

    return positions, infos


def fit_nuts(key, data, make_logdensity_fn, make_init_positions, num_chains,
             num_steps_warmup, num_steps_sampling, *,
             sampler_fun=None, **warmup_kwargs):
    """Fit one dataset's posterior, chains adapted independently and advanced together.

    Parameters
    ----------
    key : jax.Array
    data : array
        One dataset. Passed to both factories and otherwise untouched.
    make_logdensity_fn : callable
        ``f(data) -> logdensity_fn(position) -> scalar``, on unconstrained positions.
    make_init_positions : callable
        ``f(key, data) -> (num_chains, P)`` unconstrained starting positions. Must be
        traceable. Every row is used, so all ``num_chains`` of them must be genuinely
        dispersed and inside the support --
        :func:`eamax.inference.init.init_positions_from_prior` is the intended source. See
        :mod:`eamax.inference.init`.
    num_chains, num_steps_warmup, num_steps_sampling : int
    sampler_fun : callable, optional
        BlackJAX sampler constructor; defaults to ``blackjax.nuts``.
    **warmup_kwargs
        Forwarded to window adaptation.

    Returns
    -------
    positions : array
        Shape ``(num_steps_sampling, num_chains, P)``.
    infos : pytree
        Sampler diagnostics with the same leading axes.
    """
    sampler_fun = blackjax().nuts if sampler_fun is None else sampler_fun

    init_key, warmup_key, sampling_key = jax.random.split(key, 3)

    logdensity_fn = make_logdensity_fn(data)
    init_positions = make_init_positions(init_key, data)

    last_states, parameters = window_adaptation(
        sampler_fun, logdensity_fn, init_positions, num_steps_warmup, warmup_key,
        num_chains=num_chains, **warmup_kwargs,
    )

    step = sampler_fun.build_kernel()

    def kernel(chain_key, state, params):
        return step(chain_key, state, logdensity_fn, **params)

    return inference_loop_multiple_chains(
        sampling_key, kernel, last_states, num_steps_sampling, num_chains, parameters
    )


def fit_nuts_batch(key, data, make_logdensity_fn, make_init_positions, num_chains,
                   num_steps_warmup, num_steps_sampling, **kwargs):
    """:func:`fit_nuts` mapped over `data`'s leading axis.

    Every dataset in one XLA computation, nesting datasets (`vmap`) -> draws (`scan`) ->
    chains (`vmap`). All datasets must have the same number of trials, since ``data`` is one
    stacked array.

    Parameters
    ----------
    key : jax.Array
        Split once per dataset, so results do not depend on how the batch was assembled.
    data : array
        Shape ``(num_datasets, ...)``.
    make_logdensity_fn, make_init_positions : callable
        See :func:`fit_nuts`.
    num_chains, num_steps_warmup, num_steps_sampling : int
    **kwargs
        Forwarded to :func:`fit_nuts`.

    Returns
    -------
    positions : array
        Shape ``(num_datasets, num_steps_sampling, num_chains, P)``.
    infos : pytree
        Diagnostics with a matching leading ``num_datasets`` axis.
    """
    keys = jax.random.split(key, data.shape[0])

    def fit_one(dataset_key, dataset):
        return fit_nuts(
            dataset_key, dataset, make_logdensity_fn, make_init_positions, num_chains,
            num_steps_warmup, num_steps_sampling, **kwargs,
        )

    return jax.vmap(fit_one, in_axes=(0, 0))(keys, data)
