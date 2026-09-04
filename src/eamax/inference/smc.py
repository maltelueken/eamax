"""Adaptive tempered SMC: one driver with the union of the two it replaces.

The particle cloud starts from the prior and is annealed towards the posterior, which copes
with the funnel geometry of a semi-centered hierarchical parameterization better than a
single chain does. The number of tempering steps is data-dependent, so the loop is a
`while_loop` rather than a `scan`.

This driver keeps everything a hierarchical fit needs from the algorithm:

* **the evidence** -- accumulated always, because it is one scalar add per iteration and it
  is often the headline scientific output;
* **per-chain tuning and per-chain clouds** -- the shape contract, so that sharing either is
  a named call rather than a silent default;
* **a final resample**, so stored particles are genuinely equally weighted;
* **unique-particle counts** on both sides of that resample, because nothing else detects a
  collapsed cloud.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._blackjax import blackjax, extend_params, smc_resampling, tempering_param
from .diagnostics import count_unique_particles, weight_ess

#: Safety cap on tempering iterations. Reaching it means `lmbda` never got to 1, so the
#: particles are not a posterior sample and the evidence is an underestimate -- check
#: `num_iterations` against it.
DEFAULT_MAX_STEPS = 200


class SMCResult(NamedTuple):
    """Output of :func:`tempered_smc`. Every field carries a leading chain axis.

    Attributes
    ----------
    particles : array
        Shape ``(num_chains, num_particles, D)``. Equally weighted when
        ``resample_final`` was set.
    weights : array
        Shape ``(num_chains, num_particles)``. **Normalized linear-space** weights, not log
        weights -- named for what BlackJAX actually returns. They always belong to
        ``particles``: when ``resample_final`` was set they are uniform, because the
        resample has already spent them. Anything that reweights the stored cloud by these
        is therefore a no-op rather than a double-weighting.
    log_marginal_likelihood : array
        Shape ``(num_chains,)``. The SMC estimate of ``log p(data | model)``.
    num_iterations : array
        Shape ``(num_chains,)``. Compare against ``max_steps``: equality means the
        temperature never reached 1.
    num_unique : array
        Distinct particles after the final resample, shape ``(num_chains,)``.
    num_unique_smc : array
        Distinct particles before it, shape ``(num_chains,)``.
    weight_ess : array
        Weight ESS as a fraction, shape ``(num_chains,)``, of the cloud *before* the final
        resample -- the pre-resample counterpart of ``num_unique_smc``. It is deliberately
        not the ESS of ``weights``: those are uniform once the cloud has been resampled, so
        their ESS is 1 by construction and measures nothing.
    """

    particles: jnp.ndarray
    weights: jnp.ndarray
    log_marginal_likelihood: jnp.ndarray
    num_iterations: jnp.ndarray
    num_unique: jnp.ndarray
    num_unique_smc: jnp.ndarray
    weight_ess: jnp.ndarray


def smc_inference_loop(key, smc_kernel, initial_state, max_steps=DEFAULT_MAX_STEPS):
    """Temper until `lmbda` reaches 1, accumulating the log marginal likelihood.

    Each step's ``log_likelihood_increment`` is the log normalizing constant of the
    incremental importance weights between successive tempered targets. They telescope from
    the normalized prior at ``lmbda = 0`` to the posterior at ``lmbda = 1``, so their sum is
    the SMC estimate of the model evidence. It is accumulated always, since the algorithm
    produces it for free.

    Parameters
    ----------
    key : jax.Array
    smc_kernel : callable
        An SMC step function, e.g. ``blackjax.adaptive_tempered_smc(...).step``.
    initial_state : TemperedSMCState
    max_steps : int, optional
        See :data:`DEFAULT_MAX_STEPS`.

    Returns
    -------
    num_iterations : array
        Scalar.
    final_state : TemperedSMCState
    log_marginal_likelihood : array
        Scalar.
    """

    def cond(carry):
        step, state, _, _ = carry
        return (tempering_param(state) < 1) & (step < max_steps)

    # Not jitted: `lax.while_loop` stages the body out itself, so a decorator here only
    # adds a nested `pjit` to the jaxpr and a jit cache entry keyed on a closure that is
    # rebuilt on every call and so never reused.
    def one_step(carry):
        step, state, log_evidence, loop_key = carry
        loop_key, step_key = jax.random.split(loop_key, 2)
        next_state, info = smc_kernel(step_key, state)
        return step + 1, next_state, log_evidence + info.log_likelihood_increment, loop_key

    num_iterations, final_state, log_evidence, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, 0.0, key)
    )
    return num_iterations, final_state, log_evidence


def tempered_smc(key, log_prior_fn, log_likelihood_fn, initial_particles, mcmc_parameters,
                 *, num_integration_steps=100, target_ess=0.5, num_mcmc_steps=1,
                 max_steps=DEFAULT_MAX_STEPS, resampling_fn=None, resample_final=True,
                 mcmc_step_fn=None, mcmc_init_fn=None, map_chains="sequential"):
    """Run adaptive tempered SMC over several chains.

    Parameters
    ----------
    key : jax.Array
    log_prior_fn, log_likelihood_fn : callable
        ``f(flat_position) -> scalar``, on unconstrained coordinates. They are kept separate
        because tempering interpolates between them. The prior term must already include its
        change-of-variables Jacobian; see
        :meth:`eamax.hierarchical.HierarchicalFlatSpace.log_prob`, which is that term.
    initial_particles : array
        Shape ``(num_chains, num_particles, D)``. Per-chain clouds are the contract, and the
        only option: draw them with
        :func:`eamax.inference.init.init_particles_from_prior`, one independent cloud per
        chain. Sharing a single cloud across chains is not offered -- the chains would then
        differ only in their SMC randomness, so the between-chain spread would understate the
        real uncertainty, which matters most for the log marginal likelihood where that
        spread *is* the standard error of the headline number.
    mcmc_parameters : dict of array
        Inner-kernel tuning, every leaf carrying a leading ``num_chains`` axis --
        ``step_size`` and ``inverse_mass_matrix`` exactly as
        :func:`eamax.inference.warmup.window_adaptation` returns them, one adaptation per
        chain. Pass them through unmodified: a collapsed step size is something to report
        rather than overwrite.
    num_integration_steps : int or None, optional
        Static leapfrog steps for the default HMC mutation kernel. Pass ``None`` when
        supplying a kernel that does not take it.
    target_ess : float, optional
        ESS fraction that sets the next temperature.
    num_mcmc_steps : int, optional
        Mutation steps per tempering iteration. At the default of 1 there is exactly one
        move to rediversify a cloud whose ESS was just driven to ``target_ess``, so
        duplicates compound across iterations -- watch ``num_unique``.
    max_steps : int, optional
    resampling_fn : callable, optional
        Defaults to ``blackjax.smc.resampling.systematic``.
    resample_final : bool, optional
        One extra resample after the loop. Each SMC step is resample -> mutate -> reweight,
        so the returned weights belong to the *final* temperature increment and were never
        resampled away; storing the particles as if uniform therefore reports the
        penultimate, flatter target. Set ``False`` to reproduce a run made without it.
    mcmc_step_fn, mcmc_init_fn : callable, optional
        Default to ``blackjax.hmc.build_kernel()`` and ``blackjax.hmc.init``.
    map_chains : {"sequential", "vmap"}, optional
        ``"sequential"`` uses ``jax.lax.map``, so chains cost wall-clock linearly and
        per-chain clouds cost no extra peak memory. ``"vmap"`` trades memory for speed.

    Returns
    -------
    SMCResult

    Raises
    ------
    ValueError
        If ``initial_particles`` is not three-dimensional, if any tuning leaf disagrees with
        it about the number of chains, or if ``map_chains`` is unrecognised.
    """
    initial_particles = jnp.asarray(initial_particles)
    if initial_particles.ndim != 3:
        raise ValueError(
            f"initial_particles must be (num_chains, num_particles, D); got shape "
            f"{initial_particles.shape}. Draw one independent cloud per chain with "
            "`init_particles_from_prior`; sharing a single cloud across chains is "
            "deliberately not offered."
        )

    num_chains, num_particles, _ = initial_particles.shape

    wrong = {
        name: jnp.shape(leaf)
        for name, leaf in mcmc_parameters.items()
        if jnp.shape(leaf)[:1] != (num_chains,)
    }
    if wrong:
        raise ValueError(
            f"Every mcmc_parameters leaf needs a leading axis of {num_chains} chains; got "
            f"{wrong}. `window_adaptation` returns that shape, one adaptation per chain."
        )

    if map_chains not in {"sequential", "vmap"}:
        raise ValueError(f"map_chains must be 'sequential' or 'vmap', got {map_chains!r}")

    bj = blackjax()
    resampling_fn = smc_resampling().systematic if resampling_fn is None else resampling_fn
    mcmc_step_fn = bj.hmc.build_kernel() if mcmc_step_fn is None else mcmc_step_fn
    mcmc_init_fn = bj.hmc.init if mcmc_init_fn is None else mcmc_init_fn
    extend = extend_params()

    def run_chain(chain_key, tuning, particles):
        run_key, resample_key = jax.random.split(chain_key)

        parameters = dict(tuning)
        if num_integration_steps is not None:
            parameters["num_integration_steps"] = num_integration_steps

        tempered = bj.adaptive_tempered_smc(
            log_prior_fn,
            log_likelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            extend(parameters),
            resampling_fn,
            target_ess,
            num_mcmc_steps=num_mcmc_steps,
        )

        num_iterations, state, log_evidence = smc_inference_loop(
            run_key, tempered.step, tempered.init(particles), max_steps
        )

        num_unique_smc = count_unique_particles(state.particles)

        if resample_final:
            index = resampling_fn(resample_key, state.weights, num_particles)
            particles_out = state.particles[index]
            # The resample has spent the weights: what comes out is an equally weighted
            # cloud. Returning `state.weights` alongside it would pair post-resample
            # particles with pre-resample weights, and a consumer doing the natural thing
            # for an SMC file -- weighting the stored draws -- would weight the cloud twice.
            weights_out = jnp.full_like(state.weights, 1.0 / num_particles)
        else:
            particles_out = state.particles
            weights_out = state.weights

        return SMCResult(
            particles=particles_out,
            weights=weights_out,
            log_marginal_likelihood=log_evidence,
            num_iterations=num_iterations,
            num_unique=count_unique_particles(particles_out),
            num_unique_smc=num_unique_smc,
            # Of `state.weights`, i.e. before any final resample: the diagnostic only says
            # something while the weights are still uneven.
            weight_ess=weight_ess(state.weights),
        )

    keys = jax.random.split(key, num_chains)

    if map_chains == "vmap":
        return jax.vmap(run_chain)(keys, mcmc_parameters, initial_particles)

    return jax.lax.map(
        lambda args: run_chain(*args), (keys, mcmc_parameters, initial_particles)
    )
