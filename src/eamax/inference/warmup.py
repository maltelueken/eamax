"""Window adaptation: one adaptation per chain, and no repair afterwards.

Adaptation produces two things a sampler needs -- a step size and an inverse mass matrix --
and one thing a *reader* needs: chains that started somewhere different from each other.
The three source repositories treat the second as optional, and it is not.

Replicating a single warmed-up state across chains leaves R-hat with nothing to measure.
Between-chain variance starts at zero and every chain begins inside whichever mode the one
warm-up happened to find, so the diagnostic reports Monte-Carlo noise. Measured on a
bimodal target with modes at +-6 (4 chains x 2000 draws): the shared start reports max
R-hat 1.001 while putting 100% of its mass in a single mode; an overdispersed start reports
1.734 and splits 50/50, which is the truth. The draws from the shared start are not
*biased* -- every chain gets a fresh key, and adaptation has already brought the state to
stationarity -- but the number reported alongside them cannot fail, so it carries no
information. A downstream filter that drops fits on that number is filtering on noise.

So there is exactly one warm-up entry point, :func:`window_adaptation`, and it is per-chain.
A ``broadcast_warmup`` that adapted one chain and replicated it across the rest used to live
here and has been **removed**, along with the ``shared_warmup`` switch on
:func:`eamax.inference.mcmc.fit_nuts` that reached it. Consumers should not have the option:
what it saves is warm-up work, and what it spends is the one diagnostic that says whether
the fit is usable at all.

**The prescribed path is: draw dispersed starts from the prior with**
:func:`eamax.inference.init.init_positions_from_prior` **(or**
:func:`~eamax.inference.init.init_particles_from_prior` **for SMC), adapt every chain with**
:func:`window_adaptation`, **then sample.** Nothing in between.

Nothing repairs a collapsed adaptation either. A ``repair_degenerate_tuning`` that replaced
a step size two orders of magnitude below its siblings' with the healthy chains' median has
also been removed. It treated a symptom: in every population that produced a degenerate step
size, the degenerate chain was the chain with the most subjects whose ``t0`` started above
their fastest observed response time -- a start outside the support, which
:class:`eamax.inference.init.T0Support` prevents at the source. Repairing the tuning
afterwards left those chains under-dispersed at 0.16-0.83x their siblings' spread and still
breaking R-hat (1.94 / 2.60 / 2.76 across three populations), so the repair bought a
plausible-looking step size and nothing else. A chain whose adaptation collapses should be
*reported*, not rescued: start inside the support, and if a step size still collapses, that
is a finding about the posterior and belongs in the run's output.
"""

import jax
import jax.numpy as jnp

from ._blackjax import blackjax, get_filter_adapt_info_fn


def _leading_axis(tree, declared=None):
    """Infer the chain count from a pytree of per-chain positions.

    A one-dimensional array is genuinely ambiguous -- ``(4,)`` is four chains of a scalar
    parameter or one chain of a four-parameter model, and nothing in the array says which.
    So an inferred count requires every leaf to carry a chain axis *on top of* at least one
    parameter axis, and the ambiguous case has to be resolved by declaring ``num_chains``.
    Silently adapting one chain because a single position was passed is the mistake this
    exists to prevent.
    """
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError("Initial positions are empty.")

    shapes = {jnp.shape(leaf)[:1] for leaf in leaves}
    if len(shapes) != 1 or shapes == {()}:
        raise ValueError(
            f"Initial positions must share one leading chain axis; got leading shapes "
            f"{sorted(shapes)}."
        )
    inferred = int(next(iter(shapes))[0])

    if declared is not None:
        if inferred != declared:
            raise ValueError(
                f"init_positions has a leading axis of {inferred} but "
                f"num_chains={declared}."
            )
        return declared

    if min(jnp.ndim(leaf) for leaf in leaves) < 2:
        raise ValueError(
            "Initial positions need a leading chain axis, and this looks like a single "
            "position: every leaf should have at least two axes, (num_chains, ...). "
            "Draw one dispersed start per chain with `init_positions_from_prior`; adapting "
            "one chain and replicating it is deliberately not offered, because it makes "
            "R-hat unable to fail. If these really are per-chain positions of a scalar "
            "parameter, pass `num_chains` explicitly."
        )
    return inferred


def window_adaptation(sampler_fun, logdensity_fn, init_positions, num_steps, key,
                      num_chains=None, **kwargs):
    """Adapt each chain independently, from its own starting position.

    The cost is ``num_chains`` times the warm-up work. What it buys is starts that are both
    overdispersed *and* converged, which is what Gelman-Rubin assumes and what makes R-hat
    able to fail. This is the only warm-up `eamax` offers, deliberately -- see the module
    docstring.

    The tuning it returns is meant to be used as it comes. If one chain's step size
    collapses, do not overwrite it with its siblings': check that every start was inside the
    support (:class:`eamax.inference.init.T0Support`), and report the collapse.

    Adaptation info is filtered out: keeping it for every chain and every step is memory
    with no downstream consumer.

    Parameters
    ----------
    sampler_fun : callable
        BlackJAX sampler constructor, e.g. ``blackjax.nuts``.
    logdensity_fn : callable
        Unnormalised log posterior on unconstrained positions, shared across chains.
    init_positions : pytree
        Starting positions with a leading axis of length ``num_chains``. They should be
        genuinely dispersed and must all lie inside the support -- see
        :mod:`eamax.inference.init`.
    num_steps : int
        Adaptation steps per chain.
    key : jax.Array
        PRNG key, split once per chain.
    num_chains : int, optional
        Checked against the leading axis when given.
    **kwargs
        Forwarded to ``blackjax.window_adaptation`` (``target_acceptance_rate``, ...).

    Returns
    -------
    last_states : pytree
        Per-chain final states, leading chain axis.
    parameters : dict
        ``step_size`` shape ``(num_chains,)`` and ``inverse_mass_matrix`` shape
        ``(num_chains, ...)``. Pass straight to
        :func:`eamax.inference.mcmc.inference_loop_multiple_chains` as ``kernel_params``.

    Raises
    ------
    ValueError
        If the positions carry no common leading axis. A single shared position is not
        expanded into chains for you: draw one start per chain with
        :func:`eamax.inference.init.init_positions_from_prior`.
    """
    inferred = _leading_axis(init_positions, num_chains)

    kwargs.setdefault("adaptation_info_fn", get_filter_adapt_info_fn()())
    adapt = blackjax().window_adaptation(sampler_fun, logdensity_fn, **kwargs)

    def run_one(chain_key, position):
        (last_state, parameters), _ = adapt.run(chain_key, position, num_steps=num_steps)
        return last_state, parameters

    return jax.vmap(run_one)(jax.random.split(key, inferred), init_positions)
