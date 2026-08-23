"""Sampling from a race, through the same parameterization the likelihood scores.

`simulate_race` and `eamax.race.race_loglik` both go through `params_fn`, so a
parameterization cannot drift between the model you simulate from and the model you fit.
In the source repositories that consistency was asserted in a docstring and never tested;
here it is structural, and `tests/test_simulate_roundtrip.py` checks it with a score test.

The guarantee is exactly as wide as one accumulator object: sampling and scoring agree
because `Accumulator.sample` and `Accumulator.log_pdf_sf` describe the same distribution.
Pairing a density from one source with a sampler from another -- a neural flow's `log_prob`
against an Euler-Maruyama sampler, say -- is a choice the caller can still make, and is one
the source repositories deliberately do make. `simulate_race` cannot detect it.
"""

import jax.numpy as jnp


def race_sample(key, accumulator, params, t0, first_response=1):
    """Race a set of accumulators once per trial and report the winner.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    accumulator : Accumulator
        Supplies ``sample(key, params) -> (N, T)`` first-passage times, with ``inf`` where
        an accumulator never crossed.
    params : dict of array
        Per-accumulator parameters, each shape ``(N, T)`` or broadcastable.
    t0 : float or array
        Non-decision time, added to the winning first-passage time.
    first_response : int, optional
        Index of the first accumulator in the output coding.

    Returns
    -------
    rt : array
        Response times, shape ``(T,)``. Trials where *no* accumulator crossed are marked
        ``-1.0``, the sentinel :func:`eamax.race.censored_eval_rt` reads back as a
        right-censored observation.
    response : array
        Winning accumulator, shape ``(T,)``, or ``-1`` on a non-crossing trial.
    """
    fpt = accumulator.sample(key, params)

    winner = jnp.argmin(fpt, axis=0)
    fastest = jnp.min(fpt, axis=0)
    finished = jnp.isfinite(fastest)

    rt = jnp.where(finished, fastest + t0, -1.0)
    response = jnp.where(finished, winner + first_response, -1)
    return rt, response


def simulate_race(key, params_fn, theta, design, accumulator):
    """Simulate one dataset from a parameterization.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    params_fn : callable
        From :func:`eamax.design.build_params_fn` -- the same object the likelihood uses,
        which is what makes simulation and scoring consistent by construction.
    theta : array
        Unconstrained parameter vector.
    design : TrialDesign
        Supplies the trial covariates. Its ``rt`` and ``response`` are ignored (they are
        what is being generated); everything else is read. Simulating against a real
        dataset's own design is the usual way to run a prior-predictive check.
    accumulator : Accumulator
        The first-passage distribution to draw from.

    Returns
    -------
    rt : array
        Response times, shape ``(T,)``.
    response : array
        Winning accumulator, shape ``(T,)``.
    """
    params, t0 = params_fn(theta, design)
    return race_sample(key, accumulator, params, t0, design.first_response)


def simulate_dataset(key, params_fn, theta, design, accumulator):
    """`simulate_race`, returning a `TrialDesign` with the generated data filled in.

    Convenient for round-tripping: the result can be handed straight back to a likelihood
    built on the same ``params_fn``.

    Returns
    -------
    TrialDesign
        A copy of ``design`` with the generated ``rt`` and ``response``.
    """
    rt, response = simulate_race(key, params_fn, theta, design, accumulator)
    return design.replace(rt=rt, response=response)
