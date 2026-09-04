"""The race: one likelihood for any number of independent accumulators.

With independent accumulators, observing that accumulator `r` finished at time `t` while
the others had not yet finished gives

    p_r(t) * prod_{k != r} S_k(t)

so every race likelihood is a winner's density times the losers' survival functions. That
is the whole model, written once here for any number of accumulators, with the
two-accumulator case an instance rather than a special path.

Three things this module owns:

* **Non-decision time.** `t0` is a race-level shift, not an accumulator property.
  Accumulators see decision times.
* **All guard logic.** The floor, NaN containment and the padding mask apply once, here,
  to the assembled trial total. Callers get a finished number and no intermediate to
  re-clamp -- see `eamax.numerics` for why that matters.
* **Right-censoring.** Trials that never crossed have to be handled on *both* sides of the
  accumulator call: the evaluation time is substituted before, and the trial's score is
  replaced after. That is why `race_loglik` takes a closure rather than arrays.
"""

import jax.numpy as jnp

from .numerics import MIN_P, MIN_RT, contain_nan, finalize_trial_logp


def winner_mask(response, num_accumulators, first_response=0):
    """`(N, T)` boolean: is accumulator `k` the one that won trial `t`?

    Parameters
    ----------
    response : array
        Winning accumulator per trial, shape ``(T,)``.
    num_accumulators : int
        N.
    first_response : int, optional
        Index of the first accumulator in ``response``'s coding. Pass ``0`` when choices are
        labelled ``0..N-1`` and ``1`` when they are labelled ``1..N``.

    Returns
    -------
    array
        Boolean array of shape ``(N, T)``.
    """
    index = jnp.arange(num_accumulators) + first_response
    return index[:, None] == jnp.asarray(response)[None, :]


def censored_eval_rt(rt, t0, t_max):
    """Redirect the "never crossed" sentinel onto the RT whose decision time is `t_max`.

    Simulators mark a non-crossing trial with a negative RT. Such a trial is not missing
    data -- it is the observation `T > t_max`, whose likelihood is the probability that
    *every* accumulator was still running. Evaluating at `t_max + t0` is what makes the
    survival terms come out right; the caller then discards the density term.

    Parameters
    ----------
    rt : array
        Observed response times, shape ``(T,)``. Negative marks a non-crossing trial.
    t0 : float or array
        Non-decision time.
    t_max : float or None
        Simulation horizon. ``None`` disables the censoring model.

    Returns
    -------
    rt_eval : array
        Response times with the sentinel redirected onto ``t_max + t0``.
    is_censored : array or None
        Boolean, shape ``(T,)``, or ``None`` when ``t_max`` is ``None`` -- in which case a
        negative RT is simply an infeasible trial and comes out at the floor.
    """
    if t_max is None:
        return rt, None
    is_censored = rt < 0.0
    return jnp.where(is_censored, t_max + t0, rt), is_censored


def gather_by_mask(mask, *arrays):
    """Collapse `(N, T)` arrays to `(T,)` by picking the one accumulator `mask` selects.

    For hybrid races where exactly one accumulator per trial is special -- the one carrying
    a conflict pulse, say -- and its density comes from an expensive source such as a
    neural flow. Gathering first means that source is evaluated **once per trial rather
    than once per accumulator**, which on a four-response task is a 4x saving.

    The masked sum is used rather than ``jnp.take_along_axis`` because the selecting index
    is a traced data column, and this keeps the shape static. It requires exactly one
    ``True`` per column; with none it yields 0.0, with several it yields their sum.

    Parameters
    ----------
    mask : array
        Boolean, shape ``(N, T)``, with exactly one ``True`` per column.
    *arrays : array
        Arrays of shape ``(N, T)`` to collapse.

    Returns
    -------
    tuple of array
        One array of shape ``(T,)`` per input.
    """
    return tuple(jnp.sum(jnp.where(mask, a, 0.0), axis=0) for a in arrays)


def overlay_by_mask(mask, alternative, base):
    """Substitute a per-trial `(T,)` result into an `(N, T)` base where `mask` is True.

    The inverse of :func:`gather_by_mask`, closing the hybrid-race round trip.

    Parameters
    ----------
    mask : array
        Boolean, shape ``(N, T)``.
    alternative : array
        Per-trial values, shape ``(T,)``, to substitute where ``mask`` is ``True``.
    base : array
        Values of shape ``(N, T)`` to substitute into.

    Returns
    -------
    array
        Shape ``(N, T)``.
    """
    return jnp.where(mask, jnp.asarray(alternative)[None, :], base)


def race_from_arrays(
    response,
    log_pdf,
    log_sf,
    *,
    mask=None,
    is_censored=None,
    first_response=0,
    min_p=MIN_P,
):
    """Assemble a per-trial log-likelihood from per-accumulator densities.

    The lower-level half of `race_loglik`, exposed for tests and for callers who have
    already built their `(N, T)` arrays by some other route (a hybrid race, typically).
    Prefer `race_loglik`, which also owns the censoring substitution.

    Parameters
    ----------
    response : array
        Winning accumulator per trial, shape ``(T,)``.
    log_pdf, log_sf : array
        Log density and log survival, shape ``(N, T)``, raw.
    mask : array, optional
        Boolean, shape ``(T,)``. False trials contribute exactly ``0.0``, so padding a
        ragged dataset to a rectangle is safe regardless of what the padding holds.
    is_censored : array, optional
        Boolean, shape ``(T,)``, from :func:`censored_eval_rt`.
    first_response : int, optional
        See :func:`winner_mask`.

    Returns
    -------
    array
        Per-trial log-likelihood, shape ``(T,)``. Callers sum; ``eamax`` does not, so you
        are free to reduce over whichever axis suits your model.
    """
    log_pdf = contain_nan(jnp.asarray(log_pdf), min_p)
    log_sf = contain_nan(jnp.asarray(log_sf), min_p)

    is_winner = winner_mask(response, log_pdf.shape[0], first_response)

    # Sum the winner's density and the losers' survival. Both branches are materialised
    # and selected, rather than indexed, so shapes stay static under jit and vmap.
    total = jnp.sum(jnp.where(is_winner, log_pdf, 0.0), axis=0)
    total = total + jnp.sum(jnp.where(is_winner, 0.0, log_sf), axis=0)

    if is_censored is not None:
        # Nobody finished: the observation is that every accumulator was still running.
        # This is `sum(log_sf)` for any N, which is why censoring needs no routing.
        total = jnp.where(is_censored, jnp.sum(log_sf, axis=0), total)

    out = finalize_trial_logp(total, min_p)

    if mask is not None:
        out = jnp.where(mask, out, 0.0)
    return out


def race_loglik(
    rt,
    response,
    t0,
    pdf_sf_fn,
    *,
    mask=None,
    t_max=None,
    first_response=0,
    min_p=MIN_P,
    min_rt=MIN_RT,
):
    """Per-trial log-likelihood of a race, given a per-accumulator density function.

    The caller owns every outer vmap. Shapes here are one dataset's worth of trials; map
    over datasets, subjects or prior draws outside.

    Parameters
    ----------
    rt : array
        Observed response times, shape ``(T,)``. Negative marks a non-crossing trial, which
        is treated as right-censored when ``t_max`` is given and as infeasible otherwise.
    response : array
        Winning accumulator per trial, shape ``(T,)``.
    t0 : float or array
        Non-decision time; scalar, or shape ``(T,)`` if it varies by trial. Never
        per-accumulator.
    pdf_sf_fn : callable
        ``f(decision_time: (T,)) -> ((N, T), (N, T))`` giving raw log density and log
        survival for every accumulator. Typically
        ``lambda t: accumulator.log_pdf_sf(t, params)``; a hybrid race builds its arrays
        with :func:`gather_by_mask` / :func:`overlay_by_mask` inside this closure.
    mask : array, optional
        Boolean, shape ``(T,)``, for padded trials.
    t_max : float, optional
        Simulation horizon, enabling the censoring model. Must match the horizon the data
        were generated with.

    Returns
    -------
    array
        Per-trial log-likelihood, shape ``(T,)``.
    """
    rt = jnp.asarray(rt)
    rt_eval, is_censored = censored_eval_rt(rt, t0, t_max)

    decision_time = jnp.maximum(rt_eval - t0, min_rt)

    log_pdf, log_sf = pdf_sf_fn(decision_time)

    return race_from_arrays(
        response,
        log_pdf,
        log_sf,
        mask=mask,
        is_censored=is_censored,
        first_response=first_response,
        min_p=min_p,
    )
