"""The accumulator interface: a first-passage-time distribution, and nothing else.

An accumulator answers two questions about *decision time* -- how dense is it here, and
how much probability is left beyond here -- plus how to draw from it. It knows nothing
about non-decision time, about racing, about which accumulator won, or about how bad a
trial has to be before it gets floored. All of that belongs to `eamax.race`.

That split is not cosmetic. In the source repos the per-accumulator function owned the
`t0` shift, the parameter guards, the density floor *and* the invalid-RT penalty, and it
had to warn callers in prose not to clamp its output again. One consumer clamped it
anyway, flattening the penalty to a constant and silently removing the gradient that
pushes `t0` back into the valid region. Here there is no intermediate to re-clamp:
accumulators return raw log-densities, and the race applies the guards once at the end.

Parameters travel as a dict keyed by `param_names` rather than as positional arguments,
so a parameterization can be checked against an accumulator once, at construction, and so
the same design layer can drive a three-parameter Wald and a five-parameter pulsed
accumulator without reshuffling call sites.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Accumulator(Protocol):
    """First-passage-time distribution for a single racing accumulator.

    Attributes
    ----------
    param_names : tuple of str
        Names this accumulator expects in the ``params`` dict, in their canonical order. A
        parameterization is valid for this accumulator when it produces at least these keys.
    """

    param_names: tuple[str, ...]

    def log_pdf_sf(self, t, params):
        """Log density and log survival at *decision* time `t`.

        Parameters
        ----------
        t : array
            Decision times (``rt - t0``), already floored away from zero by the caller.
            Shape ``(N, T)`` or anything broadcastable to it.
        params : dict of array
            Keyed by ``param_names``; each value broadcastable to ``t``'s shape.

        Returns
        -------
        log_pdf, log_sf : array
            Raw. Not floored, not penalised, not NaN-contained -- :mod:`eamax.race` does
            all three, once, on the assembled trial total.
        """
        ...

    def sample(self, key, params):
        """Draw one first-passage time per element of the broadcast parameter shape.

        Returns
        -------
        array
            Decision times, with ``inf`` where the accumulator never crossed the boundary
            within whatever horizon the implementation has. The race turns an all-``inf``
            trial into a right-censored observation; representing "no crossing" as ``inf``
            rather than a negative sentinel is what lets ``min`` over accumulators do the
            right thing without a special case.
        """
        ...


def validate_params(accumulator, params):
    """Check that `params` covers everything `accumulator` needs.

    Called once when a parameterization is bound to an accumulator, so a missing or
    misspelled quantity surfaces as a readable error at setup rather than as a ``KeyError``
    inside a traced function -- or worse, as a silently wrong density when the name happens
    to exist but means something else.

    Parameters
    ----------
    accumulator : Accumulator
        The accumulator whose ``param_names`` must be covered.
    params : dict
        The parameterization's output.

    Returns
    -------
    dict
        ``params`` unchanged.

    Raises
    ------
    KeyError
        If any name in ``accumulator.param_names`` is missing from ``params``.
    """
    missing = [name for name in accumulator.param_names if name not in params]
    if missing:
        raise KeyError(
            f"{type(accumulator).__name__} needs parameters {list(accumulator.param_names)}; "
            f"missing {missing}. Got {sorted(params)}."
        )
    return params
