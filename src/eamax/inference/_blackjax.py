"""Lazy access to BlackJAX.

BlackJAX is an optional dependency (`eamax[inference]`), imported inside function bodies
rather than at module scope. That is what lets `eamax.inference.transforms`,
`eamax.inference.init` and `eamax.inference.posterior` -- none of which sample anything --
be imported and used in an environment that has no sampler installed, and it keeps
`import eamax` free of a dependency most of the library does not need.

Two of the four accessors exist because the names are not where you would guess:
`get_filter_adapt_info_fn` is not re-exported at BlackJAX's top level, and neither is
`smc.resampling` -- `blackjax.smc.__init__` exports `extend_params` and submodules but not
the resampling functions, so `blackjax.smc.resampling` needs its own import statement. A
shim that forgets either works in a warm environment and fails on a fresh install.
"""

_MISSING = (
    "eamax needs BlackJAX for this operation, but `blackjax` is not importable.\n\n"
    "    pip install 'eamax[inference]'\n"
    "    # or: pip install blackjax\n\n"
    "Only the samplers need it. `eamax.inference.transforms`, `.init` and `.posterior` "
    "work without it, as does the rest of eamax."
)

#: First release carrying `blackjax.diagnostics.rhat`, the rank-normalized split-R-hat of
#: Vehtari et al. (2021). Earlier releases ship only `potential_scale_reduction`, the plain
#: Gelman-Rubin (1992) statistic -- a different number, not an older spelling of the same
#: one. See `eamax.inference.diagnostics.rhat`.
RHAT_MIN_VERSION = "1.6"


def blackjax():
    """The `blackjax` module."""
    try:
        import blackjax
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return blackjax


def smc_resampling():
    """The `blackjax.smc.resampling` module, which is not re-exported at top level."""
    try:
        import blackjax.smc.resampling as resampling
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return resampling


def extend_params():
    """`blackjax.smc.extend_params`.

    Takes a single argument on every version in range; a much older signature took
    ``(num, params)``.
    """
    try:
        from blackjax.smc import extend_params
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return extend_params


def get_filter_adapt_info_fn():
    """`blackjax.adaptation.base.get_filter_adapt_info_fn`, not exported at top level."""
    try:
        from blackjax.adaptation.base import get_filter_adapt_info_fn
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return get_filter_adapt_info_fn

#: `TemperedSMCState`'s inverse-temperature field was renamed from ``lmbda`` to
#: ``tempering_param`` between 1.2.5 and 1.6.2. It is the loop's only termination
#: condition, so reading it under one name breaks on the other.
_TEMPERING_FIELDS = ("tempering_param", "lmbda")


def tempering_param(state):
    """The current inverse temperature of a `TemperedSMCState`.

    Parameters
    ----------
    state : TemperedSMCState

    Returns
    -------
    array
        Scalar in ``[0, 1]``; 1 means the tempered target has reached the posterior.

    Raises
    ------
    AttributeError
        If the state carries neither known field name, which means a version of BlackJAX
        this has not been checked against.
    """
    for field in _TEMPERING_FIELDS:
        if hasattr(state, field):
            return getattr(state, field)
    raise AttributeError(
        f"{type(state).__name__} has none of {_TEMPERING_FIELDS}; this BlackJAX renamed "
        "the tempered SMC inverse-temperature field again. Add the new name to "
        "`eamax.inference._blackjax._TEMPERING_FIELDS`."
    )
