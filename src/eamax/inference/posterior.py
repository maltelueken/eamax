"""Array-level posterior post-processing: layout, pooling, thinning, and read-back.

Four small operations sit between a sampler's raw output and anything that consumes it --
a file, a figure, a comparison against amortized draws. They are separated from
:mod:`eamax.io` because none of them needs xarray, ArviZ, or a file: they are pure array
transformations, testable without touching disk and usable by a consumer that stores its
posteriors some other way.

They are also separated from :mod:`eamax.io` for a second reason, which is a rule rather
than a convenience: **`eamax.io` reads and writes, and does nothing else.** It does not
compute R-hat, does not drop unconverged fits, and does not thin. Those are analysis
decisions with a threshold in them, and a reader that applies them silently returns an array
whose shape depends on a diagnostic. :func:`eamax.io.load_dataset_posterior` therefore hands
back every draw of every chain, and the caller composes
:func:`eamax.inference.diagnostics.rhat`, :func:`pool_chains` and :func:`thin` in whatever
order its analysis calls for -- visibly, in its own code.

The third, :func:`back_transform_then_select`, is the reason this module exists as
something other than a convenience. Reading samples back involves two steps that do not
commute, and getting them the wrong way round produces plausible-looking numbers rather
than an error. `eam-abi-robustness` gets the order right and records why in a comment
(``src/mcmc.py:270-272``); a function makes it impossible to get wrong.

Nothing here imports a sampler.
"""

import numpy as np

POSTERIOR_DIMS = ("chain", "draw", "dataset", "param")


def to_arviz_layout(positions):
    """Reorder batched sampler output into the layout ArviZ expects.

    :func:`eamax.inference.mcmc.fit_nuts_batch` returns ``(dataset, draw, chain, param)``:
    ``vmap`` over datasets puts its axis first, and the ``scan`` inside each fit puts draws
    ahead of the vmapped chains. ArviZ wants ``(chain, draw, dataset, param)``, with the
    two batch dimensions treated differently -- ``chain`` and ``draw`` are the sampling
    axes, ``dataset`` is just another named coordinate.

    Parameters
    ----------
    positions : array_like
        Samples shaped ``(dataset, draw, chain, param)``.

    Returns
    -------
    numpy.ndarray
        The same samples shaped ``(chain, draw, dataset, param)``.

    Raises
    ------
    ValueError
        If ``positions`` is not rank 4. The reorder is a fixed permutation, so a rank
        mismatch means the caller has a different layout than it thinks and the
        permutation would silently scramble it.
    """
    positions = np.asarray(positions)

    if positions.ndim != len(POSTERIOR_DIMS):
        msg = (
            f"Expected samples shaped (dataset, draw, chain, param), got an array of rank "
            f"{positions.ndim}."
        )
        raise ValueError(msg)

    return np.transpose(positions, (2, 1, 0, 3))


def pool_chains(samples):
    """Merge the chain and draw axes of ``(chain, draw, dataset, param)`` samples.

    Chain-major: every draw of chain 0, then every draw of chain 1, matching what
    ``xarray``'s ``stack(sample=("chain", "draw"))`` produces, so a pooled array and a
    pooled ``Dataset`` index the same way. The order is load-bearing whenever pooled draws
    are matched against anything else per-draw.

    Pool *after* any diagnostic that needs the chain axis -- :func:`~eamax.inference.diagnostics.rhat`
    reads between-chain variance, and there is nothing to read once the axes are merged.

    Parameters
    ----------
    samples : array_like
        Shape ``(chain, draw, dataset, param)``, as
        :func:`eamax.io.load_dataset_posterior` returns.

    Returns
    -------
    numpy.ndarray
        Shape ``(dataset, chain * draw, param)`` -- the pooled layout :func:`thin` and
        :func:`select_params` default to.

    Raises
    ------
    ValueError
        If ``samples`` is not rank 4, for the same reason as :func:`to_arviz_layout`: the
        reshape would silently scramble a different layout.
    """
    samples = np.asarray(samples)

    if samples.ndim != len(POSTERIOR_DIMS):
        msg = (
            f"Expected samples with dims {POSTERIOR_DIMS}, got an array of rank "
            f"{samples.ndim}."
        )
        raise ValueError(msg)

    num_chains, num_draws, num_datasets, num_params = samples.shape

    # (chain, draw, dataset, param) -> (dataset, chain, draw, param), then merge the middle
    # two so that chain varies slowest.
    return np.transpose(samples, (2, 0, 1, 3)).reshape(
        num_datasets, num_chains * num_draws, num_params
    )


def thin(samples, num_target, *, axis=1):
    """Stride-thin an axis down to at most ``num_target`` entries.

    Thinning here is for *comparability*, not for autocorrelation: MCMC and amortized
    posteriors are compared at equal sample size, and the MCMC side usually has more draws.
    Striding rather than truncating keeps the retained draws spread over the whole chain,
    so a slowly mixing chain is not represented by its first tenth.

    ``num_target`` is an upper bound, not a promise. The stride is an integer, so the result
    holds ``ceil(n / stride)`` draws capped at ``num_target``; with ``n = 300`` and
    ``num_target = 200`` the stride is 1 and 200 come back, but with ``n = 399`` the stride
    is 1 and again 200 come back, while ``n = 401`` gives a stride of 2 and 201 -> capped to
    200. An axis already at or below ``num_target`` is returned whole.

    Parameters
    ----------
    samples : array_like
    num_target : int
        Maximum number of entries to keep. Must be positive.
    axis : int, optional
        Axis to thin. Defaults to 1, the sample axis of the pooled
        ``(dataset, sample, param)`` layout.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If ``num_target`` is not positive.
    """
    samples = np.asarray(samples)

    if num_target <= 0:
        msg = f"num_target must be positive, got {num_target}."
        raise ValueError(msg)

    stride = max(samples.shape[axis] // num_target, 1)

    index = [slice(None)] * samples.ndim

    index[axis] = slice(None, None, stride)
    strided = samples[tuple(index)]

    index[axis] = slice(None, num_target)
    return strided[tuple(index)]


def select_params(samples, stored_names, wanted_names, *, axis=-1):
    """Select named parameters along an axis, by name rather than by position.

    Parameters
    ----------
    samples : array_like
    stored_names : sequence of str
        Names of the entries along ``axis``, in order.
    wanted_names : sequence of str
        Names to keep, in the order they should appear in the result. May repeat.
    axis : int, optional

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If ``stored_names`` does not match the length of ``axis``, or if any wanted name
        is not stored. Both are silent-corruption failures otherwise: a position-based
        selection against the wrong name list returns the wrong parameter's samples.
    """
    samples = np.asarray(samples)
    stored_names = [str(name) for name in stored_names]
    wanted_names = [str(name) for name in wanted_names]

    if samples.shape[axis] != len(stored_names):
        msg = (
            f"Axis {axis} has length {samples.shape[axis]} but {len(stored_names)} names "
            f"were given."
        )
        raise ValueError(msg)

    missing = [name for name in wanted_names if name not in stored_names]
    if missing:
        msg = f"Requested parameters {missing} are not among the stored {stored_names}."
        raise ValueError(msg)

    return np.take(samples, [stored_names.index(name) for name in wanted_names], axis=axis)


def back_transform_then_select(samples, to_constrained, stored_names, wanted_names, *, axis=-1):
    """Map samples back to the natural scale, then keep the parameters asked for.

    The order is the whole point, and it is not interchangeable. A hierarchical model's
    leading entries are prior hyperparameters bounded on both sides, transformed with a
    ``Sigmoid``, while the subject-level parameters that follow are strictly positive and
    transformed with ``Exp`` -- see :class:`eamax.inference.transforms.BlockTransform`.
    Such a transform is defined on the *whole* vector: it reads its own block boundary off
    the trailing axis's length. Selecting first hands it a shorter vector, in which the
    entries that used to be bounded are gone and positive ones have slid into their places,
    and it will happily apply ``Sigmoid`` to the first few and ``Exp`` to the rest. The
    result has the right shape, the right names, and the wrong numbers.

    Transforming first costs a little arithmetic on parameters about to be discarded, and
    removes the failure entirely.

    Parameters
    ----------
    samples : array_like
        Unconstrained samples with parameters along ``axis``.
    to_constrained : callable
        Unconstrained -> natural scale, applied to the full parameter vector. E.g.
        :meth:`eamax.inference.transforms.BlockTransform.forward`.
    stored_names : sequence of str
        Names of every parameter in ``samples``, in order -- the full vector, not the
        subset wanted.
    wanted_names : sequence of str
        Names to keep, in result order.
    axis : int, optional

    Returns
    -------
    numpy.ndarray
        Natural-scale samples holding only ``wanted_names``, in that order.
    """
    constrained = np.asarray(to_constrained(np.asarray(samples)))

    return select_params(constrained, stored_names, wanted_names, axis=axis)
