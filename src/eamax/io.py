"""NetCDF readers and writers for posterior artifacts.

Two schemas live here, not one. They are what the consumer repositories already write, and
they have almost nothing in common:

* **Dataset posteriors** (`eam-abi-robustness`) -- one ``theta`` variable with dims
  ``(chain, draw, dataset, param)``, holding many independent fits of the *same* model to
  *different* simulated datasets. Amortized (NPE) draws go in the same layout with a single
  chain, which is the point: one reader serves both sides of the comparison.
* **Hierarchical posteriors** (`cognitive-control-comparison`) -- one variable per named
  parameter with a ``subject`` dimension, plus population ``mu``/``sigma``, plus
  ``observed_data`` and ``prior`` groups, plus the SMC log marginal likelihood in the
  attributes. A single fit of one model to one real dataset.

Trying to make one schema serve both would mean a ``dataset`` axis of length 1 next to a
``subject`` axis, or per-parameter variables for a model whose parameters are only ever
handled as a block. So this module ships two writers over the shared array helpers in
:mod:`eamax.inference.posterior`, and the helpers -- not the schema -- are where the
reusable logic lives.

**This module reads and writes. It does not analyse.** No function here computes a
diagnostic, drops a fit on one, or thins. That is a boundary worth stating because the
reader used to do all three: :func:`load_dataset_posterior` took a ``psrf_threshold`` and a
``num_target_samples``, computed R-hat, dropped the datasets that failed it, and thinned
what was left. Three things were wrong with that. The returned array's *shape*
depended on a diagnostic, so a caller could not tell an empty result from a failed read. The
threshold -- an analysis decision, and the one that defines a comparison set -- was a
file-reading argument. And pooling the chains on the way out destroyed the axis R-hat is
computed over, so the caller could not have checked the filter, or applied a different one,
even if it wanted to.

So the reader now hands back every draw of every chain, back-transformed and selected by
name, and the caller composes the rest out of parts that are already public and already
tested::

    theta = load_dataset_posterior(path, to_constrained=..., param_names=[...])
    is_converged = np.all(rhat(theta, chain_axis=0, sample_axis=1) < 1.01, axis=-1)
    pooled = pool_chains(theta)[is_converged]
    samples = thin(pooled, num_target_samples, axis=1)

The order is the part that matters: diagnose while the chain axis is still there, mask the
pooled array rather than the ``Dataset`` so total non-convergence comes back as a
length-zero axis instead of raising, then thin. Running R-hat on the back-transformed values
rather than the stored unconstrained ones changes nothing -- it is rank-normalized, so it is
invariant under any monotone reparameterization, which every link here is.

:func:`~eamax.inference.posterior.pool_chains` and :func:`~eamax.inference.posterior.thin`
live in :mod:`eamax.inference.posterior`, and :func:`~eamax.inference.diagnostics.rhat` in
:mod:`eamax.inference.diagnostics` -- none of which needs xarray, so the statistic that
decides which fits are kept stays out of the optional-dependency surface entirely rather
than merely being imported from outside it.

Everything here needs xarray and ArviZ, which nothing else in `eamax` does. Install with::

    pip install 'eamax[io]'
"""

import jax
import numpy as np

from .inference.posterior import (
    POSTERIOR_DIMS,
    back_transform_then_select,
    to_arviz_layout,
)

_MISSING = (
    "eamax.io needs xarray and ArviZ, which the rest of eamax does not.\n\n"
    "    pip install 'eamax[io]'\n\n"
    "Sampling, diagnostics and the array helpers in `eamax.inference.posterior` all work "
    "without them; only reading and writing NetCDF files needs this."
)


def _xr():
    """The `xarray` module."""
    try:
        import xarray
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return xarray


def _azb():
    """The `arviz_base` module."""
    try:
        import arviz_base
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return arviz_base


# ---------------------------------------------------------------------------
# Dataset posteriors: many fits of one model, stored as a single `theta` variable.
# ---------------------------------------------------------------------------


def save_dataset_posterior(path, positions, param_names, *, layout="sampler"):
    """Write batched posterior samples to NetCDF as an ArviZ-style DataTree.

    Naming the ``param`` axis is what makes :func:`load_dataset_posterior` able to select
    parameters by name. A hierarchical model stores prior hyperparameters ahead of the
    subject-level ones that an amortized posterior never sees, so aligning the two by
    position would silently compare different quantities.

    Parameters
    ----------
    path : str or pathlib.Path
    positions : array_like
        Samples. Shaped ``(dataset, draw, chain, param)`` under the default
        ``layout="sampler"`` -- the layout :func:`eamax.inference.mcmc.fit_nuts_batch`
        returns -- or already ``(chain, draw, dataset, param)`` under ``layout="arviz"``,
        which is how independent draws with a single chain arrive.
    param_names : sequence of str
        One name per entry of the trailing axis, in order.
    layout : {"sampler", "arviz"}, optional

    Raises
    ------
    ValueError
        If ``layout`` is unrecognised, ``positions`` is not rank 4, or the number of names
        does not match the trailing axis.
    """
    if layout not in {"sampler", "arviz"}:
        msg = f"layout must be 'sampler' or 'arviz', got {layout!r}."
        raise ValueError(msg)

    samples = to_arviz_layout(positions) if layout == "sampler" else np.asarray(positions)

    if samples.ndim != len(POSTERIOR_DIMS):
        msg = f"Expected samples with dims {POSTERIOR_DIMS}, got an array of rank {samples.ndim}."
        raise ValueError(msg)

    param_names = [str(name) for name in param_names]
    if samples.shape[-1] != len(param_names):
        msg = f"Posterior has {samples.shape[-1]} parameters but {len(param_names)} names were given."
        raise ValueError(msg)

    tree = _azb().from_dict(
        {"posterior": {"theta": samples}},
        dims={"theta": ["dataset", "param"]},
        coords={"dataset": np.arange(samples.shape[2]), "param": param_names},
    )

    tree.to_netcdf(path)


def open_dataset_posterior(path):
    """Open the ``posterior`` group written by :func:`save_dataset_posterior`.

    Returns an :class:`xarray.Dataset` rather than arrays, so that ``theta`` keeps its
    ``param`` coordinate. Use :func:`load_dataset_posterior` for the full read-back.
    """
    with _xr().open_datatree(path) as tree:
        return tree["posterior"].to_dataset().load()


def load_dataset_posterior(path, *, to_constrained, param_names):
    """Read samples back onto the natural scale, keeping every chain and every draw.

    Two steps, in an order that does not commute, and nothing else. Convergence filtering,
    pooling and thinning are the caller's -- see this module's docstring for why, and for
    the four-line recipe that reproduces what this function used to do.

    **Back-transform before selecting.** The transform reads its own block boundary off the
    full parameter vector; handing it a subset shifts that boundary and silently applies the
    wrong link to the wrong entries. See
    :func:`eamax.inference.posterior.back_transform_then_select`.

    **The chain axis survives.** It is the axis :func:`eamax.inference.diagnostics.rhat`
    is computed over, so pooling here would decide, on the caller's behalf and out of its
    sight, that convergence can no longer be checked. Pool with
    :func:`eamax.inference.posterior.pool_chains` once the diagnostics are done.

    Parameters
    ----------
    path : str or pathlib.Path
        A file written by :func:`save_dataset_posterior`.
    to_constrained : callable
        Unconstrained -> natural scale, applied to the full parameter vector. The inverse
        of whatever transform the fit used -- e.g.
        :meth:`eamax.inference.transforms.BlockTransform.forward` or
        :meth:`eamax.design.spec.ParamSpec.constrain`.
    param_names : sequence of str
        Parameters to keep, by name and in result order.

    Returns
    -------
    numpy.ndarray
        Shape ``(chain, draw, dataset, len(param_names))``, natural scale. Every stored
        dataset is present, in storage order, whatever its R-hat.

    Raises
    ------
    ValueError
        If any requested name is not stored.
    """
    posterior = open_dataset_posterior(path)

    theta = posterior["theta"].transpose(*POSTERIOR_DIMS)
    stored_names = [str(name) for name in posterior.coords["param"].to_numpy()]

    return back_transform_then_select(
        theta.to_numpy(), to_constrained, stored_names, param_names
    )


# ---------------------------------------------------------------------------
# Hierarchical posteriors: one SMC fit, per-parameter variables with a subject axis.
# ---------------------------------------------------------------------------


def _report_scale(values, exp_mask):
    """Exponentiate the log-linked entries of a trailing-axis parameter block."""
    return np.where(np.asarray(exp_mask), np.exp(np.asarray(values)), np.asarray(values))


def _reconstruct_particles(flat_space, particles):
    """Map particles to ``(mu, s, subject_params)``, vmapping over every leading axis."""
    particles = np.asarray(particles)

    def one(flat):
        mu, s = flat_space.population(flat)
        return mu, s, flat_space.subject_params(flat)

    reconstruct = one
    for _ in range(particles.ndim - 1):
        reconstruct = jax.vmap(reconstruct)

    mu, s, subject = reconstruct(particles)
    return np.asarray(mu), np.asarray(s), np.asarray(subject)


def save_hierarchical_posterior(
    path,
    result,
    flat_space,
    spec,
    subjects,
    *,
    observed=None,
    prior_particles=None,
    attrs=None,
):
    """Write one hierarchical SMC fit to NetCDF as an ArviZ-style DataTree.

    Particles are reconstructed onto interpretable coordinates on the way out --
    population ``mu`` and ``sigma`` plus one variable per named parameter carrying a
    ``subject`` dimension -- using :class:`eamax.hierarchical.HierarchicalFlatSpace`. That
    reconstruction is the same semi-centered formula the sampler uses, reached through the
    one object rather than written out again here.

    ``mu`` and the per-subject parameters are reported on the natural scale: log-linked
    entries are exponentiated, identity-linked ones are left signed. ``sigma`` is **not**
    transformed, and deliberately so -- it is a standard deviation *on the log scale*, and
    exponentiating it would produce a number that is not a standard deviation of anything.

    The SMC log marginal likelihood is stored per chain in the attributes, along with the
    across-chain mean. Treat the spread across chains as the honest uncertainty: it is only
    meaningful when the chains were given genuinely independent particle clouds and tuning
    -- which is what :func:`eamax.inference.init.init_particles_from_prior` and one
    :func:`eamax.inference.warmup.window_adaptation` per chain produce, and the reason
    `eamax` offers no way to share either.

    Parameters
    ----------
    path : str or pathlib.Path
    result : eamax.inference.smc.SMCResult
        Every field carries a leading chain axis.
    flat_space : eamax.hierarchical.HierarchicalFlatSpace
        The coordinate system the particles live in.
    spec : eamax.design.spec.ParamSpec
        Supplies ``names`` and ``exp_mask``; its parameter order must match the prior's.
    subjects : sequence
        Subject labels, one per row of the reconstructed ``(S, P)`` block. Used as the
        ``subject`` coordinate, so the stored posterior can be joined back to the source
        data by identifier rather than by row number.
    observed : mapping of str to array_like, optional
        Observed quantities with dims ``(subject, trial)`` -- e.g. ``rt``, ``response``,
        and a padding ``mask``. Written to the ``observed_data`` group as given.
    prior_particles : array_like, optional
        Initial particles, shape ``(num_particles, D)`` or ``(num_chains, num_particles, D)``.
        Reconstructed onto the same scale as the posterior and written to the ``prior``
        group, which is what makes prior-to-posterior contraction measurable from the file
        alone.
    attrs : mapping, optional
        Extra scalar metadata merged into the DataTree's attributes.

    Returns
    -------
    xarray.DataTree
        The tree that was written, so a caller can inspect or extend it without reopening.

    Raises
    ------
    ValueError
        If ``spec`` and the reconstructed parameter block disagree in width, or if
        ``subjects`` does not match the number of subjects.
    """
    xr = _xr()
    azb = _azb()

    param_names = [str(name) for name in spec.names]
    exp_mask = np.asarray(spec.exp_mask)
    subject_coord = np.asarray(subjects)

    pop_mu, pop_s, pop_theta = _reconstruct_particles(flat_space, result.particles)

    num_subjects, num_params = pop_theta.shape[-2:]
    if num_params != len(param_names):
        msg = (
            f"The prior reconstructs {num_params} parameters but the spec names "
            f"{len(param_names)}: {param_names}."
        )
        raise ValueError(msg)
    if subject_coord.shape[0] != num_subjects:
        msg = f"{subject_coord.shape[0]} subject labels for {num_subjects} subjects."
        raise ValueError(msg)

    reported = _report_scale(pop_theta, exp_mask)

    posterior_ds = azb.dict_to_dataset(
        {
            "mu": _report_scale(pop_mu, exp_mask),
            "sigma": pop_s,
            **{name: reported[..., i] for i, name in enumerate(param_names)},
        },
        sample_dims=["chain", "draw"],
        coords={"subject": subject_coord, "param": param_names},
        dims={
            "mu": ["param"],
            "sigma": ["param"],
            **{name: ["subject"] for name in param_names},
        },
    )

    groups = {
        "posterior": posterior_ds,
        "sample_stats": azb.dict_to_dataset(
            {"weight": np.asarray(result.weights)}, sample_dims=["chain", "draw"]
        ),
    }

    if observed is not None:
        groups["observed_data"] = xr.Dataset(
            {
                name: (["subject", "trial"], np.asarray(values))
                for name, values in observed.items()
            },
            coords={"subject": subject_coord},
        )

    if prior_particles is not None:
        prior_mu, prior_s, prior_theta = _reconstruct_particles(flat_space, prior_particles)
        prior_reported = _report_scale(prior_theta, exp_mask)
        leading = ["chain", "draw"] if prior_mu.ndim == 3 else ["draw"]
        groups["prior"] = xr.Dataset(
            {
                "mu": ([*leading, "param"], _report_scale(prior_mu, exp_mask)),
                "sigma": ([*leading, "param"], prior_s),
                **{
                    name: ([*leading, "subject"], prior_reported[..., i])
                    for i, name in enumerate(param_names)
                },
            },
            coords={"subject": subject_coord, "param": param_names},
        )

    tree = xr.DataTree.from_dict(groups)

    lml = np.asarray(result.log_marginal_likelihood)
    tree.attrs["num_subjects"] = num_subjects
    tree.attrs["param_names"] = param_names
    tree.attrs["smc_num_iterations"] = np.asarray(result.num_iterations).tolist()
    tree.attrs["smc_num_unique"] = np.asarray(result.num_unique).tolist()
    tree.attrs["smc_num_unique_pre_resample"] = np.asarray(result.num_unique_smc).tolist()
    tree.attrs["smc_weight_ess"] = np.asarray(result.weight_ess).tolist()
    tree.attrs["log_marginal_likelihood"] = lml.tolist()
    tree.attrs["log_marginal_likelihood_mean"] = float(lml.mean())
    tree.attrs["log_marginal_likelihood_sd"] = float(lml.std(ddof=1)) if lml.size > 1 else float("nan")

    if attrs:
        tree.attrs.update(attrs)

    tree.to_netcdf(path)

    return tree
