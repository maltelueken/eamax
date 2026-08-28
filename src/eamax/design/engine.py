"""From a flat parameter vector plus a trial design to per-accumulator quantities.

This is the layer that turns `theta` into the `(N, T)` drifts, thresholds and noises an
accumulator wants, and it is the layer the three source repositories each wrote their own
version of. It now does so generically: each quantity is assembled by summing its terms, and
each term is a coefficient times a product of contrast columns (see
:mod:`eamax.design.contrasts`). There is no hard-coded `average + sgn * difference / 2` and
no fixed list of covariates -- adding a parameter or a covariate is adding a quantity or a
contrast, not editing this file.

Everything here is elementwise and branch-free on traced values: contrasts select with
`jnp.where`, never with a Python `if`, so shapes stay static under `jit` and `vmap`. The
parameterization's structure is fixed in Python before tracing; only `theta` is traced.

The same `params_fn` feeds the likelihood and the simulator. That is the point -- in the
source repos, simulation and scoring shared a parameterization only by convention and a
docstring, and nothing tested it.
"""

import jax.numpy as jnp

from .parameterization import Free


def _assemble_quantity(quantity, spec, p, accum, design):
    """Sum a quantity's terms into one value broadcastable to `(N, T)`."""
    total = jnp.zeros(())
    for t in quantity.terms:
        column = jnp.ones(())
        for contrast in t.contrasts:
            column = column * contrast.fn(accum, design)
        if isinstance(t.coefficient, Free):
            coefficient = p[spec.index(t.coefficient.name)]
        else:
            coefficient = t.coefficient.value
        total = total + coefficient * column
    return total


def accumulator_params(spec, theta, design):
    """Per-accumulator drift, noise and threshold (and any extra quantities), each `(N, T)`.

    Returns
    -------
    params : dict of array
        Keyed by quantity name, each ``(N, T)``. A quantity with a ``transform`` is emitted as
        that function of the other quantities -- e.g. the LBA threshold ``A + b`` from a
        ``b``-gap.
    t0 : array
        Non-decision time, scalar per trial (or a broadcastable shape). Zero if the spec has
        no ``t0`` quantity.
    """
    p = spec.constrain(theta)
    n = spec.num_responses
    num_trials = jnp.shape(design.rt)[-1]
    accum = (spec.first_response + jnp.arange(n))[:, None]  # (N, 1)
    ref = jnp.ones((n, num_trials))

    # Assemble every quantity's raw sum first, so a `transform` can read the others' values.
    built = {q.name: _assemble_quantity(q, spec, p, accum, design) for q in spec.quantities}
    final = {
        q.name: q.transform(built) if q.transform is not None else built[q.name]
        for q in spec.quantities
    }

    params = {name: value * ref for name, value in final.items() if name != "t0"}
    t0 = final["t0"] if "t0" in final else jnp.asarray(0.0)
    return params, t0


def build_params_fn(spec, accumulator):
    """Bind a spec to an accumulator, returning `params_fn(theta, design) -> (params, t0)`.

    The returned function is the single seam both `eamax.race` and `eamax.simulate` go
    through, which is what makes simulation and scoring structurally consistent rather than
    consistent by convention.

    The spec's coverage of the accumulator is checked once, here, so a missing or misspelled
    quantity surfaces at setup rather than as a `KeyError` inside a traced function.
    """
    produced = {q.name for q in spec.quantities if q.name != "t0"}
    missing = [name for name in accumulator.param_names if name not in produced]
    if missing:
        raise KeyError(
            f"{type(accumulator).__name__} needs {list(accumulator.param_names)}, but this "
            f"spec produces {sorted(produced)}; missing {missing}."
        )

    def params_fn(theta, design):
        params, t0 = accumulator_params(spec, theta, design)
        return {name: params[name] for name in accumulator.param_names}, t0

    return params_fn
