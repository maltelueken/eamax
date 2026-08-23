"""From a flat parameter vector plus a trial design to per-accumulator quantities.

This is the layer that turns `theta` into the `(N, T)` drifts, thresholds and noises an
accumulator wants, and it is the layer the three source repositories each wrote their own
version of. Everything here is elementwise and branch-free on traced values: conditions
select with `jnp.where`, never with a Python `if`, so shapes stay static under jit and vmap.

The same function feeds the likelihood and the simulator. That is the point -- in the
source repos, simulation and scoring shared a parameterization only by convention and a
docstring, and nothing tested it.
"""

import jax.numpy as jnp


def _evaluate_terms(p, spec, terms):
    """Sum a tuple of `(parameter_name, coefficient)` pairs against the constrained vector."""
    total = jnp.zeros(())
    for name, coefficient in terms:
        total = total + coefficient * p[spec.index(name)]
    return total


def select_by_condition(p, spec, q, condition):
    """Per-trial value of quantity `q`, routed by condition.

    `condition == 1` is the reference level (congruent, or speed-instructed). Quantities
    with no condition effect return one value for every trial.

    Resolution order: a `derived` linear combination, then a `fixed` constant, then a direct
    index. A quantity mentioned in none of them is zero -- so a model without, say, a
    threshold difference simply omits `b_d` rather than declaring it and pinning it to zero.
    """
    if q in spec.derived:
        terms_con, terms_inc = spec.derived[q]
        value_con = _evaluate_terms(p, spec, terms_con)
        if terms_con == terms_inc:
            return value_con
        return jnp.where(condition == 1, value_con, _evaluate_terms(p, spec, terms_inc))

    if q in spec.fixed:
        return jnp.asarray(spec.fixed[q])

    if q not in spec.quantities or spec.quantities[q] is None:
        return jnp.zeros(())

    i_con, i_inc = spec.quantities[q]
    if i_con == i_inc:
        return p[i_con]

    if spec.coding.get(q) == "sum":
        # `p[i_inc]` is a non-negative increment, so the ordering holds by construction.
        return jnp.where(condition == 1, p[i_con], p[i_con] + p[i_inc])
    return jnp.where(condition == 1, p[i_con], p[i_inc])


def response_offsets(p, spec):
    """Effect-coded (sum-to-zero) per-accumulator threshold offsets, shape `(N,)`.

    The last accumulator's offset is derived rather than estimated, so the per-accumulator
    baseline thresholds average to the shared threshold exactly.
    """
    if not spec.response_offsets:
        return jnp.zeros((spec.num_responses,))
    free = p[jnp.asarray(spec.response_offsets)]
    return jnp.concatenate([free, -jnp.sum(free)[None]])


def accumulator_noise(sgn, average, difference, spec):
    """Per-accumulator within-trial noise, under whichever identification the spec uses.

    Only drift-to-noise ratios are identified, so one noise has to be pinned. The two
    source conventions pin *different* ones, and they are not reparameterizations of each
    other -- this is a modelling choice, not a naming choice:

    * `noise_reference="average"`: the average noise is the reference (fixed to 1 in the
      congruent condition), and the target-match difference `s_d` is free. The mismatching
      accumulator's noise is then `1 - s_d/2`, which is *not* 1.
    * `noise_reference="mismatch"`: the mismatching accumulator's noise is the reference,
      held at `spec.noise_scale`, and the matching accumulator's noise is free. This is the
      convention in which a parameter named `s_true` means what it says.
    """
    if spec.noise_reference == "average":
        return average + sgn * difference / 2.0
    return jnp.where(sgn > 0, average, spec.noise_scale)


def trial_quantities(p, spec, design):
    """Per-trial quantities shared across accumulators.

    Returns a dict of `V, v_d, B, b_d, S, s_d, t0, c` plus any extra shared quantities the
    spec carries (`amp`, `tau`, `A`), each with trial shape except `c`, which is `(N,)`.
    """
    condition = design.condition
    out = {
        q: select_by_condition(p, spec, q, condition)
        for q in ("V", "v_d", "B", "b_d", "S", "s_d")
    }
    out["c"] = response_offsets(p, spec)
    out["t0"] = p[spec.index("t0")] if "t0" in spec.names else jnp.asarray(0.0)
    for extra in ("amp", "tau", "A"):
        if extra in spec.quantities or extra in spec.fixed or extra in spec.derived:
            out[extra] = select_by_condition(p, spec, extra, condition)
    return out


def accumulator_params(spec, quantities, design):
    """Per-accumulator drift, noise and threshold, each `(N, T)`.

    The design enters through one covariate: whether accumulator `k` is the trial's target.
    Everything else is an average plus a signed difference around it.
    """
    n = spec.num_responses
    accumulators = jnp.arange(1, n + 1)[:, None]
    target = jnp.asarray(design.target)[None, :]
    sgn = jnp.where(target == accumulators, 1.0, -1.0)  # (N, T)

    q = quantities
    v = q["V"] + sgn * q["v_d"] / 2.0
    s = accumulator_noise(sgn, q["S"], q["s_d"], spec)
    threshold = q["B"] + q["c"][:, None] + sgn * q["b_d"] / 2.0

    params = {"v": v * jnp.ones_like(sgn), "s": s * jnp.ones_like(sgn)}
    if spec.threshold_is_gap:
        # LBA convention: the estimated quantity is the gap above the start point, so the
        # absolute boundary is `A + gap` and `b > A` holds by construction.
        params["A"] = q["A"] * jnp.ones_like(sgn)
        params["b"] = params["A"] + threshold
    else:
        params["b"] = threshold * jnp.ones_like(sgn)

    for extra in ("amp", "tau"):
        if extra in q:
            params[extra] = q[extra] * jnp.ones_like(sgn)
    return params


def build_params_fn(spec, accumulator):
    """Bind a spec to an accumulator, returning `params_fn(theta, design) -> (params, t0)`.

    The returned function is the single seam both `eamax.race` and `eamax.simulate` go
    through, which is what makes simulation and scoring structurally consistent rather than
    consistent by convention.

    The spec's coverage of the accumulator is checked once, here, so a missing or misspelled
    quantity surfaces at setup rather than as a `KeyError` inside a traced function.
    """
    probe = {name: None for name in _quantities_produced(spec)}
    missing = [name for name in accumulator.param_names if name not in probe]
    if missing:
        raise KeyError(
            f"{type(accumulator).__name__} needs {list(accumulator.param_names)}, but this "
            f"spec produces {sorted(probe)}; missing {missing}."
        )

    def params_fn(theta, design):
        p = spec.constrain(theta)
        quantities = trial_quantities(p, spec, design)
        return accumulator_params(spec, quantities, design), quantities["t0"]

    return params_fn


def _quantities_produced(spec):
    produced = {"v", "s", "b"}
    if spec.threshold_is_gap:
        produced.add("A")
    produced |= {
        extra
        for extra in ("amp", "tau")
        if extra in spec.quantities or extra in spec.derived or extra in spec.fixed
    }
    return produced
