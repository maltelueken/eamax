# eamax

Evidence accumulation models in JAX: simulation, likelihoods, and parameter estimation for
the racing diffusion model and its relatives.

One implementation of the maths — the race likelihood, the accumulator densities, the
parameterization engine and the sampling layer — with the numerical guards in a single place.

## Install

```console
uv add eamax                 # core: jax + numpy
uv add 'eamax[flows]'        # + the neural spline flow (distrax, flax, orbax)
uv add 'eamax[inference]'    # + NUTS and tempered SMC (blackjax)
uv add 'eamax[io]'           # + NetCDF posterior artifacts (xarray, arviz-base, h5netcdf)
```

`tensorflow_probability` is a required runtime peer for the sampling paths and
`eamax.hierarchical`, but is deliberately **not** a declared dependency: `tfp-nightly` and
`tensorflow-probability` are separate distributions that install the same module. Install
whichever you already use, or `uv add 'eamax[tfp]'` in a fresh environment.

Call `eamax.enable_x64()` from your entry point. Race log-densities are precision-sensitive
and the reference tolerances assume float64. `eamax` never mutates JAX's global config on
import.

## The four seams

```
eamax.accumulators   first-passage distributions: Wald, LBA, pulsed variants.
                     They see decision times and return raw log-densities.
eamax.race           winner's density x losers' survival, for any N.
                     Owns t0 and every numerical guard.
eamax.design         parameterizations: flat parameter vector + trial design
                     -> per-accumulator drift, threshold, noise.
eamax.simulate       sampling, driven by the same parameterization.
```

Plus `eamax.flows` (neural density estimation) and `eamax.hierarchical` (multi-subject
priors).

```python
import eamax
eamax.enable_x64()

import jax.numpy as jnp
from eamax import race_loglik
from eamax.accumulators import Wald
from eamax.design import TrialDesign, build_params_fn, rdm_intercept_slope_spec

spec = rdm_intercept_slope_spec()             # [v_intercept, v_slope, s_true, b, t0]
params_fn = build_params_fn(spec, Wald())

design = TrialDesign(rt=rt, response=response, target=target)
params, t0 = params_fn(theta, design)         # theta is unconstrained (log space)

loglik = race_loglik(                          # (T,) per trial; you sum
    design.rt, design.response, t0,
    lambda t: Wald().log_pdf_sf(t, params),
    mask=design.mask,
)
```

The caller owns every outer `vmap`. `eamax` handles one dataset's worth of trials; map over
datasets, subjects or prior draws outside. A hierarchical likelihood is `jax.vmap` plus a
sum, which is why there isn't one in the library.

## Inference

```console
uv add 'eamax[inference]'
```

`eamax.inference` owns the sampling layer: starting values, NUTS warmup and driving,
adaptive tempered SMC, and convergence diagnostics.

```python
from eamax.inference import fit_nuts_batch, tempered_smc, T0Support, min_valid_rt

positions, infos = fit_nuts_batch(
    key, data, make_logdensity_fn, make_init_positions,
    num_chains=4, num_steps_warmup=1000, num_steps_sampling=2000,
)
```

Both factories are called *inside* the driver, per traced dataset, so starting values can be
per-dataset **and** per-chain **and** support-aware at once. Everything is in unconstrained
coordinates, so the forward transform appears exactly once.

Two defaults are opinionated. They are defaults with no override: the cheaper alternatives
are not exposed at all.

**Chains are adapted independently.** `window_adaptation` runs one window adaptation per
chain and *raises* on a position without a chain axis. Replicating one warmed state across
chains leaves R-hat with nothing to measure: between-chain variance starts at zero, so the
diagnostic cannot fail — it can report a clean R-hat even when every chain has settled into a
single mode of a multimodal target.

**Starting values are rejection-sampled into the support.** `t0` above the fastest observed
response time puts a chain on the likelihood's flat floor, where the gradient carries no
information and step-size adaptation cannot recover. Rejection rather than clipping, because a clip is a point mass — dispersion
destroyed in the one coordinate the constraint exists to protect.

`T0Support.from_spec` locates `t0` by name via `Parameterization` and raises if it is absent
or not on the log link, rather than assuming it is the last entry.

### The one supported path

```python
support = T0Support.from_spec(spec, min_valid_rt(rt))
positions, num_exhausted = init_positions_from_prior(
    prior.sample, num_chains, key, support=support,      # dispersed, and inside the support
)
last_states, tuning = window_adaptation(       # one adaptation per chain, all num_chains
    blackjax.nuts, logdensity_fn, positions, num_steps=1000, key=warmup_key,
)
# ... then sample with `tuning` exactly as it came back
```

Draw dispersed per-chain starts from the prior — `init_positions_from_prior` for NUTS,
`init_particles_from_prior` for SMC — adapt **every** chain, then sample with the tuning that
comes back, unmodified. `fit_nuts` / `fit_nuts_batch` do exactly this internally.

Three shortcuts off this path are deliberately **not offered**, because each spends a
diagnostic that says whether a fit is usable:

| Not offered | Why | Do this instead |
|---|---|---|
| sharing one warmed state across chains | Adapting one chain and replicating it makes R-hat unable to fail. | `init_positions_from_prior` + `window_adaptation` |
| sharing one SMC cloud across chains | The chains would then differ only in SMC randomness, so between-chain spread understates the uncertainty — and that spread *is* the standard error on the log marginal likelihood. | `init_particles_from_prior`, one cloud per chain |
| overwriting a collapsed step size | Replacing it with the healthy chains' median treats a symptom; the usual cause is a chain that started outside the `t0` support. | Start inside the support (`T0Support`); if a step size still collapses, report it |

The warm-up saving these buy is real — per-chain adaptation costs `num_chains`× the warm-up
work — but it is paid for out of the only numbers that say whether a fit is usable.

`eamax.io` (extra `io`) reads and writes posterior artifacts in two schemas — batched
per-dataset fits, and single hierarchical fits — over shared array helpers in
`eamax.inference.posterior`.

**`eamax.io` reads and writes, and does nothing else.** It computes no diagnostic, drops no
fit on one, and does not thin. `load_dataset_posterior` returns every draw of every chain,
back-transformed and selected by name, shaped `(chain, draw, dataset, param)`. A reader that
filtered would return an array whose *shape* depended on a threshold, and pooling on the way
out would destroy the axis R-hat is computed over — so the caller composes those steps
itself, out of parts that need no xarray:

```python
theta = load_dataset_posterior(path, to_constrained=..., param_names=[...])
is_converged = np.all(rhat(theta, chain_axis=0, sample_axis=1) < 1.01, axis=-1)
samples = thin(pool_chains(theta)[is_converged], num_target_samples, axis=1)
```

Diagnose while the chain axis is still there, mask the pooled array, then thin.

## Three design decisions worth knowing about

**`t0` and the guards live in the race, not the accumulator.** The `t0` shift, the parameter
floors and the density floor all belong to the race. Accumulators return raw log-densities,
so there is no intermediate for a caller to re-clamp — which would floor each component
separately and make the effective floor scale with the number of accumulators.

**The likelihood floor applies once, to the trial total.** Flooring each density component
separately makes the effective floor scale with the number of accumulators while inflating
deep-tail survival terms once per loser — a bias that grows with N and with tail depth, and
whose sign flatters models that push losers into the tail. Flooring the assembled total is
N-independent, and matches EMC2, which floors each trial at the same `log(1e-10)`.

**`eamax` stops at the likelihood boundary.** It owns the link function (inseparable from the
parameterization) but no prior — priors are a modelling choice left to the caller.

## One engine, every parameterization

A parameterization is a set of quantities, each a sum of terms, each term a coefficient times
a **contrast** column built from the accumulator index and the trial covariates:

```python
from eamax._tfp import tfb
from eamax.design import coef, term, quantity, parameterization, intercept, match, distractor

V, v_d = coef("V", tfb().Exp()), coef("v_d", tfb().Identity())
parameterization(
    order=[V, v_d, ...],
    quantities=[
        quantity("v", term(V, intercept()), term(v_d, match(0.5))),  # average ± difference/2
        quantity("amp", term(amp, distractor())),                    # pulse on the distractor
        ...],
    num_responses=2)
```

Each coefficient carries a TensorFlow Probability bijector as its link, so `constrain` and
`log_det_jacobian` compose over them and `eamax` owns the change of variables but no prior.
The two-accumulator `[v_intercept, v_slope, s_true, b, t0]` layout and the signed
average/difference terms over N accumulators are both just presets over this one engine
(`rdm_intercept_slope_spec`, `rdm_sat_spec`, `lba_intercept_slope_spec`, `lba_sat_spec`,
`effects_spec`), emitting stable parameter names — nothing downstream renames.

The two noise identifications are just two ways to build the `s` quantity: the "average"
convention is `S·intercept + s_d·match(0.5)`; the "mismatch" convention (what `s_true` means)
is `constant(scale)·nontarget() + s_true·target()`. Both are linear — there is no
`noise_reference` flag. And because a quantity is an open-ended sum of contrasts, richer
models — a conflict pulse routed to one accumulator via `distractor`, an arbitrary condition
contrast matrix — are ordinary specs (`pulsed_conflict_spec`).

## Validation

`tests/test_emc2_reference.py` checks the densities and the assembled race against
[EMC2](https://github.com/ampl-psych/EMC2), an independent R implementation, via committed
CSV fixtures — 1e-9 relative on pointwise density and on total dataset log-likelihood. It is
what pins the flooring decision above.

Beyond that: the Wald against `scipy.stats.invgauss`; the LBA's defective densities
integrating to 1 by quadrature; the Volterra solver reducing to the inverse Gaussian as the
pulse vanishes, at the right convergence order; the flow's survival function against a
numerical integral of its own density; and a score test — `mean grad_theta loglik(theta_true)
= 0` over simulated datasets — tying each simulator to its likelihood.

On the inference side: tempered SMC recovers a conjugate Gaussian's closed-form posterior
mean and covariance in D=2, and its log marginal likelihood lands within **0.0044 nats** of
the analytic evidence `log N(x | 0, S0 + S)` — a check on the evidence, which is often the
headline scientific output, against a known answer.

```console
uv run pytest
```

## Migration

`docs/migration/` has a guide per source repository: what moves, what stays, what breaks, and
which numbers change.
