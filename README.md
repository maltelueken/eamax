# eamax

Evidence accumulation models in JAX: simulation, likelihoods, and parameter estimation for
the racing diffusion model and its relatives.

Extracted from three research repositories — [`eam-abi-robustness`], [`racing-diffusion-conflict`]
and [`cognitive-control-comparison`] — that had each grown their own copy of the same maths.
The race likelihood alone was written out six times, with the copies quietly drifting apart
in their numerical guards.

## Install

```console
uv add eamax                 # core: jax + numpy
uv add 'eamax[flows]'        # + the neural spline flow (distrax, flax, orbax)
uv add 'eamax[inference]'    # + NUTS and tempered SMC (blackjax)
uv add 'eamax[io]'           # + NetCDF posterior artifacts (xarray, arviz-base, h5netcdf)
```

`tensorflow_probability` is a required runtime peer for the sampling paths and
`eamax.hierarchical`, but is deliberately **not** a declared dependency: `tfp-nightly` and
`tensorflow-probability` are separate distributions that install the same module, and every
consumer here is on the nightly. Install whichever you already use, or `uv add 'eamax[tfp]'`
in a fresh environment.

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
from eamax.design import TrialDesign, build_params_fn, intercept_slope_spec

spec = intercept_slope_spec()                 # [v_intercept, v_slope, s_true, b, t0]
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

Two defaults are opinionated, because the three source implementations disagreed and one
side of each disagreement was measurably wrong. They are defaults with no override: the
cheaper alternatives are not exposed at all.

**Chains are adapted independently.** `window_adaptation` runs one window adaptation per
chain and *raises* on a position without a chain axis. Replicating one warmed state across
chains — which two of the three repositories do — leaves R-hat with nothing to measure:
between-chain variance starts at zero, so the diagnostic cannot fail. On a bimodal target at
±6, the shared start reports max R-hat 1.001 while putting 100% of its mass in one mode; a
dispersed start reports 1.734 and splits 50/50.

**Starting values are rejection-sampled into the support.** `t0` above the fastest observed
response time puts a chain on the likelihood's penalty wall, where step-size adaptation
cannot recover. Rejection rather than clipping, because a clip is a point mass: at a 0.9
cap, 55% of draws were capped and the worst subject had 93% of its particles pinned to one
value — dispersion destroyed in the one coordinate the constraint exists to protect.

`T0Support.from_spec` locates `t0` by name via `ParamSpec` and raises if it is absent or not
on the log link, rather than assuming it is the last entry.

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

Three escape hatches that earlier drafts carried are **removed**, not deprecated, because a
consumer repository should not be able to reach them:

| Removed | Why | Do this instead |
|---|---|---|
| `broadcast_warmup`, `fit_nuts(..., shared_warmup=True)` | Adapting one chain and replicating it makes R-hat unable to fail — see the ±6 measurement above. | `init_positions_from_prior` + `window_adaptation` |
| `broadcast_particles` | One shared SMC cloud leaves chains differing only in SMC randomness, so between-chain spread understates the uncertainty — and that spread *is* the standard error on the log marginal likelihood. | `init_particles_from_prior`, one cloud per chain |
| `repair_degenerate_tuning` | Replacing a collapsed step size with the healthy chains' median treats a symptom. In every population that produced one, the degenerate chain was the chain with the most subjects starting outside the `t0` support; repaired chains stayed under-dispersed at 0.16–0.83× their siblings' spread and still broke R-hat (1.94 / 2.60 / 2.76). | Start inside the support (`T0Support`); if a step size still collapses, report it |

The warm-up saving these bought was real — per-chain adaptation costs `num_chains`× the
warm-up work — but it was paid for out of the only numbers that say whether a fit is usable.

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

**`t0` and the guards live in the race, not the accumulator.** In the source repositories the
per-accumulator function owned the `t0` shift, the parameter floors, the density floor *and*
the invalid-RT penalty, and warned callers in prose not to clamp its output again. One
consumer clamped it anyway, flattening the penalty to a constant and removing the gradient
that pushes `t0` back into the valid region. `eamax` exposes no intermediate to re-clamp.

**The likelihood floor applies once, to the trial total.** Flooring each density component
separately makes the effective floor scale with the number of accumulators while inflating
deep-tail survival terms once per loser — a bias that grows with N and with tail depth, and
whose sign flatters models that push losers into the tail. Flooring the assembled total is
N-independent, and matches EMC2, which floors each trial at the same `log(1e-10)`.

**`eamax` stops at the likelihood boundary.** It owns the link function (inseparable from the
parameterization) but no prior. Priors are where the three consumers genuinely disagree
scientifically; there is no shared implementation underneath to extract.

## Both parameter naming conventions

The two-accumulator repositories name their RDM parameters
`[v_intercept, v_slope, s_true, b, t0]`; the conflict-task work uses signed average and
difference terms (`V ± v_d/2`, `B ± b_d/2`, `S ± s_d/2`) over N accumulators. These are the
same map for drift and threshold — `V = v_intercept + v_slope/2`, `v_d = v_slope` — so both
are `ParamSpec`s over one engine and nothing downstream renames.

They are **not** the same for the noise, and that is a modelling difference rather than a
naming one: one pins the mismatching accumulator's noise to 1, the other pins the *average*
to 1 and leaves the mismatching accumulator at `1 - s_d/2`. `ParamSpec.noise_reference`
makes the choice explicit.

## Validation

`tests/test_emc2_reference.py` checks the densities and the assembled race against
[EMC2](https://github.com/ampl-psych/EMC2), an independent R implementation, via committed
CSV fixtures — 1e-9 relative on pointwise density and on total dataset log-likelihood. It is
the only cross-implementation check any of the source repositories had, and it is what pins
the flooring decision above.

Beyond that: the Wald against `scipy.stats.invgauss`; the LBA's defective densities
integrating to 1 by quadrature; the Volterra solver reducing to the inverse Gaussian as the
pulse vanishes, at the right convergence order; the flow's survival function against a
numerical integral of its own density; and a score test — `mean grad_theta loglik(theta_true)
= 0` over simulated datasets — tying each simulator to its likelihood.

On the inference side, where none of the source repositories had an outside reference:
tempered SMC recovers a conjugate Gaussian's closed-form posterior mean and covariance in
D=2, and its log marginal likelihood lands within **0.0044 nats** of the analytic evidence
`log N(x | 0, S0 + S)`. That last one is the check that matters most — the evidence is one
consumer's headline scientific output and had never been compared against a known answer.

```console
uv run pytest
```

## Migration

`docs/migration/` has a guide per repository: what moves, what stays, what breaks, and which
numbers change.

[`eam-abi-robustness`]: ../eam-abi-robustness
[`racing-diffusion-conflict`]: ../racing-diffusion-conflict
[`cognitive-control-comparison`]: ../cognitive-control-comparison
