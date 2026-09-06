# eamax

Evidence accumulation models in JAX: simulation, likelihoods, and parameter estimation.

## Install

eamax has separate dependency groups for different modules:

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

## The four core modules

Four core modules cover simulation and likelihoods of evidence accumulation models:

- `eamax.accumulators`: First-passage distributions: Wald, LBA, pulsed variants; they receive decision times and return raw log-densities
- `eamax.race`: Race likelihoods. Takes care of t0 and every numerical guard
- `eamax.design`: Parameterizations and presets (e.g., intercept-slope)
- `eamax.simulate`: Sampling, driven by parameterizations

Additional modules are `eamax.flows` (neural density estimation) and `eamax.hierarchical` (hierarchical
prior structures).

Priors are defined outside the package (except for structure of hierarchical priors).

## Example

Create a log likelihood function for the two-accumulator racing diffusion model with intercept-slope parameterization:

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
from eamax.inference import fit_nuts_batch, make_init_positions

positions, infos = fit_nuts_batch(
    key, data, make_logdensity_fn, make_init_positions,
    num_chains=4, num_steps_warmup=1000, num_steps_sampling=2000,
)
```

## Parameterizations

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
`effects_spec`). Parameter names are set here and not supposed to be changed downstream.

The two noise identifications are just two ways to build the `s` quantity: the "average"
convention is `S·intercept + s_d·match(0.5)`; the "mismatch" convention (what `s_true` means)
is `constant(scale)·nontarget() + s_true·target()`. Because a quantity is an open-ended sum of contrasts, richer
models, e.g., a conflict pulse routed to one accumulator via `distractor`, an arbitrary condition
contrast matrix, can be defined through the same interface (`pulsed_conflict_spec`).

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
the analytic evidence `log N(x | 0, S0 + S)`.

Run unit and validation tests with:

```console
uv run pytest
```
## Generative AI usage

Claude Code (Opus version 4.5 - 5.0) was used to partially generate and improve code in prfmodel.
All improvements were manually evaluated and approved by the author.

