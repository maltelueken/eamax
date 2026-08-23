# `cognitive-control-comparison` → `eamax`

This is the migration that pays for the whole exercise, and the one with a bug to fix first.

## Fix the penalty bug first, in place, as its own commit

`confrdm_jax.likelihoods.rdm.inv_gauss_log_pdf_sf` returns, for `rt <= t0`, a penalty

```
_LOG_FLOOR + 1e3 * min(rt_shifted - _FLOOR, 0)
```

which is **by construction `<= _LOG_FLOOR`**. Its docstring says, in bold:

> Returns `(log_pdf, log_sf)`, already clamped and penalised. **Do not clamp the result
> again** — that would flatten the `rt <= t0` penalty back to a constant and remove the
> gradient that pushes `t0` into the valid region.

This repository clamps it again, at three sites:

* `scripts/fit_hierarchical_rdm.py:191` — `return _clamp_log(log_pdf), _clamp_log(log_sf)`
* `scripts/fit_hierarchical_crdm.py:288` — the same, in `_per_accumulator_analytic`
* `scripts/fit_hierarchical_crdm.py:305-306` — `nn_log_pdf = _clamp_log(...)`, `nn_log_sf = ...`

`_clamp_log` is `maximum(nan_to_num(x), _LOG_FLOOR)`, so every penalty is clamped back to
exactly `_LOG_FLOOR`. **Every SMC fit in `results/` has run with a flat, zero-gradient `t0`
barrier.** Nothing pushed `t0` back when a particle proposed it above the fastest observed
response time; the sampler saw a plateau instead of a slope.

The fix is to drop the three `_clamp_log` wrappers. Do it before the migration, on its own,
so the fixture diff separates this effect from the floor-semantics change. Expect real
movement in the `t0` posterior and possibly in SMC behaviour near the boundary — for the
better, but it is not a no-op.

`eamax` makes the bug structurally impossible rather than forbidden by a docstring:
accumulators return raw densities and the race applies every guard once, at the end, with
no intermediate exposed. `tests/test_race.py::test_penalty_slopes_when_rt_precedes_t0` is
the regression.

## Then: this repository stops depending on `racing-diffusion-conflict` entirely

Its whole cross-repository surface is

* `confrdm_jax.flows` — `evaluate_pdf_sf`, `load_conditioner`, `make_mlp_conditioner`, `spline_flow`
* `confrdm_jax.simulators.base.HierarchicalRDMPriorLKJMVN`
* `confrdm_jax.likelihoods.rdm` — `inv_gauss_log_pdf_sf`, `_clamp_log`, `_penalize_invalid_rt`
* `confrdm_jax.mcmc.warmup` — reached through the `.pth` file for this one function

**Every one of those is now in `eamax`.** So:

1. Add a `pyproject.toml` depending on `eamax[flows]`.
2. Promote `scripts/hierarchical_fit/` to `src/ccc/` — a real package, importable without
   `sys.path` adjacency.
3. Delete `venv/lib/python3.12/site-packages/__editable__.confrdm-0.0.1.pth`.

That `.pth` file is currently the *only* record that this repository cannot run without a
sibling checkout on disk. It is not in `requirements.txt`, the README, or any config.

## What moves out of `scripts/hierarchical_fit/`

| From | To |
|---|---|
| `param_spec.py` — `ParamSpecBuilder` | `eamax.design.ParamSpecBuilder` |
| `param_spec.py` — `select_by_condition`, `reconstruct_response_offsets`, `condition_effects`, `accumulator_params` | `eamax.design.map` |
| `effects.py` | `eamax.design.effects` (verbatim) |
| `prior.py` — `build_bijector`, the reconstruction block (×2) | `eamax.hierarchical` |
| `simulate.py` — `sample_accumulator_rt`, the N-accumulator race | `eamax.simulate` |
| `fit_hierarchical_rdm.py` — `create_rdm_effect_likelihood` | `eamax.race.race_loglik` |
| `fit_hierarchical_crdm.py` — `create_crdm_effect_likelihood` | `eamax.race` + `gather_by_mask` / `overlay_by_mask` |
| `prior.py` — `init_from_prior_mode`, `sample_prior_particles` | `eamax.inference.init.{init_position_from_mode,init_particles_from_prior}` |
| `prior.py` — `reconstruct_particle` | `HierarchicalFlatSpace.{subject_params,population}` |
| `smc_fit.py` — `smc_inference_loop_lml`, `fit` | `eamax.inference.smc.tempered_smc` |
| `smc_fit.py` — the hand-written log-Jacobian | `eamax.hierarchical.joint_log_det_jacobian` |
| `output.py` — `build_datatree` | `eamax.io.save_hierarchical_posterior` |

## What stays

`PARAM_PRIORS` (your scientific content), `build_param_spec` (the *ordering* is
model-specific and must keep B-type parameters and `t0` in the trailing centered block),
`data.py`, `cli.py`, `preprocess_data.py`.

`smc_fit.py` and `output.py` no longer stay: the driver becomes a call to
`eamax.inference.smc.tempered_smc`, and `build_datatree` becomes
`eamax.io.save_hierarchical_posterior`. What is left of each is argument assembly — which
model, which effects, which CLI flags — and belongs in the fitting scripts.

## Things to get right

* **Response coding.** Your responses are 1-indexed; the two-accumulator repositories use
  0/1. `eamax` takes `first_response=1` — `TrialDesign` defaults to it, but
  `race_loglik` defaults to `0`, so pass it explicitly if you call the race directly.
* **The hierarchical likelihood disappears.** There is no `create_*_hierarchical_likelihood`
  in `eamax`, because once the race handles one subject's trials with a mask, the
  multi-subject case is `jax.vmap` plus a sum. See
  `tests/test_hierarchical.py::test_a_hierarchical_likelihood_is_just_the_race_vmapped_over_subjects`.
* **The reconstruction block.** `eamax.hierarchical.reconstruct_semicentered` replaces all
  copies — `prior.py` (×2), `smc_fit.py`, `parameter_recovery_hierarchical.py`, both
  hierarchical simulators in the sibling repository, and the private copy in its test suite.
  They all have to agree; now they are one function.
* **The noise identification is a modelling decision you have been making implicitly.** Your
  layout fixes the *average* noise `S` to 1 (congruent as reference), which leaves the
  mismatching accumulator at `1 - s_d/2`. The two-accumulator repositories fix the
  *mismatching* accumulator's noise to 1 instead. These are different constraints, not
  different names, and `eamax` makes the choice explicit via `ParamSpec.noise_reference`
  (`"average"` reproduces yours). Worth a deliberate decision rather than an inherited one —
  see `tests/test_design_spec.py::test_the_two_conventions_pin_a_different_noise`.

## Numbers change twice from the likelihood

Once from the penalty fix, once from the floor semantics — and then again from inference, on
a separate axis; see "Inference" below. The floor change is the one that
matters most here: your output is model comparison by SMC log marginal likelihood, and the
per-component floor you inherit is *N-dependent* — a four-response race effectively floors
at `4 * log(1e-12)` while inflating each loser's deep-tail survival. That systematically
flatters models that push losers into the tail, which is exactly the axis you are comparing
along. Re-run the committed `rdm_BSs`, `rdm_BS`, `rdm_Bs` and `crdm_BSs` fits and compare
Bayes factors before and after.

## Inference: three distinct changes, plus a version bump

This is where the rest of the payoff is. Do it in the two commits `README.md` describes —
the shim commit reproducing today's behaviour exactly, then one behaviour commit per change
below if you want them separable, which here you do.

**Upgrade BlackJAX first.** You pin 1.2.5. `eamax.inference.diagnostics.rhat` needs ≥1.6 and
raises a clear upgrade error below it rather than falling back to the older plain
Gelman–Rubin statistic. Everything else you use — `extend_params`,
`adaptive_tempered.as_top_level_api`, `smc.resampling.systematic`,
`SMCInfo.log_likelihood_increment` — is identical across 1.2.5 and 1.6.2. One thing is not:
`TemperedSMCState.lmbda` is `tempering_param` on the newer version. `eamax` reads whichever
exists, but anything in your code touching that field directly must be updated.

### Shim: as close to today as `eamax` will go

```python
result = tempered_smc(
    key, log_prior_fn, log_likelihood_fn,
    initial_particles,      # (num_chains, num_particles, D) -- one cloud per chain, required
    mcmc_parameters,        # leading chain axis -- one window_adaptation per chain, required
    resample_final=False,   # the one compatibility argument that still exists
    num_integration_steps=100, target_ess=0.5, num_mcmc_steps=1, max_steps=200,
)
```

Your five hard-coded constants are keyword arguments with your current values as defaults, so
the CLI can expose them without touching the driver. `resample_final=False` reproduces today.

**The shared cloud and shared tuning do not have a compatibility argument.** `eamax` used to
ship `broadcast_particles` for exactly this shim and it has been **removed**, together with
`broadcast_warmup` and `repair_degenerate_tuning`. `initial_particles` must be
`(num_chains, num_particles, D)` with genuinely independent clouds — draw them with
`init_particles_from_prior` — and `mcmc_parameters` must carry a leading chain axis from one
`window_adaptation` per chain. Passing a 2-D cloud raises rather than broadcasting.

For this repository that means Change 1 below is not optional and does not wait for a
behaviour commit; it lands with the shim. Changes 2 and 3 remain separable.

### Change 1: per-chain warmup and per-chain particle clouds

Today every SMC chain gets one identical initial cloud (`smc_fit.py:158-166`) and one shared
mutation-kernel tuning. The four `log_marginal_likelihood` values you record per chain are
therefore **far more correlated than they look**: they share a starting cloud and a step
size, so the spread across them is not the sampling variability it is being read as.

The point estimate may move a little. The **uncertainty on it will move a lot**, and in the
direction of being larger. `save_hierarchical_posterior` now records
`log_marginal_likelihood_sd` alongside the mean for exactly this reason. Any published Bayes
factor should be **recomputed, not reused** — this is a change to the error bars on your
headline result.

Give each chain its own cloud from `init_particles_from_prior`, and give the mutation kernel
per-chain tuning by passing `mcmc_parameters` whose leaves carry a leading chain axis, which
is what `window_adaptation` returns. There is no argument that keeps the old behaviour — this
is the change that forces itself into the shim commit.

Do not add a repair step on top. `eamax` has no `repair_degenerate_tuning`: if one chain's
step size collapses, the cause is almost always a start outside the `t0` support, which is
Change 3 below. Fix it there, and report any chain that still collapses rather than lending
it a sibling's tuning — chains repaired that way measured 0.16–0.83× their siblings' spread
and still broke R-hat.

### Change 2: `resample_final=True`

The post-loop resample equalizes the weights so the stored particles are an equally weighted
sample. Without it, downstream code reading `final_state.particles` as if it were a posterior
sample is quietly ignoring the weights. `SMCResult` reports `num_unique` and `num_unique_smc`
— distinct particles after and before the resample — so you can see how much of the cloud is
real either way. With `num_mcmc_steps=1`, duplicates compound across tempering iterations;
this is worth looking at before trusting either number.

### Change 3: the `t0` support constraint you have never had

`init_from_prior_mode` starts warmup from a bare `prior.mode()` with no support check, and
`sample_prior_particles` draws the cloud from the **untruncated** prior. Meanwhile you fit
real data, and your `t0` prior median (`exp(-1.609) ≈ 0.2 s`) sits close to observed minimum
response times. Particles with `t0` above a subject's fastest response land in the race
likelihood's slope-1e3 penalty region.

`racing-diffusion-conflict` measured what that does: in every population producing a
degenerate step size, the degenerate chain was exactly the chain with the most violating
subjects — R-hat 1.94 / 2.60 / 2.76, falling to ~1.01 once those chains were dropped. Use:

```python
support = T0Support.from_spec(spec, min_valid_rt(data[..., 0], mask=mask))
particles, num_exhausted = init_particles_from_prior(
    flat_space, num_particles, key, support=support
)
```

Note that this is **rejection sampling, not clipping**, and deliberately. A clip is a point
mass: at `max_fraction=0.9`, 55% of (particle, subject) draws were capped and the worst
subject had 93% of its particles pinned to a single value — dispersion destroyed in the one
coordinate the constraint exists to protect. `clip` remains only as the exhaustion fallback,
and `num_exhausted` tells you how often it fired.

Also note that the restricted cloud is **not the prior**, and must not be stored as one. Its
`t0` spread is narrower — a measured ratio of 0.529 — so any contraction measured against it
understates the contraction. `adaptive_tempered_smc` weights increments by the likelihood
alone and never corrects the initial distribution, so this is a real difference in the
sampler's starting measure, not a bookkeeping detail. `save_hierarchical_posterior` takes
`prior_particles` as an explicit argument rather than reusing the initial cloud, so pass an
unrestricted `flat_space.sample_particles(...)` draw there and keep the restricted cloud as
an initialisation artefact.

### What `save_hierarchical_posterior` changes about the output

The schema is your `build_datatree` schema: `posterior` with per-parameter `subject`-dimensioned
variables plus population `mu`/`sigma`, `sample_stats`, `observed_data`, `prior`, and the SMC
summaries in the attributes. Three differences:

* **`sample_stats` names the weights `weight`, not `log_weights`.** `TemperedSMCState.weights`
  are normalized *linear-space* weights; the old name described the wrong quantity.
* **`sigma` is documented as being on the log scale and is not exponentiated** — unchanged
  behaviour, but now stated. An exponentiated standard deviation is not a standard deviation
  of anything.
* **More SMC summaries are stored**: `log_marginal_likelihood_sd`, `smc_num_unique`,
  `smc_num_unique_pre_resample`, `smc_weight_ess`, alongside the mean and per-chain values
  you already keep.

### And the hand-written log-Jacobian

`smc_fit.py` computes the hierarchical prior's log-det-Jacobian by hand, with a comment
blaming a broadcasting bug in `JointMap.forward_log_det_jacobian`. **Your numbers are right;
the explanation is wrong.** Called with no `event_ndims` it does overcount — by
`(P-1) × ljd_cc`, which is state-dependent and would tilt the posterior over correlations
rather than cancel — but called with per-component `event_ndims` it returns the correct
scalar, matching your hand-written value at every point tested and satisfying the inverse
log-det identity. `eamax.hierarchical.joint_log_det_jacobian` does the latter, generically,
without hard-coding the component names `"s"` and `"psi_raw"`. **This changes no numbers.**

## And: add tests

This repository currently has none — `pytest` is pinned and unused — while containing the
most novel code of the three (the N-accumulator generalisation). Most of that code is now in
`eamax` and tested there. What remains here is worth covering: `build_param_spec`'s ordering
invariant (B-type parameters and `t0` trailing), `load_dataset`'s padding, and the claim in
`fit_hierarchical_crdm.py`'s docstring that exactly one accumulator is pulsed per trial —
which is asserted about all of `data/prep` but checked nowhere.
