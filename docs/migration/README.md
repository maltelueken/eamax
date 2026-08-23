# Migrating onto `eamax`

`eamax` was extracted from three repositories that had each grown their own copy of the
racing diffusion model's simulation and density code. None of them has been changed yet.
This directory records, per repository, what to delete, what to keep, what will break, and
what will change numerically.

## Do this first, in every repository: freeze golden fixtures

Before touching any code, commit two artifacts per repository:

1. **A per-trial log-likelihood table** on a fixed grid of parameters and response times,
   covering the tails and the `rt <= t0` region as well as the bulk.
2. **A fixed-seed simulator digest** — a hash or a quantile summary of a few thousand draws.
3. **An inference digest**, now that the samplers move too: for `eam-abi-robustness`, a hash
   of `fit_mcmc_gpu_batch`'s output for two datasets × 200 draws plus the per-dataset R-hat
   vector; for `racing-diffusion-conflict`, one `recover_population` particle mean and its
   iteration count; for `cognitive-control-comparison`, `fit()`'s log marginal likelihood for
   one dataset.

Every migration step below is then a diff against these, and "the numbers changed" becomes a
measurement rather than something discovered months later in a figure. Two of the three
migrations *are expected to change numbers*; without fixtures you cannot tell an intended
change from a mistake. The inference digest matters most, because the inference migration is
deliberately done in **two commits** — one that must not change numbers, and one that will.

## What is shared, and what deliberately is not

`eamax` owns the accumulator densities, the race likelihood, the parameterization layer,
simulation, the neural flow, and the hierarchical prior family.

It does not own **priors** — not `eam-abi-robustness`'s informed truncated normals, not
`racing-diffusion-conflict`'s `distrax.Joint` factories, not
`cognitive-control-comparison`'s `PARAM_PRIORS` numbers. This is where the three repositories
genuinely disagree *scientifically*, so there is no shared implementation underneath to
extract; unifying them would produce an abstraction with nothing in it. `eamax` stops at the
likelihood boundary and provides `ParamSpec.constrain` / `ParamSpec.log_det_jacobian` so the
link function — which is inseparable from the parameterization — is not duplicated either.

It also owns **inference**: starting values, NUTS warmup and driving, tempered SMC, and
convergence diagnostics, in `eamax.inference` (extra `inference`). This is a change from the
first extraction pass, which left the drivers behind on the grounds that they were
"inference, not models". That reasoning was wrong in one specific way. The three drivers had
converged on the same BlackJAX call sequence and diverged mainly in *which defects each one
carried* — a broadcast warmup that makes R-hat unable to fail, an SMC cloud drawn with no
support constraint, a log marginal likelihood computed in one repository and discarded in
another. Kept separate, each defect stays local and invisible; brought together, they can be
measured against each other — and the losing side of each comparison is then simply not
implemented, rather than kept as an option. See "The inference layer" below.

It does still not own **artifact I/O** in the sense of a single schema. `eamax.io` ships two
writers, because the two posterior layouts have almost nothing in common: one `theta`
variable with dims `(chain, draw, dataset, param)` for many fits of one model, versus
per-parameter variables with a `subject` dimension for one hierarchical fit. What is shared
is the array-level work underneath them — layout, pooling, thinning, and the back-transform
order — which lives in `eamax.inference.posterior` and needs no xarray at all.

And `eamax.io` **reads and writes, and does nothing else.** No function there computes a
diagnostic, drops a fit on one, or thins. `load_dataset_posterior` used to do all three and
no longer does: it returns every draw of every chain, and the caller runs `rhat`,
`pool_chains` and `thin` itself. A reader that filters returns an array whose *shape*
depends on a threshold, and pooling on the way out destroys the axis R-hat is computed over
— so the repository that keeps its converged fraction as a result cannot inspect or vary the
filter that produced it. See `eam-abi-robustness.md`, "Read-back changes", for the four-line
replacement.

Nor does it own two things that look shareable but are not: `SplittableKey` and
`batched_experiment` (BayesFlow calling-convention adapters with one consumer), and the
per-model log-density factories, which bundle a prior with a likelihood.

## The behavioural changes

Three, in decreasing order of how much they will move numbers.

### 1. Guard logic moved from the accumulator into the race

In the source repositories the per-accumulator function owned the `t0` shift, the parameter
guards, the density floor *and* the invalid-RT penalty, and warned callers in prose not to
clamp its output again. `eamax` accumulators return raw log-densities at decision times, and
`race_loglik` applies every guard once, at the end, exposing no intermediate to re-clamp.

This is not a stylistic change. `cognitive-control-comparison` re-clamped, which flattened
the penalty to a constant and removed the gradient that pushes `t0` back into the valid
region — see its migration note.

### 2. The likelihood floor applies once, to the trial total

`eam-abi-robustness` floors `log_pdf + log_sf` once at `log(1e-10)`;
`racing-diffusion-conflict` and `cognitive-control-comparison` clamp each component
separately at `log(1e-12)`. `eamax` adopts the former, for two reasons:

* It is the only variant validated against an outside implementation — EMC2's
  `log_likelihood_race` floors each trial at `min_ll = log(1e-10)`, the same constant at the
  same place, and `tests/test_emc2_reference.py` holds agreement to 1e-9 on total dataset
  log-likelihood.
* The per-component floor scales with the number of accumulators. A four-response race
  effectively floors at `4 * log(1e-12)` while *inflating* every deep-tail survival term once
  per loser. For work whose output is model comparison by marginal likelihood, that bias has
  the wrong sign: it flatters models that push losers into the tail.

`min_p` is a keyword argument, so a repository that needs continuity with published runs can
pin `1e-12` for one release.

### 3. NaN containment no longer floors

`contain_nan` replaces non-finite values with the log floor but leaves finite values alone,
so a value below the floor still cancels correctly against other terms before the floor is
applied to the total. Previously the two were conflated.

## The inference layer

`eamax.inference` replaces `eam-abi-robustness/src/mcmc.py`,
`racing-diffusion-conflict/src/confrdm_jax/{mcmc,smc}.py`, and
`cognitive-control-comparison/scripts/hierarchical_fit/{smc_fit,prior,output}.py` — plus the
several hundred lines of driver code inlined in the fitting scripts.

### There is one supported way to fit, and it is not optional

Every repository, every sampler, every fit:

1. **Initialize from the prior, into the support.** `init_positions_from_prior` for NUTS,
   `init_particles_from_prior` for SMC — one independent draw per chain, `support=` always
   passed.
2. **Warm up per chain.** `window_adaptation` over all `num_chains` starts. Each chain gets
   its own step size and inverse mass matrix.
3. **Sample with that tuning, unmodified.** No inspection, no substitution, no repair.

`eamax` used to expose three ways off that path, and they have been **removed** — not
deprecated, not left as opt-in keywords. A consumer repository cannot reach them:

| Removed | What it did | Why it is gone |
|---|---|---|
| `broadcast_warmup`, and `fit_nuts(..., shared_warmup=True)` | Adapted one chain, replicated its state and tuning across the rest | Between-chain variance starts at zero, so R-hat measures Monte-Carlo noise. Bimodal target at ±6: shared start reports max R-hat 1.001 with 100% of its mass in one mode; dispersed start reports 1.734 and splits 50/50 |
| `broadcast_particles` | Replicated one SMC particle cloud across chains | Chains then differ only in SMC randomness, so their spread is not sampling variability — and that spread is the standard error on the log marginal likelihood, which is one repository's entire scientific output |
| `repair_degenerate_tuning` | Replaced a collapsed step size and mass matrix with the healthy chains' median | It treats a symptom of a bad start. In every population producing a degenerate step size, the degenerate chain was the one with the most subjects violating the `t0` support; repaired chains stayed under-dispersed at 0.16–0.83× their siblings' spread and still broke R-hat (1.94 / 2.60 / 2.76) |

The two `broadcast_*` helpers saved warm-up work — per-chain adaptation costs `num_chains`×
— and that saving was real. It was paid for out of the only numbers that say whether a fit is
usable, which is the wrong account. If a study genuinely cannot afford per-chain adaptation,
that is a decision to argue for and record in the study, not a keyword to inherit.

`repair_degenerate_tuning` is the one worth restating: **do not repair, prevent.** Pass
`support=T0Support.from_spec(...)` at initialization and a collapsed step size mostly stops
happening. If one still collapses, that is a finding about the posterior — report it
alongside the fit and drop the chain; do not hand it a step size it did not adapt to.

**Migrate it in two commits, per repository.**

**(a) The shim commit — numbers must not change.** Swap the imports, keeping today's call
sequence and today's arguments wherever `eamax` still accepts them (`resample_final=False`,
the five SMC constants, the same prior and log-density objects). Verify against the inference
digest from step 0. **Warm-up and initialization are the exception**: there is no argument
that reproduces a broadcast warmup or a shared particle cloud, so a repository doing either
today gets its behaviour change in this commit whether it wants it or not. Where that is the
case, say so in the commit message and treat the digest comparison as a *report*, not a gate.

**(b) The behaviour commit — numbers will change.** Drop the remaining compatibility
arguments one at a time and report the delta. Each repository's own note says which changes
apply to it and what to measure.

### Three things every repository should know

* **The `t0` support constraint is enforced from `ParamSpec`, not by position.** All four
  current routines assume `t0` is the last parameter, positionally and unenforced.
  `T0Support.from_spec` calls `spec.index("t0")` and *raises* if the name is absent or its
  link is not `"log"`. Every current layout has `t0` last on the log link, so nothing changes
  today — but a future layout that moves it will now fail loudly instead of clipping the
  wrong parameter.

* **`TemperedSMCState.lmbda` was renamed to `tempering_param`.** Between BlackJAX 1.2.5 and
  1.6.2 the tempering field changed name. `eamax` reads whichever exists and pins the
  behaviour with a canary test, so the driver spans both versions. Anything in a consumer
  repository that touches that field directly does not.

* **`blackjax.diagnostics.rhat` needs BlackJAX ≥ 1.6.** It is rank-normalized split-R̂
  (Vehtari et al. 2021) and reproduces `arviz_stats.rhat` to six decimals, so the convergence
  filter costs no ArviZ dependency. `eamax` deliberately does **not** fall back to 1.2.5's
  `potential_scale_reduction`: that is plain Gelman–Rubin, a different statistic, and
  substituting it would quietly move which fits a threshold accepts.
  `cognitive-control-comparison` pins 1.2.5 and must upgrade; everything else it uses is
  compatible across the range.

### Two defects fixed on the way in

* **`count_unique_particles` under-reported collapse.** The ported version projects particles
  onto a random direction and counts distinct projections. Two *bit-identical* rows can
  project to values differing in the last bit, because XLA accumulates the matvec differently
  depending on row position — measured at 41 unique in a cloud of 40. `eamax` sorts on the
  projection and then compares adjacent rows exactly. The old failure direction is the unsafe
  one: it reports a healthier cloud than there is.

* **The hierarchical log-Jacobian "TFP bug" is not a bug.** Two repositories hand-compute the
  prior's log-det-Jacobian with a comment blaming a broadcasting bug in
  `JointMap.forward_log_det_jacobian`. Calling it with no `event_ndims` does overcount, by
  `(P-1) × ljd_cc` — *state-dependent*, so it would tilt the posterior over correlations
  rather than cancel. Passing per-component `event_ndims` returns the correct scalar.
  **Their numbers are right; their explanation is wrong**, and `eamax.hierarchical.joint_log_det_jacobian`
  now does it generically instead of hard-coding the component names `"s"` and `"psi_raw"`.
  Numerically identical, so this changes nothing.

## Order of work

1. `eam-abi-robustness` — shims only, **no numbers change**. Lowest risk, and it is what
   keeps the EMC2 fixtures honest in their original setting. Its *inference* behaviour commit
   is a separate matter, and is the single most consequential change in the whole migration
   — see its note.
2. `racing-diffusion-conflict` — shims plus the flow. Numbers change in the tails.
3. `cognitive-control-comparison` — the real payoff: it ends with no dependency on
   repository 2 at all. Numbers change twice from the likelihood, and three more times from
   inference; make the bug fix a separate commit.

See `eam-abi-robustness.md`, `racing-diffusion-conflict.md`, and
`cognitive-control-comparison.md`.
