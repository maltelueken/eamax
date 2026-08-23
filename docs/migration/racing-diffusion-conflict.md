# `racing-diffusion-conflict` → `eamax`

**Numbers change in the tails**, via the floor semantics (see `README.md`). The bulk of any
likelihood surface is unaffected; deep-tail survival terms move by up to a few nats per
trial, and totals shift accordingly. Measure against the golden fixtures rather than
assuming.

## What moves

| From | To |
|---|---|
| `likelihoods/rdm.py` — `inv_gauss_logpdf`, `inv_gauss_logsf` | `eamax.accumulators.wald` |
| `likelihoods/rdm.py` — `_clamp_log`, `_penalize_invalid_rt` | `eamax.numerics` (semantics changed) |
| `likelihoods/rdm.py` — `_censored_eval_rt`, `_apply_censoring` | `eamax.race` (inside `race_loglik`) |
| `likelihoods/rdm.py`, `crdm.py`, `hierarchical_*.py` — the four race likelihoods | `eamax.race.race_loglik` |
| `likelihoods/crdm_volterra.py` | `eamax.accumulators.volterra` |
| `simulators/crdm_utils.py` — pulse + Euler–Maruyama | `eamax.accumulators.{pulse,diffusion}` |
| `simulators/rdm.py`, `wald.py` — the race simulators | `eamax.simulate` + `eamax.design` |
| `simulators/base.py` — `HierarchicalRDMPriorLKJMVN`, `interval_to_mu_loc_scale` | `eamax.hierarchical` |
| `flows.py` | `eamax.flows` (extra `flows`) |
| `mcmc.py` — `warmup_multiple_chains` | `eamax.inference.warmup.window_adaptation` |
| `mcmc.py` — `warmup` (single-chain, broadcast across chains) | *deleted, no replacement — see "Inference"* |
| `mcmc.py` — `inference_loop_multiple_chains` | `eamax.inference.mcmc` (same name) |
| `smc.py` — `smc_inference_loop`, the tempered-SMC driver | `eamax.inference.smc.{smc_inference_loop,tempered_smc}` |
| `smc.py` — `count_unique_particles` | `eamax.inference.diagnostics` (defect fixed, see below) |
| `scripts/parameter_recovery.py` — `_sample_init_positions_in_support` | `eamax.inference.init.init_positions_from_prior` |
| `scripts/parameter_recovery_hierarchical.py` — the particle-cloud init | `eamax.inference.init.init_particles_from_prior` |
| the degenerate step-size repair | *deleted, no replacement — see "Inference"* |

## What stays

* **`conf_jax/prior/*.yaml` and the `create_*_prior_*` factories.** `distrax.Joint` priors
  are this repository's scientific content.
* **`simulators/base.py` — `TruncatedNormal`.** 60 lines, distrax-only, one consumer.
* **The training loop** in `scripts/train_conditioner.py`. `eamax.flows.train` provides
  `loss_fn` / `train_step` / `eval_step`; the schedule and checkpoint policy stay here.

## Structural changes to expect

* **The censoring sentinel moves down a layer.** `simulate_crdm_single_trial` currently bakes
  the race-level `(-1.0, -1)` sentinel into the *trial* simulator. In `eamax` a
  non-crossing accumulator returns `inf` and `race_sample` produces the sentinel. The
  refactor is exact — noise and crossings are independent per accumulator — and it is what
  lets the race read an all-`inf` trial as censored without a special case.
* **Volterra and the flow are separate classes**, not two backends of one. A deterministic
  solver with static discretisation knobs and a learned pytree passed by identity have
  incompatible static-argument requirements.
* **The hybrid race stays in your code.** `eamax` ships `gather_by_mask` / `overlay_by_mask`
  so the neural density is still evaluated once per trial rather than once per accumulator,
  but the routing rule — yours is the sign of `amp` — is four lines of `jnp.where` and is
  genuinely model-specific.
* **`likelihoods/__init__.py` and `simulators/__init__.py` are already re-export barrels**, so
  the 20 `_target_` strings are covered by a one-file change per barrel.

## Inference

This repository has the best sampler of the three, and it mostly migrates unchanged.
`eamax.inference`'s defaults *are* its behaviour: per-chain warmup, per-chain particle
clouds, rejection-sampled starts inside the support, a post-loop resample. `mcmc.py` and
`smc.py` are deleted outright, and the two recovery scripts lose several hundred lines of
inlined driver code.

Two things change, one defect is fixed, and one thing you have goes away.

### The degenerate step-size repair does not migrate

This is the only piece of your sampler with no destination. `eamax` has no
`repair_degenerate_tuning` and will not grow one: the mandated path is
`init_positions_from_prior` / `init_particles_from_prior` with `support=`, then
`window_adaptation` per chain, then sampling with that tuning untouched.

Your own measurement is the argument. In every population that produced a degenerate step
size, the degenerate chain was **exactly** the chain with the most subjects whose `t0` sat
above their fastest observed response time — a start outside the support, not a property of
the posterior. `T0Support` removes that cause at initialization. And the repair did not
actually rescue those chains: repaired, they came out under-dispersed at 0.16–0.83× their
siblings' spread and still broke R-hat (1.94 / 2.60 / 2.76), falling to ~1.01 only once they
were *dropped*. The step-size column looked fixed; the fit was not.

So: delete the repair, keep the detection. A step size two orders of magnitude below its
siblings' after a support-constrained start is a finding — log it, drop the chain, and report
how many chains you dropped, the way you already report R-hat.

### You gain the log marginal likelihood

`tempered_smc` accumulates it unconditionally — one scalar add per tempering iteration, no
opt-out — and returns it per chain on `SMCResult.log_marginal_likelihood`. Today
`smc_inference_loop` discards `SMCInfo.log_likelihood_increment`. The estimate is validated
against a conjugate Gaussian with a closed-form evidence: **-5.1156 measured against -5.1200
analytic, an error of 0.0044 nats** in D=2. Nothing in this repository depends on it yet, but
it is free and it is the quantity `cognitive-control-comparison` compares models with.

### `t0` no longer has to be last, or centered

`_sample_init_positions_in_support`'s hierarchical counterpart tests
`sample["theta_bt"][:, -1]`, which requires `t0` to be both the final parameter *and* in the
centered block. `T0Support` tests the reconstructed `(S, P)` block instead: one extra einsum
per draw, and both requirements lift. Where `t0` *is* last and centered the two are the same
number, so no current configuration changes.

`T0Support.from_spec` also raises if the spec has no `t0`, or if its link is not `"log"` —
turning the silent wrong-parameter clipping that the positional version would do into an
error.

### `count_unique_particles` was under-reporting collapse

The projection-and-count method has a failure the original could not have caught. Two
**bit-identical** particle rows can project to values differing in the last bit, because XLA
accumulates the matvec differently depending on the row's position — measured as 41 unique
in a cloud of 40. The fix sorts on the projection and then compares adjacent *full rows*
exactly, so bit-identical duplicates always collapse. The docstring, the projection
rationale, and the "an O(N²D) alternative would allocate ~1 GB" note all move across intact.

The old failure direction is the unsafe one: it reports a healthier cloud than there is. If
you have recorded unique-particle counts, expect them to come out the same or **lower**, and
never higher.

### Also worth knowing

* **The `warmup` / `warmup_multiple_chains` pair becomes one function.** Only
  `warmup_multiple_chains` has a counterpart, as `window_adaptation`; the single-chain
  `warmup` broadcast across chains has no equivalent and no keyword that restores it.
* **`SMCResult.weights` is named for what BlackJAX returns**: normalized *linear-space*
  weights, not log weights.
* **`TemperedSMCState.lmbda` is `tempering_param` on newer BlackJAX.** `eamax` reads
  whichever exists; this repository's pin is unpinned, so it works either way.

### Report after migrating

Per-chain step sizes and unique-particle counts, before and after. Both should be unchanged
except for the `count_unique_particles` fix — and for whatever the repair used to overwrite,
which now shows through. Report the number of chains dropped for a collapsed step size as
part of the run, not as a step-size value.

## Risks

* **Orbax checkpoints.** `eamax.flows.MLP` keeps the class name and the `linear1` / `linear2`
  attribute names, so existing checkpoints restore. Do not rename them. `save_conditioner`
  now also writes a JSON sidecar recording `context_names`; existing checkpoints have none
  and load unchanged, but new ones should carry it — a *reordered* context of the same width
  is otherwise a silent wrong-density bug, as `load_conditioner`'s original docstring admits.
* **`distrax.Joint` priors are passed as `static_argnames`** and hashed by identity. They stay
  in this repository and must not be wrapped or reconstructed, or every call recompiles.
* **`orbax` import side effects.** `eamax.flows.checkpoint` imports it lazily, so importing
  `eamax` no longer reconfigures your root logger. You can drop the `force=True` workaround.
