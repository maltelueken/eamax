# `eam-abi-robustness` → `eamax`

**Expected outcome from the likelihood migration: no numbers change.** `eamax`'s race is
bit-identical to this repository's on the LBA and agrees to one ULP on the RDM (a length-2
`sum` reduction versus a scalar `+`; far inside the 1e-9 EMC2 tolerance). Every prior, every
log-density factory and every parameter name stays put, so saved artifacts and Hydra configs
are unaffected.

**The inference migration is a different story**, and this repository has the most at stake
in it. See "Inference" below: its behaviour commit changes which datasets survive the
convergence filter, and that set is the comparison set for the NPE-vs-MCMC result.

This repository is the *source* of `eamax`'s numerics: the Wald primitives, the combined
trial floor, and the EMC2 reference fixtures all came from here.

## What moves

| From | To |
|---|---|
| `src/rdm_jax.py` — `inv_gauss_logpdf`, `inv_gauss_logsf` | `eamax.accumulators.wald` |
| `src/rdm_jax.py` — `_finalize_race_logp` | `eamax.numerics.finalize_trial_logp` |
| `src/rdm_jax.py` — `rdm_race_logpdf` | `eamax.race.race_loglik` (N-way) |
| `src/lba_jax.py` — `lba_logpdf`, `lba_logsf`, `_lba_guard`, `lba_race_logpdf` | `eamax.accumulators.lba` |
| `src/rdm_jax.py` / `src/lba_jax.py` — simulators | `eamax.simulate.race_sample` + `eamax.design` |
| `tests/reference/`, `tests/test_emc2_reference.py` | `eamax/tests/` (already copied) |
| `src/mcmc.py` — the four `make_{bounded,meta}_to_{un,}constrained` factories | `eamax.inference.transforms.BlockTransform` |
| `src/mcmc.py` — `simple_to_{un,}constrained` | `eamax.inference.transforms` (same names) |
| `src/mcmc.py` — `warmup`, `inference_loop_multiple_chains` | `eamax.inference.{warmup,mcmc}` |
| `src/mcmc.py` — `fit_mcmc_gpu`, `fit_mcmc_gpu_batch` | `eamax.inference.mcmc.{fit_nuts,fit_nuts_batch}` |
| `src/mcmc.py` — `save_mcmc_posterior`, `load_mcmc_posterior` | `eamax.io.{save,load}_dataset_posterior` |
| `src/mcmc.py` — `bounded_init_position`, `meta_init_position` | `BlockTransform.midpoint` + `eamax.inference.init` |
| `src/data.py` — `save_posterior`, `load_posterior` | `eamax.io` (folded into the two above) |

## What stays, and why

* **The eight `make_*_logdensity` factories.** They are the Hydra `_target_` strings, and
  they bundle a prior this repository owns with a likelihood `eamax` now provides. Keep them
  here as thin wrappers: `spec.constrain` → `spec.log_det_jacobian` → your log-prior →
  `race_loglik`. Zero config churn, and `tests/test_config.py` needs no edit.
* **`src/priors.py`.** Prior hyperparameters are this study's scientific content.
* **`SplittableKey`, `batched_experiment`, `static_num_obs` callers, `src/simulation.py`.**
  BayesFlow adapters. (`static_num_obs` itself is in `eamax.batching`; the `batched_experiment`
  wrapper around it is not.)
* **`src/cases.py`, `src/artifacts.py`, `src/design.py`, `src/metrics.py`.** Pipeline, not
  models. `src/mcmc.py` no longer stays — see "Inference" — but its Hydra `_target_` strings
  for `simple_to_unconstrained` / `simple_to_constrained` migrate by string substitution,
  since `eamax.inference.transforms` keeps both names.
* **`src/data.py` — `save_dataset`, `load_dataset`, `load_hdf5`.** Simulator-output IO and
  the legacy HDF5 shim. Only the two *posterior* functions move.

## Mapping the parameterization

`eamax.design.intercept_slope_spec()` reproduces `[v_intercept, v_slope, s_true, b, t0]`
exactly, including `t0` last (which `fit_mcmc_gpu`'s `init_position.at[-1]` depends on) and
`s_false` fixed to 1. `sat_spec()` reproduces the speed/accuracy variant, with `b_diff` under
`"sum"` coding so `b_accuracy > b_speed` still holds by construction.
`lba_intercept_slope_spec()` reproduces `[v_intercept, v_slope, s_true, A, B, t0]` with `B`
as the threshold gap.

**The SAT condition channel changes sign convention.** This repository carries
`is_accuracy` as the third channel of `x`; `eamax` uses `condition == 1` for the *reference*
level, which for a speed/accuracy design is speed. Pass `condition = 1 - is_accuracy`.

## Shim sketch

```python
# src/rdm_jax.py
from eamax.accumulators.wald import inv_gauss_logpdf, inv_gauss_logsf  # noqa: F401
from eamax.numerics import finalize_trial_logp
import eamax
eamax.enable_x64()   # this module's import side effect is load-bearing here; keep it

def rdm_race_logpdf(rt, drift_winner, drift_loser, s_winner, s_loser, threshold, ndt, min_p=1e-10):
    ...  # stack winner-first, call eamax.race_loglik with response = 0
```

Keep `jax.config.update("jax_enable_x64", True)` in the shim: `lba_jax` documents that it
depends on the side effect. `eamax` itself never does this — call `eamax.enable_x64()`.

## Inference

This repository's sampler is the one with the most consequential change, and it needs the
two-commit split more than the other two.

### (a) The shim commit — everything except the warmup

The transforms are byte-for-byte the same expressions, so `to_unconstrained` /
`to_constrained` produce identical values, and `inference_loop_multiple_chains` is the same
loop. **The warm-up is not reproducible, on purpose.** `src/mcmc.py:164` adapts one chain and
replicates its state and tuning across the rest; `eamax` has no equivalent — `fit_nuts`'s
`shared_warmup=True` and the `broadcast_warmup` behind it were removed rather than kept as a
compatibility switch, because a broadcast warmup is precisely what makes this repository's
R-hat filter unable to fail (see `docs/migration/README.md` for the ±6 measurement).

So for this repository the two-commit split collapses at one point: the warm-up change lands
in commit (a) regardless. Everything else in (a) still holds the digest. Record the R-hat
vector before and after in that commit's message, since it will move.

One structural difference to handle in this commit: `fit_nuts` takes
`make_init_positions(key, data)` and calls it *inside*, next to `make_logdensity_fn(data)`.
Today the initial position is one fixed vector with its last entry overwritten by
`data[:, 0].min() / 2`. Wrap that as-is:

```python
def make_init_positions(key, data):
    position = init_position_from_values(       # unconstrained, thanks to `transform=`
        configured_values,
        spec=spec,
        min_rt=min_valid_rt(data[:, 0]),
        t0_fraction=0.5,
        transform=transform,
    )
    support = T0Support.from_spec(spec, min_valid_rt(data[:, 0]))
    positions, _ = jitter_positions(position, num_chains, key, scale=0.1, support=support)
    return positions
```

Note the `jitter_positions` call rather than a `jnp.broadcast_to`. Returning `num_chains`
copies of one position is *not* a shim here: `window_adaptation` will adapt each copy
independently, from the same place, which costs the full `num_chains`× warm-up and still
leaves between-chain variance near zero — the worst of both. This repository has no prior
object to draw from, so jittered starts are its `init_positions_from_prior`; see the
behaviour section below for why `support=` is not optional.

Two notes. `init_position_from_values` takes `t0`'s index from `spec`, not from the
position's last slot; every current layout has `t0` last on the log link, so this is the same
number. And `fit_nuts` works in unconstrained coordinates throughout, which is why
`transform=` is passed here rather than the caller calling `to_unconstrained` afterwards —
the forward transform now appears exactly once.

### (b) The behaviour commit — this is the one that matters

Three things follow from the per-chain warm-up.

**Warmup cost multiplies by `num_chains`** — ×4 at the current setting. This is the real
cost, and it is not avoidable through `eamax`: an amortized study fitting a thousand
simulated datasets pays it. The lever is `num_steps_warmup`, `num_chains`, or the number of
datasets — all of which are visible decisions — rather than a shared warm-up that buys the
time back out of the diagnostic. Budget for the ×4 before the behaviour commit rather than
discovering it in the queue.

**Chains stop sharing tuning.** Each carries its own step size and mass matrix, so
`inference_loop_multiple_chains` must be passed `kernel_params` with a leading chain axis.
`fit_nuts` does this for you.

**R-hat becomes able to fail.** Today every chain starts from the same warmed state, so
between-chain variance starts at zero and the diagnostic cannot detect the failure it exists
to detect. On a controlled bimodal target at ±6, the shared start reports max R-hat 1.001
while putting 100% of its mass in one mode; a dispersed start reports 1.734 and splits 50/50.
Your convergence filter will therefore drop a **different and probably larger** set of
datasets — and that set is the comparison set for the NPE-vs-MCMC result. **Report the
converged fraction per experiment, before and after.** Note that the filter is now yours to
run: `load_dataset_posterior` no longer applies one (see "Read-back changes").

Per-chain warmup needs a dispersion source, and this repository has no prior object to draw
from, so it cannot use `init_positions_from_prior` the way the other two do. Use
`jitter_positions`, and **pass the support argument**:

```python
support = T0Support.from_spec(spec, min_valid_rt(data[:, 0]))
positions, num_exhausted = jitter_positions(
    position, num_chains, key, scale=0.1, support=support
)
```

`position` is unconstrained, as above. `num_exhausted` counts draws that hit
`max_attempts` and fell back to a clipped value; it should be zero, and a non-zero count
means `scale` is too wide for the data at hand.

Naked jitter without `support=` reproduces the exact failure `racing-diffusion-conflict`
measured: chains started with `t0` above the observed minimum RT land on the race
likelihood's slope-1e3 penalty wall, and in every population producing a degenerate step
size the degenerate chain was the one with the most violating subjects (R-hat 1.94 / 2.60 /
2.76, falling to ~1.01 once dropped). A dispersed start that lands outside the support is
worse than no dispersion at all.

**There is no repair for a chain that collapses anyway.** `repair_degenerate_tuning` was
removed with the rest: if a step size comes back two orders of magnitude below its siblings',
the fix is upstream — check `num_exhausted`, check the support — and if it persists, report
the chain and drop it. Do not give it tuning it did not adapt to; measured, that leaves the
chain under-dispersed at 0.16–0.83× its siblings' spread and still breaking R-hat, which is a
fit that *looks* repaired in the step-size column and is not.

### Read-back changes

`load_mcmc_posterior` did five things in one load-bearing order — pool chains → mask
unconverged → back-transform the **full** vector → select by name → stride-thin.
`load_dataset_posterior` keeps **only the back-transform and the selection**, in that order,
and returns every draw of every chain shaped `(chain, draw, dataset, param)`.

**`eamax.io` reads and writes; it applies no diagnostic and no threshold.** A reader that
filters returns an array whose shape depends on a statistic — so an empty result and a
failed read look alike — and pooling on the way out destroys the axis R-hat is computed
over, which means you could not check the filter or vary it even if you wanted to. Both are
your decisions, and this repository is the one where they *are* the result.

The other three steps move to the caller, unchanged in order:

```python
theta = load_dataset_posterior(path, to_constrained=transform.forward, param_names=names)
rhat_per_dataset = rhat(theta, chain_axis=0, sample_axis=1)     # (dataset, param)
is_converged = np.all(rhat_per_dataset < psrf_threshold, axis=-1)
posterior = thin(pool_chains(theta)[is_converged], num_target_samples, axis=1)
```

`pool_chains` and `thin` are in `eamax.inference.posterior`, `rhat` in
`eamax.inference.diagnostics`; none of them needs xarray. Four notes:

* **Chain-major pooling is preserved by `pool_chains`**, matching xarray's
  `stack(sample=("chain", "draw"))` and your original `reshape`. Do not hand-roll it.
* **Mask the pooled numpy array, not the `Dataset`.** That is what makes total
  non-convergence come back as a length-zero leading axis rather than raising — the case
  your plotting code relies on.
* **R-hat comes from `eamax.inference.diagnostics.rhat`, not `arviz_stats.rhat`.** They agree
  to six decimals on both a converged and a deliberately split set of 4×2000 chains, so the
  converged set is unchanged, and the filter no longer needs ArviZ. Running it on the
  back-transformed values above rather than the stored unconstrained ones also changes
  nothing: it is rank-normalized, hence invariant under any monotone link.
* **Warn or log as you like.** The dropped-dataset warning came out with the filter, so put
  your `logger.info` back where it was.

## Risks

* **~13 `_target_` strings** point at `rdm_jax.*` / `lba_jax.*`. Re-export shims keep them
  resolving; do not rename anything they reference.
* **The `src/`-rooted editable install** means modules are bare top-level names
  (`from rdm_jax import ...`). Preserve it — do not convert to a package in the same change.
* **`jax 0.11` vs `eamax`'s `>=0.9` floor.** Fine, but this repository is the one on the
  newer JAX; run its suite first after the bump.
