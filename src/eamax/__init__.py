"""Evidence accumulation models in JAX: simulation and likelihoods.

Shared implementation of the racing diffusion model and its relatives, extracted from
three research repositories that had each grown their own copy.

The library is four seams deep:

* `eamax.accumulators` -- first-passage-time distributions (`Wald`, `LBA`, pulsed
  variants). They see *decision times* and return raw log densities.
* `eamax.race` -- the winner's-density-times-losers'-survival likelihood, for any number
  of accumulators, owning non-decision time and every numerical guard.
* `eamax.design` -- parameterizations: how a flat parameter vector plus a trial design
  become per-accumulator drifts, thresholds and noise.
* `eamax.simulate` -- sampling driven by the same parameterization the likelihood uses.

`eamax.flows` (extra `flows`) and `eamax.hierarchical` sit alongside, for neural density
estimation and multi-subject priors respectively.

`eamax.inference` (extra `inference`) adds the sampling layer on top: starting values,
NUTS warmup and driving, tempered SMC, and convergence diagnostics. It imports without
BlackJAX installed -- only reaching for a sampler raises -- so importing `eamax` costs
nothing extra. `eamax.io` (extra `io`) reads and writes posterior artifacts, and is the
one module not imported here: it needs xarray and ArviZ, which nothing else does.
"""

from . import accumulators, batching, design, hierarchical, inference, numerics, race, simulate
from .accumulators import Wald, inv_gauss_logpdf, inv_gauss_logsf
from .numerics import MIN_P, MIN_RT, finalize_trial_logp, guard_positive, log_floor
from .design import Parameterization, TrialDesign, build_params_fn, parameterization
from .simulate import race_sample, simulate_dataset, simulate_race
from .race import (
    censored_eval_rt,
    gather_by_mask,
    overlay_by_mask,
    race_from_arrays,
    race_loglik,
    winner_mask,
)

__version__ = "0.1.0"

__all__ = [
    "MIN_P",
    "MIN_RT",
    "Wald",
    "Parameterization",
    "TrialDesign",
    "accumulators",
    "batching",
    "build_params_fn",
    "design",
    "censored_eval_rt",
    "enable_x64",
    "finalize_trial_logp",
    "gather_by_mask",
    "guard_positive",
    "hierarchical",
    "inference",
    "inv_gauss_logpdf",
    "inv_gauss_logsf",
    "log_floor",
    "numerics",
    "overlay_by_mask",
    "parameterization",
    "race",
    "race_from_arrays",
    "race_sample",
    "race_loglik",
    "simulate",
    "simulate_dataset",
    "simulate_race",
    "winner_mask",
]


def enable_x64():
    """Turn on JAX's 64-bit mode.

    Race log-densities are precision-sensitive: small squared terms, exp/log transforms of
    sub-unit values like `t0`, and survival functions evaluated far into the tail. Every
    consumer repo runs in float64, and the reference tests here assume it.

    This is a function rather than an import-time side effect on purpose. A library that
    mutates global JAX configuration when imported changes the numerics of unrelated code
    in the same process; `eam-abi-robustness/src/rdm_jax.py` does exactly that today, and
    a sibling module documents that it *depends* on the side effect. Call this explicitly
    from an entry point instead.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
