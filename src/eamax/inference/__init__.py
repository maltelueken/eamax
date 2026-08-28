"""Parameter estimation: starting values, warmup, NUTS, and tempered SMC.

Priors and log-densities stay with the model; this subpackage owns how a chain is started,
adapted, and run, and how the resulting draws are summarized.

Four seams:

* :mod:`~eamax.inference.transforms` -- unconstraining maps for
  ``[*bounded_block, *positive_block]`` vectors, with the matching log-det-Jacobian.
* :mod:`~eamax.inference.init` -- starting values: dispersion source x support constraint x
  exhaustion fallback. Rejection sampling against the ``t0 < min(rt)`` constraint rather
  than clipping, which would put a point mass exactly where dispersion is needed.
* :mod:`~eamax.inference.warmup` and :mod:`~eamax.inference.mcmc` -- window adaptation and
  the vmapped sampling loop. :func:`~eamax.inference.warmup.window_adaptation` adapts every
  chain independently, and is the only warm-up there is.
* :mod:`~eamax.inference.smc` -- adaptive tempered SMC with per-chain particle clouds and
  the log marginal likelihood, plus :mod:`~eamax.inference.diagnostics` for R-hat, unique
  particle counts and weight ESS.
* :mod:`~eamax.inference.posterior` -- pooling, thinning and the back-transform-then-select
  order, as plain array functions. They live here rather than in :mod:`eamax.io` because
  reading a file and deciding which fits to keep are different jobs: `eamax.io` reads and
  writes, and applies no diagnostic and no threshold of its own.

There is one prescribed path. Whatever the sampler, a fit starts by drawing dispersed
per-chain starts from the prior with :func:`~eamax.inference.init.init_positions_from_prior`
(NUTS) or :func:`~eamax.inference.init.init_particles_from_prior` (SMC), adapts **each**
chain with :func:`~eamax.inference.warmup.window_adaptation`, and samples with the tuning
that comes back, unmodified. Replicating one warmed state across chains, or sharing a single
particle cloud, leaves R-hat and the log-marginal-likelihood spread with nothing to measure,
so neither is offered.

Sampling needs BlackJAX (``pip install 'eamax[inference]'``) and the TFP peer, but importing
does not: no submodule imports BlackJAX at module scope, so ``transforms``, ``init`` and
``posterior`` are usable without it and only reaching for a sampler raises. Reading and
writing posterior files lives in :mod:`eamax.io`, behind its own extra.
"""

from . import diagnostics, init, mcmc, posterior, smc, transforms, warmup
from .diagnostics import count_unique_particles, rhat, weight_ess
from .init import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_T0_FRACTION,
    T0Support,
    init_particles_from_prior,
    init_position_from_mode,
    init_position_from_values,
    init_positions_from_prior,
    jitter_positions,
    min_valid_rt,
    rejection_sample,
)
from .mcmc import fit_nuts, fit_nuts_batch, inference_loop_multiple_chains
from .posterior import (
    back_transform_then_select,
    pool_chains,
    select_params,
    thin,
    to_arviz_layout,
)
from .smc import (
    DEFAULT_MAX_STEPS,
    SMCResult,
    smc_inference_loop,
    tempered_smc,
)
from .transforms import (
    SIMPLE,
    BlockTransform,
    simple_to_constrained,
    simple_to_unconstrained,
)
from .warmup import (
    window_adaptation,
)

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_T0_FRACTION",
    "SIMPLE",
    "BlockTransform",
    "SMCResult",
    "T0Support",
    "back_transform_then_select",
    "count_unique_particles",
    "diagnostics",
    "fit_nuts",
    "fit_nuts_batch",
    "inference_loop_multiple_chains",
    "init",
    "init_particles_from_prior",
    "init_position_from_mode",
    "init_position_from_values",
    "init_positions_from_prior",
    "jitter_positions",
    "mcmc",
    "min_valid_rt",
    "pool_chains",
    "posterior",
    "rejection_sample",
    "rhat",
    "select_params",
    "simple_to_constrained",
    "simple_to_unconstrained",
    "smc",
    "smc_inference_loop",
    "tempered_smc",
    "thin",
    "to_arviz_layout",
    "transforms",
    "warmup",
    "weight_ess",
    "window_adaptation",
]
