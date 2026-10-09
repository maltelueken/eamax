# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions are
bumped with `bump-my-version`, which also turns the `[Unreleased]` heading into the release
heading.

## [Unreleased]

## [0.2.0] - 2026-10-09

### Added

- `eamax.flows`: optional per-context affine stage on log decision time
  (`make_mlp_conditioner(..., affine=True)`). The conditioner emits a location and scale per
  context, so the spline only models the shape of a standardised log decision time. Zeroed
  affine outputs reproduce the plain flow, and the survival function stays exact. Affine
  flows can also narrow `spline_range` and set `boundary_slopes`.
- `eamax.flows`: log-scaled conditioner inputs (`input_scaling`, affine only), with
  `log_input_scaling` computing the fixed location and scale in closed form from the uniform
  training box.
- `eamax.flows.DeepMLP` and `num_hidden` for conditioners with more than one hidden layer.
  `num_hidden=1` still builds a plain `MLP`, so existing checkpoints keep their structure.
- `eamax.flows`: conditioners can record their training box (`context_bounds`), and
  `FlowAccumulator` clips its inputs to that box (or to a `context_bounds` passed to it).
  This keeps stray SMC/NUTS particles off the flow's extrapolation region; inside the box the
  likelihood is unchanged.
- `eamax.flows.spline_knots` for inspecting where a flow's knots sit in log decision time.
- `eamax.flows.Maximum`, an `nnx` metric that tracks the running maximum of one argument.
- Helpers `architecture_metadata`, `conditioner_layout`, `context_bounds`, `input_scaling`,
  `num_hidden_layers`, `spline_settings` and the constant `MIN_SCALE`.
- This changelog, and `bump-my-version` configuration for releases.

### Changed

- `eamax.flows.train_step` now passes `grad_norm` (the raw global gradient norm, before any
  clipping) to `metrics.update` alongside `loss`.
- The checkpoint sidecar now records the conditioner's architecture settings (affine layout,
  spline settings, input scaling, depth, training box). `load_conditioner` raises
  `ValueError` when the template conditioner disagrees with them, instead of silently loading
  the weights as a different density. A template without a training box adopts the one in
  the sidecar. Sidecars written before this release load as a plain one-hidden-layer flow
  with no recorded box.

### Fixed

- `eamax.__version__` now matches the package version (it reported `0.1.0` in 0.1.1).

## [0.1.1] - 2026-09-13

### Changed

- Refactored `eamax.design` into parameterizations, presets, contrasts, links and an
  evaluation engine. The old `spec`, `map` and `legacy` modules are removed.
- Removed the invalid-RT penalty slope from the race likelihood. Infeasible trials (e.g.
  `t0` above the response time) now come out at the likelihood floor, and
  `race_from_arrays` no longer takes `rt_shifted` or `min_rt`.
- Renamed `EulerMaruyamaPulsedWald` to `SimulatedPulsedWald`. Importing the old name raises
  an error pointing to the new one.

### Added

- `eamax.accumulators.first_passage_from_mean`.
- `accumulator_params(..., broadcast=False)` keeps trial-invariant quantities at shape
  `(N, 1)`. Accumulators that set `broadcasts_params` (such as `FlowAccumulator`) receive
  them unexpanded and evaluate once per accumulator instead of once per trial.

### Fixed

- Integrated drift in the pulsed Wald diffusion: the sampler now integrates the pulse's
  drift exactly rather than by Euler-Maruyama.
- `FlowAccumulator` evaluated the flow once per trial instead of once per accumulator for
  trial-invariant parameters.
- Edge cases in design links and parameterizations, hierarchical priors, inference
  initialisation, MCMC/SMC, posterior handling and `eamax.io`.

## [0.1.0] - 2026-08-23

### Added

- Initial release: simulation and likelihoods for racing diffusion, Wald, LBA and pulsed
  Wald accumulators in JAX, with design/parameterization utilities, hierarchical priors,
  neural spline-flow density estimation (`eamax.flows`), NUTS and tempered SMC inference
  (`eamax.inference`), and NetCDF I/O (`eamax.io`).

[Unreleased]: https://github.com/maltelueken/eamax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/maltelueken/eamax/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/maltelueken/eamax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/maltelueken/eamax/releases/tag/v0.1.0
