"""First-passage-time distributions for single racing accumulators."""

from .base import Accumulator, validate_params
from .diffusion import (
    SimulatedPulsedWald,
    first_passage_euler_maruyama,
    first_passage_from_mean,
)
from .pulse import normalized_gamma, normalized_gamma_derivative
from .volterra import VolterraPulsedWald, solve_volterra_fpt
from .lba import LBA, lba_logpdf, lba_logsf
from .wald import Wald, inv_gauss_logpdf, inv_gauss_logsf, wald_params

__all__ = [
    "LBA",
    "Accumulator",
    "SimulatedPulsedWald",
    "VolterraPulsedWald",
    "Wald",
    "inv_gauss_logpdf",
    "inv_gauss_logsf",
    "lba_logpdf",
    "lba_logsf",
    "normalized_gamma",
    "normalized_gamma_derivative",
    "solve_volterra_fpt",
    "first_passage_euler_maruyama",
    "first_passage_from_mean",
    "validate_params",
    "wald_params",
]


def __getattr__(name):
    # `EulerMaruyamaPulsedWald` was the name until the sampler stopped accumulating the
    # drift by Euler-Maruyama. A hint beats the bare `ImportError` a downstream repo would
    # otherwise get, since nothing about the model itself changed -- only the name.
    if name == "EulerMaruyamaPulsedWald":
        raise AttributeError(
            "EulerMaruyamaPulsedWald was renamed to SimulatedPulsedWald. It no longer "
            "accumulates the drift by Euler-Maruyama: the pulse's integrated drift is "
            "evaluated in closed form, which is exact at every grid point. Same "
            "parameters, same distribution, more accurate at a given `dt`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
