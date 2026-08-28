"""The Wald (inverse Gaussian) accumulator: a constant-drift diffusion to a fixed bound.

A diffusion with drift `v` and within-trial noise `s` started at zero first reaches a
boundary `b` at an inverse-Gaussian time with mean `mu = b / v` and shape
`lam = (b / s)^2`. Racing several of these is the racing diffusion model.

The two primitives here are validated against `scipy.stats.invgauss` and against the R
package EMC2 (see `tests/test_emc2_reference.py`).
"""

import jax.numpy as jnp
from jax.scipy import stats as jstats

from .._tfp import tfd
from ..numerics import MIN_P, guard_positive


def inv_gauss_logpdf(t, mu, lam):
    """Log density of the inverse Gaussian with mean `mu` and shape `lam`."""
    e = -(lam / (2 * t)) * (t**2 / mu**2 - 2 * t / mu + 1)
    return e + 0.5 * jnp.log(lam) - 0.5 * jnp.log(2 * t**3 * jnp.pi)


def inv_gauss_logsf(t, mu, lam):
    """Log survival function of the inverse Gaussian, for `t > 0`.

    Uses the stable form of Giner & Smyth (2016), *statmod: Probability Calculations for
    the Inverse Gaussian Distribution*, R Journal 8(1): it works in the `(mu/lam, t/lam)`
    parameterization and combines the two normal-CDF terms through `log1p`, so the far
    right tail does not catastrophically cancel.

    Hand-rolled rather than delegated to `tfd.InverseGaussian.log_survival_function`,
    which underflows to `-inf` for moderately large `t` -- that would zero out the
    gradient of every loser's "hadn't finished yet" term, which is most of a race
    likelihood.
    """
    mu = mu / lam
    t = t / lam
    r = 1.0 / jnp.sqrt(t)
    a = jstats.norm.logcdf(-r * (t / mu - 1.0))
    b = 2.0 / mu + jstats.norm.logcdf(-r * (t + mu) / mu)
    # b <= a always holds mathematically, but floating-point arithmetic can violate it;
    # clamp b - a <= 0 so the log1p argument never drops below -1 (-> NaN).
    return a + jnp.log1p(-jnp.exp(jnp.minimum(b - a, 0.0)))


def wald_params(v, s, b, min_param=MIN_P):
    """Map `(drift, noise, threshold)` to the inverse Gaussian's `(mu, lam)`."""
    v = guard_positive(v, min_param)
    s = guard_positive(s, min_param)
    b = guard_positive(b, min_param)
    return b / v, (b / s) ** 2


class Wald:
    """Constant-drift diffusion to a fixed boundary; first-passage time is Wald.

    Parameters
    ----------
    min_param : float, optional
        Floor applied to ``v``, ``s`` and ``b`` before they enter the closed forms, so an
        unconstrained sampler proposing a non-positive value gets a finite log-density
        rather than a NaN.
    """

    param_names = ("v", "s", "b")

    def __init__(self, min_param=MIN_P):
        self.min_param = min_param

    def log_pdf_sf(self, t, params):
        mu, lam = wald_params(params["v"], params["s"], params["b"], self.min_param)
        return inv_gauss_logpdf(t, mu, lam), inv_gauss_logsf(t, mu, lam)

    def sample(self, key, params):
        mu, lam = wald_params(params["v"], params["s"], params["b"], self.min_param)
        return tfd().InverseGaussian(mu, lam).sample(seed=key)

    def __repr__(self):
        return f"Wald(min_param={self.min_param!r})"
