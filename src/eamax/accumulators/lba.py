"""The linear ballistic accumulator.

Each accumulator draws a start point uniformly on `[0, A]` and a drift rate from a
normal truncated to be positive, then travels deterministically to threshold `b`, so its
first-passage time is `(b - start) / drift`. Within-trial noise is replaced by
between-trial variability in both quantities, which is what makes the density closed-form.

For decision time `t`, drift mean `v`, drift SD `s`, start-point range `A` and threshold
`b`, with `z1 = (b - A - t*v)/(t*s)` and `z2 = (b - t*v)/(t*s)`:

    f(t) = (1/A) [ -v Phi(z1) + s phi(z1) + v Phi(z2) - s phi(z2) ]
    F(t) = 1 + ((b-A-t v)/A) Phi(z1) - ((b-t v)/A) Phi(z2)
             + (t s/A) phi(z1) - (t s/A) phi(z2)

both divided by `Phi(v/s)`, the probability that the truncated drift is positive -- i.e.
that the accumulator terminates at all.

The LBA's density is a difference of nearly equal terms that underflows to zero at small
`t`, where the closed form simply has no value to report. That is why `lba_logpdf` carries
a floor, unlike the other accumulators here: it marks the density as unrepresentable rather
than acting as a likelihood floor, so it is distinct from the race's trial floor in
`eamax.numerics`.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats as jstats

from .._tfp import tfd
from ..numerics import MIN_P, log_floor


def _lba_guard(t, s, sp_max, b, min_p):
    """Clamp inputs into the domain where the pdf/cdf formulae are valid.

    The closed forms take the raw threshold `b`, so they are callable with any `b`,
    including the degenerate `b < A` where the formula integrates untruncated drift mass
    below zero and can return a negative density. Capping `sp_max` just below `b` keeps
    them inside their support for arbitrary inputs. Model code passes `b = A + B` with
    `B > 0`, so the cap is inert on the real code path.
    """
    t = jnp.maximum(t, min_p)
    s = jnp.maximum(s, min_p)
    b = jnp.maximum(b, min_p)
    sp_max = jnp.maximum(jnp.minimum(sp_max, b - min_p), min_p)
    return t, s, sp_max, b


def _lba_log_truncation(v, s):
    """`log Phi(v/s)`: log probability that a truncated drift rate is positive.

    `norm.logcdf` rather than `log(norm.cdf(...))`, so this stays finite when `v/s` is very
    negative -- an accumulator that almost never terminates.
    """
    return jstats.norm.logcdf(v / s)


def lba_logpdf(t, v, s, sp_max, b, min_p=MIN_P):
    """Log density of one LBA accumulator's first passage at decision time `t`."""
    t, s, sp_max, b = _lba_guard(t, s, sp_max, b, min_p)

    z1 = (b - sp_max - t * v) / (t * s)
    z2 = (b - t * v) / (t * s)

    bracket = (
        -v * jstats.norm.cdf(z1)
        + s * jstats.norm.pdf(z1)
        + v * jstats.norm.cdf(z2)
        - s * jstats.norm.pdf(z2)
    )

    log_d = jnp.log(jnp.maximum(bracket, min_p)) - jnp.log(sp_max) - _lba_log_truncation(v, s)

    # What gets floored has to be the *density*, not the bracket. Clamping only the bracket
    # leaves `-log(sp_max) - log Phi(v/s)` to be applied afterwards, and those two terms are
    # usually positive, so they lift the clamped value back *above* the floor: at A=0.6,
    # b=1.8, v=3.5, s=1.2, t=0.05 that returns -22.51 where the true density is e^-146. So
    # substitute the floor outright wherever the bracket underflowed, and keep a plain lower
    # bound for the rarer `sp_max > 1` case where normalization pushes a non-underflowed
    # value below it instead.
    floor = log_floor(min_p)
    return jnp.where(bracket > min_p, jnp.maximum(log_d, floor), floor)


def lba_logsf(t, v, s, sp_max, b, min_p=MIN_P):
    """Log survival of one LBA accumulator: it has *not* finished by decision time `t`."""
    t, s, sp_max, b = _lba_guard(t, s, sp_max, b, min_p)

    z1 = (b - sp_max - t * v) / (t * s)
    z2 = (b - t * v) / (t * s)

    cdf = (
        1.0
        + ((b - sp_max - t * v) / sp_max) * jstats.norm.cdf(z1)
        - ((b - t * v) / sp_max) * jstats.norm.cdf(z2)
        + (t * s / sp_max) * jstats.norm.pdf(z1)
        - (t * s / sp_max) * jstats.norm.pdf(z2)
    )
    cdf = cdf / jnp.maximum(jnp.exp(_lba_log_truncation(v, s)), min_p)

    return jnp.log(jnp.clip(1.0 - cdf, min_p, 1.0))


class LBA:
    """Linear ballistic accumulator.

    Parameters are `v` (drift mean), `s` (drift SD), `A` (start-point range) and `b` (the
    absolute threshold). Note `b`, not the threshold *gap*: a model that parameterizes by
    the gap `B` -- keeping `b > A` true by construction and every parameter positive and
    log-transformable -- should form `b = A + B` in its parameterization, which is where
    that modelling choice belongs.
    """

    param_names = ("v", "s", "A", "b")

    def __init__(self, min_param=MIN_P):
        self.min_param = min_param

    def log_pdf_sf(self, t, params):
        v, s, sp_max, b = params["v"], params["s"], params["A"], params["b"]
        return (
            lba_logpdf(t, v, s, sp_max, b, self.min_param),
            lba_logsf(t, v, s, sp_max, b, self.min_param),
        )

    def sample(self, key, params):
        d = tfd()
        v, s, sp_max, b = (
            jnp.asarray(params["v"]),
            jnp.asarray(params["s"]),
            jnp.asarray(params["A"]),
            jnp.asarray(params["b"]),
        )
        shape = jnp.broadcast_shapes(v.shape, s.shape, sp_max.shape, b.shape)
        v, s, sp_max, b = (jnp.broadcast_to(x, shape) for x in (v, s, sp_max, b))

        key_start, key_drift = jax.random.split(key)
        start = d.Uniform(jnp.zeros_like(sp_max), sp_max).sample(seed=key_start)
        drift = d.TruncatedNormal(v, s, 0.0, jnp.inf).sample(seed=key_drift)
        return (b - start) / drift

    def __repr__(self):
        return f"LBA(min_param={self.min_param!r})"
