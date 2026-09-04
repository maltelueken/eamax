"""Numerical first-passage density for a time-varying drift, via the Fortet equation.

A diffusion with time-varying drift has no closed-form first-passage density, but it does
satisfy a Volterra integral equation of the second kind (Fortet's equation; see Smith 2000
for the psychological application). Discretising it on a grid and substituting forward
gives the density directly:

    g(t_k) = phi(b, t_k | 0, 0) * (v(t_k) + (b - M(t_k)) / t_k)
             + 2 * dt * sum_{j<k} g(t_j) * psi(b, t_k | b, t_j)

where `M` is the integrated drift, `v` the instantaneous drift, and `psi` the flux across
the boundary from an earlier crossing.

This is the *reference* density, not the production one: the recursion is inherently
sequential and its kernel is `O(num_steps^2)`, which is far too slow to sit inside a
sampler. Its purpose is to validate faster approximations -- a trained neural flow, chiefly
-- and to check that the pulsed model reduces to the analytic Wald as `amp -> 0`.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy import stats

from ..numerics import MIN_P, guard_positive
from .pulse import DEFAULT_SHAPE, instantaneous_drift, integrated_drift, num_steps_for, time_grid

_TINY = 1e-30


@partial(jax.jit, static_argnames=["num_steps", "a_shape"])
def solve_volterra_fpt(v, amp, tau, s, b, dt, num_steps, a_shape=DEFAULT_SHAPE):
    """Solve Fortet's equation for the first-passage density and CDF on a grid.

    Parameters
    ----------
    v, amp, tau, s, b : float
        Scalar parameters. Vectorise with ``jax.vmap`` over parameter sets; the time
        recursion itself cannot be vectorised.
    dt : float
        Grid step.
    num_steps : int
        Number of grid points. Static -- it sets the kernel's shape.
    a_shape : float, optional
        Gamma shape of the conflict pulse.

    Returns
    -------
    g : array
        First-passage density on ``dt, ..., num_steps * dt``.
    G : array
        Cumulative distribution on the same grid.
    """
    t = time_grid(dt, num_steps)
    m = integrated_drift(t, v, amp, tau, a_shape)
    v_inst = instantaneous_drift(t, v, amp, tau, a_shape)

    sqrt_t = jnp.sqrt(t)
    # Free-diffusion density at the boundary, times the probability flux through it.
    phi_0 = stats.norm.pdf((b - m) / (s * sqrt_t)) / (s * sqrt_t)
    h0 = phi_0 * (v_inst + (b - m) / t)

    # Kernel: flux at t_n given an earlier crossing at t_j. Lower-triangular in effect --
    # `dt_diff` is clamped so the unused upper triangle stays finite rather than NaN.
    dt_diff = jnp.maximum(t[:, None] - t[None, :], 1e-10)
    m_diff = m[:, None] - m[None, :]
    phi = stats.norm.pdf(m_diff / (s * jnp.sqrt(dt_diff))) / (s * jnp.sqrt(dt_diff))
    psi = 0.5 * phi * (-v_inst[:, None] + m_diff / dt_diff)

    def step(g, n):
        # Only entries j < n are non-zero in `g` at this point, so the full dot product is
        # the triangular sum the equation asks for.
        value = jnp.maximum(h0[n] + 2.0 * dt * jnp.dot(g, psi[n]), 0.0)
        return g.at[n].set(value), value

    _, g = jax.lax.scan(step, jnp.zeros(num_steps), jnp.arange(num_steps))

    # Trapezoidal cumulative: the half-weight on the current point keeps G(t_1) unbiased.
    return g, dt * (jnp.cumsum(g) - 0.5 * g)


class VolterraPulsedWald:
    """Reference density for a pulsed accumulator, by numerical solution of Fortet's equation.

    Density only -- pair it with `SimulatedPulsedWald` to sample. Deliberately a
    separate class from a flow-based density rather than a "backend" of one: this is a
    deterministic solver with static discretisation knobs, a flow is a learned pytree passed
    by identity, and behind one constructor their static-argument requirements conflict.

    Cost is `O(num_steps^2)` per parameter set, sequentially. At `dt = 1e-3` and
    `t_max = 4` that is a 4000x4000 kernel per set; this is a validation tool, not a
    production likelihood.

    Parameters
    ----------
    dt : float
        Grid step.
    t_max : float
        Grid horizon. Decision times beyond it are extrapolated by ``jnp.interp``, which
        holds the endpoint -- so ``t_max`` must cover the data.
    a_shape : float, optional
        Gamma shape of the conflict pulse.
    min_param : float, optional
        Floor applied to ``tau``, ``s`` and ``b``.
    """

    param_names = ("v", "amp", "tau", "s", "b")

    def __init__(self, dt, t_max, a_shape=DEFAULT_SHAPE, min_param=MIN_P):
        self.dt = float(dt)
        self.t_max = float(t_max)
        self.a_shape = float(a_shape)
        self.min_param = min_param
        self.num_steps = num_steps_for(dt, t_max)

    def log_pdf_sf(self, t, params):
        flat = {
            name: jnp.ravel(jnp.broadcast_to(jnp.asarray(params[name]), jnp.shape(t)))
            for name in self.param_names
        }
        for name in ("tau", "s", "b"):
            flat[name] = guard_positive(flat[name], self.min_param)

        grid = time_grid(self.dt, self.num_steps)

        def one(v, amp, tau, s, b, decision_time):
            g, cdf = solve_volterra_fpt(
                v, amp, tau, s, b, self.dt, self.num_steps, self.a_shape
            )
            density = jnp.interp(decision_time, grid, g)
            cumulative = jnp.interp(decision_time, grid, cdf)
            return (
                jnp.log(jnp.maximum(density, _TINY)),
                jnp.log(jnp.maximum(1.0 - cumulative, _TINY)),
            )

        log_pdf, log_sf = jax.vmap(one)(
            flat["v"], flat["amp"], flat["tau"], flat["s"], flat["b"], jnp.ravel(t)
        )
        return log_pdf.reshape(jnp.shape(t)), log_sf.reshape(jnp.shape(t))

    def __repr__(self):
        return f"VolterraPulsedWald(dt={self.dt!r}, t_max={self.t_max!r})"
