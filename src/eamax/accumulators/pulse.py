"""The conflict pulse: a transient, gamma-shaped addition to an accumulator's drift.

In conflict tasks (flanker, Simon, Stroop) an irrelevant stimulus feature is thought to
drive an early, automatic activation that decays as controlled processing takes over. The
diffusion model for conflict tasks (Ulrich et al., 2015) models that as a gamma-shaped
pulse added to one accumulator's drift:

    C(t) = amp * exp(-t/tau) * (e * t / ((a - 1) * tau))^(a - 1)     [integrated]
    g(t) = dC/dt = C(t) * ((a - 1)/t - 1/tau)                        [instantaneous]

`C(t)` is the *integrated* signal -- the accumulator's mean displacement -- and is what is
normalised, so `amp` is its peak height regardless of `tau`. That decoupling of amplitude
from time scale is what makes the two parameters separately interpretable and separately
priorable; parameterising the instantaneous drift instead would entangle them.

The effect is transient: `C` decays back to zero, so conflict perturbs *when* an
accumulator crosses, not where it ends up. `a` (the shape) is fixed at 2, which is what
makes the pulse rise-then-decay with a single time constant.

`g` is undefined at `t = 0`; the evaluation grid starts at `dt > 0`.
"""

import jax.numpy as jnp

#: Gamma shape. Fixed at 2: it is what makes the pulse a single rise-and-decay rather than
#: a family of shapes competing with `tau` for the same signal.
DEFAULT_SHAPE = 2.0


def normalized_gamma(t, amp, tau, a_shape=DEFAULT_SHAPE):
    """Integrated conflict drift `C(t)`, scaled so its peak value is `amp`."""
    return amp * jnp.exp(-t / tau) * (jnp.exp(1.0) * t / (a_shape - 1.0) / tau) ** (a_shape - 1.0)


def normalized_gamma_derivative(t, amp, tau, a_shape=DEFAULT_SHAPE):
    """Instantaneous conflict drift `g(t)`, the time derivative of `normalized_gamma`."""
    return normalized_gamma(t, amp, tau, a_shape) * ((a_shape - 1.0) / t - 1.0 / tau)


def integrated_drift(t, v, amp, tau, a_shape=DEFAULT_SHAPE):
    """Mean position `M(t)` of a pulsed accumulator started at zero."""
    return v * t + normalized_gamma(t, amp, tau, a_shape)


def instantaneous_drift(t, v, amp, tau, a_shape=DEFAULT_SHAPE):
    """Total drift rate `v(t)` of a pulsed accumulator."""
    return v + normalized_gamma_derivative(t, amp, tau, a_shape)


def time_grid(dt, num_steps):
    """Evaluation grid `dt, 2*dt, ..., num_steps*dt`, starting past the pulse's singularity."""
    return jnp.arange(1, num_steps + 1) * dt


def num_steps_for(dt, t_max):
    """Number of grid points covering `[dt, t_max]`. Static -- it sets array shapes."""
    return int(round(t_max / dt))
