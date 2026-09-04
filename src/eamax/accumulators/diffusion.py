"""Grid sampling for accumulators whose drift varies over time.

A pulsed accumulator has no closed-form first-passage density, so it is simulated by
walking `dx = v(t) dt + s dW` down a fixed grid and testing for boundary crossings.

The noise is additive and constant, so the path splits into a deterministic mean and a
scaled Brownian motion, `x(t) = M(t) + s W(t)`. When the integrated drift `M` is known in
closed form -- as it is for the conflict pulse -- the grid values of `x` are drawn from
their *exact* joint law and Euler-Maruyama's `O(dt)` drift error never arises.
`first_passage_from_mean` takes that mean path directly; `first_passage_euler_maruyama`
wraps it with a Riemann sum for drifts whose integral is not available.

What remains discretised is the crossing test, and two corrections make it far more
accurate than a naive one:

* **Brownian bridge.** Between two grid points that both sit below the boundary, the path
  may still have crossed. Conditional on its endpoints the increment is a Brownian bridge,
  whose maximum exceeds `b` with probability `exp(-2 (b - x_prev)(b - x) / (s^2 dt))`.
  Sampling that event recovers the crossings a grid test silently drops -- which are
  exactly the fast ones, so omitting it biases response times upward.
* **Within-step dequantisation.** The crossing time is `(index + u) * dt` with `u` uniform,
  rather than `index * dt`. Without it every simulated time is a multiple of `dt`, which is
  visible in any quantile-based summary and interacts badly with a floor on `rt - t0`.

A non-crossing accumulator returns `inf`, not a negative sentinel. That is what lets the
race take a plain `min` over accumulators and read an all-`inf` trial as right-censored,
with no special case.

Memory is the binding constraint: the mean path alone is `size * num_steps` floats, and
the crossing test needs several arrays that shape. `chunk_size` bounds peak memory by
mapping over slices instead, trading a little speed for a smaller footprint.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..batching import map_in_chunks
from ..numerics import MIN_P, guard_positive
from .pulse import DEFAULT_SHAPE, integrated_drift, num_steps_for, time_grid


def first_passage_from_mean(key, mean, s, b, dt):
    """First-passage times of diffusions with additive noise about a known mean path.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    mean : array
        Mean position `M(t)` on the grid, shape ``(..., num_steps)``. The path is
        ``M(t) + s W(t)``, so this is the accumulator's displacement in the absence of
        noise -- an *integrated* drift, not a rate.
    s : array
        Within-trial noise, shape ``(...)`` or broadcastable.
    b : array
        Absorbing boundary, shape ``(...)`` or broadcastable.
    dt : float
        Grid step.

    Returns
    -------
    array
        First-passage times, shape ``(...)``, with ``inf`` where the boundary was never
        reached.
    """
    s = jnp.broadcast_to(jnp.asarray(s)[..., None], mean.shape)
    b = jnp.broadcast_to(jnp.asarray(b)[..., None], mean.shape)

    key_noise, key_bridge, key_time = jax.random.split(key, 3)

    # Exact at the grid points: the increments of `s W` are iid normal, and `mean` carries
    # the whole drift contribution with no quadrature error of its own.
    noise = jnp.cumsum(jax.random.normal(key_noise, mean.shape) * jnp.sqrt(dt), axis=-1)
    x = mean + s * noise
    x_prev = jnp.concatenate([jnp.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)

    crossing = x >= b
    # Brownian-bridge correction: recover crossings that happened *between* grid points.
    p_touch = jnp.exp(-2.0 * (b - x_prev) * (b - x) / (s**2 * dt))
    touched = jax.random.uniform(key_bridge, x.shape) < p_touch
    crossing = crossing | (touched & (x_prev < b) & (x < b))

    first_index = jnp.argmax(crossing, axis=-1)
    offset = jax.random.uniform(key_time, first_index.shape)
    crossed = jnp.any(crossing, axis=-1)

    return jnp.where(crossed, (first_index + offset) * dt, jnp.inf)


def first_passage_euler_maruyama(key, drift, s, b, dt):
    """First-passage times of independent diffusions with time-varying drift.

    Euler-Maruyama: the mean path is accumulated as `cumsum(drift * dt)`, a Riemann sum
    carrying an `O(dt)` error. Prefer :func:`first_passage_from_mean` whenever the drift's
    integral is available in closed form -- for the conflict pulse it is, so
    :class:`SimulatedPulsedWald` uses that instead.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    drift : array
        Drift *rate* on the grid, shape ``(..., num_steps)``.
    s : array
        Within-trial noise, shape ``(...)`` or broadcastable.
    b : array
        Absorbing boundary, shape ``(...)`` or broadcastable.
    dt : float
        Grid step.

    Returns
    -------
    array
        First-passage times, shape ``(...)``, with ``inf`` where the boundary was never
        reached.
    """
    return first_passage_from_mean(key, jnp.cumsum(drift * dt, axis=-1), s, b, dt)


class SimulatedPulsedWald:
    """A diffusion whose drift carries a conflict pulse, sampled on a fixed grid.

    The pulse's integrated drift is known in closed form, so the grid values of the path
    are exact and only the crossing test is discretised.

    Sampling only: this class has no `log_pdf_sf`, because the model it samples from has no
    closed-form density. Pair it with `VolterraPulsedWald` for a numerical reference density
    or with `eamax.flows.FlowAccumulator` for a learned one -- and note that doing so breaks
    the "one object, one distribution" guarantee `eamax.simulate` otherwise gives you, since
    the density then comes from a different implementation than the draws.

    Parameters
    ----------
    dt : float
        Grid step. Smaller resolves the crossing test more finely and is proportionally
        more expensive.
    t_max : float
        Simulation horizon. Accumulators still running at ``t_max`` return ``inf``, and the
        race reads an all-``inf`` trial as right-censored -- so this must match the
        ``t_max`` passed to :func:`eamax.race.race_loglik`.
    a_shape : float, optional
        Gamma shape of the conflict pulse.
    min_param : float, optional
        Floor applied to ``tau``, ``s`` and ``b``.
    chunk_size : int, optional
        Maximum elements per integration call. Bounds peak memory, which is
        ``size * num_steps`` floats otherwise.
    """

    param_names = ("v", "amp", "tau", "s", "b")

    def __init__(self, dt, t_max, a_shape=DEFAULT_SHAPE, min_param=MIN_P, chunk_size=None):
        self.dt = float(dt)
        self.t_max = float(t_max)
        self.a_shape = float(a_shape)
        self.min_param = min_param
        self.chunk_size = chunk_size
        self.num_steps = num_steps_for(dt, t_max)

    def mean_grid(self, params):
        """Mean position `M(t)`: the constant drift's ramp plus the integrated pulse.

        Returns
        -------
        array
            Shape ``(..., num_steps)``.
        """
        t = time_grid(self.dt, self.num_steps)
        v = jnp.asarray(params["v"])[..., None]
        amp = jnp.asarray(params["amp"])[..., None]
        tau = guard_positive(jnp.asarray(params["tau"]), self.min_param)[..., None]
        return integrated_drift(t, v, amp, tau, self.a_shape)

    def _sample_flat(self, key, params):
        return first_passage_from_mean(
            key,
            self.mean_grid(params),
            guard_positive(params["s"], self.min_param),
            guard_positive(params["b"], self.min_param),
            self.dt,
        )

    def sample(self, key, params):
        params = {name: jnp.asarray(params[name]) for name in self.param_names}
        shape = jnp.broadcast_shapes(*(jnp.shape(value) for value in params.values()))
        params = {name: jnp.broadcast_to(value, shape) for name, value in params.items()}

        if self.chunk_size is None or int(np.prod(shape)) <= self.chunk_size:
            return self._sample_flat(key, params)
        return map_in_chunks(self._sample_flat, key, params, self.chunk_size)

    def __repr__(self):
        return (
            f"SimulatedPulsedWald(dt={self.dt!r}, t_max={self.t_max!r}, "
            f"chunk_size={self.chunk_size!r})"
        )
