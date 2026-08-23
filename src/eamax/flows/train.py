r"""Training a conditioner: the right-censored objective and one optimiser step.

The training *loop* stays with the consumer -- it owns the simulator, the schedule and the
checkpoint policy. What belongs here is the objective, because getting it wrong is a
statistical error rather than a plumbing one.

A trial that never crosses within the simulator's horizon is not missing data: it is the
observation `T > t_max`, whose likelihood is the survival function. Dropping such trials --
or, equivalently, replacing their log-density with a constant, which zeroes their gradient
-- makes the flow fit `p(t | T < t_max) = p(t) / (1 - q)` instead of `p(t)`, where
`q = P(T > t_max)`. The fitted survival then decays to 0 rather than to `q`, and the density
is inflated by `-log(1 - q)` uniformly in `t`. Both errors depend on the parameters, so they
do not cancel out of a likelihood ratio and they bias the posterior.

How much censoring there is, measured in the source repository over its training prior
(`dt = 0.004`, `t_max = 4.0`, 2000 draws x 500 trials): 3.16% of trials overall, 25.5% of
draws losing at least one trial, 1.5% losing more than half. It is concentrated rather than
diffuse -- draws with `q > 0.5` have mean drift 0.22 against 4.95 for clean draws, i.e. the
slow-drift, high-boundary corner where the flow has the least data and the most distorted
target.

Scoring censored trials by `log S(t_max)` is the standard right-censored MLE and removes the
bias at no extra cost, since the flow's survival is exact given the flow.
"""

import jax.numpy as jnp
from flax import nnx
from jax.scipy import stats

from .model import spline_flow


def loss_fn(conditioner, data, context, t_max=None):
    """Mean right-censored negative log-likelihood of the first-passage time.

    Parameters
    ----------
    conditioner : MLP
        Differentiated against, so it stays the first positional argument.
    data : array
        Decision times; any shape that squeezes to the batch shape ``context`` implies.
        Non-crossing trials carry a non-positive or non-finite sentinel.
    context : array
        Conditioning parameters, natural scale.
    t_max : float, optional
        Horizon of the simulator that produced ``data``, or ``None`` for a simulator that
        cannot censor (an exact inverse-Gaussian draw, say). With no sentinels present the
        survival branch is never selected and the result is bit-identical to the uncensored
        loss.

    Returns
    -------
    array
        Scalar mean negative log-likelihood.
    """
    flat = jnp.squeeze(data)
    is_valid = jnp.isfinite(flat) & (flat > 0.0)

    # Both branches are evaluated, so the substituted value must stay inside the support or
    # the unused branch contributes NaN to the gradient.
    safe = jnp.where(is_valid, flat, 1.0 if t_max is None else t_max)

    log_pdf, flow = spline_flow(safe, context, conditioner)
    log_sf = stats.norm.logsf(flow.bijector.inverse(safe))

    return -jnp.mean(jnp.where(is_valid, log_pdf, log_sf))


@nnx.jit(static_argnames="t_max")
def train_step(conditioner, optimizer, metrics, data, context, t_max=None):
    """One optimiser step on `loss_fn`, updating `conditioner` in place.

    `metrics` accumulates but is never reset here -- the caller owns the averaging window
    and must call `metrics.reset()` after each `compute()` for a windowed mean.
    """
    loss, grads = nnx.value_and_grad(loss_fn)(conditioner, data, context, t_max)
    metrics.update(loss=loss)
    optimizer.update(conditioner, grads)


def eval_step(conditioner, metrics, data, context, t_max=None):
    """Score a batch without updating the conditioner."""
    metrics.update(loss=loss_fn(conditioner, data, context, t_max))
