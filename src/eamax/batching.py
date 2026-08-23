"""Utilities for running a model over many parameter sets or many trials at once.

`eamax` deliberately handles one dataset's worth of trials and leaves every outer axis to
the caller -- the three consumers batch over three different things (datasets, subjects,
prior draws), and supporting all of them inside would mean `in_axes` plumbing on every
parameter. What is worth sharing is the small amount of machinery that is the same however
you batch: making a sample shape static, and bounding peak memory when the batch is large.

`eam-abi-robustness`'s `batched_experiment` is *not* here. It is shaped around BayesFlow's
calling convention -- positional parameters splatted from a prior dict, a stateful
by-reference `rng`, a `{"x": ...}` return contract -- and only one consumer speaks that
protocol. It belongs in that repository's adapter layer alongside `SplittableKey`, for the
same reason.
"""

import jax
import jax.numpy as jnp
import numpy as np


def static_num_obs(num_obs):
    """Normalise a trial count to the single Python int a JAX sample shape needs.

    Trial counts arrive from config as ints, from a design simulator as 0-d arrays, and
    from batched keyword arguments as shape-`(1,)` or fully-populated arrays. A sample shape
    must be a concrete int, and a batch must agree on one value, so this collapses the
    former and rejects the latter.

    Parameters
    ----------
    num_obs : int or array
        Trial count, scalar or an array whose entries all agree.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``num_obs`` holds more than one distinct value.
    """
    array = np.asarray(num_obs)
    if array.ndim == 0:
        return int(array)

    unique = np.unique(array)
    if unique.size != 1:
        raise ValueError(
            f"num_obs must be a single value across a batch (a JAX sample shape is static); "
            f"got {unique.size} distinct values: {unique[:5]}..."
        )
    return int(unique[0])


def map_in_chunks(fn, key, params, chunk_size):
    """Apply `fn(key, params_slice)` over slices of the flattened parameter axis.

    For work whose peak memory scales with the batch size -- SDE integration on a fine grid,
    chiefly, where the drift array alone is `size * num_steps` floats. Trades a little speed
    for a bounded footprint.

    Padding to a whole number of chunks keeps every slice the same shape, so the body is
    traced once. Pad values are discarded. Each chunk draws its own key, so results are
    independent of the chunk size only up to the PRNG stream -- changing `chunk_size`
    changes the draws, though not their distribution.

    Parameters
    ----------
    fn : callable
        ``f(key, params) -> array`` with the same leading shape as its inputs.
    key : jax.Array
        PRNG key, split once per chunk.
    params : dict of array
        Arrays that already share a common shape.
    chunk_size : int
        Maximum number of elements per call.

    Returns
    -------
    array
        Same shape as the inputs.
    """
    shape = jnp.shape(next(iter(params.values())))
    size = int(np.prod(shape))

    flat = {name: jnp.reshape(value, (size,)) for name, value in params.items()}
    num_chunks = -(-size // chunk_size)
    padded = num_chunks * chunk_size
    flat = {
        name: jnp.concatenate([value, jnp.repeat(value[:1], padded - size)])
        for name, value in flat.items()
    }
    stacked = {name: jnp.reshape(value, (num_chunks, chunk_size)) for name, value in flat.items()}

    out = jax.lax.map(
        lambda arg: fn(arg[0], arg[1]), (jax.random.split(key, num_chunks), stacked)
    )
    return jnp.reshape(out, (padded,))[:size].reshape(shape)


def split_like(key, tree_shape):
    """One PRNG key per element of `tree_shape`, shaped like it.

    Simulating a batch of datasets wants one key per dataset so that results do not depend
    on how the batch was split up.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    tree_shape : tuple of int
        Target shape.

    Returns
    -------
    jax.Array
        Keys with shape ``tree_shape``.
    """
    shape = tuple(tree_shape)
    return jax.random.split(key, int(np.prod(shape))).reshape(shape)
