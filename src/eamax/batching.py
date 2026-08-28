"""Utilities for running a model over many parameter sets or many trials at once.

`eamax` handles one dataset's worth of trials and leaves every outer axis to the caller, so
you are free to map over datasets, subjects or prior draws however suits your problem. This
module holds the one piece of batching machinery that is the same however you map: bounding
peak memory when the batch is large.
"""

import jax
import jax.numpy as jnp
import numpy as np


def map_in_chunks(fn, key, params, chunk_size):
    """Apply `fn(key, params_slice)` over slices of the flattened parameter axis.

    For work whose peak memory scales with the batch size, such as integrating a diffusion
    on a fine grid. Trades a little speed for a bounded memory footprint.

    Each chunk draws its own key, so changing ``chunk_size`` changes the draws but not their
    distribution.

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
