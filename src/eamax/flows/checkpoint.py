"""Saving and restoring a trained conditioner, with the metadata Orbax does not keep.

Orbax restores by *structure*: the conditioner handed to `load_conditioner` supplies the
abstract state, and the checkpoint fills it in. Nothing on disk records `num_in`,
`num_mid`, `num_bins` or -- more dangerously -- which parameter each context column means.
A width mismatch surfaces as a shape error; a *reordered* context of the same width does
not surface at all, it just produces wrong densities.

So this module writes a small JSON sidecar next to the checkpoint recording all of that,
and `load_conditioner` checks it when present. Checkpoints written before the sidecar
existed still load; they simply skip the check.

`orbax.checkpoint` is imported lazily. It is slow to import and it reconfigures the root
logger on the way in, which a library has no business doing to a process that only wanted a
density function.
"""

import json
from pathlib import Path

import jax
from flax import nnx

SIDECAR_NAME = "eamax_conditioner.json"


def _orbax():
    import orbax.checkpoint as ocp

    return ocp


def write_metadata(path, context_names, num_mid=None, num_bins=None):
    """Record what a checkpoint's context columns mean, next to the checkpoint."""
    Path(path).mkdir(parents=True, exist_ok=True)
    (Path(path) / SIDECAR_NAME).write_text(
        json.dumps(
            {
                "context_names": list(context_names),
                "num_in": len(context_names),
                "num_mid": num_mid,
                "num_bins": num_bins,
            },
            indent=2,
        )
    )


def read_metadata(path):
    """Read a checkpoint's sidecar, or None if it has none."""
    sidecar = Path(path) / SIDECAR_NAME
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text())


def save_conditioner(conditioner, path, step=0, context_names=None, num_mid=None, num_bins=None):
    """Write the conditioner's NNX state to an Orbax checkpoint at `path`.

    The directory is erased and recreated first and only one step is kept: checkpoints are
    keyed by output directory, so rerunning a configuration replaces its predecessor rather
    than accumulating. Passing `context_names` also writes the sidecar, which is strongly
    recommended -- it is the only record of what the flow was conditioned on.

    Parameters
    ----------
    conditioner : MLP
        The module whose state is written.
    path : str or Path
        Checkpoint directory. Erased and recreated.
    step : int, optional
        Checkpoint step to write.
    context_names : sequence of str, optional
        Written to the sidecar. Strongly recommended.
    num_mid, num_bins : int, optional
        Written to the sidecar for reference.
    """
    ocp = _orbax()
    _, state = nnx.split(conditioner)

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    with ocp.CheckpointManager(ocp.test_utils.erase_and_create_empty(path), options=options) as mngr:
        mngr.save(step, args=ocp.args.StandardSave(state))
        mngr.wait_until_finished()

    if context_names is not None:
        write_metadata(path, context_names, num_mid=num_mid, num_bins=num_bins)


def load_conditioner(conditioner, path, step=0, context_names=None):
    """Restore checkpointed weights into a freshly built `conditioner`.

    Parameters
    ----------
    conditioner : MLP
        Used only for its structure, so it must have been built with the same ``num_in`` /
        ``num_mid`` / ``num_bins`` as the checkpoint.
    path : str or Path
        Checkpoint directory.
    step : int, optional
        Checkpoint step to restore.
    context_names : sequence of str, optional
        If given and the checkpoint has a sidecar, the two are compared. This is the check
        that turns a silently-wrong-density bug into an error.

    Returns
    -------
    MLP
        A new merged module; ``conditioner`` itself is not modified.

    Raises
    ------
    ValueError
        If ``context_names`` disagrees with the sidecar's recorded ordering.
    """
    ocp = _orbax()

    metadata = read_metadata(path)
    if context_names is not None and metadata is not None:
        recorded = list(metadata.get("context_names", []))
        if recorded and recorded != list(context_names):
            raise ValueError(
                f"Checkpoint at {path} was trained on context {recorded}, but this "
                f"conditioner is being used with {list(context_names)}. Matching widths "
                "with a different order produces silently wrong densities."
            )

    graphdef, abstract_state = nnx.split(nnx.eval_shape(lambda: conditioner))

    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ("x",)), jax.sharding.PartitionSpec()
    )
    abstract_state = jax.tree_util.tree_map(
        lambda x: x.update(sharding=sharding), abstract_state
    )

    with ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions()) as mngr:
        restored = mngr.restore(step, args=ocp.args.StandardRestore(abstract_state))
        mngr.wait_until_finished()

    return nnx.merge(graphdef, restored)
