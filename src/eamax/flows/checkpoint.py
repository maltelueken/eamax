"""Saving and restoring a trained conditioner, with the metadata Orbax does not keep.

Orbax restores by *structure*: the conditioner handed to `load_conditioner` supplies the
abstract state, and the checkpoint fills it in. Nothing on disk records `num_in`,
`num_mid`, `num_bins` or -- more dangerously -- which parameter each context column means.
A width mismatch surfaces as a shape error; a *reordered* context of the same width does
not surface at all, it just produces wrong densities.

So this module writes a small JSON sidecar next to the checkpoint recording all of that,
and `load_conditioner` checks it when present. Checkpoints written before the sidecar
existed still load; they simply skip the check.

The sidecar also records the conditioner's architecture settings that live in its graph
rather than its weights -- affine layout, spline settings, log-input scaling, depth and
training box (see :mod:`eamax.flows.model`). A layout or depth mismatch would fail inside
Orbax with an error naming no setting; a spline or scaling mismatch would not fail at all,
it would evaluate the same weights as a different density. A sidecar without these keys is
read as a plain one-layer flow with no recorded box.

`orbax.checkpoint` is imported lazily. It is slow to import and it reconfigures the root
logger on the way in, which a library has no business doing to a process that only wanted a
density function.
"""

import json
import math
from pathlib import Path

import jax
from flax import nnx

from .model import (
    DEFAULT_BOUNDARY_SLOPES,
    RANGE_MAX,
    conditioner_layout,
    context_bounds,
    input_scaling,
    num_hidden_layers,
    spline_settings,
)

SIDECAR_NAME = "eamax_conditioner.json"


def _orbax():
    import orbax.checkpoint as ocp

    return ocp


def architecture_metadata(conditioner):
    """The sidecar entries describing a conditioner's graph-level settings."""
    scaling = input_scaling(conditioner)
    bounds = context_bounds(conditioner)
    spline_range, boundary_slopes = spline_settings(conditioner)
    return {
        "affine": conditioner_layout(conditioner)[1],
        "spline_range": spline_range,
        "boundary_slopes": boundary_slopes,
        "input_scaling": None if scaling is None else {k: list(v) for k, v in scaling.items()},
        "num_hidden": num_hidden_layers(conditioner),
        "context_bounds": None if bounds is None else [list(pair) for pair in bounds],
    }


def write_metadata(path, context_names, num_mid=None, num_bins=None, architecture=None):
    """Record what a checkpoint's context columns mean, next to the checkpoint.

    `architecture` is merged in as-is; :func:`save_conditioner` passes
    :func:`architecture_metadata` of the conditioner it saves.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    metadata = {
        "context_names": list(context_names),
        "num_in": len(context_names),
        "num_mid": num_mid,
        "num_bins": num_bins,
    }
    metadata.update(architecture or {})
    (Path(path) / SIDECAR_NAME).write_text(json.dumps(metadata, indent=2))


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
        write_metadata(
            path, context_names, num_mid=num_mid, num_bins=num_bins,
            architecture=architecture_metadata(conditioner),
        )


def load_conditioner(conditioner, path, step=0, context_names=None):
    """Restore checkpointed weights into a freshly built `conditioner`.

    Parameters
    ----------
    conditioner : MLP
        Used only for its structure, so it must have been built with the same ``num_in`` /
        ``num_mid`` / ``num_bins`` and architecture settings as the checkpoint. If it records
        no training box and the sidecar does, the returned module adopts the sidecar's.
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
        If ``context_names`` disagrees with the sidecar's recorded ordering, or the
        conditioner's architecture settings or training box disagree with the sidecar's.
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
    adopted_bounds = None
    if metadata is not None:
        adopted_bounds = _check_architecture(conditioner, metadata, path)

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

    restored = nnx.merge(graphdef, restored)
    if adopted_bounds is not None:
        restored.context_bounds = adopted_bounds
    return restored


def _same_scaling(recorded, expected):
    if recorded is None or expected is None:
        return recorded is None and expected is None
    return all(
        len(recorded[key]) == len(expected[key])
        and all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
            for a, b in zip(recorded[key], expected[key], strict=True)
        )
        for key in ("eps", "loc", "scale")
    )


def _check_architecture(conditioner, metadata, path):
    """Refuse a template whose graph-level settings differ from the sidecar's.

    Returns the sidecar's training box when the template has none, for the caller to set on
    the restored module, and ``None`` otherwise.
    """
    recorded_depth = int(metadata.get("num_hidden", 1))
    if recorded_depth != num_hidden_layers(conditioner):
        raise ValueError(
            f"Checkpoint at {path} was trained with num_hidden={recorded_depth}, but the "
            f"conditioner it is being loaded into has {num_hidden_layers(conditioner)}."
        )
    recorded_affine = bool(metadata.get("affine", False))
    if recorded_affine != conditioner_layout(conditioner)[1]:
        raise ValueError(
            f"Checkpoint at {path} was trained with affine={recorded_affine}, but the "
            f"conditioner it is being loaded into has affine={conditioner_layout(conditioner)[1]}."
        )
    recorded_settings = (
        float(metadata.get("spline_range", RANGE_MAX)),
        str(metadata.get("boundary_slopes", DEFAULT_BOUNDARY_SLOPES)),
    )
    if recorded_settings != spline_settings(conditioner):
        raise ValueError(
            f"Checkpoint at {path} was trained with (spline_range, boundary_slopes) = "
            f"{recorded_settings}, but the conditioner it is being loaded into has "
            f"{spline_settings(conditioner)}."
        )
    recorded_scaling = metadata.get("input_scaling")
    if not _same_scaling(recorded_scaling, input_scaling(conditioner)):
        raise ValueError(
            f"Checkpoint at {path} was trained with input_scaling={recorded_scaling}, but the "
            f"conditioner it is being loaded into has {input_scaling(conditioner)}."
        )

    recorded_bounds = metadata.get("context_bounds")
    if recorded_bounds is None:
        return None
    recorded_bounds = tuple((float(low), float(high)) for low, high in recorded_bounds)
    expected_bounds = context_bounds(conditioner)
    if expected_bounds is None:
        return recorded_bounds
    if expected_bounds != recorded_bounds:
        raise ValueError(
            f"Checkpoint at {path} was trained on the box {recorded_bounds}, but the conditioner "
            f"it is being loaded into records {expected_bounds}."
        )
    return None
