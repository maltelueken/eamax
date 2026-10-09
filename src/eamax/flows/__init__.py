"""Neural density estimation for accumulators with no closed form.

Requires the `flows` extra: `pip install 'eamax[flows]'`.
"""

from .checkpoint import (
    architecture_metadata,
    load_conditioner,
    read_metadata,
    save_conditioner,
    write_metadata,
)
from .model import (
    MIN_SCALE,
    MLP,
    DeepMLP,
    FlowAccumulator,
    conditioner_layout,
    context_bounds,
    evaluate_pdf_sf,
    input_scaling,
    log_input_scaling,
    make_mlp_conditioner,
    num_hidden_layers,
    spline_flow,
    spline_knots,
    spline_settings,
)
from .train import Maximum, eval_step, loss_fn, train_step

__all__ = [
    "MIN_SCALE",
    "MLP",
    "DeepMLP",
    "FlowAccumulator",
    "Maximum",
    "architecture_metadata",
    "conditioner_layout",
    "context_bounds",
    "eval_step",
    "evaluate_pdf_sf",
    "input_scaling",
    "load_conditioner",
    "log_input_scaling",
    "loss_fn",
    "make_mlp_conditioner",
    "num_hidden_layers",
    "read_metadata",
    "save_conditioner",
    "spline_flow",
    "spline_knots",
    "spline_settings",
    "train_step",
    "write_metadata",
]
