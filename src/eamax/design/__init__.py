"""Parameterizations: how a flat parameter vector plus a trial design become accumulator
drifts, thresholds and noise.

Each per-accumulator quantity is a sum of contrast-weighted terms (see
:mod:`eamax.design.contrasts`), and each estimated coefficient carries a TensorFlow
Probability bijector as its link. Build one with :func:`parameterization`, or reach for a
ready-made :mod:`preset <eamax.design.presets>`; bind it to an accumulator with
:func:`build_params_fn`.
"""

from .contrasts import (
    Contrast,
    condition,
    constant_column,
    contrast_column,
    distractor,
    intercept,
    match,
    nontarget,
    response_offset,
    target,
)
from .effects import CODE_ORDER, CODE_TO_QTYPE, IDENTITY_QTYPES, effects_label, parse_effects
from .engine import accumulator_params, build_params_fn
from .links import bijector_for_link, identity, link_name, log
from .parameterization import (
    Const,
    Free,
    Parameterization,
    Quantity,
    Term,
    coef,
    constant,
    parameterization,
    quantity,
    term,
)
from .presets import (
    effects_spec,
    lba_intercept_slope_spec,
    lba_sat_spec,
    pulsed_conflict_spec,
    rdm_intercept_slope_spec,
    rdm_sat_spec,
)
from .trial import TrialDesign

__all__ = [
    "CODE_ORDER",
    "CODE_TO_QTYPE",
    "IDENTITY_QTYPES",
    "Const",
    "Contrast",
    "Free",
    "Parameterization",
    "Quantity",
    "Term",
    "TrialDesign",
    "accumulator_params",
    "bijector_for_link",
    "build_params_fn",
    "coef",
    "condition",
    "constant",
    "constant_column",
    "contrast_column",
    "distractor",
    "effects_label",
    "effects_spec",
    "identity",
    "intercept",
    "lba_intercept_slope_spec",
    "lba_sat_spec",
    "link_name",
    "log",
    "match",
    "nontarget",
    "parameterization",
    "parse_effects",
    "pulsed_conflict_spec",
    "quantity",
    "rdm_intercept_slope_spec",
    "rdm_sat_spec",
    "response_offset",
    "target",
    "term",
]
