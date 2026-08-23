"""Specs reproducing the intercept/slope parameterization the two-accumulator repos use.

`eam-abi-robustness` and `racing-diffusion-conflict` name their RDM parameters
`[v_intercept, v_slope, s_true, b, t0]`: accumulator 0 (the non-target) drifts at
`v_intercept` with its noise fixed to 1, and accumulator 1 (the target) drifts at
`v_intercept + v_slope` with free noise `s_true`.

Those names are load-bearing downstream -- they are Hydra config keys, MCMC parameter-name
contracts, neural-network inference-variable names, and the coordinate labels inside saved
posterior artifacts. Renaming them to the average/difference convention would break all of
that for no scientific gain, so the specs here keep them and express the relationship to
the general engine declaratively:

    V   = v_intercept + v_slope / 2
    v_d = v_slope

which is exact. The noise is *not* exact under the same translation, and that is a real
modelling difference rather than a naming one: this convention pins the mismatching
accumulator's noise to 1, whereas the average/difference convention pins the *average* to 1
and leaves the mismatching accumulator at `1 - s_d/2`. Hence `noise_reference="mismatch"`
here. See `eamax.design.map.accumulator_noise`.
"""

from .spec import ParamSpec

_LOG = "log"


def intercept_slope_spec(noise_scale=1.0):
    """The two-accumulator RDM as `[v_intercept, v_slope, s_true, b, t0]`.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification. 1 in
        both source repositories.

    Returns
    -------
    ParamSpec
        Whose ``names`` are exactly the two repos' parameter names, in their order (``t0``
        last, which several of their call sites depend on).
    """
    names = ("v_intercept", "v_slope", "s_true", "b", "t0")
    return ParamSpec(
        names=names,
        links=(_LOG,) * len(names),
        derived={
            "V": ((("v_intercept", 1.0), ("v_slope", 0.5)),) * 2,
            "v_d": ((("v_slope", 1.0),),) * 2,
            "S": ((("s_true", 1.0),),) * 2,
            "B": ((("b", 1.0),),) * 2,
        },
        num_responses=2,
        noise_reference="mismatch",
        noise_scale=noise_scale,
    )


def sat_spec(noise_scale=1.0):
    """The speed/accuracy variant: `[v_intercept, v_slope, s_true, b, b_diff, t0]`.

    The manipulation is a condition effect on the threshold under `"sum"` coding -- the
    accuracy-instructed threshold is `b + b_diff` rather than a second free parameter. That
    keeps `b_accuracy > b_speed` true by construction and every parameter positive and
    log-transformable, which is why `eam-abi-robustness` parameterizes it that way.

    The design's ``condition`` must be ``1`` for speed-instructed trials (the reference
    level) and ``0`` for accuracy-instructed ones.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification.

    Returns
    -------
    ParamSpec
    """
    names = ("v_intercept", "v_slope", "s_true", "b", "b_diff", "t0")
    return ParamSpec(
        names=names,
        links=(_LOG,) * len(names),
        derived={
            "V": ((("v_intercept", 1.0), ("v_slope", 0.5)),) * 2,
            "v_d": ((("v_slope", 1.0),),) * 2,
            "S": ((("s_true", 1.0),),) * 2,
            # Speed keeps `b`; accuracy adds the non-negative increment.
            "B": ((("b", 1.0),), (("b", 1.0), ("b_diff", 1.0))),
        },
        num_responses=2,
        noise_reference="mismatch",
        noise_scale=noise_scale,
    )


def lba_intercept_slope_spec(noise_scale=1.0):
    """The two-accumulator LBA as `[v_intercept, v_slope, s_true, A, B, t0]`.

    ``A`` is the start-point range and ``B`` the threshold *gap*, so the absolute boundary
    is ``A + B`` and ``b > A`` holds by construction -- the same trick :func:`sat_spec` uses
    for the speed/accuracy threshold, applied to a different invariant.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification.

    Returns
    -------
    ParamSpec
    """
    names = ("v_intercept", "v_slope", "s_true", "A", "B", "t0")
    return ParamSpec(
        names=names,
        links=(_LOG,) * len(names),
        derived={
            "V": ((("v_intercept", 1.0), ("v_slope", 0.5)),) * 2,
            "v_d": ((("v_slope", 1.0),),) * 2,
            "S": ((("s_true", 1.0),),) * 2,
            "B": ((("B", 1.0),),) * 2,
            "A": ((("A", 1.0),),) * 2,
        },
        num_responses=2,
        noise_reference="mismatch",
        noise_scale=noise_scale,
        threshold_is_gap=True,
    )
