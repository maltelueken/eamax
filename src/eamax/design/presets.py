"""Ready-made parameterizations, as thin presets over the contrast engine.

Two families of models, expressed once. The parameter *names* these emit -- ``v_intercept``,
``v_slope``, ``s_true``, ``b``, ``b_diff``, ``A``, ``B``, ``t0`` for the two-accumulator
specs, and the ``V``/``v_d``/``B``/``b_d``/``S``/``s_d``/``c_*`` layout for the effects specs
-- are stable, so downstream config keys and saved coordinate labels stay fixed. The presets
express each model declaratively through :mod:`eamax.design.contrasts` rather than through a
hand-written map.

Two facts are choices of contrast rather than flags:

* The within-trial noise identification. The "average" convention frees the target-match
  difference around a fixed average, ``s = S*intercept + s_d*match(0.5)``. The "mismatch"
  convention -- what ``s_true`` means -- pins the non-target accumulator's noise to a
  constant and frees the target's: ``s = constant(scale)*nontarget() + s_true*target()``.
  Both are linear.
* The LBA threshold gap. ``b`` carries a ``transform`` reading the ``A`` quantity
  (``lambda q: q["A"] + q["b"]``), so the engine emits the absolute boundary ``A + B`` and
  ``b > A`` holds by construction.
"""

from .contrasts import condition, distractor, intercept, match, nontarget, response_offset, target
from .effects import IDENTITY_QTYPES, parse_effects
from .links import identity, log
from .parameterization import coef, constant, parameterization, quantity, term


def rdm_intercept_slope_spec(noise_scale=1.0):
    """The two-accumulator RDM as ``[v_intercept, v_slope, s_true, b, t0]``.

    Accumulator drift is ``v_intercept`` for the non-target and ``v_intercept + v_slope`` for
    the target; the non-target's noise is pinned to ``noise_scale`` and the target's is the
    free ``s_true`` (the "mismatch" identification). ``t0`` is last, which downstream code
    depends on.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification.

    Returns
    -------
    Parameterization
    """
    v_intercept = coef("v_intercept", log())
    v_slope = coef("v_slope", log())
    s_true = coef("s_true", log())
    b = coef("b", log())
    t0 = coef("t0", log())
    return parameterization(
        order=[v_intercept, v_slope, s_true, b, t0],
        quantities=[
            quantity("v", term(v_intercept, intercept()), term(v_slope, target())),
            quantity("s", term(constant(noise_scale), nontarget()), term(s_true, target())),
            quantity("b", term(b, intercept())),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=2,
    )


def rdm_sat_spec(noise_scale=1.0):
    """The speed/accuracy variant: ``[v_intercept, v_slope, s_true, b, b_diff, t0]``.

    The manipulation is a condition effect on the threshold: accuracy-instructed trials
    (``condition == 0``) get ``b + b_diff`` rather than a second free threshold. With
    ``b_diff`` on the log link the increment is non-negative, so ``b_accuracy > b_speed``
    holds by construction and every parameter stays positive and log-transformable.

    The design's ``condition`` must be ``1`` for speed-instructed trials (the reference
    level) and ``0`` for accuracy-instructed ones.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification.

    Returns
    -------
    Parameterization
    """
    v_intercept = coef("v_intercept", log())
    v_slope = coef("v_slope", log())
    s_true = coef("s_true", log())
    b = coef("b", log())
    b_diff = coef("b_diff", log())
    t0 = coef("t0", log())
    return parameterization(
        order=[v_intercept, v_slope, s_true, b, b_diff, t0],
        quantities=[
            quantity("v", term(v_intercept, intercept()), term(v_slope, target())),
            quantity("s", term(constant(noise_scale), nontarget()), term(s_true, target())),
            # Speed keeps `b`; accuracy (condition 0) adds the non-negative increment.
            quantity("b", term(b, intercept()), term(b_diff, condition(0))),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=2,
    )


def lba_intercept_slope_spec(noise_scale=1.0):
    """The two-accumulator LBA as ``[v_intercept, v_slope, s_true, A, B, t0]``.

    ``A`` is the start-point range and ``B`` the threshold *gap*, so the absolute boundary is
    ``A + B`` and ``b > A`` holds by construction -- the same trick
    :func:`rdm_sat_spec` uses for the speed/accuracy threshold, applied to a different
    invariant.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification.

    Returns
    -------
    Parameterization
    """
    v_intercept = coef("v_intercept", log())
    v_slope = coef("v_slope", log())
    s_true = coef("s_true", log())
    a = coef("A", log())
    b = coef("B", log())
    t0 = coef("t0", log())
    return parameterization(
        order=[v_intercept, v_slope, s_true, a, b, t0],
        quantities=[
            quantity("v", term(v_intercept, intercept()), term(v_slope, target())),
            quantity("s", term(constant(noise_scale), nontarget()), term(s_true, target())),
            quantity("A", term(a, intercept())),
            # Absolute boundary is A + B: `b` is the gap above the start point, expressed as a
            # transform of `A`, so `b > A` holds by construction.
            quantity("b", term(b, intercept()), transform=lambda q: q["A"] + q["b"]),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=2,
    )


def lba_sat_spec(noise_scale=1.0):
    """The LBA speed/accuracy variant: ``[v_intercept, v_slope, s_true, A, B, b_diff, t0]``.

    :func:`lba_intercept_slope_spec` with :func:`rdm_sat_spec`'s threshold manipulation: ``A``
    is the start-point range and the gap above it is ``B + b_diff`` on accuracy-instructed
    trials (``condition == 0``) and ``B`` on speed-instructed ones (``condition == 1``). With
    ``B`` and ``b_diff`` on the log link the gap stays positive and grows under accuracy, so
    both ``b > A`` and ``b_accuracy > b_speed`` hold by construction.

    The design's ``condition`` must be ``1`` for speed-instructed trials (the reference level)
    and ``0`` for accuracy-instructed ones.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification.

    Returns
    -------
    Parameterization
    """
    v_intercept = coef("v_intercept", log())
    v_slope = coef("v_slope", log())
    s_true = coef("s_true", log())
    a = coef("A", log())
    b = coef("B", log())
    b_diff = coef("b_diff", log())
    t0 = coef("t0", log())
    return parameterization(
        order=[v_intercept, v_slope, s_true, a, b, b_diff, t0],
        quantities=[
            quantity("v", term(v_intercept, intercept()), term(v_slope, target())),
            quantity("s", term(constant(noise_scale), nontarget()), term(s_true, target())),
            quantity("A", term(a, intercept())),
            # Gap above the start point is B (speed) or B + b_diff (accuracy); the absolute
            # boundary A + gap keeps b > A by construction.
            quantity(
                "b",
                term(b, intercept()),
                term(b_diff, condition(0)),
                transform=lambda q: q["A"] + q["b"],
            ),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=2,
    )


def effects_spec(effects, num_responses=2, *, fixed_noise=1.0):
    """The average/difference conflict model, from an effect-code string or quantity set.

    Racing accumulator models of conflict tasks decompose each per-accumulator quantity into
    an average over accumulators and a target-minus-others difference, then let some subset
    of those differ between conditions (congruent versus incongruent). Which subset is the
    model comparison, named by an effect-code string such as ``"VBs"`` (see
    :mod:`eamax.design.effects`).

    The within-trial noise uses the "average" identification: the average noise is fixed to
    ``fixed_noise`` unless ``"S"`` is among the effects, and the target-match difference
    ``s_d`` is free. Averages take the log link; differences (``v_d``, ``b_d``, ``s_d``) and
    the per-accumulator offsets take the identity link, because the whole point of estimating
    a difference is to learn its sign.

    The coefficient *order* is fixed -- ``V, v_d, b_d, c_1.., s_d, S?, B, t0`` -- so the
    trailing ``B``-and-``t0`` block is the one a semi-centered hierarchical prior centres;
    that is why ``num_centered`` is set here rather than left to the caller.

    Parameters
    ----------
    effects : str or iterable of str
        An effect-code string (``"VBs"``) or an iterable of quantity names
        (``{"V", "B", "s_d"}``). See :func:`eamax.design.effects.parse_effects`.
    num_responses : int, optional
        Number of accumulators.
    fixed_noise : float, optional
        The average noise, held fixed when ``"S"`` is not among the effects.

    Returns
    -------
    Parameterization
    """
    if isinstance(effects, str):
        effects = parse_effects(effects)
    effects = set(effects)

    order = []

    def make(name, qtype):
        bijector = identity() if qtype in IDENTITY_QTYPES else log()
        c = coef(name, bijector)
        order.append(c)
        return c

    def effect_terms(q, base):
        """Terms for quantity family `q` on base contrast `base`, appending its coefficients.

        A free condition effect splits into a reference-level and a non-reference-level term;
        with no effect it is one shared coefficient.
        """
        if q in effects:
            con = make(f"{q}_con", q)
            inc = make(f"{q}_inc", q)
            return [term(con, base * condition(1)), term(inc, base * condition(0))]
        return [term(make(q, q), base)]

    v_terms = effect_terms("V", intercept()) + effect_terms("v_d", match(0.5))
    b_terms = effect_terms("b_d", match(0.5))
    for r in range(1, num_responses):
        b_terms.append(term(make(f"c_{r}", "c"), response_offset(r, num_responses)))
    s_terms = effect_terms("s_d", match(0.5))
    if "S" in effects:
        s_terms += effect_terms("S", intercept())
    else:
        s_terms.append(term(constant(fixed_noise), intercept()))
    b_terms += effect_terms("B", intercept())
    t0 = make("t0", "t0")

    num_centered = (2 if "B" in effects else 1) + 1  # the B block plus t0

    return parameterization(
        order=order,
        quantities=[
            quantity("v", *v_terms),
            quantity("s", *s_terms),
            quantity("b", *b_terms),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=num_responses,
        num_centered=num_centered,
    )


def pulsed_conflict_spec(num_responses=2, *, fixed_noise=1.0):
    """A pulsed-conflict RDM: the conflict pulse rides only the distractor accumulator.

    ``[V, v_d, B, amp, tau, t0]``. The average/difference drift is as usual; ``amp`` -- the
    conflict pulse's peak height -- is routed by :func:`eamax.design.contrasts.distractor` to
    the accumulator the distractor selects and is zero on every other accumulator, while
    ``tau`` (the pulse's time constant) is shared. The within-trial noise is fixed to
    ``fixed_noise``.

    Pair it with a pulsed accumulator (:class:`eamax.accumulators.VolterraPulsedWald`,
    :class:`eamax.accumulators.SimulatedPulsedWald`).

    Returns
    -------
    Parameterization
    """
    V = coef("V", log())
    v_d = coef("v_d", identity())
    B = coef("B", log())
    amp = coef("amp", log())
    tau = coef("tau", log())
    t0 = coef("t0", log())
    return parameterization(
        order=[V, v_d, B, amp, tau, t0],
        quantities=[
            quantity("v", term(V, intercept()), term(v_d, match(0.5))),
            quantity("s", term(constant(fixed_noise), intercept())),
            quantity("b", term(B, intercept())),
            quantity("amp", term(amp, distractor())),
            quantity("tau", term(tau, intercept())),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=num_responses,
    )
