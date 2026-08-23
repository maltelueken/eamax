"""Condition-effect codes: which quantities are allowed to differ between conditions.

Racing accumulator models of conflict tasks decompose each per-accumulator quantity into
an *average* over accumulators and a *difference* between the one matching the target and
the ones that do not, then let some subset of those differ between conditions (congruent
versus incongruent, speed versus accuracy). Which subset is the model comparison.

Single-letter codes select the quantities; case distinguishes the average (uppercase) from
its difference (lowercase):

    V -> V,  v -> v_d,  B -> B,  b -> b_d,  S -> S,  s -> s_d

so `"VBs"` frees the average drift, the average threshold, and the noise difference.

The difference terms carry an identity link rather than an exp link. They are signed by
nature -- the whole point of estimating `b_d` is to find out whether the target-matching
accumulator has a *higher* or *lower* threshold -- and forcing them positive would decide
that in advance. The `+/-1` match sign convention is retained; making the differences
signed just lets the data pick the direction.
"""

CODE_TO_QTYPE = {"V": "V", "v": "v_d", "B": "B", "b": "b_d", "S": "S", "s": "s_d"}

#: Canonical order for labels, so a set of effects always spells the same filename.
CODE_ORDER = "VvBbSs"

#: Quantities on the identity (signed, natural-scale) link. Everything else uses `exp`.
IDENTITY_QTYPES = frozenset({"v_d", "b_d", "s_d", "c"})


def parse_effects(effects_str):
    """Parse a string of effect codes (e.g. `"VBs"`) into a set of quantity names."""
    qtypes = set()
    for char in effects_str:
        if char not in CODE_TO_QTYPE:
            raise ValueError(
                f"Unknown effect code {char!r}; valid codes are {''.join(CODE_TO_QTYPE)}"
            )
        qtypes.add(CODE_TO_QTYPE[char])
    return qtypes


def effects_label(effects):
    """Canonical filename/label for a set of effect quantities."""
    label = "".join(c for c in CODE_ORDER if CODE_TO_QTYPE[c] in effects)
    return label or "baseline"
