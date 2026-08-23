"""`TrialDesign`: the per-trial covariates a parameterization reads.

A registered pytree, so a whole design can be passed through `jax.vmap` and `jax.jit`
without unpacking it field by field at every call site.

Every field is optional except the ones a given model actually reads, and all of them are
per-trial arrays of the same length -- there is no design *matrix* here. The three source
repositories all route covariates rather than regressing on them (accumulator identity
picks a sign, congruency picks between two parameters, the distractor picks which
accumulator carries a pulse), and inventing a matrix formalism none of them uses would add
machinery without adding reach.
"""

import jax
import jax.numpy as jnp


class TrialDesign:
    """Per-trial covariates for one dataset's worth of trials.

    Parameters
    ----------
    rt : array
        Response times, shape ``(T,)``. Negative marks a non-crossing trial.
    response : array, optional
        Winning accumulator per trial, shape ``(T,)``.
    target : array, optional
        The correct response per trial, shape ``(T,)``. Accumulator ``k`` is "matching" on
        trials where ``target == k``, which is the sign that splits every average into a
        match and a mismatch value.
    condition : array, optional
        ``1`` for the reference level (congruent, or speed-instructed), ``0`` otherwise.
        Defaults to all-reference, which makes every condition effect inert.
    distractor : array, optional
        The response the distracting stimulus feature is associated with, shape ``(T,)``.
        Selects the accumulator carrying a conflict pulse; exactly one per trial.
    mask : array, optional
        Boolean, shape ``(T,)``, False on padding. Masked trials contribute exactly zero.
    first_response : int, optional
        Index of the first accumulator in ``response``/``target``'s coding.
    """

    def __init__(
        self,
        rt,
        response=None,
        target=None,
        condition=None,
        distractor=None,
        mask=None,
        first_response=1,
    ):
        self.rt = jnp.asarray(rt)
        self.response = None if response is None else jnp.asarray(response)
        self.target = None if target is None else jnp.asarray(target)
        self.condition = jnp.ones_like(self.rt) if condition is None else jnp.asarray(condition)
        self.distractor = None if distractor is None else jnp.asarray(distractor)
        self.mask = None if mask is None else jnp.asarray(mask)
        self.first_response = first_response

    @classmethod
    def from_columns(cls, data, columns, mask=None, first_response=1):
        """Build from a `(T, C)` array plus the column names in order.

        The empirical datasets arrive as ``[rt, response, target, congruency, distractor]``;
        the two-accumulator simulators produce ``[rt, response]`` or
        ``[rt, response, condition]``. Naming the columns at the boundary keeps that
        difference out of the model code.

        Parameters
        ----------
        data : array
            Shape ``(T, C)``.
        columns : sequence of str
            Column names, in order.
        mask : array, optional
            Boolean, shape ``(T,)``.
        first_response : int, optional
            Index of the first accumulator in the response coding.

        Returns
        -------
        TrialDesign
        """
        data = jnp.asarray(data)
        fields = {name: data[:, i] for i, name in enumerate(columns)}
        return cls(
            rt=fields["rt"],
            response=fields.get("response"),
            target=fields.get("target"),
            condition=fields.get("condition", fields.get("congruency")),
            distractor=fields.get("distractor"),
            mask=mask,
            first_response=first_response,
        )

    def replace(self, **changes):
        current = {
            "rt": self.rt,
            "response": self.response,
            "target": self.target,
            "condition": self.condition,
            "distractor": self.distractor,
            "mask": self.mask,
            "first_response": self.first_response,
        }
        return TrialDesign(**{**current, **changes})

    def __repr__(self):
        present = [
            name
            for name in ("response", "target", "condition", "distractor", "mask")
            if getattr(self, name) is not None
        ]
        return f"TrialDesign(num_trials={self.rt.shape[-1]}, fields={present})"


_ARRAY_FIELDS = ("rt", "response", "target", "condition", "distractor", "mask")


def _flatten(design):
    children = tuple(getattr(design, name) for name in _ARRAY_FIELDS)
    return children, design.first_response


def _unflatten(first_response, children):
    design = object.__new__(TrialDesign)
    for name, value in zip(_ARRAY_FIELDS, children):
        setattr(design, name, value)
    design.first_response = first_response
    return design


jax.tree_util.register_pytree_node(TrialDesign, _flatten, _unflatten)
