"""Starting values: where a sampler begins, and why it matters that it begins in support.

A starting rule is three independent choices:

    (source of dispersion) x (support constraint) x (what to do when rejection gives up)

This module supplies all three, so the choice is explicit.

**The constraint.** For a race model the likelihood is undefined at ``t0 >= min(rt)``, and
`eamax.race` scores every trial in that region at the floor. The floor is flat, so a chain
that *starts* there starts on a plateau: the gradient says nothing about which way `t0`
should move, and window adaptation cannot help, because flatness is not a curvature scale.
Nothing in the likelihood pushes such a chain back down. Dispersion has to be support-aware
to be worth anything.

**Rejection, not clipping.** Capping the offending coordinate looks cheaper and is worse: a
cap is a point mass, and it lands in the one coordinate the routine exists to keep dispersed.
Rejecting whole draws instead yields exactly the source distribution conditioned on the
constraint.

Everything here speaks **unconstrained** coordinates, the ones a sampler explores. A prior
that draws on the natural scale is composed with a transform by the caller; see
:func:`init_positions_from_prior`.

This module needs no sampler; it imports nothing from `_blackjax`.
"""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

#: Highest fraction of the fastest observed response time that a starting `t0` may take.
#: Close to 1 on purpose: the true `t0` often sits just below `min_rt`, so a low ceiling
#: would put every chain below the truth and leave window adaptation to walk them all up.
DEFAULT_MAX_T0_FRACTION = 0.97

#: Redraws before a rejection sampler gives up on one draw.
DEFAULT_MAX_ATTEMPTS = 2000


def min_valid_rt(rt, *, mask=None, sentinel=0.0, axis=-1):
    """Smallest response time that is a real observation.

    The non-crossing sentinel is excluded before taking the minimum, so a censored trial
    never masquerades as the fastest response.

    Parameters
    ----------
    rt : array
        Response times, shape ``(T,)`` for one subject or ``(S, T)`` for several.
    mask : array, optional
        Boolean of ``rt``'s shape; ``False`` marks padding, which is excluded.
    sentinel : float, optional
        Entries at or below this mark a trial that never crossed -- an observation of
        ``T > t_max``, not a fast response. `eamax.simulate` writes ``-1.0``.
    axis : int, optional
        Trial axis.

    Returns
    -------
    array
        Scalar for a ``(T,)`` input, shape ``(S,)`` for ``(S, T)``.

    Raises
    ------
    ValueError
        If any subject has no valid observation at all -- every trial censored, or masked
        out as padding. The minimum over an empty set is ``inf``, which is the identity
        element of ``min`` rather than an answer, and it propagates silently: an infinite
        ``log_t0_max`` in :meth:`T0Support.from_spec` disables the constraint for exactly
        the subject whose data cannot support it. Checked only when ``rt`` is concrete;
        under tracing there is nothing to inspect.
    """
    rt = jnp.asarray(rt)
    valid = rt > sentinel
    if mask is not None:
        valid = valid & jnp.asarray(mask)

    smallest = jnp.min(jnp.where(valid, rt, jnp.inf), axis=axis)

    try:
        empty = ~jnp.isfinite(smallest)
        any_empty = bool(jnp.any(empty))
    except jax.errors.TracerBoolConversionError:  # pragma: no cover - traced call
        return smallest

    if any_empty:
        where = jnp.argwhere(jnp.atleast_1d(empty)).ravel().tolist()
        raise ValueError(
            f"No valid response time for {len(where)} of {jnp.atleast_1d(smallest).size} "
            f"entries (index {where}): every trial is at or below the non-crossing sentinel "
            f"{sentinel}, or masked out. The minimum of an empty set is inf, which would "
            "silently disable the t0 support constraint for those subjects. Drop them, or "
            "pass a min_rt of your own."
        )

    return smallest


@dataclass(frozen=True)
class T0Support:
    """The `t0 < min(rt)` constraint, as a predicate on unconstrained parameters.

    Attributes
    ----------
    index : int
        Position of ``t0`` in the parameter vector.
    log_t0_max : array
        ``log(max_fraction * min_rt)``. Scalar for one subject, shape ``(S,)`` for several.
    """

    index: int
    log_t0_max: jnp.ndarray

    @classmethod
    def from_spec(cls, spec, min_rt, *, max_fraction=DEFAULT_MAX_T0_FRACTION):
        """Build from a :class:`eamax.design.Parameterization` and the fastest observed RT.

        ``t0`` is located by name from the spec rather than assumed to be the last entry, so
        a spec that orders its parameters differently raises instead of silently clipping the
        wrong one.

        Parameters
        ----------
        spec : Parameterization
            Supplies ``t0``'s index and confirms its link.
        min_rt : float or array
            Fastest valid response time; scalar, or shape ``(S,)`` per subject. See
            :func:`min_valid_rt`.
        max_fraction : float, optional
            See :data:`DEFAULT_MAX_T0_FRACTION`.

        Returns
        -------
        T0Support

        Raises
        ------
        ValueError
            If ``spec`` has no ``t0``, or ``t0`` is not on the log link -- the comparison
            is against an unconstrained value, which is only ``log(t0)`` under that link.
        """
        if "t0" not in spec.names:
            raise ValueError(
                f"The t0 support constraint needs a parameter named 't0'; this spec has "
                f"{list(spec.names)}. Build a T0Support directly if it is named otherwise."
            )

        index = spec.index("t0")
        if spec.links[index] != "log":
            raise ValueError(
                f"t0 must be on the log ('Exp') link for this constraint, but it is on "
                f"'{spec.links[index]}'. The cap compares against an unconstrained value, "
                "which is only log(t0) under the log link."
            )

        return cls(index=index, log_t0_max=jnp.log(max_fraction * jnp.asarray(min_rt)))

    def _values(self, theta):
        return jnp.asarray(theta)[..., self.index]

    def _cap_for(self, values):
        """``log_t0_max`` reduced to something that broadcasts against `values`.

        A per-subject cap lines up entry by entry with the ``(S,)`` column of an ``(S, P)``
        draw. Against a single ``(P,)`` vector there is one ``t0`` for every bound, and
        :meth:`violates` treats that as breaching unless it clears them all, so the binding
        bound -- the one a clip has to respect -- is the tightest.
        """
        cap = jnp.asarray(self.log_t0_max)
        return jnp.min(cap) if cap.ndim > jnp.ndim(values) else cap

    def violates_t0(self, t0):
        """Does any of these unconstrained ``t0`` values breach the cap?

        Split out from :meth:`violates` so a caller that already holds the ``t0`` column --
        :func:`init_particles_from_prior` reads it straight off the centered block -- can
        test it without reconstructing a whole parameter vector to index into.
        """
        return jnp.any(jnp.asarray(t0) > self.log_t0_max)

    def violates(self, theta):
        """Does any entry breach the cap?

        Parameters
        ----------
        theta : array
            Unconstrained parameters, shape ``(P,)`` for one subject or ``(S, P)``. A
            ``(S, P)`` input with per-subject bounds is checked subject by subject: it
            takes only one of them to ruin the draw.

        Returns
        -------
        array
            Scalar boolean.
        """
        return self.violates_t0(self._values(theta))

    def clip(self, theta):
        """Pin breaching entries to the cap.

        The exhaustion fallback, not a strategy -- see this module's docstring for what a
        cap does to dispersion when it binds often.

        Accepts the same shapes as :meth:`violates`: a ``(P,)`` vector or an ``(S, P)``
        block, against a scalar or per-subject cap.
        """
        theta = jnp.asarray(theta)
        values = self._values(theta)
        return theta.at[..., self.index].set(jnp.minimum(values, self._cap_for(values)))


def rejection_sample(sample_fn, violates_fn, num_draws, key, *,
                     max_attempts=DEFAULT_MAX_ATTEMPTS, repair_fn=None):
    """Draw until the constraint is satisfied, independently per draw.

    Parameters
    ----------
    sample_fn : callable
        ``f(key) -> pytree``, one draw.
    violates_fn : callable
        ``f(pytree) -> bool``, a traceable scalar predicate.
    num_draws : int
        Leading axis of the result.
    key : jax.Array
        PRNG key, split once per draw.
    max_attempts : int, optional
        Redraws before giving up on one draw.
    repair_fn : callable, optional
        ``f(pytree) -> pytree``, applied to a draw that gave up, and *only* to such a draw:
        it is evaluated on every draw so the selection stays traceable, but its result is
        kept only where ``max_attempts`` ran out. That is what leaves the accepted draws
        exactly the source distribution conditioned on the constraint, rather than the
        source distribution pushed through a projection. When ``None`` an exhausted draw is
        returned as drawn, which is the honest default -- not every constraint has a
        sensible projection.

    Returns
    -------
    draws : pytree
        Leading axis ``num_draws``.
    num_exhausted : array
        Scalar count of draws that hit ``max_attempts``. Report it; it should be 0.
    """

    def draw_one(draw_key):
        first_key, loop_key = jax.random.split(draw_key)

        def cond(carry):
            attempt, sample, _ = carry
            return violates_fn(sample) & (attempt < max_attempts)

        def body(carry):
            attempt, _, current = carry
            current, next_key = jax.random.split(current)
            return attempt + 1, sample_fn(next_key), current

        _, sample, _ = jax.lax.while_loop(
            cond, body, (0, sample_fn(first_key), loop_key)
        )

        exhausted = violates_fn(sample)
        if repair_fn is not None:
            # `exhausted` is a traced scalar, so the repair cannot be branched on; select
            # instead. Applying it unconditionally would push *every* accepted draw through
            # the projection, which is exactly the point mass this module exists to avoid.
            sample = jax.tree.map(
                lambda repaired, drawn: jnp.where(exhausted, repaired, drawn),
                repair_fn(sample),
                sample,
            )
        return sample, exhausted

    draws, exhausted = jax.vmap(draw_one)(jax.random.split(key, num_draws))
    return draws, jnp.sum(exhausted)


def init_positions_from_prior(sample_fn, num_chains, key, *, support=None,
                              max_attempts=DEFAULT_MAX_ATTEMPTS):
    """One unconstrained starting position per chain, drawn from the prior.

    The prior supplies the dispersion, which is what makes R-hat measure something. It is
    also already the right scale, unlike an arbitrary jitter width.

    Parameters
    ----------
    sample_fn : callable
        ``f(key) -> (P,)`` **unconstrained** draw. A prior that draws on the natural scale
        composes with a transform here, e.g.
        ``lambda k: jnp.log(jnp.stack(prior.sample(seed=k)))``.
    num_chains : int
    key : jax.Array
    support : T0Support, optional
        When given, draws are rejected and redrawn until they satisfy it.
    max_attempts : int, optional

    Returns
    -------
    positions : array
        Shape ``(num_chains, P)``.
    num_exhausted : array
        Scalar; 0 when no draw hit ``max_attempts``.
    """
    if support is None:
        keys = jax.random.split(key, num_chains)
        return jax.vmap(sample_fn)(keys), jnp.asarray(0)

    return rejection_sample(
        sample_fn, support.violates, num_chains, key,
        max_attempts=max_attempts, repair_fn=support.clip,
    )


def init_particles_from_prior(flat_space, num_particles, key, *, support=None,
                              max_attempts=DEFAULT_MAX_ATTEMPTS):
    """Prior particles in flat unconstrained coordinates.

    The constraint is tested on the *reconstructed* ``(S, P)`` subject parameters rather
    than on a raw component of the prior's dict, so it does not require ``t0`` to sit in any
    particular block.

    The accepted region is where the posterior lives -- outside it the likelihood is the
    floor, not a density -- so restricting the cloud costs no posterior mass. It is
    nonetheless **not** the prior, and that matters: tempered SMC weights increments by the
    likelihood alone and never corrects the initial distribution. Store it as an
    initialisation artefact, not under a ``prior`` group, or anything measuring posterior
    contraction against it will be wrong.

    Parameters
    ----------
    flat_space : HierarchicalFlatSpace
    num_particles : int
    key : jax.Array
    support : T0Support, optional
        Its ``log_t0_max`` is normally per-subject, shape ``(S,)``.
    max_attempts : int, optional

    Returns
    -------
    particles : array
        Shape ``(num_particles, D)``.
    num_exhausted : array
        Scalar.
    """
    if support is None:
        return flat_space.sample_particles(key, num_particles), jnp.asarray(0)

    num_ncp = flat_space.prior.num_params_ncp

    def violates(flat):
        # The centered block passes through the reconstruction unchanged and sits in the
        # trailing columns of the subject parameters, so when `t0` lives there the cap can be
        # tested against that one column. Reading it off the flat vector skips a full
        # unravel, the bijector forward (a `CorrelationCholesky` on a P x P factor among
        # them) and the (S, P) reconstruction -- per attempt, inside a vmapped while_loop.
        if support.index >= num_ncp:
            return support.violates_t0(
                flat_space.centered_block(flat)[:, support.index - num_ncp]
            )
        return support.violates(flat_space.subject_params(flat))

    def repair(flat):
        # Only the centered block passes through to the subject parameters unchanged, so
        # only there can a breach be projected out without inverting the reconstruction.
        # Elsewhere an exhausted draw is returned as drawn and counted; the count is the
        # signal that the budget, or the cap, needs revisiting.
        if support.index < num_ncp:
            return flat
        unconstrained = flat_space.unravel(flat)
        column = support.index - num_ncp
        centered = unconstrained["theta_bt"]
        capped = centered.at[:, column].set(
            jnp.minimum(centered[:, column], support.log_t0_max)
        )
        return flat_space.ravel({**unconstrained, "theta_bt": capped})

    return rejection_sample(
        flat_space.sample, violates, num_particles, key,
        max_attempts=max_attempts, repair_fn=repair,
    )


def init_position_from_mode(flat_space, *, num_chains=None, key=None, jitter=0.0,
                            support=None, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """The prior's mode, optionally jittered into several dispersed starts.

    With the defaults this is one deterministic point with no dispersion and no support
    check. That is reproducible, but it is worth knowing what it costs: every chain begins at
    the same place, so between-chain variance starts at zero and any diagnostic built on it
    reports Monte-Carlo noise.

    Parameters
    ----------
    flat_space : HierarchicalFlatSpace
    num_chains : int, optional
        Without it, returns a single ``(D,)`` position.
    key : jax.Array, optional
        Required when ``jitter`` is nonzero.
    jitter : float, optional
        Standard deviation of a Gaussian perturbation, in unconstrained units.
    support : T0Support, optional
    max_attempts : int, optional

    Returns
    -------
    positions : array
        Shape ``(D,)``, or ``(num_chains, D)`` when ``num_chains`` is given.
    num_exhausted : array
        Scalar; only meaningful when jittering against a ``support``.

    Raises
    ------
    ValueError
        If ``jitter`` is nonzero and ``key`` is ``None``, or ``num_chains`` is ``None``.
    """
    mode = flat_space.mode()

    if num_chains is None:
        if jitter:
            raise ValueError("Jittering needs num_chains; one position cannot be dispersed.")
        return mode, jnp.asarray(0)

    if not jitter:
        return jnp.broadcast_to(mode, (num_chains, mode.size)), jnp.asarray(0)

    if key is None:
        raise ValueError("Jittering needs a PRNG key.")

    return jitter_positions(
        mode, num_chains, key, scale=jitter, support=support, max_attempts=max_attempts
    )


def init_position_from_values(values, *, spec=None, t0_index=None, offset=0, min_rt=None,
                              t0_fraction=0.5, transform=None):
    """A fixed natural-scale starting vector, with `t0` optionally derived from the data.

    The ``t0`` entry is located by name and set to a fraction of the fastest observed
    response time.

    This is the one builder with no dispersion at all. It is a poor way to start several
    chains -- prefer :func:`init_positions_from_prior` -- and is kept only for callers that
    need a fixed starting vector.

    Parameters
    ----------
    values : sequence of float
        Natural-scale starting values, the full vector including any leading bounded block.
        See :meth:`eamax.inference.transforms.BlockTransform.midpoint` for that block.
    spec : Parameterization, optional
        Supplies ``t0``'s index, offset by ``offset``.
    t0_index : int, optional
        Given directly instead of via ``spec``.
    offset : int, optional
        Length of any leading block that ``spec`` does not describe.
    min_rt : float, optional
        When given, ``t0`` is set to ``t0_fraction * min_rt``. See :func:`min_valid_rt`.
    t0_fraction : float, optional
    transform : BlockTransform, optional
        When given, the result is returned unconstrained.

    Returns
    -------
    position : array
        Shape ``(P,)``, natural scale unless ``transform`` is given.
    num_exhausted : array
        Always scalar 0. There is no rejection here -- nothing is drawn -- but every
        builder in this module returns ``(positions, num_exhausted)``, and a lone builder
        returning a bare array unpacks into the first two *coordinates* of the starting
        vector for a caller following that pattern, silently and without raising.

    Raises
    ------
    ValueError
        If ``min_rt`` is given but ``t0``'s index cannot be resolved.
    """
    position = jnp.asarray(values, dtype=float)

    if min_rt is not None:
        if t0_index is None:
            if spec is None:
                raise ValueError("Deriving t0 from min_rt needs either `spec` or `t0_index`.")
            if "t0" not in spec.names:
                raise ValueError(
                    f"Deriving t0 from min_rt needs a parameter named 't0'; this spec has "
                    f"{list(spec.names)}."
                )
            t0_index = offset + spec.index("t0")
        position = position.at[t0_index].set(t0_fraction * jnp.asarray(min_rt))

    if transform is not None:
        position = transform.inverse(position)

    return position, jnp.asarray(0)


def jitter_positions(position, num_chains, key, *, scale=0.1, support=None,
                     max_attempts=DEFAULT_MAX_ATTEMPTS):
    """Disperse one unconstrained position into `num_chains` starts.

    The route for a caller with no prior object to hand. An arbitrary width stands in for
    the prior's own scale, so this is weaker than drawing from the prior, but it is enough to
    give R-hat something to measure, which a replicated position is not.

    ``support`` is not really optional: undirected jitter easily lands a chain on the
    likelihood's flat floor, so omitting it while jittering warns.

    Parameters
    ----------
    position : array
        Unconstrained, shape ``(P,)``.
    num_chains : int
    key : jax.Array
    scale : float, optional
        Gaussian standard deviation in unconstrained units.
    support : T0Support, optional
    max_attempts : int, optional

    Returns
    -------
    positions : array
        Shape ``(num_chains, P)``.
    num_exhausted : array
        Scalar.
    """
    position = jnp.asarray(position)

    if support is None and scale:
        warnings.warn(
            "Jittering without a support constraint: a start with t0 above the fastest "
            "response time lands on the likelihood's flat floor, where the gradient carries "
            "no information and step-size adaptation cannot recover. Pass "
            "`support=T0Support.from_spec(...)`.",
            stacklevel=2,
        )

    def sample_fn(draw_key):
        return position + scale * jax.random.normal(draw_key, position.shape)

    if support is None:
        return jax.vmap(sample_fn)(jax.random.split(key, num_chains)), jnp.asarray(0)

    return rejection_sample(
        sample_fn, support.violates, num_chains, key,
        max_attempts=max_attempts, repair_fn=support.clip,
    )
