"""Starting values: where a sampler begins, and why it matters that it begins in support.

The four source routines look unrelated -- a fixed list with the last entry overwritten
from the data, per-chain rejection sampling from the prior, whole-particle rejection over
every subject, and a bare deterministic prior mode -- but they are four points in one
product:

    (source of dispersion) x (support constraint) x (what to do when rejection gives up)

and they differ in which factors they leave out. This module supplies all three, so the
choice becomes explicit rather than implied by which repository the code was copied from.

**The constraint.** For a race model the likelihood is undefined at ``t0 >= min(rt)``, and
`eamax.race` covers that region with a slope-1e3 penalty whose whole purpose is to push
`t0` back down. A chain that *starts* there starts against a wall: window adaptation sees
divergence after divergence and its only response is to shrink the step size, which does
not help, because the wall is not a curvature scale. Measured on the RDM at 500 trials,
perturbing every parameter alike by ~0.35 in log space gives 1001/1000 divergences, max
R-hat 1.53 and min ESS 7. Dispersion has to be support-aware to be worth anything.

**Rejection, not clipping.** Capping the offending coordinate looks cheaper and is worse: a
cap is a point mass. Measured on the hierarchical RDM at a 0.9 cap, 55% of
(particle, subject) draws were capped, 98% of particles had at least one, and the worst
subject had 93% of its particles pinned to the single value ``log(0.9 * min_rt)`` -- which
is dispersion destroyed in the one coordinate the routine exists to keep dispersed.
Rejecting whole draws yields exactly the source distribution conditioned on the constraint.

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
#: Close to 1 on purpose: the true `t0` sits at a median 0.807 of `min_rt` on the RDM and
#: 0.926 on the conflict RDM, so a low ceiling puts every chain below the truth and leaves
#: window adaptation to walk them all back up. At 0.9 the cap excluded the true `t0` for
#: 4.5% of subjects outright.
DEFAULT_MAX_T0_FRACTION = 0.97

#: Redraws before a rejection sampler gives up on one draw.
DEFAULT_MAX_ATTEMPTS = 2000


def min_valid_rt(rt, *, mask=None, sentinel=0.0, axis=-1):
    """Smallest response time that is a real observation.

    Both source repositories state in prose that the non-crossing sentinel has to be
    excluded before computing this, and neither provides a function that does it -- so the
    exclusion is written out at each call site, or forgotten.

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
    """
    rt = jnp.asarray(rt)
    valid = rt > sentinel
    if mask is not None:
        valid = valid & jnp.asarray(mask)
    return jnp.min(jnp.where(valid, rt, jnp.inf), axis=axis)


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
        """Build from a :class:`eamax.design.ParamSpec` and the fastest observed RT.

        All four source routines locate ``t0`` as *the last entry*, positionally and
        unenforced -- a comment in one of them says so outright. A spec knows the name, so
        this asks it, the same way :mod:`eamax.design.map` already does. The lookup costs
        nothing and turns "silently clipped the wrong parameter" into an exception.

        Parameters
        ----------
        spec : ParamSpec
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
                f"t0 must be on the 'log' link for this constraint, but it is on "
                f"'{spec.links[index]}'. The cap compares against an unconstrained value, "
                "which is only log(t0) under the log link."
            )

        return cls(index=index, log_t0_max=jnp.log(max_fraction * jnp.asarray(min_rt)))

    def _values(self, theta):
        return jnp.asarray(theta)[..., self.index]

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
        return jnp.any(self._values(theta) > self.log_t0_max)

    def clip(self, theta):
        """Pin breaching entries to the cap.

        The exhaustion fallback, not a strategy -- see this module's docstring for what a
        cap does to dispersion when it binds often.
        """
        theta = jnp.asarray(theta)
        return theta.at[..., self.index].set(jnp.minimum(self._values(theta), self.log_t0_max))


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
        ``f(pytree) -> pytree``, applied to a draw that gave up. When ``None`` an exhausted
        draw is returned as drawn, which is the honest default -- not every constraint has
        a sensible projection.

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
            sample = repair_fn(sample)
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
    than on a raw component of the prior's dict. The source version tests
    ``sample["theta_bt"][:, -1]``, which silently requires ``t0`` to be both the last
    parameter and in the centered block; going through the reconstruction costs one einsum
    per draw and requires neither. Where ``t0`` *is* last and centered the two are the same
    number, so nothing changes for any current configuration.

    The accepted region is where the posterior lives -- outside it the likelihood is a
    penalty, not a density -- so restricting the cloud costs no posterior mass. It is
    nonetheless **not** the prior, and that matters: ``adaptive_tempered_smc`` weights
    increments by the likelihood alone and never corrects the initial distribution, so only
    mutation can. Store it as an initialisation artefact, not under a ``prior`` group, or
    anything measuring posterior contraction against it will be wrong.

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
    check -- exactly what `cognitive-control-comparison` starts its warmup from. That is
    reproducible on purpose, but it is worth knowing what it costs: every chain begins at
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

    `eam-abi-robustness`'s configured initial position, whose `t0` entry is overwritten
    with ``min(rt) / 2`` inside the fitting driver. Two things change here: the entry is
    located by name rather than as index ``-1``, and the fraction is a named argument
    rather than a division buried in the expression.

    This is the one builder with no dispersion at all. It is kept because it is what the
    existing configurations specify, not because it is a good way to start four chains.

    Parameters
    ----------
    values : sequence of float
        Natural-scale starting values, the full vector including any leading bounded block.
        See :meth:`eamax.inference.transforms.BlockTransform.midpoint` for that block.
    spec : ParamSpec, optional
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
    array
        Shape ``(P,)``, natural scale unless ``transform`` is given.

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

    return position if transform is None else transform.inverse(position)


def jitter_positions(position, num_chains, key, *, scale=0.1, support=None,
                     max_attempts=DEFAULT_MAX_ATTEMPTS):
    """Disperse one unconstrained position into `num_chains` starts.

    The route for a consumer with no prior object to hand. It is the weakest of the four
    sources -- an arbitrary width standing in for the prior's own scale -- but it is enough
    to give R-hat something to measure, which a replicated position is not.

    ``support`` is not really optional. Undirected jitter is precisely what produced
    1001/1000 divergences in the measurement recorded on this module, so omitting it while
    jittering warns.

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
            "response time lands on the likelihood's penalty region, where step-size "
            "adaptation cannot recover. Pass `support=T0Support.from_spec(...)`.",
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
