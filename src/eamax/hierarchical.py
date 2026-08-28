"""Semi-centered hierarchical prior over subject-level parameters.

Multi-subject fits need a population distribution over per-subject parameters. The family
used by every consumer here is a multivariate normal on the unconstrained (log) scale, with
an LKJ prior on the correlation structure and an inverse-gamma prior on the between-subject
scales:

    s        ~ InverseGamma(concentration, scale)        per parameter
    mu       ~ Normal(mu_loc, mu_scale)                  per parameter
    psi_raw  ~ CholeskyLKJ(P, lkj_concentration)
    L        = diag(s) @ psi_raw

**Semi-centered.** The leading `P - num_centered` parameters use the non-centered
parameterization (standard-normal offsets `z`, scaled up by `L`); the trailing
`num_centered` use the centered one (`theta_bt`, drawn from the MVN conditional on `z`).
Neither parameterization works for every parameter -- non-centering helps where the data are
weak relative to the population spread and hurts where they are strong -- so the split is
part of the model, and it is why parameter *order* matters: whichever parameters should be
centered have to be last.

`reconstruct_semicentered` is the reason most of this module exists. In the source
repositories that four-line formula was written out verbatim in about six places -- both
hierarchical simulators, the likelihood wrapper, the particle reconstruction, the
post-processing, and a test's private copy -- and one of those docstrings noted that "all
three must agree" back when there were only three. All six have to compute the same thing
or the fit is silently wrong, so it is one function here.

`HierarchicalFlatSpace` is the other half. A sampler explores one flat real vector, so
something has to own the map between that vector and the five-component dict -- along with
the change-of-variables term, which `log_prob` here deliberately omits because it scores the
*constrained* parameterization. Putting that with the prior rather than with a sampler is
what lets `eamax.inference` stay ignorant of these component names, and it is the same
argument `default_bijector` already makes: the keys are the prior's own business.

TFP is imported lazily; see `eamax._tfp` for why it is not a declared dependency.
"""

import jax.numpy as jnp

from ._tfp import tfb, tfd

DEFAULT_LKJ_CONCENTRATION = 2.0

#: Tail index of the between-subject scales. `P(s > x) ~ x**-concentration`, so at the
#: historically common value of 4 the tail is heavy enough that a rare population draws
#: `s ~ 1` in log space, and hence subject parameters spanning orders of magnitude. For a
#: neural likelihood that is outside the box the flow was trained on, where its log-density
#: has gradient spikes and flat plateaus that collapse step-size adaptation. Lowering
#: `inverse_gamma_scale` cannot fix it -- that shifts `s` down but leaves the tail index
#: unchanged. Raising the concentration (and scaling the scale up to hold the median fixed)
#: is what bounds the worst case.
DEFAULT_INVERSE_GAMMA_CONCENTRATION = 4.0


def cholesky_factor(s, psi_raw):
    """Covariance Cholesky `L = diag(s) @ psi_raw`."""
    return jnp.asarray(s)[:, None] * jnp.asarray(psi_raw)


def reconstruct_semicentered(mu, s, psi_raw, z, theta_bt):
    """Per-subject unconstrained parameters from the prior's five named components.

    The single definition of a formula the source repositories wrote out six times.

    Parameters
    ----------
    mu : array
        Population means, shape ``(P,)``.
    s : array
        Between-subject scales, shape ``(P,)``.
    psi_raw : array
        Correlation Cholesky factor, shape ``(P, P)``.
    z : array
        Standard-normal offsets for the non-centered block, shape ``(S, P_ncp)``.
    theta_bt : array
        The centered block, shape ``(S, num_centered)``.

    Returns
    -------
    array
        Per-subject parameters on the unconstrained scale, shape ``(S, P)`` -- the input
        :meth:`eamax.design.Parameterization.constrain` expects, one row per subject.
    """
    z = jnp.asarray(z)
    num_ncp = z.shape[-1]
    factor = cholesky_factor(s, psi_raw)[:num_ncp, :num_ncp]
    theta_ncp = jnp.asarray(mu)[:num_ncp] + jnp.einsum("nj,ij->ni", z, factor)
    return jnp.concatenate([theta_ncp, jnp.asarray(theta_bt)], axis=-1)


def reconstruct_from_dict(params):
    """:func:`reconstruct_semicentered` from the prior's own sample dict.

    Parameters
    ----------
    params : dict of array
        Keyed by ``mu``, ``s``, ``psi_raw``, ``z``, ``theta_bt``.

    Returns
    -------
    array
        Shape ``(S, P)``.
    """
    return reconstruct_semicentered(
        params["mu"], params["s"], params["psi_raw"], params["z"], params["theta_bt"]
    )


def default_bijector():
    """Constrained <-> unconstrained bijector for the prior's named components.

    Belongs to the prior rather than to the consumer: the component names are the prior's
    own dict keys, and which of them need constraining is a property of the family.
    """
    return tfb().JointMap(
        {
            "s": tfb().Exp(),  # InverseGamma -> positive
            "mu": tfb().Identity(),  # Normal -> real
            "psi_raw": tfb().CorrelationCholesky(),  # CholeskyLKJ -> unconstrained vector
            "z": tfb().Identity(),  # N(0, I)
            "theta_bt": tfb().Identity(),  # already unconstrained
        }
    )


class HierarchicalLKJMVNPrior:
    """Semi-centered hierarchical LKJ-MVN prior over subject-level parameters.

    `sample`, `mode` and `log_prob` all speak the five-component dict
    (`s`, `mu`, `psi_raw`, `z`, `theta_bt`); call `reconstruct_from_dict` to turn a draw
    into `(S, P)` per-subject parameters.

    Parameters
    ----------
    num_subjects : int
        S.
    num_params : int
        P, the number of per-subject parameters.
    inverse_gamma_scale : array
        Scale of the InverseGamma prior on ``s``, shape ``(P,)``.
    mu_loc, mu_scale : array
        Hyperparameters of the Normal prior on ``mu``, shape ``(P,)``.
    num_centered : int, optional
        How many *trailing* parameters use the centered parameterization.
    inverse_gamma_concentration : float, optional
        See :data:`DEFAULT_INVERSE_GAMMA_CONCENTRATION`.
    lkj_concentration : float, optional
        CholeskyLKJ concentration.

    Raises
    ------
    ValueError
        If any hyperparameter array does not have shape ``(P,)``.
    """

    def __init__(
        self,
        num_subjects,
        num_params,
        *,
        inverse_gamma_scale,
        mu_loc,
        mu_scale,
        num_centered=2,
        inverse_gamma_concentration=DEFAULT_INVERSE_GAMMA_CONCENTRATION,
        lkj_concentration=DEFAULT_LKJ_CONCENTRATION,
    ):
        self.num_subjects = num_subjects
        self.num_params = num_params
        self.num_centered = num_centered
        self.num_params_ncp = num_params - num_centered

        arrays = {
            "inverse_gamma_scale": jnp.asarray(inverse_gamma_scale),
            "mu_loc": jnp.asarray(mu_loc),
            "mu_scale": jnp.asarray(mu_scale),
        }
        wrong = {name: value.shape for name, value in arrays.items() if value.shape != (num_params,)}
        if wrong:
            raise ValueError(
                f"Hyperparameter arrays must have shape ({num_params},); got {wrong}. "
                "A length mismatch usually means the hyperparameters came from a different "
                "model's parameter spec than the one being fitted."
            )

        self.inverse_gamma_scale = arrays["inverse_gamma_scale"]
        self.mu_loc = arrays["mu_loc"]
        self.mu_scale = arrays["mu_scale"]
        self.inverse_gamma_concentration = inverse_gamma_concentration
        self.lkj_concentration = lkj_concentration

        d = tfd()
        num_ncp = self.num_params_ncp

        def centered_block(z, s, mu, psi_raw):
            # The centered block is drawn from the MVN conditional on the non-centered one,
            # so the joint stays exactly the intended multivariate normal however the split
            # is placed.
            factor = cholesky_factor(s, psi_raw)
            lower_left = factor[num_ncp:, :num_ncp]
            lower_right = factor[num_ncp:, num_ncp:]
            conditional_mean = mu[num_ncp:] + jnp.einsum("nk,jk->nj", z, lower_left)
            return d.Independent(
                d.MultivariateNormalTriL(loc=conditional_mean, scale_tril=lower_right),
                reinterpreted_batch_ndims=1,
            )

        self._joint = d.JointDistributionNamed(
            {
                "s": d.Independent(
                    d.InverseGamma(
                        concentration=inverse_gamma_concentration, scale=self.inverse_gamma_scale
                    ),
                    reinterpreted_batch_ndims=1,
                ),
                "mu": d.Independent(
                    d.Normal(loc=self.mu_loc, scale=self.mu_scale), reinterpreted_batch_ndims=1
                ),
                "psi_raw": d.CholeskyLKJ(num_params, lkj_concentration),
                "z": d.Independent(
                    d.Normal(
                        loc=jnp.zeros((num_subjects, num_ncp)),
                        scale=jnp.ones((num_subjects, num_ncp)),
                    ),
                    reinterpreted_batch_ndims=2,
                ),
                "theta_bt": centered_block,
            }
        )

    @classmethod
    def from_spec(cls, num_subjects, spec, **kwargs):
        """Build from a :class:`eamax.design.Parameterization`.

        Takes the spec's hyperparameter arrays and its centered block.

        Parameters
        ----------
        num_subjects : int
            S.
        spec : Parameterization
            Supplies ``num_params``, ``num_centered`` and the hyperparameter arrays.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        HierarchicalLKJMVNPrior
        """
        return cls(
            num_subjects,
            num_params=spec.num_params,
            num_centered=spec.num_centered,
            inverse_gamma_scale=spec.hyperparameter_array("inverse_gamma_scale"),
            mu_loc=spec.hyperparameter_array("mu_loc"),
            mu_scale=spec.hyperparameter_array("mu_scale"),
            **kwargs,
        )

    def sample(self, seed):
        """One draw, as the five-component dict."""
        return self._joint.sample(seed=seed)

    def log_prob(self, params):
        return self._joint.log_prob(params)

    def mode(self):
        """Modal value of each component, for initialising a sampler.

        Computed analytically per marginal: `InverseGamma` mode is
        `scale / (concentration + 1)`; `Normal` is its location; `CholeskyLKJ` with
        concentration >= 1 is the identity; `z` is zero; and with `psi_raw = I` the
        conditional mean of the centered block is just its slice of ``mu``.

        Returns
        -------
        dict of array
            Keyed by ``s``, ``mu``, ``psi_raw``, ``z``, ``theta_bt``.
        """
        return {
            "s": self.inverse_gamma_scale / (self.inverse_gamma_concentration + 1.0),
            "mu": self.mu_loc,
            "psi_raw": jnp.eye(self.num_params),
            "z": jnp.zeros((self.num_subjects, self.num_params_ncp)),
            "theta_bt": jnp.broadcast_to(
                self.mu_loc[self.num_params_ncp :],
                (self.num_subjects, self.num_centered),
            ),
        }

    def sample_subject_params(self, seed):
        """One draw, reconstructed as `(mu, s, (S, P) per-subject parameters)`."""
        params = self.sample(seed)
        return params["mu"], params["s"], reconstruct_from_dict(params)

    def flat_space(self, bijector=None):
        """A :class:`HierarchicalFlatSpace` over this prior's coordinates."""
        return HierarchicalFlatSpace(self, bijector)

    def __repr__(self):
        return (
            f"HierarchicalLKJMVNPrior(num_subjects={self.num_subjects}, "
            f"num_params={self.num_params}, num_centered={self.num_centered})"
        )


def joint_log_det_jacobian(bijector, unconstrained):
    """Total forward log-det-Jacobian of a `tfb.JointMap`, reduced component by component.

    Calling ``JointMap.forward_log_det_jacobian`` without ``event_ndims`` lets each
    component fall back to its bijector's *minimum* event rank -- 0 for ``Exp``, which
    therefore returns a shape-``(P,)`` array rather than a scalar. TFP then broadcasts the
    components against one another before reducing, so a scalar term such as
    ``CorrelationCholesky``'s gets added once per element of the unreduced one. Summing
    that overcounts it by ``P``, and because the overcount is ``(P - 1)`` times a
    *state-dependent* quantity it does not cancel -- it tilts the posterior over
    correlations.

    Reducing each component with ``event_ndims`` equal to its own full rank makes every
    term a scalar before it is summed, which is correct for any `JointMap` regardless of
    its components' shapes.

    The two source repositories work around the same problem by hand-writing the two
    non-trivial terms and naming the prior's components inline. That is right for
    :func:`default_bijector` and silently wrong for any other, which is why this is
    written generically and lives with the prior rather than with a sampler.

    Parameters
    ----------
    bijector : tfb.JointMap
        Constrained <-> unconstrained map, keyed like the prior's components.
    unconstrained : dict of array
        Unconstrained components -- the input to ``bijector.forward``.

    Returns
    -------
    array
        Scalar total log determinant.
    """
    total = 0.0
    for name, component in bijector.bijectors.items():
        value = jnp.asarray(unconstrained[name])
        total = total + component.forward_log_det_jacobian(value, event_ndims=value.ndim)
    return total


class HierarchicalFlatSpace:
    """Flat unconstrained coordinates for a :class:`HierarchicalLKJMVNPrior`.

    A sampler explores a single flat real vector; the prior speaks a five-component dict
    on the constrained scale. This owns the round trip between them and the
    change-of-variables term that goes with it, so a sampler never has to know the
    component names -- which is what lets one driver serve this prior and any other.

    The layout is :func:`jax.flatten_util.ravel_pytree`'s, i.e. dict keys in *sorted*
    order (``mu, psi_raw, s, theta_bt, z``), not the order the prior declares them. It is
    fixed at construction from the prior's mode, so the unravel closure is static under
    ``jit``. Use :attr:`component_slices` rather than hard-coding offsets.

    Parameters
    ----------
    prior : HierarchicalLKJMVNPrior
        The prior whose coordinates these are.
    bijector : tfb.JointMap, optional
        Defaults to :func:`default_bijector`.

    Attributes
    ----------
    num_flat_params : int
        D, the length of the flat vector.
    component_slices : dict of slice
        Where each component lives in the flat vector.
    """

    def __init__(self, prior, bijector=None):
        from jax.flatten_util import ravel_pytree

        self.prior = prior
        self.bijector = default_bijector() if bijector is None else bijector

        template = self.bijector.inverse(prior.mode())
        flat, unravel = ravel_pytree(template)

        self._unravel = unravel
        self.num_flat_params = int(flat.size)

        offset = 0
        slices = {}
        for name in sorted(template):
            size = int(jnp.size(template[name]))
            slices[name] = slice(offset, offset + size)
            offset += size
        self.component_slices = slices

    def unravel(self, flat):
        """Flat vector, shape ``(D,)``, to the unconstrained component dict."""
        return self._unravel(jnp.asarray(flat))

    def ravel(self, unconstrained):
        """Unconstrained component dict to a flat vector, shape ``(D,)``."""
        from jax.flatten_util import ravel_pytree

        flat, _ = ravel_pytree(unconstrained)
        return flat

    def forward(self, flat):
        """Flat vector to the *constrained* component dict."""
        return self.bijector.forward(self.unravel(flat))

    def log_det_jacobian(self, flat):
        """Scalar log determinant of :meth:`forward`'s Jacobian.

        Public, and separate from :meth:`log_prob`, so the correction can be tested on its
        own against an autodiff determinant.
        """
        return joint_log_det_jacobian(self.bijector, self.unravel(flat))

    def log_prob(self, flat):
        """Prior log density in flat unconstrained coordinates.

        Includes the change of variables. :meth:`HierarchicalLKJMVNPrior.log_prob`
        deliberately does not -- it scores the constrained parameterization -- so a sampler
        exploring unconstrained space must add the Jacobian, and this is where that happens
        once rather than in every consumer.
        """
        flat = jnp.asarray(flat)
        return self.prior.log_prob(self.forward(flat)) + self.log_det_jacobian(flat)

    def subject_params(self, flat):
        """Per-subject unconstrained parameters, shape ``(S, P)``.

        The semi-centered reconstruction, reached from flat coordinates. This is the
        formula the source repositories wrote out six times and then four more times inside
        their samplers; see :func:`reconstruct_semicentered`.
        """
        return reconstruct_from_dict(self.forward(flat))

    def population(self, flat):
        """Population location and scale as ``(mu, s)``, each shape ``(P,)``."""
        params = self.forward(flat)
        return params["mu"], params["s"]

    def mode(self):
        """The prior's mode in flat coordinates, shape ``(D,)``."""
        return self.ravel(self.bijector.inverse(self.prior.mode()))

    def sample(self, seed):
        """One prior draw in flat coordinates, shape ``(D,)``."""
        return self.ravel(self.bijector.inverse(self.prior.sample(seed)))

    def sample_particles(self, seed, num_particles):
        """`num_particles` independent prior draws, shape ``(num_particles, D)``."""
        import jax

        return jax.vmap(self.sample)(jax.random.split(seed, num_particles))

    def __repr__(self):
        return (
            f"HierarchicalFlatSpace(num_flat_params={self.num_flat_params}, "
            f"prior={self.prior!r})"
        )

def interval_to_mu_loc_scale(percentile_interval, subject_scale):
    """Split a target 95% interval for subject parameters into `mu_scale` and `s`.

    A design helper for choosing hyperparameters: given how wide you want subject-level
    parameters to range, and how much of that spread should be within-population rather
    than uncertainty about the population mean, this returns the `mu_loc` / `mu_scale` that
    leave the rest. Raises when `subject_scale` already exceeds the requested interval.
    """
    import numpy as np

    lower, upper = np.asarray(percentile_interval[0]), np.asarray(percentile_interval[1])
    subject_scale = np.asarray(subject_scale)

    mu_loc = (lower + upper) / 2.0
    total_variance = ((upper - lower) / (2 * 1.959963984540054)) ** 2
    remaining = total_variance - subject_scale**2

    if np.any(remaining <= 0):
        bad = np.where(remaining <= 0)[0]
        raise ValueError(
            f"subject_scale is too large relative to the requested interval for parameter "
            f"indices {bad.tolist()}. Reduce subject_scale or widen the interval."
        )
    return mu_loc, np.sqrt(remaining)
