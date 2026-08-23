"""Lazy access to the TensorFlow Probability JAX substrate.

TFP is a required runtime peer for `eamax`'s sampling paths and for
`eamax.hierarchical`, but it is deliberately *not* a declared dependency: `tfp-nightly`
and `tensorflow-probability` are separate PyPI distributions that both install the
`tensorflow_probability` module. All three consumer repos are on `tfp-nightly`, so
declaring `tensorflow-probability` would install both into one environment and they would
fight over the same import path.

Importing lazily also keeps TFP -- and the several seconds it takes to import -- off the
path of anyone who only wants the closed-form densities.
"""

_MISSING = (
    "eamax needs the TensorFlow Probability JAX substrate for this operation, but "
    "`tensorflow_probability` is not importable.\n\n"
    "Install exactly one of these distributions (they provide the same module and must "
    "not be installed together):\n"
    "    pip install tfp-nightly            # what the eamax consumer repos use\n"
    "    pip install tensorflow-probability # or: pip install 'eamax[tfp]'\n\n"
    "Closed-form densities (`eamax.accumulators.wald`, `.lba`) do not need TFP; only "
    "sampling and `eamax.hierarchical` do."
)


def tfd():
    """The `tensorflow_probability.substrates.jax.distributions` module."""
    try:
        from tensorflow_probability.substrates import jax as tfp
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return tfp.distributions


def tfb():
    """The `tensorflow_probability.substrates.jax.bijectors` module."""
    try:
        from tensorflow_probability.substrates import jax as tfp
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return tfp.bijectors
