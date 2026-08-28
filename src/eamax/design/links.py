"""The links a parameterization can put on a coefficient, in one place.

A *link* is the transformation from a coefficient's unconstrained real value to its natural
scale -- ``log`` for a strictly positive quantity, ``identity`` for a signed one. Each is a
TensorFlow Probability bijector, and the correspondence between a link's name and its bijector
lives here alone, so :mod:`~eamax.design.parameterization`, the presets, and the inference
layer all read it from a single definition rather than each reconstructing ``tfb().Exp()`` and
an ``isinstance`` check of their own.

Use the constructors when building a coefficient::

    from eamax.design import coef, log, identity
    coef("V", log())       # strictly positive
    coef("v_d", identity())  # signed difference
"""

from .._tfp import tfb

# The bijector class for the log link, cached for the name lookup below. Only ``Exp`` counts
# as the log link; every other bijector reports as ``"identity"`` for the string view.
_LOG_BIJECTOR = tfb().Exp


def log():
    """The log link: an ``Exp`` bijector, for a strictly positive quantity (``log(x)``)."""
    return tfb().Exp()


def identity():
    """The identity link: an ``Identity`` bijector, for an already-unconstrained quantity."""
    return tfb().Identity()


_BY_NAME = {"log": log, "identity": identity}


def bijector_for_link(name):
    """The bijector for a link name (``"log"`` or ``"identity"``)."""
    try:
        return _BY_NAME[name]()
    except KeyError:
        raise ValueError(f"Unknown link {name!r}; expected one of {sorted(_BY_NAME)}")


def link_name(bijector):
    """The link name for a bijector: ``"log"`` for ``Exp``, otherwise ``"identity"``."""
    return "log" if isinstance(bijector, _LOG_BIJECTOR) else "identity"
