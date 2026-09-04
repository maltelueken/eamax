"""The links a parameterization can put on a coefficient, in one place.

A *link* is the transformation from a coefficient's unconstrained real value to its natural
scale -- ``log`` for a strictly positive quantity, ``identity`` for a signed one. Each is a
TensorFlow Probability bijector, and the correspondence between a link's name and its bijector
lives here alone, so :mod:`~eamax.design.parameterization`, the presets, and the inference
layer all read it from a single definition rather than each reconstructing ``tfb().Exp()`` and
an ``isinstance`` check of their own.

Nothing here touches TFP at import time. This module is on ``import eamax``'s path (via
:mod:`eamax.design`), and :mod:`eamax._tfp` exists precisely so that the several seconds of a
TFP import -- and the requirement to have it installed at all -- stay off the path of anyone
who only wants the closed-form densities. The bijector *classes* the name lookup compares
against are therefore resolved on first use and cached, not at import.

Use the constructors when building a coefficient::

    from eamax.design import coef, log, identity
    coef("V", log())       # strictly positive
    coef("v_d", identity())  # signed difference
"""

from .._tfp import tfb

# Bijector classes for the name lookup, filled in on first use by `_link_classes` so that
# importing this module does not import TFP. Keyed by link name, in lookup order.
_LINK_CLASSES = None


def _link_classes():
    """``{link name: bijector class}``, resolved from TFP once and cached."""
    global _LINK_CLASSES

    if _LINK_CLASSES is None:
        bijectors = tfb()
        _LINK_CLASSES = {"log": bijectors.Exp, "identity": bijectors.Identity}
    return _LINK_CLASSES


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
    """The link name for a bijector: ``"log"`` for ``Exp``, ``"identity"`` for ``Identity``.

    Only those two have names. A coefficient may carry any bijector -- ``constrain``,
    ``to_natural`` and ``log_det_jacobian`` call it directly and never consult this -- but
    the *string* view of a link is what round-trips through
    :meth:`eamax.design.Parameterization.of_names`, and that round trip can only reproduce a
    bijector it can name. Reporting a ``Softplus`` as ``"identity"`` would rebuild it as
    ``Identity()`` and back-transform stored samples on the wrong link with no error, so an
    unnameable bijector raises here instead.

    Raises
    ------
    ValueError
        If ``bijector`` is neither ``Exp`` nor ``Identity``.
    """
    for name, cls in _link_classes().items():
        if isinstance(bijector, cls):
            return name

    raise ValueError(
        f"No link name for bijector {bijector!r}: only Exp ('log') and Identity "
        "('identity') have one. A coefficient may still carry this bijector -- constrain, "
        "to_natural and log_det_jacobian use it directly -- but anything that goes through "
        "link names (Parameterization.links, .of_names, T0Support.from_spec) cannot "
        "represent it, and naming it 'identity' would silently rebuild it as Identity()."
    )
