"""Method registry.

The three repair methods are exposed via ``METHOD_REGISTRY`` so the runner and
CLI scripts can refer to them by string. Adding a new method means: subclass
``RepairMethod``, implement the abstract hooks, and register it here.
"""

from .base import RepairMethod, MethodConfig, IterationContext
from .cet import CeTMethod
from .alpha_procet import AlphaProCeTMethod
from .beta_procet import BetaProCeTMethod


METHOD_REGISTRY = {
    "cet":           CeTMethod,
    "alpha-procet":  AlphaProCeTMethod,
    "beta-procet":   BetaProCeTMethod,
    # Aliases — paper uses α/β-ProCeT, but ASCII identifiers are friendlier on the CLI.
    "alpha_procet":  AlphaProCeTMethod,
    "beta_procet":   BetaProCeTMethod,
    "procet":        AlphaProCeTMethod,   # bare "ProCeT" in the paper = α-ProCeT
    "aprocet":       BetaProCeTMethod,
}


def build_method(name, cfg):
    """Instantiate a registered method.

    Args:
        name: Key into ``METHOD_REGISTRY`` (e.g. ``'cet'``, ``'alpha-procet'``).
        cfg: ``MethodConfig`` passed to the method constructor.

    Raises:
        KeyError if ``name`` is unknown.
    """
    try:
        cls = METHOD_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown method '{name}'. Available: {sorted(METHOD_REGISTRY)}"
        )
    return cls(cfg)


__all__ = [
    "RepairMethod", "MethodConfig", "IterationContext",
    "CeTMethod", "AlphaProCeTMethod", "BetaProCeTMethod",
    "METHOD_REGISTRY", "build_method",
]
