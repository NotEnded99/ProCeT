"""
ProCeT — Provably-Certified Training / Repair for Neural Control Barrier Functions.

Package layout:
    procet.core         Shared framework: I/O, metrics, LBP bounds, SOCP, audit.
    procet.methods      The three method variants (CeT, α-ProCeT, β-ProCeT).
    procet.runner       The shared outer repair loop (template method).

Public API:
    from procet import run_repair, METHOD_REGISTRY
"""

from .runner import run_repair
from .methods import METHOD_REGISTRY, build_method

__all__ = ["run_repair", "METHOD_REGISTRY", "build_method"]

__version__ = "0.1.0"
