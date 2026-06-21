"""Dynamics system registry.

Maps user-facing system keys (``simple2d``, ``barr1`` …) to the corresponding
dynamics classes defined in ``lbp_neural_cbf.cbf``. The registry is the single
source of truth for argparse ``choices`` and for the system-specific splitting
depth schedule.
"""

from lbp_neural_cbf.cbf.fossil_dynamics import (
    Barrier1System,
    Barrier2System,
    Barrier3System,
    Barrier4System,
)
from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem, CartPoleSystem


DYNAMICS_SYSTEMS = {
    "simple2d": Simple2DSystem,
    "barr1":    Barrier1System,
    "barr2":    Barrier2System,
    "barr3":    Barrier3System,
    "barr4":    Barrier4System,
    "cartpole": CartPoleSystem,
}

# Activation functions accepted by the repair scripts. Note: the ProCeT family
# (α/β) only supports smooth activations — ``SUPPORTED_ACTIVATIONS_PROCET`` —
# because their SOCP protection requires differentiable bounds. CeT accepts all
# four since it relies on plain gradient descent.
SUPPORTED_ACTIVATIONS = ["Relu", "Tanh", "Sigmoid", "LeakyRelu"]
SUPPORTED_ACTIVATIONS_PROCET = ["Tanh", "Sigmoid"]

# Per-system default verification depth. ``barr4`` and ``cartpole`` are higher-
# dimensional and need deeper splitting to resolve boundary cases.
SYSTEM_DEPTH = {
    "simple2d": 12,
    "barr1":    12,
    "barr2":    12,
    "barr3":    12,
    "barr4":    14,
    "cartpole": 14,
}
