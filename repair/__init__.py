"""
ICGAR: Iterative Certificate-Gradient Aligned Refinement

This module implements the ICGAR method for repairing Neural Control Barrier Functions
while preserving verified-region invariance through manifold-constrained optimization.
"""

from .icgar_repair import ICGARRepair, icgar_repair_pipeline
from .tangent_space import compute_tangent_space, compute_projection_matrix
from .lbp_bounds import LBPBoundsComputer, LBPLowerBoundComputer
from .alpha_schedule import compute_alpha, AlphaScheduler

__all__ = [
    "ICGARRepair",
    "icgar_repair_pipeline",
    "compute_tangent_space",
    "compute_projection_matrix",
    "LBPBoundsComputer",
    "LBPLowerBoundComputer",
    "compute_alpha",
    "AlphaScheduler",
]
