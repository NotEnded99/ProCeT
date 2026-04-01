"""
New_repair module for CBF verification and parameter repair.

This module contains implementations of algorithms for repairing neural
control barrier functions using verification-based approaches.
"""

from .geometry_module import (
    compute_lbp_bounds,
    compute_jacobian_matrix,
    compute_lbp_bounds_batch,
    compute_lbp_bounds_with_crown
)

__all__ = [
    "compute_lbp_bounds",
    "compute_jacobian_matrix",
    "compute_lbp_bounds_batch",
    "compute_lbp_bounds_with_crown",
]
