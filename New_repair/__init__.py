"""
New_repair module for CBF verification and parameter repair.

This module contains implementations of algorithms for repairing neural
control barrier functions using verification-based approaches.
"""

from .geometry_module import (
    compute_simplex_bound,
    compute_jacobian_matrix,
)

__all__ = [
    "compute_simplex_bound",
    "compute_jacobian_matrix",
]
