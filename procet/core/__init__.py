"""Shared framework layer used by all three repair methods.

Modules:
    systems     — DYNAMICS_SYSTEMS mapping (system key -> class)
    io          — model export (ONNX), full-LBP verification wrapper
    metrics     — safety metrics + NumpyJSONEncoder
    lbp_loss    — LBP bound machinery + weighted McCormick repair loss
    selection   — Top-N vulnerable region selection + repair target pruning
    jacobian    — Jacobian of LBP bounds w.r.t. parameters (single/multi-thread)
    socp        — SOCP projection update (Equation (9) of the paper)
    audit       — Protection audit: re-check protected simplices after each step
"""

from .systems import DYNAMICS_SYSTEMS, SUPPORTED_ACTIVATIONS, SYSTEM_DEPTH
from .io import pytorch_to_onnx, verify_model
from .metrics import (
    NumpyJSONEncoder,
    compute_simplex_volume,
    compute_total_volume,
    compute_safety_metrics_v8,
)
from .lbp_loss import (
    compute_min_L_with_mccormick,
    compute_h_max_via_network_bounds,
    compute_repair_loss_and_grad_lbp_weighted,
    simple_gradient_update,
)
from .selection import (
    select_top_n_v_safe_lbp,
    select_top_n_v_unsafe_lbp,
    select_repair_targets,
    select_repair_targets_with_pruning,
)
from .jacobian import (
    compute_jacobian_for_lbp_bounds,
    compute_jacobian_for_lbp_bounds_v1,
)
from .socp import socp_project_and_update_formula9
from .audit import check_protection_status

__all__ = [
    # systems
    "DYNAMICS_SYSTEMS", "SUPPORTED_ACTIVATIONS", "SYSTEM_DEPTH",
    # io
    "pytorch_to_onnx", "verify_model",
    # metrics
    "NumpyJSONEncoder", "compute_simplex_volume", "compute_total_volume",
    "compute_safety_metrics_v8",
    # lbp_loss
    "compute_min_L_with_mccormick", "compute_h_max_via_network_bounds",
    "compute_repair_loss_and_grad_lbp_weighted", "simple_gradient_update",
    # selection
    "select_top_n_v_safe_lbp", "select_top_n_v_unsafe_lbp",
    "select_repair_targets", "select_repair_targets_with_pruning",
    # jacobian
    "compute_jacobian_for_lbp_bounds", "compute_jacobian_for_lbp_bounds_v1",
    # socp
    "socp_project_and_update_formula9",
    # audit
    "check_protection_status",
]
