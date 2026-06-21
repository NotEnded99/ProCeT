"""CeT — Certified Training baseline.

Plain gradient descent on the LBP-with-McCormick repair loss. No Jacobian
protection, no SOCP, no protection audit. This is the simplest variant and
the baseline against which α/β-ProCeT are compared in the paper.

Paper name:    ``\\textsc{CeT}``
CLI key:       ``cet``
Supported activations: ``Relu``, ``Tanh``, ``Sigmoid``, ``LeakyRelu``
"""

import torch

from ..core.lbp_loss import simple_gradient_update
from ..core.selection import select_repair_targets
from .base import IterationContext, MethodConfig, RepairMethod


class CeTMethod(RepairMethod):
    """Certified Training: plain GD on the LBP repair loss."""

    name = "cet"
    display_name = "CeT"
    supported_activations = ("Relu", "Tanh", "Sigmoid", "LeakyRelu")
    output_dir_suffix = "CeT"

    def __init__(self, cfg: MethodConfig):
        # CeT defaults: lambda_ = 1.0 (no category weighting).
        super().__init__(cfg)

    # ----- Iteration hooks -----

    def select_targets(self, regions, ctx: IterationContext):
        # Phase 2 (the runner always runs phase-2 style here since depth is fixed).
        return select_repair_targets(
            regions["F_h_positive_in_unsafe"],
            regions["F_safe_cbf_violation"],
            regions["F_depth_limit_reached_unsafe"],
            regions["F_depth_limit_reached_safe"],
            regions["F_unsafe_cannot_split"],
            current_phase=2,
        )

    def needs_top_n(self) -> bool:
        # CeT neither protects nor audits — no Top-N selection needed.
        return False

    def audits(self) -> bool:
        return False

    def get_lr(self, max_depth_limit: int) -> float:
        # Original main_CeT.py: depth_lr_map = {12: 5e-3, 14: 5e-3}
        return {12: 5e-3, 14: 5e-3}.get(max_depth_limit, self.cfg.lr)

    def prepare_inner_loop(self, ctx: IterationContext) -> None:
        # No Jacobians to compute — CeT does plain GD.
        return None

    def compute_update(self, ctx: IterationContext, g_F_clipped: torch.Tensor,
                       current_lr: float) -> tuple:
        update_norm = simple_gradient_update(ctx.model, g_F_clipped, current_lr)
        g_norm = g_F_clipped.norm().item()
        return g_norm, update_norm
