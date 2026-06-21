"""α-ProCeT — Protected repair via SOCP projection (Equation 9).

The flagship variant. On every iteration:

  1. Select Top-N most vulnerable V_safe / V_unsafe regions and snapshot them
     as *protected* simplices (kept fixed across the inner loop).
  2. Compute Jacobians ``J_safe = ∇_θ φ_θ`` and ``J_unsafe = ∇_θ ψ_θ`` once,
     before the inner loop.
  3. In each inner step, solve the SOCP (Equation 8): minimise the epigraph
     distance to the repair gradient subject to safe-side and unsafe-side
     protection constraints and an L2 trust region.
  4. Audit the protected simplices after each step (re-compute their bounds
     under the new parameters and log success / failure counts).

Paper name:    ``\\textsc{$\\alpha$-ProCeT}``
CLI key:       ``alpha-procet``  (alias: ``procet``, ``alpha_procet``)
Supported activations: ``Tanh``, ``Sigmoid`` (smooth only — required for SOCP)
"""

import numpy as np
import torch

from ..core.jacobian import compute_jacobian_for_lbp_bounds_v1
from ..core.selection import select_repair_targets_with_pruning
from ..core.socp import socp_project_and_update_formula9
from .base import IterationContext, MethodConfig, RepairMethod


class AlphaProCeTMethod(RepairMethod):
    """α-ProCeT: SOCP-protected repair with full Top-N audit."""

    name = "alpha-procet"
    display_name = "α-ProCeT"
    supported_activations = ("Tanh", "Sigmoid")
    output_dir_suffix = "alphaProCeT_prab"

    def __init__(self, cfg: MethodConfig):
        super().__init__(cfg)

    # ----- Iteration hooks -----

    def select_targets(self, regions, ctx: IterationContext):
        total_region_count = (
            len(regions["V_safe"]) + len(regions["V_unsafe"])
            + len(regions["F_h_positive_in_unsafe"])
            + len(regions["F_safe_cbf_violation"])
            + len(regions["F_depth_limit_reached_unsafe"])
            + len(regions["F_depth_limit_reached_safe"])
            + len(regions["F_unsafe_cannot_split"])
        )
        return select_repair_targets_with_pruning(
            regions["F_h_positive_in_unsafe"],
            regions["F_safe_cbf_violation"],
            regions["F_depth_limit_reached_unsafe"],
            regions["F_depth_limit_reached_safe"],
            regions["F_unsafe_cannot_split"],
            current_phase=2,
            model=ctx.model,
            dynamics_model=ctx.dynamics_model,
            lbp_linearizer=ctx.lbp_linearizer,
            device=ctx.device,
            dtype=ctx.dtype,
            total_region_count=total_region_count,
        )

    def needs_top_n(self) -> bool:
        # Protection + audit both rely on the Top-N snapshot.
        return True

    def audits(self) -> bool:
        return True

    def get_lr(self, max_depth_limit: int) -> float:
        # Original main_ProCeT_prab.py: depth_lr_map = {12: 2e-2, 14: 2e-2}
        return {12: 2e-2, 14: 2e-2}.get(max_depth_limit, self.cfg.lr)

    def get_beta(self):
        return 0.999, 0.999

    def prepare_inner_loop(self, ctx: IterationContext) -> None:
        """Compute J_safe and J_unsafe from the snapshot simplices."""
        print(f"\n[Iteration {ctx.iteration + 1}.2] Computing Jacobian matrices...")
        ctx.J_safe = compute_jacobian_for_lbp_bounds_v1(
            ctx.model, ctx.top_n_v_safe, "safe",
            ctx.dynamics_model, ctx.lbp_linearizer, ctx.device, ctx.dtype,
        )
        ctx.J_unsafe = compute_jacobian_for_lbp_bounds_v1(
            ctx.model, ctx.top_n_v_unsafe, "unsafe",
            ctx.dynamics_model, ctx.lbp_linearizer, ctx.device, ctx.dtype,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_update(self, ctx: IterationContext, g_F_clipped: torch.Tensor,
                       current_lr: float) -> tuple:
        beta_s, beta_us = self.get_beta()
        rho_safe = (
            np.maximum(ctx.top_n_v_safe_margins, 1e-12)
            if len(ctx.top_n_v_safe) > 0 else np.array([])
        )
        rho_unsafe = (
            np.maximum(-ctx.top_n_v_unsafe_h_ub, 1e-12)
            if len(ctx.top_n_v_unsafe) > 0 else np.array([])
        )
        return socp_project_and_update_formula9(
            model=ctx.model,
            g_raw=g_F_clipped,
            grad_phi=ctx.J_safe,
            grad_psi=ctx.J_unsafe,
            rho_safe=rho_safe,
            rho_unsafe=rho_unsafe,
            beta_s=beta_s,
            beta_us=beta_us,
            lr=current_lr,
            delta_theta_norm_bound=self.cfg.delta_theta_norm_bound,
        )

    # α-ProCeT encodes its extra hyper-params (K inner steps, ζ trust region)
    # in the result filename so different settings don't clobber each other.
    def result_file_stem(self) -> str:
        c = self.cfg
        return (f"result_{c.system}_{c.activation}_{self.output_dir_suffix}"
                f"_depth{c.max_depth_limit}_w{c.lambda_}"
                f"_k{c.num_inner_steps}_zeta{c.delta_theta_norm_bound}")
