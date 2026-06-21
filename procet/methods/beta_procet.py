"""β-ProCeT — Adaptive two-phase repair (CeT → α-ProCeT).

Strategy: start cheap, escalate only when needed.

  Phase 1 (CeT mode): direct gradient updates, no Jacobian computation,
      no SOCP. Fast per-step but offers no protection guarantee. Runs the
      protection *audit* on Top-N simplices after each step so we can still
      measure protection retention.

  Phase 2 (ProCeT mode): triggered the first time the patience counter hits
      1 in Phase 1. The triggering iteration is *backtracked* — pre-inner-loop
      parameters are restored — and the inner loop is re-run with the full
      α-ProCeT machinery (Top-N snapshot, Jacobians, SOCP). All subsequent
      iterations stay in Phase 2.

The escalation is one-way: once in Phase 2, the method never returns to
Phase 1. This is the algorithmic difference vs α-ProCeT (which is in
"Phase 2" from iteration 1).

Paper name:    ``\\textsc{$\\beta$-ProCeT}``
CLI key:       ``beta-procet``  (alias: ``aprocet``, ``beta_procet``)
Supported activations: ``Tanh``, ``Sigmoid``
"""

import numpy as np
import torch

from ..core.jacobian import compute_jacobian_for_lbp_bounds_v1
from ..core.lbp_loss import simple_gradient_update
from ..core.selection import select_repair_targets_with_pruning
from ..core.socp import socp_project_and_update_formula9
from .base import IterationContext, MethodConfig, RepairMethod


class BetaProCeTMethod(RepairMethod):
    """β-ProCeT: start in CeT mode, escalate to α-ProCeT on first plateau."""

    name = "beta-procet"
    display_name = "β-ProCeT"
    supported_activations = ("Tanh", "Sigmoid")
    output_dir_suffix = "betaProCeT_prab"

    def __init__(self, cfg: MethodConfig):
        super().__init__(cfg)
        # Mutable phase flag — flipped exactly once, in ``on_iteration_end``.
        self.use_protection = False

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
        # Top-N is needed in BOTH phases — Phase 1 audits the snapshot,
        # Phase 2 uses it for both protection and audit.
        return True

    def audits(self) -> bool:
        return True

    def get_lr(self, max_depth_limit: int) -> float:
        # Original main_aProCeT_prab.py: CeT-mode and ProCeT-mode use different maps.
        if self.use_protection:
            return {12: 2e-2, 14: 2e-2}.get(max_depth_limit, self.cfg.lr)
        return {12: 5e-3, 14: 5e-3}.get(max_depth_limit, self.cfg.lr)

    def get_beta(self):
        return 0.999, 0.999

    def prepare_inner_loop(self, ctx: IterationContext) -> None:
        """Compute Jacobians only when in Phase 2."""
        if not self.use_protection:
            return
        print(f"\n[Iteration {ctx.iteration + 1}.2] Computing Jacobian matrices (ProCeT mode)...")
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
        if not self.use_protection:
            # Phase 1: plain gradient step.
            update_norm = simple_gradient_update(ctx.model, g_F_clipped, current_lr)
            return g_F_clipped.norm().item(), update_norm

        # Phase 2: same SOCP as α-ProCeT.
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

    def on_iteration_end(self, ctx: IterationContext, certified_pct: float,
                         prev_certified_pct, patience_counter: int) -> dict:
        """Trigger the one-way CeT → ProCeT escalation.

        Fires the first time patience hits 1 *while still in Phase 1*. The
        runner will then: restore the pre-iteration params, re-prepare the
        inner loop (now computing Jacobians), re-run the inner loop in SOCP
        mode, and re-verify.
        """
        if patience_counter >= 1 and not self.use_protection:
            self.use_protection = True
            return {"backtrack_and_redo": True, "reset_patience": True}
        return {}

    def current_mode_label(self) -> str:
        """Human-readable current phase (for logs and result JSON)."""
        return "ProCeT" if self.use_protection else "CeT"

    # Same filename encoding as α-ProCeT — both methods expose K and ζ.
    def result_file_stem(self) -> str:
        c = self.cfg
        return (f"result_{c.system}_{c.activation}_{self.output_dir_suffix}"
                f"_depth{c.max_depth_limit}_w{c.lambda_}"
                f"_k{c.num_inner_steps}_zeta{c.delta_theta_norm_bound}")
