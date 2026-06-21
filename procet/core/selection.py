"""Repair-target and Top-N vulnerable region selection.

Two families of selectors live here:

``select_repair_targets`` / ``select_repair_targets_with_pruning``
    Decide *which failed simplices* to push the repair loss on this iteration.

``select_top_n_v_safe_lbp`` / ``select_top_n_v_unsafe_lbp``
    Find the ``N`` already-certified regions whose margin is smallest — these
    are the regions the ProCeT protection constraints will keep certified.
"""

import numpy as np
import torch

from .lbp_loss import compute_min_L_with_mccormick, compute_h_max_via_network_bounds


# ---------------------------------------------------------------------------
# Repair-target selection
# ---------------------------------------------------------------------------

def select_repair_targets(F_h_positive_in_unsafe, F_safe_cbf_violation,
                          F_depth_limit_reached_unsafe, F_depth_limit_reached_safe,
                          F_unsafe_cannot_split, current_phase):
    """Plain (no pruning) target selector used by CeT.

    Phase 1: only definitive failures.
    Phase 2: definitive + uncertain failures.
    """
    if current_phase == 1:
        return list(F_safe_cbf_violation), list(F_h_positive_in_unsafe), "Definitive_only"
    return (
        list(F_safe_cbf_violation) + list(F_depth_limit_reached_safe),
        list(F_h_positive_in_unsafe) + list(F_unsafe_cannot_split) + list(F_depth_limit_reached_unsafe),
        "All",
    )


def select_repair_targets_with_pruning(
    F_h_positive_in_unsafe, F_safe_cbf_violation,
    F_depth_limit_reached_unsafe, F_depth_limit_reached_safe,
    F_unsafe_cannot_split, current_phase,
    model=None, dynamics_model=None, lbp_linearizer=None,
    device=None, dtype=None, total_region_count=0,
):
    """LBP-aware repair-target selector used by α/β-ProCeT.

    When the total region count is huge (> 100k), only the 50 % easiest-to-fix
    uncertain regions are kept, ranked by their LBP bounds:
        - F_depth_limit_reached_unsafe: 50 % smallest ``h_max``
        - F_depth_limit_reached_safe  : 50 % largest  ``min_L``
    Otherwise identical to ``select_repair_targets`` in phase 2.
    """
    REGION_THRESHOLD = 100_000

    if current_phase == 1:
        return list(F_safe_cbf_violation), list(F_h_positive_in_unsafe), "Phase1_Definitive"

    if (total_region_count > REGION_THRESHOLD
            and model is not None and dynamics_model is not None and lbp_linearizer is not None):

        # F_depth_limit_reached_unsafe: keep 50 % with smallest h_max
        if len(F_depth_limit_reached_unsafe) > 0:
            n_select = max(1, int(len(F_depth_limit_reached_unsafe) * 0.5))
            h_max_values = []
            BATCH = 256
            for bs in range(0, len(F_depth_limit_reached_unsafe), BATCH):
                be = min(bs + BATCH, len(F_depth_limit_reached_unsafe))
                h_max_batch = compute_h_max_via_network_bounds(
                    F_depth_limit_reached_unsafe[bs:be], dynamics_model, lbp_linearizer, device, dtype
                )
                h_max_values.append(h_max_batch.detach().cpu().numpy())
                del h_max_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            h_max_all = np.concatenate(h_max_values, axis=0)
            selected_indices = np.argsort(h_max_all)[:n_select]
            selected_depth_unsafe = [F_depth_limit_reached_unsafe[i] for i in selected_indices]
            print(f"    [Region Pruning] F_depth_limit_reached_unsafe: "
                  f"{len(F_depth_limit_reached_unsafe)} -> {len(selected_depth_unsafe)} "
                  f"(50%, h_max range: [{h_max_all.min():.4f}, {h_max_all.max():.4f}])")
        else:
            selected_depth_unsafe = list(F_depth_limit_reached_unsafe)

        # F_depth_limit_reached_safe: keep 50 % with largest min_L
        if len(F_depth_limit_reached_safe) > 0:
            n_select = max(1, int(len(F_depth_limit_reached_safe) * 0.5))
            min_L_values = []
            BATCH = 256
            for bs in range(0, len(F_depth_limit_reached_safe), BATCH):
                be = min(bs + BATCH, len(F_depth_limit_reached_safe))
                min_L_batch = compute_min_L_with_mccormick(
                    F_depth_limit_reached_safe[bs:be], dynamics_model, lbp_linearizer, device, dtype
                )
                min_L_values.append(min_L_batch.detach().cpu().numpy())
                del min_L_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            min_L_all = np.concatenate(min_L_values, axis=0)
            selected_indices = np.argsort(min_L_all)[-n_select:]
            selected_depth_safe = [F_depth_limit_reached_safe[i] for i in selected_indices]
            print(f"    [Region Pruning] F_depth_limit_reached_safe: "
                  f"{len(F_depth_limit_reached_safe)} -> {len(selected_depth_safe)} "
                  f"(50%, min_L range: [{min_L_all.min():.4f}, {min_L_all.max():.4f}])")
        else:
            selected_depth_safe = list(F_depth_limit_reached_safe)

        return (
            list(F_safe_cbf_violation) + selected_depth_safe,
            list(F_h_positive_in_unsafe) + list(F_unsafe_cannot_split) + selected_depth_unsafe,
            "Phase2_Sampled",
        )

    return (
        list(F_safe_cbf_violation) + list(F_depth_limit_reached_safe),
        list(F_h_positive_in_unsafe) + list(F_unsafe_cannot_split) + list(F_depth_limit_reached_unsafe),
        "Phase2_All",
    )


# ---------------------------------------------------------------------------
# Top-N vulnerable region selection (protection snapshots)
# ---------------------------------------------------------------------------

def select_top_n_v_safe_lbp(model, V_safe, dynamics_model, lbp_linearizer, top_n, cbf_margin=0.0):
    """Top-N safe regions with the smallest ``min_L`` margin.

    Returns ``(selected_regions, selected_min_L_margins)``. The margins are
    reused as ``ρ_safe`` values downstream in the SOCP constraint.
    """
    if len(V_safe) == 0:
        return [], np.array([])
    n_available = len(V_safe)
    actual_n = min(top_n, n_available)
    BATCH = 256
    device = next(model.parameters()).device
    dtype = torch.float32

    all_margins = []
    for bs in range(0, n_available, BATCH):
        be = min(bs + BATCH, n_available)
        batch = V_safe[bs:be]
        try:
            min_L = compute_min_L_with_mccormick(batch, dynamics_model, lbp_linearizer, device, dtype)
            margins = min_L.detach().cpu().numpy() if isinstance(min_L, torch.Tensor) else np.array(min_L)
        except (ValueError, RuntimeError) as e:
            print(f"    [Warning] compute_min_L failed for safe batch [{bs}:{be}]: {e}")
            margins = np.full(len(batch), 1e10)
        all_margins.append(margins)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    margins = np.concatenate(all_margins, axis=0)
    if actual_n == n_available:
        selected_indices = list(range(n_available))
    else:
        selected_indices = np.argsort(margins)[:actual_n].tolist()
    return [V_safe[i] for i in selected_indices], margins[selected_indices]


def select_top_n_v_unsafe_lbp(model, V_unsafe, dynamics_model, lbp_linearizer, top_n):
    """Top-N unsafe regions with the largest ``h_max`` (closest to ``h ≥ 0``).

    Returns ``(selected_regions, selected_h_max)``. ``-h_max`` is reused as
    ``ρ_unsafe`` downstream in the SOCP constraint.
    """
    if len(V_unsafe) == 0:
        return [], np.array([])
    n_available = len(V_unsafe)
    actual_n = min(top_n, n_available)
    BATCH = 256
    device = next(model.parameters()).device
    dtype = torch.float32

    all_h_max = []
    for bs in range(0, n_available, BATCH):
        be = min(bs + BATCH, n_available)
        batch = V_unsafe[bs:be]
        try:
            h_max = compute_h_max_via_network_bounds(batch, dynamics_model, lbp_linearizer, device, dtype)
            h_max_np = h_max.detach().cpu().numpy() if isinstance(h_max, torch.Tensor) else np.array(h_max)
        except (ValueError, RuntimeError) as e:
            print(f"    [Warning] compute_h_max failed for unsafe batch [{bs}:{be}]: {e}")
            h_max_np = np.full(len(batch), -1e10)
        all_h_max.append(h_max_np)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    h_max_all = np.concatenate(all_h_max, axis=0).ravel()
    if actual_n == n_available:
        selected_indices = list(range(n_available))
    else:
        selected_indices = np.argsort(h_max_all)[-actual_n:].tolist()
    return [V_unsafe[i] for i in selected_indices], h_max_all[selected_indices]
