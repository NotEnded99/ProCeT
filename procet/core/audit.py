"""Protection audit.

After every inner-step update, recompute LBP bounds on the *protected*
simplices (the Top-N most vulnerable V_safe / V_unsafe regions selected at
the start of the iteration) and check whether they are still certified:

    Safe region   certified ⟺  min_L > 0     (CBF condition lower bound)
    Unsafe region certified ⟺  h_max < 0     (barrier function upper bound)

This is a *measurement* only — it does not affect the update. The success and
failure counts are aggregated into ``protection_log`` and saved alongside the
final result JSON so the paper can report protection retention rates.
"""

import numpy as np
import torch

from .lbp_loss import compute_min_L_with_mccormick, compute_h_max_via_network_bounds


def check_protection_status(model, top_n_v_safe, top_n_v_unsafe,
                            dynamics_model, lbp_linearizer, device, dtype,
                            batch_size=64):
    """Recompute bounds on the protected simplices under the CURRENT model
    parameters and report per-side certification counts and statistics.

    Returns:
        dict with keys:
            ``safe_{total,success,failure,min_L_min,min_L_mean}``
            ``unsafe_{total,success,failure,h_max_max,h_max_mean}``
    """
    result = {
        "safe_total":      len(top_n_v_safe),
        "safe_success":    0,
        "safe_failure":    0,
        "safe_min_L_min":  None,
        "safe_min_L_mean": None,
        "unsafe_total":      len(top_n_v_unsafe),
        "unsafe_success":    0,
        "unsafe_failure":    0,
        "unsafe_h_max_max":  None,
        "unsafe_h_max_mean": None,
    }

    # --- Safe regions: min_L must stay > 0 ---
    if len(top_n_v_safe) > 0:
        safe_min_L = []
        for bs in range(0, len(top_n_v_safe), batch_size):
            be = min(bs + batch_size, len(top_n_v_safe))
            batch = top_n_v_safe[bs:be]
            try:
                min_L = compute_min_L_with_mccormick(batch, dynamics_model, lbp_linearizer, device, dtype)
                min_L_np = min_L.detach().cpu().numpy()
            except (ValueError, RuntimeError) as e:
                print(f"    [Protection Audit] compute_min_L failed for safe batch [{bs}:{be}]: {e}")
                min_L_np = np.full(len(batch), np.nan)
            safe_min_L.append(min_L_np)
            del min_L
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        arr = np.concatenate(safe_min_L, axis=0)
        valid = arr[~np.isnan(arr)]
        result["safe_success"] = int(np.sum(arr > 0))
        result["safe_failure"] = int(np.sum(arr <= 0))
        if valid.size > 0:
            result["safe_min_L_min"]  = float(np.min(valid))
            result["safe_min_L_mean"] = float(np.mean(valid))

    # --- Unsafe regions: h_max (real barrier upper bound) must stay < 0 ---
    if len(top_n_v_unsafe) > 0:
        h_max_list = []
        for bs in range(0, len(top_n_v_unsafe), batch_size):
            be = min(bs + batch_size, len(top_n_v_unsafe))
            batch = top_n_v_unsafe[bs:be]
            try:
                h_max = compute_h_max_via_network_bounds(batch, dynamics_model, lbp_linearizer, device, dtype)
                h_max_np = h_max.detach().cpu().numpy()
            except (ValueError, RuntimeError) as e:
                print(f"    [Protection Audit] compute_h_max failed for unsafe batch [{bs}:{be}]: {e}")
                h_max_np = np.full(len(batch), np.nan)
            h_max_list.append(h_max_np)
            del h_max
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        arr = np.concatenate(h_max_list, axis=0)
        valid_h = arr[~np.isnan(arr)]
        result["unsafe_success"] = int(np.sum(arr < 0))
        result["unsafe_failure"] = int(np.sum(arr >= 0))
        if valid_h.size > 0:
            result["unsafe_h_max_max"]  = float(np.max(valid_h))
            result["unsafe_h_max_mean"] = float(np.mean(valid_h))

    return result
