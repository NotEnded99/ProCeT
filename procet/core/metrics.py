"""Simplex volume, safety metrics, and JSON encoding helpers."""

import json
import math

import numpy as np
import torch


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy / torch scalar types.

    ``safety_metrics`` and inner-step logs surface ``numpy.float32`` /
    ``numpy.int64`` / ``torch.Tensor`` values, which the default encoder
    rejects.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Simplex volume
# ---------------------------------------------------------------------------

def compute_simplex_volume(simplex):
    """Volume (area in 2-D, volume in n-D) of a simplex given as [n+1, n] verts."""
    if hasattr(simplex, "cpu"):
        simplex = simplex.detach().cpu().numpy()
    num_vertices = simplex.shape[0]
    n = simplex.shape[1]

    if n == 0:
        return 0.0
    if num_vertices != n + 1:
        raise ValueError(f"Invalid simplex shape: expected [n+1, n], got {simplex.shape}")

    origin = simplex[0]
    vectors = simplex[1:] - origin
    det = np.linalg.det(vectors)
    return abs(det) / math.factorial(n)


def compute_total_volume(simplices_list):
    if not simplices_list:
        return 0.0
    return sum(compute_simplex_volume(s) for s in simplices_list)


# ---------------------------------------------------------------------------
# Safety metrics
# ---------------------------------------------------------------------------

def compute_safety_metrics_v8(
    V_safe,
    V_unsafe,
    F_h_positive_in_unsafe,
    F_safe_cbf_violation,
    F_depth_limit_reached_unsafe,
    F_depth_limit_reached_safe,
    F_unsafe_cannot_split,
):
    """Volume-aware safety metrics based on the harmonic mean of
    safe-side and unsafe-side pass rates.

    The harmonic mean penalises lopsided certificates: a model that
    certifies 100 % of the safe set but only 50 % of the unsafe set
    scores ``H = 2·1·0.5 / (1+0.5) ≈ 66.7 %``, not 100 %.
    """
    volume_v_safe            = compute_total_volume(V_safe)
    volume_v_unsafe          = compute_total_volume(V_unsafe)
    volume_f_h               = compute_total_volume(F_h_positive_in_unsafe)
    volume_f_safe_violation  = compute_total_volume(F_safe_cbf_violation)
    volume_f_depth_unsafe    = compute_total_volume(F_depth_limit_reached_unsafe)
    volume_f_depth_safe      = compute_total_volume(F_depth_limit_reached_safe)
    volume_f_unsafe_split    = compute_total_volume(F_unsafe_cannot_split)

    total_volume = (
        volume_v_safe + volume_v_unsafe + volume_f_h + volume_f_safe_violation
        + volume_f_depth_unsafe + volume_f_depth_safe + volume_f_unsafe_split
    )
    total_uncertain_volume = (
        volume_f_depth_unsafe + volume_f_depth_safe + volume_f_unsafe_split
    )

    true_safe_volume   = volume_v_safe + volume_f_safe_violation + volume_f_depth_unsafe
    true_unsafe_volume = volume_v_unsafe + volume_f_h + volume_f_depth_safe

    R_safe   = volume_v_safe / true_safe_volume   if true_safe_volume > 0   else 0.0
    R_unsafe = volume_v_unsafe / true_unsafe_volume if true_unsafe_volume > 0 else 0.0

    HarmonicMeanPassRate = (
        2.0 * R_safe * R_unsafe / (R_safe + R_unsafe)
        if (R_safe + R_unsafe) > 0 else 0.0
    )

    standard_pass_rate = (
        (volume_v_safe + volume_v_unsafe) / total_volume * 100
        if total_volume > 0 else 0.0
    )
    unsafe_intersect_volume = volume_v_unsafe + volume_f_h
    usr = (
        volume_v_unsafe / unsafe_intersect_volume * 100
        if unsafe_intersect_volume > 0 else 0.0
    )
    f_h_ratio = (
        volume_f_h / unsafe_intersect_volume * 100
        if unsafe_intersect_volume > 0 else 0.0
    )
    uncertainty_ratio = (
        total_uncertain_volume / total_volume * 100
        if total_volume > 0 else 0.0
    )

    return {
        "R_safe": R_safe,
        "R_unsafe": R_unsafe,
        "HarmonicMeanPassRate": HarmonicMeanPassRate,
        "true_safe_volume": true_safe_volume,
        "true_unsafe_volume": true_unsafe_volume,
        "standard_pass_rate": standard_pass_rate,
        "usr": usr,
        "f_h_ratio": f_h_ratio,
        "uncertainty_ratio": uncertainty_ratio,
        "unsafe_intersect_volume": unsafe_intersect_volume,
        "total_volume": total_volume,
        "volumes": {
            "V_safe":          volume_v_safe,
            "V_unsafe":        volume_v_unsafe,
            "F_h":             volume_f_h,
            "F_safe_violation":volume_f_safe_violation,
            "F_depth_unsafe":  volume_f_depth_unsafe,
            "F_depth_safe":    volume_f_depth_safe,
            "F_unsafe_split":  volume_f_unsafe_split,
            "total_uncertain": total_uncertain_volume,
        },
    }
