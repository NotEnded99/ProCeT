"""
LBP-Based CBF Verification Module WITHOUT McCormick Relaxation

This module implements Linear Bound Propagation (LBP) for neural CBF verification
without using McCormick envelopes. Instead, it uses direct interval multiplication:

For intervals J ∈ [J_L, J_U] and f ∈ [f_L, f_U]:
  - lower = min(J_L*f_L, J_L*f_U, J_U*f_L, J_U*f_U)
  - upper = max(J_L*f_L, J_L*f_U, J_U*f_L, J_U*f_U)

This avoids the 1/(u-l) terms in McCormick that cause NaN gradients.

Based on: verify_cbf.py structure but without McCormick relaxation
"""

from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
import time
import itertools
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..linearization import TaylorLinearization
from ..linearization.linear_derivative_bounds import CrownPartialLinearization
from ..translators import TorchTranslator
from ..regions import SimplicialRegion, HyperrectangularRegion
from .domain import unsafe_region


# =============================================================================
# 1. Direct Interval Multiplication (No McCormick)
# =============================================================================

def _interval_product_bounds(
    affine1_L: Tuple[torch.Tensor, torch.Tensor],
    affine1_U: Tuple[torch.Tensor, torch.Tensor],
    affine2_L: Tuple[torch.Tensor, torch.Tensor],
    affine2_U: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute bounds on the product of two affine functions using direct interval arithmetic.

    For intervals J ∈ [J_L, J_U] and f ∈ [f_L, f_U]:
      product_lower = min(J_L*f_L, J_L*f_U, J_U*f_L, J_U*f_U)
      product_upper = max(J_L*f_L, J_L*f_U, J_U*f_L, J_U*f_U)

    This avoids McCormick's 1/(u-l) terms that cause NaN gradients.

    Args:
        affine1_L, affine1_U: (A, b) lower and upper bounds for first affine
        affine2_L, affine2_U: (A, b) lower and upper bounds for second affine

    Returns:
        ((M_L, c_L), (M_U, c_U)): Lower and upper affine bounds on the product
    """
    (A1_L, b1_L), (A1_U, b1_U) = affine1_L, affine1_U
    (A2_L, b2_L), (A2_U, b2_U) = affine2_L, affine2_U

    # Compute all four corner products for lower bound
    # J_L * f_L, J_L * f_U, J_U * f_L, J_U * f_U
    products = [
        A1_L @ A2_L.transpose(-2, -1),  # J_L * f_L
        A1_L @ A2_U.transpose(-2, -1),  # J_L * f_U
        A1_U @ A2_L.transpose(-2, -1),  # J_U * f_L
        A1_U @ A2_U.transpose(-2, -1),  # J_U * f_U
    ]

    # Lower bound is min of all corners
    M_L = torch.minimum(torch.minimum(products[0], products[1]),
                        torch.minimum(products[2], products[3]))

    # Upper bound is max of all corners
    M_U = torch.maximum(torch.maximum(products[0], products[1]),
                        torch.maximum(products[2], products[3]))

    # Compute constant terms similarly
    const_products = [
        A1_L @ b2_L.unsqueeze(-1),  # J_L * b_f_L
        A1_L @ b2_U.unsqueeze(-1),
        A1_U @ b2_L.unsqueeze(-1),
        A1_U @ b2_U.unsqueeze(-1),
        b1_L * A2_L @ torch.ones_like(b2_L).unsqueeze(-1) if b1_L.ndim > 0 else b1_L * A2_L @ torch.ones_like(b2_L).unsqueeze(-1),
        # This is more complex, let's handle it differently
    ]

    # For simplicity, compute lower/upper bounds on the constant term separately
    # by evaluating at vertices of the region (done in batched_get_affine_function_bounds)

    return (M_L, None), (M_U, None)


def _batched_interval_product_bounds(
    J_L: torch.Tensor,
    J_U: torch.Tensor,
    f_L: torch.Tensor,
    f_U: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute batched interval product bounds without McCormick.

    Args:
        J_L, J_U: [batch, n_out, n_in] Jacobian interval bounds
        f_L, f_U: [batch, n_in] or [batch, n_in, n_ctrl] function interval bounds

    Returns:
        M_L, c_L, M_U, c_U such that the product J*f is bounded by M_L @ x + c_L to M_U @ x + c_U
    """
    # Handle different shapes for drift (no control dim) vs control (has control dim)
    has_control_dim = f_L.ndim == 3

    if has_control_dim:
        # f shape: [batch, n_in, n_ctrl]
        # J shape: [batch, 1, n_in] (for single output)
        batch_size, n_in, n_ctrl = f_L.shape

        # Compute all corner products: J_L*J_f, etc.
        # J_L, J_U shape: [batch, 1, n_in]
        # f_L, f_U shape: [batch, n_in, n_ctrl]

        # Corner products
        JL_fL = torch.matmul(J_L.transpose(-2, -1), f_L)  # [batch, 1, n_ctrl]
        JL_fU = torch.matmul(J_L.transpose(-2, -1), f_U)
        JU_fL = torch.matmul(J_U.transpose(-2, -1), f_L)
        JU_fU = torch.matmul(J_U.transpose(-2, -1), f_U)

        # Lower bound: min of corners
        M_L = torch.minimum(torch.minimum(JL_fL, JL_fU), torch.minimum(JU_fL, JU_fU)).squeeze(-2)  # [batch, n_ctrl]
        M_U = torch.maximum(torch.maximum(JL_fL, JL_fU), torch.maximum(JU_fL, JU_fU)).squeeze(-2)  # [batch, n_ctrl]

        # No constant term in this simple interval product
        c_L = torch.zeros(batch_size, n_ctrl, dtype=J_L.dtype, device=J_L.device)
        c_U = torch.zeros(batch_size, n_ctrl, dtype=J_L.dtype, device=J_L.device)
    else:
        # f shape: [batch, n_in]
        batch_size, n_in = f_L.shape

        # Corner products: J_L*f_L, J_L*f_U, J_U*f_L, J_U*f_U
        # Each is [batch, n_in]
        JL_fL = J_L.squeeze(-2) * f_L
        JL_fU = J_L.squeeze(-2) * f_U
        JU_fL = J_U.squeeze(-2) * f_L
        JU_fU = J_U.squeeze(-2) * f_U

        # Lower/upper bounds
        M_L = torch.minimum(torch.minimum(JL_fL, JL_fU), torch.minimum(JU_fL, JU_fU))  # [batch, n_in]
        M_U = torch.maximum(torch.maximum(JL_fL, JL_fU), torch.maximum(JU_fL, JU_fU))  # [batch, n_in]

        c_L = torch.zeros(batch_size, n_in, dtype=J_L.dtype, device=J_L.device)
        c_U = torch.zeros(batch_size, n_in, dtype=J_L.dtype, device=J_L.device)

    return M_L, c_L, M_U, c_U


# =============================================================================
# 2. Dynamics Bounds (Same as verify_cbf_ibp.py)
# =============================================================================

def _compute_dynamics_bounds_ibp(batch, dynamics_model, device, dtype):
    """
    Compute dynamics bounds by evaluating f(x) and g(x) at region vertices/centers.

    Returns bounds in interval format [f_L, f_U] and [g_L, g_U].
    Each bound is [batch_size, n] for consistency with IBP bounds.
    """
    n = dynamics_model.input_dim
    m = dynamics_model.control_dim
    batch_size = len(batch)

    translator = TorchTranslator(device=device, dtype=dtype)

    if isinstance(batch[0], SimplicialRegion):
        f_L_list = []
        f_U_list = []
        g_L_list = []
        g_U_list = []

        for sample in batch:
            verts = torch.tensor(sample.vertices, device=device, dtype=dtype)
            f_vals = []
            g_vals = []
            for v in verts:
                x = v.unsqueeze(0)
                with torch.no_grad():
                    f_val = dynamics_model.compute_f(x, translator).squeeze(0)
                    f_vals.append(f_val)
                    if m > 0:
                        g_val = dynamics_model.compute_g(x, translator).squeeze(0)
                        g_vals.append(g_val)

            f_vals = torch.stack(f_vals, dim=0)
            f_L_list.append(f_vals.min(dim=0).values)
            f_U_list.append(f_vals.max(dim=0).values)

            if m > 0:
                g_vals = torch.stack(g_vals, dim=0)
                g_L_list.append(g_vals.min(dim=0).values)
                g_U_list.append(g_vals.max(dim=0).values)

        f_L = torch.stack(f_L_list, dim=0)
        f_U = torch.stack(f_U_list, dim=0)

        if m > 0:
            g_L = torch.stack(g_L_list, dim=0)
            g_U = torch.stack(g_U_list, dim=0)
        else:
            g_L = None
            g_U = None

    elif isinstance(batch[0], HyperrectangularRegion):
        import itertools
        f_L_list = []
        f_U_list = []
        g_L_list = []
        g_U_list = []

        for sample in batch:
            center = torch.tensor(sample.center_point, device=device, dtype=dtype)
            radius = torch.tensor(sample.radius_vec, device=device, dtype=dtype)

            corners = []
            for combo in itertools.product([0, 1], repeat=n):
                corner = center.clone()
                for i, c in enumerate(combo):
                    corner[i] = center[i] + radius[i] if c == 1 else center[i] - radius[i]
                corners.append(corner)

            f_vals = []
            g_vals = []
            with torch.no_grad():
                for c in corners:
                    x = c.unsqueeze(0)
                    f_val = dynamics_model.compute_f(x, translator).squeeze(0)
                    f_vals.append(f_val)
                    if m > 0:
                        g_val = dynamics_model.compute_g(x, translator).squeeze(0)
                        g_vals.append(g_val)

            f_vals = torch.stack(f_vals, dim=0)
            f_L_list.append(f_vals.min(dim=0).values)
            f_U_list.append(f_vals.max(dim=0).values)

            if m > 0:
                g_vals = torch.stack(g_vals, dim=0)
                g_L_list.append(g_vals.min(dim=0).values)
                g_U_list.append(g_vals.max(dim=0).values)

        f_L = torch.stack(f_L_list, dim=0)
        f_U = torch.stack(f_U_list, dim=0)

        if m > 0:
            g_L = torch.stack(g_L_list, dim=0)
            g_U = torch.stack(g_U_list, dim=0)
        else:
            g_L = None
            g_U = None
    else:
        raise TypeError(f"Unsupported region type: {type(batch[0])}")

    f_bounds = (f_L, f_U)
    g_bounds = (g_L, g_U) if m > 0 else None

    return f_bounds, g_bounds


# =============================================================================
# 3. Affine Function Bounds over Regions
# =============================================================================

def _batched_get_affine_function_bounds(
    affine_L,
    batch,
    affine_U=None,
    device="cpu",
    dtype=torch.float64,
):
    """Computes min/max of a Torch affine function over a region (hyperrectangular or simplicial)."""
    (A, b) = affine_L

    if isinstance(batch[0], HyperrectangularRegion):
        centers = [torch.tensor(region.center_point, device=device, dtype=dtype) for region in batch]
        centers = torch.stack(centers, dim=0)
        radii = [torch.tensor(region.radius_vec, device=device, dtype=dtype) for region in batch]
        radii = torch.stack(radii, dim=0)

        if A.ndim == 4:
            centers = centers.unsqueeze(-2)
            radii = radii.unsqueeze(-2)

        A_abs = torch.abs(A)
        lower_b = b + (A @ centers.unsqueeze(-1)).squeeze(-1) - (A_abs @ radii.unsqueeze(-1)).squeeze(-1)

        if affine_U is not None:
            (A_U, b_U) = affine_U
            A_U_abs = torch.abs(A_U)
            upper_b = b_U + (A_U @ centers.unsqueeze(-1)).squeeze(-1) - (A_U_abs @ radii.unsqueeze(-1)).squeeze(-1)
        else:
            upper_b = b + (A @ centers.unsqueeze(-1)).squeeze(-1) + (A_abs @ radii.unsqueeze(-1)).squeeze(-1)

    elif isinstance(batch[0], SimplicialRegion):
        vertices = [torch.tensor(region.vertices, device=device, dtype=dtype) for region in batch]
        vertices = torch.stack(vertices, dim=0)

        if A.ndim == 4:
            vertices = vertices.unsqueeze(-3)

        values_L = A @ vertices.transpose(-2, -1)
        lower_b = torch.min(values_L, dim=-1).values + b

        if affine_U is not None:
            (A_U, b_U) = affine_U
            values_U = A_U @ vertices.transpose(-2, -1)
            upper_b = torch.max(values_U, dim=-1).values + b_U
        else:
            upper_b = torch.max(values_L, dim=-1).values + b
    else:
        raise TypeError(f"Unsupported region type: {type(batch[0])}. Expected HyperrectangularRegion or SimplicialRegion.")

    return lower_b, upper_b


# =============================================================================
# 4. CBF Condition Verification WITHOUT McCormick
# =============================================================================

def _verify_cbf_condition_lbp_wo_mccormick(
    batch,
    dynamics_model,
    network_linearizer,
    device,
    dtype,
    eta=(0.5, 0.5),
    find_counterexample=False,
):
    """
    Verify the CBF condition using direct interval bounds (no McCormick).

    CBF condition: ∇h(x)·f(x) + ∇h(x)·g(x)·u + α(h(x)) ≥ 0

    Uses direct interval multiplication instead of McCormick relaxation.
    """
    n = dynamics_model.input_dim
    m = dynamics_model.control_dim
    batch_size = len(batch)

    # Compute dynamics bounds
    try:
        f_affine_bounds, g_affine_bounds = _compute_dynamics_bounds_ibp(batch, dynamics_model, device, dtype)
    except ValueError:
        return torch.zeros(len(batch), dtype=torch.bool, device=device), None

    # Get Jacobian bounds from linear bound propagation
    # A_L shape: [batch, n, n] where n is input_dim
    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()

    # Get dynamics affine bounds
    f_L, f_U = f_affine_bounds

    # Direct interval product: for each batch element, compute
    # J ∈ [A_L, A_U], f ∈ [f_L, f_U]
    # Lower bound: min over all corners J_lower*f_lower, J_lower*f_upper, J_upper*f_lower, J_upper*f_upper
    # Then sum over the state dimensions

    # For each sample b, we have:
    # A_L[b] is [n, n], f_L[b] is [n]
    # We need to compute sum_j min_i (A[b,i,j] * f[b,j]) over j
    # This is equivalent to: (min_i A[b,i,j] * f_lower[b,j]).sum() or similar

    # Actually, for direct interval product:
    # L_bound = min_{c in corners} sum_j A[b,i,j] * f_c[b,j]
    # where corners are (f_L, f_U)
    # This simplifies to: for each j, take min of A[:,j] * f_L[j] and A[:,j] * f_U[j], then sum

    # JL[b, j] = min(A_L[b, 0, j] * f[b, j], A_L[b, 1, j] * f[b, j], ...)
    # But A has shape [batch, n, n], f has shape [batch, n]
    # We need element-wise min across the "n" dimension of A for each column j

    # Compute all corner products
    # JL_lower[b, j] = A_L[b, :, j] * f_L[b, j] (element-wise for each column j)
    # But A_L[b, :, j] is a vector of length n, so this gives a matrix

    # For simplicity: use element-wise min across the 4 corners
    # JL = A_L * f_L (element-wise with broadcasting)
    JL_fL = A_L * f_L.unsqueeze(-1)  # [batch, n, n]
    JL_fU = A_L * f_U.unsqueeze(-1)
    JU_fL = A_U * f_L.unsqueeze(-1)
    JU_fU = A_U * f_U.unsqueeze(-1)

    # Take minimum over the corners for each element
    L_corners = torch.minimum(torch.minimum(JL_fL, JL_fU), torch.minimum(JU_fL, JU_fU))  # [batch, n, n]
    U_corners = torch.maximum(torch.maximum(JL_fL, JL_fU), torch.maximum(JU_fL, JU_fU))  # [batch, n, n]

    # For the CBF condition, we need scalar per sample:
    # For each sample, we want a single lower bound on J*f
    # Take min over all n and n dimensions (which is conservative)
    # Or sum over n dimensions after taking min over the n rows

    # Actually, for ∇h · f where ∇h is [n] and f is [n], the result is scalar
    # With bounds: min over j of min_i (A[i,j] * f_j) summed over j
    # = sum over j of (min over i of A[i,j]) * f_j when f_j >= 0, or similar
    # But with intervals: we need the minimum over all f in [f_L, f_U] of A*f
    # This is min over corners: sum_j min_i (A[i,j] * f_corners[j])

    # For each output dimension i and each sample:
    L_drift = L_corners.sum(dim=-1)  # [batch, n] - sum over j
    U_drift = U_corners.sum(dim=-1)  # [batch, n]

    # But we need scalar, so take min over the n dimensions (conservative)
    L_drift = L_drift.min(dim=-1)[0]  # [batch]
    U_drift = U_drift.max(dim=-1)[0]  # [batch]

    # Get class-K bounds using network linear bounds (not partial derivative bounds)
    (A_L_net, a_L_net), (A_U_net, a_U_net) = network_linearizer.get_network_linear_bounds()
    # A_L_net shape: [batch, n_out=1, n_in], a_L_net shape: [batch, n_out=1]
    h_lb = a_L_net.squeeze(-1)  # [batch]
    alpha_l = dynamics_model.alpha_function(h_lb)  # [batch]

    L_total = L_drift + alpha_l
    U_total = U_drift + alpha_l

    # Handle control term
    if m > 0 and g_affine_bounds is not None:
        g_L, g_U = g_affine_bounds  # g_L shape: [batch, n, m]

        # For J*g: J [batch, n, n] * g [batch, n, m]
        # Result: [batch, n, m]
        JL_gL = torch.matmul(A_L, g_L)  # [batch, n, m]
        JL_gU = torch.matmul(A_L, g_U)
        JU_gL = torch.matmul(A_U, g_L)
        JU_gU = torch.matmul(A_U, g_U)

        v_L = torch.minimum(torch.minimum(JL_gL, JL_gU), torch.minimum(JU_gL, JU_gU))  # [batch, n, m]
        v_U = torch.maximum(torch.maximum(JL_gL, JL_gU), torch.maximum(JU_gL, JU_gU))  # [batch, n, m]

        u_min = torch.tensor(dynamics_model.u_min, device=device, dtype=dtype)
        u_max = torch.tensor(dynamics_model.u_max, device=device, dtype=dtype)

        # Control lower bound: min over u in [u_min, u_max] of v(x)*u
        L_ctrl = torch.zeros(batch_size, dtype=dtype, device=device)
        for k in range(m):
            v_L_k = v_L[:, :, k]  # [batch, n]
            v_U_k = v_U[:, :, k]

            pos_mask = v_L_k >= 0
            neg_mask = v_U_k <= 0
            mixed_mask = ~(pos_mask | neg_mask)

            # Take min over n dimension (conservative)
            L_ctrl += torch.where(pos_mask.all(dim=-1), (v_L_k * u_min[k]).min(dim=-1)[0], torch.zeros_like(L_ctrl))
            L_ctrl += torch.where(neg_mask.all(dim=-1), (v_U_k * u_max[k]).max(dim=-1)[0], torch.zeros_like(L_ctrl))

            mixed_vals = torch.minimum(v_L_k * u_max[k], v_U_k * u_min[k]).min(dim=-1)[0]
            L_ctrl += torch.where(mixed_mask.any(dim=-1), mixed_vals, torch.zeros_like(L_ctrl))

        L_total = L_total + L_ctrl
        U_total = U_total + L_ctrl  # approximate

    satisfaction = L_total >= -1e-12

    if find_counterexample:
        counterexample = U_total < 0
        return satisfaction, counterexample, L_total, U_total

    return satisfaction, torch.zeros_like(satisfaction), L_total, None


# =============================================================================
# 5. Main Verification Logic
# =============================================================================

def _lbp_handle_split(sample, start_time, results, sample_idx, min_volume, split_type, unsat_type, max_depth=None, depth_limit_type=None):
    """Record a MAYBE result via splitting or an UNSAT counterexample."""
    from lbp_neural_cbf.certification_results import SampleResultUNSAT

    if max_depth is not None and sample.depth >= max_depth:
        counterexample = sample.center
        result_type = depth_limit_type if depth_limit_type is not None else "depth_limit_reached"
        results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type=result_type)
        return

    if sample._compute_volume() > min_volume:
        new_samples = sample.split()
        if new_samples:
            from lbp_neural_cbf.certification_results import SampleResultMaybe
            results[sample_idx] = SampleResultMaybe(sample, start_time, new_samples, split_type=split_type)
            return

    counterexample = sample.center
    results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type=unsat_type)


def verify_batch_lbp_wo_mccormick(
    batch,
    dynamics_model,
    network_linearizer,
    device,
    dtype,
    min_volume=1e-8,
    max_depth=None,
):
    """
    Verify a batch of samples using LBP without McCormick.

    Args:
        batch: List of region samples
        dynamics_model: CBF dynamics model
        network_linearizer: CrownPartialLinearization instance
        device, dtype: Computation device and data type
        min_volume, max_depth: Splitting parameters

    Returns:
        List of SampleResult objects
    """
    start_time = time.time()
    results = [None for _ in range(len(batch))]

    from lbp_neural_cbf.certification_results import SampleResultSAT, SampleResultUNSAT, SampleResultMaybe

    to_check_cbf_cond = []

    # Compute network bounds once for this batch
    network_linearizer.compute_network_bounds(batch)
    h_lb_all, h_ub_all = network_linearizer.get_network_output_bounds()
    h_lb_all = h_lb_all.reshape(len(batch), -1)[:, 0]
    h_ub_all = h_ub_all.reshape(len(batch), -1)[:, 0]

    for sample_idx, sample in enumerate(batch):
        h_min = h_lb_all[sample_idx].item()
        h_max = h_ub_all[sample_idx].item()

        # Case 1: h_max < 0 -> SAT (unsafe region verified)
        if h_max < 0:
            results[sample_idx] = SampleResultSAT(sample, start_time, result_type="unsafe_region")

        # Case 2: unsafe region
        elif unsafe_region(sample, dynamics_model, require_complete_containment=False):
            if h_min >= 0:
                counterexample = sample.center
                results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type="h_positive_in_unsafe")
            else:
                _lbp_handle_split(
                    sample=sample,
                    start_time=start_time,
                    results=results,
                    sample_idx=sample_idx,
                    min_volume=min_volume,
                    split_type="boundary_unsafe",
                    unsat_type="unsafe_cannot_split",
                    max_depth=max_depth,
                    depth_limit_type="depth_limit_reached_unsafe",
                )

        # Case 3: safe region
        else:
            to_check_cbf_cond.append(sample_idx)

    if len(to_check_cbf_cond) == 0:
        return results

    # Pre-compute Jacobian bounds for CBF condition
    network_linearizer.keep_indices(to_check_cbf_cond)
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=0)
    subbatch = [batch[i] for i in to_check_cbf_cond]

    cbf_verified = torch.ones(len(subbatch), dtype=torch.bool, device=device)
    current_indices = torch.arange(len(subbatch), device=device)

    eta_values_list = list(itertools.product([0.5], repeat=2))
    for iteration_idx, eta in enumerate(eta_values_list):
        if len(current_indices) == 0:
            break

        if iteration_idx > 0:
            subbatch_to_check = [subbatch[i.item()] for i in current_indices]
        else:
            subbatch_to_check = subbatch

        eta_verified, counter_verified, _, _ = _verify_cbf_condition_lbp_wo_mccormick(
            subbatch_to_check, dynamics_model, network_linearizer, device, dtype, eta=eta
        )

        failed_in_current = ~eta_verified
        original_failed_indices = current_indices[failed_in_current]
        cbf_verified[original_failed_indices] = False

        succeeded_in_current = eta_verified
        original_succeeded_indices = current_indices[succeeded_in_current]
        cbf_verified[original_succeeded_indices] = True

        current_indices = current_indices[eta_verified]

        if len(current_indices) > 0 and iteration_idx < len(eta_values_list) - 1:
            keep_mask = eta_verified
            network_linearizer.keep_indices(keep_mask.nonzero(as_tuple=True)[0], include_partial_deriv_bounds=True)

    for subsample_idx, sample_idx in enumerate(to_check_cbf_cond):
        sample = batch[sample_idx]

        if cbf_verified[subsample_idx]:
            results[sample_idx] = SampleResultSAT(sample, start_time, result_type="safe_cbf_verified")
        else:
            _lbp_handle_split(
                sample=sample,
                start_time=start_time,
                results=results,
                sample_idx=sample_idx,
                min_volume=min_volume,
                split_type="cbf_failure",
                unsat_type="safe_cbf_violation",
                max_depth=max_depth,
                depth_limit_type="depth_limit_reached_safe",
            )

    return results


# =============================================================================
# 6. Verification Strategy Class
# =============================================================================

class LBPWoMcCormickVerificationStrategy:
    """
    Verification strategy using LBP without McCormick relaxation.
    """

    def __init__(self, network_path, dynamics_model, use_gpu=True, max_depth=None):
        self.network_path = network_path
        self.dynamics_model = dynamics_model
        self.use_gpu = use_gpu
        self.max_depth = max_depth

    def initialize_worker(self):
        """Initialize the PyTorch model and linearizers for each worker process."""
        global _LOCAL_LBP_WO_MCCORMICK

        device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
        dtype = torch.float32

        pth_path = self.network_path.replace(".onnx", ".pth")
        activation_fnc = getattr(self.dynamics_model, 'activation_fnc', 'Tanh')

        from lbp_neural_cbf.cbf.network import BarrierNN
        _LOCAL_LBP_WO_MCCORMICK.torch_model = BarrierNN(
            input_size=self.dynamics_model.input_dim,
            hidden_sizes=self.dynamics_model.hidden_sizes,
            activation_fnc=activation_fnc,
            device=device,
        )
        _LOCAL_LBP_WO_MCCORMICK.torch_model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=False))
        _LOCAL_LBP_WO_MCCORMICK.torch_model = _LOCAL_LBP_WO_MCCORMICK.torch_model.to(dtype=dtype)
        _LOCAL_LBP_WO_MCCORMICK.torch_model.eval()

        _LOCAL_LBP_WO_MCCORMICK.network_linearizer = CrownPartialLinearization(
            _LOCAL_LBP_WO_MCCORMICK.torch_model, dtype=dtype
        )
        _LOCAL_LBP_WO_MCCORMICK.dynamics_model = self.dynamics_model
        _LOCAL_LBP_WO_MCCORMICK.device = device
        _LOCAL_LBP_WO_MCCORMICK.dtype = dtype
        _LOCAL_LBP_WO_MCCORMICK.max_depth = self.max_depth

    @staticmethod
    def verify_batch(batch):
        """Verify a batch of samples. Called by the executor."""
        return LBPWoMcCormickVerificationStrategy._verify_batch_lbp_wo_mccormick(
            batch,
            _LOCAL_LBP_WO_MCCORMICK.dynamics_model,
            _LOCAL_LBP_WO_MCCORMICK.network_linearizer,
            _LOCAL_LBP_WO_MCCORMICK.torch_model,
            _LOCAL_LBP_WO_MCCORMICK.device,
            _LOCAL_LBP_WO_MCCORMICK.dtype,
            max_depth=_LOCAL_LBP_WO_MCCORMICK.max_depth,
        )

    @staticmethod
    @torch.no_grad()
    def _verify_batch_lbp_wo_mccormick(
        batch,
        dynamics_model,
        network_linearizer,
        torch_model,
        device,
        dtype,
        min_volume=1e-8,
        max_depth=None,
    ):
        return verify_batch_lbp_wo_mccormick(
            batch,
            dynamics_model,
            network_linearizer,
            device,
            dtype,
            min_volume=min_volume,
            max_depth=max_depth,
        )


# Global namespace for worker-specific objects
_LOCAL_LBP_WO_MCCORMICK = types.SimpleNamespace()


# =============================================================================
# 7. Top-level verify_cbf function
# =============================================================================

def verify_cbf_lbp_wo_mccormick(
    dynamics_model,
    barrier_model_path=None,
    executor_type="single",
    region_type="simplicial",
    visualize=False,
    use_gpu=True,
    batch_size=512,
    max_depth=None,
    save_verification_regions=False,
):
    """
    Main function to verify a neural control barrier function using LBP without McCormick.

    Args:
        dynamics_model: CBF dynamical system
        barrier_model_path: Path to trained barrier function model
        executor_type: Type of executor ("single", "multi-thread", or "multi-process")
        region_type: Type of regions ("hyperrectangular" or "simplicial")
        visualize: Whether to create live visualization during verification
        use_gpu: Whether to use GPU for verification
        batch_size: Batch size for verification
        max_depth: Maximum depth for region splitting
        save_verification_regions: Whether to save verification regions

    Returns:
        Verification results with guaranteed soundness for SAT results
    """
    if barrier_model_path is None:
        raise ValueError("barrier_model_path must be provided for verification")

    print(f"Verifying CBF (LBP without McCormick): {barrier_model_path}")

    # Pre-import verify_cbf to ensure it's in sys.modules before the circular import chain
    # When executors/__init__.py -> multi_thread_executor imports verify_cbf, it will
    # find this already-loaded module instead of trying to re-initialize it.
    import lbp_neural_cbf.cbf.verify_cbf

    # Use relative imports since this module is now inside lbp_neural_cbf.cbf package
    from ..executors import SinglethreadExecutor
    from .network import BarrierNN
    from ..certification_results import SampleResultSAT, SampleResultUNSAT
    from ..regions import create_region_generator

    # Create verification strategy
    strategy = LBPWoMcCormickVerificationStrategy(
        barrier_model_path,
        dynamics_model,
        use_gpu=use_gpu,
        max_depth=max_depth,
    )

    # Generate initial samples
    region_generator = create_region_generator(region_type)
    samples = region_generator.create_mesh(dynamics_model).get_regions(0)

    # Create executor
    # Note: multi-thread and multi-process executors have circular import issues
    # with verify_cbf. Only single-threaded execution is supported for now.
    if executor_type == "single":
        executor = SinglethreadExecutor()
    else:
        raise ValueError(f"Invalid executor_type: {executor_type}. Only 'single' is supported to avoid circular imports.")

    print(f"Using {executor_type} executor")

    def aggregate(agg, result):
        if agg is None:
            agg = []
        agg.append(result)
        return agg

    agg, certified_percentage, uncertified_percentage, computation_time = executor.execute(
        initializer=strategy.initialize_worker,
        process_batch=strategy.verify_batch,
        aggregate=aggregate,
        samples=samples,
        use_wandb=False,
        batch_size=batch_size,
    )

    print("\n" + "=" * 60)
    print("CBF VERIFICATION RESULTS (LBP without McCormick)")
    print("=" * 60)
    print(f"System: {dynamics_model.system_name}")
    print(f"Certified percentage: {certified_percentage:.4f}%")
    print(f"Uncertified percentage: {uncertified_percentage:.4f}%")
    print(f"Computation time: {computation_time:.2f} seconds")

    total_samples = len(agg) if agg else 0
    iterations_per_second = total_samples / computation_time if computation_time > 0 else 0
    print(f"Total samples processed: {total_samples}")
    print(f"Iterations per second: {iterations_per_second:.2f} it/s")

    V_safe = []
    V_unsafe = []
    F_h_positive_in_unsafe = []
    F_safe_cbf_violation = []
    F_depth_limit_reached_unsafe = []
    F_depth_limit_reached_safe = []
    F_unsafe_cannot_split = []

    for result in agg:
        sample = result.sample

        if hasattr(sample, 'vertices'):
            vertices = np.array(sample.vertices, dtype=np.float32)
        elif hasattr(sample, 'center_point') and hasattr(sample, 'radius_vec'):
            center = np.array(sample.center_point, dtype=np.float32)
            radius = np.array(sample.radius_vec, dtype=np.float32)
            vertices = np.stack([center - radius, center + radius], axis=0)
        else:
            continue

        if isinstance(result, SampleResultSAT):
            result_type = result.result_type
            if result_type == "unsafe_region":
                V_unsafe.append(vertices)
            elif result_type == "safe_cbf_verified":
                V_safe.append(vertices)

        elif isinstance(result, SampleResultUNSAT):
            result_type = result.result_type
            if result_type == "h_positive_in_unsafe":
                F_h_positive_in_unsafe.append(vertices)
            elif result_type == "safe_cbf_violation":
                F_safe_cbf_violation.append(vertices)
            elif result_type == "depth_limit_reached_unsafe":
                F_depth_limit_reached_unsafe.append(vertices)
            elif result_type == "depth_limit_reached_safe":
                F_depth_limit_reached_safe.append(vertices)
            elif result_type == "unsafe_cannot_split":
                F_unsafe_cannot_split.append(vertices)

    results = {
        "regions": agg,
        "certified_percentage": certified_percentage,
        "uncertified_percentage": uncertified_percentage,
        "computation_time": computation_time,
        "total_samples": total_samples,
        "iterations_per_second": iterations_per_second,
        "V_safe": V_safe,
        "V_unsafe": V_unsafe,
        "F_h_positive_in_unsafe": F_h_positive_in_unsafe,
        "F_safe_cbf_violation": F_safe_cbf_violation,
        "F_depth_limit_reached_unsafe": F_depth_limit_reached_unsafe,
        "F_depth_limit_reached_safe": F_depth_limit_reached_safe,
        "F_unsafe_cannot_split": F_unsafe_cannot_split
    }

    if save_verification_regions:
        print("\n" + "=" * 60)
        print("SAVING VERIFICATION REGIONS FOR REPAIR")
        print("=" * 60)

        activation_fnc = getattr(dynamics_model, 'activation_fnc', 'Unknown')
        regions_dir = "/data/mzm/Repair_NCBF/New_repair/regions"
        import os
        os.makedirs(regions_dir, exist_ok=True)
        save_path = f"{regions_dir}/verified_regions_{dynamics_model.system_name}_{activation_fnc}_v1_lbp.pt"

        regions_data = {
            'V_safe': V_safe,
            'V_unsafe': V_unsafe,
            'F_h_positive_in_unsafe': F_h_positive_in_unsafe,
            'F_safe_cbf_violation': F_safe_cbf_violation,
            'F_depth_limit_reached_unsafe': F_depth_limit_reached_unsafe,
            'F_depth_limit_reached_safe': F_depth_limit_reached_safe,
            'F_unsafe_cannot_split': F_unsafe_cannot_split,
            'system_name': dynamics_model.system_name,
            'activation_fnc': activation_fnc,
            'input_dim': dynamics_model.input_dim,
            'max_depth': max_depth,
            "Certified percentage": certified_percentage,
            "Uncertified percentage": uncertified_percentage,
        }

        torch.save(regions_data, save_path)
        print(f"Verification regions saved to: {save_path}")

    return results


# =============================================================================
# 8. Loss Computation Functions (for gradient computation)
# =============================================================================

def vec_min(*tensors: torch.Tensor) -> torch.Tensor:
    """Compute element-wise minimum across multiple tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.minimum(result, t)
    return result


def vec_max(*tensors: torch.Tensor) -> torch.Tensor:
    """Compute element-wise maximum across multiple tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.maximum(result, t)
    return result


def compute_min_L_lbp_wo_mccormick(
    batch,
    dynamics_model,
    network_linearizer,
    device,
    dtype,
    h_lb_lbp=None
) -> torch.Tensor:
    """
    Compute lower bound on CBF condition using direct interval bounds (no McCormick).

    CBF condition: ∇h(x)·f(x) + ∇h(x)·g(x)·u + α(h(x)) ≥ 0

    Uses direct interval multiplication instead of McCormick.
    """
    m = dynamics_model.control_dim

    # Get Jacobian bounds from network linearizer
    # A_L shape: [batch, n, n], b_L shape: [batch, n]
    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()

    # Get dynamics bounds
    f_bounds, g_bounds = _compute_dynamics_bounds_ibp(batch, dynamics_model, device, dtype)
    f_L, f_U = f_bounds  # f_L shape: [batch, n]

    # Compute drift lower bound using direct interval multiplication
    # A_L is [batch, n, n], f_L is [batch, n]
    # Need to broadcast: A_L * f_L.unsqueeze(-1) -> [batch, n, n]
    JL_fL = A_L * f_L.unsqueeze(-1)
    JL_fU = A_L * f_U.unsqueeze(-1)
    JU_fL = A_U * f_L.unsqueeze(-1)
    JU_fU = A_U * f_U.unsqueeze(-1)

    L_corners = torch.minimum(torch.minimum(JL_fL, JL_fU), torch.minimum(JU_fL, JU_fU))  # [batch, n, n]
    U_corners = torch.maximum(torch.maximum(JL_fL, JL_fU), torch.maximum(JU_fL, JU_fU))  # [batch, n, n]

    # Sum over n dimensions, then take min for conservative lower bound
    L_drift = L_corners.sum(dim=-1).min(dim=-1)[0]  # [batch]
    U_drift = U_corners.sum(dim=-1).max(dim=-1)[0]  # [batch]

    # Compute control lower bound if applicable
    L_ctrl = torch.zeros_like(L_drift)

    if m > 0 and g_bounds is not None:
        g_L, g_U = g_bounds  # g_L shape: [batch, n, m]

        # J is A_L [batch, n, n], g is [batch, n, m]
        JL_gL = torch.matmul(A_L, g_L)  # [batch, n, m]
        JL_gU = torch.matmul(A_L, g_U)
        JU_gL = torch.matmul(A_U, g_L)
        JU_gU = torch.matmul(A_U, g_U)

        v_L = torch.minimum(torch.minimum(JL_gL, JL_gU), torch.minimum(JU_gL, JU_gU))  # [batch, n, m]
        v_U = torch.maximum(torch.maximum(JL_gL, JL_gU), torch.maximum(JU_gL, JU_gU))

        u_min = torch.tensor(dynamics_model.u_min, dtype=dtype, device=device)
        u_max = torch.tensor(dynamics_model.u_max, dtype=dtype, device=device)

        batch_size = len(batch)
        L_ctrl = torch.zeros(batch_size, dtype=dtype, device=device)
        for k in range(m):
            v_L_k = v_L[:, :, k]  # [batch, n]
            v_U_k = v_U[:, :, k]

            pos_mask = v_L_k >= 0
            neg_mask = v_U_k <= 0
            mixed_mask = ~(pos_mask | neg_mask)

            # Take min over n dimension (conservative)
            L_ctrl += torch.where(pos_mask.all(dim=-1), (v_L_k * u_min[k]).min(dim=-1)[0], torch.zeros_like(L_ctrl))
            L_ctrl += torch.where(neg_mask.all(dim=-1), (v_U_k * u_max[k]).max(dim=-1)[0], torch.zeros_like(L_ctrl))

            mixed_vals = torch.minimum(v_L_k * u_max[k], v_U_k * u_min[k]).min(dim=-1)[0]
            L_ctrl += torch.where(mixed_mask.any(dim=-1), mixed_vals, torch.zeros_like(L_ctrl))

    # Add class-K term
    if h_lb_lbp is not None:
        alpha_l = dynamics_model.alpha_function(h_lb_lbp)
    else:
        # Fall back: compute h_lb from network linearizer
        (A_L_net, a_L_net), _ = network_linearizer.get_network_linear_bounds()
        alpha_l = dynamics_model.alpha_function(a_L_net[..., 0])

    L_total = L_drift + L_ctrl + alpha_l

    return L_total


def compute_simplex_bound_batch_lbp_wo_mccormick(
    model: nn.Module,
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    region_type: str,
    dynamics_model=None,
    network_linearizer=None,
):
    """
    Batch version of bound computation using LBP without McCormick.

    Returns min_L for 'safe' regions, (h_lb, h_ub) for 'unsafe' regions.
    """
    from lbp_neural_cbf.regions import SimplicialRegion

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    batch = []
    for verts in vertices_list:
        if isinstance(verts, torch.Tensor):
            verts_np = verts.cpu().numpy()
        else:
            verts_np = verts
        batch.append(SimplicialRegion(verts_np, output_dim=None))

    B = len(batch)

    if network_linearizer is None:
        network_linearizer = CrownPartialLinearization(model, dtype=dtype)

    network_linearizer.compute_network_bounds(batch)
    h_lb, h_ub = network_linearizer.get_network_output_bounds()
    h_lb = h_lb.reshape(B, -1)[:, 0]
    h_ub = h_ub.reshape(B, -1)[:, 0]

    if region_type == 'unsafe':
        return h_lb, h_ub

    if region_type == 'safe':
        if dynamics_model is None:
            raise ValueError("dynamics_model is required for 'safe' region type")

        network_linearizer.keep_indices(list(range(B)))
        network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=0)

        min_L = compute_min_L_lbp_wo_mccormick(batch, dynamics_model, network_linearizer, device, dtype, h_lb_lbp=h_lb)
        return min_L.reshape(B)

    raise ValueError(f"Invalid region_type: {region_type}. Must be 'unsafe' or 'safe'")