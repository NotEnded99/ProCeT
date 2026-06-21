"""LBP bound machinery + weighted McCormick repair loss.

This module is the foundation of the LBP-based repair pipeline. It exposes:

``compute_min_L_with_mccormick``
    Lower bound of the CBF condition ``L(x) = ∇h·f + α(h) + sup_u ∇h·g·u``
    on a batch of simplicial regions, via full LBP with McCormick envelopes.

``compute_h_max_via_network_bounds``
    Upper bound of the barrier function ``h(x)`` via standard CROWN network
    output bounds. This matches the V_unsafe verification condition
    ``h_max < 0``.

``compute_repair_loss_and_grad_lbp_weighted``
    Combined safe + unsafe repair loss (softplus-margin) with per-sample
    category weighting, plus gradient w.r.t. flattened model parameters.

``simple_gradient_update``
    Plain GD step used by the CeT method.

The two bound functions are kept here (rather than in a separate module)
because the loss depends on them directly and every other consumer
(``selection``, ``jacobian``, ``audit``) is happy to import from
``lbp_loss`` rather than carry a duplicate definition.
"""

import numpy as np
import torch
import torch.nn as nn

from lbp_neural_cbf.cbf.verify_cbf import (
    _batched_compute_mccormick_product_lower_bound,
    _batched_get_affine_function_bounds,
    _compute_dynamics_bounds_taylor,
)
from lbp_neural_cbf.regions import SimplicialRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_simplicial_regions(batch):
    """Convert a list of (n+1, n) arrays / tensors / regions into SimplicialRegion."""
    regions = []
    for verts in batch:
        if isinstance(verts, np.ndarray):
            regions.append(SimplicialRegion(verts.astype(np.float32), output_dim=None))
        elif hasattr(verts, "detach"):
            regions.append(SimplicialRegion(verts.detach().cpu().numpy().astype(np.float32), output_dim=None))
        else:
            regions.append(verts)
    return regions


# ---------------------------------------------------------------------------
# LBP bounds (McCormick)
# ---------------------------------------------------------------------------

def compute_min_L_with_mccormick(batch, dynamics_model, network_linearizer, device, dtype, h_lb_lbp=None):
    """Lower bound ``min_L`` of the CBF condition on each region in ``batch``.

    CBF condition: ``∇h(x)·f(x) + ∇h(x)·g(x)·u + α(h(x)) ≥ 0``.
    Uses McCormick envelopes for interval products.
    """
    n = dynamics_model.input_dim
    m = dynamics_model.control_dim

    regions = _to_simplicial_regions(batch)

    # 0. Network bounds + ∂h/∂x bounds
    network_linearizer.compute_network_bounds(regions)
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=0)

    # 1. Dynamics affine bounds via Taylor linearisation
    f_affine_bounds, g_affine_bounds = _compute_dynamics_bounds_taylor(regions, dynamics_model, device, dtype)
    f_affine_L, f_affine_U = f_affine_bounds

    # 2. Jacobian affine bounds from LBP
    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
    J_affine_L, J_affine_U = (A_L, b_L), (A_U, b_U)

    # 3. Lower bound of drift term J(x)·f(x) via McCormick
    eta_drift = 0.5
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_affine_L, J_affine_U, f_affine_L, f_affine_U, regions,
        eta=eta_drift, device=device, dtype=dtype,
    )
    M_D, c_D = M_D.sum(dim=-2), c_D.sum(dim=-1)

    # 4. Lower bound of class-K term α(h)
    (A_L_net, a_L_net), _ = network_linearizer.get_network_linear_bounds()
    alpha_A_L = dynamics_model.alpha_function(A_L_net[..., 0, :])
    alpha_a_L = dynamics_model.alpha_function(a_L_net[..., 0])

    M_total, c_total = M_D + alpha_A_L, c_D + alpha_a_L

    # 5. Lower bound of control term sup_u J(x)·g(x)·u
    if m > 0:
        g_affine_L = g_affine_bounds[0][0], g_affine_bounds[0][1]
        g_affine_U = g_affine_bounds[1][0], g_affine_bounds[1][1]

        eta_control_L = 0.5
        M_v_L, c_v_L = _batched_compute_mccormick_product_lower_bound(
            J_affine_L, J_affine_U, g_affine_L, g_affine_U, regions,
            eta=eta_control_L, device=device, dtype=dtype,
        )
        M_v_L, c_v_L = M_v_L.sum(dim=-2), c_v_L.sum(dim=-1)

        v_affine_L = (M_v_L, c_v_L)
        v_L_min, v_L_max = _batched_get_affine_function_bounds(v_affine_L, regions, device=device, dtype=dtype)

        u_min = torch.tensor(dynamics_model.u_min, device=device, dtype=dtype)
        u_max = torch.tensor(dynamics_model.u_max, device=device, dtype=dtype)

        M_v_L_u_min, c_v_L_u_min = M_v_L * u_min.unsqueeze(-1), c_v_L * u_min
        M_v_L_u_max, c_v_L_u_max = M_v_L * u_max.unsqueeze(-1), c_v_L * u_max

        for sample_idx, sample in enumerate(batch):
            M_C = torch.zeros(n, device=device, dtype=dtype)
            c_C = torch.tensor(0.0, device=device, dtype=dtype)
            if m > 0:
                v_Lsample_min = v_L_min[sample_idx]
                v_Lsample_max = v_L_max[sample_idx]

                pos_mask = v_Lsample_min >= 0
                if pos_mask.any():
                    M_C += (M_v_L_u_max[sample_idx, pos_mask]).sum(dim=0)
                    c_C += (c_v_L_u_max[sample_idx, pos_mask]).sum()

                neg_mask = v_Lsample_max <= 0
                if neg_mask.any():
                    M_C += (M_v_L_u_min[sample_idx, neg_mask]).sum(dim=0)
                    c_C += (c_v_L_u_min[sample_idx, neg_mask]).sum()

                mixed_mask = ~(pos_mask | neg_mask)
                if mixed_mask.any():
                    v_u_min_b, _ = _batched_get_affine_function_bounds(
                        (M_v_L_u_min[sample_idx, mixed_mask], c_v_L_u_min[sample_idx, mixed_mask]),
                        [regions[sample_idx]], device=device, dtype=dtype,
                    )
                    v_u_max_b, _ = _batched_get_affine_function_bounds(
                        (M_v_L_u_max[sample_idx, mixed_mask], c_v_L_u_max[sample_idx, mixed_mask]),
                        [regions[sample_idx]], device=device, dtype=dtype,
                    )
                    c_C += torch.maximum(v_u_min_b, v_u_max_b).sum()

            M_total[sample_idx] += M_C
            c_total[sample_idx] += c_C

    # 6. Minimise the affine lower bound over each region
    min_L, _ = _batched_get_affine_function_bounds(
        (M_total.unsqueeze(1), c_total.unsqueeze(1)), regions, device=device, dtype=dtype
    )
    min_L = min_L.squeeze(-1)
    return min_L


def compute_h_max_via_network_bounds(batch, dynamics_model, network_linearizer, device, dtype):
    """Upper bound of ``h(x)`` via standard CROWN network output bounds.

    Matches the V_unsafe verification condition: ``h_max < 0`` ⟹ SAT unsafe.
    The returned tensor is differentiable and supports ``.backward()`` for
    Jacobian computations.
    """
    regions = _to_simplicial_regions(batch)

    network_linearizer.compute_network_bounds(regions)
    _, ub = network_linearizer.get_network_output_bounds(sample_idx=None)
    # Guarantee 1-D (one h_max per region) so downstream argsort works.
    while ub.dim() > 1:
        ub = ub.squeeze(-1)
    return ub


# ---------------------------------------------------------------------------
# Plain GD update (CeT)
# ---------------------------------------------------------------------------

def simple_gradient_update(model, g_F, lr):
    """Apply ``θ ← θ − lr · g_F`` in-place; return ``‖lr·g_F‖``."""
    params = list(model.parameters())
    g_flat = g_F.clone()
    offset = 0
    for p in params:
        numel = p.numel()
        grad_slice = g_flat[offset:offset + numel].view(p.shape)
        with torch.no_grad():
            p.sub_(lr * grad_slice)
        offset += numel
    return (g_F * lr).norm().item()


# ---------------------------------------------------------------------------
# Weighted LBP repair loss
# ---------------------------------------------------------------------------

def _safe_lbp_weighted(model, dynamics_model, safe_simplices, lbp_linearizer,
                       cbf_margin, beta, grad_clip_norm, verbose, weights):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    if len(safe_simplices) == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    regions = _to_simplicial_regions(safe_simplices)

    BATCH_SIZE = 512
    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_weighted_loss_sum = 0.0
    total_weight_sum = 0.0

    for batch_start in range(0, len(regions), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(regions))
        batch_regions = regions[batch_start:batch_end]
        B = len(batch_regions)

        batch_weights = (
            torch.tensor(weights[batch_start:batch_end], device=device, dtype=dtype)
            if weights is not None else
            torch.ones(B, device=device, dtype=dtype)
        )

        min_L_batch = compute_min_L_with_mccormick(
            batch_regions, dynamics_model, lbp_linearizer, device, dtype
        ).reshape(-1)

        # softplus(cbf_margin − min_L): pushes min_L ≥ cbf_margin
        loss_batch = torch.nn.functional.softplus(cbf_margin - min_L_batch, beta=beta)

        if not torch.isfinite(loss_batch).all():
            if verbose:
                print(f"  [Warning] safe LBP McCormick batch [{batch_start}:{batch_start + B}] has NaN/Inf, skipping")
            del min_L_batch, loss_batch
            torch.cuda.empty_cache()
            continue

        weighted_loss = loss_batch * batch_weights
        total_weighted_loss_sum += weighted_loss.sum().item()
        total_weight_sum += batch_weights.sum().item()

        model.zero_grad()
        weighted_loss.sum().backward()

        grad_batch = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        if grad_batch.isnan().any() or grad_batch.isinf().any():
            raise ValueError(f"Safe LBP McCormick batch [{batch_start}:{batch_start + B}] produced NaN/Inf gradient")
        g_raw.add_(grad_batch)

        del min_L_batch, loss_batch, weighted_loss, grad_batch
        torch.cuda.empty_cache()

    if total_weight_sum == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    mean_loss = total_weighted_loss_sum / total_weight_sum
    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)
    if verbose:
        print(f"  [Safe LBP with McCormick] loss={mean_loss:.6f}, |g|={grad_norm:.4f}, n={len(regions)}")
    return mean_loss, g_raw


def _unsafe_lbp_weighted(model, dynamics_model, unsafe_simplices, lbp_linearizer,
                         margin, beta, grad_clip_norm, verbose, weights):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    if len(unsafe_simplices) == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    regions = _to_simplicial_regions(unsafe_simplices)

    BATCH_SIZE = 512
    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_weighted_loss_sum = 0.0
    total_weight_sum = 0.0

    for batch_start in range(0, len(regions), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(regions))
        batch_regions = regions[batch_start:batch_end]
        B = len(batch_regions)

        batch_weights = (
            torch.tensor(weights[batch_start:batch_end], device=device, dtype=dtype)
            if weights is not None else
            torch.ones(B, device=device, dtype=dtype)
        )

        h_ub_batch = compute_h_max_via_network_bounds(
            batch_regions, dynamics_model, lbp_linearizer, device, dtype
        )

        # softplus(h_ub + margin): pushes h_ub + margin ≤ 0
        loss_batch = torch.nn.functional.softplus(h_ub_batch + margin, beta=beta)

        if not torch.isfinite(loss_batch).all():
            if verbose:
                print(f"  [Warning] unsafe LBP McCormick batch [{batch_start}:{batch_start + B}] has NaN/Inf, skipping")
            del h_ub_batch, loss_batch
            torch.cuda.empty_cache()
            continue

        weighted_loss = loss_batch * batch_weights
        total_weighted_loss_sum += weighted_loss.sum().item()
        total_weight_sum += batch_weights.sum().item()

        model.zero_grad()
        weighted_loss.sum().backward()

        grad_batch = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        if grad_batch.isnan().any() or grad_batch.isinf().any():
            raise ValueError(f"Unsafe LBP McCormick batch [{batch_start}:{batch_start + B}] produced NaN/Inf gradient")
        g_raw.add_(grad_batch)

        del h_ub_batch, loss_batch, weighted_loss, grad_batch
        torch.cuda.empty_cache()

    if total_weight_sum == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    mean_loss = total_weighted_loss_sum / total_weight_sum
    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)
    if verbose:
        print(f"  [Unsafe LBP with McCormick] loss={mean_loss:.6f}, |g|={grad_norm:.4f}, n={len(regions)}")
    return mean_loss, g_raw


def compute_repair_loss_and_grad_lbp_weighted(
    model, dynamics_model, safe_simplices, unsafe_simplices, lbp_linearizer,
    margin=0.0, cbf_margin=0.0, beta=5.0, grad_clip_norm=10.0, verbose=False,
    safe_weights=None, unsafe_weights=None,
):
    """Combined weighted safe + unsafe repair loss (full LBP + McCormick).

    Per-sample ``safe_weights`` / ``unsafe_weights`` implement the definitive-
    vs-uncertain category weighting used by the ProCeT family: definitive
    violations (``F_h_positive_in_unsafe``, ``F_safe_cbf_violation``) get
    weight ``lambda_`` while uncertain regions get weight 1.

    Returns:
        ``(mean_loss, grad_wrt_flat_params)``
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_weighted_loss_sum = 0.0
    total_weight_sum = 0.0
    n_valid = 0

    if len(unsafe_simplices) > 0:
        loss_unsafe, grad_unsafe = _unsafe_lbp_weighted(
            model, dynamics_model, unsafe_simplices, lbp_linearizer,
            margin=margin, beta=beta, grad_clip_norm=grad_clip_norm,
            verbose=verbose, weights=unsafe_weights,
        )
        if torch.isfinite(grad_unsafe).all():
            g_raw.add_(grad_unsafe)
            w_sum = sum(unsafe_weights) if unsafe_weights is not None else len(unsafe_simplices)
            total_weighted_loss_sum += loss_unsafe * w_sum
            total_weight_sum += w_sum
            n_valid += 1

    if len(safe_simplices) > 0:
        loss_safe, grad_safe = _safe_lbp_weighted(
            model, dynamics_model, safe_simplices, lbp_linearizer,
            cbf_margin=cbf_margin, beta=beta, grad_clip_norm=grad_clip_norm,
            verbose=verbose, weights=safe_weights,
        )
        if torch.isfinite(grad_safe).all():
            g_raw.add_(grad_safe)
            w_sum = sum(safe_weights) if safe_weights is not None else len(safe_simplices)
            total_weighted_loss_sum += loss_safe * w_sum
            total_weight_sum += w_sum
            n_valid += 1

    if n_valid == 0 or total_weight_sum == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    total_loss = total_weighted_loss_sum / total_weight_sum
    grad_norm = g_raw.norm().item()
    if verbose:
        print(f"  [Repair Loss LBP+McCormick+CategoryWeight] total_loss={total_loss:.6f}, "
              f"|g|={grad_norm:.4f}, unsafe={len(unsafe_simplices)}, safe={len(safe_simplices)}, "
              f"total_weight={total_weight_sum:.4f}")
    return total_loss, g_raw
