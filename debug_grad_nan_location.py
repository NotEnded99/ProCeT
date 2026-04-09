"""
Pinpoint exact NaN location in Tanh gradient backward pass.
"""

import sys
import os
import torch
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

from lbp_neural_cbf.cbf.fossil_dynamics import Barrier3System
from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.translators import TorchTranslator
from lbp_neural_cbf.regions import SimplicialRegion
from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization
from lbp_neural_cbf.cbf.verify_cbf import (
    _compute_dynamics_bounds_taylor,
    _batched_compute_mccormick_product_lower_bound,
    _batched_get_affine_function_bounds,
)

def isolate_nan_in_forward(activation, idx=0, region_type='safe'):
    """
    Test each step of the forward pass with NaN detection to find exact step that causes gradient NaN.
    """
    print(f"\n{'='*70}")
    print(f"ISOLATE NaN: {activation}, region {idx} ({region_type})")
    print(f"{'='*70}")

    device = torch.device('cuda')

    dynamics_class = Barrier3System
    dynamics_model = dynamics_class(alpha=1.0)
    dynamics_model.activation_fnc = activation

    model_dir = f"data/New_models_Hard_{activation}"
    model_path = f"{model_dir}/{dynamics_model.system_name}_cbf.pth"

    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device,
        activation_fnc=activation
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}.pt"
    regions_data = torch.load(regions_path, map_location=device, weights_only=False)

    V_safe = regions_data['V_safe']
    dtype = torch.float32

    simplex_vertices = V_safe[idx]
    if isinstance(simplex_vertices, torch.Tensor):
        simplex_vertices = simplex_vertices.cpu().numpy()

    batch = [SimplicialRegion(simplex_vertices, output_dim=None)]
    params = list(model.parameters())

    # Build computation graph step by step
    network_lin = CrownPartialLinearization(model, dtype=dtype)
    network_lin.compute_network_bounds(batch)

    print("\n--- Step 1: compute_partial_derivative_bounds ---")
    network_lin.compute_partial_derivative_bounds(input_idx=None, output_idx=None)
    A_L, b_L, A_U, b_U = network_lin.get_partial_derivative_bounds()
    A_L_s = A_L.squeeze(1)
    b_L_s = b_L.squeeze(1)
    A_U_s = A_U.squeeze(1)
    b_U_s = b_U.squeeze(1)
    print(f"  Jacobian: A_L range=[{A_L_s.min().item():.4f}, {A_L_s.max().item():.4f}], NaN={torch.isnan(A_L_s).any().item()}")

    print("\n--- Step 2: dynamics bounds ---")
    f_bounds, _ = _compute_dynamics_bounds_taylor(batch, dynamics_model, device=device, dtype=dtype)
    f_L, f_U = f_bounds
    print(f"  f bounds: A_L[0] range=[{f_L[0].min().item():.4f}, {f_L[0].max().item():.4f}], NaN={torch.isnan(f_L[0]).any().item()}")

    print("\n--- Step 3: J_affine bounds (get_affine) ---")
    J_L = (A_L_s, b_L_s)
    J_U = (A_U_s, b_U_s)
    y1_min, y1_max = _batched_get_affine_function_bounds(J_L, batch, device=device, dtype=dtype)
    y2_min, y2_max = _batched_get_affine_function_bounds(f_L, batch, device=device, dtype=dtype)
    print(f"  y1 bounds: min=[{y1_min.min().item():.4f}, {y1_min.max().item():.4f}], NaN={torch.isnan(y1_min).any().item()}")
    print(f"  y2 bounds: min=[{y2_min.min().item():.4f}, {y2_min.max().item():.4f}], NaN={torch.isnan(y2_min).any().item()}")

    print("\n--- Step 4: McCormick C values ---")
    eta = 0.5
    C1 = eta * y1_min + (1 - eta) * y1_max
    C2 = eta * y2_min + (1 - eta) * y2_max
    const_part = -(eta * y1_min * y2_min + (1 - eta) * y1_max * y2_max)
    print(f"  C1: [{C1.min().item():.4f}, {C1.max().item():.4f}], NaN={torch.isnan(C1).any().item()}")
    print(f"  C2: [{C2.min().item():.4f}, {C2.max().item():.4f}], NaN={torch.isnan(C2).any().item()}")
    print(f"  const_part: [{const_part.min().item():.4f}, {const_part.max().item():.4f}], NaN={torch.isnan(const_part).any().item()}")

    print("\n--- Step 5: Full McCormick product ---")
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_L, J_U, f_L, f_U, batch, eta=eta, device=device, dtype=dtype,
    )
    M_D_sum = M_D.sum(dim=-2)
    c_D_sum = c_D.sum(dim=-1)
    print(f"  M_D_sum: [{M_D_sum.min().item():.4f}, {M_D_sum.max().item():.4f}], NaN={torch.isnan(M_D_sum).any().item()}")
    print(f"  c_D_sum: [{c_D_sum.min().item():.4f}, {c_D_sum.max().item():.4f}], NaN={torch.isnan(c_D_sum).any().item()}")

    print("\n--- Step 6: Alpha function ---")
    (A_L_net, a_L_net), _ = network_lin.get_network_linear_bounds()
    A_L_net = A_L_net.squeeze(1)
    a_L_net = a_L_net.squeeze(1)
    alpha_A_L = dynamics_model.alpha_function(A_L_net)
    alpha_a_L = dynamics_model.alpha_function(a_L_net)
    print(f"  alpha_A_L: [{alpha_A_L.min().item():.4f}, {alpha_A_L.max().item():.4f}], NaN={torch.isnan(alpha_A_L).any().item()}")
    print(f"  alpha_a_L: [{alpha_a_L.min().item():.4f}, {alpha_a_L.max().item():.4f}], NaN={torch.isnan(alpha_a_L).any().item()}")

    M_total = M_D_sum + alpha_A_L
    c_total = c_D_sum + alpha_a_L
    print(f"  M_total: [{M_total.min().item():.4f}, {M_total.max().item():.4f}], NaN={torch.isnan(M_total).any().item()}")
    print(f"  c_total: [{c_total.min().item():.4f}, {c_total.max().item():.4f}], NaN={torch.isnan(c_total).any().item()}")

    print("\n--- Step 7: min_L ---")
    min_L, _ = _batched_get_affine_function_bounds(
        (M_total.unsqueeze(1), c_total.unsqueeze(1)),
        batch, device=device, dtype=dtype,
    )
    min_L = min_L.squeeze(-1)
    print(f"  min_L: {min_L[0].item():.6f}, NaN={torch.isnan(min_L).any().item()}, requires_grad={min_L.requires_grad}")

    print("\n--- Step 8: Test gradient with nan_to_num ---")
    model.zero_grad()
    # Use torch.no_grad to check if the forward result is valid
    with torch.no_grad():
        min_L_check = min_L[0].item()
    print(f"  Forward value (no grad): {min_L_check:.6f}")

    # Now test gradient with nan_to_num on inputs
    print("\n--- Testing if nan_to_num on min_L helps ---")
    min_L_nan = torch.nan_to_num(min_L[0], nan=0.0)
    try:
        grad_nan = torch.autograd.grad(min_L_nan, params, retain_graph=False)
        grad_vec_nan = torch.cat([g.flatten() for g in grad_nan])
        print(f"  nan_to_num gradient: has NaN={torch.isnan(grad_vec_nan).any().item()}")
    except Exception as e:
        print(f"  nan_to_num gradient: FAILED - {e}")

    # Test with clamp on min_L to prevent extreme values
    print("\n--- Testing if clamp on min_L helps ---")
    min_L_clamped = torch.clamp(min_L[0], min=-1e6, max=1e6)
    model.zero_grad()
    try:
        grad_clamp = torch.autograd.grad(min_L_clamped, params, retain_graph=False)
        grad_vec_clamp = torch.cat([g.flatten() for g in grad_clamp])
        print(f"  clamp gradient: has NaN={torch.isnan(grad_vec_clamp).any().item()}, has Inf={torch.isinf(grad_vec_clamp).any().item()}")
        if torch.isnan(grad_vec_clamp).any():
            nan_count = torch.isnan(grad_vec_clamp).sum().item()
            print(f"    NaN count: {nan_count}")
    except Exception as e:
        print(f"  clamp gradient: FAILED - {e}")

    print("\n" + "="*70)

if __name__ == "__main__":
    isolate_nan_in_forward('Tanh', idx=0, region_type='safe')
    isolate_nan_in_forward('Relu', idx=0, region_type='safe')