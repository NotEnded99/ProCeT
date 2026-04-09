"""
Isolate the exact MulBackward0 NaN location.
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

def analyze_network_bounds(activation, idx=0, region_type='safe'):
    """Analyze the network bounds and intermediate values."""
    print(f"\n{'='*70}")
    print(f"ANALYZE: {activation}, region {idx} ({region_type})")
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
    V_unsafe = regions_data['V_unsafe']

    dtype = torch.float32

    if region_type == 'safe':
        simplex_vertices = V_safe[idx]
        if isinstance(simplex_vertices, torch.Tensor):
            simplex_vertices = simplex_vertices.cpu().numpy()
    else:
        simplex_vertices = V_unsafe[idx]
        if isinstance(simplex_vertices, torch.Tensor):
            simplex_vertices = simplex_vertices.cpu().numpy()

    batch = [SimplicialRegion(simplex_vertices, output_dim=None)]

    print(f"Simplex vertices:\n{simplex_vertices}")

    # Create CrownPartialLinearization
    network_linearizer = CrownPartialLinearization(model, dtype=dtype)
    network_linearizer.compute_network_bounds(batch)

    # Check pre-activation bounds for each layer
    print(f"\n--- Pre-activation bounds (y_lb, y_ub) ---")
    for i in range(len(network_linearizer.fc_layers)):
        key = f"layer_{i}_pre_act_bounds"
        if key in network_linearizer.forward_bounds:
            lb = network_linearizer.forward_bounds[key]['lb']
            ub = network_linearizer.forward_bounds[key]['ub']
            print(f"  layer_{i}: y_lb=[{lb.min().item():.4f}, {lb.max().item():.4f}], "
                  f"y_ub=[{ub.min().item():.4f}, {ub.max().item():.4f}]")
            # Check for extreme values
            if (lb.abs() > 10).any() or (ub.abs() > 10).any():
                print(f"    WARNING: Large pre-activation values detected!")

    print(f"\n--- Post-activation bounds ---")
    for i in range(len(network_linearizer.fc_layers)):
        key = f"layer_{i}_post_act_bounds"
        if key in network_linearizer.forward_bounds:
            lb = network_linearizer.forward_bounds[key]['lb']
            ub = network_linearizer.forward_bounds[key]['ub']
            print(f"  layer_{i}: z_lb=[{lb.min().item():.4f}, {lb.max().item():.4f}], "
                  f"z_ub=[{ub.min().item():.4f}, {ub.max().item():.4f}]")

    # Compute partial derivative bounds
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
    print(f"\n--- Jacobian Bounds ---")
    print(f"A_L: shape={A_L.shape}, range=[{A_L.min().item():.6f}, {A_L.max().item():.6f}]")
    print(f"b_L: shape={b_L.shape}, range=[{b_L.min().item():.6f}, {b_L.max().item():.6f}]")
    print(f"A_U: shape={A_U.shape}, range=[{A_U.min().item():.6f}, {A_U.max().item():.6f}]")
    print(f"b_U: shape={b_U.shape}, range=[{b_U.min().item():.6f}, {b_U.max().item():.6f}]")

    # Check for extreme values
    for name, tensor in [('A_L', A_L), ('b_L', b_L), ('A_U', A_U), ('b_U', b_U)]:
        if torch.isnan(tensor).any():
            print(f"  NaN in {name}!")
        if torch.isinf(tensor).any():
            print(f"  Inf in {name}!")
        if (tensor.abs() > 1e6).any():
            print(f"  WARNING: Large values in {name} (max={tensor.abs().max().item():.2e})")

    # Compute dynamics bounds
    f_affine_bounds, g_affine_bounds = _compute_dynamics_bounds_taylor(
        batch, dynamics_model, device=device, dtype=dtype
    )
    f_affine_L, f_affine_U = f_affine_bounds

    print(f"\n--- Dynamics Bounds ---")
    print(f"f A_L: shape={f_affine_L[0].shape}, range=[{f_affine_L[0].min().item():.6f}, {f_affine_L[0].max().item():.6f}]")
    print(f"f b_L: shape={f_affine_L[1].shape}, range=[{f_affine_L[1].min().item():.6f}, {f_affine_L[1].max().item():.6f}]")
    print(f"f A_U: shape={f_affine_U[0].shape}, range=[{f_affine_U[0].min().item():.6f}, {f_affine_U[0].max().item():.6f}]")
    print(f"f b_U: shape={f_affine_U[1].shape}, range=[{f_affine_U[1].min().item():.6f}, {f_affine_U[1].max().item():.6f}]")

    # McCormick product
    A_L_s = A_L.squeeze(1)
    b_L_s = b_L.squeeze(1)
    A_U_s = A_U.squeeze(1)
    b_U_s = b_U.squeeze(1)
    J_affine_L = (A_L_s, b_L_s)
    J_affine_U = (A_U_s, b_U_s)

    print(f"\n--- McCormick Product Intermediates ---")
    eta = 0.5

    # Get y bounds for McCormick
    y1_min, y1_max = _batched_get_affine_function_bounds(J_affine_L, batch, device=device, dtype=dtype)
    y2_min, y2_max = _batched_get_affine_function_bounds(f_affine_L, batch, device=device, dtype=dtype)

    print(f"y1_min: [{y1_min.min().item():.4f}, {y1_min.max().item():.4f}]")
    print(f"y1_max: [{y1_max.min().item():.4f}, {y1_max.max().item():.4f}]")
    print(f"y2_min: [{y2_min.min().item():.4f}, {y2_min.max().item():.4f}]")
    print(f"y2_max: [{y2_max.min().item():.4f}, {y2_max.max().item():.4f}]")

    C1 = eta * y1_min + (1 - eta) * y1_max
    C2 = eta * y2_min + (1 - eta) * y2_max
    const_part = -(eta * y1_min * y2_min + (1 - eta) * y1_max * y2_max)

    print(f"C1: [{C1.min().item():.4f}, {C1.max().item():.4f}]")
    print(f"C2: [{C2.min().item():.4f}, {C2.max().item():.4f}]")
    print(f"const_part: [{const_part.min().item():.4f}, {const_part.max().item():.4f}]")

    if torch.isnan(C1).any() or torch.isnan(C2).any() or torch.isnan(const_part).any():
        print("  NaN detected in C1, C2, or const_part!")
        return

    C1_pos, C1_neg = C1.clamp(min=0), C1.clamp(max=0)
    C2_pos, C2_neg = C2.clamp(min=0), C2.clamp(max=0)

    print(f"C1_pos: [{C1_pos.min().item():.4f}, {C1_pos.max().item():.4f}]")
    print(f"C1_neg: [{C1_neg.min().item():.4f}, {C1_neg.max().item():.4f}]")

    # Compute McCormick
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_affine_L, J_affine_U, f_affine_L, f_affine_U, batch, eta=eta, device=device, dtype=dtype,
    )

    print(f"M_D: shape={M_D.shape}, range=[{M_D.min().item():.6f}, {M_D.max().item():.6f}]")
    print(f"c_D: shape={c_D.shape}, range=[{c_D.min().item():.6f}, {c_D.max().item():.6f}]")

    if torch.isnan(M_D).any() or torch.isnan(c_D).any():
        print("  NaN in M_D or c_D!")
        return
    if torch.isinf(M_D).any() or torch.isinf(c_D).any():
        print("  Inf in M_D or c_D!")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    # Test with different simplex indices to find which ones cause NaN
    for idx in range(3):
        analyze_network_bounds('Tanh', idx=idx, region_type='safe')
        analyze_network_bounds('Relu', idx=idx, region_type='safe')