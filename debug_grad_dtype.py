"""
Test if float64 precision fixes the NaN gradient issue.
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

def test_float64(activation, idx=0, region_type='safe'):
    """Test if float64 resolves NaN gradients."""
    print(f"\n{'='*70}")
    print(f"TEST FLOAT64: {activation}, region {idx} ({region_type})")
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

    # Test with float32 (original)
    print(f"\n--- Testing float32 ---")
    dtype32 = torch.float32
    simplex_vertices = V_safe[idx]
    if isinstance(simplex_vertices, torch.Tensor):
        simplex_vertices = simplex_vertices.cpu().numpy()

    batch32 = [SimplicialRegion(simplex_vertices, output_dim=None)]

    network_lin32 = CrownPartialLinearization(model, dtype=dtype32)
    network_lin32.compute_network_bounds(batch32)
    network_lin32.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    A_L32, b_L32, A_U32, b_U32 = network_lin32.get_partial_derivative_bounds()
    A_L32 = A_L32.squeeze(1)
    b_L32 = b_L32.squeeze(1)
    A_U32 = A_U32.squeeze(1)
    b_U32 = b_U32.squeeze(1)

    f_bounds32, _ = _compute_dynamics_bounds_taylor(
        batch32, dynamics_model, device=device, dtype=dtype32
    )
    f_L32, f_U32 = f_bounds32

    J_L32 = (A_L32, b_L32)
    J_U32 = (A_U32, b_U32)

    eta = 0.5
    M_D32, c_D32 = _batched_compute_mccormick_product_lower_bound(
        J_L32, J_U32, f_L32, f_U32, batch32, eta=eta, device=device, dtype=dtype32,
    )
    M_D32 = M_D32.sum(dim=-2)
    c_D32 = c_D32.sum(dim=-1)

    (A_L_net32, a_L_net32), _ = network_lin32.get_network_linear_bounds()
    A_L_net32 = A_L_net32.squeeze(1)
    a_L_net32 = a_L_net32.squeeze(1)
    alpha_A_L32 = dynamics_model.alpha_function(A_L_net32)
    alpha_a_L32 = dynamics_model.alpha_function(a_L_net32)

    M_total32 = M_D32 + alpha_A_L32
    c_total32 = c_D32 + alpha_a_L32

    min_L32, _ = _batched_get_affine_function_bounds(
        (M_total32.unsqueeze(1), c_total32.unsqueeze(1)),
        batch32, device=device, dtype=dtype32,
    )
    min_L32 = min_L32.squeeze(-1)

    print(f"min_L (float32): {min_L32[0].item():.6f}")

    params = list(model.parameters())
    model.zero_grad()
    try:
        grad32 = torch.autograd.grad(min_L32[0], params, retain_graph=False)
        grad_vec32 = torch.cat([g.flatten() for g in grad32])
        print(f"Gradient (float32): has NaN={torch.isnan(grad_vec32).any().item()}, has Inf={torch.isinf(grad_vec32).any().item()}")
    except Exception as e:
        print(f"Gradient (float32): FAILED - {e}")

    # Test with float64
    print(f"\n--- Testing float64 ---")
    dtype64 = torch.float64

    batch64 = [SimplicialRegion(simplex_vertices, output_dim=None)]

    network_lin64 = CrownPartialLinearization(model, dtype=dtype64)
    network_lin64.compute_network_bounds(batch64)
    network_lin64.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    A_L64, b_L64, A_U64, b_U64 = network_lin64.get_partial_derivative_bounds()
    A_L64 = A_L64.squeeze(1)
    b_L64 = b_L64.squeeze(1)
    A_U64 = A_U64.squeeze(1)
    b_U64 = b_U64.squeeze(1)

    f_bounds64, _ = _compute_dynamics_bounds_taylor(
        batch64, dynamics_model, device=device, dtype=dtype64
    )
    f_L64, f_U64 = f_bounds64

    J_L64 = (A_L64, b_L64)
    J_U64 = (A_U64, b_U64)

    M_D64, c_D64 = _batched_compute_mccormick_product_lower_bound(
        J_L64, J_U64, f_L64, f_U64, batch64, eta=eta, device=device, dtype=dtype64,
    )
    M_D64 = M_D64.sum(dim=-2)
    c_D64 = c_D64.sum(dim=-1)

    (A_L_net64, a_L_net64), _ = network_lin64.get_network_linear_bounds()
    A_L_net64 = A_L_net64.squeeze(1)
    a_L_net64 = a_L_net64.squeeze(1)
    alpha_A_L64 = dynamics_model.alpha_function(A_L_net64)
    alpha_a_L64 = dynamics_model.alpha_function(a_L_net64)

    M_total64 = M_D64 + alpha_A_L64
    c_total64 = c_D64 + alpha_a_L64

    min_L64, _ = _batched_get_affine_function_bounds(
        (M_total64.unsqueeze(1), c_total64.unsqueeze(1)),
        batch64, device=device, dtype=dtype64,
    )
    min_L64 = min_L64.squeeze(-1)

    print(f"min_L (float64): {min_L64[0].item():.6f}")

    model.zero_grad()
    try:
        grad64 = torch.autograd.grad(min_L64[0], params, retain_graph=False)
        grad_vec64 = torch.cat([g.flatten() for g in grad64])
        print(f"Gradient (float64): has NaN={torch.isnan(grad_vec64).any().item()}, has Inf={torch.isinf(grad_vec64).any().item()}")
        if torch.isnan(grad_vec64).any():
            nan_count = torch.isnan(grad_vec64).sum().item()
            print(f"  NaN count: {nan_count}")
        else:
            print(f"  All gradients valid!")
            # Compare norms
            print(f"  |grad32|={grad_vec32.norm().item():.6f}" if not torch.isnan(grad_vec32).any() else f"  |grad32|=NaN")
            print(f"  |grad64|={grad_vec64.norm().item():.6f}")
    except Exception as e:
        print(f"Gradient (float64): FAILED - {e}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    test_float64('Tanh', idx=0, region_type='safe')
    test_float64('Relu', idx=0, region_type='safe')