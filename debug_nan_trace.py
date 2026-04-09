"""
Debug script to trace the exact NaN location in Tanh Jacobian computation.
"""

import sys
import os
import torch
import numpy as np

# Setup path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set seeds for reproducibility
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

def debug_tanh_jacobian(activation, idx=0, region_type='safe'):
    """Debug the Tanh Jacobian computation step by step."""
    print(f"\n{'='*70}")
    print(f"DEBUG Tanh JACOBIAN: {activation}, region {idx} ({region_type})")
    print(f"{'='*70}")

    device = torch.device('cuda')

    # Load dynamics
    dynamics_class = Barrier3System
    dynamics_model = dynamics_class(alpha=1.0)
    dynamics_model.activation_fnc = activation

    # Load model
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

    # Load regions
    regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}.pt"
    regions_data = torch.load(regions_path, map_location=device, weights_only=False)

    V_safe = regions_data['V_safe']
    V_unsafe = regions_data['V_unsafe']

    translator = TorchTranslator(device=device)
    dtype = torch.float32

    # Get one simplex
    if region_type == 'safe':
        simplex_vertices = V_safe[idx]
        if isinstance(simplex_vertices, torch.Tensor):
            simplex_vertices = simplex_vertices.cpu().numpy()
    else:
        simplex_vertices = V_unsafe[idx]
        if isinstance(simplex_vertices, torch.Tensor):
            simplex_vertices = simplex_vertices.cpu().numpy()

    print(f"Simplex vertices:\n{simplex_vertices}")

    batch = [SimplicialRegion(simplex_vertices, output_dim=None)]

    # Step 1: Create CrownPartialLinearization
    print(f"\n--- Step 1: CrownPartialLinearization ---")
    network_linearizer = CrownPartialLinearization(model, dtype=dtype)
    network_linearizer.compute_network_bounds(batch)
    print(f"Network bounds computed")

    # Check forward bounds
    for key in network_linearizer.forward_bounds:
        lb = network_linearizer.forward_bounds[key]['lb']
        ub = network_linearizer.forward_bounds[key]['ub']
        has_nan = torch.isnan(lb).any().item() or torch.isnan(ub).any().item()
        has_inf = torch.isinf(lb).any().item() or torch.isinf(ub).any().item()
        print(f"  {key}: shape={lb.shape}, NaN={has_nan}, Inf={has_inf}")
        if has_nan or has_inf:
            # Find first NaN/Inf location
            nan_mask = torch.isnan(lb) | torch.isnan(ub) | torch.isinf(lb) | torch.isinf(ub)
            nan_idx = nan_mask.nonzero(as_tuple=True)
            print(f"    First NaN/Inf at indices: {[(i.item(), j.item()) for i, j in zip(nan_idx[0][:3], nan_idx[1][:3])]}")

    # Step 2: Compute partial derivative bounds
    print(f"\n--- Step 2: compute_partial_derivative_bounds ---")
    try:
        network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)
        print(f"Partial derivative bounds computed")

        A_L = network_linearizer.derivative_bounds['A_L']
        b_L = network_linearizer.derivative_bounds['b_L']
        A_U = network_linearizer.derivative_bounds['A_U']
        b_U = network_linearizer.derivative_bounds['b_U']

        print(f"  A_L shape: {A_L.shape}, has NaN: {torch.isnan(A_L).any().item()}")
        print(f"  b_L shape: {b_L.shape}, has NaN: {torch.isnan(b_L).any().item()}")
        print(f"  A_U shape: {A_U.shape}, has NaN: {torch.isnan(A_U).any().item()}")
        print(f"  b_U shape: {b_U.shape}, has NaN: {torch.isnan(b_U).any().item()}")
    except Exception as e:
        print(f"ERROR in compute_partial_derivative_bounds: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Get Jacobian bounds
    print(f"\n--- Step 3: Get Jacobian Bounds ---")
    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
    print(f"A_L shape: {A_L.shape if hasattr(A_L, 'shape') else 'scalar'}")
    print(f"b_L shape: {b_L.shape if hasattr(b_L, 'shape') else 'scalar'}")

    A_L_squeezed = A_L.squeeze(1)  # [B, 1, n, n] -> [B, n, n]
    b_L_squeezed = b_L.squeeze(1)  # [B, 1, n] -> [B, n]
    A_U_squeezed = A_U.squeeze(1)
    b_U_squeezed = b_U.squeeze(1)

    print(f"After squeeze - A_L_squeezed shape: {A_L_squeezed.shape}")
    J_affine_L = (A_L_squeezed, b_L_squeezed)
    J_affine_U = (A_U_squeezed, b_U_squeezed)

    has_nan_A_L = torch.isnan(A_L_squeezed).any().item()
    has_nan_b_L = torch.isnan(b_L_squeezed).any().item()
    print(f"J_affine_L has NaN: A_L={has_nan_A_L}, b_L={has_nan_b_L}")

    # Step 4: Compute dynamics bounds
    print(f"\n--- Step 4: _compute_dynamics_bounds_taylor ---")
    try:
        f_affine_bounds, g_affine_bounds = _compute_dynamics_bounds_taylor(
            batch, dynamics_model, device=device, dtype=dtype
        )
        print(f"Dynamics bounds computed")
        f_affine_L, f_affine_U = f_affine_bounds
        print(f"f_affine_L[0] shape: {f_affine_L[0].shape if hasattr(f_affine_L[0], 'shape') else 'scalar'}")
        print(f"f_affine_L[0] has NaN: {torch.isnan(f_affine_L[0]).any().item()}")
        print(f"f_affine_L[1] has NaN: {torch.isnan(f_affine_L[1]).any().item()}")
        print(f"f_affine_U[0] has NaN: {torch.isnan(f_affine_U[0]).any().item()}")
        print(f"f_affine_U[1] has NaN: {torch.isnan(f_affine_U[1]).any().item()}")
    except Exception as e:
        print(f"ERROR in _compute_dynamics_bounds_taylor: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: McCormick product
    print(f"\n--- Step 5: McCormick Product (J · f) ---")
    eta = 0.5

    try:
        M_D, c_D = _batched_compute_mccormick_product_lower_bound(
            J_affine_L,     # (A_L, b_L) for J
            J_affine_U,     # (A_U, b_U) for J
            f_affine_L,     # (A_L, b_L) for f
            f_affine_U,     # (A_U, b_U) for f
            batch,
            eta=eta,
            device=device,
            dtype=dtype,
        )
        print(f"McCormick product computed")
        print(f"M_D shape: {M_D.shape}, has NaN: {torch.isnan(M_D).any().item()}")
        print(f"c_D shape: {c_D.shape}, has NaN: {torch.isnan(c_D).any().item()}")
        M_D_sum, c_D_sum = M_D.sum(dim=-2), c_D.sum(dim=-1)
        print(f"M_D_sum has NaN: {torch.isnan(M_D_sum).any().item()}")
        print(f"c_D_sum has NaN: {torch.isnan(c_D_sum).any().item()}")
    except Exception as e:
        print(f"ERROR in McCormick product: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 6: Alpha function
    print(f"\n--- Step 6: Alpha Function ---")
    (A_L_net, a_L_net), (A_U_net, a_U_net) = network_linearizer.get_network_linear_bounds()
    A_L_net = A_L_net.squeeze(1)
    a_L_net = a_L_net.squeeze(1)
    print(f"A_L_net shape: {A_L_net.shape}")
    print(f"a_L_net shape: {a_L_net.shape}")
    print(f"a_L_net[0]: {a_L_net[0].item():.6f}")

    alpha_A_L = dynamics_model.alpha_function(A_L_net)
    alpha_a_L = dynamics_model.alpha_function(a_L_net)
    print(f"alpha_A_L has NaN: {torch.isnan(alpha_A_L).any().item()}")
    print(f"alpha_a_L has NaN: {torch.isnan(alpha_a_L).any().item()}")

    M_total = M_D_sum + alpha_A_L
    c_total = c_D_sum + alpha_a_L
    print(f"M_total has NaN: {torch.isnan(M_total).any().item()}")
    print(f"c_total has NaN: {torch.isnan(c_total).any().item()}")

    # Step 7: Get affine function bounds (min_L)
    print(f"\n--- Step 7: min_L computation ---")
    try:
        min_L, _ = _batched_get_affine_function_bounds(
            (M_total.unsqueeze(1), c_total.unsqueeze(1)),
            batch,
            device=device,
            dtype=dtype,
        )
        min_L = min_L.squeeze(-1)
        print(f"min_L shape: {min_L.shape}, first value: {min_L[0].item():.6f}")
        print(f"min_L has NaN: {torch.isnan(min_L).any().item()}")
    except Exception as e:
        print(f"ERROR in min_L computation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 8: Gradient computation
    print(f"\n--- Step 8: Gradient Computation ---")
    params = list(model.parameters())

    # Clone the model to ensure requires_grad
    model_grad = model
    params_grad = list(model_grad.parameters())

    print(f"min_L[0] requires_grad: {min_L[0].requires_grad}")

    try:
        grad = torch.autograd.grad(
            outputs=min_L[0],
            inputs=params_grad,
            retain_graph=False,
        )
        grad_vec = torch.cat([g.flatten() for g in grad])
        print(f"Gradient computed, has NaN: {torch.isnan(grad_vec).any().item()}")
        print(f"Gradient shape: {grad_vec.shape}")
    except Exception as e:
        print(f"ERROR in gradient computation: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")

if __name__ == "__main__":
    # Test with a specific simplex that causes NaN
    debug_tanh_jacobian('Tanh', idx=0, region_type='safe')
    debug_tanh_jacobian('Relu', idx=0, region_type='safe')