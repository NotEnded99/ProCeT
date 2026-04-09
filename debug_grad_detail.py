"""
Debug script to trace gradient NaN location in Tanh autograd.
"""

import sys
import os
import torch
import numpy as np

# Setup path
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

def debug_gradient_autograd(activation, idx=0, region_type='safe'):
    """Debug the gradient computation in detail."""
    print(f"\n{'='*70}")
    print(f"DEBUG GRADIENT: {activation}, region {idx} ({region_type})")
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

    # Get one simplex
    if region_type == 'safe':
        simplex_vertices = V_safe[idx]
        if isinstance(simplex_vertices, torch.Tensor):
            simplex_vertices = simplex_vertices.cpu().numpy()
    else:
        simplex_vertices = V_unsafe[idx]
        if isinstance(simplex_vertices, torch.Tensor):
            simplex_vertices = simplex_vertices.cpu().numpy()

    batch = [SimplicialRegion(simplex_vertices, output_dim=None)]

    # Create CrownPartialLinearization
    network_linearizer = CrownPartialLinearization(model, dtype=dtype)
    network_linearizer.compute_network_bounds(batch)
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    # Get Jacobian bounds
    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
    A_L = A_L.squeeze(1)  # [B, 1, n, n] -> [B, n, n]
    b_L = b_L.squeeze(1)  # [B, 1, n] -> [B, n]
    A_U = A_U.squeeze(1)
    b_U = b_U.squeeze(1)
    J_affine_L = (A_L, b_L)
    J_affine_U = (A_U, b_U)

    # Compute dynamics bounds
    f_affine_bounds, g_affine_bounds = _compute_dynamics_bounds_taylor(
        batch, dynamics_model, device=device, dtype=dtype
    )
    f_affine_L, f_affine_U = f_affine_bounds

    # McCormick product
    eta = 0.5
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_affine_L, J_affine_U, f_affine_L, f_affine_U, batch, eta=eta, device=device, dtype=dtype,
    )
    M_D, c_D = M_D.sum(dim=-2), c_D.sum(dim=-1)

    # Alpha function
    (A_L_net, a_L_net), (A_U_net, a_U_net) = network_linearizer.get_network_linear_bounds()
    A_L_net = A_L_net.squeeze(1)
    a_L_net = a_L_net.squeeze(1)
    alpha_A_L = dynamics_model.alpha_function(A_L_net)
    alpha_a_L = dynamics_model.alpha_function(a_L_net)

    M_total = M_D + alpha_A_L
    c_total = c_D + alpha_a_L

    # min_L computation
    min_L, _ = _batched_get_affine_function_bounds(
        (M_total.unsqueeze(1), c_total.unsqueeze(1)),
        batch,
        device=device,
        dtype=dtype,
    )
    min_L = min_L.squeeze(-1)

    print(f"\n--- Gradient Checkpoints ---")
    print(f"min_L value: {min_L[0].item():.6f}")
    print(f"min_L requires_grad: {min_L.requires_grad}")
    print(f"min_L.grad_fn: {min_L.grad_fn}")

    # Check parameter norms
    params = list(model.parameters())
    print(f"\n--- Parameter Statistics ---")
    for i, p in enumerate(params):
        print(f"  param[{i}]: shape={p.shape}, mean={p.mean().item():.4f}, std={p.std().item():.4f}, "
              f"min={p.min().item():.4f}, max={p.max().item():.4f}")

    # Now compute gradient with hooks to find where NaN appears
    print(f"\n--- Backward Pass with NaN Detection ---")

    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)

    try:
        grad = torch.autograd.grad(
            outputs=min_L[0],
            inputs=params,
            retain_graph=False,
        )
        grad_vec = torch.cat([g.flatten() for g in grad])
        print(f"Gradient computed successfully!")
        print(f"  has NaN: {torch.isnan(grad_vec).any().item()}")
        print(f"  has Inf: {torch.isinf(grad_vec).any().item()}")
        nan_count = torch.isnan(grad_vec).sum().item()
        inf_count = torch.isinf(grad_vec).sum().item()
        print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
        if nan_count > 0:
            nan_indices = torch.isnan(grad_vec).nonzero(as_tuple=True)[0][:10]
            print(f"  First NaN indices: {nan_indices}")
            for idx_nan in nan_indices:
                # Find which param this belongs to
                cumsum = 0
                for pi, p in enumerate(params):
                    if idx_nan < cumsum + p.numel():
                        local_idx = idx_nan.item() - cumsum
                        print(f"    NaN at global idx {idx_nan.item()} -> param[{pi}], local_idx={local_idx}, value={grad_vec[idx_nan].item()}")
                        break
                    cumsum += p.numel()
    except Exception as e:
        print(f"ERROR during backward: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")

if __name__ == "__main__":
    debug_gradient_autograd('Tanh', idx=0, region_type='safe')
    debug_gradient_autograd('Relu', idx=0, region_type='safe')