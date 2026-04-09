"""
Pinpoint exact MulBackward0 NaN location by wrapping each operation.
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

def check_gradients_detailed(activation, idx=0, region_type='safe'):
    """Check gradient values step by step with per-parameter tracking."""
    print(f"\n{'='*70}")
    print(f"GRADIENT DETAIL: {activation}, region {idx} ({region_type})")
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

    # === Full computation with gradient hooks ===
    network_linearizer = CrownPartialLinearization(model, dtype=dtype)
    network_linearizer.compute_network_bounds(batch)
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
    A_L = A_L.squeeze(1)
    b_L = b_L.squeeze(1)
    A_U = A_U.squeeze(1)
    b_U = b_U.squeeze(1)
    J_affine_L = (A_L, b_L)
    J_affine_U = (A_U, b_U)

    f_affine_bounds, _ = _compute_dynamics_bounds_taylor(
        batch, dynamics_model, device=device, dtype=dtype
    )
    f_affine_L, f_affine_U = f_affine_bounds

    eta = 0.5
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_affine_L, J_affine_U, f_affine_L, f_affine_U, batch, eta=eta, device=device, dtype=dtype,
    )
    M_D_sum = M_D.sum(dim=-2)
    c_D_sum = c_D.sum(dim=-1)

    (A_L_net, a_L_net), _ = network_linearizer.get_network_linear_bounds()
    A_L_net = A_L_net.squeeze(1)
    a_L_net = a_L_net.squeeze(1)
    alpha_A_L = dynamics_model.alpha_function(A_L_net)
    alpha_a_L = dynamics_model.alpha_function(a_L_net)

    M_total = M_D_sum + alpha_A_L
    c_total = c_D_sum + alpha_a_L

    min_L, _ = _batched_get_affine_function_bounds(
        (M_total.unsqueeze(1), c_total.unsqueeze(1)),
        batch,
        device=device,
        dtype=dtype,
    )
    min_L = min_L.squeeze(-1)

    print(f"\n=== Testing gradient for each parameter ===")
    params = list(model.parameters())
    param_names = []
    for name, _ in model.named_parameters():
        param_names.append(name)

    grad_norms = []
    for i, (p, name) in enumerate(zip(params, param_names)):
        # Compute gradient w.r.t. this parameter only
        model.zero_grad()
        try:
            grad_i = torch.autograd.grad(
                outputs=min_L[0],
                inputs=p,
                retain_graph=True,
            )
            g = grad_i[0].flatten()
            grad_norm = g.norm().item()
            has_nan = torch.isnan(g).any().item()
            has_inf = torch.isinf(g).any().item()
            grad_norms.append(grad_norm)
            print(f"  param[{i}] {name[:30]:30s}: |g|={grad_norm:12.6f}, NaN={has_nan}, Inf={has_inf}")
            if has_nan or has_inf:
                nan_count = torch.isnan(g).sum().item()
                inf_count = torch.isinf(g).sum().item()
                print(f"    NaN={nan_count}, Inf={inf_count}")
                # Find first bad value
                bad_idx = torch.where(torch.isnan(g) | torch.isinf(g))[0]
                if len(bad_idx) > 0:
                    print(f"    First bad at idx {bad_idx[0].item()}")
        except Exception as e:
            grad_norms.append(float('inf'))
            print(f"  param[{i}] {name[:30]:30s}: ERROR - {e}")

    print(f"\n--- Summary ---")
    print(f"Total params with NaN/Inf gradients:")
    total_bad = sum(1 for g in grad_norms if np.isnan(g) or np.isinf(g))
    print(f"  Count: {total_bad}/{len(params)}")
    if total_bad > 0:
        print(f"  First bad param indices: {[i for i, g in enumerate(grad_norms) if np.isnan(g) or np.isinf(g)]}")

    # Full gradient with nan_to_num
    model.zero_grad()
    try:
        grad_full = torch.autograd.grad(
            outputs=min_L[0],
            inputs=params,
            retain_graph=False,
        )
        grad_vec = torch.cat([g.flatten() for g in grad_full])
        grad_vec_clean = torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"\nFull gradient (with nan_to_num): has NaN={torch.isnan(grad_vec_clean).any().item()}")
        print(f"NaN replaced: {(torch.isnan(grad_vec) & ~torch.isnan(grad_vec_clean)).sum().item()}")
        print(f"Inf replaced: {(torch.isinf(grad_vec) & ~torch.isinf(grad_vec_clean)).sum().item()}")
    except Exception as e:
        print(f"\nFull gradient FAILED: {e}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    check_gradients_detailed('Tanh', idx=0, region_type='safe')
    check_gradients_detailed('Relu', idx=0, region_type='safe')