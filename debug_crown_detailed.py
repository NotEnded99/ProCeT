"""
Check if the issue is in prj_vertex_lb / prj_vertex_ub McCormick einsum.
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
)

def check_crown_intermediates(activation, idx=0, region_type='safe'):
    """
    Inspect all intermediate values in crown linearization that could cause NaN in backward.
    """
    print(f"\n{'='*70}")
    print(f"CROWN INTERMEDIATES: {activation}, region {idx} ({region_type})")
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

    # Build computation
    network_lin = CrownPartialLinearization(model, dtype=dtype)
    network_lin.compute_network_bounds(batch)

    # Inspect pre_act_bounds for all layers
    print("\n--- Pre-activation bounds for each layer ---")
    for i in range(len(network_lin.fc_layers)):
        key = f"layer_{i}_pre_act_bounds"
        if key in network_lin.forward_bounds:
            bounds = network_lin.forward_bounds[key]
            print(f"  layer_{i}:")
            print(f"    lb: [{bounds['lb'].min().item():.4f}, {bounds['lb'].max().item():.4f}]")
            print(f"    ub: [{bounds['ub'].min().item():.4f}, {bounds['ub'].max().item():.4f}]")
            print(f"    prj_vertex_lb: {'None' if bounds['prj_vertex_lb'] is None else 'present'}")
            print(f"    prj_vertex_ub: {'None' if bounds['prj_vertex_ub'] is None else 'present'}")
            if bounds['prj_vertex_lb'] is not None:
                print(f"    prj_vertex_lb shape: {bounds['prj_vertex_lb'].shape}")
                print(f"    prj_vertex_lb range: [{bounds['prj_vertex_lb'].min().item():.4f}, {bounds['prj_vertex_lb'].max().item():.4f}]")
                print(f"    prj_vertex_lb has NaN: {torch.isnan(bounds['prj_vertex_lb']).any().item()}")

    # Compute partial derivative bounds
    print("\n--- Computing partial derivative bounds ---")
    network_lin.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    # Inspect the derivative bounds at each layer during the iterative computation
    print("\n--- Iterative Jacobian bound computation ---")

    L = len(network_lin.fc_layers)
    A_L_running, b_L_running, A_U_running, b_U_running = network_lin._get_jacobian_bounds_for_layer(L)
    print(f"After get_jacobian_bounds_for_layer({L}):")
    print(f"  A_L_running shape: {A_L_running.shape}")
    print(f"  A_L_running range: [{A_L_running.min().item():.6f}, {A_L_running.max().item():.6f}]")
    print(f"  has NaN: {torch.isnan(A_L_running).any().item()}")
    print(f"  has Inf: {torch.isinf(A_L_running).any().item()}")

    A_L_running = A_L_running.unsqueeze(0)
    b_L_running = b_L_running.unsqueeze(0)
    A_U_running = A_U_running.unsqueeze(0)
    b_U_running = b_U_running.unsqueeze(0)

    for i in range(L - 1, 0, -1):
        Lambda_L, lambda_L, Lambda_U, lambda_U = network_lin._get_jacobian_bounds_for_layer(i)
        pre_act_bounds = network_lin.forward_bounds[f"layer_{i-1}_pre_act_bounds"]

        print(f"\nIter i={i} (going backwards):")
        print(f"  Lambda_L shape: {Lambda_L.shape if hasattr(Lambda_L, 'shape') else 'N/A'}")
        print(f"  Lambda_L range: [{Lambda_L.min().item():.6f}, {Lambda_L.max().item():.6f}]")
        print(f"  lambda_L range: [{lambda_L.min().item():.6f}, {lambda_L.max().item():.6f}]")
        print(f"  has NaN: Lambda_L={torch.isnan(Lambda_L).any().item()}, lambda_L={torch.isnan(lambda_L).any().item()}")

        if pre_act_bounds["prj_vertex_lb"] is not None:
            print(f"  Using prj_vertex approach")
            pvlb = pre_act_bounds["prj_vertex_lb"]
            pvub = pre_act_bounds["prj_vertex_ub"]
            print(f"    prj_vertex_lb shape: {pvlb.shape}, range=[{pvlb.min().item():.4f}, {pvlb.max().item():.4f}]")
            print(f"    prj_vertex_ub shape: {pvub.shape}, range=[{pvub.min().item():.4f}, {pvub.max().item():.4f}]")

        # Compute the McCormick product
        try:
            A_L_new, b_L_new, A_U_new, b_U_new = network_lin._vectorized_mccormick_product(
                (A_L_running, b_L_running, A_U_running, b_U_running),
                (Lambda_L, lambda_L, Lambda_U, lambda_U),
                pre_act_bounds,
            )
            print(f"  After _vectorized_mccormick_product:")
            print(f"    A_L_new range: [{A_L_new.min().item():.6f}, {A_L_new.max().item():.6f}]")
            print(f"    b_L_new range: [{b_L_new.min().item():.6f}, {b_L_new.max().item():.6f}]")
            print(f"    has NaN: A_L_new={torch.isnan(A_L_new).any().item()}, b_L_new={torch.isnan(b_L_new).any().item()}")
            print(f"    has Inf: A_L_new={torch.isinf(A_L_new).any().item()}, b_L_new={torch.isinf(b_L_new).any().item()}")
        except Exception as e:
            print(f"  ERROR in _vectorized_mccormick_product: {e}")

        # Propagate
        A_L_running, b_L_running, A_U_running, b_U_running = network_lin._propagate_bounds_one_layer(
            i, A_L_new, b_L_new, A_U_new, b_U_new
        )
        print(f"  After _propagate_bounds_one_layer:")
        print(f"    A_L_running shape: {A_L_running.shape}")
        print(f"    A_L_running range: [{A_L_running.min().item():.6f}, {A_L_running.max().item():.6f}]")
        print(f"    has NaN: {torch.isnan(A_L_running).any().item()}")

    print("\n" + "="*70)

if __name__ == "__main__":
    check_crown_intermediates('Tanh', idx=0, region_type='safe')
    check_crown_intermediates('Relu', idx=0, region_type='safe')