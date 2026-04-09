"""
Trace gradient through each layer to find exact NaN layer.
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

def trace_gradient_by_layer(activation, idx=0, region_type='safe'):
    """
    Test gradient computation for each layer separately.
    """
    print(f"\n{'='*70}")
    print(f"TRACE GRADIENT BY LAYER: {activation}, region {idx} ({region_type})")
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
    network_lin.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    A_L, b_L, A_U, b_U = network_lin.get_partial_derivative_bounds()
    A_L_s = A_L.squeeze(1)
    b_L_s = b_L.squeeze(1)
    A_U_s = A_U.squeeze(1)
    b_U_s = b_U.squeeze(1)
    J_L = (A_L_s, b_L_s)
    J_U = (A_U_s, b_U_s)

    f_bounds, _ = _compute_dynamics_bounds_taylor(batch, dynamics_model, device=device, dtype=dtype)
    f_L, f_U = f_bounds

    eta = 0.5
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_L, J_U, f_L, f_U, batch, eta=eta, device=device, dtype=dtype,
    )
    M_D_sum = M_D.sum(dim=-2)
    c_D_sum = c_D.sum(dim=-1)

    (A_L_net, a_L_net), _ = network_lin.get_network_linear_bounds()
    A_L_net = A_L_net.squeeze(1)
    a_L_net = a_L_net.squeeze(1)
    alpha_A_L = dynamics_model.alpha_function(A_L_net)
    alpha_a_L = dynamics_model.alpha_function(a_L_net)

    M_total = M_D_sum + alpha_A_L
    c_total = c_D_sum + alpha_a_L

    min_L, _ = _batched_get_affine_function_bounds(
        (M_total.unsqueeze(1), c_total.unsqueeze(1)),
        batch, device=device, dtype=dtype,
    )
    min_L = min_L.squeeze(-1)

    # Now test gradient for each layer
    params = list(model.parameters())
    layer_names = ['layer0_fc', 'layer0_bias', 'layer1_fc', 'layer1_bias', 'layer2_fc', 'layer2_bias', 'layer3_fc', 'layer3_bias']

    print("\n--- Gradient by layer ---")
    for i, (p, name) in enumerate(zip(params, layer_names)):
        model.zero_grad()
        try:
            grad_i = torch.autograd.grad(
                outputs=min_L[0],
                inputs=p,
                retain_graph=True,
            )
            g = grad_i[0].flatten()
            has_nan = torch.isnan(g).any().item()
            has_inf = torch.isinf(g).any().item()
            grad_norm = g.norm().item()
            print(f"  {name}: |g|={grad_norm:12.6f}, NaN={has_nan}, Inf={has_inf}")
            if has_nan:
                print(f"    NaN at first few: {g[:10].detach().cpu().numpy()}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Now let's look at the derivative relaxation values for each layer
    print("\n--- Activation derivative relaxation analysis ---")
    for i in range(len(network_lin.fc_layers)):
        key = f"layer_{i}_pre_act_bounds"
        if key in network_lin.forward_bounds:
            lb = network_lin.forward_bounds[key]['lb']
            ub = network_lin.forward_bounds[key]['ub']
            act_relax = network_lin.activation_relaxation
            if hasattr(act_relax, 'relax_activation_derivative'):
                gamma_L, delta_L, gamma_U, delta_U = act_relax.relax_activation_derivative(lb, ub)
                print(f"  layer_{i}: y_lb=[{lb.min().item():.4f}, {lb.max().item():.4f}], y_ub=[{ub.min().item():.4f}, {ub.max().item():.4f}]")
                print(f"    gamma_L: [{gamma_L.min().item():.4f}, {gamma_L.max().item():.4f}], has NaN={torch.isnan(gamma_L).any().item()}")
                print(f"    gamma_U: [{gamma_U.min().item():.4f}, {gamma_U.max().item():.4f}], has NaN={torch.isnan(gamma_U).any().item()}")
                # Check for extreme values
                if (gamma_L.abs() > 10).any() or (gamma_U.abs() > 10).any():
                    print(f"    WARNING: Large gamma values detected!")

    print("\n" + "="*70)

if __name__ == "__main__":
    trace_gradient_by_layer('Tanh', idx=0, region_type='safe')
    trace_gradient_by_layer('Relu', idx=0, region_type='safe')