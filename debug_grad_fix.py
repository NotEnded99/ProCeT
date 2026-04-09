"""
Test if diag_embed + einsum in McCormick product causes NaN in backward.
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

def test_diag_embed_grad():
    """Test if diag_embed in jacobian computation causes NaN."""
    print(f"\n{'='*70}")
    print(f"TEST DIAG_EMBED GRADIENT")
    print(f"{'='*70}")

    # Simulate the diag_embed operation from _get_jacobian_bounds_for_layer
    # Lambda_L = torch.diag_embed(term_L.transpose(-2, -1)).permute(0, 2, 1, 3)

    device = torch.device('cuda')
    dtype = torch.float32

    # Create test tensors simulating layer 3's computation
    # term_L: [b, k, p] = [1, 32, 32]
    b, k, p = 1, 32, 32

    # Tanh case: non-zero values
    term_L_tanh = torch.randn(b, k, p, device=device, dtype=dtype) * 0.3
    term_L_tanh = torch.clamp(term_L_tanh, min=-0.5, max=0.5)

    # Relu case: mostly zero
    term_L_relu = torch.randn(b, k, p, device=device, dtype=dtype) * 0.3
    term_L_relu = torch.clamp(term_L_relu, min=-0.5, max=0.5)
    # Make most values zero (like ReLU derivative)
    mask = torch.rand(b, k, p, device=device, dtype=dtype) > 0.3
    term_L_relu = term_L_relu * mask.float()

    for name, term_L in [('Tanh', term_L_tanh), ('Relu', term_L_relu)]:
        print(f"\n--- {name} ---")
        print(f"term_L: range=[{term_L.min().item():.4f}, {term_L.max().item():.4f}]")

        term_L_t = term_L.transpose(-2, -1)  # [b, p, k]
        print(f"term_L_t shape: {term_L_t.shape}")

        diag_L = torch.diag_embed(term_L_t)  # [b, p, k, p]
        print(f"diag_L shape: {diag_L.shape}")

        Lambda_L = diag_L.permute(0, 2, 1, 3)  # [b, k, p, p]
        print(f"Lambda_L shape: {Lambda_L.shape}")
        print(f"Lambda_L: range=[{Lambda_L.min().item():.4f}, {Lambda_L.max().item():.4f}]")

        # Test gradient through this
        Lambda_L.requires_grad_(True)
        try:
            loss = Lambda_L.sum()
            grad = torch.autograd.grad(loss, Lambda_L, retain_graph=False)[0]
            print(f"Gradient through diag_embed: has NaN={torch.isnan(grad).any().item()}")
        except Exception as e:
            print(f"Gradient through diag_embed: FAILED - {e}")

def test_einsum_grad():
    """Test if einsum in McCormick product causes NaN."""
    print(f"\n{'='*70}")
    print(f"TEST EINSUM GRADIENT")
    print(f"{'='*70}")

    device = torch.device('cuda')
    dtype = torch.float32

    # Simulate: einsum("bjpm,bmv->bjpv", Pi_L_pos, prj_vertex_lb) + pi_L.unsqueeze(-1)
    b, j, p, m, V = 1, 2, 32, 32, 3

    # Pi_L_pos: [b, j, p, m]
    Pi_L_pos = torch.randn(b, j, p, m, device=device, dtype=dtype) * 0.1
    Pi_L_pos = Pi_L_pos.clamp(min=-1, max=1)

    # prj_vertex_lb: [b, m, V]
    prj_vertex_lb = torch.randn(b, m, V, device=device, dtype=dtype) * 2.0

    for name, val in [('Tanh', 0.3), ('Relu', 0.0)]:
        print(f"\n--- {name} ---")
        Pi_test = Pi_L_pos.clone()
        if val == 0.0:
            # Make it sparse like ReLU
            mask = torch.rand_like(Pi_test) > 0.5
            Pi_test = Pi_test * mask.float()
            Pi_test = Pi_test.clamp(min=-1, max=1)

        print(f"Pi_L_pos: range=[{Pi_test.min().item():.4f}, {Pi_test.max().item():.4f}]")
        print(f"prj_vertex_lb: range=[{prj_vertex_lb.min().item():.4f}, {prj_vertex_lb.max().item():.4f}]")

        Pi_test.requires_grad_(True)
        prj_test = prj_vertex_lb.clone().requires_grad_(True)

        try:
            result = torch.einsum("bjpm,bmv->bjpv", Pi_test, prj_test)
            loss = result.sum()
            grad_pi, grad_prj = torch.autograd.grad(loss, [Pi_test, prj_test], retain_graph=False)
            print(f"Einsum gradient: has NaN={torch.isnan(grad_pi).any().item() or torch.isnan(grad_prj).any().item()}")
        except Exception as e:
            print(f"Einsum gradient: FAILED - {e}")

def test_tanh_activation_grad():
    """Test if Tanh activation relaxation's derivative computation causes NaN in backward."""
    print(f"\n{'='*70}")
    print(f"TEST TANH ACTIVATION RELAXATION GRADIENT")
    print(f"{'='*70}")

    from lbp_neural_cbf.linearization.activations.tanh import TanhActivationRelaxation

    device = torch.device('cuda')
    dtype = torch.float32

    tanh_relax = TanhActivationRelaxation()

    # Test with pre-activation bounds similar to what we see in practice
    y_lb = torch.tensor([[-3.0, 2.0]], device=device, dtype=dtype)  # [1, 32]
    y_ub = torch.tensor([[-2.8, 2.3]], device=device, dtype=dtype)  # [1, 32]

    # Forward pass
    gamma_L, delta_L, gamma_U, delta_U = tanh_relax.relax_activation_derivative(y_lb, y_ub)

    print(f"gamma_L: range=[{gamma_L.min().item():.4f}, {gamma_L.max().item():.4f}], has NaN={torch.isnan(gamma_L).any().item()}")
    print(f"delta_L: range=[{delta_L.min().item():.4f}, {delta_L.max().item():.4f}], has NaN={torch.isnan(delta_L).any().item()}")

    # Backward pass
    y_lb.requires_grad_(True)
    y_ub.requires_grad_(True)

    try:
        gamma_L_test, delta_L_test, gamma_U_test, delta_U_test = tanh_relax.relax_activation_derivative(y_lb, y_ub)
        loss = gamma_L_test.sum() + delta_L_test.sum()
        grad_lb, grad_ub = torch.autograd.grad(loss, [y_lb, y_ub], retain_graph=False)
        print(f"Gradient: has NaN={torch.isnan(grad_lb).any().item() or torch.isnan(grad_ub).any().item()}")
    except Exception as e:
        print(f"Gradient: FAILED - {e}")

def test_network_forward_and_backward():
    """Test actual network forward and backward to confirm NaN location."""
    print(f"\n{'='*70}")
    print(f"TEST NETWORK FORWARD/BACKWARD")
    print(f"{'='*70}")

    device = torch.device('cuda')
    dtype = torch.float32

    from lbp_neural_cbf.cbf.fossil_dynamics import Barrier3System
    from lbp_neural_cbf.cbf.network import BarrierNN

    dynamics_model = Barrier3System(alpha=1.0)

    for activation in ['Tanh', 'Relu']:
        print(f"\n--- {activation} ---")
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

        # Get a simple input
        x = torch.randn(1, 2, device=device, dtype=dtype, requires_grad=True)

        # Forward
        h = model(x)
        print(f"Forward output: {h.item():.6f}, has NaN={torch.isnan(h).any().item()}")

        # Backward
        model.zero_grad()
        try:
            h.backward()
            grad = x.grad
            print(f"Input gradient: has NaN={torch.isnan(grad).any().item()}")
        except Exception as e:
            print(f"Input gradient: FAILED - {e}")

if __name__ == "__main__":
    test_diag_embed_grad()
    test_einsum_grad()
    test_tanh_activation_grad()
    test_network_forward_and_backward()