"""
Debug script to identify NaN source in the inner loop QP solver.
"""

import sys
import os
import torch
import numpy as np
import cvxpy as cp

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
from New_repair.geometry_module_new import compute_simplex_bound_batch, compute_jacobian_matrix
import copy

def debug_inner_loop(activation, system_name):
    """Debug the inner loop repair with QP solver."""
    print(f"\n{'='*70}")
    print(f"DEBUG INNER LOOP: {activation} activation with {system_name}")
    print(f"{'='*70}")

    device = torch.device('cuda')

    # Load dynamics
    dynamics_class = Barrier3System
    dynamics_model = dynamics_class(alpha=1.0)
    dynamics_model.activation_fnc = activation

    # Load model
    model_dir = f"data/New_models_Hard_{activation}"
    model_path = f"{model_dir}/{dynamics_model.system_name}_cbf.pth"

    print(f"Loading model from: {model_path}")
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
    print(f"Loading regions from: {regions_path}")
    regions_data = torch.load(regions_path, map_location=device, weights_only=False)

    V_safe = regions_data['V_safe']
    V_unsafe = regions_data['V_unsafe']
    F_h_positive = regions_data['F_h_positive_in_unsafe']
    F_safe_violation = regions_data['F_safe_cbf_violation']
    F_depth = regions_data['F_depth_limit_reached']
    F_unsafe_split = regions_data['F_unsafe_cannot_split']

    print(f"V_safe: {len(V_safe)}, V_unsafe: {len(V_unsafe)}")
    print(f"F_h_positive: {len(F_h_positive)}, F_safe_violation: {len(F_safe_violation)}")
    print(f"F_depth: {len(F_depth)}, F_unsafe_split: {len(F_unsafe_split)}")

    translator = TorchTranslator(device=device)

    # Compute Jacobian matrix
    print(f"\n--- Computing Jacobian Matrix ---")
    try:
        J = compute_jacobian_matrix(
            model,
            V_safe,
            V_unsafe,
            dynamics_model=dynamics_model,
            translator=translator,
            max_workers=1  # Single threaded for debugging
        )
        print(f"J.shape: {J.shape}")
        print(f"J has NaN: {torch.isnan(J).any().item()}")
        print(f"J has Inf: {torch.isinf(J).any().item()}")
        if torch.isnan(J).any() or torch.isinf(J).any():
            nan_count = torch.isnan(J).sum().item()
            inf_count = torch.isinf(J).sum().item()
            print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
            # Find which rows have NaN/Inf
            nan_rows = torch.isnan(J).any(dim=1).nonzero(as_tuple=True)[0]
            inf_rows = torch.isinf(J).any(dim=1).nonzero(as_tuple=True)[0]
            print(f"  NaN rows: {nan_rows[:10] if len(nan_rows) > 10 else nan_rows}")
            print(f"  Inf rows: {inf_rows[:10] if len(inf_rows) > 10 else inf_rows}")
    except Exception as e:
        print(f"ERROR computing J: {e}")
        import traceback
        traceback.print_exc()
        return

    # Simulate one inner loop step
    print(f"\n--- Simulating Inner Loop Step ---")
    from New_repair.optimizer_module import inner_loop_repair_with_qp

    try:
        inner_history = inner_loop_repair_with_qp(
            model=model,
            J=J,
            F_h_positive_in_unsafe=F_h_positive,
            F_safe_cbf_violation=F_safe_violation,
            F_depth_limit_reached=F_depth,
            F_unsafe_cannot_split=F_unsafe_split,
            dynamics_model=dynamics_model,
            translator=translator,
            num_inner_steps=3,
            batch_ratio=0.2,
            lr=1e-3,
            V_safe=V_safe,
            V_unsafe=V_unsafe,
            lambda_penalty=1.0,
            lambda_stability=0.1,
            lambda_barrier=0.1,
            gamma_safe=0.1,
            gamma_unsafe=0.1,
            verified_batch_ratio=0.1,
            verbose=True,
            seed=42,
        )
        print(f"Inner loop completed: {len(inner_history)} steps")
        for i, h in enumerate(inner_history):
            print(f"  Step {i+1}: loss={h['loss']:.6f}, grad_norm={h['grad_norm']:.6f}, update_norm={h['update_norm']:.6f}")
    except Exception as e:
        print(f"ERROR in inner loop: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")

def debug_qp_solver():
    """Debug the QP solver specifically."""
    print(f"\n{'='*70}")
    print(f"DEBUG QP SOLVER")
    print(f"{'='*70}")

    # Create a simple test case
    P = 100  # number of parameters
    N = 50   # number of constraints

    # Create random J and g
    np.random.seed(42)
    J_np = np.random.randn(N, P)
    g_np = np.random.randn(P)

    # Normalize J rows
    epsilon = 1e-8
    J_norms = np.linalg.norm(J_np, axis=1, keepdims=True)
    J_hat = J_np / (J_norms + epsilon)
    g_hat = g_np / (np.linalg.norm(g_np) + epsilon)

    print(f"J_hat shape: {J_hat.shape}, g_hat shape: {g_hat.shape}")
    print(f"J_hat has NaN: {np.isnan(J_hat).any()}")
    print(f"g_hat has NaN: {np.isnan(g_hat).any()}")

    # Build QP
    lam = cp.Variable(N, nonneg=True)
    residual = J_hat.T @ lam - g_hat
    objective = cp.Minimize(0.5 * cp.sum_squares(residual))
    prob = cp.Problem(objective)

    print(f"\nSolving QP...")
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        print(f"QP status: {prob.status}")
        print(f"QP optimal value: {prob.value}")
        lam_value = lam.value
        print(f"lam has NaN: {np.isnan(lam_value).any() if lam_value is not None else 'None'}")
    except Exception as e:
        print(f"QP ERROR: {e}")

    # Now test with the actual J from Tanh model
    print(f"\n--- Testing with actual J from Tanh model ---")
    device = torch.device('cuda')

    dynamics_class = Barrier3System
    dynamics_model = dynamics_class(alpha=1.0)
    dynamics_model.activation_fnc = 'Tanh'

    model_dir = f"data/New_models_Hard_Tanh"
    model_path = f"{model_dir}/barr3_cbf.pth"

    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device,
        activation_fnc='Tanh'
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    regions_path = f"New_repair/regions/verified_regions_barr3_Tanh.pt"
    regions_data = torch.load(regions_path, map_location=device, weights_only=False)

    V_safe = regions_data['V_safe']
    V_unsafe = regions_data['V_unsafe']

    translator = TorchTranslator(device=device)

    # Compute J with only first 100 vertices
    print(f"Computing J with V_safe[:50] + V_unsafe[:50]...")
    test_V_safe = V_safe[:50]
    test_V_unsafe = V_unsafe[:50]

    J_test = compute_jacobian_matrix(
        model,
        test_V_safe,
        test_V_unsafe,
        dynamics_model=dynamics_model,
        translator=translator,
        max_workers=1
    )

    print(f"J_test.shape: {J_test.shape}")
    print(f"J_test has NaN: {torch.isnan(J_test).any().item()}")

    # Test QP with this J
    J_np_test = J_test.detach().cpu().numpy()
    g_np_test = np.random.randn(J_np_test.shape[1])

    # Normalize
    J_norms_test = np.linalg.norm(J_np_test, axis=1, keepdims=True)
    J_hat_test = J_np_test / (J_norms_test + epsilon)
    g_hat_test = g_np_test / (np.linalg.norm(g_np_test) + epsilon)

    print(f"J_hat_test has NaN: {np.isnan(J_hat_test).any()}")

    lam_test = cp.Variable(J_np_test.shape[0], nonneg=True)
    residual_test = J_hat_test.T @ lam_test - g_hat_test
    objective_test = cp.Minimize(0.5 * cp.sum_squares(residual_test))
    prob_test = cp.Problem(objective_test)

    print(f"Solving test QP...")
    try:
        prob_test.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        print(f"Test QP status: {prob_test.status}")
        lam_value_test = lam_test.value
        if lam_value_test is not None:
            print(f"lam_test has NaN: {np.isnan(lam_value_test).any()}")
            print(f"lam_test max: {np.max(lam_value_test):.4f}")
    except Exception as e:
        print(f"Test QP ERROR: {e}")

if __name__ == "__main__":
    print("Debug NaN issue - Inner Loop QP")
    print("="*70)

    debug_qp_solver()
    debug_inner_loop('Tanh', 'barr3')