"""
Test if nan_to_num on J fixes the QP solver.
"""

import sys
import os
import torch
import numpy as np
import cvxpy as cp

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

from lbp_neural_cbf.cbf.fossil_dynamics import Barrier3System
from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.translators import TorchTranslator
from New_repair.geometry_module_new import compute_jacobian_matrix

def test_qp_with_nan_handling(activation):
    """Test if handling NaN in J fixes the QP solver."""
    print(f"\n{'='*70}")
    print(f"TEST QP WITH NAN HANDLING: {activation}")
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

    translator = TorchTranslator(device=device)

    # Use a small subset for speed
    n_test = 20
    V_safe_test = V_safe[:n_test]
    V_unsafe_test = V_unsafe[:n_test]

    print(f"\nComputing J with {n_test} safe + {n_test} unsafe regions...")

    # Compute J without multi-threading to see all debug messages
    J = compute_jacobian_matrix(
        model,
        V_safe_test,
        V_unsafe_test,
        dynamics_model=dynamics_model,
        translator=translator,
        max_workers=1
    )

    print(f"J.shape: {J.shape}")
    print(f"J has NaN: {torch.isnan(J).any().item()}")
    nan_count = torch.isnan(J).sum().item()
    total_count = J.numel()
    print(f"NaN count: {nan_count}/{total_count} ({100*nan_count/total_count:.2f}%)")

    # Check J row by row
    nan_rows = torch.isnan(J).any(dim=1)
    print(f"Rows with NaN: {nan_rows.sum().item()}/{J.shape[0]}")

    # Test QP with raw J (will fail for Tanh)
    print(f"\n--- QP with raw J ---")
    J_np = J.detach().cpu().numpy()
    g_np = np.random.randn(J_np.shape[1])

    # Normalize J
    epsilon = 1e-8
    J_norms = np.linalg.norm(J_np, axis=1, keepdims=True)
    J_hat = J_np / (J_norms + epsilon)
    g_norm = np.linalg.norm(g_np)
    g_hat = g_np / (g_norm + epsilon)

    if np.isnan(J_hat).any():
        print("J_hat has NaN after normalization!")

    lam = cp.Variable(J_hat.shape[0], nonneg=True)
    residual = J_hat.T @ lam - g_hat
    objective = cp.Minimize(0.5 * cp.sum_squares(residual))
    prob = cp.Problem(objective)

    print("Solving raw QP...")
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        print(f"Raw QP status: {prob.status}")
    except Exception as e:
        print(f"Raw QP FAILED: {e}")

    # Test QP with nan_to_num J
    print(f"\n--- QP with nan_to_num J ---")
    J_clean = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    J_clean_np = J_clean.detach().cpu().numpy()

    J_norms_clean = np.linalg.norm(J_clean_np, axis=1, keepdims=True)
    J_hat_clean = J_clean_np / (J_norms_clean + epsilon)

    if np.isnan(J_hat_clean).any():
        print("J_hat_clean STILL has NaN!")
    else:
        print("J_hat_clean is clean (no NaN)")

    lam_clean = cp.Variable(J_hat_clean.shape[0], nonneg=True)
    residual_clean = J_hat_clean.T @ lam_clean - g_hat
    objective_clean = cp.Minimize(0.5 * cp.sum_squares(residual_clean))
    prob_clean = cp.Problem(objective_clean)

    print("Solving clean QP...")
    try:
        prob_clean.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        print(f"Clean QP status: {prob_clean.status}")
        if prob_clean.status in ["optimal", "optimal_inaccurate"]:
            print(f"Clean QP optimal value: {prob_clean.value:.6f}")
    except Exception as e:
        print(f"Clean QP FAILED: {e}")

    # Check the rows of J to understand which simplexes cause NaN
    print(f"\n--- Analyzing NaN rows ---")
    for i in range(J.shape[0]):
        if torch.isnan(J[i]).any():
            print(f"  Row {i}: all NaN" if torch.isnan(J[i]).all() else f"  Row {i}: partial NaN, first NaN at idx {torch.isnan(J[i]).nonzero()[0][0].item()}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    test_qp_with_nan_handling('Tanh')
    test_qp_with_nan_handling('Relu')