"""
Debug script to identify NaN source in Tanh/Sigmoid vs Relu.
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
from New_repair.geometry_module_new import compute_simplex_bound_batch, compute_jacobian_matrix

def debug_activation(activation, system_name, num_samples=5):
    """Debug a specific activation function with a specific system."""
    print(f"\n{'='*70}")
    print(f"DEBUG: {activation} activation with {system_name}")
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

    print(f"V_safe: {len(V_safe)}, V_unsafe: {len(V_unsafe)}")
    print(f"F_h_positive: {len(F_h_positive)}, F_safe_violation: {len(F_safe_violation)}")

    translator = TorchTranslator(device=device)

    # Test unsafe region (h_lb computation)
    print(f"\n--- Testing Unsafe Region (h_lb computation) ---")
    if len(V_unsafe) > 0:
        sample_indices = np.random.choice(len(V_unsafe), min(num_samples, len(V_unsafe)), replace=False)
        for idx in sample_indices:
            simplex = V_unsafe[idx]
            try:
                h_lb, h_ub = compute_simplex_bound_batch(
                    model, [simplex], 'unsafe',
                    dynamics_model=None, translator=translator
                )
                print(f"  V_unsafe[{idx}]: h_lb={h_lb.item():.6f}, h_ub={h_ub.item():.6f}, NaN={torch.isnan(h_lb).any().item() or torch.isnan(h_ub).any().item()}")
            except Exception as e:
                print(f"  V_unsafe[{idx}]: ERROR - {e}")

    # Test safe region (min_L computation)
    print(f"\n--- Testing Safe Region (min_L computation) ---")
    if len(V_safe) > 0:
        sample_indices = np.random.choice(len(V_safe), min(num_samples, len(V_safe)), replace=False)
        for idx in sample_indices:
            simplex = V_safe[idx]
            try:
                min_L = compute_simplex_bound_batch(
                    model, [simplex], 'safe',
                    dynamics_model=dynamics_model, translator=translator
                )
                print(f"  V_safe[{idx}]: min_L={min_L.item():.6f}, NaN={torch.isnan(min_L).any().item()}")
            except Exception as e:
                print(f"  V_safe[{idx}]: ERROR - {e}")

    # Test F_h_positive (failure region - unsafe)
    print(f"\n--- Testing F_h_positive_in_unsafe (failure regions) ---")
    if len(F_h_positive) > 0:
        sample_indices = np.random.choice(len(F_h_positive), min(num_samples, len(F_h_positive)), replace=False)
        for idx in sample_indices:
            simplex = F_h_positive[idx]
            try:
                h_lb, h_ub = compute_simplex_bound_batch(
                    model, [simplex], 'unsafe',
                    dynamics_model=None, translator=translator
                )
                print(f"  F_h[{idx}]: h_lb={h_lb.item():.6f}, h_ub={h_ub.item():.6f}, NaN={torch.isnan(h_lb).any().item() or torch.isnan(h_ub).any().item()}")
            except Exception as e:
                print(f"  F_h[{idx}]: ERROR - {e}")

    # Test F_safe_violation (failure region - safe)
    print(f"\n--- Testing F_safe_cbf_violation (failure regions) ---")
    if len(F_safe_violation) > 0:
        sample_indices = np.random.choice(len(F_safe_violation), min(num_samples, len(F_safe_violation)), replace=False)
        for idx in sample_indices:
            simplex = F_safe_violation[idx]
            try:
                min_L = compute_simplex_bound_batch(
                    model, [simplex], 'safe',
                    dynamics_model=dynamics_model, translator=translator
                )
                print(f"  F_safe[{idx}]: min_L={min_L.item():.6f}, NaN={torch.isnan(min_L).any().item()}")
            except Exception as e:
                print(f"  F_safe[{idx}]: ERROR - {e}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    print("Debug NaN issue for Tanh vs Relu activation")
    print("="*70)

    # Test Relu first (should work)
    debug_activation('Relu', 'barr3')

    # Test Tanh (likely to have NaN)
    debug_activation('Tanh', 'barr3')

    # Test Sigmoid (likely to have NaN)
    debug_activation('Sigmoid', 'barr3')