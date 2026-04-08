"""Quick test for compute_jacobian_matrix_fast fix."""
import torch
import sys
sys.path.insert(0, '/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4')

from New_repair.geometry_module_new import compute_jacobian_matrix_fast
from lbp_neural_cbf.dynamics.barr_system import BarrSystem

# Load model
model = torch.load('data/New_models_Hard_Tanh/barr1_cbf.pth', weights_only=False)
model.eval()

# Load verified regions
data = torch.load('New_repair/regions/verified_regions_barr1_Tanh.pt', weights_only=False)
V_safe = data['V_safe']
V_unsafe = data['V_unsafe']

print(f"V_safe count: {len(V_safe)}")
print(f"V_unsafe count: {len(V_unsafe)}")
print(f"V_safe[0] shape: {V_safe[0].shape}")

# Create dynamics model
dynamics = BarrSystem()

# Test compute_jacobian_matrix_fast
print("\nTesting compute_jacobian_matrix_fast...")
J = compute_jacobian_matrix_fast(
    model=model,
    V_safe=V_safe[:10],  # Use only first 10 for quick test
    V_unsafe=V_unsafe[:10],
    dynamics_model=dynamics,
    translator=None,
)

print(f"Jacobian shape: {J.shape}")
print("Test passed!")
