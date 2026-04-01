#!/usr/bin/env python3
"""
Test that verification fix works correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lbp_neural_cbf.cbf.fossil_dynamics import Barrier3System
from repair.verification_module import ICGARVerification

print("=" * 60)
print("Testing ICGAR Verification Fix")
print("=" * 60)
print()

# Test barr3 system
dynamics_model = Barrier3System()
model_path = "data/mine_models_relu/barr3_cbf.onnx"

# print(f"System: {dynamics_model.system_name}")
# print(f"Model: {model_path}")
# print(f"Expected baseline: 72.36% certified")
# print()

verifier = ICGARVerification(
    model_path=model_path,
    dynamics_model=dynamics_model,
    max_depth=13,
    region_type="simplicial",
    executor_type="single",
    use_gpu=True,
    batch_size=512,
    verbose=False
)

results = verifier.run_verification()

certified = results['certified_percentage']
uncertified = results['uncertified_percentage']
diff = abs(certified - 72.36)

print("Verification Results:")
print(f"  Certified: {certified:.2f}%")
print(f"  Uncertified: {uncertified:.2f}%")
print(f"  Time: {results['computation_time']:.2f}s")
print(f"  Samples: {results['total_samples']}")
print()

if diff < 0.5:
    print(f"✅ PASS: Certified percentage {certified:.2f}% matches expected 72.36%")
    print(f"   Difference: {diff:.2f}% < 0.5%")
else:
    print(f"❌ FAIL: Certified percentage {certified:.2f}% does not match expected 72.36%")
    print(f"   Difference: {diff:.2f}% >= 0.5%")

print("=" * 60)
