#!/usr/bin/env python3
"""
Test verification baselines for all systems
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System, Barrier4System
from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem
from repair.verification_module import quick_verification


def test_system(system_name, dynamics_model, expected_certified, model_path=None):
    """Test a single system."""
    if model_path is None:
        model_path = f"data/mine_models_relu/{system_name}_cbf.onnx"

    print(f"\n{'='*60}")
    print(f"Testing {system_name} system")
    print(f"{'='*60}")

    try:
        cert_pct, uncert_pct = quick_verification(
            model_path=model_path,
            dynamics_model=dynamics_model,
            max_depth=13,
            verbose=True
        )

        print(f"Expected certified: {expected_certified:.2f}%")
        print(f"Actual certified: {cert_pct:.2f}%")

        diff = abs(cert_pct - expected_certified)
        if diff < 0.5:
            print(f"✅ PASS: Difference {diff:.2f}% < 0.5%")
        else:
            print(f"❌ FAIL: Difference {diff:.2f}% >= 0.5%")

        return diff < 0.5
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    results = {}

    # Test simple2d (expected: 100.00%)
    results['simple2d'] = test_system(
        'simple2d',
        Simple2DSystem(),
        100.00
    )

    # Test barr1 (expected: 56.65%)
    results['barr1'] = test_system(
        'barr1',
        Barrier1System(),
        56.65
    )

    # Test barr2 (expected: 94.53%)
    results['barr2'] = test_system(
        'barr2',
        Barrier2System(),
        94.53
    )

    # Test barr3 (expected: 72.36%)
    results['barr3'] = test_system(
        'barr3',
        Barrier3System(),
        72.36
    )

    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for system_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{system_name}: {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total - passed} test(s) failed")
