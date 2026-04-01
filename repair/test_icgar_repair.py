#!/usr/bin/env python3
"""
Test script for ICGAR repair functionality
Tests the repair on different barrier systems and measures improvement.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf

import repair.icgar_repair_fixed_v2 as icgar_rearing


def verify_model(dynamics_model, model_path, max_depth=13):
    """Verify a model and return verification results."""
    results = verify_cbf(
        dynamics_model,
        barrier_model_path=model_path,
        executor_type="single",
        region_type="simplicial",
        visualize=False,
        use_wandb=False,
        batch_size=512,
        max_depth=max_depth
    )

    # Calculate pass rate
    total_regions = len(results['regions'])
    verified_regions = sum(1 for r in results['regions'] if r.issat())
    failed_regions = sum(1 for r in results['regions'] if r.isunsat())
    maybe_regions = total_regions - verified_regions - failed_regions

    pass_rate = 100.0 * verified_regions / total_regions if total_regions > 0 else 0.0

    return {
        'total_regions': total_regions,
        'verified': verified_regions,
        'failed': failed_regions,
        'maybe': maybe_regions,
        'pass_rate': pass_rate
    }


def run_repair_test(system_type, output_dir):
    """Run repair test for a specific system."""
    print("\n" + "=" * 80)
    print(f"TESTING SYSTEM: {system_type}")
    print("=" * 80)

    # Load dynamics model
    if system_type == "barr1":
        dynamics_model = Barrier1System()
    elif system_type == "barr2":
        dynamics_model = Barrier2System()
    elif system_type == "barr3":
        dynamics_model = Barrier3System()
    else:
        raise ValueError(f"Unknown system type: {system_type}")

    # Path to original model
    original_model_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.pth"

    if not os.path.exists(original_model_path):
        print(f"ERROR: Original model not found at {original_model_path}")
        return None

    # Verify original model first
    print(f"\n1. Verifying original model at {original_model_path}...")
    # Create ONNX path for verification
    original_onnx_path = original_model_path.replace('.pth', '.onnx')
    if not os.path.exists(original_onnx_path):
        print(f"Warning: ONNX version not found, verification may fail")
        original_onnx_path = None

    # Note: verify_cbf expects ONNX, but we can try to convert
    if original_onnx_path:
        try:
            original_results = verify_model(dynamics_model, original_onnx_path)
            print(f"Original model results:")
            print(f"  Verified: {original_results['verified']}/{original_results['total_regions']} ({original_results['pass_rate']:.2f}%)")
            print(f"  Failed: {original_results['failed']}")
            print(f"  Maybe: {original_results['maybe']}")
        except Exception as e:
            print(f"Original verification failed: {e}")
            original_results = None
    else:
        original_results = None

    # Run ICGAR repair
    print(f"\n2. Running ICGAR repair...")
    repair_config = {
        'learning_rate': 1e-3,
        'regularization_lambda': 1e-4,
        'max_iterations': 200,
        'verify_frequency': 10,
        'alpha_schedule': 'exponential_decay',
        'alpha_params': {'tau': 50, 'alpha_0': 1.0},
        'rank_threshold': 0.9,
        'max_rank': 50,
        'verbose': True
    }

    repair_results = icgar_repair.icgar_repair_pipeline(
        original_model_path,
        dynamics_model,
        output_dir,
        repair_config
    )

    # Verify repaired model
    print(f"\n3. Repaired model verification...")
    repaired_onnx_path = repair_results.get('repaired_onnx_path')

    if repaired_onnx_path and os.path.exists(repaired_onnx_path):
        try:
            repaired_results = verify_model(dynamics_model, repaired_onnx_path)
            print(f"Repaired model results:")
            print(f"  Verified: {repaired_results['verified']}/{repaired_results['total_regions']} ({repaired_results['pass_rate']:.2f}%)")
            print(f"  Failed: {repaired_results['failed']}")
            print(f"  Maybe: {repaired_results['maybe']}")

            # Calculate improvement
            if original_results:
                improvement = repaired_results['verified'] - original_results['verified']
                improvement_pct = repaired_results['pass_rate'] - original_results['pass_rate']
                print(f"\nImprovement: +{improvement} verified regions ({improvement_pct:.2f}%)")
            else:
                improvement = None
                improvement_pct = None
        except Exception as e:
            print(f"Repaired verification failed: {e}")
            repaired_results = None
            improvement = None
            improvement_pct = None
    else:
        print("Repaired ONNX not available")
        repaired_results = None
        improvement = None
        improvement_pct = None

    # Compile final results
    test_results = {
        'system_type': system_type,
        'original_results': original_results,
        'repair_results': repair_results,
        'repaired_results': repaired_results,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

    return test_results


def main():
    """Main test function."""
    output_dir = "/data/icgar_repaired_models_v2"
    os.makedirs(output_dir, exist_ok=True)

    systems_to_test = ["barr3"]  # Start with barr3

    all_results = {}

    for system_type in systems_to_test:
        test_results = run_repair_test(system_type, output_dir)
        all_results[system_type] = test_results

    # Save all results
    results_path = os.path.join(output_dir, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for system_type, results in all_results.items():
        print(f"\n{system_type}:")
        if results['original_results']:
            print(f"  Original: {results['original_results']['pass_rate']:.2f}% verified")
        if results['repaired_results']:
            print(f"  Repaired: {results['repaired_results']['pass_rate']:.2f}% verified")
        if results['improvement'] is not None:
            print(f"  Improvement: +{results['improvement_pct']:.2f}%")

    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    main()
