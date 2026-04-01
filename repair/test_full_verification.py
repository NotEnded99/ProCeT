#!/usr/bin/env python3
import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf


def verify_system(system_name, model_path, max_depth=13):
    print("\n" + "=" * 80)
    print(f"Verifying {system_name} from {model_path}")
    print("=" * 80)

    if "barr1" in system_name:
        dynamics_model = Barrier1System()
    elif "barr2" in system_name:
        dynamics_model = Barrier2System()
    elif "barr3" in system_name:
        dynamics_model = Barrier3System()
    else:
        raise ValueError(f"Unknown system type: {system_name}")

    results = verify_cbf(
        dynamics_model,
        barrier_model_path=model_path,
        executor_type="single",
        region_type="simplicial",
        visualize=False,
        use_wandb=False,
        use_gpu=False,
        batch_size=64,
        max_depth=max_depth
    )

    total_regions = len(results['regions'])
    verified_regions = sum(1 for r in results['regions'] if r.issat())
    failed_regions = sum(1 for r in results['regions'] if r.isunsat())
    maybe_regions = total_regions - verified_regions - failed_regions

    pass_rate = 100.0 * verified_regions / total_regions if total_regions > 0 else 0.0
    fail_rate = 100.0 * failed_regions / total_regions if total_regions > 0 else 0.0

    print("\nVerification Results:")
    print(f"  Total regions: {total_regions}")
    print(f"  Verified (SAT): {verified_regions} ({pass_rate:.2f}%)")
    print(f"  Failed (UNSAT): {failed_regions} ({fail_rate:.2f}%)")
    print(f"  Unknown (MAYBE): {maybe_regions}")

    return {
        "total": total_regions,
        "verified": verified_regions,
        "failed": failed_regions,
        "maybe": maybe_regions,
        "pass_rate": pass_rate
    }


def main():
    original_rates = {
        "barr1": 56.65,
        "barr2": 94.53,
        "barr3": 72.36
    }

    repaired_dir = "/data/icgar_repaired_models_v2"
    systems_to_test = ["barr1", "barr2", "barr3"]

    all_results = {}

    for system_type in systems_to_test:
        original_path = f"data/mine_models_relu/{system_type}_cbf.onnx"
        repaired_path = os.path.join(repaired_dir, f"{system_type}_icgar_repaired.onnx")

        print("\n" + "#" * 80)
        print(f"# SYSTEM: {system_type}")
        print("#" * 80)

        if os.path.exists(original_path):
            original_results = verify_system(f"{system_type} (original)", original_path)
        else:
            print(f"Original model not found at {original_path}")
            original_results = None

        if os.path.exists(repaired_path):
            repaired_results = verify_system(f"{system_type} (repaired)", repaired_path)
        else:
            print(f"Repaired model not found at {repaired_path}")
            repaired_results = None

        if original_results and repaired_results:
            improvement = repaired_results["verified"] - original_results["verified"]
            improvement_pct = repaired_results["pass_rate"] - original_results["pass_rate"]
            print("\n" + "-" * 80)
            print(f"IMPROVEMENT for {system_type}:")
            print(f"  Verified regions: +{improvement}")
            print(f"  Pass rate: {original_results['pass_rate']:.2f}% -> {repaired_results['pass_rate']:.2f}%")
            print(f"  Improvement: +{improvement_pct:.2f}%")
            print(f"  Original baseline: {original_rates[system_type]:.2f}%")
            print(f"  Net improvement vs baseline: +{repaired_results['pass_rate'] - original_rates[system_type]:.2f}%")
            print("-" * 80)

        all_results[system_type] = {
            "original": original_results,
            "repaired": repaired_results,
            "original_baseline": original_rates.get(system_type, None)
        }

    results_path = os.path.join(repaired_dir, "full_verification_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for system_type, results in all_results.items():
        print(f"\n{system_type}:")
        if results["original"]:
            print(f"  Original pass rate: {results['original']['pass_rate']:.2f}%")
        if results["repaired"]:
            print(f"  Repaired pass rate: {results['repaired']['pass_rate']:.2f}%")
        if results["original_baseline"]:
            print(f"  Original baseline: {results['original_baseline']:.2f}%")
        if results["original"] and results["repaired"]:
            improvement = results["repaired"]["pass_rate"] - results["original"]["pass_rate"]
            print(f"  Improvement: +{improvement:.2f}%")

    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    main()
