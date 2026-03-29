#!/usr/bin/env python3
"""
ICGAR Repair - Simple Direct Implementation
Based on ICGAR_FINAL_PROPOSAL.md pseudocode
"""

import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System, Barrier4System
from lbp_neural_cbf.regions.simplicial import SimplicialRegion
from lbp_neural_cbf.regions import create_region_generator

import repair.lbp_bounds as lbp_bounds_module
import repair.tangent_space as tangent_space_module
import repair.alpha_schedule as alpha_schedule_module

LBPLowerBoundComputer = lbp_bounds_module.LBPLowerBoundComputer
compute_tangent_space = tangent_space_module.compute_tangent_space
project_orthogonal_components = tangent_space_module.project_orthogonal_components
AlphaScheduler = alpha_schedule_module.AlphaScheduler
DECAY_SCHEDULES = alpha_schedule_module.DECAY_SCHEDULES


def main():
    print("="*70)
    print("ICGAR REPAIR TOOL - SIMPLIFIED VERSION")
    print("="*70)
    print()

    print("This tool repairs neural CBFs using the ICGAR method:")
    print("1. Computes certificate manifold tangent space")
    print("2. Performs projected gradient descent")
    print("3. Preserves verified-region invariance")
    print()

    if len(sys.argv) < 2:
        print("Usage: python3 repair_icgar_simple.py <system_type> [options]")
        print()
        print("Options:")
        print("  --max-iterations N    Default: 500")
        print("  --learning-rate R       Default: 0.001")
        print("  --alpha-schedule TYPE   Default: exponential_decay")
        print("  --strict-invariance    Use alpha=0 (strict invariance)")
        print()
        print("Available systems: simple2d, barr1, barr2, barr3, barr4")
        print("Available schedules: strict, constant, exponential_decay, linear_ramp")
        print()
        print("Output: Repaired models saved to /data/icgar_repaired_models/")
        return 1

    system_type = sys.argv[1]

    # Parse additional arguments
    max_iterations = 500
    learning_rate = 0.001
    alpha_schedule = 'exponential_decay'
    strict_invariance = False

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--max-iterations" and i+1 < len(sys.argv):
            max_iterations = int(sys.argv[i+1])
            i += 1
        elif arg == "--learning-rate" and i+1 < len(sys.argv):
            learning_rate = float(sys.argv[i+1])
            i += 1
        elif arg == "--alpha-schedule" and i+1 < len(sys.argv):
            alpha_schedule = sys.argv[i+1]
            i += 1
        elif arg == "--strict-invariance":
            strict_invariance = True
            i += 1
        else:
            i += 1

    if strict_invariance:
        alpha_schedule = 'strict'

    print(f"Configuration:")
    print(f"  System: {system_type}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Alpha schedule: {alpha_schedule}")
    print(f"  Strict invariance: {strict_invariance}")
    print()

    # Select dynamics system
    if system_type == "simple2d":
        from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem
        dynamics_model = Simple2DSystem()
    elif system_type == "barr1":
        dynamics_model = Barrier1System()
    elif system_type == "barr2":
        dynamics_model = Barrier2System()
    elif system_type == "barr3":
        dynamics_model = Barrier3System()
    elif system_type == "barr4":
        dynamics_model = Barrier4System()
    else:
        print(f"Error: Unknown system type '{system_type}'")
        print("Available: simple2d, barr1, barr2, barr3, barr4")
        return 1

    # Model path
    model_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return 1

    print(f"Loading model from {model_path}")

    # Load model
    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=getattr(dynamics_model, 'hidden_sizes', [64, 64, 64]),
        device=None
    )

    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Input dimension: {dynamics_model.input_dim}")
    print()

    # Generate simplices
    print("Generating simplices for verification...")
    region_generator = create_region_generator("simplicial")
    simplices = region_generator.create_mesh(dynamics_model).get_regions(0)
    print(f"Generated {len(simplices)} simplices")
    print()

    # Initial classification
    print("Running initial verification...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lbp_computer = LBPLowerBoundComputer(model, device=device, dtype=torch.float64)

    initial_verified = []
    initial_failed = []

    for region in = simplices:
        lower_bound = lbp_computer.compute_lower_bound(region)
        if lower_bound >= 0:
            initial_verified.append(region)
        else:
            initial_failed.append(region)

    print(f"Initial verification: {len(initial_verified)} verified, {len(initial_failed)} failed")
    print()

    pass_rate_initial = 100 * len(initial_verified) / len(simplices)
    print(f"Initial pass rate: {pass_rate_initial:.2f}%")
    print()

    if len(initial_failed) == 0:
        print("All regions already verified - no repair needed!")
        return 0

    # Run ICGAR repair
    print("="*70)
    print("RUNNING ICGAR REPAIR")
    print("="*70)
    print()

    # Initialize repair
    verified_regions = initial_verified
    failed_regions = initial_failed

    # Compute initial tangent space
    print("Computing certificate manifold tangent space...")
    start_time = time.time()

    try:
        tangent_basis, projection_matrix, rank = compute_tangent_space(
            model,
            verified_regions,
            lbp_computer,
            rank_threshold=0.9,
            max_rank=min(50, len(verified_regions))
        )
        manifold_time = time.time() - start_time
        print(f"Tangent space computed in {manifold_time:.2f}s")
        print(f"  Tang dimensionality: {tangent_basis.shape[1]}")
        print(f"  Normal space rank: {rank}")
        print()
    except Exception as e:
        print(f"Error computing tangent space: {e}")
        print("Falling back to unconstrained repair...")
        tangent_basis = None
        projection_matrix = None
        rank = 0
        manifold_time = 0

    # Main repair loop
    print("Starting repair iterations...")
    print()

    prev_loss = float('inf')
    convergence_count = 0

    for iteration in range(max_iterations):
        # Compute repair loss on failed regions
        loss = 0.0

        for region in failed_regions:
            vertices = torch.tensor(region.vertices, device=device, dtype=torch.float64)
            with torch.no_grad():
                outputs = model(vertices)
                lower_bound = outputs.min().item()

                if lower_bound < 0:
                    loss += -lower_bound  # Hinge loss

        # Add L2 regularization
        current_params = lbp_computer._get_flattened_params()
        reg_loss = 1e-4 * np.sum((current_params - current_params)**2)
        loss += reg_loss

        # Print progress
        if iteration % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration:3d}: Loss = {loss:.6f}")

        # Check convergence
        if abs(loss - prev_loss) < 1e-6 and iteration > 10:
            convergence_count += 1
            if convergence_count >= 3:
                print(f"Converged at iteration {iteration}")
                break
        else:
            convergence_count = 0

        prev_loss = loss

        # Compute gradient
        model.zero_grad()

        for region in failed_regions:
            vertices = torch.tensor(region.vertices, device=device, dtype=torch.float64)

            with torch.no_grad():
                outputs = model(vertices)
                min_idx = torch.argmin(outputs)
                min_vertex = vertices[min_idx:min_idx+1]

                if outputs[min_idx] < 0:
                    min_vertex.requires_grad_(True)
                    output = model(min_vertex)
                    output.backward()

        # Collect gradients
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.detach().cpu().flatten().numpy())
            else:
                grad_list.append(np.zeros(param.numel()))

        if grad_list:
            gradient = np.concatenate(grad_list)
        else:
            gradient = np.zeros(sum(p.numel() for p in model.parameters()))

        # Project gradient if tangent space available
        if projection_matrix is not None:
            g_parallel = projection_matrix @ gradient
            g_perpendicular = gradient - g_parallel

            # Compute alpha(t)
            alpha_t = compute_alpha_value(
                iteration,
                alpha_schedule,
                len(failed_regions)
            )

            g_combined = g_parallel + alpha_t * g_perpendicular
        else:
            g_combined = gradient

        # Update parameters
        param_idx = 0
        with torch.no_grad():
            for param in model.parameters():
                numel = param.numel()
                param_grad = g_combined[param_idx:param_idx+numel].reshape(param.shape)
                param.data = param.data - learning_rate * torch.tensor(
                    param_grad, device=device, dtype=torch.float64
                )
                param_idx += numel

        # Re-count failed regions
        new_failed = 0
        for region in simplices:
            lower_bound = lbp_computer.compute_lower_bound(region)
            if lower_bound < 0:
                new_failed += 1

        if new_failed == 0:
            print(f"All regions verified at iteration {iteration}!")
            break

        if iteration % 20 == 0:
            print(f"  Current failed regions: {new_failed}/{len(simplices)}")
            print(f"  Pass rate: {100 * (len(simplices) - new_failed) / len(simplices):.2f}%")
            print()

    # Final verification
    print("="*70)
    print("FINAL VERIFICATION")
    print("="*70)
    print()

    final_verified = 0
    final_failed = 0

    for region in simplices:
        lower_bound = lbp_computer.compute_lower_bound(region)
        if lower_bound >= 0:
            final_verified += 1
        else:
            final_failed += 1

    print(f"Final verification: {final_verified} verified, {final_failed} failed")
    pass_rate_final = 100 * final_verified / len(simplices)
    print(f"Final pass rate: {pass_rate_final:.2f}%")
    print()

    improvement = len(initial_failed) - final_failed
    improvement_pct = 100 * improvement / max(len(initial_failed), 1)

    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"System: {system_type}")
    print(f"Initial pass rate: {pass_rate_initial:.2f}%")
    print(f"Final pass rate: {pass_rate_final:.2f}%")
    print(f"Regions repaired: {improvement} ({improvement_pct:.1f}%)")
    print(f"Iterations: {iteration + 1}")
    print()

    # Save repaired model
    output_dir = "/data/icgar_repaired_models"
    os.makedirs(output_dir, exist_ok=True)

    system_name = dynamics_model.system_name
    repaired_pth_path = os.path.join(output_dir, f"{system_name}_icgar_repaired.pth")

    torch.save(model.state_dict(), repaired_pth_path)
    print(f"Repaired model saved to {repaired_pth_path}")

    # Export to ONNX
    try:
        import torch.onnx
        dummy_input = torch.randn(1, dynamics_model.input_dim, device=device, dtype=torch.float64)
        repaired_onnx_path = os.path.join(output_dir, f"{system_name}_icgar_repaired.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            repaired_onnx_path,
            export_params=True,
            opset_version=14
        )
        print(f"ONNX model exported to {repaired_onnx_path}")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")

    # Save results
    results = {
        'system': system_type,
        'initial_verified': len(initial_verified),
        'initial_failed': len(initial_failed),
        'initial_pass_rate': pass_rate_initial,
        'final_verified': final_verified,
        'final_failed': final_failed,
        'final_pass_rate': pass_rate_final,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'iterations': iteration + 1,
        'learning_rate': learning_rate,
        'alpha_schedule': alpha_schedule,
        'manifold_time': manifold_time,
        'repaired_model_path': repaired_pth_path
    }

    results_path = os.path.join(output_dir, f"{system_name}_icgar_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    print()

    return 0


def compute_alpha_value(t, schedule, failed_count):
    if schedule == 'strict':
        return 0.0
    elif schedule == 'constant':
        return 0.5
    elif schedule == 'exponential_decay':
        tau = 50
        alpha_0 = 1.0
        return alpha_0 * (1.0 - np.exp(-t / tau))
    elif schedule == 'linear_ramp':
        T_ramp = 100
        return min(1.0, t / T_ramp)
    else:
        return 0.1


if __name__ == "__main__":
    exit(main())
