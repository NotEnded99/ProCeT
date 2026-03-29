#!/usr/bin/env python3
import torch.onnx
"""
ICGAR Repair Implementation
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
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


class ICGARRepair:
    def __init__(
        self,
        model,
        dynamics_model,
        device=None,
        dtype=torch.float64,
        learning_rate=1e-3,
        regularization_lambda=1e-4,
        max_iterations=1000,
        verify_frequency=10,
        alpha_schedule='exponential_decay',
        alpha_params=None,
        rank_threshold=0.9,
        max_rank=50,
        tolerance=1e-6,
        verbose=True
    ):
        self.model = model
        self.dynamics_model = dynamics_model

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype

        self.learning_rate = learning_rate
        self.regularization_lambda = regularization_lambda
        self.max_iterations = max_iterations
        self.verify_frequency = verify_frequency
        self.tolerance = tolerance
        self.verbose = verbose

        self.rank_threshold = rank_threshold
        self.max_rank = max_rank

        if alpha_params is None:
            alpha_params = DECAY_SCHEDULES['exponential']
        self.alpha_scheduler = AlphaScheduler(alpha_schedule, **alpha_params)

        self.lbp_computer = LBPLowerBoundComputer(
            self.model, device=self.device, dtype=self.dtype
        )

        self.initial_params = self.lbp_computer._get_flattened_params()

        self.metrics = {
            'iterations': 0,
            'loss_history': [],
            'failed_regions_history': [],
            'invariance_violations': 0,
            'converged': False,
            'alpha_history': []
        }

    def repair(self, verified_regions, failed_regions, callback=None):
        print("=" * 60)
        print("ICGAR REPAIR ALGORITHM")
        print("=" * 60)
        print(f"Verified regions: {len(verified_regions)}")
        print(f"Failed regions: {len(failed_regions)}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Regularization Lambda: {self.regularization_lambda}")
        print(f"Alpha schedule: {self.alpha_scheduler.schedule_type}")
        print()

        print("Phase 1: Computing certificate manifold tangent space...")
        start_time = time.time()

        tangent_basis, projection_matrix, rank = compute_tangent_space(
            self.model,
            verified_regions,
            self.lbp_computer,
            rank_threshold=self.rank_threshold,
            max_rank=self.max_rank
        )

        self.projection_matrix = projection_matrix
        manifold_time = time.time() - start_time

        print(f"  Tangent space dimension: {tangent_basis.shape[1]}")
        print(f"  Normal space rank: {rank}")
        print(f"  Manifold computation time: {manifold_time:.2f}s")
        print()

        print("Phase 2: Starting projected gradient descent...")

        current_failed = failed_regions.copy()
        prev_failed_count = len(failed_regions)
        prev_loss = float('inf')
        t = 0
        convergence_count = 0

        while t < self.max_iterations and len(current_failed) > 0:

            loss = self._compute_repair_loss(current_failed)
            self.metrics['loss_history'].append(loss)

            if abs(loss - prev_loss) < self.tolerance and t > 10:
                convergence_count += 1
                if convergence_count >= 5:
                    print(f"  Converged at iteration {t}")
                    self.metrics['converged'] = True
                    break
            else:
                convergence_count = 0

            gradient = self._compute_repair_gradient(current_failed)

            g_parallel, g_perpendicular = project_orthogonal_components(
                gradient, self.projection_matrix
            )

            alpha_t = self.alpha_scheduler(
                t=t,
                failed_regions_count=len(current_failed),
                failed_regions_prev=prev_failed_count,
                loss=loss,
                loss_history=self.metrics['loss_history']
            )
            self.metrics['alpha_history'].append(alpha_t)

            g_combined = g_parallel + alpha_t * g_perpendicular

            self._update_parameters(g_combined)

            if t % self.verify_frequency == 0 or t == 0:
                verify_results = self._verify_progress()
                verified_count = verify_results['verified']
                current_failed = verify_results['failed']
                self.metrics['failed_regions_history'].append(len(current_failed))

                if self.verbose:
                    print(f"  Iter {t}: Loss={loss:.4f}, "
                          f"Verified={verified_count}, Failed={len(current_failed)}, "
                          f"alpha={alpha_t:.4f}")

                violations = self._check_invariance_violations(verified_regions)
                self.metrics['invariance_violations'] += violations

                if len(current_failed) == 0:
                    print(f"  All regions verified at iteration {t}!")
                    self.metrics['converged'] = True
                    break

            prev_failed_count = len(current_failed)
            prev_loss = loss

            if callback is not None:
                callback(t, loss, len(current_failed), alpha_t)

            t += 1

        print()
        print("Phase 3: Running final verification...")
        final_results = self._verify_progress()

        self.metrics['iterations'] = t
        self.metrics['final_loss'] = loss

        results = {
            'model': self.model,
            'initial_failed': len(failed_regions),
            'final_failed': len(final_results['failed']),
            'initial_verified': len(verified_regions),
            'final_verified': final_results['verified'],
            'iterations': t,
            'converged': self.metrics['converged'],
            'invariance_violations': self.metrics['invariance_violations'],
            'metrics': self.metrics
        }

        self._print_results(results)

        return results

    def _compute_repair_loss(self, failed_regions):
        return self.lbp_computer.compute_loss_on_regions(
            failed_regions,
            reg_lambda=self.regularization_lambda,
            initial_params=self.initial_params
        )

    def _compute_repair_gradient(self, failed_regions):
        self.model.zero_grad()

        for region in failed_regions:
            vertices = torch.tensor(
                region.vertices, device=self.device, dtype=self.dtype
            )

            outputs = self.model(vertices)
            min_idx = torch.argmin(outputs)
            min_vertex = vertices[min_idx:min_idx+1]

            if outputs[min_idx] < 0:
                min_vertex.requires_grad_(True)
                output = self.model(min_vertex)
                output.backward()

        if self.regularization_lambda > 0:
            current_params = self.lbp_computer._get_flattened_params()
            reg_gradient = 2 * self.regularization_lambda * \
                          (current_params - self.initial_params)

            idx = 0
            for param in self.model.parameters():
                numel = param.numel()
                if param.grad is not None:
                    param_grad = reg_gradient[idx:idx+numel].reshape(param.shape)
                    param.grad.data = param.grad.data + torch.tensor(
                        param_grad, device=self.device, dtype=self.dtype
                    )
                idx += numel

        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().cpu().flatten().numpy())
            else:
                gradients.append(np.zeros(param.numel()))

        return np.concatenate(gradients)

    def _update_parameters(self, gradient):
        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                param_grad = gradient[idx:idx+numel].reshape(param.shape)
                param.data = param.data - self.learning_rate * torch.tensor(
                    param_grad, device=self.device, dtype=self.dtype
                )
                idx += numel

    def _verify_progress(self):
        region_generator = create_region_generator("simplicial")
        initial_simplices = region_generator.create_mesh(
            self.dynamics_model
        ).get_regions(0)

        verified = []
        failed = []

        for region in initial_simplices:
            try:
                lower_bound, upper_bound = self.lbp_computer.compute_bounds(region)

                if lower_bound >= 0:
                    verified.append(region)
                else:
                    failed.append(region)
            except Exception:
                failed.append(region)

        return {'verified': verified, 'failed': failed}

    def _check_invariance_violations(self, original_verified_regions):
        violations = 0
        tolerance = 1e-6

        for region in original_verified_regions:
            try:
                lower_bound = self.lbp_computer.compute_lower_bound(region)

                if lower_bound < -tolerance:
                    violations += 1
            except Exception:
                violations += 1

        return violations

    def _print_results(self, results):
        print("=" * 60)
        print("ICGAR REPAIR RESULTS")
        print("=" * 60)
        print("Initial verified: " + str(results['initial_verified']))
        print("Initial failed: " + str(results['initial_failed']))
        print("Final verified: " + str(results['final_verified']))
        print("Final failed: " + str(results['final_failed']))
        print("Iterations: " + str(results['iterations']))
        print("Converged: " + str(results['converged']))
        print("Invariance violations: " + str(results['invariance_violations']))

        if results['initial_failed'] > 0:
            improvement = results['initial_failed'] - results['final_failed']
            improvement_pct = 100 * improvement / results['initial_failed']
            print(f"Regions repaired: {improvement} ({improvement_pct:.1f}%)")

        print("=" * 60)


def icgar_repair_pipeline(model_path, dynamics_model, output_dir, repair_config=None):
    print("=" * 70)
    print("ICGAR REPAIR PIPELINE")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    if repair_config is None:
        repair_config = {
            'learning_rate': 1e-3,
            'regularization_lambda': 1e-4,
            'max_iterations': 500,
            'verify_frequency': 10,
            'alpha_schedule': 'exponential_decay',
            'alpha_params': {'tau': 50, 'alpha_0': 1.0},
            'rank_threshold': 0.9,
            'max_rank': 50,
            'verbose': True
        }

    print(f"Loading model from {model_path}")
    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=getattr(dynamics_model, 'hidden_sizes', [64, 64, 64]),
        device=None
    )
    model.load_state_dict(
        torch.load(model_path, map_location='cpu', weights_only=False)
    )
    model.eval()

    print("\nRunning initial verification...")
    region_generator = create_region_generator("simplicial")
    initial_simplices = region_generator.create_mesh(
        dynamics_model
    ).get_regions(0)

    lbp_computer = LBPLowerBoundComputer(model)
    initial_verified = []
    initial_failed = []

    for region in initial_simplices:
        lower_bound = lbp_computer.compute_lower_bound(region)
        if lower_bound >= 0:
            initial_verified.append(region)
        else:
            initial_failed.append(region)

    print(f"Initial verification: {len(initial_verified)} verified, "
          f"{len(initial_failed)} failed")

    if len(initial_failed) == 0:
        print("All regions already verified - no repair needed!")
        return {
            'initial_verified': len(initial_verified),
            'initial_failed': 0,
            'final_verified': len(initial_verified),
            'final_failed': 0,
            'improvement': 0,
            'repaired': False
        }

    print("\nRunning ICGAR repair...")
    icgar = ICGARRepair(model, dynamics_model, **repair_config)
    repair_results = icgar.repair(initial_verified, initial_failed)

    system_name = getattr(dynamics_model, 'system_name', 'cbf')
    repaired_pth_path = os.path.join(
        output_dir, f"{system_name}_icgar_repaired.pth"
    )
    repaired_onnx_path = os.path.join(
        output_dir, f"{system_name}_icgar_repaired.onnx"
    )

    torch.save(model.state_dict(), repaired_pth_path)
    print(f"Model saved to {repaired_pth_path}")

    try:
        import torch.onnx
        torch_onnx_available = True
    except ImportError:
        torch_onnx_available = False
    dummy_input = torch.randn(
        1, dynamics_model.input_dim,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float64
    )

    try:
        torch.onnx.export(
            model,
            dummy_input,
            repaired_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"Model exported to ONNX: {repaired_onnx_path}")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        repaired_onnx_path = None

    results = {
        'initial_verified': len(initial_verified),
        'initial_failed': len(initial_failed),
        'final_verified': repair_results['final_verified'],
        'final_failed': repair_results['final_failed'],
        'iterations': repair_results['iterations'],
        'converged': repair_results['converged'],
        'invariance_violations': repair_results['invariance_violations'],
        'improvement': len(initial_failed) - repair_results['final_failed'],
        'improvement_pct': 100 * (len(initial_failed) - repair_results['final_failed']) /
                          max(len(initial_failed), 1),
        'repaired_model_path': repaired_pth_path,
        'repaired_onnx_path': repaired_onnx_path,
        'repair_config': repair_config,
        'metrics': repair_results['metrics'],
        'repaired': True
    }

    results_path = os.path.join(
        output_dir, f"{system_name}_icgar_results.json"
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ICGAR Repair for Neural Control Barrier Functions"
    )
    parser.add_argument(
        "--system-type", type=str, default="barr3",
        choices=["simple2d", "barr1", "barr2", "barr3", "barr4"],
        help="Type of dynamical system"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model .pth file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/data/icgar_repaired_models",
        help="Output directory for repaired models"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=500,
        help="Maximum repair iterations"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate for gradient descent"
    )
    parser.add_argument(
        "--alpha-schedule", type=str, default="exponential_decay",
        choices=["strict", "constant", "linear_ramp", "exponential_decay",
                 "inverse_decay", "feedback", "loss_based", "cosine"],
        help="Alpha schedule type"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress"
    )

    args = parser.parse_args()

    if args.system_type == "simple2d":
        from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem
        dynamics_model = Simple2DSystem()
    elif args.system_type == "barr1":
        dynamics_model = Barrier1System()
    elif args.system_type == "barr2":
        dynamics_model = Barrier2System()
    elif args.system_type == "barr3":
        dynamics_model = Barrier3System()
    elif args.system_type == "barr4":
        dynamics_model = Barrier4System()

    if args.model_path is None:
        args.model_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.pth"

    repair_config = {
        'learning_rate': args.learning_rate,
        'regularization_lambda': 1e-4,
        'max_iterations': args.max_iterations,
        'verify_frequency': 10,
        'alpha_schedule': args.alpha_schedule,
        'alpha_params': {'tau': 50, 'alpha_0': 1.0},
        'rank_threshold': 0.9,
        'max_rank': 50,
        'verbose': args.verbose
    }

    results = icgar_repair_pipeline(
        args.model_path,
        dynamics_model,
        args.output_dir,
        repair_config
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"System: {args.system_type}")
    print("Initial: " + str(results['initial_verified']) + " verified, " +
          str(results['initial_failed']) + " failed")
    print("Final: " + str(results['final_verified']) + " verified, " +
          str(results['final_failed']) + " failed")
    print(f"Improvement: {results['improvement']} regions "
          f"({results['improvement_pct']:.1f}%)")
    print("=" * 70)
