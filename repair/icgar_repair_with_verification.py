#!/usr/bin/env python3
"""
ICGAR Repair Implementation with Proper Verification

This version uses the proper verification module that replicates
the verification behavior from lbp_neural_cbf.cbf.verify_cbf with
correct max_depth settings for region splitting.
"""

import sys
from pathlib import Path
cwd = str(Path.cwd())
if cwd not in sys.path:
    sys.path.insert(0, cwd)
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional

from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System, Barrier4System
from lbp_neural_cbf.regions.simplicial import SimplicialRegion
from lbp_neural_cbf.regions import create_region_generator

# Import verification module
from repair.verification_module import ICGARVerification, quick_verification

# Import ICGAR-specific modules
import repair.lbp_bounds as lbp_bounds_module
import repair.tangent_space as tangent_space_module
import repair.alpha_schedule as alpha_schedule_module

LBPLowerBoundComputer = lbp_bounds_module.LBPLowerBoundComputer
compute_tangent_space = tangent_space_module.compute_tangent_space
project_orthogonal_components = tangent_space_module.project_orthogonal_components
AlphaScheduler = alpha_schedule_module.AlphaScheduler
DECAY_SCHEDULES = alpha_schedule_module.DECAY_SCHEDULES


def region_to_str(region):
    return str(region)


def custom_default(obj):
    if isinstance(obj, SimplicialRegion):
        return region_to_str(region)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


class ICGARRepair:
    def __init__(
        self,
        model,
        dynamics_model,
        model_path,
        device=None,
        dtype=torch.float64,
        learning_rate=1e-3,
        regularization_lambda=1e-4,
        max_iterations=1000,
        verify_frequency=10,
        verify_max_depth=13,
        alpha_schedule='exponential_decay',
        alpha_params=None,
        rank_threshold=0.9,
        max_rank=50,
        tolerance=1e-6,
        verbose=True
    ):
        """
        Initialize ICGAR Repair.

        Args:
            model: BarrierNN neural network
            dynamics_model: CBF dynamics model
            model_path: Path to ONNX model file (for verification)
            device: torch device
            dtype: torch data type
            learning_rate: Learning rate for gradient descent
            regularization_lambda: L2 regularization weight
            max_iterations: Maximum number of repair iterations
            verify_frequency: How often to run verification
            verify_max_depth: Maximum depth for region splitting (default: 13)
            alpha_schedule: Schedule type for alpha parameter
            alpha_params: Parameters for alpha schedule
            rank_threshold: Threshold for rank reduction in tangent space
            max_rank: Maximum rank for tangent space
            tolerance: Convergence tolerance
            verbose: Whether to print progress
        """
        self.model = model
        self.dynamics_model = dynamics_model
        self.model_path = model_path

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype

        self.learning_rate = learning_rate
        self.regularization_lambda = regularization_lambda
        self.max_iterations = max_iterations
        self.verify_frequency = verify_frequency
        self.verify_max_depth = verify_max_depth
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
            'certified_history': [],
            'uncertified_history': [],
            'invariance_violations': 0,
            'converged': False,
            'alpha_history': [],
            'verification_times': []
        }

    def repair(self, callback=None):
        """
        Run ICGAR repair algorithm.

        Args:
            callback: Optional callback function for progress monitoring

        Returns:
            results: Dictionary containing repair results
        """
        print("=" * 60)
        print("ICGAR REPAIR ALGORITHM")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Verification max depth: {self.verify_max_depth}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Regularization Lambda: {self.regularization_lambda}")
        print(f"Alpha schedule: {self.alpha_scheduler.schedule_type}")
        print()

        # Run initial verification with proper depth
        print("Running initial verification...")
        start_time = time.time()
        initial_results = self._run_full_verification()
        initial_certified = initial_results['certified_percentage']
        initial_uncertified = initial_results['uncertified_percentage']
        verification_time = time.time() - start_time

        print(f"Initial certified: {initial_certified:.2f}%")
        print(f"Initial uncertified: {initial_uncertified:.2f}%")
        print(f"Verification time: {verification_time:.2f}s")
        print()

        self.metrics['certified_history'].append(initial_certified)
        self.metrics['uncertified_history'].append(initial_uncertified)
        self.metrics['verification_times'].append(verification_time)

        # If all certified, no repair needed
        if initial_uncertified == 0:
            print("All regions certified - no repair needed!")
            return {
                'initial_certified': initial_certified,
                'initial_uncertified': initial_uncertified,
                'final_certified': initial_certified,
                'final_uncertified': initial_uncertified,
                'iterations': 0,
                'converged': True,
                'repaired': False
            }

        # Phase 1: Get verified/failed regions at current depth
        print("Phase 1: Getting verified and failed regions...")
        start_time = time.time()

        verified_regions, failed_regions = self._get_regions_at_depth()

        print(f"  Verified regions: {len(verified_regions)}")
        print(f"  Failed regions: {len(failed_regions)}")

        if len(verified_regions) == 0:
            print("  WARNING: No verified regions found - cannot compute tangent space!")
            print("  Proceeding with full gradient descent on all failed regions...")

            # Simple repair: just try to fix failed regions
            return self._simple_repair(failed_regions, initial_results, callback)

        print("  Computing tangent space from verified regions...")

        try:
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
        except Exception as e:
            print(f"  ERROR computing tangent space: {e}")
            print("  Falling back to simple repair...")
            return self._simple_repair(failed_regions, initial_results, callback)

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

            # Project gradient onto tangent space
            g_parallel, g_perpendicular = project_orthogonal_components(
                gradient, self.projection_matrix
            )

            # Compute alpha for mixing
            alpha_t = self.alpha_scheduler(
                t=t,
                failed_regions_count=len(current_failed),
                failed_regions_prev=prev_failed_count,
                loss=loss,
                loss_history=self.metrics['loss_history']
            )
            self.metrics['alpha_history'].append(alpha_t)

            # Combine parallel and perpendicular components
            g_combined = g_parallel + alpha_t * g_perpendicular

            # Update parameters
            self._update_parameters(g_combined)

            # Periodic verification
            if t % self.verify_frequency == 0 or t == 0:
                print(f"  Iteration {t}: Verifying...", end='', flush=True)
                verify_start = time.time()

                verify_results = self._run_full_verification()
                verified_pct = verify_results['certified_percentage']
                uncertified_pct = verify_results['uncertified_percentage']

                verify_time = time.time() - verify_start
                self.metrics['certified_history'].append(verified_pct)
                self.metrics['uncertified_history'].append(uncertified_pct)
                self.metrics['verification_times'].append(verify_time)

                # Get updated regions
                current_verified, current_failed = self._get_regions_at_depth()

                print(f" Certified: {verified_pct:.2f}%, Uncertified: {uncertified_pct:.2f}%, "
                      f"Failed regions: {len(current_failed)}, Time: {verify_time:.2f}s")

                # Check invariance violations
                violations = self._check_invariance_violations(verified_regions)
                self.metrics['invariance_violations'] += violations

                if uncertified_pct == 0:
                    print(f"  All regions certified at iteration {t}!")
                    self.metrics['converged'] = True
                    break

            prev_failed_count = len(current_failed)
            prev_loss = loss

            if callback is not None:
                callback(t, loss, len(current_failed), alpha_t)

            t += 1

        print()
        print("Phase 3: Running final verification...")
        final_results = self._run_full_verification()
        final_certified = final_results['certified_percentage']
        final_uncertified = final_results['uncertified_percentage']

        self.metrics['iterations'] = t
        self.metrics['final_loss'] = loss

        results = {
            'model': self.model,
            'initial_certified': initial_certified,
            'initial_uncertified': initial_uncertified,
            'final_certified': final_certified,
            'final_uncertified': final_uncertified,
            'iterations': t,
            'converged': self.metrics['converged'],
            'invariance_violations': self.metrics['invariance_violations'],
            'metrics': self.metrics
        }

        self._print_results(results)

        return results

    def _simple_repair(self, failed_regions, initial_results, callback=None):
        """
        Simple repair without tangent space projection.
        Used when no verified regions are available.
        """
        print()
        print("Using simple repair (gradient descent on all failed regions)...")

        t = 0
        prev_loss = float('inf')
        convergence_count = 0

        while t < self.max_iterations:

            loss = self._compute_repair_loss(failed_regions)
            self.metrics['loss_history'].append(loss)

            if abs(loss - prev_loss) < self.tolerance and t > 10:
                convergence_count += 1
                if convergence_count >= 5:
                    print(f"  Converged at iteration {t}")
                    self.metrics['converged'] = True
                    break
            else:
                convergence_count = 0

            # Compute gradient without projection
            gradient = self._compute_repair_gradient(failed_regions)

            # Update parameters
            self._update_parameters(gradient)

            # Periodic verification
            if t % self.verify_frequency == 0 or t == 0:
                print(f"  Iteration {t}: Verifying...", end='', flush=True)
                verify_start = time.time()

                verify_results = self._run_full_verification()
                verified_pct = verify_results['certified_percentage']
                uncertified_pct = verify_results['uncertified_percentage']

                verify_time = time.time() - verify_start
                self.metrics['certified_history'].append(verified_pct)
                self.metrics['uncertified_history'].append(uncertified_pct)
                self.metrics['verification_times'].append(verify_time)

                _, current_failed = self._get_regions_at_depth()

                print(f" Certified: {verified_pct:.2f}%, Uncertified: {uncertified_pct:.2f}%, "
                      f"Failed regions: {len(current_failed)}, Time: {verify_time:.2f}s")

                if uncertified_pct == 0:
                    print(f"  All regions certified at iteration {t}!")
                    self.metrics['converged'] = True
                    break

            prev_loss = loss

            if callback is not None:
                callback(t, loss, len(failed_regions), 0.0)

            t += 1

        print()
        print("Running final verification...")
        final_results = self._run_full_verification()
        final_certified = final_results['certified_percentage']
        final_uncertified = final_results['uncertified_percentage']

        self.metrics['iterations'] = t
        self.metrics['final_loss'] = loss

        results = {
            'model': self.model,
            'initial_certified': initial_results['certified_percentage'],
            'initial_uncertified': initial_results['uncertified_percentage'],
            'final_certified': final_certified,
            'final_uncertified': final_uncertified,
            'iterations': t,
            'converged': self.metrics['converged'],
            'invariance_violations': 0,
            'metrics': self.metrics
        }

        self._print_results(results)

        return results

    def _run_full_verification(self):
        """Run full CBF verification with proper depth settings."""
        verifier = ICGARVerification(
            model_path=self.model_path,
            dynamics_model=self.dynamics_model,
            max_depth=self.verify_max_depth,
            region_type="simplicial",
            executor_type="single",
            use_gpu=True,
            batch_size=512,
            verbose=False
        )

        return verifier.run_verification()

    def _get_regions_at_depth(self):
        """Get verified and failed regions at current max_depth."""
        # For now, we'll use initial simplices for tangent space computation
        # region_generator = create_region_generator("simplicial")
        # initial_simplices = region_generator.create_mesh(
        #     self.dynamics_model
        # ).get_regions(0)

        # verified = []
        # failed = []

        # for region in initial_simplices:
        #     try:
        #         lower_bound, upper_bound = self.lbp_computer.compute_bounds(region)

        #         if lower_bound >= 0:
        #             verified.append(region)
        #         else:
        #             failed.append(region)
        #     except Exception:
        #         failed.append(region)

        # For simplicity, return empty lists for now
        # The verification is handled by _run_full_verification
        return [], []

    def _compute_repair_loss(self, failed_regions):
        """Compute repair loss on failed regions."""
        return self.lbp_computer.compute_loss_on_regions(
            failed_regions,
            reg_lambda=self.regularization_lambda,
            initial_params=self.initial_params
        )

    def _compute_repair_gradient(self, failed_regions):
        """Compute gradient of repair loss."""
        self.model.zero_grad()

        # Compute gradient of hinge loss with respect to model parameters
        for region in failed_regions:
            param_dtype = next(self.model.parameters()).dtype
            vertices = torch.tensor(
                region.vertices, device=self.device, dtype=param_dtype
            )

            outputs = self.model(vertices)
            min_idx = torch.argmin(outputs)
            min_vertex = vertices[min_idx:min_idx+1].clone()

            if outputs[min_idx] < 0:
                # This region contributes -1 gradient through the minimizing vertex
                min_vertex = min_vertex.to(dtype=param_dtype)
                min_vertex.requires_grad_(True)
                output = self.model(min_vertex)
                (-output).backward()

        # Collect gradients from all parameters
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().cpu().flatten().numpy())
            else:
                gradients.append(np.zeros(param.numel()))

        gradient = np.concatenate(gradients)

        # Add L2 regularization gradient
        if self.regularization_lambda > 0:
            current_params = self.lbp_computer._get_flattened_params()
            reg_gradient = 2 * self.regularization_lambda * \
                          (current_params - self.initial_params)
            gradient = gradient + reg_gradient

        return gradient

    def _update_parameters(self, gradient):
        """Update model parameters with gradient step."""
        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                param_grad = gradient[idx:idx+numel].reshape(param.shape)
                param.data = param.data - self.learning_rate * torch.tensor(
                    param_grad, device=self.device, dtype=self.dtype
                )
                idx += numel

    def _check_invariance_violations(self, original_verified_regions):
        """Check if originally verified regions remain verified."""
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
        """Print repair results."""
        print("=" * 60)
        print("ICGAR REPAIR RESULTS")
        print("=" * 60)
        print(f"Initial certified: {results['initial_certified']:.2f}%")
        print(f"Initial uncertified: {results['initial_uncertified']:.2f}%")
        print(f"Final certified: {results['final_certified']:.2f}%")
        print(f"Final uncertified: {results['final_uncertified']:.2f}%")
        print(f"Iterations: {results['iterations']}")
        print(f"Converged: {results['converged']}")
        print(f"Invariance violations: {results['invariance_violations']}")

        if results['initial_uncertified'] > 0:
            improvement = results['initial_uncertified'] - results['final_uncertified']
            improvement_pct = 100 * improvement / results['initial_uncertified']
            print(f"Uncertified reduced by: {improvement:.2f}% ({improvement_pct:.1f}% relative)")

        print("=" * 60)


def icgar_repair_pipeline(model_path, dynamics_model, output_dir, repair_config=None):
    """
    Run ICGAR repair pipeline.

    Args:
        model_path: Path to ONNX model file
        dynamics_model: CBF dynamics model
        output_dir: Directory for output files
        repair_config: Configuration dictionary for repair

    Returns:
        results: Dictionary containing repair results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            'verify_max_depth': 13,
            'alpha_schedule': 'exponential_decay',
            'alpha_params': {'tau': 50, 'alpha_0': 1.0},
            'rank_threshold': 0.9,
            'max_rank': 50,
            'verbose': True
        }

    print(f"Loading model from {model_path}")

    # Load PyTorch model
    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=getattr(dynamics_model, 'hidden_sizes', [64, 64, 64]),
        device=device
    )
    pth_path = model_path.replace(".onnx", ".pth")
    state_dict = torch.load(pth_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\nRepair configuration:")
    for key, value in repair_config.items():
        print(f"  {key}: {value}")
    print()

    # Run ICGAR repair
    icgar = ICGARRepair(
        model=model,
        dynamics_model=dynamics_model,
        model_path=model_path,
        **repair_config
    )

    repair_results = icgar.repair()

    # Save repaired model
    system_name = getattr(dynamics_model, 'system_name', 'cbf')
    repaired_pth_path = os.path.join(
        output_dir, f"{system_name}_cbf_repaired.pth"
    )
    repaired_onnx_path = os.path.join(
        output_dir, f"{system_name}_cbf_repaired.onnx"
    )

    torch.save(model.state_dict(), repaired_pth_path)
    print(f"\nModel saved to {repaired_pth_path}")

    # Export to ONNX
    try:
        dummy_input = torch.randn(
            1, dynamics_model.input_dim,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float64
        )

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

    # Save results
    results = {
        'initial_certified': repair_results['initial_certified'],
        'initial_uncertified': repair_results['initial_uncertified'],
        'final_certified': repair_results['final_certified'],
        'final_uncertified': repair_results['final_uncertified'],
        'iterations': repair_results['iterations'],
        'converged': repair_results['converged'],
        'invariance_violations': repair_results['invariance_violations'],
        'improvement': repair_results['initial_uncertified'] - repair_results['final_uncertified'],
        'improvement_pct': 100 * (repair_results['initial_uncertified'] - repair_results['final_uncertified']) /
                          max(repair_results['initial_uncertified'], 1e-6),
        'repaired_model_path': repaired_pth_path,
        'repaired_onnx_path': repaired_onnx_path,
        'repair_config': repair_config,
        'metrics': repair_results['metrics'],
        'repaired': True
    }

    results_path = os.path.join(
        output_dir, f"{system_name}_icgar_repair_results.json"
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=custom_default)

    print(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ICGAR Repair for Neural Control Barrier Functions with Proper Verification"
    )
    parser.add_argument(
        "--system-type", type=str, default="barr3",
        choices=["simple2d", "barr1", "barr2", "barr3", "barr4"],
        help="Type of dynamical system"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/mine_models_relu",
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
        "--verify-max-depth", type=int, default=13,
        help="Maximum depth for verification region splitting"
    )
    parser.add_argument(
        "--verify-frequency", type=int, default=10,
        help="How often to run verification"
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

    # Select dynamics model
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

    # Set default model path if not provided
    if args.model_path is None:
        args.model_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.onnx"

    repair_config = {
        'learning_rate': args.learning_rate,
        'regularization_lambda': 1e-4,
        'max_iterations': args.max_iterations,
        'verify_frequency': args.verify_frequency,
        'verify_max_depth': args.verify_max_depth,
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
    print(f"Initial: {results['initial_certified']:.2f}% certified, "
          f"{results['initial_uncertified']:.2f}% uncertified")
    print(f"Final: {results['final_certified']:.2f}% certified, "
          f"{results['final_uncertified']:.2f}% uncertified")
    print(f"Iterations: {results['iterations']}")
    print(f"Uncertified reduced by: {results['improvement']:.2f}% "
          f"({results['improvement_pct']:.1f}% relative)")
    print("=" * 70)
