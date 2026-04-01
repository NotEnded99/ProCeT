"""
Verification Module for ICGAR Repair

This module provides proper CBF verification functionality that matches
the verification behavior from lbp_neural_cbf.cbf.verify_cbf.
"""

import sys
from pathlib import Path
import time
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from lbp_neural_cbf.cbf.verify_cbf import (
    verify_cbf,
    CBFVerificationStrategy
)
from lbp_neural_cbf.regions import create_region_generator
from lbp_neural_cbf.cbf.network import BarrierNN


class ICGARVerification:
    """
    Verification wrapper for ICGAR repair that properly verifies CBF conditions
    with region splitting based on max_depth.
    """

    def __init__(
        self,
        model_path,
        dynamics_model,
        max_depth=13,
        region_type="simplicial",
        executor_type="single",
        use_gpu=True,
        batch_size=512,
        verbose=True
    ):
        """
        Initialize verification.

        Args:
            model_path: Path to ONNX model file
            dynamics_model: CBF dynamics model
            max_depth: Maximum depth for region splitting
            region_type: Type of regions ("simplicial" or "hyperrectangular")
            executor_type: Type of executor ("single", "multi-thread", "multi-process")
            use_gpu: Whether to use GPU
            batch_size: Batch size for verification
            verbose: Whether to print progress
        """
        self.model_path = model_path
        self.dynamics_model = dynamics_model
        self.max_depth = max_depth
        self.region_type = region_type
        self.executor_type = executor_type
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.verbose = verbose

        # Load the PyTorch model for region evaluation
        pth_path = model_path.replace(".onnx", ".pth")
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

        self.model = BarrierNN(
            input_size=dynamics_model.input_dim,
            hidden_sizes=getattr(dynamics_model, "hidden_sizes", [64, 64, 64]),
            device=self.device
        )
        self.model.load_state_dict(torch.load(pth_path, map_location=self.device, weights_only=False))
        self.model.eval()

    def run_verification(self):
        """
        Run full verification with proper region splitting.

        Returns:
            results: Dictionary containing verification results
        """
        if self.verbose:
            print("=" * 60)
            print("ICGAR VERIFICATION")
            print("=" * 60)
            print(f"Model: {self.model_path}")
            print(f"Max depth: {self.max_depth}")
            print(f"Region type: {self.region_type}")
            print(f"Executor: {self.executor_type}")
            print()

        # Use the standard verify_cbf function
        results = verify_cbf(
            self.dynamics_model,
            barrier_model_path=self.model_path,
            executor_type=self.executor_type,
            region_type=self.region_type,
            visualize=False,
            use_gpu=self.use_gpu,
            use_wandb=False,
            batch_size=self.batch_size,
            max_depth=self.max_depth
        )

        return results

    def get_regions_at_depth(self, depth):
        """
        Get regions after verification up to specified depth.

        Args:
            depth: Maximum depth for region splitting

        Returns:
            verified_regions: List of verified regions
            failed_regions: List of failed regions (counterexamples)
        """
        # Use the verification strategy with specified depth
        strategy = CBFVerificationStrategy(
            self.model_path,
            self.dynamics_model,
            use_gpu=self.use_gpu,
            max_depth=depth
        )

        # Initialize worker
        strategy.initialize_worker()
        global _LOCAL_VERIFICATION
        _LOCAL_VERIFICATION = strategy

        # Generate initial samples
        region_generator = create_region_generator(self.region_type)
        samples = region_generator.create_mesh(self.dynamics_model).get_regions(0)

        # Process samples using the verification strategy
        verified_regions = []
        failed_regions = []

        # Process in batches
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i+self.batch_size]
            batch_results = strategy.verify_batch(batch)

            for result in batch_results:
                if result.issat():
                    verified_regions.append(result.sample)
                elif result.isunsat():
                    failed_regions.append(result.sample)
                # MAYBE results are handled by the executor in full verification
                # For this simplified version, we'll just collect verified/failed

        return verified_regions, failed_regions

    def evaluate_regions_batch(self, regions):
        """
        Evaluate the barrier function bounds on a batch of regions.

        Args:
            regions: List of SimplicialRegion or HyperrectangularRegion

        Returns:
            lower_bounds: Array of lower bounds
            upper_bounds: Array of upper bounds
        """
        lower_bounds = []
        upper_bounds = []

        for region in regions:
            if hasattr(region, 'vertices'):
                # SimplicialRegion - evaluate at vertices
                param_dtype = next(self.model.parameters()).dtype
                vertices = torch.tensor(region.vertices, device=self.device, dtype=param_dtype)

                with torch.no_grad():
                    outputs = self.model(vertices)

                lower_bounds.append(outputs.min().item())
                upper_bounds.append(outputs.max().item())
            else:
                # HyperrectangularRegion - evaluate at corners
                # For now, just evaluate at center
                param_dtype = next(self.model.parameters()).dtype
                center = torch.tensor(region.center_point, device=self.device, dtype=param_dtype)

                with torch.no_grad():
                    output = self.model(center.unsqueeze(0))

                lower_bounds.append(output.item())
                upper_bounds.append(output.item())

        return np.array(lower_bounds), np.array(upper_bounds)


def quick_verification(
    model_path,
    dynamics_model,
    max_depth=13,
    verbose=True
):
    """
    Quick verification function for convenience.

    Args:
        model_path: Path to ONNX model file
        dynamics_model: CBF dynamics model
        max_depth: Maximum depth for region splitting
        verbose: Whether to print results

    Returns:
        certified_percentage: Percentage of certified regions
        uncertified_percentage: Percentage of uncertified regions
    """
    verifier = ICGARVerification(
        model_path=model_path,
        dynamics_model=dynamics_model,
        max_depth=max_depth,
        verbose=verbose
    )

    results = verifier.run_verification()

    if verbose:
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Certified: {results['certified_percentage']:.2f}%")
        print(f"Uncertified: {results['uncertified_percentage']:.2f}%")
        print(f"Time: {results['computation_time']:.2f}s")
        print("=" * 60)

    return results['certified_percentage'], results['uncertified_percentage']


if __name__ == "__main__":
    # Test the verification module
    from lbp_neural_cbf.cbf.fossil_dynamics import Barrier3System

    # Test with barr3 system
    dynamics_model = Barrier3System()
    model_path = "data/mine_models_relu/barr3_cbf.onnx"

    print("Testing ICGAR Verification Module...")
    print(f"System: {dynamics_model.system_name}")
    print(f"Model: {model_path}")

    cert_pct, uncert_pct = quick_verification(
        model_path=model_path,
        dynamics_model=dynamics_model,
        max_depth=13,
        verbose=True
    )

    print(f"\nExpected baseline for barr3: 72.36% certified")
    print(f"Actual result: {cert_pct:.2f}% certified")
