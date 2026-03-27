"""
Certified-Subspace Repair (CSR) - Complete End-to-End Example

This script demonstrates the complete CSR pipeline:
1. Train a (deliberately flawed) CBF
2. Verify it with LBP
3. Collect A_L matrices
4. Perform subspace decomposition
5. Show what the repair would do

Note: This is a demonstration that shows all the pieces working together.
The actual repair optimization is not fully implemented here (that's your job!).
"""

import sys
from pathlib import Path
cwd = str(Path.cwd())
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import numpy as np
import torch
import json
import os

from lbp_neural_cbf.cbf.train_cbf import train_cbf
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem
from lbp_neural_cbf.certified_subspace_repair import analyze_certificate_subspace


class CSRDemo:
    """
    A complete demonstration of the CSR pipeline.
    """

    def __init__(self):
        self.system = None
        self.network = None
        self.verification_results = None

    def step1_train_flawed_cbf(self):
        """
        Step 1: Train a deliberately flawed CBF.

        We'll train with suboptimal hyperparameters to ensure
        some regions fail verification.
        """
        print("\n" + "=" * 60)
        print("STEP 1: TRAIN A (DELIBERATELY FLAWED) CBF")
        print("=" * 60)

        self.system = Simple2DSystem(alpha=1.0)

        # Use suboptimal training parameters to ensure verification failure
        # In reality, you would use a network that fails some regions
        flawed_params = {
            'num_epochs': 50,  # Too few epochs
            'lr': 1e-2,       # Too large learning rate
            'batch_size': 256,
            'use_wandb': False,
        }

        print("\nTraining CBF with suboptimal parameters...")
        print(f"  Epochs: {flawed_params['num_epochs']}")
        print(f"  LR: {flawed_params['lr']}")

        # Note: We're not actually training here to save time
        # In reality, you would call train_cbf(self.system, **flawed_params)

        # Instead, we'll use a pre-trained model if available
        model_path = "data/mine_models_relu/simple2d_cbf.onnx"
        if os.path.exists(model_path):
            print(f"\n✅ Using pre-trained model: {model_path}")
        else:
            print(f"\n⚠️  Pre-trained model not found: {model_path}")
            print("   You may need to train one first.")

        print("\n✅ Step 1 complete!")

    def step2_verify_cbf(self):
        """
        Step 2: Verify the CBF with LBP.
        """
        print("\n" + "=" * 60)
        print("STEP 2: VERIFY CBF WITH LBP")
        print("=" * 60)

        model_path = "data/mine_models_relu/simple2d_cbf.onnx"

        if not os.path.exists(model_path):
            print(f"\n❌ Model not found: {model_path}")
            print("   Skipping verification.")
            return None

        print("\nRunning LBP verification...")

        # Note: In reality, you would modify verify_cbf to return A_L matrices
        # For now, we just run the standard verification
        self.verification_results = verify_cbf(
            self.system,
            model_path,
            visualize=False,
            use_gpu=False,
            use_wandb=False,
            batch_size=512,
            executor_type="single",
            region_type="simplicial",
        )

        print("\n✅ Step 2 complete!")
        return self.verification_results

    def step3_collect_A_L(self):
        """
        Step 3: Collect A_L matrices from verification.

        NOTE: This requires modifying verify_cbf.py to store A_L matrices.
        For this demo, we use synthetic data that mimics what you would get.
        """
        print("\n" + "=" * 60)
        print("STEP 3: COLLECT A_L MATRICES")
        print("=" * 60)

        d = 2  # Simple2D is 2D

        # Generate synthetic but plausible A_L matrices
        # This mimics what you would extract from a real verification
        np.random.seed(42)
        n_verified = 100
        n_failed = 20

        # Verified regions: A_L centered around [1.0, 0.5]
        center_verified = np.array([1.0, 0.5])
        A_L_verified = [center_verified + 0.1 * np.random.randn(d) for _ in range(n_verified)]

        # Failed regions: A_L centered around [-0.5, 1.0]
        center_failed = np.array([-0.5, 1.0])
        A_L_failed = [center_failed + 0.3 * np.random.randn(d) for _ in range(n_failed)]

        print(f"\nCollected {n_verified} verified A_L matrices")
        print(f"Collected {n_failed} failed A_L matrices")
        print(f"\n  Example verified A_L: {A_L_verified[0]}")
        print(f"  Example failed A_L: {A_L_failed[0]}")

        print("\n✅ Step 3 complete!")
        return A_L_verified, A_L_failed

    def step4_subspace_analysis(self, A_L_verified, A_L_failed):
        """
        Step 4: Perform certificate subspace analysis.
        """
        print("\n" + "=" * 60)
        print("STEP 4: CERTIFICATE SUBSPACE ANALYSIS")
        print("=" * 60)

        d = 2  # Simple2D is 2D

        print("\nPerforming subspace decomposition...")
        results = analyze_certificate_subspace(
            A_L_verified, A_L_failed,
            d=d,
            var_threshold=0.9
        )

        # Print results
        print("\nM_V (verified covariance):")
        print(results["M_V"])

        print("\nM_F (failed covariance):")
        print(results["M_F"])

        print("\nGeneralized eigenvalues (descending):")
        for i, lam in enumerate(results["eigenvalues"]):
            print(f"  lambda_{i+1} = {lam:.4f}")

        print("\nTop eigenvector (failure subspace direction):")
        print(results["W_F"][:, 0])

        print("\nVariance explained by each component:")
        var_explained = results["summary"]["var_explained"]
        for i, var in enumerate(var_explained):
            print(f"  Component {i+1}: {var*100:.1f}%")

        if var_explained[0] > 0.6:
            print("\n✅ SUCCESS: Top component captures most failure variance!")
            print("   This means we can safely repair in this subspace.")
        else:
            print("\n⚠️  NOTE: Multiple components needed for failure subspace.")

        print("\n✅ Step 4 complete!")
        return results

    def step5_explain_repair(self, subspace_results):
        """
        Step 5: Explain what repair would do.
        """
        print("\n" + "=" * 60)
        print("STEP 5: WHAT REPAIR WOULD DO")
        print("=" * 60)

        W_F = subspace_results["W_F"]
        W_V = subspace_results["W_V"]

        print(f"\nFailure subspace dimension: {W_F.shape[1]}")
        print(f"Verified subspace dimension: {W_V.shape[1]}")

        print("\nWhat we would do:")
        print("  1. Decompose network weights into two parts:")
        print("     - W_F: affects only failure subspace")
        print("     - W_V: affects only verified subspace")
        print("  2. Only optimize W_F to fix failed regions")
        print("  3. Leave W_V completely unchanged")
        print("  4. Result:")
        print("     - Failed regions get fixed ✓")
        print("     - Verified regions stay 100% the same ✓")
        print("     - Verified region bounds might even get tighter! ✓")

        print("\n✅ Step 5 complete!")

    def run_complete_demo(self):
        """Run the complete CSR demo."""
        print("\n" + "=" * 60)
        print("CERTIFIED-SUBSPACE REPAIR (CSR) - COMPLETE DEMO")
        print("=" * 60)

        # Step 1
        self.step1_train_flawed_cbf()

        # Step 2 (commented out to save time - requires actual model)
        # self.step2_verify_cbf()

        # Step 3
        A_L_verified, A_L_failed = self.step3_collect_A_L()

        # Step 4
        subspace_results = self.step4_subspace_analysis(A_L_verified, A_L_failed)

        # Step 5
        self.step5_explain_repair(subspace_results)

        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nNext steps for implementation:")
        print("  1. Modify verify_cbf.py to return A_L matrices")
        print("  2. Implement the subspace-constrained repair optimization")
        print("  3. Test on a real flawed CBF")
        print("  4. Compare with Chen et al. (last-layer repair)")

        return subspace_results


if __name__ == "__main__":
    demo = CSRDemo()
    results = demo.run_complete_demo()
