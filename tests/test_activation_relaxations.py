"""
Comprehensive test suite for all activation function relaxations.

This module tests the correctness and visualizes the bounds for all implemented
activation relaxations including ReLU, LeakyReLU, Sigmoid, and Tanh.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import Dict, List, Tuple, Callable

from lbp_neural_cbf.linearization.activations.relu import (
    ReLUActivationRelaxation,
)
from lbp_neural_cbf.linearization.activations.leaky_relu import (
    LeakyReLUActivationRelaxation,
)
from lbp_neural_cbf.linearization.activations.sigmoid import (
    SigmoidActivationRelaxation,
)
from lbp_neural_cbf.linearization.activations.tanh import (
    TanhActivationRelaxation,
)
from lbp_neural_cbf.linearization.activations.activation_relaxations import (
    ActivationRelaxation,
)


class ActivationTestConfig:
    """Configuration for testing an activation function."""

    def __init__(
        self,
        name: str,
        relaxation_class: type,
        activation_func: Callable[[np.ndarray], np.ndarray],
        test_intervals: List[Tuple[float, float]],
        init_kwargs: Dict = None,
    ):
        self.name = name
        self.relaxation_class = relaxation_class
        self.activation_func = activation_func
        self.test_intervals = test_intervals
        self.init_kwargs = init_kwargs or {}


# Define all activation configurations with region-specific test cases
ACTIVATION_CONFIGS = [
    ActivationTestConfig(
        name="ReLU",
        relaxation_class=ReLUActivationRelaxation,
        activation_func=lambda x: np.maximum(0, x),
        test_intervals=[
            (-2.0, 2.0),  # Crossing zero (main case)
            (-1.0, 1.0),  # Symmetric crossing zero
            (-0.5, 0.1),  # Asymmetric crossing zero
            (-2.0, -0.5),  # Pure negative (inactive)
            (0.5, 2.0),  # Pure positive (active)
            (-0.01, 0.01),  # Small interval crossing zero
        ],
    ),
    ActivationTestConfig(
        name="LeakyReLU",
        relaxation_class=LeakyReLUActivationRelaxation,
        activation_func=lambda x: np.where(x >= 0, x, 0.01 * x),
        test_intervals=[
            (-2.0, 2.0),  # Crossing zero (main case)
            (-1.0, 1.0),  # Symmetric crossing zero
            (-0.5, 0.1),  # Asymmetric crossing zero
            (-2.0, -0.5),  # Pure negative
            (0.5, 2.0),  # Pure positive
            (-0.01, 0.01),  # Small interval
        ],
        init_kwargs={"negative_slope": 0.01},
    ),
    ActivationTestConfig(
        name="Sigmoid",
        relaxation_class=SigmoidActivationRelaxation,
        activation_func=lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
        test_intervals=[
            (-2.0, 2.0),  # Crossing inflection point (main case)
            (-1.0, 1.0),  # Symmetric around zero
            (-3.0, 0.5),  # Asymmetric (left heavy)
            (-0.5, 3.0),  # Asymmetric (right heavy)
            (-5.0, -2.0),  # Pure convex region
            (2.0, 5.0),  # Pure concave region
            (-0.1, 0.1),  # Small interval crossing zero
            (-10.0, 10.0),  # Large interval
            # Derivative-specific test intervals (x_inf ≈ 1.317)
            (-1.0, 1.0),  # Purely concave region for derivative (Case A)
            (-1.2, 1.2),  # Purely concave region for derivative (Case A, close to inflection)
            (-0.5, 0.5),  # Purely concave region for derivative (Case A, smaller)
            (-3.0, -1.5),  # Purely convex region for derivative (Case B, left)
            (1.5, 3.0),  # Purely convex region for derivative (Case B, right)
            (-5.0, -2.0),  # Purely convex region for derivative (Case B, far left)
            (2.0, 5.0),  # Purely convex region for derivative (Case B, far right)
            (-2.0, 2.0),  # Mixed concavity region for derivative (Case C)
            (-1.5, 1.5),  # Mixed concavity region for derivative (Case C, smaller)
            (-3.0, 1.0),  # Mixed concavity region for derivative (Case C, asymmetric left)
            (-1.0, 3.0),  # Mixed concavity region for derivative (Case C, asymmetric right)
        ],
    ),
    ActivationTestConfig(
        name="Tanh",
        relaxation_class=TanhActivationRelaxation,
        activation_func=lambda x: np.tanh(x),
        test_intervals=[
            (-1.0, 1.0),  # Symmetric crossing zero (main case)
            (-2.0, 0.5),  # Asymmetric crossing zero (left heavy)
            (-0.5, 2.0),  # Asymmetric crossing zero (right heavy)
            (-0.1, 0.1),  # Small interval crossing zero
            (-2.0, -0.5),  # Pure convex region
            (0.5, 2.0),  # Pure concave region
            (-0.01, 0.01),  # Very small interval
            (-5.0, 5.0),  # Large interval
            # Derivative-specific test intervals (x_inf ≈ 0.6585)
            (-0.5, 0.5),  # Purely concave region for derivative (Case A)
            (-2.0, -0.8),  # Purely convex region for derivative (Case B, left)
            (0.8, 2.0),  # Purely convex region for derivative (Case B, right)
            (-1.5, 1.5),  # Mixed concavity region for derivative (Case C)
            (-0.8, 0.8),  # Mixed concavity region for derivative (Case C, smaller)
            (-3.0, 3.0),  # Mixed concavity region for derivative (Case C, large)
        ],
    ),
]


class TestActivationRelaxations:
    """Test suite for all activation relaxations."""

    @pytest.mark.parametrize("config", ACTIVATION_CONFIGS)
    def test_individual_activation_soundness(self, config):
        """Test soundness for each activation individually - this will fail for broken implementations."""
        relaxation = config.relaxation_class(**config.init_kwargs)

        failed_intervals = []

        for lb_val, ub_val in config.test_intervals:
            lb = torch.tensor([lb_val])
            ub = torch.tensor([ub_val])

            alpha_L, beta_L, alpha_U, beta_U = relaxation.relax_activation(lb, ub)

            # Test soundness over dense grid
            x_vals = np.linspace(lb_val, ub_val, 1000)
            activation_vals = config.activation_func(x_vals)
            lower_vals = alpha_L.item() * x_vals + beta_L.item()
            upper_vals = alpha_U.item() * x_vals + beta_U.item()

            max_lower_violation = np.max(lower_vals - activation_vals)
            max_upper_violation = np.max(activation_vals - upper_vals)

            if max_lower_violation > 1e-6 or max_upper_violation > 1e-6:
                failed_intervals.append(
                    {
                        "interval": (lb_val, ub_val),
                        "lower_violation": max_lower_violation,
                        "upper_violation": max_upper_violation,
                    }
                )

        # Assert that no intervals failed for this activation
        if failed_intervals:
            failure_details = []
            for fail in failed_intervals:
                failure_details.append(f"Interval {fail['interval']}: lower_viol={fail['lower_violation']:.2e}, " f"upper_viol={fail['upper_violation']:.2e}")

            pytest.fail(f"{config.name} relaxation has unsound bounds in {len(failed_intervals)} intervals:\n" + "\n".join(failure_details))

    def test_activation_bounds_at_endpoints(self):
        """Test that bounds are exact or conservative at interval endpoints."""

        for config in ACTIVATION_CONFIGS:
            relaxation = config.relaxation_class(**config.init_kwargs)

            for lb_val, ub_val in config.test_intervals:
                lb = torch.tensor([lb_val])
                ub = torch.tensor([ub_val])

                alpha_L, beta_L, alpha_U, beta_U = relaxation.relax_activation(lb, ub)

                alpha_L_val = alpha_L.item()
                beta_L_val = beta_L.item()
                alpha_U_val = alpha_U.item()
                beta_U_val = beta_U.item()

                # Check bounds at endpoints
                activation_at_lb = config.activation_func(np.array([lb_val]))[0]
                activation_at_ub = config.activation_func(np.array([ub_val]))[0]

                lower_at_lb = alpha_L_val * lb_val + beta_L_val
                lower_at_ub = alpha_L_val * ub_val + beta_L_val
                upper_at_lb = alpha_U_val * lb_val + beta_U_val
                upper_at_ub = alpha_U_val * ub_val + beta_U_val

                # Lower bounds should be <= activation values
                assert lower_at_lb <= activation_at_lb + 1e-6, f"{config.name}: Lower bound violation at lb: {lower_at_lb} > {activation_at_lb}"
                assert lower_at_ub <= activation_at_ub + 1e-6, f"{config.name}: Lower bound violation at ub: {lower_at_ub} > {activation_at_ub}"

                # Upper bounds should be >= activation values
                assert upper_at_lb >= activation_at_lb - 1e-6, f"{config.name}: Upper bound violation at lb: {upper_at_lb} < {activation_at_lb}"
                assert upper_at_ub >= activation_at_ub - 1e-6, f"{config.name}: Upper bound violation at ub: {upper_at_ub} < {activation_at_ub}"

    def test_activation_derivative_soundness(self):
        """Test that all activation derivative relaxations produce sound bounds."""

        # Define derivative functions for each activation
        derivative_functions = {
            "ReLU": lambda x: np.where(x > 0, 1.0, 0.0),  # Step function (technically undefined at 0)
            "LeakyReLU": lambda x: np.where(x > 0, 1.0, 0.01),  # Step function with negative slope
            "Sigmoid": lambda x: np.exp(-np.clip(x, -500, 500)) / (1 + np.exp(-np.clip(x, -500, 500))) ** 2,  # σ(x)(1-σ(x))
            "Tanh": lambda x: 1 - np.tanh(x) ** 2,  # sech²(x)
        }

        all_passed = True
        results = {}

        for config in ACTIVATION_CONFIGS:
            if config.name not in derivative_functions:
                continue

            print(f"\n{'='*60}")
            print(f"TESTING {config.name.upper()} DERIVATIVE RELAXATION")
            print(f"{'='*60}")

            # Initialize relaxation
            relaxation = config.relaxation_class(**config.init_kwargs)
            derivative_func = derivative_functions[config.name]

            config_results = []
            config_passed = True

            for i, (lb_val, ub_val) in enumerate(config.test_intervals):
                print(f"\nDerivative Test {i+1}: Interval [{lb_val}, {ub_val}]")
                print("-" * 50)

                # Create tensors
                lb = torch.tensor([lb_val])
                ub = torch.tensor([ub_val])

                # Get derivative bounds
                gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

                gamma_L_val = gamma_L.item()
                delta_L_val = delta_L.item()
                gamma_U_val = gamma_U.item()
                delta_U_val = delta_U.item()

                print(f"Lower bound: {gamma_L_val:.6f} * y + {delta_L_val:.6f}")
                print(f"Upper bound: {gamma_U_val:.6f} * y + {delta_U_val:.6f}")

                # Verify soundness over dense grid
                x_vals = np.linspace(lb_val, ub_val, 1000)

                # Handle discontinuous derivatives (ReLU, LeakyReLU)
                if config.name in ["ReLU", "LeakyReLU"]:
                    # For step functions, test away from discontinuity
                    x_vals = x_vals[np.abs(x_vals) > 1e-8]

                if len(x_vals) > 0:
                    derivative_vals = derivative_func(x_vals)
                    lower_vals = gamma_L_val * x_vals + delta_L_val
                    upper_vals = gamma_U_val * x_vals + delta_U_val

                    # Check soundness
                    lower_violations = lower_vals - derivative_vals
                    upper_violations = derivative_vals - upper_vals

                    max_lower_violation = np.max(lower_violations)
                    max_upper_violation = np.max(upper_violations)

                    # Compute tightness metrics
                    avg_gap = np.mean(upper_vals - lower_vals)

                    print(f"Max lower violation: {max_lower_violation:.8f}")
                    print(f"Max upper violation: {max_upper_violation:.8f}")
                    print(f"Average gap: {avg_gap:.6f}")

                    # Check if bounds are sound (allowing small numerical errors)
                    is_sound = max_lower_violation <= 1e-6 and max_upper_violation <= 1e-6
                else:
                    # Degenerate case
                    is_sound = True
                    max_lower_violation = 0.0
                    max_upper_violation = 0.0
                    avg_gap = abs(delta_U_val - delta_L_val)

                test_result = {
                    "interval": (lb_val, ub_val),
                    "bounds": (
                        gamma_L_val,
                        delta_L_val,
                        gamma_U_val,
                        delta_U_val,
                    ),
                    "max_lower_violation": max_lower_violation,
                    "max_upper_violation": max_upper_violation,
                    "avg_gap": avg_gap,
                    "is_sound": is_sound,
                }

                config_results.append(test_result)

                if is_sound:
                    print("✓ PASSED: Derivative bounds are sound")
                else:
                    print("✗ FAILED: Derivative bounds are not sound!")
                    config_passed = False
                    all_passed = False

            results[config.name] = config_results

        # Overall summary
        print(f"\n{'='*60}")
        working_derivatives = [name for name, results_list in results.items() if all(r["is_sound"] for r in results_list)]
        failing_derivatives = [name for name, results_list in results.items() if not all(r["is_sound"] for r in results_list)]

        print(f"✓ Working derivative relaxations: {working_derivatives}")
        if failing_derivatives:
            print(f"⚠ Failing derivative relaxations: {failing_derivatives}")
        print(f"{'='*60}")

        # Assert that all derivative relaxations are sound
        assert all_passed, f"Some derivative relaxations failed soundness tests: {failing_derivatives}"

    @pytest.mark.parametrize("activation_name", ["Tanh", "Sigmoid"])
    def test_specific_derivative_soundness(self, activation_name):
        """Test derivative soundness for specific activations known to work correctly."""

        derivative_functions = {
            "Tanh": lambda x: 1 - np.tanh(x) ** 2,  # sech²(x)
            "Sigmoid": lambda x: np.exp(-np.clip(x, -500, 500)) / (1 + np.exp(-np.clip(x, -500, 500))) ** 2,  # σ(x)(1-σ(x))
        }

        if activation_name not in derivative_functions:
            pytest.skip(f"Derivative function not defined for {activation_name}")

        config = next(c for c in ACTIVATION_CONFIGS if c.name == activation_name)
        relaxation = config.relaxation_class(**config.init_kwargs)
        derivative_func = derivative_functions[activation_name]

        for lb_val, ub_val in config.test_intervals:
            lb = torch.tensor([lb_val])
            ub = torch.tensor([ub_val])

            gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

            # Test soundness
            x_vals = np.linspace(lb_val, ub_val, 100)
            derivative_vals = derivative_func(x_vals)
            lower_vals = gamma_L.item() * x_vals + delta_L.item()
            upper_vals = gamma_U.item() * x_vals + delta_U.item()

            max_lower_violation = np.max(lower_vals - derivative_vals)
            max_upper_violation = np.max(derivative_vals - upper_vals)

            assert max_lower_violation <= 1e-6, f"Lower bound violation: {max_lower_violation}"
            assert max_upper_violation <= 1e-6, f"Upper bound violation: {max_upper_violation}"

    def test_tanh_derivative_specific_cases(self):
        """Test specific mathematical properties of tanh derivative (sech²) relaxation."""

        relaxation = TanhActivationRelaxation()

        # Test case 1: Purely concave region [-0.5, 0.5] (within inflection points)
        lb, ub = torch.tensor([-0.5]), torch.tensor([0.5])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        # In concave region: lower bound should be secant, upper bound should be tangent
        x_vals = np.linspace(-0.5, 0.5, 1000)
        derivative_vals = 1 - np.tanh(x_vals) ** 2
        lower_vals = gamma_L.item() * x_vals + delta_L.item()
        upper_vals = gamma_U.item() * x_vals + delta_U.item()

        # Check soundness
        assert np.all(lower_vals <= derivative_vals + 1e-6), "Lower bound violation in concave region"
        assert np.all(upper_vals >= derivative_vals - 1e-6), "Upper bound violation in concave region"

        # Test case 2: Purely convex region [1.0, 2.0] (outside inflection points)
        lb, ub = torch.tensor([1.0]), torch.tensor([2.0])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        x_vals = np.linspace(1.0, 2.0, 1000)
        derivative_vals = 1 - np.tanh(x_vals) ** 2
        lower_vals = gamma_L.item() * x_vals + delta_L.item()
        upper_vals = gamma_U.item() * x_vals + delta_U.item()

        # Check soundness
        assert np.all(lower_vals <= derivative_vals + 1e-6), "Lower bound violation in convex region"
        assert np.all(upper_vals >= derivative_vals - 1e-6), "Upper bound violation in convex region"

        # Test case 3: Mixed region crossing inflection points [-1.0, 1.0]
        lb, ub = torch.tensor([-1.0]), torch.tensor([1.0])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        x_vals = np.linspace(-1.0, 1.0, 1000)
        derivative_vals = 1 - np.tanh(x_vals) ** 2
        lower_vals = gamma_L.item() * x_vals + delta_L.item()
        upper_vals = gamma_U.item() * x_vals + delta_U.item()

        # Check soundness
        assert np.all(lower_vals <= derivative_vals + 1e-6), "Lower bound violation in mixed region"
        assert np.all(upper_vals >= derivative_vals - 1e-6), "Upper bound violation in mixed region"

        # Test case 4: Maximum point (derivative is maximized at x=0)
        lb, ub = torch.tensor([-0.1]), torch.tensor([0.1])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        # At x=0, derivative should equal 1.0
        max_derivative = 1.0
        upper_at_zero = gamma_U.item() * 0.0 + delta_U.item()

        # Upper bound should be at least the maximum value
        assert upper_at_zero >= max_derivative - 1e-6, f"Upper bound too low at maximum: {upper_at_zero} < {max_derivative}"

    def test_tanh_derivative_comprehensive_cases(self):
        """Test all three cases of tanh derivative relaxation explicitly to ensure coverage."""

        relaxation = TanhActivationRelaxation()

        # The inflection points for tanh derivative: x_inf ≈ 0.6585
        # From tanh.py: x_inf = np.arctanh(1.0 / np.sqrt(3.0))
        x_inf = np.arctanh(1.0 / np.sqrt(3.0))
        print(f"Tanh derivative inflection point: x_inf = {x_inf:.4f}")

        # Case A: Purely concave region (-x_inf <= lb < ub <= x_inf)
        print("\nTesting Case A: Purely concave region")
        test_intervals_case_a = [
            (-0.5, 0.5),  # Well within inflection points
            (-0.6, 0.6),  # Close to inflection points
            (-0.3, 0.4),  # Asymmetric within inflection points
            (-0.1, 0.1),  # Small interval around maximum
        ]

        for lb_val, ub_val in test_intervals_case_a:
            print(f"  Testing interval [{lb_val}, {ub_val}]")
            assert -x_inf <= lb_val < ub_val <= x_inf, "Should be purely concave"

            lb, ub = torch.tensor([lb_val]), torch.tensor([ub_val])
            gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

            # Verify soundness
            x_vals = np.linspace(lb_val, ub_val, 100)
            derivative_vals = 1 - np.tanh(x_vals) ** 2
            lower_vals = gamma_L.item() * x_vals + delta_L.item()
            upper_vals = gamma_U.item() * x_vals + delta_U.item()

            max_lower_viol = np.max(lower_vals - derivative_vals)
            max_upper_viol = np.max(derivative_vals - upper_vals)

            assert max_lower_viol <= 1e-6, f"Case A lower bound violation: {max_lower_viol}"
            assert max_upper_viol <= 1e-6, f"Case A upper bound violation: {max_upper_viol}"
            print(f"    ✓ Case A passed with violations: lower={max_lower_viol:.2e}, upper={max_upper_viol:.2e}")

        # Case B: Purely convex region (lb >= x_inf OR ub <= -x_inf)
        print("\nTesting Case B: Purely convex region")
        test_intervals_case_b = [
            (0.8, 2.0),  # Right convex region (lb >= x_inf)
            (1.0, 3.0),  # Well into right convex region
            (-2.0, -0.8),  # Left convex region (ub <= -x_inf)
            (-3.0, -1.0),  # Well into left convex region
        ]

        for lb_val, ub_val in test_intervals_case_b:
            print(f"  Testing interval [{lb_val}, {ub_val}]")
            assert (lb_val >= x_inf) or (ub_val <= -x_inf), "Should be purely convex"

            lb, ub = torch.tensor([lb_val]), torch.tensor([ub_val])
            gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

            # Verify soundness
            x_vals = np.linspace(lb_val, ub_val, 100)
            derivative_vals = 1 - np.tanh(x_vals) ** 2
            lower_vals = gamma_L.item() * x_vals + delta_L.item()
            upper_vals = gamma_U.item() * x_vals + delta_U.item()

            max_lower_viol = np.max(lower_vals - derivative_vals)
            max_upper_viol = np.max(derivative_vals - upper_vals)

            assert max_lower_viol <= 1e-6, f"Case B lower bound violation: {max_lower_viol}"
            assert max_upper_viol <= 1e-6, f"Case B upper bound violation: {max_upper_viol}"
            print(f"    ✓ Case B passed with violations: lower={max_lower_viol:.2e}, upper={max_upper_viol:.2e}")

        # Case C: Mixed concavity region (crosses inflection points)
        print("\nTesting Case C: Mixed concavity region")
        test_intervals_case_c = [
            (-1.0, 1.0),  # Symmetric crossing both inflection points
            (-1.5, 1.5),  # Larger symmetric crossing
            (-0.8, 0.8),  # Crossing both inflection points, smaller
            (-2.0, 1.0),  # Asymmetric crossing, left heavy
            (-1.0, 2.0),  # Asymmetric crossing, right heavy
            (-3.0, 3.0),  # Large interval crossing both inflection points
        ]

        for lb_val, ub_val in test_intervals_case_c:
            print(f"  Testing interval [{lb_val}, {ub_val}]")
            # Case C: NOT purely concave AND NOT purely convex
            assert not (-x_inf <= lb_val < ub_val <= x_inf), "Should not be purely concave"
            assert not ((lb_val >= x_inf) or (ub_val <= -x_inf)), "Should not be purely convex"

            lb, ub = torch.tensor([lb_val]), torch.tensor([ub_val])
            gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

            # Verify soundness
            x_vals = np.linspace(lb_val, ub_val, 100)
            derivative_vals = 1 - np.tanh(x_vals) ** 2
            lower_vals = gamma_L.item() * x_vals + delta_L.item()
            upper_vals = gamma_U.item() * x_vals + delta_U.item()

            max_lower_viol = np.max(lower_vals - derivative_vals)
            max_upper_viol = np.max(derivative_vals - upper_vals)

            assert max_lower_viol <= 1e-6, f"Case C lower bound violation: {max_lower_viol}"
            assert max_upper_viol <= 1e-6, f"Case C upper bound violation: {max_upper_viol}"
            print(f"    ✓ Case C passed with violations: lower={max_lower_viol:.2e}, upper={max_upper_viol:.2e}")

        print("\n✓ All tanh derivative cases (A, B, C) tested successfully!")

    def test_sigmoid_derivative_comprehensive_cases(self):
        """Test all three cases of sigmoid derivative relaxation explicitly to ensure coverage."""

        relaxation = SigmoidActivationRelaxation()

        # Sigmoid derivative inflection points: x_inf ≈ 1.317
        # From sigmoid.py: x_inf = torch.log((3 + sqrt_3) / (3 - sqrt_3))
        sqrt_3 = np.sqrt(3.0)
        x_inf = np.log((3 + sqrt_3) / (3 - sqrt_3))
        print(f"Sigmoid derivative inflection point: x_inf = {x_inf:.4f}")

        # Case A: Purely concave region (lb >= -x_inf and ub <= x_inf)
        print("\nTesting Case A: Purely concave region")
        test_intervals_case_a = [
            (-1.0, 1.0),  # Around zero, within inflection points
            (-0.5, 0.5),  # Smaller around zero
            (-1.2, 1.2),  # Close to inflection points
            (-0.8, 0.8),  # Medium interval within inflection points
        ]

        for lb_val, ub_val in test_intervals_case_a:
            print(f"  Testing interval [{lb_val}, {ub_val}]")
            assert lb_val >= -x_inf and ub_val <= x_inf, "Should be purely concave"

            lb, ub = torch.tensor([lb_val]), torch.tensor([ub_val])
            gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

            # Verify soundness
            x_vals = np.linspace(lb_val, ub_val, 100)
            derivative_vals = np.exp(-np.clip(x_vals, -500, 500)) / (1 + np.exp(-np.clip(x_vals, -500, 500))) ** 2
            lower_vals = gamma_L.item() * x_vals + delta_L.item()
            upper_vals = gamma_U.item() * x_vals + delta_U.item()

            max_lower_viol = np.max(lower_vals - derivative_vals)
            max_upper_viol = np.max(derivative_vals - upper_vals)

            assert max_lower_viol <= 1e-6, f"Case A lower bound violation: {max_lower_viol}"
            assert max_upper_viol <= 1e-6, f"Case A upper bound violation: {max_upper_viol}"
            print(f"    ✓ Case A passed with violations: lower={max_lower_viol:.2e}, upper={max_upper_viol:.2e}")

        # Case B: Purely convex region (lb >= x_inf OR ub <= -x_inf)
        print("\nTesting Case B: Purely convex region")
        test_intervals_case_b = [
            (1.5, 3.0),  # Right convex region (lb >= x_inf)
            (2.0, 5.0),  # Well into right convex region
            (-3.0, -1.5),  # Left convex region (ub <= -x_inf)
            (-5.0, -2.0),  # Well into left convex region
        ]

        for lb_val, ub_val in test_intervals_case_b:
            print(f"  Testing interval [{lb_val}, {ub_val}]")
            assert (lb_val >= x_inf) or (ub_val <= -x_inf), "Should be purely convex"

            lb, ub = torch.tensor([lb_val]), torch.tensor([ub_val])
            gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

            # Verify soundness
            x_vals = np.linspace(lb_val, ub_val, 100)
            derivative_vals = np.exp(-np.clip(x_vals, -500, 500)) / (1 + np.exp(-np.clip(x_vals, -500, 500))) ** 2
            lower_vals = gamma_L.item() * x_vals + delta_L.item()
            upper_vals = gamma_U.item() * x_vals + delta_U.item()

            max_lower_viol = np.max(lower_vals - derivative_vals)
            max_upper_viol = np.max(derivative_vals - upper_vals)

            assert max_lower_viol <= 1e-6, f"Case B lower bound violation: {max_lower_viol}"
            assert max_upper_viol <= 1e-6, f"Case B upper bound violation: {max_upper_viol}"
            print(f"    ✓ Case B passed with violations: lower={max_lower_viol:.2e}, upper={max_upper_viol:.2e}")

        # Case C: Mixed concavity region (crosses inflection points)
        print("\nTesting Case C: Mixed concavity region")
        test_intervals_case_c = [
            (-2.0, 2.0),  # Symmetric crossing both inflection points
            (-1.5, 1.5),  # Smaller symmetric crossing
            (-3.0, 1.0),  # Asymmetric crossing, left heavy
            (-1.0, 3.0),  # Asymmetric crossing, right heavy
            (-4.0, 4.0),  # Large interval crossing both inflection points
        ]

        for lb_val, ub_val in test_intervals_case_c:
            print(f"  Testing interval [{lb_val}, {ub_val}]")
            # Case C: NOT purely concave AND NOT purely convex
            assert not (lb_val >= -x_inf and ub_val <= x_inf), "Should not be purely concave"
            assert not ((lb_val >= x_inf) or (ub_val <= -x_inf)), "Should not be purely convex"

            lb, ub = torch.tensor([lb_val]), torch.tensor([ub_val])
            gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

            # Verify soundness
            x_vals = np.linspace(lb_val, ub_val, 100)
            derivative_vals = np.exp(-np.clip(x_vals, -500, 500)) / (1 + np.exp(-np.clip(x_vals, -500, 500))) ** 2
            lower_vals = gamma_L.item() * x_vals + delta_L.item()
            upper_vals = gamma_U.item() * x_vals + delta_U.item()

            max_lower_viol = np.max(lower_vals - derivative_vals)
            max_upper_viol = np.max(derivative_vals - upper_vals)

            assert max_lower_viol <= 1e-6, f"Case C lower bound violation: {max_lower_viol}"
            assert max_upper_viol <= 1e-6, f"Case C upper bound violation: {max_upper_viol}"
            print(f"    ✓ Case C passed with violations: lower={max_lower_viol:.2e}, upper={max_upper_viol:.2e}")

        print("\n✓ All sigmoid derivative cases (A, B, C) tested successfully!")

    def test_sigmoid_derivative_specific_cases(self):
        """Test specific mathematical properties of sigmoid derivative (σ'(x) = σ(x)(1-σ(x))) relaxation."""

        relaxation = SigmoidActivationRelaxation()

        # Test case 1: Symmetric region around zero [-1.0, 1.0] (crossing the inflection point)
        lb, ub = torch.tensor([-1.0]), torch.tensor([1.0])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        # Sigmoid derivative is concave everywhere (bell-shaped curve)
        x_vals = np.linspace(-1.0, 1.0, 1000)
        derivative_vals = np.exp(-np.clip(x_vals, -500, 500)) / (1 + np.exp(-np.clip(x_vals, -500, 500))) ** 2
        lower_vals = gamma_L.item() * x_vals + delta_L.item()
        upper_vals = gamma_U.item() * x_vals + delta_U.item()

        # Check soundness
        assert np.all(lower_vals <= derivative_vals + 1e-6), "Lower bound violation in symmetric region"
        assert np.all(upper_vals >= derivative_vals - 1e-6), "Upper bound violation in symmetric region"

        # Test case 2: Left tail region [-5.0, -2.0] (where derivative approaches 0)
        lb, ub = torch.tensor([-5.0]), torch.tensor([-2.0])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        x_vals = np.linspace(-5.0, -2.0, 1000)
        derivative_vals = np.exp(-np.clip(x_vals, -500, 500)) / (1 + np.exp(-np.clip(x_vals, -500, 500))) ** 2
        lower_vals = gamma_L.item() * x_vals + delta_L.item()
        upper_vals = gamma_U.item() * x_vals + delta_U.item()

        # Check soundness
        assert np.all(lower_vals <= derivative_vals + 1e-6), "Lower bound violation in left tail region"
        assert np.all(upper_vals >= derivative_vals - 1e-6), "Upper bound violation in left tail region"

        # Test case 3: Right tail region [2.0, 5.0] (where derivative approaches 0)
        lb, ub = torch.tensor([2.0]), torch.tensor([5.0])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        x_vals = np.linspace(2.0, 5.0, 1000)
        derivative_vals = np.exp(-np.clip(x_vals, -500, 500)) / (1 + np.exp(-np.clip(x_vals, -500, 500))) ** 2
        lower_vals = gamma_L.item() * x_vals + delta_L.item()
        upper_vals = gamma_U.item() * x_vals + delta_U.item()

        # Check soundness
        assert np.all(lower_vals <= derivative_vals + 1e-6), "Lower bound violation in right tail region"
        assert np.all(upper_vals >= derivative_vals - 1e-6), "Upper bound violation in right tail region"

        # Test case 4: Maximum point (derivative is maximized at x=0)
        lb, ub = torch.tensor([-0.1]), torch.tensor([0.1])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        # At x=0, derivative should equal 0.25 (maximum value)
        max_derivative = 0.25
        upper_at_zero = gamma_U.item() * 0.0 + delta_U.item()

        # Upper bound should be at least the maximum value
        assert upper_at_zero >= max_derivative - 1e-6, f"Upper bound too low at maximum: {upper_at_zero} < {max_derivative}"

        # Test case 5: Asymmetric region [-2.0, 0.5] (left-heavy around inflection)
        lb, ub = torch.tensor([-2.0]), torch.tensor([0.5])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        x_vals = np.linspace(-2.0, 0.5, 1000)
        derivative_vals = np.exp(-np.clip(x_vals, -500, 500)) / (1 + np.exp(-np.clip(x_vals, -500, 500))) ** 2
        lower_vals = gamma_L.item() * x_vals + delta_L.item()
        upper_vals = gamma_U.item() * x_vals + delta_U.item()

        # Check soundness
        assert np.all(lower_vals <= derivative_vals + 1e-6), "Lower bound violation in asymmetric region"
        assert np.all(upper_vals >= derivative_vals - 1e-6), "Upper bound violation in asymmetric region"

        # Test case 6: Small interval around maximum [-0.5, 0.5]
        lb, ub = torch.tensor([-0.5]), torch.tensor([0.5])
        gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

        x_vals = np.linspace(-0.5, 0.5, 1000)
        derivative_vals = np.exp(-np.clip(x_vals, -500, 500)) / (1 + np.exp(-np.clip(x_vals, -500, 500))) ** 2
        lower_vals = gamma_L.item() * x_vals + delta_L.item()
        upper_vals = gamma_U.item() * x_vals + delta_U.item()

        # Check soundness
        assert np.all(lower_vals <= derivative_vals + 1e-6), "Lower bound violation around maximum"
        assert np.all(upper_vals >= derivative_vals - 1e-6), "Upper bound violation around maximum"

        # Verify that the upper bound captures the maximum appropriately
        max_val_in_interval = np.max(derivative_vals)
        max_upper_val = np.max(upper_vals)
        assert max_upper_val >= max_val_in_interval - 1e-6, f"Upper bound doesn't capture maximum: {max_upper_val} < {max_val_in_interval}"

    def test_generate_comprehensive_visualizations(self):
        """Generate comprehensive visualization plots for all activation relaxations."""

        # Create output directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(test_dir, "activation_relaxation_plots")
        os.makedirs(plot_dir, exist_ok=True)

        print(f"\nGenerating activation relaxation visualizations in {plot_dir}/")

        for config in ACTIVATION_CONFIGS:
            print(f"Creating visualization for {config.name}...")

            # Initialize relaxation
            relaxation = config.relaxation_class(**config.init_kwargs)

            # For complex activation functions, create a larger subplot grid
            n_intervals = len(config.test_intervals)

            if n_intervals <= 4:
                n_rows, n_cols = 2, 2
            elif n_intervals <= 6:
                n_rows, n_cols = 2, 3
            elif n_intervals <= 8:
                n_rows, n_cols = 3, 3
            else:
                n_rows, n_cols = 3, 4

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

            # Handle single subplot case
            if n_intervals == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, (lb_val, ub_val) in enumerate(config.test_intervals):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Get bounds
                lb = torch.tensor([lb_val])
                ub = torch.tensor([ub_val])
                alpha_L, beta_L, alpha_U, beta_U = relaxation.relax_activation(lb, ub)

                alpha_L_val = alpha_L.item()
                beta_L_val = beta_L.item()
                alpha_U_val = alpha_U.item()
                beta_U_val = beta_U.item()

                # Create extended range for visualization
                margin = max(0.5, 0.3 * (ub_val - lb_val))
                x_plot = np.linspace(lb_val - margin, ub_val + margin, 1000)

                # Compute functions
                y_activation = config.activation_func(x_plot)
                y_lower = alpha_L_val * x_plot + beta_L_val
                y_upper = alpha_U_val * x_plot + beta_U_val

                # Plot activation function
                ax.plot(
                    x_plot,
                    y_activation,
                    "b-",
                    linewidth=3,
                    label=f"{config.name}(x)",
                )

                # Plot bounds
                ax.plot(
                    x_plot,
                    y_lower,
                    "r--",
                    linewidth=2,
                    label=f"Lower: {alpha_L_val:.3f}x + {beta_L_val:.3f}",
                )
                ax.plot(
                    x_plot,
                    y_upper,
                    "g--",
                    linewidth=2,
                    label=f"Upper: {alpha_U_val:.3f}x + {beta_U_val:.3f}",
                )

                # Mark the test interval
                ax.axvline(lb_val, color="k", linestyle=":", alpha=0.7)
                ax.axvline(ub_val, color="k", linestyle=":", alpha=0.7)
                ax.fill_betweenx(
                    ax.get_ylim(),
                    lb_val,
                    ub_val,
                    alpha=0.1,
                    color="gray",
                    label="Interval",
                )

                # Mark endpoints
                ax.plot(
                    lb_val,
                    config.activation_func(np.array([lb_val]))[0],
                    "ko",
                    markersize=6,
                )
                ax.plot(
                    ub_val,
                    config.activation_func(np.array([ub_val]))[0],
                    "ko",
                    markersize=6,
                )

                # Highlight the relaxation region within interval
                x_interval = np.linspace(lb_val, ub_val, 100)
                y_activation_interval = config.activation_func(x_interval)
                y_lower_interval = alpha_L_val * x_interval + beta_L_val
                y_upper_interval = alpha_U_val * x_interval + beta_U_val

                ax.fill_between(
                    x_interval,
                    y_lower_interval,
                    y_upper_interval,
                    alpha=0.2,
                    color="yellow",
                    label="Relaxation",
                )

                # Classify interval type for title
                interval_type = self._classify_interval(config.name, lb_val, ub_val)
                ax.set_title(f"{interval_type}: [{lb_val}, {ub_val}]", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
                ax.set_xlabel("x")
                ax.set_ylabel("y")

                # Check and display soundness in title
                x_check = np.linspace(lb_val, ub_val, 100)
                activation_check = config.activation_func(x_check)
                lower_check = alpha_L_val * x_check + beta_L_val
                upper_check = alpha_U_val * x_check + beta_U_val

                max_lower_viol = np.max(lower_check - activation_check)
                max_upper_viol = np.max(activation_check - upper_check)
                is_sound = max_lower_viol <= 1e-6 and max_upper_viol <= 1e-6

                # Add soundness indicator to title
                sound_indicator = "✓" if is_sound else "✗"
                current_title = ax.get_title()
                ax.set_title(f"{current_title} {sound_indicator}")

            # Hide unused subplots
            for i in range(len(config.test_intervals), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(
                f"{config.name} Activation Function Linear Relaxations",
                fontsize=16,
            )
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(plot_dir, f"{config.name.lower()}_relaxations.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"✓ Saved {config.name} visualization to {plot_path}")

        # Create comparison plot for main intervals
        self._create_activation_comparison_plot(plot_dir)

        print(f"✓ All visualizations saved in {plot_dir}/")

    def _classify_interval(self, activation_name: str, lb: float, ub: float) -> str:
        """Classify the type of interval for better plot titles."""

        if activation_name in ["ReLU", "LeakyReLU"]:
            if lb < 0 and ub > 0:
                return "Crossing Zero"
            elif ub <= 0:
                return "Negative Region"
            else:
                return "Positive Region"

        elif activation_name in ["Sigmoid", "Tanh"]:
            if lb < 0 and ub > 0:
                if abs(ub - (-lb)) < 0.1:
                    return "Symmetric"
                elif abs(lb) > abs(ub):
                    return "Left Heavy"
                else:
                    return "Right Heavy"
            elif ub <= 0:
                return "Convex Region"
            else:
                return "Concave Region"

        # Default classification based on interval size
        interval_size = ub - lb
        if interval_size < 0.1:
            return "Small"
        elif interval_size > 5:
            return "Large"
        else:
            return "Medium"

    def _create_activation_comparison_plot(self, plot_dir: str):
        """Create a comparison plot showing all activations on representative intervals."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Test intervals for comparison - choose representative cases
        comparison_intervals = [
            (-2.0, 2.0, "Main Crossing Case"),
            (-1.0, 1.0, "Symmetric Small"),
            (-3.0, 0.5, "Asymmetric Left"),
            (0.5, 3.0, "Positive Region"),
        ]

        for plot_idx, (lb_val, ub_val, case_name) in enumerate(comparison_intervals):
            ax = axes[plot_idx]

            colors = ["blue", "red", "green", "purple"]
            linestyles = ["-", "--", "-.", ":"]

            for config_idx, config in enumerate(ACTIVATION_CONFIGS):
                # Skip intervals that don't make sense for certain activations
                if case_name == "Positive Region" and config.name in ["ReLU", "LeakyReLU"] and lb_val > 0:
                    # For ReLU in positive region, use a different interval to show the linear behavior
                    test_lb, test_ub = lb_val, ub_val
                else:
                    test_lb, test_ub = lb_val, ub_val

                relaxation = config.relaxation_class(**config.init_kwargs)

                # Get bounds
                lb = torch.tensor([test_lb])
                ub = torch.tensor([test_ub])
                alpha_L, beta_L, alpha_U, beta_U = relaxation.relax_activation(lb, ub)

                alpha_L_val = alpha_L.item()
                beta_L_val = beta_L.item()
                alpha_U_val = alpha_U.item()
                beta_U_val = beta_U.item()

                # Create plot range
                margin = 0.5
                x_plot = np.linspace(test_lb - margin, test_ub + margin, 1000)

                # Compute functions
                y_activation = config.activation_func(x_plot)
                y_lower = alpha_L_val * x_plot + beta_L_val
                y_upper = alpha_U_val * x_plot + beta_U_val

                # Plot with different colors and styles for each activation
                color = colors[config_idx % len(colors)]

                ax.plot(
                    x_plot,
                    y_activation,
                    color=color,
                    linewidth=3,
                    label=f"{config.name}(x)",
                )
                ax.plot(
                    x_plot,
                    y_lower,
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"{config.name} Lower",
                )
                ax.plot(
                    x_plot,
                    y_upper,
                    color=color,
                    linestyle=":",
                    linewidth=2,
                    alpha=0.7,
                    label=f"{config.name} Upper",
                )

                # Highlight interval
                x_interval = np.linspace(test_lb, test_ub, 50)
                y_lower_interval = alpha_L_val * x_interval + beta_L_val
                y_upper_interval = alpha_U_val * x_interval + beta_U_val

                ax.fill_between(
                    x_interval,
                    y_lower_interval,
                    y_upper_interval,
                    color=color,
                    alpha=0.1,
                )

            # Mark the interval
            ax.axvline(test_lb, color="k", linestyle=":", alpha=0.7)
            ax.axvline(test_ub, color="k", linestyle=":", alpha=0.7)
            ax.fill_betweenx(ax.get_ylim(), test_lb, test_ub, alpha=0.05, color="gray")

            ax.set_title(f"{case_name}: [{test_lb}, {test_ub}]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.suptitle("Activation Function Relaxations Comparison", fontsize=16)
        plt.tight_layout()

        # Save comparison plot
        comparison_path = os.path.join(plot_dir, "activation_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved comparison plot to {comparison_path}")

    def test_generate_derivative_visualizations(self):
        """Generate comprehensive visualization plots for activation derivative relaxations."""

        # Define derivative functions
        derivative_functions = {
            "Sigmoid": lambda x: np.exp(-np.clip(x, -500, 500)) / (1 + np.exp(-np.clip(x, -500, 500))) ** 2,
            "Tanh": lambda x: 1 - np.tanh(x) ** 2,
        }

        # Create output directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(test_dir, "activation_relaxation_plots")
        os.makedirs(plot_dir, exist_ok=True)

        print(f"\nGenerating derivative relaxation visualizations in {plot_dir}/")

        for config in ACTIVATION_CONFIGS:
            if config.name not in derivative_functions:
                continue

            print(f"Creating derivative visualization for {config.name}...")

            # Initialize relaxation
            relaxation = config.relaxation_class(**config.init_kwargs)
            derivative_func = derivative_functions[config.name]

            # Select key intervals for visualization - use more intervals now that we have derivative-specific ones
            key_intervals = config.test_intervals
            # Limit to 12 intervals for reasonable visualization size
            if len(key_intervals) > 12:
                key_intervals = key_intervals[:12]

            n_intervals = len(key_intervals)
            if n_intervals <= 4:
                n_rows, n_cols = 2, 2
            elif n_intervals <= 6:
                n_rows, n_cols = 2, 3
            elif n_intervals <= 9:
                n_rows, n_cols = 3, 3
            elif n_intervals <= 12:
                n_rows, n_cols = 3, 4
            else:
                n_rows, n_cols = 3, 3

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

            # Handle single subplot case
            if n_intervals == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, (lb_val, ub_val) in enumerate(key_intervals):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Get derivative bounds
                lb = torch.tensor([lb_val])
                ub = torch.tensor([ub_val])
                gamma_L, delta_L, gamma_U, delta_U = relaxation.relax_activation_derivative(lb, ub)

                gamma_L_val = gamma_L.item()
                delta_L_val = delta_L.item()
                gamma_U_val = gamma_U.item()
                delta_U_val = delta_U.item()

                # Create extended range for visualization
                margin = max(0.5, 0.3 * (ub_val - lb_val))
                x_plot = np.linspace(lb_val - margin, ub_val + margin, 1000)

                # Compute functions
                y_derivative = derivative_func(x_plot)
                y_lower = gamma_L_val * x_plot + delta_L_val
                y_upper = gamma_U_val * x_plot + delta_U_val

                # Plot derivative function
                ax.plot(
                    x_plot,
                    y_derivative,
                    "b-",
                    linewidth=3,
                    label=f"{config.name}'(x)",
                )

                # Plot bounds
                ax.plot(
                    x_plot,
                    y_lower,
                    "r--",
                    linewidth=2,
                    label=f"Lower: {gamma_L_val:.3f}x + {delta_L_val:.3f}",
                )
                ax.plot(
                    x_plot,
                    y_upper,
                    "g--",
                    linewidth=2,
                    label=f"Upper: {gamma_U_val:.3f}x + {delta_U_val:.3f}",
                )

                # Mark the test interval
                ax.axvline(lb_val, color="k", linestyle=":", alpha=0.7)
                ax.axvline(ub_val, color="k", linestyle=":", alpha=0.7)
                ax.fill_betweenx(
                    ax.get_ylim(),
                    lb_val,
                    ub_val,
                    alpha=0.1,
                    color="gray",
                    label="Interval",
                )

                # Mark endpoints
                ax.plot(
                    lb_val,
                    derivative_func(np.array([lb_val]))[0],
                    "ko",
                    markersize=6,
                )
                ax.plot(
                    ub_val,
                    derivative_func(np.array([ub_val]))[0],
                    "ko",
                    markersize=6,
                )

                # Highlight the relaxation region within interval
                x_interval = np.linspace(lb_val, ub_val, 100)
                y_derivative_interval = derivative_func(x_interval)
                y_lower_interval = gamma_L_val * x_interval + delta_L_val
                y_upper_interval = gamma_U_val * x_interval + delta_U_val

                ax.fill_between(
                    x_interval,
                    y_lower_interval,
                    y_upper_interval,
                    alpha=0.2,
                    color="yellow",
                    label="Relaxation",
                )

                # Classify interval type for title
                interval_type = self._classify_derivative_interval(config.name, lb_val, ub_val)
                ax.set_title(f"{interval_type}: [{lb_val}, {ub_val}]", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_xlabel("x")
                ax.set_ylabel("f'(x)")

                # Set reasonable y-limits
                if config.name == "Tanh":
                    ax.set_ylim(0, 1.1)
                elif config.name == "Sigmoid":
                    ax.set_ylim(0, 0.3)

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle(f"{config.name} Derivative Linear Relaxations", fontsize=16)
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(plot_dir, f"{config.name.lower()}_derivative_relaxations.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"✓ Saved {config.name} derivative visualization to {plot_path}")

        print(f"✓ All derivative visualizations saved in {plot_dir}/")

    def _classify_derivative_interval(self, activation_name: str, lb: float, ub: float) -> str:
        """Classify the type of interval for derivative plots."""

        if activation_name == "Tanh":
            x_inf = np.arctanh(1.0 / np.sqrt(3.0))  # ≈ 0.6585

            if -x_inf <= lb and ub <= x_inf:
                return "Concave Region"
            elif (x_inf <= lb) or (ub <= -x_inf):
                return "Convex Region"
            else:
                return "Mixed Region"

        elif activation_name == "Sigmoid":
            # Sigmoid derivative has inflection points like tanh derivative
            # but for simplicity we classify regions by crossing zero
            if lb < 0 and ub > 0:
                return "Crossing Zero"
            elif ub <= 0:
                return "Negative Region"
            else:
                return "Positive Region"

        return "Standard"


def create_comparison_plot():
    """Create a comparison plot showing all activations on the same interval."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Test interval for comparison
    lb_val, ub_val = -2.0, 2.0

    for i, config in enumerate(ACTIVATION_CONFIGS):
        ax = axes[i]

        # Initialize relaxation
        relaxation = config.relaxation_class(**config.init_kwargs)

        # Get bounds
        lb = torch.tensor([lb_val])
        ub = torch.tensor([ub_val])
        alpha_L, beta_L, alpha_U, beta_U = relaxation.relax_activation(lb, ub)

        alpha_L_val = alpha_L.item()
        beta_L_val = beta_L.item()
        alpha_U_val = alpha_U.item()
        beta_U_val = beta_U.item()

        # Create plot range
        x_plot = np.linspace(lb_val - 0.5, ub_val + 0.5, 1000)

        # Compute functions
        y_activation = config.activation_func(x_plot)
        y_lower = alpha_L_val * x_plot + beta_L_val
        y_upper = alpha_U_val * x_plot + beta_U_val

        # Plot
        ax.plot(x_plot, y_activation, "b-", linewidth=3, label=f"{config.name}(x)")
        ax.plot(
            x_plot,
            y_lower,
            "r--",
            linewidth=2,
            label=f"Lower: {alpha_L_val:.3f}x + {beta_L_val:.3f}",
        )
        ax.plot(
            x_plot,
            y_upper,
            "g--",
            linewidth=2,
            label=f"Upper: {alpha_U_val:.3f}x + {beta_U_val:.3f}",
        )

        # Mark the interval
        ax.axvline(lb_val, color="k", linestyle=":", alpha=0.7)
        ax.axvline(ub_val, color="k", linestyle=":", alpha=0.7)
        ax.fill_betweenx(ax.get_ylim(), lb_val, ub_val, alpha=0.1, color="gray")

        # Highlight relaxation
        x_interval = np.linspace(lb_val, ub_val, 100)
        y_lower_interval = alpha_L_val * x_interval + beta_L_val
        y_upper_interval = alpha_U_val * x_interval + beta_U_val

        ax.fill_between(
            x_interval,
            y_lower_interval,
            y_upper_interval,
            alpha=0.2,
            color="yellow",
        )

        ax.set_title(f"{config.name} on [{lb_val}, {ub_val}]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.suptitle("Activation Function Relaxations Comparison", fontsize=16)
    plt.tight_layout()

    # Save comparison plot
    test_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(test_dir, "activation_relaxation_plots")
    os.makedirs(plot_dir, exist_ok=True)
    comparison_path = os.path.join(plot_dir, "activation_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot to {comparison_path}")


if __name__ == "__main__":
    # Run comprehensive tests
    test_suite = TestActivationRelaxations()
    print("Running comprehensive activation relaxation tests...")

    try:
        print("\nRunning endpoint tests...")
        test_suite.test_activation_bounds_at_endpoints()
        print("✓ All endpoint tests passed!")

        print("\nRunning derivative relaxation tests...")
        derivative_results = test_suite.test_activation_derivative_soundness()
        print("✓ All derivative tests completed!")

        print("\nRunning specific tanh derivative tests...")
        test_suite.test_tanh_derivative_specific_cases()
        print("✓ Tanh derivative specific tests passed!")

        print("\nRunning specific sigmoid derivative tests...")
        test_suite.test_sigmoid_derivative_specific_cases()
        print("✓ Sigmoid derivative specific tests passed!")

    except Exception as e:
        print(f"\n⚠ Some tests failed: {e}")
        print("Continuing with visualization generation...")

    print("\nGenerating visualizations...")
    # Run visualization tests
    test_instance = TestActivationRelaxations()
    test_instance.test_generate_comprehensive_visualizations()
    test_instance.test_generate_derivative_visualizations()
    create_comparison_plot()

    print("\n🎉 All tests completed and visualizations created!")
