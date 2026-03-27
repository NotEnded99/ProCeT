"""
Test suite for linear_derivative_bounds.py

This module provides comprehensive testing for the CrownPartialLinearization class,
which implements the ∂-CROWN method for computing partial derivative bounds.

The test suite covers:

1. **compute_network_bounds method testing**:
   - Tests with hyperrectangular regions (grid-based domains)
   - Tests with simplicial regions (triangular/tetrahedral domains)
   - Validates that bounds are computed for all network layers
   - Ensures bounds are well-formed (finite, lower <= upper)

2. **compute_partial_derivative_bounds method testing**:
   - Tests partial derivative bounds computation for various network architectures
   - Tests different activation functions (ReLU, Tanh, Sigmoid, LeakyReLU)
   - Tests different input/output index combinations
   - Validates output format and tensor shapes

3. **Empirical validation**:
   - Verifies that computed bounds are sound (contain true derivative values)
   - Samples random points within regions to compute true derivatives
   - Compares bounds with empirical ranges from sampling
   - Tests both hyperrectangular and simplicial regions

4. **Edge cases and error handling**:
   - Single layer networks (purely linear)
   - Invalid input/output indices
   - Very small regions (numerical stability)
   - Unsupported region types (proper error messages)

The tests use pytest fixtures to parameterize over:
- Different activation functions
- Various network architectures (small to large)
- Multiple region specifications
- Different output/input index combinations

All tests focus on empirical validation to ensure the mathematical soundness
of the bounds computation, which is critical for verification applications.
"""

import pytest
import numpy as np
import torch
from typing import Tuple, List

from lbp_neural_cbf.linearization import CrownPartialLinearization
from lbp_neural_cbf.regions import HyperrectangularRegion, SimplicialRegion

# Set PyTorch to use float64 by default for maximum numerical precision
torch.set_default_dtype(torch.float64)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(params=[torch.float64])
def dtype(request):
    """Fixture providing different data types."""
    return request.param


@pytest.fixture(
    params=[
        ("ReLU", torch.nn.ReLU()),
        ("Tanh", torch.nn.Tanh()),
        ("Sigmoid", torch.nn.Sigmoid()),
        ("LeakyReLU", torch.nn.LeakyReLU(negative_slope=0.01)),
    ]
)
def activation_function(request):
    """Fixture providing different activation functions."""
    return request.param


@pytest.fixture(
    params=[
        # Simple networks
        {"input_dim": 2, "hidden_dims": [3], "output_dim": 1},
        {"input_dim": 3, "hidden_dims": [4, 5], "output_dim": 2},
        # Medium networks
        {"input_dim": 4, "hidden_dims": [8, 6], "output_dim": 3},
        # Larger networks
        {"input_dim": 5, "hidden_dims": [10, 8, 6], "output_dim": 2},
    ]
)
def network_architecture(request):
    """Fixture providing different network architectures."""
    return request.param


@pytest.fixture
def hyperrectangular_regions():
    """Fixture providing test hyperrectangular regions."""
    return [
        # ===== 2D REGIONS =====
        # Basic 2D regions around origin
        {"center": np.array([0.0, 0.0], dtype=np.float64), "radius": np.array([0.5, 0.5], dtype=np.float64)},
        {"center": np.array([0.0, 0.0], dtype=np.float64), "radius": np.array([1.0, 1.0], dtype=np.float64)},
        {"center": np.array([0.0, 0.0], dtype=np.float64), "radius": np.array([0.1, 0.1], dtype=np.float64)},
        # Translated 2D regions
        {"center": np.array([1.0, -0.5], dtype=np.float64), "radius": np.array([0.3, 0.8], dtype=np.float64)},
        {"center": np.array([-2.0, 3.0], dtype=np.float64), "radius": np.array([0.4, 0.2], dtype=np.float64)},
        {"center": np.array([5.0, -1.5], dtype=np.float64), "radius": np.array([0.7, 0.6], dtype=np.float64)},
        {"center": np.array([-0.8, -0.3], dtype=np.float64), "radius": np.array([0.2, 0.9], dtype=np.float64)},
        # Asymmetric 2D regions
        {"center": np.array([0.5, 0.2], dtype=np.float64), "radius": np.array([2.0, 0.1], dtype=np.float64)},  # Very wide, thin
        {"center": np.array([-0.3, 0.7], dtype=np.float64), "radius": np.array([0.05, 1.5], dtype=np.float64)},  # Very tall, narrow
        {"center": np.array([1.2, -0.4], dtype=np.float64), "radius": np.array([0.01, 0.01], dtype=np.float64)},  # Very small
        {"center": np.array([0.0, 0.0], dtype=np.float64), "radius": np.array([3.0, 0.2], dtype=np.float64)},  # Large width, small height
        # Edge cases - 2D
        {"center": np.array([10.0, -10.0], dtype=np.float64), "radius": np.array([0.1, 0.1], dtype=np.float64)},  # Far from origin
        {"center": np.array([0.001, 0.001], dtype=np.float64), "radius": np.array([0.0001, 0.0001], dtype=np.float64)},  # Tiny region
        {"center": np.array([0.0, 0.0], dtype=np.float64), "radius": np.array([5.0, 5.0], dtype=np.float64)},  # Large region
        # ===== 3D REGIONS =====
        # Basic 3D regions around origin
        {"center": np.array([0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.2, 0.4, 0.6], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([1.0, 1.0, 1.0], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.5, 0.5, 0.5], dtype=np.float64)},
        # Translated 3D regions with varied scales
        {"center": np.array([0.5, -0.3, 1.2], dtype=np.float64), "radius": np.array([0.1, 0.25, 0.15], dtype=np.float64)},
        {"center": np.array([-1.0, 2.0, -0.5], dtype=np.float64), "radius": np.array([0.3, 0.2, 0.8], dtype=np.float64)},
        {"center": np.array([2.5, 1.0, -3.0], dtype=np.float64), "radius": np.array([0.4, 0.6, 0.2], dtype=np.float64)},
        {"center": np.array([-0.2, -1.5, 0.8], dtype=np.float64), "radius": np.array([0.7, 0.1, 0.5], dtype=np.float64)},
        {"center": np.array([7.23, -4.12, 0.891], dtype=np.float64), "radius": np.array([0.023, 0.156, 0.0089], dtype=np.float64)},
        {"center": np.array([-3.5, 2.7, -4.3], dtype=np.float64), "radius": np.array([0.35, 0.27, 0.43], dtype=np.float64)},  # Fixed: was -12.5, 8.7, -15.3
        # Asymmetric 3D regions (extreme aspect ratios)
        {"center": np.array([0.1, 0.2, 0.3], dtype=np.float64), "radius": np.array([2.0, 0.1, 0.05], dtype=np.float64)},  # Elongated in x
        {"center": np.array([0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.1, 3.0, 0.1], dtype=np.float64)},  # Elongated in y
        {"center": np.array([0.5, -0.2, 0.8], dtype=np.float64), "radius": np.array([0.05, 0.05, 2.5], dtype=np.float64)},  # Elongated in z
        {"center": np.array([1.0, 0.0, -1.0], dtype=np.float64), "radius": np.array([0.001, 0.001, 0.001], dtype=np.float64)},  # Very small
        {"center": np.array([-0.456, 1.234, -0.789], dtype=np.float64), "radius": np.array([4.5, 0.0012, 0.0034], dtype=np.float64)},  # Extreme x elongation
        {"center": np.array([2.1, -3.4, 5.6], dtype=np.float64), "radius": np.array([0.0067, 0.00023, 6.78], dtype=np.float64)},  # Extreme z elongation
        # Edge cases - 3D
        {"center": np.array([15.0, -20.0, 25.0], dtype=np.float64), "radius": np.array([0.01, 0.02, 0.015], dtype=np.float64)},  # Far from origin
        {"center": np.array([0.00001, -0.00002, 0.00003], dtype=np.float64), "radius": np.array([1e-6, 2e-6, 1.5e-6], dtype=np.float64)},  # Tiny region
        {"center": np.array([0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([10.0, 8.0, 12.0], dtype=np.float64)},  # Large region
        # ===== 4D REGIONS =====
        # Basic 4D regions around origin
        {"center": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.5, 0.3, 0.8, 0.2], dtype=np.float64)},
        {"center": np.array([0.12, -0.34, 0.56, -0.78], dtype=np.float64), "radius": np.array([1.23, 0.0045, 0.167, 0.0089], dtype=np.float64)},
        {"center": np.array([-1.414, 2.718, -3.142, 1.618], dtype=np.float64), "radius": np.array([0.0141, 0.00271, 0.314, 0.00161], dtype=np.float64)},
        {"center": np.array([5.0, -7.5, 2.25, -12.125], dtype=np.float64), "radius": np.array([0.005, 0.075, 0.0225, 0.01212], dtype=np.float64)},
        # Translated 4D regions with varied characteristics
        {"center": np.array([2.0, -1.5, 0.3, 2.5], dtype=np.float64), "radius": np.array([0.4, 0.2, 0.6, 0.3], dtype=np.float64)},
        {"center": np.array([-0.8, 3.0, -2.0, 1.0], dtype=np.float64), "radius": np.array([0.1, 0.5, 0.2, 0.8], dtype=np.float64)},
        {"center": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64), "radius": np.array([0.05, 0.1, 0.15, 0.2], dtype=np.float64)},
        {"center": np.array([8.67, -5.43, 12.1, -9.89], dtype=np.float64), "radius": np.array([0.0867, 0.0543, 0.012, 0.098], dtype=np.float64)},
        {"center": np.array([-15.2, 6.78, -3.45, 20.1], dtype=np.float64), "radius": np.array([0.152, 0.00678, 0.345, 0.00201], dtype=np.float64)},
        {"center": np.array([0.987, -0.654, 0.321, -0.123], dtype=np.float64), "radius": np.array([0.0098, 0.654, 0.00321, 0.123], dtype=np.float64)},
        # Asymmetric 4D regions (extreme aspect ratios)
        {"center": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([3.0, 0.05, 0.05, 0.05], dtype=np.float64)},  # Long in dim 0
        {"center": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64), "radius": np.array([0.1, 2.0, 0.1, 0.1], dtype=np.float64)},  # Long in dim 1
        {"center": np.array([1.0, 0.0, -1.0, 2.0], dtype=np.float64), "radius": np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)},  # Very small
        {"center": np.array([-2.5, 1.8, -0.9, 3.7], dtype=np.float64), "radius": np.array([0.00025, 0.00018, 4.5, 0.0037], dtype=np.float64)},  # Long in dim 2
        {"center": np.array([1.23, -4.56, 7.89, -0.12], dtype=np.float64), "radius": np.array([0.123, 0.456, 0.00789, 5.67], dtype=np.float64)},  # Long in dim 3
        # Edge cases - 4D
        {"center": np.array([25.0, -18.0, 30.0, -42.0], dtype=np.float64), "radius": np.array([0.025, 0.018, 0.003, 0.042], dtype=np.float64)},  # Far from origin
        {"center": np.array([1e-5, -2e-5, 3e-5, -4e-5], dtype=np.float64), "radius": np.array([1e-7, 2e-7, 3e-7, 4e-7], dtype=np.float64)},  # Tiny region
        {"center": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([15.0, 12.0, 20.0, 8.0], dtype=np.float64)},  # Large region
        # ===== 5D REGIONS =====
        # Basic 5D regions
        {"center": np.zeros(5, dtype=np.float64), "radius": np.array([0.1, 0.2, 0.3, 0.15, 0.25], dtype=np.float64)},
        {"center": np.array([0.123, -0.456, 0.789, -0.012, 0.345], dtype=np.float64), "radius": np.array([1.23, 0.00456, 0.0789, 0.012, 0.00345], dtype=np.float64)},
        {"center": np.array([-2.718, 3.14159, -1.414, 1.618, -0.577], dtype=np.float64), "radius": np.array([0.02718, 0.000314, 0.1414, 0.001618, 0.577], dtype=np.float64)},
        {"center": np.array([6.022, -9.109, 4.669, -2.998, 8.314], dtype=np.float64), "radius": np.array([0.006022, 0.09109, 0.004669, 0.2998, 0.008314], dtype=np.float64)},
        # Translated 5D regions with diverse characteristics
        {"center": np.array([2.0, -1.0, 0.5, 3.0, -2.5], dtype=np.float64), "radius": np.array([0.3, 0.2, 0.4, 0.1, 0.5], dtype=np.float64)},
        {"center": np.array([-1.5, 2.5, -0.8, 1.2, 0.0], dtype=np.float64), "radius": np.array([0.15, 0.25, 0.35, 0.05, 0.45], dtype=np.float64)},
        {"center": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64), "radius": np.array([0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float64)},
        {"center": np.array([7.89, -12.34, 5.67, -8.90, 15.43], dtype=np.float64), "radius": np.array([0.0789, 0.1234, 0.0567, 0.89, 0.001543], dtype=np.float64)},
        {"center": np.array([-25.1, 18.7, -9.3, 32.6, -14.8], dtype=np.float64), "radius": np.array([0.251, 0.00187, 0.93, 0.0326, 0.148], dtype=np.float64)},
        {"center": np.array([0.707, -0.866, 0.500, -0.259, 0.966], dtype=np.float64), "radius": np.array([0.0707, 0.866, 0.005, 0.259, 0.00966], dtype=np.float64)},
        # Diverse scales and positions
        {"center": np.array([5.0, -3.0, 0.0, 2.0, -1.0], dtype=np.float64), "radius": np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)},
        # Asymmetric 5D regions (extreme aspect ratios)
        {"center": np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([4.0, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)},  # Elongated in dim 0
        {"center": np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.1, 3.0, 0.1, 0.1, 0.1], dtype=np.float64)},  # Elongated in dim 1
        {"center": np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64), "radius": np.array([0.001, 0.001, 0.001, 0.001, 0.001], dtype=np.float64)},  # Tiny
        {"center": np.array([-1.2, 3.4, -5.6, 7.8, -9.0], dtype=np.float64), "radius": np.array([0.12, 0.0034, 5.67, 0.0078, 0.09], dtype=np.float64)},  # Elongated in dim 2
        {"center": np.array([2.468, -1.357, 9.024, -6.813, 4.579], dtype=np.float64), "radius": np.array([0.002468, 0.1357, 0.009024, 6.813, 0.004579], dtype=np.float64)},  # Elongated in dim 3
        {"center": np.array([0.111, -2.222, 3.333, -4.444, 5.555], dtype=np.float64), "radius": np.array([0.111, 0.002222, 0.003333, 0.004444, 7.777], dtype=np.float64)},  # Elongated in dim 4
        # Edge cases - 5D
        {"center": np.array([50.0, -35.0, 42.0, -28.0, 63.0], dtype=np.float64), "radius": np.array([0.05, 0.035, 0.0042, 0.28, 0.0063], dtype=np.float64)},  # Far from origin
        {"center": np.array([1e-6, -2e-6, 3e-6, -4e-6, 5e-6], dtype=np.float64), "radius": np.array([1e-8, 2e-8, 3e-8, 4e-8, 5e-8], dtype=np.float64)},  # Tiny region
        {"center": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([25.0, 18.0, 30.0, 12.0, 22.0], dtype=np.float64)},  # Large region
        # ===== MIXED DIMENSIONAL SPECIAL CASES =====
        # Unit hypercubes in different dimensions
        {"center": np.array([0.0, 0.0], dtype=np.float64), "radius": np.array([1.0, 1.0], dtype=np.float64)},  # 2D unit square
        {"center": np.array([0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([1.0, 1.0, 1.0], dtype=np.float64)},  # 3D unit cube
        {"center": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)},  # 4D unit hypercube
        {"center": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)},  # 5D unit hypercube
        # Regions with one very small dimension (near-degenerate cases)
        {"center": np.array([0.0, 0.0], dtype=np.float64), "radius": np.array([1.0, 1e-8], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.5, 1e-6, 0.5], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.3, 0.3, 1e-7, 0.3], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.2, 0.2, 0.2, 1e-5, 0.2], dtype=np.float64)},
        # Regions with one very large dimension
        {"center": np.array([0.0, 0.0], dtype=np.float64), "radius": np.array([10.0, 0.1], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.2, 8.0, 0.2], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([0.1, 0.1, 0.1, 15.0], dtype=np.float64)},
        {"center": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64), "radius": np.array([12.0, 0.15, 0.15, 0.15, 0.15], dtype=np.float64)},
        # Negative centers with various scales
        {"center": np.array([-5.0, -5.0], dtype=np.float64), "radius": np.array([0.5, 0.5], dtype=np.float64)},
        {"center": np.array([-2.0, -1.0, -3.0], dtype=np.float64), "radius": np.array([0.2, 0.3, 0.4], dtype=np.float64)},
        {"center": np.array([-1.0, -2.0, -0.5, -3.0], dtype=np.float64), "radius": np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)},
        {"center": np.array([-0.5, -1.5, -2.5, -0.2, -1.8], dtype=np.float64), "radius": np.array([0.05, 0.15, 0.25, 0.1, 0.2], dtype=np.float64)},
        # Mixed positive/negative centers
        {"center": np.array([5.0, -5.0], dtype=np.float64), "radius": np.array([0.3, 0.3], dtype=np.float64)},
        {"center": np.array([2.0, -1.0, 3.0], dtype=np.float64), "radius": np.array([0.4, 0.2, 0.6], dtype=np.float64)},
        {"center": np.array([1.5, -2.5, 0.5, -1.0], dtype=np.float64), "radius": np.array([0.3, 0.4, 0.1, 0.2], dtype=np.float64)},
        {"center": np.array([3.0, -1.0, 2.0, -3.0, 1.5], dtype=np.float64), "radius": np.array([0.2, 0.3, 0.15, 0.4, 0.25], dtype=np.float64)},
    ]


@pytest.fixture
def simplicial_regions():
    """Fixture providing test simplicial regions."""
    return [
        # 2D triangles
        {"vertices": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])},
        {"vertices": np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]])},
        # 3D tetrahedra
        {"vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
        {"vertices": np.array([[-0.2, -0.2, -0.2], [0.8, -0.2, -0.2], [-0.2, 0.8, -0.2], [-0.2, -0.2, 0.8]])},
        # 4D simplices (5 vertices for 4D simplex)
        {
            "vertices": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.4, 0.0, 0.0],
                    [0.0, 0.0, 0.6, 0.0],
                    [0.0, 0.0, 0.0, 0.3],
                ]
            )
        },
        {
            "vertices": np.array(
                [
                    [-0.1, -0.1, -0.1, -0.1],
                    [0.4, -0.1, -0.1, -0.1],
                    [-0.1, 0.3, -0.1, -0.1],
                    [-0.1, -0.1, 0.5, -0.1],
                    [-0.1, -0.1, -0.1, 0.2],
                ]
            )
        },
        # 5D simplex
        {
            "vertices": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.3, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.6],
                ]
            )
        },
    ]


def generate_all_output_input_combinations(input_dim: int, output_dim: int):
    """Generate all possible (output_idx, input_idx) combinations for given dimensions."""
    combinations = []
    for output_idx in range(output_dim):
        for input_idx in range(input_dim):
            combinations.append((output_idx, input_idx))
    return combinations


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_test_network(input_dim: int, hidden_dims: List[int], output_dim: int, activation: str = "ReLU", seed: int = 42, dtype=torch.float64) -> torch.nn.Sequential:
    """Create a test network with specified architecture."""
    torch.manual_seed(seed)

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
        if activation == "ReLU":
            layers.append(torch.nn.ReLU())
        elif activation == "Tanh":
            layers.append(torch.nn.Tanh())
        elif activation == "Sigmoid":
            layers.append(torch.nn.Sigmoid())
        elif activation == "LeakyReLU":
            layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
        prev_dim = hidden_dim

    # Output layer (no activation)
    layers.append(torch.nn.Linear(prev_dim, output_dim))

    network = torch.nn.Sequential(*layers)

    # Initialize weights and convert to specified dtype
    with torch.no_grad():
        for layer in network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0.01)

    return network.to(dtype)


def compute_true_derivative(network: torch.nn.Sequential, x: np.ndarray, output_idx: int, input_idx: int) -> float:
    """Compute true partial derivative using PyTorch autodiff."""
    # Get the network's dtype from the first linear layer
    network_dtype = next(network.parameters()).dtype
    x_tensor = torch.tensor(x, dtype=network_dtype, requires_grad=True)
    y = network(x_tensor.unsqueeze(0)).squeeze(0)
    y[output_idx].backward()
    return x_tensor.grad[input_idx].item()


def sample_points_in_region(region, num_samples: int, seed: int = 42) -> np.ndarray:
    """Sample random points within a region."""
    np.random.seed(seed)

    if isinstance(region, HyperrectangularRegion):
        # Sample uniformly within hyperrectangle
        samples = []
        for _ in range(num_samples):
            random_offset = np.random.uniform(-1, 1, size=len(region.radius_vec))
            sample_point = region.center_point + random_offset * region.radius_vec
            samples.append(sample_point)
        return np.array(samples)

    elif isinstance(region, SimplicialRegion):
        # Sample uniformly within simplex using Dirichlet distribution
        samples = []
        n_vertices = len(region.vertices)

        for _ in range(num_samples):
            # Generate Dirichlet weights for barycentric coordinates
            alphas = np.ones(n_vertices)
            weights = np.random.dirichlet(alphas)

            # Compute barycentric combination
            sample_point = np.sum([w * v for w, v in zip(weights, region.vertices)], axis=0)
            samples.append(sample_point)
        return np.array(samples)

    else:
        raise ValueError(f"Unsupported region type: {type(region)}")


def validate_derivative_bounds_soundness(
    network: torch.nn.Sequential,
    region,
    output_idx: int,
    input_idx: int,
    bounds_lower: Tuple,
    bounds_upper: Tuple,
    num_samples: int = 100,
) -> dict:
    """Validate that computed partial derivative bounds are sound by sampling."""

    # Sample points in the region
    sample_points = sample_points_in_region(region, num_samples)

    # Compute true derivatives at sample points
    true_derivatives = []
    for point in sample_points:
        derivative = compute_true_derivative(network, point, output_idx, input_idx)
        true_derivatives.append(derivative)

    if len(true_derivatives) == 0:
        return {"valid": False, "error": "No valid derivative computations"}

    true_derivatives = np.array(true_derivatives)
    empirical_min = np.min(true_derivatives)
    empirical_max = np.max(true_derivatives)

    # Extract bounds (assuming affine bounds format)
    A_L, b_L = bounds_lower
    A_U, b_U = bounds_upper

    # Get the network's dtype to match tensors
    network_dtype = next(network.parameters()).dtype

    # Evaluate bounds over the region
    # For hyperrectangular regions, use center ± radius approach
    if isinstance(region, HyperrectangularRegion):
        center_torch = torch.tensor(region.center_point, dtype=network_dtype)
        radius_torch = torch.tensor(region.radius_vec, dtype=network_dtype)

        # Lower bound evaluation
        bounds_min = (A_L @ center_torch + b_L) - (torch.abs(A_L) @ radius_torch)
        bounds_max = (A_U @ center_torch + b_U) + (torch.abs(A_U) @ radius_torch)

    elif isinstance(region, SimplicialRegion):
        # For simplicial regions, evaluate bounds at all vertices and take extrema
        vertices_torch = torch.tensor(region.vertices, dtype=network_dtype)

        lower_values = A_L @ vertices_torch.T + b_L
        upper_values = A_U @ vertices_torch.T + b_U

        bounds_min = torch.min(lower_values)
        bounds_max = torch.max(upper_values)

    # Convert to numpy for comparison
    bounds_min = bounds_min.item() if hasattr(bounds_min, "item") else bounds_min
    bounds_max = bounds_max.item() if hasattr(bounds_max, "item") else bounds_max

    # Check soundness (region-wide bounds)
    lower_sound = bounds_min <= empirical_min + 1e-6  # Small tolerance for numerical errors
    upper_sound = empirical_max <= bounds_max + 1e-6

    # Point-wise validation: check that for each point, A_L @ point + b_L <= derivative <= A_U @ point + b_U
    pointwise_violations = []
    for i, (point, derivative) in enumerate(zip(sample_points, true_derivatives)):
        point_torch = torch.tensor(point, dtype=network_dtype)

        # Compute bounds at this specific point
        lower_bound_at_point = (A_L @ point_torch + b_L).item()
        upper_bound_at_point = (A_U @ point_torch + b_U).item()

        # Check if bounds hold at this point
        lower_violation = derivative < lower_bound_at_point - 1e-6
        upper_violation = derivative > upper_bound_at_point + 1e-6

        if lower_violation or upper_violation:
            pointwise_violations.append(
                {
                    "point_index": i,
                    "point": point,
                    "derivative": derivative,
                    "lower_bound": lower_bound_at_point,
                    "upper_bound": upper_bound_at_point,
                    "lower_violation": lower_violation,
                    "upper_violation": upper_violation,
                }
            )

    # Overall point-wise soundness
    pointwise_sound = len(pointwise_violations) == 0

    return {
        "valid": True,
        "lower_sound": lower_sound,
        "upper_sound": upper_sound,
        "pointwise_sound": pointwise_sound,
        "empirical_range": (empirical_min, empirical_max),
        "bounds_range": (bounds_min, bounds_max),
        "num_samples": len(true_derivatives),
        "pointwise_violations": (pointwise_violations[:5] if pointwise_violations else []),  # Report up to 5 violations for debugging
    }


def validate_network_bounds_soundness(network: torch.nn.Sequential, region, crown_linearization, num_samples: int = 100) -> dict:
    """Validate that computed network bounds are sound by sampling."""

    # Get the network's dtype from the first linear layer
    network_dtype = next(network.parameters()).dtype

    # Sample points in the region
    sample_points = sample_points_in_region(region, num_samples)

    # Compute layer-by-layer activations for all sample points
    layer_activations = {"pre": {}, "post": {}}  # Store activations for each layer

    for point in sample_points:
        x_tensor = torch.tensor(point, dtype=network_dtype)

        # Forward pass through each layer manually to capture intermediate activations
        current_input = x_tensor.unsqueeze(0)
        layer_idx = 0

        for i, layer in enumerate(network):
            if isinstance(layer, torch.nn.Linear):
                # Pre-activation (linear transformation)
                pre_activation = layer(current_input).squeeze(0)

                if layer_idx not in layer_activations["pre"]:
                    layer_activations["pre"][layer_idx] = []
                layer_activations["pre"][layer_idx].append(pre_activation.detach().numpy())

                current_input = pre_activation.unsqueeze(0)

            elif hasattr(layer, "forward"):  # Activation function
                # Post-activation (after activation function)
                post_activation = layer(current_input).squeeze(0)

                if layer_idx not in layer_activations["post"]:
                    layer_activations["post"][layer_idx] = []
                layer_activations["post"][layer_idx].append(post_activation.detach().numpy())

                current_input = post_activation.unsqueeze(0)
                layer_idx += 1

        # Handle final layer if it has no activation
        if layer_idx not in layer_activations["post"]:
            # Final output is the same as pre-activation of last layer
            final_output = current_input.squeeze(0)
            if layer_idx not in layer_activations["post"]:
                layer_activations["post"][layer_idx] = []
            layer_activations["post"][layer_idx].append(final_output.detach().numpy())

    if len(layer_activations["pre"]) == 0:
        return {"valid": False, "error": "No valid network evaluations"}

    # Convert lists to numpy arrays
    for activation_type in ["pre", "post"]:
        for layer_idx in layer_activations[activation_type]:
            layer_activations[activation_type][layer_idx] = np.array(layer_activations[activation_type][layer_idx])

    # Check each layer's bounds
    validation_results = {}
    num_layers = len(crown_linearization.fc_layers)

    for layer_idx in range(num_layers):
        pre_act_key = f"layer_{layer_idx}_pre_act_bounds"
        post_act_key = f"layer_{layer_idx}_post_act_bounds"

        if pre_act_key not in crown_linearization.forward_bounds:
            continue

        pre_bounds = crown_linearization.forward_bounds[pre_act_key]
        post_bounds = crown_linearization.forward_bounds[post_act_key]

        # Validate pre-activation bounds
        if layer_idx in layer_activations["pre"]:
            pre_activations = layer_activations["pre"][layer_idx]
            # Access batch index 0 for single-region batches
            pre_lb = pre_bounds["lb"][0].detach().numpy() if pre_bounds["lb"].dim() > 1 else pre_bounds["lb"].detach().numpy()
            pre_ub = pre_bounds["ub"][0].detach().numpy() if pre_bounds["ub"].dim() > 1 else pre_bounds["ub"].detach().numpy()

            empirical_pre_min = np.min(pre_activations, axis=0)
            empirical_pre_max = np.max(pre_activations, axis=0)

            pre_lower_sound = np.all(pre_lb <= empirical_pre_min + 1e-6)
            pre_upper_sound = np.all(empirical_pre_max <= pre_ub + 1e-6)

            # Point-wise validation for pre-activation bounds
            pre_pointwise_violations = []
            for i, activation in enumerate(pre_activations):
                # Check if bounds hold at each activation value
                lower_violations = activation < pre_lb - 1e-6
                upper_violations = activation > pre_ub + 1e-6

                if np.any(lower_violations) or np.any(upper_violations):
                    pre_pointwise_violations.append(
                        {
                            "point_index": i,
                            "activation": activation,
                            "lower_bound": pre_lb,
                            "upper_bound": pre_ub,
                            "lower_violations": np.where(lower_violations)[0].tolist(),
                            "upper_violations": np.where(upper_violations)[0].tolist(),
                        }
                    )

            pre_pointwise_sound = len(pre_pointwise_violations) == 0

            validation_results[f"layer_{layer_idx}_pre_activation"] = {
                "lower_sound": pre_lower_sound,
                "upper_sound": pre_upper_sound,
                "pointwise_sound": pre_pointwise_sound,
                "empirical_range": (empirical_pre_min, empirical_pre_max),
                "bounds_range": (pre_lb, pre_ub),
                "layer_idx": layer_idx,
                "activation_type": "pre",
                "pointwise_violations": (pre_pointwise_violations[:3] if pre_pointwise_violations else []),  # Report up to 3 violations
            }

        # Validate post-activation bounds
        if layer_idx in layer_activations["post"]:
            post_activations = layer_activations["post"][layer_idx]
            # Access batch index 0 for single-region batches
            post_lb = post_bounds["lb"][0].detach().numpy() if post_bounds["lb"].dim() > 1 else post_bounds["lb"].detach().numpy()
            post_ub = post_bounds["ub"][0].detach().numpy() if post_bounds["ub"].dim() > 1 else post_bounds["ub"].detach().numpy()

            empirical_post_min = np.min(post_activations, axis=0)
            empirical_post_max = np.max(post_activations, axis=0)

            post_lower_sound = np.all(post_lb <= empirical_post_min + 1e-6)
            post_upper_sound = np.all(empirical_post_max <= post_ub + 1e-6)

            # Point-wise validation for post-activation bounds
            post_pointwise_violations = []
            for i, activation in enumerate(post_activations):
                # Check if bounds hold at each activation value
                lower_violations = activation < post_lb - 1e-6
                upper_violations = activation > post_ub + 1e-6

                if np.any(lower_violations) or np.any(upper_violations):
                    post_pointwise_violations.append(
                        {
                            "point_index": i,
                            "activation": activation,
                            "lower_bound": post_lb,
                            "upper_bound": post_ub,
                            "lower_violations": np.where(lower_violations)[0].tolist(),
                            "upper_violations": np.where(upper_violations)[0].tolist(),
                        }
                    )

            post_pointwise_sound = len(post_pointwise_violations) == 0

            validation_results[f"layer_{layer_idx}_post_activation"] = {
                "lower_sound": post_lower_sound,
                "upper_sound": post_upper_sound,
                "pointwise_sound": post_pointwise_sound,
                "empirical_range": (empirical_post_min, empirical_post_max),
                "bounds_range": (post_lb, post_ub),
                "layer_idx": layer_idx,
                "activation_type": "post",
                "pointwise_violations": (post_pointwise_violations[:3] if post_pointwise_violations else []),  # Report up to 3 violations
            }

    # Also validate final network outputs (using standard forward pass)
    network_outputs = []
    for point in sample_points:
        x_tensor = torch.tensor(point, dtype=network_dtype)
        y = network(x_tensor.unsqueeze(0)).squeeze(0)
        network_outputs.append(y.detach().numpy())

    if len(network_outputs) > 0:
        network_outputs = np.array(network_outputs)  # Shape: (num_samples, output_dim)

        # Get final layer bounds (last layer's post-activation bounds)
        final_layer_idx = num_layers - 1
        final_post_key = f"layer_{final_layer_idx}_post_act_bounds"

        if final_post_key in crown_linearization.forward_bounds:
            final_bounds = crown_linearization.forward_bounds[final_post_key]
            # Access batch index 0 for single-region batches
            final_lb = final_bounds["lb"][0].detach().numpy() if final_bounds["lb"].dim() > 1 else final_bounds["lb"].detach().numpy()
            final_ub = final_bounds["ub"][0].detach().numpy() if final_bounds["ub"].dim() > 1 else final_bounds["ub"].detach().numpy()

            empirical_final_min = np.min(network_outputs, axis=0)
            empirical_final_max = np.max(network_outputs, axis=0)

            final_lower_sound = np.all(final_lb <= empirical_final_min + 1e-6)
            final_upper_sound = np.all(empirical_final_max <= final_ub + 1e-6)

            # Point-wise validation for final network outputs
            final_pointwise_violations = []
            for i, output in enumerate(network_outputs):
                # Check if bounds hold at each output value
                lower_violations = output < final_lb - 1e-6
                upper_violations = output > final_ub + 1e-6

                if np.any(lower_violations) or np.any(upper_violations):
                    final_pointwise_violations.append(
                        {
                            "point_index": i,
                            "output": output,
                            "lower_bound": final_lb,
                            "upper_bound": final_ub,
                            "lower_violations": np.where(lower_violations)[0].tolist(),
                            "upper_violations": np.where(upper_violations)[0].tolist(),
                        }
                    )

            final_pointwise_sound = len(final_pointwise_violations) == 0

            validation_results["final_network_output"] = {
                "lower_sound": final_lower_sound,
                "upper_sound": final_upper_sound,
                "pointwise_sound": final_pointwise_sound,
                "empirical_range": (empirical_final_min, empirical_final_max),
                "bounds_range": (final_lb, final_ub),
                "layer_idx": final_layer_idx,
                "activation_type": "final_output",
                "pointwise_violations": (final_pointwise_violations[:3] if final_pointwise_violations else []),  # Report up to 3 violations
            }

    # Overall validation result
    all_valid = all(result["lower_sound"] and result["upper_sound"] for result in validation_results.values())
    all_pointwise_valid = all(result.get("pointwise_sound", True) for result in validation_results.values())

    return {
        "valid": True,
        "overall_sound": all_valid,
        "overall_pointwise_sound": all_pointwise_valid,
        "layer_results": validation_results,
        "num_samples": num_samples,
    }


# =============================================================================
# TESTS FOR compute_network_bounds
# =============================================================================


class TestComputeNetworkBounds:
    """Test the compute_network_bounds method."""

    def test_network_bounds_hyperrectangular(self, activation_function, network_architecture, hyperrectangular_regions, dtype):
        """Test compute_network_bounds with hyperrectangular regions and empirical validation."""
        activation_name, _ = activation_function

        # Filter regions to match network input dimension
        input_dim = network_architecture["input_dim"]
        matching_regions = [r for r in hyperrectangular_regions if len(r["center"]) == input_dim]

        if not matching_regions:
            pytest.skip(f"No hyperrectangular regions available for input_dim={input_dim}")

        # Create network
        network = create_test_network(activation=activation_name, dtype=dtype, **network_architecture)
        crown_linearization = CrownPartialLinearization(network, dtype=dtype)

        # Set tolerance based on dtype for numerical precision
        tolerance = 1e-6 if dtype == torch.float32 else 1e-10

        for region_spec in matching_regions:  # Test all matching regions
            region = HyperrectangularRegion(region_spec["center"], region_spec["radius"])

            # Test that compute_network_bounds runs without error (passing as batch)
            crown_linearization.compute_network_bounds([region])

            # Check that forward bounds were computed for all layers
            num_layers = len(crown_linearization.fc_layers)
            for i in range(num_layers):
                pre_act_key = f"layer_{i}_pre_act_bounds"
                post_act_key = f"layer_{i}_post_act_bounds"

                assert pre_act_key in crown_linearization.forward_bounds
                assert post_act_key in crown_linearization.forward_bounds

                # Check that bounds are well-formed (accessing batch index 0)
                pre_bounds = crown_linearization.forward_bounds[pre_act_key]
                post_bounds = crown_linearization.forward_bounds[post_act_key]

                for bounds in [pre_bounds, post_bounds]:
                    assert "lb" in bounds and "ub" in bounds
                    # Access batch dimension [0] for single-region batches
                    lb = bounds["lb"][0] if bounds["lb"].dim() > 1 else bounds["lb"]
                    ub = bounds["ub"][0] if bounds["ub"].dim() > 1 else bounds["ub"]
                    assert torch.all(lb <= ub + tolerance), f"Lower bounds should be <= upper bounds (within tolerance={tolerance})"
                    assert torch.all(torch.isfinite(lb)), "Lower bounds should be finite"
                    assert torch.all(torch.isfinite(ub)), "Upper bounds should be finite"

            # Empirical validation for network bounds
            validation_result = validate_network_bounds_soundness(network, region, crown_linearization, num_samples=50)

            assert validation_result["valid"], f"Network bounds validation failed: {validation_result.get('error', 'Unknown error')}"
            assert validation_result["overall_sound"], f"Network bounds not sound for {activation_name} activation. " f"Layer validation details: {validation_result['layer_results']}"

            # Check individual layer results for more detailed error reporting
            for layer_key, layer_result in validation_result["layer_results"].items():
                assert layer_result["lower_sound"], (
                    f"Lower network bounds not sound for {activation_name} activation, {layer_key}. "
                    f"Empirical range: {layer_result['empirical_range']}, "
                    f"Bounds range: {layer_result['bounds_range']}"
                )
                assert layer_result["upper_sound"], (
                    f"Upper network bounds not sound for {activation_name} activation, {layer_key}. "
                    f"Empirical range: {layer_result['empirical_range']}, "
                    f"Bounds range: {layer_result['bounds_range']}"
                )

    def test_network_bounds_simplicial(self, activation_function, network_architecture, simplicial_regions, dtype):
        """Test compute_network_bounds with simplicial regions and empirical validation."""
        activation_name, _ = activation_function

        # Filter regions to match network input dimension
        input_dim = network_architecture["input_dim"]
        matching_regions = [r for r in simplicial_regions if r["vertices"].shape[1] == input_dim]

        if not matching_regions:
            pytest.skip(f"No simplicial regions available for input_dim={input_dim}")

        # Create network
        network = create_test_network(activation=activation_name, dtype=dtype, **network_architecture)
        crown_linearization = CrownPartialLinearization(network, dtype=dtype)

        # Set tolerance based on dtype for numerical precision
        tolerance = 1e-6 if dtype == torch.float32 else 1e-10

        for region_spec in matching_regions:  # Test all matching regions
            region = SimplicialRegion(region_spec["vertices"])

            # Test that compute_network_bounds runs without error (passing as batch)
            crown_linearization.compute_network_bounds([region])

            # Check that forward bounds were computed for all layers
            num_layers = len(crown_linearization.fc_layers)
            for i in range(num_layers):
                pre_act_key = f"layer_{i}_pre_act_bounds"
                post_act_key = f"layer_{i}_post_act_bounds"

                assert pre_act_key in crown_linearization.forward_bounds
                assert post_act_key in crown_linearization.forward_bounds

                # Check that bounds are well-formed (accessing batch index 0)
                pre_bounds = crown_linearization.forward_bounds[pre_act_key]
                post_bounds = crown_linearization.forward_bounds[post_act_key]

                for bounds in [pre_bounds, post_bounds]:
                    assert "lb" in bounds and "ub" in bounds
                    # Access batch dimension [0] for single-region batches
                    lb = bounds["lb"][0] if bounds["lb"].dim() > 1 else bounds["lb"]
                    ub = bounds["ub"][0] if bounds["ub"].dim() > 1 else bounds["ub"]
                    assert torch.all(lb <= ub + tolerance), f"Lower bounds should be <= upper bounds (within tolerance={tolerance})"
                    assert torch.all(torch.isfinite(lb)), "Lower bounds should be finite"
                    assert torch.all(torch.isfinite(ub)), "Upper bounds should be finite"

            # Empirical validation for network bounds
            validation_result = validate_network_bounds_soundness(network, region, crown_linearization, num_samples=50)

            assert validation_result["valid"], f"Network bounds validation failed: {validation_result.get('error', 'Unknown error')}"
            assert validation_result["overall_sound"], f"Network bounds not sound for {activation_name} activation. " f"Layer validation details: {validation_result['layer_results']}"

            # Check individual layer results for more detailed error reporting
            for layer_key, layer_result in validation_result["layer_results"].items():
                assert layer_result["lower_sound"], (
                    f"Lower network bounds not sound for {activation_name} activation, {layer_key}. "
                    f"Empirical range: {layer_result['empirical_range']}, "
                    f"Bounds range: {layer_result['bounds_range']}"
                )
                assert layer_result["upper_sound"], (
                    f"Upper network bounds not sound for {activation_name} activation, {layer_key}. "
                    f"Empirical range: {layer_result['empirical_range']}, "
                    f"Bounds range: {layer_result['bounds_range']}"
                )


# =============================================================================
# TESTS FOR compute_partial_derivative_bounds
# =============================================================================


class TestComputePartialDerivativeBounds:
    """Test the compute_partial_derivative_bounds method."""

    def test_partial_derivative_bounds_hyperrectangular(self, activation_function, network_architecture, hyperrectangular_regions, dtype):
        """Test compute_partial_derivative_bounds with hyperrectangular regions and empirical validation."""
        activation_name, _ = activation_function

        # Filter regions to match network input dimension
        matching_regions = [r for r in hyperrectangular_regions if len(r["center"]) == network_architecture["input_dim"]]

        if not matching_regions:
            pytest.skip(f"No hyperrectangular regions available for input_dim={network_architecture['input_dim']}")

        # Generate all possible output/input index combinations for this network
        all_combinations = generate_all_output_input_combinations(network_architecture["input_dim"], network_architecture["output_dim"])

        # Create network
        network = create_test_network(activation=activation_name, dtype=dtype, **network_architecture)
        crown_linearization = CrownPartialLinearization(network, dtype=dtype)

        for region_spec in matching_regions:  # Test all matching regions
            region = HyperrectangularRegion(region_spec["center"], region_spec["radius"])

            # Test all output/input index combinations for this region
            for output_idx, input_idx in all_combinations:

                # Ensure forward bounds are available before computing derivatives
                crown_linearization.compute_network_bounds([region])

                # Test that compute_partial_derivative_bounds runs without error (passing as batch)
                crown_linearization.compute_partial_derivative_bounds(input_idx, output_idx)

                # Retrieve stored bounds using get_partial_derivative_bounds
                A_L, b_L, A_U, b_U = crown_linearization.get_partial_derivative_bounds()

                # Check that bounds are tensors of correct shape
                expected_shape = (network_architecture["input_dim"],)
                assert A_L.flatten().shape == expected_shape, f"A_L shape should be {expected_shape}"
                assert A_U.flatten().shape == expected_shape, f"A_U shape should be {expected_shape}"
                assert b_L.squeeze(-1).shape == (), "b_L should be scalar"
                assert b_U.squeeze(-1).shape == (), "b_U should be scalar"

                # Check that bounds are finite
                assert torch.all(torch.isfinite(A_L)), "A_L should be finite"
                assert torch.all(torch.isfinite(A_U)), "A_U should be finite"
                assert torch.isfinite(b_L), "b_L should be finite"
                assert torch.isfinite(b_U), "b_U should be finite"

                # Empirical validation
                validation_result = validate_derivative_bounds_soundness(network, region, output_idx, input_idx, (A_L, b_L), (A_U, b_U), num_samples=30)

                assert validation_result["valid"], f"Validation failed: {validation_result.get('error', 'Unknown error')}"
                assert validation_result["lower_sound"], (
                    f"Lower bound not sound for {activation_name} activation, "
                    f"indices ({output_idx}, {input_idx}). "
                    f"Empirical range: {validation_result['empirical_range']}, "
                    f"Bounds range: {validation_result['bounds_range']}"
                )
                assert validation_result["upper_sound"], (
                    f"Upper bound not sound for {activation_name} activation, "
                    f"indices ({output_idx}, {input_idx}). "
                    f"Empirical range: {validation_result['empirical_range']}, "
                    f"Bounds range: {validation_result['bounds_range']}"
                )
                assert validation_result["pointwise_sound"], (
                    f"Point-wise bounds not sound for {activation_name} activation, " f"indices ({output_idx}, {input_idx}). " f"Violations: {validation_result['pointwise_violations']}"
                )

    def test_partial_derivative_bounds_simplicial(self, activation_function, network_architecture, simplicial_regions, dtype):
        """Test compute_partial_derivative_bounds with simplicial regions and empirical validation."""
        activation_name, _ = activation_function

        # Filter regions to match network input dimension
        matching_regions = [r for r in simplicial_regions if r["vertices"].shape[1] == network_architecture["input_dim"]]

        if not matching_regions:
            pytest.skip(f"No simplicial regions available for input_dim={network_architecture['input_dim']}")

        # Generate all possible output/input index combinations for this network
        all_combinations = generate_all_output_input_combinations(network_architecture["input_dim"], network_architecture["output_dim"])

        # Create network
        network = create_test_network(activation=activation_name, dtype=dtype, **network_architecture)
        crown_linearization = CrownPartialLinearization(network, dtype=dtype)

        for region_spec in matching_regions:  # Test all matching regions
            region = SimplicialRegion(region_spec["vertices"])

            # Test all output/input index combinations for this region
            for output_idx, input_idx in all_combinations:

                # Ensure forward bounds are available before computing derivatives
                crown_linearization.compute_network_bounds([region])

                # Test that compute_partial_derivative_bounds runs without error (passing as batch)
                crown_linearization.compute_partial_derivative_bounds(input_idx, output_idx)

                # Retrieve stored bounds using get_partial_derivative_bounds for sample_idx=0
                A_L, b_L, A_U, b_U = crown_linearization.get_partial_derivative_bounds()

                # Check that bounds are tensors of correct shape
                expected_shape = (network_architecture["input_dim"],)
                assert A_L.flatten().shape == expected_shape, f"A_L shape should be {expected_shape}"
                assert A_U.flatten().shape == expected_shape, f"A_U shape should be {expected_shape}"
                assert b_L.squeeze(-1).shape == (), "b_L should be scalar"
                assert b_U.squeeze(-1).shape == (), "b_U should be scalar"

                # Check that bounds are finite
                assert torch.all(torch.isfinite(A_L)), "A_L should be finite"
                assert torch.all(torch.isfinite(A_U)), "A_U should be finite"
                assert torch.isfinite(b_L), "b_L should be finite"
                assert torch.isfinite(b_U), "b_U should be finite"

                # Empirical validation
                validation_result = validate_derivative_bounds_soundness(network, region, output_idx, input_idx, (A_L, b_L), (A_U, b_U), num_samples=30)

                assert validation_result["valid"], f"Validation failed: {validation_result.get('error', 'Unknown error')}"
                assert validation_result["lower_sound"], (
                    f"Lower bound not sound for {activation_name} activation, "
                    f"indices ({output_idx}, {input_idx}). "
                    f"Empirical range: {validation_result['empirical_range']}, "
                    f"Bounds range: {validation_result['bounds_range']}"
                )
                assert validation_result["upper_sound"], (
                    f"Upper bound not sound for {activation_name} activation, "
                    f"indices ({output_idx}, {input_idx}). "
                    f"Empirical range: {validation_result['empirical_range']}, "
                    f"Bounds range: {validation_result['bounds_range']}"
                )
                assert validation_result["pointwise_sound"], (
                    f"Point-wise bounds not sound for {activation_name} activation, " f"indices ({output_idx}, {input_idx}). " f"Violations: {validation_result['pointwise_violations']}"
                )


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_layer_network(self, activation_function, dtype):
        """Test with single layer (linear) network."""
        activation_name, _ = activation_function

        # Create single layer network (just linear transformation)
        # For single layer networks, we need to provide a dummy activation relaxation
        # since there are no activations to detect
        network = create_test_network(input_dim=2, hidden_dims=[], output_dim=1, activation=activation_name, dtype=dtype)

        # Import the activation relaxation directly for single layer networks
        from lbp_neural_cbf.linearization.activations import ReLUActivationRelaxation

        dummy_relaxation = ReLUActivationRelaxation()

        crown_linearization = CrownPartialLinearization(network, activation_relaxation=dummy_relaxation, dtype=dtype)

        region = HyperrectangularRegion(np.array([0.0, 0.0], dtype=np.float64), np.array([0.5, 0.5], dtype=np.float64))

        crown_linearization.compute_network_bounds([region])

        # Should work without error (passing as batch)
        crown_linearization.compute_partial_derivative_bounds(0, 0)

        # Retrieve stored bounds to verify it worked
        A_L, b_L, A_U, b_U = crown_linearization.get_partial_derivative_bounds()
        assert A_L is not None and A_U is not None

    def test_invalid_indices(self, activation_function, dtype):
        """Test with invalid output/input indices."""
        activation_name, _ = activation_function

        network = create_test_network(input_dim=2, hidden_dims=[3], output_dim=2, activation=activation_name, dtype=dtype)
        crown_linearization = CrownPartialLinearization(network, dtype=dtype)

        region = HyperrectangularRegion(np.array([0.0, 0.0], dtype=np.float64), np.array([0.5, 0.5], dtype=np.float64))

        # Test invalid output index (passing as batch)
        crown_linearization.compute_network_bounds([region])
        with pytest.raises(IndexError):
            crown_linearization.compute_partial_derivative_bounds(0, 2)  # output_idx=2 invalid

        # Test invalid input index (passing as batch)
        crown_linearization.compute_network_bounds([region])
        with pytest.raises(IndexError):
            crown_linearization.compute_partial_derivative_bounds(2, 0)  # input_idx=2 invalid

    def test_very_small_regions(self, activation_function, dtype):
        """Test with very small regions."""
        activation_name, _ = activation_function

        network = create_test_network(input_dim=2, hidden_dims=[3], output_dim=1, activation=activation_name, dtype=dtype)
        crown_linearization = CrownPartialLinearization(network, dtype=dtype)

        # Very small hyperrectangular region
        tiny_region = HyperrectangularRegion(np.array([0.0, 0.0], dtype=np.float64), np.array([1e-6, 1e-6], dtype=np.float64))

        # Should work without numerical issues (passing as batch)
        crown_linearization.compute_network_bounds([tiny_region])
        crown_linearization.compute_partial_derivative_bounds(0, 0)

        # Retrieve stored bounds
        A_L, b_L, A_U, b_U = crown_linearization.get_partial_derivative_bounds()

        # Check that bounds are finite
        assert torch.all(torch.isfinite(A_L)), "A_L should be finite for tiny regions"
        assert torch.all(torch.isfinite(A_U)), "A_U should be finite for tiny regions"
        assert torch.isfinite(b_L), "b_L should be finite for tiny regions"
        assert torch.isfinite(b_U), "b_U should be finite for tiny regions"

    def test_unsupported_region_type(self, activation_function, dtype):
        """Test with unsupported region type."""
        activation_name, _ = activation_function

        network = create_test_network(input_dim=2, hidden_dims=[3], output_dim=1, activation=activation_name, dtype=dtype)
        crown_linearization = CrownPartialLinearization(network, dtype=dtype)

        # Create a mock unsupported region type
        class UnsupportedRegion:
            def __init__(self):
                self.centroid = np.array([0.0, 0.0], dtype=np.float64)

        unsupported_region = UnsupportedRegion()

        # Should raise TypeError for unsupported region type (passing as batch)
        with pytest.raises(TypeError, match="Unsupported region type"):
            crown_linearization.compute_network_bounds([unsupported_region])
            crown_linearization.compute_partial_derivative_bounds(0, 0)


if __name__ == "__main__":
    # Allow running as a script for debugging
    pytest.main([__file__, "-v"])
