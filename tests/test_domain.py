"""
Test suite for domain.py

This module provides comprehensive testing for the domain system including:

1. **Core Domain Classes**:
   - CircleDomain: Circular/spherical domains in arbitrary dimensions
   - BoxDomain: Rectangular/box domains
   - UnionDomain: Union of multiple domains
   - IntersectionDomain: Intersection of multiple domains
   - ComplementDomain: Complement of a domain within bounds

2. **Mathematical Correctness**:
   - Containment checking for single points and batches
   - Constraint function evaluation and mathematical properties
   - Geometric intersection and containment tests
   - Sampling from domains with statistical validation

3. **Shape Convention Handling**:
   - Standard convention: (batch_size, dim)
   - CBF/dynamics convention: (dim, batch_size)
   - Both NumPy arrays and PyTorch tensors

4. **Empirical Validation**:
   - Verifies mathematical soundness through random sampling
   - Tests boundary conditions and edge cases
   - Validates constraint function properties (positive inside domain)
   - Statistical tests for sampling uniformity

5. **Integration Tests**:
   - Domain parsing from dictionary definitions
   - Visualization functionality (when matplotlib available)
   - Interaction with region objects (hyperrectangular/simplicial)
   - Translator compatibility for differentiable operations

The tests use extensive fixtures to parameterize over:
- Different domain types and configurations
- Various input shapes and conventions
- Multiple dimensions (1D to 5D)
- Edge cases and degenerate scenarios
"""

import pytest
import numpy as np
from typing import List, Tuple, Union, Dict, Any
from unittest.mock import Mock, patch
import warnings

import numpy
import random as rand

# Import the domain module
from lbp_neural_cbf.cbf.domain import (
    Domain,
    CircleDomain,
    BoxDomain,
    UnionDomain,
    IntersectionDomain,
    ComplementDomain,
    parse_domain_definition,
    visualize_domain_2d,
    unsafe_region,
    sat_box_simplex_test,
    _is_torch_tensor,
    _convert_torch_to_numpy,
)
from lbp_neural_cbf.regions import HyperrectangularRegion, SimplicialRegion
from lbp_neural_cbf.translators import NumpyTranslator, TorchTranslator

# Try to import torch for tensor testing
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def tolerance():
    """Default numerical tolerance for floating point comparisons."""
    return 1e-10


@pytest.fixture
def sample_sizes():
    """Different sample sizes for statistical testing."""
    return [100, 1000, 5000]


@pytest.fixture
def dimensions():
    """Test dimensions from 1D to 5D."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def circle_domain_2d():
    """Basic 2D circle domain centered at origin."""
    return CircleDomain(center=[0.0, 0.0], radius=1.0)


@pytest.fixture
def circle_domain_3d():
    """Basic 3D sphere domain."""
    return CircleDomain(center=[1.0, -0.5, 2.0], radius=0.8)


@pytest.fixture
def box_domain_2d():
    """Basic 2D box domain."""
    return BoxDomain(bounds=[[-1.0, 1.0], [-0.5, 1.5]])


@pytest.fixture
def box_domain_3d():
    """Basic 3D box domain."""
    return BoxDomain(bounds=[[-2.0, 2.0], [-1.0, 3.0], [-1.5, 1.0]])


@pytest.fixture
def circle_domains_various():
    """Collection of circle domains in different dimensions."""
    return [
        CircleDomain(center=[0.0], radius=1.0),  # 1D
        CircleDomain(center=[0.0, 0.0], radius=1.0),  # 2D
        CircleDomain(center=[1.0, -1.0], radius=0.5),  # 2D translated
        CircleDomain(center=[0.0, 0.0, 0.0], radius=2.0),  # 3D
        CircleDomain(center=[0.5, -0.3, 1.2], radius=0.7),  # 3D translated
        CircleDomain(center=[0.0, 0.0, 0.0, 0.0], radius=1.5),  # 4D
        CircleDomain(center=[0.1, -0.2, 0.3, -0.4, 0.5], radius=0.3),  # 5D
    ]


@pytest.fixture
def box_domains_various():
    """Collection of box domains in different dimensions."""
    return [
        BoxDomain(bounds=[[-1.0, 1.0]]),  # 1D
        BoxDomain(bounds=[[-1.0, 1.0], [-0.5, 1.5]]),  # 2D
        BoxDomain(bounds=[[0.0, 2.0], [-1.0, 0.0]]),  # 2D translated
        BoxDomain(bounds=[[-2.0, 2.0], [-1.0, 3.0], [-1.5, 1.0]]),  # 3D
        BoxDomain(bounds=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),  # 4D
        BoxDomain(bounds=[[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]),  # 5D
    ]


@pytest.fixture
def test_points_2d():
    """Test points for 2D domains."""
    return [
        # Inside unit circle/square
        np.array([0.0, 0.0]),
        np.array([0.5, 0.3]),
        np.array([-0.2, 0.7]),
        # On boundary (approximately)
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.707, 0.707]),
        # Outside
        np.array([2.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([1.5, 1.5]),
        np.array([-2.0, -2.0]),
    ]


@pytest.fixture
def test_points_3d():
    """Test points for 3D domains."""
    return [
        # Inside unit sphere/cube
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.3, -0.4]),
        # On boundary (approximately)
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.707, 0.707, 0.0]),
        # Outside
        np.array([0.8, -0.5, 2.0]),
        np.array([1.5, -0.5, 2.0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([2.0, 2.0, 2.0]),
        np.array([-2.0, -2.0, -2.0]),
    ]


@pytest.fixture
def hyperrectangular_regions_1d():
    """Sample 1D hyperrectangular regions for testing."""
    return [
        {"center": np.array([0.0]), "radius": np.array([0.5])},
        {"center": np.array([1.0]), "radius": np.array([0.3])},
        {"center": np.array([-1.0]), "radius": np.array([0.2])},
    ]


@pytest.fixture
def hyperrectangular_regions_2d():
    """Sample 2D hyperrectangular regions for testing."""
    return [
        {"center": np.array([0.1, 0.1]), "radius": np.array([0.1, 0.1]), "circle_domain_intersects": True, "circle_domain_contains": True, "box_domain_intersects": True, "box_domain_contains": True},
        {"center": np.array([0.0, 0.0]), "radius": np.array([0.5, 0.5]), "circle_domain_intersects": True, "circle_domain_contains": True, "box_domain_intersects": True, "box_domain_contains": True},
        {
            "center": np.array([1.0, 0.0]),
            "radius": np.array([0.5, 0.5]),
            "circle_domain_intersects": True,
            "circle_domain_contains": False,
            "box_domain_intersects": True,
            "box_domain_contains": False,
        },
        {
            "center": np.array([1.0, -1.0]),
            "radius": np.array([0.3, 0.7]),
            "circle_domain_intersects": True,
            "circle_domain_contains": False,
            "box_domain_intersects": True,
            "box_domain_contains": False,
        },
        {
            "center": np.array([-1.0, 2.0]),
            "radius": np.array([0.2, 0.4]),
            "circle_domain_intersects": False,
            "circle_domain_contains": False,
            "box_domain_intersects": False,
            "box_domain_contains": False,
        },
    ]


@pytest.fixture
def hyperrectangular_regions_3d():
    """Sample 3D hyperrectangular regions for testing."""
    return [
        {
            "center": np.array([1.0, -0.5, 2.0]),
            "radius": np.array([0.3, 0.3, 0.3]),
            "circle_domain_intersects": True,
            "circle_domain_contains": True,
            "box_domain_intersects": False,
            "box_domain_contains": False,
        },
        {
            "center": np.array([0.0, 0.0, 2.0]),
            "radius": np.array([1.0, 0.5, 0.8]),
            "circle_domain_intersects": True,
            "circle_domain_contains": False,
            "box_domain_intersects": False,
            "box_domain_contains": False,
        },
        {
            "center": np.array([-0.5, 1.5, -1.0]),
            "radius": np.array([0.2, 0.4, 0.6]),
            "circle_domain_intersects": False,
            "circle_domain_contains": False,
            "box_domain_intersects": True,
            "box_domain_contains": False,
        },
        {
            "center": np.array([-0.5, 1.5, -1.0]),
            "radius": np.array([0.2, 0.4, 0.2]),
            "circle_domain_intersects": False,
            "circle_domain_contains": False,
            "box_domain_intersects": True,
            "box_domain_contains": True,
        },
        {
            "center": np.array([2.0, -2.0, 2.0]),
            "radius": np.array([0.3, 0.3, 0.3]),
            "circle_domain_intersects": False,
            "circle_domain_contains": False,
            "box_domain_intersects": False,
            "box_domain_contains": False,
        },
    ]


@pytest.fixture
def hyperrectangular_regions(hyperrectangular_regions_1d, hyperrectangular_regions_2d, hyperrectangular_regions_3d):
    """Sample hyperrectangular regions for testing."""
    return hyperrectangular_regions_1d + hyperrectangular_regions_2d + hyperrectangular_regions_3d


@pytest.fixture
def simplicial_regions_1d():
    """
    Sample 1D simplicial regions for testing.

    Assumption: simplices are never degenerate (e.g. points). The assumption is enforced, by the number of vertices, in the SimplicialRegion class.
    """
    return [
        # 1D line segment
        {"vertices": np.array([[0.0], [1.0]])},
        # 1D line segment (translated)
        {"vertices": np.array([[-1.0], [0.0]])},
    ]


@pytest.fixture
def simplicial_regions_2d():
    """
    Sample 2D simplicial regions for testing.

    Assumption: simplices are never degenerate (e.g. line segments or points). The assumption is enforced, by the number of vertices, in the SimplicialRegion class.
    """
    return [
        # 2D triangle
        {"vertices": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]), "circle_domain_intersects": True, "circle_domain_contains": False, "box_domain_intersects": True, "box_domain_contains": True},
        # 2D triangle (translated)
        {
            "vertices": np.array([[-1.0, -1.0], [0.0, -1.0], [-0.5, 0.0]]),
            "circle_domain_intersects": True,
            "circle_domain_contains": False,
            "box_domain_intersects": True,
            "box_domain_contains": False,
        },
        # 2D triangle does not intersect circle domain
        {"vertices": np.array([[2.0, 2.0], [3.0, 2.0], [2.5, 3.0]]), "circle_domain_intersects": False, "circle_domain_contains": False, "box_domain_intersects": False, "box_domain_contains": False},
        # 2D triangle is completely inside circle domain
        {"vertices": np.array([[0.1, 0.1], [0.4, 0.1], [0.25, 0.4]]), "circle_domain_intersects": True, "circle_domain_contains": True, "box_domain_intersects": True, "box_domain_contains": True},
    ]


@pytest.fixture
def simplicial_regions_3d():
    """
    Sample 3D simplicial regions for testing.

    Assumption: simplices are never degenerate (e.g. faces, line segments, or points). The assumption is enforced, by the number of vertices, in the SimplicialRegion class.
    """
    return [
        # 3D tetrahedron
        {
            "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]]),
            "circle_domain_intersects": False,
            "circle_domain_contains": False,
            "box_domain_intersects": True,
            "box_domain_contains": True,
        },
        # 3D tetrahedron (translated)
        {
            "vertices": np.array([[-1.0, -1.0, -1.0], [0.0, -1.0, -1.0], [-0.5, 0.0, -1.0], [-0.5, -0.5, 0.0]]),
            "circle_domain_intersects": False,
            "circle_domain_contains": False,
            "box_domain_intersects": True,
            "box_domain_contains": True,
        },
        # 3D tetrahedron intersects circle domain
        {
            "vertices": np.array([[0.5, -0.5, 0.5], [1.5, -0.5, 1.5], [1.0, 0.5, 1.5], [1.0, 0.0, 2.5]]),
            "circle_domain_intersects": True,
            "circle_domain_contains": False,
            "box_domain_intersects": True,
            "box_domain_contains": False,
        },
        # 3D tetrahedron completely inside circle domain
        {
            "vertices": np.array([[1.0, -0.5, 2.0], [1.2, -0.5, 2.0], [1.1, -0.3, 2.0], [1.1, -0.4, 2.2]]),
            "circle_domain_intersects": True,
            "circle_domain_contains": True,
            "box_domain_intersects": False,
            "box_domain_contains": False,
        },
    ]


@pytest.fixture
def simplicial_regions(simplicial_regions_1d, simplicial_regions_2d, simplicial_regions_3d):
    """Sample simplicial regions for testing."""
    return simplicial_regions_1d + simplicial_regions_2d + simplicial_regions_3d


# =============================================================================
# UTILITY TESTS
# =============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_torch_tensor(self):
        """Test torch tensor detection."""
        # NumPy arrays
        assert not _is_torch_tensor(np.array([1, 2, 3]))
        assert not _is_torch_tensor([1, 2, 3])
        assert not _is_torch_tensor(5)

        if TORCH_AVAILABLE:
            # PyTorch tensors
            assert _is_torch_tensor(torch.tensor([1, 2, 3]))
            assert _is_torch_tensor(torch.zeros(3, 4))

    def test_convert_torch_to_numpy(self):
        """Test torch to numpy conversion."""
        if TORCH_AVAILABLE:
            # CPU tensor
            tensor_cpu = torch.tensor([1.0, 2.0, 3.0])
            result = _convert_torch_to_numpy(tensor_cpu)
            assert isinstance(result, np.ndarray)
            np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

            # GPU tensor (if available)
            if torch.cuda.is_available():
                tensor_gpu = torch.tensor([1.0, 2.0, 3.0]).cuda()
                result = _convert_torch_to_numpy(tensor_gpu)
                assert isinstance(result, np.ndarray)
                np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

        # Non-tensor input should be returned as-is
        array = np.array([1, 2, 3])
        result = _convert_torch_to_numpy(array)
        assert result is array

    def test_sat_box_simplex_test_2d_intersect(self):
        """Test Separating Axis Theorem implementation."""
        # Box and simplex that intersect
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex = np.array([[0.5, 0.5], [1.5, 0.5], [1.0, 1.5]])
        assert sat_box_simplex_test(box_min, box_max, simplex)

    def test_sat_box_simplex_test_2d_no_intersect_two_axes(self):
        # Box and simplex that don't intersect - two axes
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_outside = np.array([[2.0, 2.0], [3.0, 2.0], [2.5, 3.0]])
        assert not sat_box_simplex_test(box_min, box_max, simplex_outside)

    def test_sat_box_simplex_test_2d_no_intersect_y_axis(self):
        # Box and simplex that don't intersect - one axis
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_outside_one_axis = np.array([[0.5, 1.5], [1.5, 1.5], [1.0, 2.0]])
        assert not sat_box_simplex_test(box_min, box_max, simplex_outside_one_axis)

    def test_sat_box_simplex_test_2d_no_intersect_x_axis(self):
        # Box and simplex that don't intersect - other axis
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_outside_other_axis = np.array([[1.5, 0.5], [2.5, 0.5], [2.0, 1.0]])
        assert not sat_box_simplex_test(box_min, box_max, simplex_outside_other_axis)

    def test_sat_box_simplex_test_2d_box_in_simplex(self):
        # Box completely inside simplex
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_enclosing = np.array([[-1.0, -1.0], [2.0, 0.0], [0.0, 2.0]])
        assert sat_box_simplex_test(box_min, box_max, simplex_enclosing)

    def test_sat_box_simplex_test_2d_simplex_in_box(self):
        # Simplex completely inside box
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_inside = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]])
        assert sat_box_simplex_test(box_min, box_max, simplex_inside)

    def test_sat_box_simplex_test_2d_touch_at_point(self):
        # Box and simplex that touch at a point
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_touching = np.array([[1.0, 1.0], [1.5, 1.0], [1.0, 1.5]])
        assert sat_box_simplex_test(box_min, box_max, simplex_touching)

    def test_sat_box_simplex_test_2d_touch_at_edge(self):
        # Box and simplex that share an edge
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_edge = np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 1.0]])
        assert sat_box_simplex_test(box_min, box_max, simplex_edge)

    def test_sat_box_simplex_test_2d_degenerate_simplex_line(self):
        # Degenerate simplex (line segment)
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_line = np.array([[0.5, 0.5], [0.5, 1.5]])
        assert sat_box_simplex_test(box_min, box_max, simplex_line)

    def test_sat_box_simplex_test_2d_degenerate_simplex_point(self):
        # Degenerate simplex (point)
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 1.0])
        simplex_point = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert sat_box_simplex_test(box_min, box_max, simplex_point)

    def test_sat_box_simplex_test_2d_degenerate_box(self):
        # Degenerate box (line segment)
        box_min = np.array([0.0, 0.0])
        box_max = np.array([1.0, 0.0])  # Zero height
        simplex = np.array([[0.5, -0.5], [0.5, 0.5], [1.5, 0.0]])
        assert sat_box_simplex_test(box_min, box_max, simplex)


# =============================================================================
# CIRCLE DOMAIN TESTS
# =============================================================================


class TestCircleDomain:
    """Test CircleDomain functionality."""

    def test_initialization_2d(self):
        """Test CircleDomain initialization in 2D."""
        # 2D circle
        circle = CircleDomain(center=[1.0, -0.5], radius=2.0)
        assert circle.dim == 2
        np.testing.assert_allclose(circle.center, [1.0, -0.5])
        assert circle.radius == 2.0

    def test_initialization_3d(self):
        """Test CircleDomain initialization in 3D."""
        # 3D sphere
        sphere = CircleDomain(center=[0.0, 0.0, 0.0], radius=1.5)
        assert sphere.dim == 3
        np.testing.assert_allclose(sphere.center, [0.0, 0.0, 0.0])
        assert sphere.radius == 1.5

    def test_contains_single_points_2d(self, circle_domain_2d, test_points_2d):
        """Test containment checking for single points in 2D."""
        # 2D tests
        for point in test_points_2d:
            result = circle_domain_2d.contains(point)
            expected = np.sum((point - circle_domain_2d.center) ** 2) <= circle_domain_2d.radius**2
            assert result == expected

    def test_contains_single_points_3d(self, circle_domain_3d, test_points_3d):
        """Test containment checking for single points in 3D."""
        # 3D tests
        for point in test_points_3d:
            result = circle_domain_3d.contains(point)
            distance_sq = np.sum((point - circle_domain_3d.center) ** 2)
            expected = distance_sq <= circle_domain_3d.radius**2
            assert result == expected

    def test_contains_batch_points_2d(self, circle_domain_2d, test_points_2d):
        """Test containment checking for batch of points in 2D."""
        # Standard convention: (batch_size, dim)
        points = np.stack(test_points_2d)
        results = circle_domain_2d.contains(points)

        assert isinstance(results, np.ndarray)
        assert results.shape == (len(test_points_2d),)

        # Check individual results
        for i, point in enumerate(test_points_2d):
            expected = np.sum((point - circle_domain_2d.center) ** 2) <= circle_domain_2d.radius**2
            assert results[i] == expected

    def test_contains_batch_points_3d(self, circle_domain_3d, test_points_3d):
        """Test containment checking for batch of points in 3D."""
        # Standard convention: (batch_size, dim)
        points = np.stack(test_points_3d)
        results = circle_domain_3d.contains(points)

        assert isinstance(results, np.ndarray)
        assert results.shape == (len(test_points_3d),)

        # Check individual results
        for i, point in enumerate(test_points_3d):
            expected = np.sum((point - circle_domain_3d.center) ** 2) <= circle_domain_3d.radius**2
            assert results[i] == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_contains_torch_tensors(self, circle_domain_2d, test_points_2d):
        """Test containment with PyTorch tensors."""
        points = torch.tensor(np.stack(test_points_2d))
        results = circle_domain_2d.contains(points)

        assert isinstance(results, np.ndarray)
        assert results.shape == (len(test_points_2d),)

        for i, point in enumerate(test_points_2d):
            expected = np.sum((point - circle_domain_2d.center) ** 2) <= circle_domain_2d.radius**2
            assert results[i] == expected

    def test_constraint_function(self, circle_domain_2d, tolerance):
        """Test constraint function properties."""
        # Points inside should have positive constraint
        inside_points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.2, 0.7]])
        for point in inside_points:
            constraint = circle_domain_2d.constraint(point)
            assert constraint > tolerance  # Should be positive inside

        # Points outside should have negative constraint
        outside_points = np.array([[2.0, 0.0], [0.0, 2.0], [1.5, 1.5]])
        for point in outside_points:
            constraint = circle_domain_2d.constraint(point)
            assert constraint < -tolerance  # Should be negative outside

        # Points on boundary should have constraint ≈ 0
        boundary_points = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])  # Use exact value
        for point in boundary_points:
            constraint = circle_domain_2d.constraint(point)
            assert abs(constraint) < 1e-3  # Relax tolerance for boundary points

    def test_constraint_function_batch(self, circle_domain_2d):
        """Test constraint function with batch of points."""
        points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.2, 0.7], [2.0, 0.0], [0.0, 2.0], [1.5, 1.5], [1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])

        constraints = circle_domain_2d.constraint(points)
        assert isinstance(constraints, np.ndarray)
        assert constraints.shape == (points.shape[0],)

        # Check individual constraints
        for i, point in enumerate(points):
            expected = circle_domain_2d.constraint(point)
            assert abs(constraints[i] - expected) < 1e-6

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_constraint_with_translator(self, circle_domain_2d):
        """Test constraint function with translator."""
        torch_translator = TorchTranslator(dtype=torch.float64)
        np_translator = NumpyTranslator(dtype=np.float64)

        points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.2, 0.7], [2.0, 0.0], [0.0, 2.0], [1.5, 1.5], [1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])

        # Individual point checks
        for point in points:
            constraint_np = circle_domain_2d.constraint(point, translator=np_translator)
            constraint_torch = circle_domain_2d.constraint(point, translator=torch_translator)
            np.testing.assert_approx_equal(constraint_np, constraint_torch.item())

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_constraint_with_translator_batch(self, circle_domain_2d):
        """Test constraint function with translator and batch of points."""
        torch_translator = TorchTranslator(dtype=torch.float64)
        np_translator = NumpyTranslator(dtype=np.float64)

        points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.2, 0.7], [2.0, 0.0], [0.0, 2.0], [1.5, 1.5], [1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])

        # Batch point checks
        constraints_np = circle_domain_2d.constraint(points, translator=np_translator)
        constraints_torch = circle_domain_2d.constraint(points, translator=torch_translator)
        np.testing.assert_allclose(constraints_np, constraints_torch.numpy())

    def test_sampling_uniformity(self, circle_domains_various, sample_sizes):
        """Test that sampling produces uniformly distributed points."""
        for domain in circle_domains_various:
            for n_samples in sample_sizes:
                if n_samples > 1000 and domain.dim > 3:
                    continue  # Skip large samples for high dimensions

                points = domain.sample_points(n_samples)

                # Basic checks
                assert points.shape == (n_samples, domain.dim)
                assert np.all(domain.contains(points))

                # Check that points are distributed within radius
                distances = np.linalg.norm(points - domain.center, axis=1)
                assert np.all(distances <= domain.radius + 1e-10)

                # For 2D, check angular distribution uniformity (statistical test)
                if domain.dim == 2 and n_samples >= 1000:
                    angles = np.arctan2(points[:, 1] - domain.center[1], points[:, 0] - domain.center[0])
                    # Normalize to [0, 2π]
                    angles = (angles + 2 * np.pi) % (2 * np.pi)

                    # Chi-square test for uniformity
                    n_bins = 8
                    hist, _ = np.histogram(angles, bins=n_bins, range=(0, 2 * np.pi))
                    expected = n_samples / n_bins

                    # Simple uniformity check (not rigorous chi-square)
                    relative_errors = np.abs(hist - expected) / expected
                    assert np.all(relative_errors < 0.3)  # Allow 30% deviation

    def test_sampling_edge_cases(self, circle_domain_2d):
        """Test sampling edge cases."""
        # Zero points
        points = circle_domain_2d.sample_points(0)
        assert points.shape == (0, 2)

        # Single point
        points = circle_domain_2d.sample_points(1)
        assert points.shape == (1, 2)
        assert circle_domain_2d.contains(points[0])

        # Negative points (should raise TypeError)
        with pytest.raises(TypeError, match="num_points must be a non-negative integer"):
            circle_domain_2d.sample_points(-5)

    def test_geometric_operations_hyperrectangle_2d(self, circle_domain_2d, hyperrectangular_regions_2d):
        """Test geometric intersection and containment operations."""

        # TODO: Test case with only intersecting on line segment (not corner)

        # Test with hyperrectangular regions
        for region_data in hyperrectangular_regions_2d:
            center = region_data["center"]
            radius = region_data["radius"]

            # Test intersection
            intersects = circle_domain_2d.intersects_hyperrect(center, radius)
            assert isinstance(intersects, (bool, np.bool_))
            if "circle_domain_intersects" in region_data:
                assert intersects == region_data["circle_domain_intersects"]

            # Test containment
            contains = circle_domain_2d.contains_hyperrect(center, radius)
            assert isinstance(contains, (bool, np.bool_))
            if "circle_domain_contains" in region_data:
                assert contains == region_data["circle_domain_contains"]

            # Logical consistency: if contains, then must intersect
            if contains:
                assert intersects

    def test_geometric_operations_hyperrectangle_3d(self, circle_domain_3d, hyperrectangular_regions_3d):
        """Test geometric intersection and containment operations."""

        # TODO: Test case with only intersecting on face (not corner)

        # Test with hyperrectangular regions
        for region_data in hyperrectangular_regions_3d:
            center = region_data["center"]
            radius = region_data["radius"]

            # Test intersection
            intersects = circle_domain_3d.intersects_hyperrect(center, radius)
            assert isinstance(intersects, (bool, np.bool_))
            if "circle_domain_intersects" in region_data:
                assert intersects == region_data["circle_domain_intersects"]

            # Test containment
            contains = circle_domain_3d.contains_hyperrect(center, radius)
            assert isinstance(contains, (bool, np.bool_))
            if "circle_domain_contains" in region_data:
                assert contains == region_data["circle_domain_contains"]

            # Logical consistency: if contains, then must intersect
            if contains:
                assert intersects

    def test_geometric_operations_simplicial_2d(self, circle_domain_2d, simplicial_regions_2d):
        """Test geometric intersection and containment operations."""

        # TODO: Test case with only intersecting on line segment (not corner)

        # Test with simplicial regions
        for region_data in simplicial_regions_2d:
            vertices = region_data["vertices"]

            # Test intersection
            intersects = circle_domain_2d.intersects_simplex(vertices)
            assert isinstance(intersects, (bool, np.bool_))
            if "circle_domain_intersects" in region_data:
                assert intersects == region_data["circle_domain_intersects"]

            # Test containment
            contains = circle_domain_2d.contains_simplex(vertices)
            assert isinstance(contains, (bool, np.bool_))
            if "circle_domain_contains" in region_data:
                assert contains == region_data["circle_domain_contains"]

            # Logical consistency
            if contains:
                assert intersects

    def test_geometric_operations_simplicial_3d(self, circle_domain_3d, simplicial_regions_3d):
        """Test geometric intersection and containment operations."""

        # TODO: Test case with only intersecting on face (not corner)

        # Test with simplicial regions
        for region_data in simplicial_regions_3d:
            vertices = region_data["vertices"]

            # Test intersection
            intersects = circle_domain_3d.intersects_simplex(vertices)
            assert isinstance(intersects, (bool, np.bool_))
            if "circle_domain_intersects" in region_data:
                assert intersects == region_data["circle_domain_intersects"]

            # Test containment
            contains = circle_domain_3d.contains_simplex(vertices)
            assert isinstance(contains, (bool, np.bool_))
            if "circle_domain_contains" in region_data:
                assert contains == region_data["circle_domain_contains"]

            # Logical consistency
            if contains:
                assert intersects

    def test_dimension_validation(self):
        """Test dimension validation in methods."""
        circle = CircleDomain(center=[0.0, 0.0], radius=1.0)

        # Wrong dimension for contains
        with pytest.raises(ValueError, match="Expected points with dimensions 2, got 1"):
            circle.contains(np.array([1.0]))

        with pytest.raises(ValueError, match="Expected points with dimensions 2, got 3"):
            circle.contains(np.array([1.0, 2.0, 3.0]))

        # Wrong shape for batch
        with pytest.raises(ValueError, match="Expected points with dimensions 2, got 1"):
            circle.contains(np.array([[1.0], [2.0], [3.0]]))


# =============================================================================
# BOX DOMAIN TESTS
# =============================================================================


class TestBoxDomain:
    """Test BoxDomain functionality."""

    def test_initialization(self):
        """Test BoxDomain initialization."""
        # 2D box
        box = BoxDomain(bounds=[[-1.0, 2.0], [-0.5, 1.5]])
        assert box.dim == 2
        np.testing.assert_allclose(box.low_bounds, [-1.0, -0.5])
        np.testing.assert_allclose(box.high_bounds, [2.0, 1.5])

        # Invalid bounds should raise error
        with pytest.raises(ValueError, match="Invalid bounds"):
            BoxDomain(bounds=[[1.0, 0.0]])  # low >= high

        with pytest.raises(ValueError, match="Bounds cannot be empty"):
            BoxDomain(bounds=[])

    def test_contains_single_points(self, box_domain_2d, test_points_2d):
        """Test containment checking for single points."""
        for point in test_points_2d:
            result = box_domain_2d.contains(point)
            expected = np.all((point >= box_domain_2d.low_bounds) & (point <= box_domain_2d.high_bounds))
            assert result == expected

    def test_contains_batch_points(self, box_domain_2d):
        """Test containment checking for batch of points."""
        # Standard convention: (batch_size, dim)
        points = np.array([[0.0, 0.0], [0.5, 1.0], [2.0, 0.0], [-0.5, 0.5]])
        results = box_domain_2d.contains(points)

        assert isinstance(results, np.ndarray)
        assert results.shape == (4,)

        # Check against bounds [-1, 1] x [-0.5, 1.5]
        expected = [True, True, False, True]
        for i, exp in enumerate(expected):
            assert results[i] == exp

    def test_constraint_function(self, box_domain_2d):
        """Test constraint function properties."""
        # Points inside should have positive constraint
        inside_points = np.array([[0.0, 0.25], [0.5, 1.0], [-0.5, 0.5], [0.0, 0.5]])
        expected_min_distances = np.array([0.75, 0.5, 0.5, 1.0])
        for point, expected in zip(inside_points, expected_min_distances):
            constraint = box_domain_2d.constraint(point)
            np.testing.assert_approx_equal(constraint, expected)

        # Points outside should have negative constraint
        outside_points = np.array([[3.0, 0.0], [-0.75, 2.0], [-2.0, -1.75]])  # Outside top right  # Outside top  # Outside bottom left
        expected_min_distances = np.array([-2.0, -0.5, -1.25])
        for point, expected in zip(outside_points, expected_min_distances):
            constraint = box_domain_2d.constraint(point)
            np.testing.assert_approx_equal(constraint, expected)

        # Points on boundary should have constraint ≈ 0
        boundary_points = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -0.5], [0.0, 1.5]])
        for point in boundary_points:
            constraint = box_domain_2d.constraint(point)
            np.testing.assert_approx_equal(constraint, 0.0)

    def test_constraint_function_batch(self, box_domain_2d):
        """Test constraint function with batch of points."""
        points = np.array(
            [
                [0.0, 0.25],  # Inside
                [0.5, 1.0],  # Inside
                [-0.5, 0.5],  # Inside
                [3.0, 0.0],  # Outside
                [-0.75, 2.0],  # Outside
                [-2.0, -1.75],  # Outside
                [-1.0, 0.0],  # Boundary
                [1.0, 0.0],  # Boundary
                [0.0, -0.5],  # Boundary
                [0.0, 1.5],  # Boundary
            ]
        )

        constraints = box_domain_2d.constraint(points)
        assert isinstance(constraints, np.ndarray)
        assert constraints.shape == (points.shape[0],)

        for point, constraint in zip(points, constraints):
            expected = box_domain_2d.constraint(point)
            np.testing.assert_approx_equal(constraint, expected)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_constraint_with_translator(self, box_domain_2d):
        """Test constraint function with translator."""
        torch_translator = TorchTranslator()
        np_translator = NumpyTranslator()

        points = np.array(
            [
                [0.0, 0.25],  # Inside
                [0.5, 1.0],  # Inside
                [-0.5, 0.5],  # Inside
                [3.0, 0.0],  # Outside
                [-0.75, 2.0],  # Outside
                [-2.0, -1.75],  # Outside
                [-1.0, 0.0],  # Boundary
                [1.0, 0.0],  # Boundary
                [0.0, -0.5],  # Boundary
                [0.0, 1.5],  # Boundary
            ]
        )

        # Individual point checks
        for point in points:
            constraint_np = box_domain_2d.constraint(point, translator=np_translator)
            constraint_torch = box_domain_2d.constraint(point, translator=torch_translator)
            np.testing.assert_approx_equal(constraint_np, constraint_torch.item())

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_constraint_with_translator_batch(self, box_domain_2d):
        """Test constraint function with translator and batch of points."""
        torch_translator = TorchTranslator()
        np_translator = NumpyTranslator()

        points = np.array(
            [
                [0.0, 0.25],  # Inside
                [0.5, 1.0],  # Inside
                [-0.5, 0.5],  # Inside
                [3.0, 0.0],  # Outside
                [-0.75, 2.0],  # Outside
                [-2.0, -1.75],  # Outside
                [-1.0, 0.0],  # Boundary
                [1.0, 0.0],  # Boundary
                [0.0, -0.5],  # Boundary
                [0.0, 1.5],  # Boundary
            ]
        )

        # Batch point checks
        constraints_np = box_domain_2d.constraint(points, translator=np_translator)
        constraints_torch = box_domain_2d.constraint(points, translator=torch_translator)
        np.testing.assert_allclose(constraints_np, constraints_torch.numpy())

    def test_sampling_uniformity(self, box_domains_various, sample_sizes):
        """Test that sampling produces uniformly distributed points."""
        for domain in box_domains_various:
            for n_samples in sample_sizes:
                if n_samples > 1000 and domain.dim > 3:
                    continue

                points = domain.sample_points(n_samples)

                # Basic checks
                assert points.shape == (n_samples, domain.dim)
                assert np.all(domain.contains(points))

                # Check that points are within bounds
                assert np.all(points >= domain.low_bounds)
                assert np.all(points <= domain.high_bounds)

                # For sufficient samples, check uniform distribution
                if n_samples >= 1000:
                    for dim in range(domain.dim):
                        dim_values = points[:, dim]
                        low, high = domain.bounds[dim]

                        # Check that values span the range reasonably
                        assert np.min(dim_values) < low + 0.1 * (high - low)
                        assert np.max(dim_values) > high - 0.1 * (high - low)

                        # Check mean is approximately at center
                        expected_mean = (low + high) / 2
                        actual_mean = np.mean(dim_values)
                        assert abs(actual_mean - expected_mean) < 0.1 * (high - low)

    def test_geometric_operations_hyperrectangle_2d(self, box_domain_2d, hyperrectangular_regions_2d):
        """Test geometric intersection and containment operations of hyperrectangle in 2D."""

        # TODO: Test case with only intersecting on line segment (not corner)

        # Test with hyperrectangular regions
        for region_data in hyperrectangular_regions_2d:
            center = region_data["center"]
            radius = region_data["radius"]

            intersects = box_domain_2d.intersects_hyperrect(center, radius)
            assert isinstance(intersects, (bool, np.bool_))
            if "box_domain_intersects" in region_data:
                assert intersects == region_data["box_domain_intersects"]

            contains = box_domain_2d.contains_hyperrect(center, radius)
            assert isinstance(contains, (bool, np.bool_))
            if "box_domain_contains" in region_data:
                assert contains == region_data["box_domain_contains"]

            if contains:
                assert intersects

    def test_geometric_operations_hyperrectangle_3d(self, box_domain_3d, hyperrectangular_regions_3d):
        """Test geometric intersection and containment operations of hyperrectangle in 3D."""

        # TODO: Test case with only intersecting on face (not corner)

        # Test with hyperrectangular regions
        for region_data in hyperrectangular_regions_3d:
            center = region_data["center"]
            radius = region_data["radius"]

            intersects = box_domain_3d.intersects_hyperrect(center, radius)
            assert isinstance(intersects, (bool, np.bool_))
            if "box_domain_intersects" in region_data:
                assert intersects == region_data["box_domain_intersects"]

            contains = box_domain_3d.contains_hyperrect(center, radius)
            assert isinstance(contains, (bool, np.bool_))
            if "box_domain_contains" in region_data:
                assert contains == region_data["box_domain_contains"]

            if contains:
                assert intersects

    def test_geometric_operations_simplicial_2d(self, box_domain_2d, simplicial_regions_2d):
        """Test geometric intersection and containment operations of simplicies in 2D."""

        # TODO: Test case with only intersecting on line segment (not corner)

        # Test with simplicial regions
        for region_data in simplicial_regions_2d:
            vertices = region_data["vertices"]

            intersects = box_domain_2d.intersects_simplex(vertices)
            assert isinstance(intersects, (bool, np.bool_))
            if "box_domain_intersects" in region_data:
                assert intersects == region_data["box_domain_intersects"]

            contains = box_domain_2d.contains_simplex(vertices)
            assert isinstance(contains, (bool, np.bool_))
            if "box_domain_contains" in region_data:
                assert contains == region_data["box_domain_contains"]

            if contains:
                assert intersects

    def test_geometric_operations_simplicial_3d(self, box_domain_3d, simplicial_regions_3d):
        """Test geometric intersection and containment operations of simplicies in 3D."""

        # TODO: Test case with only intersecting on face (not corner)

        # Test with simplicial regions
        for region_data in simplicial_regions_3d:
            vertices = region_data["vertices"]

            intersects = box_domain_3d.intersects_simplex(vertices)
            assert isinstance(intersects, (bool, np.bool_))
            if "box_domain_intersects" in region_data:
                assert intersects == region_data["box_domain_intersects"]

            contains = box_domain_3d.contains_simplex(vertices)
            assert isinstance(contains, (bool, np.bool_))
            if "box_domain_contains" in region_data:
                assert contains == region_data["box_domain_contains"]

            if contains:
                assert intersects


# =============================================================================
# COMPOSITE DOMAIN TESTS
# =============================================================================


class TestUnionDomain:
    """Test UnionDomain functionality."""

    def test_initialization(self, circle_domain_2d, box_domain_2d):
        """Test UnionDomain initialization."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])
        assert union.dim == 2
        assert len(union.domains) == 2

        # Mismatched dimensions should raise error
        circle_3d = CircleDomain(center=[0, 0, 0], radius=1.0)
        with pytest.raises(ValueError, match="All domains must have the same dimension"):
            UnionDomain([circle_domain_2d, circle_3d])

        # Empty domain list should raise error
        with pytest.raises(ValueError, match="requires at least one sub-domain"):
            UnionDomain([])

    def test_contains(self, circle_domain_2d, box_domain_2d):
        """Test union containment logic."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])

        # Point in circle but not box (circle has radius 1, box is [-1,1] x [-0.5,1.5])
        point_circle = np.array([0.0, -0.6])  # Inside circle but below box y-bounds
        assert circle_domain_2d.contains(point_circle)
        assert not box_domain_2d.contains(point_circle)
        assert union.contains(point_circle)

        # Point in box but not circle
        point_box = np.array([0.0, 1.4])  # Inside box but outside circle
        assert not circle_domain_2d.contains(point_box)
        assert box_domain_2d.contains(point_box)
        assert union.contains(point_box)

        # Point in both
        point_both = np.array([0.0, 0.0])
        assert circle_domain_2d.contains(point_both)
        assert box_domain_2d.contains(point_both)
        assert union.contains(point_both)

        # Point in neither
        point_neither = np.array([2.0, 2.0])
        assert not circle_domain_2d.contains(point_neither)
        assert not box_domain_2d.contains(point_neither)
        assert not union.contains(point_neither)

    def test_contains_batch(self, circle_domain_2d, box_domain_2d):
        """Test union containment for batch of points."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        results = union.contains(points)
        expected = [True, True, True, True, False]

        for i, exp in enumerate(expected):
            assert results[i] == exp

    def test_constraint_function(self, circle_domain_2d, box_domain_2d):
        """Test union constraint function."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        inside = [True, True, True, True, False]

        for point, is_inside in zip(points, inside):
            # Test single point
            constraint = union.constraint(point)

            if is_inside:
                assert constraint > 0
            else:
                assert constraint <= 0

            # The maximum should be at least as large as the larger individual constraint
            circle_constraint = circle_domain_2d.constraint(point)
            box_constraint = box_domain_2d.constraint(point)
            np.testing.assert_approx_equal(constraint, max(circle_constraint, box_constraint))

    def test_constraint_function_batch(self, circle_domain_2d, box_domain_2d):
        """Test union constraint function."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        batch_constraints = union.constraint(points)
        assert isinstance(batch_constraints, np.ndarray)
        assert batch_constraints.shape == (points.shape[0],)

        for point, batch_constraint in zip(points, batch_constraints):
            constraint = union.constraint(point)
            np.testing.assert_approx_equal(constraint, batch_constraint)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_constraint_function_translator(self, circle_domain_2d, box_domain_2d):
        """Test union constraint function."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])
        translator = TorchTranslator()

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both?  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        for point in points:
            # Test single point
            constraint = union.constraint(point)

            point_tensor = translator.to_format(point)
            constraint_trans = union.constraint(point_tensor, translator=translator)

            # Should be approximately the same for this case
            if hasattr(constraint_trans, "item"):
                constraint_trans = constraint_trans.item()

            np.testing.assert_approx_equal(constraint, constraint_trans)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_constraint_function_translator_batch(self, circle_domain_2d, box_domain_2d):
        """Test union constraint function."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])
        translator = TorchTranslator(dtype=torch.float64)

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both?  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        constraint = union.constraint(points)

        points_tensor = translator.to_format(points)
        constraint_trans = union.constraint(points_tensor, translator=translator)

        np.testing.assert_allclose(constraint, constraint_trans.numpy())

    def test_sampling(self, circle_domain_2d, box_domain_2d, sample_sizes):
        """Test union sampling."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])

        for n_samples in sample_sizes:
            if n_samples > 1000:
                continue  # Skip large samples for speed

            points = union.sample_points(n_samples)

            # All points should be in the union
            assert np.all(union.contains(points))

            # Points should be distributed between domains
            in_circle = circle_domain_2d.contains(points)
            in_box = box_domain_2d.contains(points)

            # At least some points should be in each domain (statistical)
            if n_samples >= 100:
                assert np.sum(in_circle) > 0
                assert np.sum(in_box) > 0

    def test_geometric_operations_hyperrectangle_2d(self, circle_domain_2d, box_domain_2d, hyperrectangular_regions_2d):
        """Test geometric intersection operations of hyperrectangle in 2D."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])

        # Test with hyperrectangular regions
        for region_data in hyperrectangular_regions_2d:
            center = region_data["center"]
            radius = region_data["radius"]

            intersects = union.intersects_hyperrect(center, radius)
            intersects_box = box_domain_2d.intersects_hyperrect(center, radius)
            intersects_circle = circle_domain_2d.intersects_hyperrect(center, radius)
            assert intersects == (intersects_box or intersects_circle)

    def test_geometric_operations_hyperrectangle_3d(self, circle_domain_3d, box_domain_3d, hyperrectangular_regions_3d):
        """Test geometric intersection operations of hyperrectangle in 3D."""
        union = UnionDomain([circle_domain_3d, box_domain_3d])

        # Test with hyperrectangular regions
        for region_data in hyperrectangular_regions_3d:
            center = region_data["center"]
            radius = region_data["radius"]

            intersects = union.intersects_hyperrect(center, radius)
            intersects_box = box_domain_3d.intersects_hyperrect(center, radius)
            intersects_circle = circle_domain_3d.intersects_hyperrect(center, radius)
            assert intersects == (intersects_box or intersects_circle)

    def test_geometric_operations_simplicial_2d(self, circle_domain_2d, box_domain_2d, simplicial_regions_2d):
        """Test geometric intersection operations of simplicies in 2D."""
        union = UnionDomain([circle_domain_2d, box_domain_2d])

        # Test with simplicial regions
        for region_data in simplicial_regions_2d:
            vertices = region_data["vertices"]

            intersects = union.intersects_simplex(vertices)
            intersects_box = box_domain_2d.intersects_simplex(vertices)
            intersects_circle = circle_domain_2d.intersects_simplex(vertices)
            assert intersects == (intersects_box or intersects_circle)

    def test_geometric_operations_simplicial_3d(self, circle_domain_3d, box_domain_3d, simplicial_regions_3d):
        """Test geometric intersection operations of simplicies in 3D."""
        union = UnionDomain([circle_domain_3d, box_domain_3d])

        # Test with simplicial regions
        for region_data in simplicial_regions_3d:
            vertices = region_data["vertices"]

            intersects = union.intersects_simplex(vertices)
            intersects_box = box_domain_3d.intersects_simplex(vertices)
            intersects_circle = circle_domain_3d.intersects_simplex(vertices)
            assert intersects == (intersects_box or intersects_circle)


class TestIntersectionDomain:
    """Test IntersectionDomain functionality."""

    def test_initialization(self, circle_domain_2d, box_domain_2d):
        """Test IntersectionDomain initialization."""
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])
        assert intersection.dim == 2
        assert len(intersection.domains) == 2

    def test_contains(self, circle_domain_2d, box_domain_2d):
        """Test intersection containment logic."""
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])

        # Point in both domains
        point_both = np.array([0.0, 0.0])
        assert circle_domain_2d.contains(point_both)
        assert box_domain_2d.contains(point_both)
        assert intersection.contains(point_both)

        # Point in circle but not box
        point_circle = np.array([0.0, -0.6])  # Inside circle but below box y-bounds
        assert circle_domain_2d.contains(point_circle)
        assert not box_domain_2d.contains(point_circle)
        assert not intersection.contains(point_circle)

        # Point in box but not circle
        point_box = np.array([0.0, 1.4])  # Inside box but outside circle
        assert not circle_domain_2d.contains(point_box)
        assert box_domain_2d.contains(point_box)
        assert not intersection.contains(point_box)

        # Point in neither
        point_neither = np.array([2.0, 2.0])
        assert not circle_domain_2d.contains(point_neither)
        assert not box_domain_2d.contains(point_neither)
        assert not intersection.contains(point_neither)

    def test_contains_batch(self, circle_domain_2d, box_domain_2d):
        """Test intersection containment for batch of points."""
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        results = intersection.contains(points)
        expected = [True, True, False, False, False]

        for i, exp in enumerate(expected):
            assert results[i] == exp

    def test_constraint_function(self, circle_domain_2d, box_domain_2d):
        """Test intersection constraint function (smooth minimum)."""
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        inside = [True, True, False, False, False]

        for point, is_inside in zip(points, inside):
            # Test single point in intersection
            constraint = intersection.constraint(point)

            if is_inside:
                assert constraint > 0
            else:
                assert constraint <= 0

            circle_constraint = circle_domain_2d.constraint(point)
            box_constraint = box_domain_2d.constraint(point)
            np.testing.assert_approx_equal(constraint, min(circle_constraint, box_constraint))

    def test_constraint_function_batch(self, circle_domain_2d, box_domain_2d):
        """Test intersection constraint function (smooth minimum) with batch of points."""
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        batch_constraints = intersection.constraint(points)
        assert isinstance(batch_constraints, np.ndarray)
        assert batch_constraints.shape == (points.shape[0],)

        for point, batch_constraint in zip(points, batch_constraints):
            constraint = intersection.constraint(point)
            np.testing.assert_approx_equal(constraint, batch_constraint)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_constraint_function(self, circle_domain_2d, box_domain_2d):
        """Test intersection constraint function (smooth minimum)."""
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        for point in points:
            # Test single point
            constraint = intersection.constraint(point)

            translator = TorchTranslator()
            point_tensor = translator.to_format(point)
            constraint_trans = intersection.constraint(point_tensor, translator=translator)

            if hasattr(constraint_trans, "item"):
                constraint_trans = constraint_trans.item()

            np.testing.assert_approx_equal(constraint, constraint_trans)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_constraint_function_batch(self, circle_domain_2d, box_domain_2d):
        """Test intersection constraint function (smooth minimum) with batch of points."""
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])

        points = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, -0.6], [0.0, 1.4], [2.0, 2.0]])  # In both  # In both  # In circle only (below box y-bounds)  # In box only (outside circle)  # In neither

        constraint = intersection.constraint(points)

        translator = TorchTranslator(dtype=torch.float64)
        points_tensor = translator.to_format(points)
        constraint_trans = intersection.constraint(points_tensor, translator=translator)

        np.testing.assert_allclose(constraint, constraint_trans.numpy())

    def test_sampling_rejection_method(self, circle_domain_2d, box_domain_2d):
        """Test intersection sampling using rejection method."""
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])

        # Small sample size to avoid timeout issues
        points = intersection.sample_points(50)

        # All sampled points should be in the intersection
        assert np.all(intersection.contains(points))
        assert np.all(circle_domain_2d.contains(points))
        assert np.all(box_domain_2d.contains(points))

    def test_geometric_operations_hyperrectangle_2d(self, circle_domain_2d, box_domain_2d, hyperrectangular_regions_2d):
        """Test geometric containment operations of hyperrectangle in 2D."""
        union = IntersectionDomain([circle_domain_2d, box_domain_2d])

        # Test with hyperrectangular regions
        for region_data in hyperrectangular_regions_2d:
            center = region_data["center"]
            radius = region_data["radius"]

            contains = union.contains_hyperrect(center, radius)
            contains_box = box_domain_2d.contains_hyperrect(center, radius)
            contains_circle = circle_domain_2d.contains_hyperrect(center, radius)
            assert contains == (contains_box and contains_circle)

    def test_geometric_operations_hyperrectangle_3d(self, circle_domain_3d, box_domain_3d, hyperrectangular_regions_3d):
        """Test geometric containment operations of hyperrectangle in 3D."""
        union = IntersectionDomain([circle_domain_3d, box_domain_3d])

        # Test with hyperrectangular regions
        for region_data in hyperrectangular_regions_3d:
            center = region_data["center"]
            radius = region_data["radius"]

            contains = union.contains_hyperrect(center, radius)
            contains_box = box_domain_3d.contains_hyperrect(center, radius)
            contains_circle = circle_domain_3d.contains_hyperrect(center, radius)
            assert contains == (contains_box and contains_circle)

    def test_geometric_operations_simplicial_2d(self, circle_domain_2d, box_domain_2d, simplicial_regions_2d):
        """Test geometric containment operations of simplicies in 2D."""
        union = IntersectionDomain([circle_domain_2d, box_domain_2d])

        # Test with simplicial regions
        for region_data in simplicial_regions_2d:
            vertices = region_data["vertices"]

            contains = union.contains_simplex(vertices)
            contains_box = box_domain_2d.contains_simplex(vertices)
            contains_circle = circle_domain_2d.contains_simplex(vertices)
            assert contains == (contains_box and contains_circle)

    def test_geometric_operations_simplicial_3d(self, circle_domain_3d, box_domain_3d, simplicial_regions_3d):
        """Test geometric containment operations of simplicies in 3D."""
        union = IntersectionDomain([circle_domain_3d, box_domain_3d])

        # Test with simplicial regions
        for region_data in simplicial_regions_3d:
            vertices = region_data["vertices"]

            contains = union.contains_simplex(vertices)
            contains_box = box_domain_3d.contains_simplex(vertices)
            contains_circle = circle_domain_3d.contains_simplex(vertices)
            assert contains == (contains_box and contains_circle)


class TestComplementDomain:
    """Test ComplementDomain functionality."""

    def test_initialization(self, circle_domain_2d):
        """Test ComplementDomain initialization."""
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        complement = ComplementDomain(circle_domain_2d, bounds)

        assert complement.dim == 2
        assert complement.domain is circle_domain_2d
        assert complement.bounds == bounds

        # Mismatched dimensions should raise error
        bounds_3d = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
        with pytest.raises(ValueError, match="dimension.*must match"):
            ComplementDomain(circle_domain_2d, bounds_3d)

    def test_contains(self, circle_domain_2d):
        """Test complement containment logic."""
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        complement = ComplementDomain(circle_domain_2d, bounds)

        # Point outside circle but inside bounds
        point_complement = np.array([1.5, 1.5])
        assert not circle_domain_2d.contains(point_complement)
        assert complement.bounding_box.contains(point_complement)
        assert complement.contains(point_complement)

        # Point inside circle
        point_circle = np.array([0.0, 0.0])
        assert circle_domain_2d.contains(point_circle)
        assert not complement.contains(point_circle)

        # Point outside bounds
        point_outside = np.array([3.0, 3.0])
        assert not complement.bounding_box.contains(point_outside)
        assert not complement.contains(point_outside)

    def test_constraint_function(self, circle_domain_2d):
        """Test complement constraint function."""
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        complement = ComplementDomain(circle_domain_2d, bounds)

        # Point in complement should have positive constraint
        point_complement = np.array([1.5, 1.5])
        constraint = complement.constraint(point_complement)
        assert constraint > 0

        # Point in original domain should have negative constraint
        point_original = np.array([0.0, 0.0])
        constraint = complement.constraint(point_original)
        assert constraint < 0

        # Point outside bounds should have negative constraint
        point_outside = np.array([3.0, 3.0])
        constraint = complement.constraint(point_outside)
        assert constraint < 0

    def test_sampling(self, circle_domain_2d):
        """Test complement sampling."""
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        complement = ComplementDomain(circle_domain_2d, bounds)

        points = complement.sample_points(100)

        # All points should be in complement
        assert np.all(complement.contains(points))

        # All points should be outside original domain
        assert np.all(~circle_domain_2d.contains(points))

        # All points should be inside bounding box
        assert np.all(complement.bounding_box.contains(points))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestDomainParsing:
    """Test domain parsing from dictionary definitions."""

    def test_parse_circle_domain(self):
        """Test parsing circle domain from dictionary."""
        domain_def = {"type": "circle", "center": [1.0, -0.5], "radius": 2.0}
        input_domain = [[-5.0, 5.0], [-5.0, 5.0]]

        domain = parse_domain_definition(domain_def, input_domain)

        assert isinstance(domain, CircleDomain)
        assert domain.dim == 2
        np.testing.assert_allclose(domain.center, [1.0, -0.5])
        assert domain.radius == 2.0

    def test_parse_circle_exterior(self):
        """Test parsing circle exterior (complement) domain."""
        domain_def = {"type": "circle_exterior", "center": [0.0, 0.0], "radius": 1.0}
        input_domain = [[-2.0, 2.0], [-2.0, 2.0]]

        domain = parse_domain_definition(domain_def, input_domain)

        assert isinstance(domain, ComplementDomain)
        assert isinstance(domain.domain, CircleDomain)
        assert domain.dim == 2

    def test_parse_box_domain(self):
        """Test parsing box domain from dictionary."""
        domain_def = {"type": "box", "bounds": [[-1.0, 1.0], [-0.5, 1.5]]}
        input_domain = [[-5.0, 5.0], [-5.0, 5.0]]

        domain = parse_domain_definition(domain_def, input_domain)

        assert isinstance(domain, BoxDomain)
        assert domain.dim == 2
        assert domain.bounds == [[-1.0, 1.0], [-0.5, 1.5]]

    def test_parse_union_domain(self):
        """Test parsing union domain from dictionary."""
        domain_def = {"type": "union", "regions": [{"type": "circle", "center": [0.0, 0.0], "radius": 1.0}, {"type": "box", "bounds": [[-0.5, 0.5], [-0.5, 0.5]]}]}
        input_domain = [[-5.0, 5.0], [-5.0, 5.0]]

        domain = parse_domain_definition(domain_def, input_domain)

        assert isinstance(domain, UnionDomain)
        assert len(domain.domains) == 2
        assert isinstance(domain.domains[0], CircleDomain)
        assert isinstance(domain.domains[1], BoxDomain)

    def test_parse_intersection_domain(self):
        """Test parsing intersection domain from dictionary."""
        domain_def = {"type": "intersection", "regions": [{"type": "circle", "center": [0.0, 0.0], "radius": 1.5}, {"type": "box", "bounds": [[-1.0, 1.0], [-1.0, 1.0]]}]}
        input_domain = [[-5.0, 5.0], [-5.0, 5.0]]

        domain = parse_domain_definition(domain_def, input_domain)

        assert isinstance(domain, IntersectionDomain)
        assert len(domain.domains) == 2

    def test_parse_complement_domain(self):
        """Test parsing complement domain from dictionary."""
        domain_def = {"type": "complement", "of": {"type": "circle", "center": [0.0, 0.0], "radius": 1.0}}
        input_domain = [[-2.0, 2.0], [-2.0, 2.0]]

        domain = parse_domain_definition(domain_def, input_domain)

        assert isinstance(domain, ComplementDomain)
        assert isinstance(domain.domain, CircleDomain)

    def test_parse_unknown_domain_type(self):
        """Test parsing unknown domain type raises error."""
        domain_def = {"type": "unknown"}
        input_domain = [[-1.0, 1.0], [-1.0, 1.0]]

        with pytest.raises(ValueError, match="Unknown domain type"):
            parse_domain_definition(domain_def, input_domain)


class TestDomainVisualization:
    """Test domain visualization functionality."""

    def test_visualize_domain_2d_no_matplotlib(self, circle_domain_2d):
        """Test visualization when matplotlib is not available."""
        # Mock import failure to simulate missing matplotlib
        with patch("builtins.__import__", side_effect=ImportError("No module named 'matplotlib'")):
            result = visualize_domain_2d(circle_domain_2d, [[-2, 2], [-2, 2]])
            assert result is None

    @pytest.mark.skipif(True, reason="Matplotlib import may not be available in test environment")
    def test_visualize_domain_2d_with_matplotlib(self, circle_domain_2d):
        """Test visualization when matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            bounds = [[-2.0, 2.0], [-2.0, 2.0]]
            ax = visualize_domain_2d(circle_domain_2d, bounds, resolution=50)
            assert ax is not None
            plt.close("all")  # Clean up
        except ImportError:
            pytest.skip("Matplotlib not available")


class TestUnsafeRegion:
    """Test unsafe region checking functionality."""

    def test_unsafe_region_intersection(self, circle_domain_2d):
        """Test unsafe region intersection checking."""
        # Mock dynamics with unsafe domain
        dynamics = Mock()
        dynamics.unsafe_domain = circle_domain_2d

        # Mock sample region that intersects
        sample = Mock()
        sample.intersects_domain = Mock(return_value=True)
        sample.contained_in_domain = Mock(return_value=False)

        # Test intersection check
        result = unsafe_region(sample, dynamics, require_complete_containment=False)
        assert result is True
        sample.intersects_domain.assert_called_once_with(circle_domain_2d)

        # Test containment check
        result = unsafe_region(sample, dynamics, require_complete_containment=True)
        assert result is False
        sample.contained_in_domain.assert_called_once_with(circle_domain_2d)

    def test_unsafe_region_no_unsafe_domain(self):
        """Test unsafe region when no unsafe domain is defined."""
        dynamics = Mock()
        dynamics.unsafe_domain = None

        sample = Mock()

        result = unsafe_region(sample, dynamics)
        assert result is False

    def test_unsafe_region_invalid_domain_type(self):
        """Test unsafe region with invalid domain type."""
        dynamics = Mock()
        dynamics.unsafe_domain = "not a domain"

        sample = Mock()

        with pytest.raises(ValueError, match="unsafe_domain must be a Domain object"):
            unsafe_region(sample, dynamics)


# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================


class TestMathematicalProperties:
    """Test mathematical properties and consistency."""

    def test_constraint_boundary_consistency(self, circle_domains_various, box_domains_various, tolerance):
        """Test that constraint function is consistent with containment."""
        all_domains = circle_domains_various + box_domains_various

        for domain in all_domains:
            # Generate test points
            points = domain.sample_points(100)

            # Add some points outside domain
            if isinstance(domain, CircleDomain):
                outside_points = domain.center + 2 * domain.radius * np.random.randn(50, domain.dim)
            else:  # BoxDomain
                # Generate points outside bounds
                range_size = domain.high_bounds - domain.low_bounds
                outside_points = (domain.low_bounds + domain.high_bounds) / 2
                outside_points = outside_points + 2 * range_size * (np.random.rand(50, domain.dim) - 0.5)

            all_test_points = np.vstack([points, outside_points])

            # Check consistency
            for point in all_test_points:
                contains = domain.contains(point)
                constraint = domain.constraint(point)

                if contains:
                    assert constraint >= -tolerance, f"Constraint should be >= 0 for contained point, got {constraint}"
                else:
                    assert constraint <= tolerance, f"Constraint should be <= 0 for non-contained point, got {constraint}"

    def test_sampling_containment_consistency(self, circle_domains_various, box_domains_various):
        """Test that all sampled points are actually contained in domain."""
        all_domains = circle_domains_various + box_domains_various

        for domain in all_domains:
            points = domain.sample_points(200)

            # All sampled points must be contained
            containment = domain.contains(points)
            assert np.all(containment), f"Sampled points not contained in {type(domain).__name__}"

            # All sampled points must have non-negative constraint
            constraints = []
            for point in points:
                constraint = domain.constraint(point)
                constraints.append(constraint)

            constraints = np.array(constraints)
            assert np.all(constraints >= -1e-10), f"Sampled points have negative constraints in {type(domain).__name__}"

    def test_geometric_consistency(self, circle_domain_2d, box_domain_2d, hyperrectangular_regions_2d):
        """Test consistency between different geometric operations."""
        domains = [circle_domain_2d, box_domain_2d]

        for domain in domains:
            for region_data in hyperrectangular_regions_2d:
                center = region_data["center"]
                radius = region_data["radius"]

                intersects = domain.intersects_hyperrect(center, radius)
                contains = domain.contains_hyperrect(center, radius)

                # If domain contains region, it must intersect
                if contains:
                    assert intersects, f"Contains implies intersects failed for {type(domain).__name__}"

                # Test via point sampling for verification
                if intersects:
                    # Generate some points in the hyperrect
                    test_points = np.random.uniform(low=center - radius, high=center + radius, size=(100, 2))

                    # At least some should be in domain (if intersection is non-trivial)
                    in_domain = domain.contains(test_points)
                    # This is a probabilistic test, so we're lenient
                    if np.sum(in_domain) == 0:
                        warnings.warn(f"No sampled points in intersection for {type(domain).__name__}")

    def test_union_intersection_properties(self, circle_domain_2d, box_domain_2d):
        """Test mathematical properties of union and intersection."""
        # Create test domains
        union = UnionDomain([circle_domain_2d, box_domain_2d])
        intersection = IntersectionDomain([circle_domain_2d, box_domain_2d])

        # Generate test points
        test_points = np.random.uniform(-2, 2, size=(100, 2))

        for point in test_points:
            in_circle = circle_domain_2d.contains(point)
            in_box = box_domain_2d.contains(point)
            in_union = union.contains(point)
            in_intersection = intersection.contains(point)

            # Union: should be in union if in either domain
            assert in_union == (in_circle or in_box)

            # Intersection: should be in intersection if in both domains
            assert in_intersection == (in_circle and in_box)

            # Constraint consistency
            union_constraint = union.constraint(point)
            intersection_constraint = intersection.constraint(point)

            if in_union:
                assert union_constraint >= -1e-10
            else:
                assert union_constraint <= 1e-10

            if in_intersection:
                assert intersection_constraint >= -1e-10
            else:
                assert intersection_constraint <= 1e-10

    def test_complement_properties(self, circle_domain_2d):
        """Test mathematical properties of complement domain."""
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        complement = ComplementDomain(circle_domain_2d, bounds)

        # Generate test points
        test_points = np.random.uniform(-3, 3, size=(200, 2))

        for point in test_points:
            in_circle = circle_domain_2d.contains(point, translator=NumpyTranslator())
            in_bounds = complement.bounding_box.contains(point, translator=NumpyTranslator())
            in_complement = complement.contains(point, translator=NumpyTranslator())

            # Complement: should be in complement if in bounds but not in circle
            expected = in_bounds and not in_circle
            assert in_complement == expected

            # Constraint consistency
            constraint = complement.constraint(point, NumpyTranslator())
            if in_complement:
                assert constraint >= -1e-10
            else:
                assert constraint <= 1e-10


# =============================================================================
# STRESS TESTS AND EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and stress scenarios."""

    def test_very_small_domains(self):
        """Test domains with very small dimensions."""
        # Very small circle
        tiny_circle = CircleDomain(center=[0.0, 0.0], radius=1e-10)
        assert tiny_circle.radius == 1e-10

        # Point at center should be contained
        assert tiny_circle.contains(np.array([0.0, 0.0]))

        # Very small box
        tiny_box = BoxDomain(bounds=[[-1e-10, 1e-10], [-1e-10, 1e-10]])
        assert tiny_box.contains(np.array([0.0, 0.0]))

    def test_very_large_domains(self):
        """Test domains with very large dimensions."""
        # Very large circle
        large_circle = CircleDomain(center=[0.0, 0.0], radius=1e6)
        assert large_circle.contains(np.array([1e5, 1e5]))

        # Very large box
        large_box = BoxDomain(bounds=[[-1e6, 1e6], [-1e6, 1e6]])
        assert large_box.contains(np.array([1e5, 1e5]))

    def test_high_dimensional_domains(self):
        """Test domains in high dimensions."""
        # 10D circle
        center_10d = np.zeros(10)
        circle_10d = CircleDomain(center=center_10d.tolist(), radius=1.0)

        assert circle_10d.dim == 10
        assert circle_10d.contains(np.zeros(10))

        # Sample some points
        points = circle_10d.sample_points(50)
        assert points.shape == (50, 10)
        assert np.all(circle_10d.contains(points))

        # 10D box
        bounds_10d = [[-1.0, 1.0] for _ in range(10)]
        box_10d = BoxDomain(bounds=bounds_10d)

        assert box_10d.dim == 10
        assert box_10d.contains(np.zeros(10))

    def test_degenerate_shapes(self):
        """Test degenerate or edge case shapes."""
        # Very flat box (almost 1D)
        flat_box = BoxDomain(bounds=[[-1.0, 1.0], [-1e-10, 1e-10]])
        assert flat_box.contains(np.array([0.0, 0.0]))

        # Single point sampling should work
        points = flat_box.sample_points(10)
        assert points.shape == (10, 2)
        assert np.all(np.abs(points[:, 1]) <= 1e-10)

    def test_numerical_stability(self):
        """Test numerical stability in edge cases."""
        # Domain at machine precision limits
        eps = np.finfo(np.float64).eps

        # Circle with radius near machine epsilon
        tiny_circle = CircleDomain(center=[0.0, 0.0], radius=eps * 100)

        # Should still work for basic operations
        assert tiny_circle.contains(np.array([0.0, 0.0]))
        constraint = tiny_circle.constraint(np.array([0.0, 0.0]))
        assert constraint > 0

    def test_empty_sampling(self):
        """Test sampling edge cases."""
        circle = CircleDomain(center=[0.0, 0.0], radius=1.0)

        # Zero samples
        points = circle.sample_points(0)
        assert points.shape == (0, 2)

        # Negative samples (should raise TypeError)
        with pytest.raises(TypeError, match="num_points must be a non-negative integer"):
            circle.sample_points(-5)

        # Invalid input type
        with pytest.raises(TypeError, match="num_points must be a non-negative integer"):
            circle.sample_points(5.5)

    def test_domain_validation_improvements(self):
        """Test improved domain validation."""
        # Test repr functionality
        circle = CircleDomain(center=[0.0, 0.0], radius=1.0)
        repr_str = repr(circle)
        assert "CircleDomain" in repr_str
        assert "dim=2" in repr_str

        # Test zero radius sampling validation
        zero_circle = CircleDomain(center=[0.0, 0.0], radius=0.0)
        with pytest.raises(ValueError, match="Cannot sample from domain with non-positive radius"):
            zero_circle.sample_points(10)

        # Test negative radius sampling validation
        neg_circle = CircleDomain(center=[0.0, 0.0], radius=-1.0)
        with pytest.raises(ValueError, match="Cannot sample from domain with non-positive radius"):
            neg_circle.sample_points(10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_tensor_edge_cases(self):
        """Test edge cases with PyTorch tensors."""
        circle = CircleDomain(center=[0.0, 0.0], radius=1.0)

        # Empty tensor
        empty_tensor = torch.empty(0, 2)
        result = circle.contains(empty_tensor, translator=TorchTranslator(device=empty_tensor.device))
        assert result.shape == (0,)

        # Single point tensor
        single_point = torch.tensor([[0.0, 0.0]])
        result = circle.contains(single_point, translator=TorchTranslator(device=single_point.device))
        assert result.shape == (1,)
        assert result[0] == True

        # Test with requires_grad
        point_grad = torch.tensor([[0.5, 0.5]], requires_grad=True)
        result = circle.contains(point_grad, translator=TorchTranslator(device=point_grad.device))
        assert result.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__])
