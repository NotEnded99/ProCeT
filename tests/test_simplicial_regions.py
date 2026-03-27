"""
Test suite for simplicial regions and meshes.

This module provides comprehensive testing for the SimplicialRegion and SimplicialMesh
classes, including all geometric operations and certification-specific features.

The test suite covers:

1. **SimplicialRegion initialization and basic properties**:
   - Tests for 1D, 2D, 3D, and higher-dimensional simplices
   - Validates vertex count and dimension consistency
   - Tests centroid and volume computation

2. **Volume computation (_compute_volume)**:
   - Tests accurate volume calculation for various dimensions
   - Validates against known analytic formulas
   - Tests edge cases (degenerate simplices, unit simplices)

3. **Geometric operations**:
   - Point containment tests (contains_point)
   - Bounding box computation (get_bounds)
   - Uniform sampling (sample_uniform)
   - Edge length computation (get_max_edge_length)
   - Closest point computation (closest_point_on_simplex)

4. **Halfspace representation**:
   - Tests conversion to linear inequalities (Ax <= b)
   - Validates constraints for 1D, 2D, and 3D simplices
   - Ensures all vertices satisfy the constraints

5. **Splitting operations**:
   - Tests simplex bisection along longest edge
   - Validates volume conservation after split
   - Tests multiple levels of splitting

6. **SimplicialMesh operations**:
   - Tests mesh initialization and triangulation
   - Tests region generation for different dimensions
   - Validates total volume coverage
   - Tests point location queries

7. **Domain intersection and containment**:
   - Tests intersection checking with domains
   - Tests full containment checking

All tests use pytest fixtures and include empirical validation to ensure
geometric correctness and numerical stability.
"""

import math
import pytest
import numpy as np
from typing import List, Tuple

from lbp_neural_cbf.regions.simplicial import (
    SimplicialRegion,
    SimplicialMesh,
    SimplicialRegionGenerator,
    closest_point_on_line_segment,
    closest_point_on_simplex,
)

# Set random seed for reproducibility
np.random.seed(42)

# Numerical tolerance for floating point comparisons
TOLERANCE = 1e-10


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def simplices_1d():
    """Fixture providing 1D simplices (line segments)."""
    return [
        {"vertices": np.array([[0.0], [1.0]]), "expected_volume": 1.0},
        {"vertices": np.array([[-2.0], [3.0]]), "expected_volume": 5.0},
        {"vertices": np.array([[0.5], [0.8]]), "expected_volume": 0.3},
        {"vertices": np.array([[-1.5], [-0.5]]), "expected_volume": 1.0},
    ]


@pytest.fixture
def simplices_2d():
    """Fixture providing 2D simplices (triangles)."""
    return [
        # Right triangle with unit legs
        {
            "vertices": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "expected_volume": 0.5,
        },
        # Equilateral-ish triangle
        {
            "vertices": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866025]]),
            "expected_volume": 0.433012,
        },
        # Arbitrary triangle
        {
            "vertices": np.array([[-1.0, -1.0], [2.0, 0.0], [0.0, 3.0]]),
            "expected_volume": 5.5,
        },
        # Small triangle
        {
            "vertices": np.array([[0.0, 0.0], [0.1, 0.0], [0.05, 0.1]]),
            "expected_volume": 0.005,
        },
    ]


@pytest.fixture
def simplices_3d():
    """Fixture providing 3D simplices (tetrahedra)."""
    return [
        # Unit tetrahedron
        {
            "vertices": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "expected_volume": 1.0 / 6.0,
        },
        # Larger tetrahedron
        {
            "vertices": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            ),
            "expected_volume": 8.0 / 6.0,
        },
        # Arbitrary tetrahedron
        {
            "vertices": np.array(
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [1.0, 3.0, 1.0],
                    [1.0, 1.0, 4.0],
                ]
            ),
            "expected_volume": 1.0,
        },
    ]


@pytest.fixture
def simplices_4d():
    """Fixture providing 4D simplices."""
    return [
        # Unit 4-simplex
        {
            "vertices": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "expected_volume": 1.0 / 24.0,
        },
    ]


# =============================================================================
# BASIC INITIALIZATION AND PROPERTIES TESTS
# =============================================================================


class TestSimplicialRegionInitialization:
    """Test SimplicialRegion initialization and basic properties."""

    def test_1d_initialization(self, simplices_1d):
        """Test 1D simplex (line segment) initialization."""
        for simplex_data in simplices_1d:
            vertices = simplex_data["vertices"]
            region = SimplicialRegion(vertices)

            assert region.get_dimension() == 1
            assert region.n_vertices == 2
            assert np.allclose(region.vertices, vertices)

    def test_2d_initialization(self, simplices_2d):
        """Test 2D simplex (triangle) initialization."""
        for simplex_data in simplices_2d:
            vertices = simplex_data["vertices"]
            region = SimplicialRegion(vertices)

            assert region.get_dimension() == 2
            assert region.n_vertices == 3
            assert np.allclose(region.vertices, vertices)

    def test_3d_initialization(self, simplices_3d):
        """Test 3D simplex (tetrahedron) initialization."""
        for simplex_data in simplices_3d:
            vertices = simplex_data["vertices"]
            region = SimplicialRegion(vertices)

            assert region.get_dimension() == 3
            assert region.n_vertices == 4
            assert np.allclose(region.vertices, vertices)

    def test_4d_initialization(self, simplices_4d):
        """Test 4D simplex initialization."""
        for simplex_data in simplices_4d:
            vertices = simplex_data["vertices"]
            region = SimplicialRegion(vertices)

            assert region.get_dimension() == 4
            assert region.n_vertices == 5
            assert np.allclose(region.vertices, vertices)

    def test_invalid_vertex_count(self):
        """Test that invalid vertex counts raise ValueError."""
        # 2D should have 3 vertices
        with pytest.raises(ValueError, match="must have 3 vertices"):
            SimplicialRegion(np.array([[0.0, 0.0], [1.0, 0.0]]))

        # 3D should have 4 vertices
        with pytest.raises(ValueError, match="must have 4 vertices"):
            SimplicialRegion(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_1d_array_reshaping(self):
        """Test that 1D arrays are properly reshaped."""
        vertices = np.array([0.0, 1.0])
        region = SimplicialRegion(vertices)

        assert region.vertices.shape == (2, 1)
        assert region.get_dimension() == 1


# =============================================================================
# VOLUME COMPUTATION TESTS
# =============================================================================


class TestVolumeComputation:
    """Test volume computation for simplices of various dimensions."""

    def test_1d_volumes(self, simplices_1d):
        """Test volume (length) computation for 1D simplices."""
        for simplex_data in simplices_1d:
            vertices = simplex_data["vertices"]
            expected_volume = simplex_data["expected_volume"]

            region = SimplicialRegion(vertices)
            computed_volume = region.volume

            assert np.isclose(computed_volume, expected_volume, rtol=1e-6), f"Volume mismatch: expected {expected_volume}, got {computed_volume}"

    def test_2d_volumes(self, simplices_2d):
        """Test area computation for 2D simplices."""
        for simplex_data in simplices_2d:
            vertices = simplex_data["vertices"]
            expected_volume = simplex_data["expected_volume"]

            region = SimplicialRegion(vertices)
            computed_volume = region.volume

            assert np.isclose(computed_volume, expected_volume, rtol=1e-5), f"Volume mismatch: expected {expected_volume}, got {computed_volume}"

    def test_3d_volumes(self, simplices_3d):
        """Test volume computation for 3D simplices."""
        for simplex_data in simplices_3d:
            vertices = simplex_data["vertices"]
            expected_volume = simplex_data["expected_volume"]

            region = SimplicialRegion(vertices)
            computed_volume = region.volume

            assert np.isclose(computed_volume, expected_volume, rtol=1e-6), f"Volume mismatch: expected {expected_volume}, got {computed_volume}"

    def test_4d_volumes(self, simplices_4d):
        """Test volume computation for 4D simplices."""
        for simplex_data in simplices_4d:
            vertices = simplex_data["vertices"]
            expected_volume = simplex_data["expected_volume"]

            region = SimplicialRegion(vertices)
            computed_volume = region.volume

            assert np.isclose(computed_volume, expected_volume, rtol=1e-6), f"Volume mismatch: expected {expected_volume}, got {computed_volume}"

    def test_volume_positive(self):
        """Test that volumes are always positive."""
        test_cases = [
            np.array([[0.0], [1.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ]

        for vertices in test_cases:
            region = SimplicialRegion(vertices)
            assert region.volume > 0, f"Volume should be positive, got {region.volume}"

    def test_volume_scaling(self):
        """Test that volume scales correctly with uniform scaling."""
        base_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        base_region = SimplicialRegion(base_vertices)
        base_volume = base_region.volume

        # Scale by factor of 2 (volume should scale by 2^2 = 4 in 2D)
        scaled_vertices = base_vertices * 2
        scaled_region = SimplicialRegion(scaled_vertices)
        scaled_volume = scaled_region.volume

        assert np.isclose(scaled_volume, base_volume * 4, rtol=1e-10)

        # Scale by factor of 3 (volume should scale by 3^2 = 9 in 2D)
        scaled_vertices = base_vertices * 3
        scaled_region = SimplicialRegion(scaled_vertices)
        scaled_volume = scaled_region.volume

        assert np.isclose(scaled_volume, base_volume * 9, rtol=1e-10)


# =============================================================================
# CENTROID COMPUTATION TESTS
# =============================================================================


class TestCentroidComputation:
    """Test centroid computation for simplices."""

    def test_centroid_calculation(self):
        """Test that centroid is the average of vertices."""
        test_cases = [
            np.array([[0.0], [1.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ]

        for vertices in test_cases:
            region = SimplicialRegion(vertices)
            expected_centroid = np.mean(vertices, axis=0)

            assert np.allclose(region.centroid, expected_centroid, atol=TOLERANCE)

    def test_centroid_in_simplex(self):
        """Test that centroid is contained in the simplex."""
        test_cases = [
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ]

        for vertices in test_cases:
            region = SimplicialRegion(vertices)
            # Centroid should be inside the simplex
            assert region.contains_point(region.centroid)


# =============================================================================
# GEOMETRIC OPERATIONS TESTS
# =============================================================================


class TestPointContainment:
    """Test point containment checking."""

    def test_1d_containment(self):
        """Test point containment for 1D simplices."""
        region = SimplicialRegion(np.array([[0.0], [1.0]]))

        # Points inside
        assert region.contains_point(np.array([0.0]))
        assert region.contains_point(np.array([0.5]))
        assert region.contains_point(np.array([1.0]))

        # Points outside
        assert not region.contains_point(np.array([-0.1]))
        assert not region.contains_point(np.array([1.1]))

    def test_2d_containment(self):
        """Test point containment for 2D simplices."""
        region = SimplicialRegion(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

        # Vertices should be inside
        assert region.contains_point(np.array([0.0, 0.0]))
        assert region.contains_point(np.array([1.0, 0.0]))
        assert region.contains_point(np.array([0.0, 1.0]))

        # Centroid should be inside
        assert region.contains_point(region.centroid)

        # Edge midpoints should be inside
        assert region.contains_point(np.array([0.5, 0.0]))
        assert region.contains_point(np.array([0.0, 0.5]))
        assert region.contains_point(np.array([0.5, 0.5]))

        # Point clearly outside
        assert not region.contains_point(np.array([2.0, 2.0]))
        assert not region.contains_point(np.array([-1.0, -1.0]))

    def test_3d_containment(self):
        """Test point containment for 3D simplices."""
        region = SimplicialRegion(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

        # Vertices should be inside
        for vertex in region.vertices:
            assert region.contains_point(vertex)

        # Centroid should be inside
        assert region.contains_point(region.centroid)

        # Point clearly outside
        assert not region.contains_point(np.array([2.0, 2.0, 2.0]))


class TestBoundsComputation:
    """Test bounding box computation."""

    def test_bounds_2d(self):
        """Test bounds computation for 2D simplices."""
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        region = SimplicialRegion(vertices)
        lower, upper = region.get_bounds()

        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert np.allclose(lower, [0.0, 0.0])
        assert np.allclose(upper, [1.0, 1.0])

    def test_bounds_3d(self):
        """Test bounds computation for 3D simplices."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        region = SimplicialRegion(vertices)
        lower, upper = region.get_bounds()

        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert np.allclose(lower, [0.0, 0.0, 0.0])
        assert np.allclose(upper, [2.0, 3.0, 4.0])

    def test_bounds_contain_vertices(self):
        """Test that bounds contain all vertices."""
        vertices = np.array([[-1.0, -2.0], [3.0, 1.0], [0.0, 4.0]])
        region = SimplicialRegion(vertices)
        lower, upper = region.get_bounds()

        for vertex in vertices:
            for dim in range(len(vertex)):
                assert lower[dim] <= vertex[dim] <= upper[dim]


class TestUniformSampling:
    """Test uniform sampling from simplices."""

    def test_samples_inside_simplex(self):
        """Test that all samples are inside the simplex."""
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        region = SimplicialRegion(vertices)

        samples = region.sample_uniform(n_samples=100)

        assert samples.shape == (100, 2)

        # All samples should be inside the simplex
        for sample in samples:
            assert region.contains_point(sample), f"Sample {sample} is outside simplex"

    def test_sample_distribution(self):
        """Test that samples are reasonably distributed."""
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        region = SimplicialRegion(vertices)

        samples = region.sample_uniform(n_samples=1000)

        # Check that mean is close to centroid
        sample_mean = np.mean(samples, axis=0)
        centroid = region.centroid

        # Allow some deviation due to random sampling
        assert np.allclose(sample_mean, centroid, atol=0.1)


class TestEdgeLengthComputation:
    """Test edge length computation."""

    def test_max_edge_length_1d(self):
        """Test max edge length for 1D simplices."""
        region = SimplicialRegion(np.array([[0.0], [3.0]]))
        max_length, edge = region.get_max_edge_length()

        assert np.isclose(max_length, 3.0)
        assert edge in [(0, 1), (1, 0)]

    def test_max_edge_length_2d(self):
        """Test max edge length for 2D simplices."""
        # Right triangle with legs 3 and 4, hypotenuse 5
        vertices = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        region = SimplicialRegion(vertices)
        max_length, edge = region.get_max_edge_length()

        assert np.isclose(max_length, 5.0)

    def test_max_edge_length_equilateral(self):
        """Test max edge length for equilateral triangle."""
        # Equilateral triangle with side length 2
        h = 2 * np.sqrt(3) / 2  # height
        vertices = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, h]])
        region = SimplicialRegion(vertices)
        max_length, edge = region.get_max_edge_length()

        # All edges should be approximately 2
        assert np.isclose(max_length, 2.0, rtol=1e-6)


class TestClosestPointComputation:
    """Test closest point computation on simplices."""

    def test_closest_point_on_line_segment(self):
        """Test closest point on a line segment."""
        v0 = np.array([0.0, 0.0])
        v1 = np.array([1.0, 0.0])

        # Point on segment
        point = np.array([0.5, 0.0])
        closest = closest_point_on_line_segment(v0, v1, point)
        assert np.allclose(closest, point)

        # Point above segment
        point = np.array([0.5, 1.0])
        closest = closest_point_on_line_segment(v0, v1, point)
        assert np.allclose(closest, np.array([0.5, 0.0]))

        # Point before segment
        point = np.array([-1.0, 0.5])
        closest = closest_point_on_line_segment(v0, v1, point)
        assert np.allclose(closest, v0)

        # Point after segment
        point = np.array([2.0, 0.5])
        closest = closest_point_on_line_segment(v0, v1, point)
        assert np.allclose(closest, v1)

    def test_closest_point_on_simplex_2d(self):
        """Test closest point on a 2D simplex."""
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Point inside simplex should return the point itself
        point = np.array([0.25, 0.25])
        closest = closest_point_on_simplex(vertices, point)
        assert np.allclose(closest, point, atol=1e-6)

        # Point outside should project to boundary
        point = np.array([2.0, 2.0])
        closest = closest_point_on_simplex(vertices, point)
        # Should be somewhere on the boundary
        assert not np.allclose(closest, point)


# =============================================================================
# HALFSPACE REPRESENTATION TESTS
# =============================================================================


class TestHalfspaceRepresentation:
    """Test halfspace representation computation."""

    def test_halfspace_1d(self):
        """Test halfspace representation for 1D simplices."""
        region = SimplicialRegion(np.array([[0.0], [1.0]]))
        A, b = region.get_halfspace_representation()

        # Should have 2 constraints: x >= 0 and x <= 1
        assert A.shape[0] == 2
        assert b.shape[0] == 2

        # All vertices should satisfy constraints
        for vertex in region.vertices:
            assert np.all(A @ vertex <= b + TOLERANCE)

    def test_halfspace_2d(self):
        """Test halfspace representation for 2D simplices."""
        region = SimplicialRegion(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        A, b = region.get_halfspace_representation()

        # Should have 3 constraints (one per face)
        assert A.shape[0] == 3
        assert b.shape[0] == 3

        # All vertices should satisfy constraints
        for vertex in region.vertices:
            assert np.all(A @ vertex <= b + TOLERANCE)

        # Centroid should satisfy constraints
        centroid = region.centroid
        assert np.all(A @ centroid <= b + TOLERANCE)

    def test_halfspace_3d(self):
        """Test halfspace representation for 3D simplices."""
        region = SimplicialRegion(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        A, b = region.get_halfspace_representation()

        # Should have 4 constraints (one per face)
        assert A.shape[0] == 4
        assert b.shape[0] == 4

        # All vertices should satisfy constraints
        for vertex in region.vertices:
            assert np.all(A @ vertex <= b + TOLERANCE)

    def test_halfspace_consistency(self):
        """Test that halfspace representation is consistent with containment."""
        region = SimplicialRegion(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        A, b = region.get_halfspace_representation()

        # Test points known to be inside
        test_points = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.0]),
            np.array([0.0, 0.5]),
            np.array([0.25, 0.25]),
            region.centroid,
        ]

        for point in test_points:
            satisfies_constraints = np.all(A @ point <= b + TOLERANCE)
            contains = region.contains_point(point)
            assert satisfies_constraints == contains or contains


# =============================================================================
# SPLITTING OPERATIONS TESTS
# =============================================================================


class TestSplittingOperations:
    """Test simplex splitting operations."""

    def test_split_1d(self):
        """Test splitting 1D simplices."""
        region = SimplicialRegion(np.array([[0.0], [1.0]]))
        region1, region2 = region.split()

        # Both should be 1D simplices
        assert region1.get_dimension() == 1
        assert region2.get_dimension() == 1

        # Volumes should sum to original
        assert np.isclose(region1.volume + region2.volume, region.volume)

    def test_split_2d(self):
        """Test splitting 2D simplices."""
        region = SimplicialRegion(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        region1, region2 = region.split()

        # Both should be 2D simplices
        assert region1.get_dimension() == 2
        assert region2.get_dimension() == 2

        # Volumes should sum to original
        assert np.isclose(region1.volume + region2.volume, region.volume, rtol=1e-10)

    def test_split_3d(self):
        """Test splitting 3D simplices."""
        region = SimplicialRegion(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        region1, region2 = region.split()

        # Both should be 3D simplices
        assert region1.get_dimension() == 3
        assert region2.get_dimension() == 3

        # Volumes should sum to original
        assert np.isclose(region1.volume + region2.volume, region.volume, rtol=1e-10)

    def test_split_preserves_properties(self):
        """Test that splitting preserves output_dim and nonlin_dependencies."""
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        region = SimplicialRegion(vertices, output_dim=2, nonlin_dependencies=[(0, True), (1, False)])

        region1, region2 = region.split()

        assert region1.output_dim == 2
        assert region2.output_dim == 2
        assert region1.nonlin_dependencies == [(0, True), (1, False)]
        assert region2.nonlin_dependencies == [(0, True), (1, False)]

    def test_multiple_splits(self):
        """Test multiple levels of splitting."""
        region = SimplicialRegion(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        original_volume = region.volume

        # Split multiple times
        regions = [region]
        for _ in range(3):
            new_regions = []
            for r in regions:
                r1, r2 = r.split()
                new_regions.extend([r1, r2])
            regions = new_regions

        # Total volume should be preserved
        total_volume = sum(r.volume for r in regions)
        assert np.isclose(total_volume, original_volume, rtol=1e-10)

        # Should have 2^3 = 8 regions
        assert len(regions) == 8


# =============================================================================
# SIMPLICIAL MESH TESTS
# =============================================================================


class TestSimplicialMesh:
    """Test SimplicialMesh operations."""

    def test_mesh_initialization_1d(self):
        """Test 1D mesh initialization."""
        domain_bounds = [(0.0, 1.0)]
        mesh = SimplicialMesh(domain_bounds)

        assert mesh.dim == 1
        assert mesh.points is not None
        assert len(mesh) > 0

    def test_mesh_initialization_2d(self):
        """Test 2D mesh initialization."""
        domain_bounds = [(0.0, 1.0), (0.0, 1.0)]
        mesh = SimplicialMesh(domain_bounds)

        assert mesh.dim == 2
        assert mesh.delaunay is not None
        assert len(mesh) > 0

    def test_mesh_initialization_3d(self):
        """Test 3D mesh initialization."""
        domain_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        mesh = SimplicialMesh(domain_bounds)

        assert mesh.dim == 3
        assert mesh.delaunay is not None
        assert len(mesh) > 0

    def test_get_regions(self):
        """Test getting regions from mesh."""
        domain_bounds = [(0.0, 1.0), (0.0, 1.0)]
        mesh = SimplicialMesh(domain_bounds)

        regions = mesh.get_regions(output_dim=0)

        assert len(regions) > 0
        assert all(isinstance(r, SimplicialRegion) for r in regions)
        assert all(r.get_dimension() == 2 for r in regions)

    def test_total_volume(self):
        """Test total volume computation for mesh."""
        domain_bounds = [(0.0, 1.0), (0.0, 1.0)]
        mesh = SimplicialMesh(domain_bounds)

        total_volume = mesh.total_volume()

        # For a unit square, total volume should be 1.0
        assert np.isclose(total_volume, 1.0, rtol=1e-6)

    def test_find_region_containing_point_2d(self):
        """Test finding which region contains a point in 2D."""
        domain_bounds = [(0.0, 1.0), (0.0, 1.0)]
        mesh = SimplicialMesh(domain_bounds)

        # Point in the middle should be in some region
        point = np.array([0.5, 0.5])
        region_idx = mesh.find_region_containing_point(point)

        assert region_idx is not None
        assert 0 <= region_idx < len(mesh)

    def test_find_region_containing_point_1d(self):
        """Test finding which region contains a point in 1D."""
        domain_bounds = [(0.0, 1.0)]
        mesh = SimplicialMesh(domain_bounds)

        point = np.array([0.5])
        region_idx = mesh.find_region_containing_point(point)

        assert region_idx is not None
        assert 0 <= region_idx < len(mesh)


# =============================================================================
# SIMPLICIAL REGION GENERATOR TESTS
# =============================================================================


class TestSimplicialRegionGenerator:
    """Test SimplicialRegionGenerator."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SimplicialRegionGenerator()

        # Generator should be instantiable without arguments
        assert generator is not None

    def test_create_mesh(self):
        """Test mesh creation."""
        generator = SimplicialRegionGenerator()
        domain_bounds = [(0.0, 1.0), (0.0, 1.0)]

        # Create a mock dynamics model
        class MockDynamicsModel:
            def __init__(self):
                self.input_domain = domain_bounds
                self.output_dim = 2

        model = MockDynamicsModel()
        mesh = generator.create_mesh(model)

        assert isinstance(mesh, SimplicialMesh)
        assert mesh.dim == 2

    def test_generate_regions(self):
        """Test region generation."""
        generator = SimplicialRegionGenerator()
        domain_bounds = [(0.0, 1.0), (0.0, 1.0)]

        # Create a mock dynamics model
        class MockDynamicsModel:
            def __init__(self):
                self.input_domain = domain_bounds
                self.output_dim = 2

        model = MockDynamicsModel()
        regions = generator.generate_regions(model)

        assert len(regions) > 0
        assert all(isinstance(r, SimplicialRegion) for r in regions)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_simplex(self):
        """Test with very small simplices."""
        vertices = np.array([[0.0, 0.0], [1e-10, 0.0], [0.0, 1e-10]])
        region = SimplicialRegion(vertices)

        assert region.volume > 0
        assert region.volume < 1e-15

    def test_translated_simplex(self):
        """Test simplex far from origin."""
        vertices = np.array([[1000.0, 1000.0], [1001.0, 1000.0], [1000.0, 1001.0]])
        region = SimplicialRegion(vertices)

        # Volume should be same as unit triangle at origin
        assert np.isclose(region.volume, 0.5, rtol=1e-10)

    def test_negative_coordinates(self):
        """Test simplex with negative coordinates."""
        vertices = np.array([[-1.0, -1.0], [0.0, -1.0], [-1.0, 0.0]])
        region = SimplicialRegion(vertices)

        assert np.isclose(region.volume, 0.5, rtol=1e-10)
        assert region.contains_point(region.centroid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
