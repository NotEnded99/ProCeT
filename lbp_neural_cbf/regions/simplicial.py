"""
Simplicial region and mesh implementations.

This module provides concrete implementations for simplicial regions
and Delaunay-based meshes, following the unified region interface.
"""

import math
from itertools import combinations
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay

from ..translators import NumpyTranslator

from .base import AbstractMesh, AbstractRegion, AbstractRegionGenerator


def closest_point_on_line_segment(v0: np.ndarray, v1: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Find the closest point on a line segment to a given point."""
    edge_vec = v1 - v0
    edge_len_sq = np.dot(edge_vec, edge_vec)

    if edge_len_sq < 1e-12:  # Degenerate edge
        return v0

    # Project point onto line and clamp to segment
    t = np.dot(point - v0, edge_vec) / edge_len_sq
    t = np.clip(t, 0, 1)

    return v0 + t * edge_vec


def closest_point_on_simplex(vertices: np.ndarray, point: np.ndarray):
    """
    Recursively find the closest point on a simplex to a given point using NumPy's lstsq,
    with active constraint analysis to reduce recursion.

    Parameters:
    - vertices: np.ndarray of shape (m, n), m = n+1 vertices of the simplex in nD
    - point: np.ndarray of shape (n,), the external point
    Returns:
    - closest: np.ndarray of shape (n,), the closest point on the simplex
    """
    m, n = vertices.shape

    # If the simplex is a single point
    if m == 1:
        return vertices[0]

    # Construct matrix A and offset v0
    v0 = vertices[0]
    A = (vertices[1:] - v0).T  # shape (n, m-1)

    # Solve least squares: find alpha such that x = v0 + A @ alpha is close to point
    b = point - v0
    alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    x_proj = v0 + A @ alpha

    # Compute barycentric coordinates
    lmbda = np.empty(m)
    lmbda[0] = 1 - np.sum(alpha)
    lmbda[1:] = alpha

    # Check if point lies inside the simplex
    if np.all(lmbda >= -1e-12) and np.abs(np.sum(lmbda) - 1) <= 1e-12:
        return x_proj

    # Identify violated constraints (negative barycentric coordinates)
    violated = np.where(lmbda < -1e-12)[0]

    # Recurse only on faces that exclude at least one violated vertex
    min_dist = np.inf
    closest = None
    for i in violated:
        face = np.delete(vertices, i, axis=0)
        candidate = closest_point_on_simplex(face, point)
        dist = np.linalg.norm(candidate - point)
        if dist < min_dist:
            min_dist = dist
            closest = candidate

    return closest


class SimplicialRegion(AbstractRegion):
    """
    Simplicial certification region with direct geometric operations.

    This unified class combines the geometric simplex functionality
    with certification-specific features, eliminating the need for wrapper classes.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        output_dim: int = None,
        nonlin_dependencies: List[Tuple[int, bool]] = None,
        numeric_translator=None,
        depth: int = 0,
    ):
        if numeric_translator is None:
            numeric_translator = NumpyTranslator()
        self.translator = numeric_translator

        # Store geometric parameters and standardize shape
        self.vertices = vertices
        if self.vertices.ndim == 1:
            # Reshape 1D array [v1, v2, ...] to [[v1], [v2], ...]
            self.vertices = self.vertices.reshape(-1, 1)

        self.n_vertices = self.vertices.shape[-2]
        self.dim = self.vertices.shape[-1]

        # Validate simplex - now we can safely access shape[1]
        expected_vertices = self.dim + 1
        if self.n_vertices != expected_vertices:
            raise ValueError(f"Simplex in {self.dim}D space must have {expected_vertices} vertices, got {self.n_vertices}")

        # Compute min_radius for splitting decisions
        if isinstance(self.vertices, np.ndarray):
            max_edge_length, _ = self.get_max_edge_length()
            self.min_radius = 1e-3 * max_edge_length

        # Initialize unified region (this will call _compute_centroid and _compute_volume)
        super().__init__(output_dim, nonlin_dependencies, depth)

    def get_dimension(self) -> int:
        """Get the dimension of the simplicial region."""
        return self.dim

    def _compute_centroid(self) -> np.ndarray:
        """Compute the centroid of the simplex."""
        return self.translator.mean(self.vertices, dim=-2)

    def _compute_volume(self) -> float:
        """Compute the volume of the simplex."""
        if self.vertices.ndim != 2:
            return 0.0

        dim = self.get_dimension()
        if dim == 1:
            # Line segment
            return np.linalg.norm(self.vertices[1] - self.vertices[0])
        else:
            # Higher dimensions: |det(v1-v0, v2-v0, ..., vn-v0)| / n!
            edge_vectors = self.vertices[1:] - self.vertices[0]
            det = np.linalg.det(edge_vectors)
            return abs(det) / math.factorial(dim)

    def get_bounds(self) -> np.ndarray:
        """Get bounding box as (min, max) pairs for each dimension."""
        return np.min(self.vertices, axis=-2), np.max(self.vertices, axis=-2)

    def contains_point(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a point is inside the simplex using barycentric coordinates."""
        point = np.asarray(point)

        # Handle 1D case specially
        if self.get_dimension() == 1:
            x_min = min(self.vertices[:, 0])
            x_max = max(self.vertices[:, 0])
            return x_min - tolerance <= point[0] <= x_max + tolerance

        # For higher dimensions, solve for barycentric coordinates
        A = np.vstack([self.vertices.T, np.ones(self.n_vertices)])
        b = np.append(point, 1.0)

        # Check if matrix is well-conditioned before solving
        cond_num = np.linalg.cond(A)
        if cond_num > 1e12:
            return False

        # Check matrix rank to avoid singular matrices
        rank = np.linalg.matrix_rank(A)
        if rank < A.shape[0]:
            # Use least squares for rank-deficient case
            lambdas, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

            # Check if the solution is reasonable (low residual)
            if len(residuals) > 0 and residuals[0] > tolerance * 100:
                return False
        else:
            # Matrix is full rank, use direct solve
            lambdas = np.linalg.solve(A, b)

        # Point is inside if all barycentric coordinates are non-negative
        return np.all(lambdas >= -tolerance)

    def sample_uniform(self, n_samples: int = 1) -> np.ndarray:
        """Sample points uniformly from the simplex."""
        # Use Dirichlet distribution to sample barycentric coordinates
        alphas = np.ones(self.n_vertices)
        barycentric = np.random.dirichlet(alphas, size=n_samples)

        # Convert to Cartesian coordinates
        samples = barycentric @ self.vertices
        return samples

    def get_max_edge_length(self) -> Tuple[float, Tuple[int, int]]:
        """Get the length of the longest edge in the simplex."""
        max_length = 0
        longest_edge = (0, 1)
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                length = np.linalg.norm(self.vertices[i] - self.vertices[j])
                if length > max_length:
                    max_length = length
                    longest_edge = (i, j)
        return max_length, longest_edge

    def get_halfspace_representation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the halfspace representation of the simplex: Ax <= b.

        For a simplex defined by vertices, this computes the linear inequalities
        that define the region boundaries.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (A, b) where Ax <= b defines the simplex
        """
        if self.dim == 1:
            # For 1D, the simplex is a line segment [x_min, x_max]
            # Constraints: x >= x_min and x <= x_max
            # In standard form: -x <= -x_min and x <= x_max
            x_min = np.min(self.vertices[:, 0])
            x_max = np.max(self.vertices[:, 0])
            A = np.array([[-1.0], [1.0]])
            b = np.array([-x_min, x_max])
            return A, b

        # For higher dimensions, compute the hyperplanes defining the simplex faces
        # Each face is defined by n vertices (where n = dim)
        # The hyperplane equation is: normal · (x - point_on_plane) = 0
        # Rearranged: normal · x <= normal · point_on_plane

        normals = []
        offsets = []

        # Generate all combinations of dim vertices to form faces
        from itertools import combinations

        for face_vertices_indices in combinations(range(self.n_vertices), self.dim):
            face_vertices = self.vertices[list(face_vertices_indices)]

            # Compute normal to the face
            if self.dim == 2:
                # For 2D triangle, each edge defines a constraint
                # Edge vector
                edge = face_vertices[1] - face_vertices[0]
                # Normal vector (rotate 90 degrees)
                normal = np.array([-edge[1], edge[0]])
            else:
                # For higher dimensions, use the null space of the face vertices
                # Center the vertices at the first vertex
                centered_vertices = face_vertices[1:] - face_vertices[0]

                # Check for rank deficiency (collinear/coplanar vertices)
                rank = np.linalg.matrix_rank(centered_vertices)
                # Compute normal using SVD of the centered vertices
                # The normal is in the null space of the face
                U, s, Vt = np.linalg.svd(centered_vertices, full_matrices=True)
                # The normal is the last row of V^T (corresponding to smallest singular value)
                normal = Vt[-1]

            # Normalize the normal vector
            normal_magnitude = np.linalg.norm(normal)
            normal = normal / normal_magnitude

            # Compute offset: normal · point_on_face
            offset = np.dot(normal, face_vertices[0])

            # Determine the correct orientation (inward or outward normal)
            # Check if the centroid of the simplex is on the correct side
            centroid = np.mean(self.vertices, axis=0)
            distance_to_centroid = np.dot(normal, centroid) - offset

            # If the centroid is on the "positive" side, flip the normal
            # We want the inequality normal·x <= offset to be satisfied inside the simplex
            if distance_to_centroid > 1e-12:
                normal = -normal
                offset = -offset

            normals.append(normal)
            offsets.append(offset)

        if not normals:
            # This should not happen for a valid simplex, but provide a meaningful error
            raise ValueError(
                f"Failed to compute halfspace representation for simplex with vertices:\n{self.vertices}\n"
                f"This may indicate degenerate geometry or numerical issues."
            )

        A = np.array(normals)
        b = np.array(offsets)

        return A, b

    def _determine_split_dimension(self, taylor_approximation, dynamics, timeout=False):
        """
        Determine how to split this simplicial region.

        For simplicial regions, we ignore the dimension parameter and instead
        split along the longest edge.

        Args:
            taylor_approximation: Taylor approximation function (not used for simplicial)
            dynamics: True dynamics function (not used for simplicial)
            timeout: Whether this is a timeout case

        Returns:
            The dimension to split on (returns 0 for simplicial regions since
            split method ignores the dimension parameter)
        """
        # For simplicial regions, the split method determines the edge automatically
        # so we just return 0 as a placeholder dimension
        return 0

    def split(self, split_criterion=None, taylor_approximation=None, dynamics=None, timeout=False) -> Tuple["SimplicialRegion", "SimplicialRegion"]:
        """
        Split the simplex by bisecting its longest edge.

        Args:
            split_criterion: For compatibility with base class (ignored for simplicial regions)
            taylor_approximation: Taylor approximation function (not used for simplicial)
            dynamics: True dynamics function (not used for simplicial)
            timeout: Whether this is a timeout case (not used for simplicial)

        Returns:
            Tuple of two new simplicial regions
        """
        max_length, (v1_idx, v2_idx) = self.get_max_edge_length()

        # Compute midpoint of longest edge
        midpoint = (self.vertices[v1_idx] + self.vertices[v2_idx]) / 2

        # Create two new simplices by replacing each vertex with the midpoint
        vertices1 = self.vertices.copy()
        vertices1[v2_idx] = midpoint

        vertices2 = self.vertices.copy()
        vertices2[v1_idx] = midpoint

        # Create new regions
        region1 = SimplicialRegion(
            vertices1,
            output_dim=self.output_dim,
            nonlin_dependencies=self.nonlin_dependencies,
            depth=self.depth + 1,
        )

        region2 = SimplicialRegion(
            vertices2,
            output_dim=self.output_dim,
            nonlin_dependencies=self.nonlin_dependencies,
            depth=self.depth + 1,
        )

        return region1, region2

    def intersects_domain(self, domain) -> bool:
        """Check if this simplicial region intersects with a domain."""
        return domain.intersects_simplex(self.vertices)

    def contained_in_domain(self, domain) -> bool:
        """Check if this simplicial region is completely contained in a domain."""
        return domain.contains_simplex(self.vertices)


class SimplicialMesh(AbstractMesh):
    """
    Delaunay-based mesh using simplicial regions.
    """

    def __init__(self, domain_bounds: List[Tuple[float, float]]):
        """
        Initialize a simplicial mesh over a given domain.

        Args:
            domain_bounds: List of (min, max) bounds for each dimension
        """
        super().__init__(domain_bounds)
        self.delaunay = None  # Store the Delaunay triangulation object
        self.points = None  # Store the points used for triangulation

        # Create initial triangulation of the domain
        self._initialize_mesh()

    def _initialize_mesh(self):
        """Create initial simplicial mesh covering the domain using Delaunay triangulation."""
        # Create a grid of points for all dimensions
        grid_points = []
        n_points_per_dim = 2

        for dim_idx in range(self.dim):
            min_val, max_val = self.domain_bounds[dim_idx]
            points = np.linspace(min_val, max_val, n_points_per_dim)
            grid_points.append(points)

        # Create meshgrid and flatten to get all combinations
        mesh = np.meshgrid(*grid_points, indexing="ij")
        self.points = np.vstack([m.ravel() for m in mesh]).T

        # Handle special case for 1D (Delaunay doesn't work for 1D)
        if self.dim == 1:
            # For 1D, create line segments manually
            self.delaunay = None  # No Delaunay object for 1D
        else:
            # Use Delaunay triangulation for 2D and higher
            self.delaunay = Delaunay(self.points)

    def get_regions(self, output_dim: int, nonlin_dependencies: Optional[List[Tuple[int, bool]]] = None) -> List[SimplicialRegion]:
        """Get all regions in the mesh for a specific output dimension."""
        regions = []

        if self.dim == 1:
            # For 1D, create line segments manually
            for i in range(len(self.points) - 1):
                vertices = np.array([self.points[i], self.points[i + 1]])
                region = SimplicialRegion(
                    vertices,
                    output_dim=output_dim,
                    nonlin_dependencies=nonlin_dependencies,
                )
                regions.append(region)
        else:
            # Create simplices from triangulation
            for simplex_indices in self.delaunay.simplices:
                simplex_vertices = self.points[simplex_indices]
                region = SimplicialRegion(
                    simplex_vertices,
                    output_dim=output_dim,
                    nonlin_dependencies=nonlin_dependencies,
                )
                regions.append(region)

        return regions

    def total_volume(self) -> float:
        """Calculate total volume covered by the mesh."""
        regions = self.get_regions(0)  # Use any output dimension for volume calculation
        return sum(region.volume for region in regions)

    def find_region_containing_point(self, point: np.ndarray) -> Optional[int]:
        """Find which region contains a given point."""
        if self.delaunay is None:
            # For 1D case, search manually
            if self.dim == 1:
                regions = self.get_regions(0)
                for i, region in enumerate(regions):
                    if region.contains_point(point):
                        return i
                return None
            else:
                return None

        # Use Delaunay's find_simplex method
        simplex_index = self.delaunay.find_simplex(point)
        return simplex_index if simplex_index >= 0 else None

    def __len__(self):
        if self.delaunay is None:
            # For 1D case
            return len(self.points) - 1 if len(self.points) > 1 else 0
        return len(self.delaunay.simplices)


class SimplicialRegionGenerator(AbstractRegionGenerator):
    """Generator for simplicial regions using mesh-based approach."""

    def generate_regions(
        self,
        dynamics_model,
        nonlin_dependencies_func: Optional[Callable[[int], Optional[List[Tuple[int, bool]]]]] = None,
    ) -> List[SimplicialRegion]:
        """Generate simplicial regions for verification."""
        mesh = self.create_mesh(dynamics_model)

        all_regions_with_output_dim = []
        for j in range(dynamics_model.output_dim):
            # Get dependencies for this output dimension if function is provided
            nonlin_deps = nonlin_dependencies_func(j) if nonlin_dependencies_func is not None else None

            # Get regions from mesh with dependencies
            regions = mesh.get_regions(j, nonlin_deps)
            all_regions_with_output_dim.extend(regions)

        return all_regions_with_output_dim

    def create_mesh(self, dynamics_model) -> SimplicialMesh:
        """Create a simplicial mesh for the given dynamics model."""
        return SimplicialMesh(dynamics_model.input_domain)
