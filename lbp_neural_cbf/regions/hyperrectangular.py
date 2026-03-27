"""
Hyperrectangular region and mesh implementations.

This module provides concrete implementations for hyperrectangular regions
and grid-based meshes, following the unified region interface.
"""

from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import numpy as np

from .base import AbstractMesh, AbstractRegion, AbstractRegionGenerator


def get_box_corners(box_min: np.ndarray, box_max: np.ndarray) -> List[np.ndarray]:
    """Generate all corners of a hyperrectangular box efficiently."""
    dim = len(box_min)
    corners = []

    # Use binary representation to generate all corners
    for i in range(2**dim):
        corner = box_min.copy()
        for d in range(dim):
            if (i >> d) & 1:
                corner[d] = box_max[d]
        corners.append(corner)

    return corners


class HyperrectangularRegion(AbstractRegion):
    """
    Hyperrectangular certification region with direct geometric operations.

    This unified class combines the geometric hyperrectangle functionality
    with certification-specific features, eliminating the need for wrapper classes.
    """

    def __init__(
        self,
        center: np.ndarray,
        radius: np.ndarray,
        output_dim: int = None,
        nonlin_dependencies: List[Tuple[int, bool]] = None,
        depth: int = 0,
    ):
        # Store geometric parameters
        self.center_point = np.array(center)
        self.radius_vec = np.array(radius)
        self.min_radius = 1e-6 * self.radius_vec

        # Compute vertices for compatibility if needed
        self.vertices = self._compute_vertices()

        # Initialize unified region (this will call _compute_centroid and _compute_volume)
        super().__init__(output_dim, nonlin_dependencies, depth)

    def _compute_vertices(self) -> np.ndarray:
        """Compute all vertices (corners) of the hyperrectangle."""
        n_dim = len(self.center_point)
        n_vertices = 2**n_dim
        vertices = np.zeros((n_vertices, n_dim))

        for i in range(n_vertices):
            for j in range(n_dim):
                # Use bit representation to determine sign
                sign = 1 if (i >> j) & 1 else -1
                vertices[i, j] = self.center_point[j] + sign * self.radius_vec[j]

        return vertices

    def get_dimension(self) -> int:
        """Get the dimension of the hyperrectangular region."""
        return len(self.center_point)

    def _compute_centroid(self) -> np.ndarray:
        """Compute the centroid of the hyperrectangle."""
        return self.center_point.copy()

    def _compute_volume(self) -> float:
        """Compute the volume of the hyperrectangle."""
        return np.prod(2 * self.radius_vec).item()

    def get_bounds(self) -> np.ndarray:
        """Get bounding box as (min, max) pairs for each dimension."""
        return np.column_stack([self.center_point - self.radius_vec, self.center_point + self.radius_vec])

    def contains_point(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a point is inside the hyperrectangle."""
        point = np.asarray(point)
        for i in range(len(self.center_point)):
            if not (self.center_point[i] - self.radius_vec[i] - tolerance <= point[i] <= self.center_point[i] + self.radius_vec[i] + tolerance):
                return False
        return True

    def sample_uniform(self, n_samples: int = 1) -> np.ndarray:
        """Sample points uniformly from the hyperrectangle."""
        samples = []
        for _ in range(n_samples):
            sample = np.random.uniform(self.center_point - self.radius_vec, self.center_point + self.radius_vec)
            samples.append(sample)
        return np.array(samples)

    def compute_volume(self) -> float:
        """Public method for volume computation."""
        return self.volume

    def compute_centroid(self) -> np.ndarray:
        """Public method for centroid computation."""
        return self.centroid

    def split(self, split_criterion=None, taylor_approximation=None, dynamics=None, timeout=False) -> List["HyperrectangularRegion"]:
        """Split the hyperrectangular region into smaller regions."""
        if split_criterion is None:
            # If approximation functions are provided, use them to determine best split dimension
            if taylor_approximation is not None and dynamics is not None:
                split_params = self._determine_split_dimension(taylor_approximation, dynamics, timeout)
                if split_params is None:
                    return []  # No split recommended
                split_dim = split_params["split_dim"]
            else:
                # Default: split along the largest dimension
                split_dim = np.argmax(self.radius_vec)
        elif isinstance(split_criterion, dict):
            # Handle dictionary format from nextsplitdim (for backward compatibility)
            split_dim = split_criterion.get("split_dim", np.argmax(self.radius_vec))
        else:
            # Handle direct integer specification for backward compatibility
            split_dim = split_criterion

        split_radius = self.radius_vec[split_dim] / 2

        # Left hyperrectangle
        left_center = self.center_point.copy()
        left_center[split_dim] -= split_radius
        left_radius = self.radius_vec.copy()
        left_radius[split_dim] = split_radius

        # Right hyperrectangle
        right_center = self.center_point.copy()
        right_center[split_dim] += split_radius
        right_radius = self.radius_vec.copy()
        right_radius[split_dim] = split_radius

        new_regions = []
        for center, radius in [(left_center, left_radius), (right_center, right_radius)]:
            new_region = HyperrectangularRegion(
                center=center,
                radius=radius,
                output_dim=self.output_dim,
                nonlin_dependencies=self.nonlin_dependencies,
                depth=self.depth + 1,
            )
            new_region.min_radius = self.min_radius
            new_regions.append(new_region)

        return new_regions

    def _determine_split_dimension(self, taylor_approximation, dynamics, timeout=False):
        """
        Identify the dimension with the highest approximation error for splitting.

        Returns:
            Dictionary with split parameters, or None if no split is recommended
        """
        if timeout:
            split_dims = list(range(self.get_dimension()))
        else:
            split_dims = self.nonlin_dependencies

        sample, delta = self.center_point, self.radius_vec
        approximation_error = taylor_approximation(sample) - dynamics(sample).flatten()[self.output_dim]

        error_list = np.ones(len(split_dims)) * 10e-9  # Initialize near zero

        if all(delta < self.min_radius):
            return None

        for i in range(len(split_dims)):
            j = split_dims[i]  # Get the current split dimension index

            delta_j = delta[j]
            if delta_j < self.min_radius[j]:
                continue

            left_point = sample.copy()
            left_point[j] -= 0.5 * delta_j

            # Calculate the Taylor approximation at the left point
            left_approximation = taylor_approximation(left_point) - approximation_error
            left_dynamics = dynamics(left_point).flatten()[self.output_dim]
            left_error = abs(left_approximation - left_dynamics)

            right_point = sample.copy()
            right_point[j] += 0.5 * delta_j

            # Calculate the Taylor approximation at the right point
            right_approximation = taylor_approximation(right_point) - approximation_error
            right_dynamics = dynamics(right_point).flatten()[self.output_dim]
            right_error = abs(right_approximation - right_dynamics)

            max_error = np.maximum(left_error, right_error)

            if max_error > 0.0:
                error_list[i] = max_error

        if np.sum(error_list) < 1e-9 and not timeout:
            return self._determine_split_dimension(taylor_approximation, dynamics, timeout=True)

        delta_maxmin_ratio = np.max(delta[split_dims]) / np.min(delta[split_dims])
        if delta_maxmin_ratio.item() > 1e2:
            # Softmax calculation
            probabilities = error_list / np.sum(error_list)
            split_dim_idx = np.random.choice(len(error_list), p=probabilities)
        else:
            split_dim_idx = np.argmax(error_list)

        best_split_dim = split_dims[split_dim_idx]
        return {"split_dim": best_split_dim, "timeout": timeout}

    @property
    def center(self) -> np.ndarray:
        """Compatibility property for existing code."""
        return self.center_point

    @property
    def radius(self) -> np.ndarray:
        """Compatibility property for existing code."""
        return self.radius_vec

    def __repr__(self):
        return f"HyperrectangularRegion(center={self.center_point}, radius={self.radius_vec}, output_dim={self.output_dim})"

    def intersects_domain(self, domain) -> bool:
        """Check if this hyperrectangular region intersects with a domain."""
        return domain.intersects_hyperrect(self.center_point, self.radius_vec)

    def contained_in_domain(self, domain) -> bool:
        """Check if this hyperrectangular region is completely contained in a domain."""
        return domain.contains_hyperrect(self.center_point, self.radius_vec)


class HyperrectangularMesh(AbstractMesh):
    """
    Grid-based mesh using hyperrectangular regions.
    """

    def __init__(self, domain_bounds: List[Tuple[float, float]], delta: float = None):
        """
        Initialize a hyperrectangular mesh over a given domain.

        Args:
            domain_bounds: List of (min, max) bounds for each dimension
            delta: Grid spacing (half-width of each hyperrectangle)
        """
        super().__init__(domain_bounds)
        self.delta = delta
        self.regions = []

        if delta is not None:
            self._initialize_grid_mesh()

    def _initialize_grid_mesh(self):
        """Create initial grid-based mesh covering the domain."""

        X_train, _ = self.generate_grid(delta=self.delta)

        self.regions = []
        for x in X_train:
            region = HyperrectangularRegion(center=x, radius=np.full_like(x, self.delta))
            self.regions.append(region)

    def get_regions(self, output_dim: int, nonlin_dependencies: Optional[List[Tuple[int, bool]]] = None) -> List[HyperrectangularRegion]:
        """Get all regions in the mesh for a specific output dimension."""
        certification_regions = []
        for region in self.regions:
            # Create a copy with the specific output dimension and dependencies
            region_copy = HyperrectangularRegion(
                center=region.center_point,
                radius=region.radius_vec,
                output_dim=output_dim,
                nonlin_dependencies=nonlin_dependencies,
            )
            certification_regions.append(region_copy)
        return certification_regions

    def total_volume(self) -> float:
        """Calculate total volume covered by the mesh."""
        return sum(region.volume for region in self.regions)

    def find_region_containing_point(self, point: np.ndarray) -> Optional[int]:
        """Find which region contains a given point."""
        for i, region in enumerate(self.regions):
            if region.contains_point(point):
                return i
        return None

    def __len__(self):
        return len(self.regions)

    def generate_grid(self, delta=0.01, batch_size=256, dynamics_model=None):
        """
        Generate data points for training or verification.
        Generate a fixed grid of points with spacing at most delta.

        Args:
            delta: Grid spacing (can be scalar or array for each dimension)
            batch_size: Number of samples (unused for grid generation, kept for compatibility)
            dynamics_model: The dynamics model to use for generating outputs

        Returns:
            X_train: Input data with shape [input_dim, batch_size]
            y_train: Output data with shape [input_dim, batch_size] or None if no dynamics_model
        """
        input_size = self.dim
        input_domain = self.domain_bounds

        # Handle scalar delta
        if np.isscalar(delta):
            delta = [delta] * input_size

        # Ensure domain size matches input_size
        assert len(input_domain) == input_size, f"Input domain size {len(input_domain)} must match input size {input_size}"

        # Generate grid points for each dimension based on its domain
        grid_points_per_dim = []
        for i in range(input_size):
            min_val, max_val = input_domain[i]
            # Remove edge of domain, as this is covered by the hypercubes
            min_val = min_val + delta[i]
            max_val = max_val - delta[i]
            num_points = int(np.ceil((max_val - min_val) / (2 * delta[i]))) + 1
            grid_points_per_dim.append(np.linspace(min_val, max_val, num_points))

        # Create meshgrid from the points
        mesh = np.meshgrid(*grid_points_per_dim)
        X_train = np.vstack(list(map(np.ravel, mesh))).T

        if dynamics_model is None:
            y_train = None
        else:
            # Get outputs in [output_dim, batch_size] format
            y_train = dynamics_model(X_train)

        return X_train, y_train


class HyperrectangularRegionGenerator(AbstractRegionGenerator):
    """Generator for hyperrectangular regions using grid-based approach."""

    def generate_regions(
        self,
        dynamics_model,
        nonlin_dependencies_func: Optional[Callable[[int], Optional[List[Tuple[int, bool]]]]] = None,
    ) -> List[HyperrectangularRegion]:
        """Generate hyperrectangular regions for verification."""
        mesh = self.create_mesh(dynamics_model)

        samples = []
        for j in range(dynamics_model.output_dim):
            # Get dependencies for this output dimension if function is provided
            nonlin_deps = nonlin_dependencies_func(j) if nonlin_dependencies_func is not None else None

            # Get regions from mesh with dependencies
            regions = mesh.get_regions(j, nonlin_deps)
            samples.extend(regions)

        return samples

    def create_mesh(self, dynamics_model) -> HyperrectangularMesh:
        """Create a hyperrectangular mesh for the given dynamics model."""
        return HyperrectangularMesh(domain_bounds=dynamics_model.input_domain, delta=dynamics_model.delta)
