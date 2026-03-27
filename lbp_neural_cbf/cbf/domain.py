"""
Domain operations for control barrier function systems.

This module contains utilities for defining, manipulating, and checking
safe/unsafe sets, geometric operations, and domain sampling.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial import Delaunay
from scipy.special import gamma  # Required for n-ball volume
from torch.distributions.uniform import Uniform

from ..regions.hyperrectangular import HyperrectangularRegion, get_box_corners
from ..regions.simplicial import SimplicialRegion, closest_point_on_simplex
from ..translators import NumpyTranslator, TorchTranslator


def _is_torch_tensor(x) -> bool:
    """Check if x is a PyTorch tensor without importing torch."""
    return hasattr(x, "__class__") and "torch" in str(type(x)).lower() and hasattr(x, "dtype")


def _convert_torch_to_numpy(x):
    """Convert PyTorch tensor to numpy array safely."""
    if hasattr(x, "cpu"):
        # Handle both CPU and CUDA tensors
        if hasattr(x, "is_cuda") and x.is_cuda:
            return x.cpu().detach().numpy()
        else:
            return x.detach().numpy()
    return x


def sat_box_simplex_test(box_min: np.ndarray, box_max: np.ndarray, simplex_vertices: np.ndarray) -> bool:
    """Separating Axis Theorem test for box-simplex intersection."""
    # Test box face normals (coordinate axes)
    for dim in range(len(box_min)):
        simplex_proj_min = np.min(simplex_vertices[:, dim])
        simplex_proj_max = np.max(simplex_vertices[:, dim])

        if simplex_proj_max < box_min[dim] or simplex_proj_min > box_max[dim]:
            return False

    # For 2D, test simplex edge normals
    if len(box_min) == 2 and len(simplex_vertices) >= 2:
        box_corners = get_box_corners(box_min, box_max)

        for i in range(len(simplex_vertices)):
            v1 = simplex_vertices[i]
            v2 = simplex_vertices[(i + 1) % len(simplex_vertices)]
            edge = v2 - v1

            if np.linalg.norm(edge) < 1e-10:
                continue

            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)

            simplex_projs = [np.dot(v, normal) for v in simplex_vertices]
            simplex_proj_min = min(simplex_projs)
            simplex_proj_max = max(simplex_projs)

            box_projs = [np.dot(corner, normal) for corner in box_corners]
            box_proj_min = min(box_projs)
            box_proj_max = max(box_projs)

            if simplex_proj_max < box_proj_min or box_proj_max < simplex_proj_min:
                return False

    return True


def _smooth_maximum(constraints: List, translator, beta: float = 10.0):
    """Compute smooth maximum for differentiability using log-sum-exp."""
    if len(constraints) == 1:
        return constraints[0]

    # Require formal operations for translator
    if not hasattr(translator, "exp") or not hasattr(translator, "log"):
        raise ValueError("Translator must support exp() and log() operations for smooth maximum")

    # Use numerically stable log-sum-exp
    result = constraints[0]
    for constraint in constraints[1:]:
        # Numerically stable soft maximum computation
        max_val = translator.max([result, constraint])

        # Mathematically sound soft maximum: log(exp(β(a-max)) + exp(β(b-max))) / β + max
        exp_a = translator.exp(beta * (result - max_val))
        exp_b = translator.exp(beta * (constraint - max_val))
        result = translator.log(exp_a + exp_b) / beta + max_val

    return result


def _smooth_minimum(constraints: List, translator, beta: float = 10.0):
    if len(constraints) == 1:
        return constraints[0]

    # Require formal operations for translator
    if not hasattr(translator, "exp") or not hasattr(translator, "log"):
        raise ValueError("Translator must support exp() and log() operations for smooth minimum")

    # Use numerically stable log-sum-exp for minimum
    result = constraints[0]
    for constraint in constraints[1:]:
        # Numerically stable soft minimum computation
        min_val = translator.min([result, constraint])

        # Mathematically sound soft minimum: -log(exp(-β(a-min)) + exp(-β(b-min))) / β + min
        exp_a = translator.exp(-beta * (result - min_val))
        exp_b = translator.exp(-beta * (constraint - min_val))
        result = -translator.log(exp_a + exp_b) / beta + min_val

    return result


class Domain(ABC):
    """Abstract base class for domain definitions."""

    def __init__(self, dim: int):
        """Initialize domain with dimension validation."""
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        """Return the dimension of this domain."""
        return self._dim

    def __repr__(self) -> str:
        """Return string representation of the domain."""
        return f"{self.__class__.__name__}(dim={self.dim})"

    @abstractmethod
    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) x are in this domain."""
        pass

    @abstractmethod
    def sample_points(self, num_points: int, **kwargs) -> np.ndarray:
        """Sample random points from this domain."""
        pass

    @abstractmethod
    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Compute constraint value (positive inside domain)."""
        pass

    def intersects_region(self, region) -> bool:
        """Check if this domain intersects with a region (unified interface)."""
        return region.intersects_domain(self)

    def contains_region(self, region) -> bool:
        """Check if this domain completely contains a region (unified interface)."""
        return region.contained_in_domain(self)

    # Abstract methods for domain-specific geometric operations
    @abstractmethod
    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Check intersection with hyperrectangular region."""
        pass

    @abstractmethod
    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Check if domain contains hyperrectangular region."""
        pass

    @abstractmethod
    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Check intersection with simplicial region."""
        pass

    @abstractmethod
    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """Check if domain contains simplicial region."""
        pass


class CircleDomain(Domain):
    """Circular/spherical domain in arbitrary dimensions."""

    def __init__(self, center: List[float], radius: float):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        super().__init__(len(self.center))

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are inside circle/sphere."""
        # For contains, we typically don't need translator operations
        # but we should handle both tensor types
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)
        center = translator.to_format(self.center)

        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected points with dimensions {self.dim}, got {x.shape[-1]}")

        # Standard convention: (batch_size, dim)
        diff = x - center
        dist_sq = translator.sum(diff * diff, dim=-1)

        return dist_sq <= self.radius**2

    def volume(self) -> float:
        """Calculate the volume of the n-ball."""
        n = self.dim
        return (np.pi ** (n / 2) / gamma(n / 2 + 1)) * self.radius**n

    def sample_points(self, num_points: int, device=None, use_torch=False, **kwargs):
        """
        Sample points uniformly from the n-ball (circle/sphere) using a unified method.
        """
        if not isinstance(num_points, (int, np.integer)) or num_points < 0:
            raise TypeError("num_points must be a non-negative integer")
        if self.radius <= 0:
            raise ValueError("Cannot sample from domain with non-positive radius")

        if num_points == 0:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        if use_torch:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Generate points from a standard Gaussian distribution
            gaussian_samples = torch.randn(num_points, self.dim, device=device)

            # Normalize to get uniformly distributed points on the sphere surface
            norms = torch.norm(gaussian_samples, p=2, dim=1, keepdim=True)
            unit_vectors = gaussian_samples / torch.clamp(norms, min=1e-12)

            # Sample radii to ensure uniform distribution within the n-ball volume
            u = torch.rand(num_points, 1, device=device)
            radii = self.radius * torch.pow(u, 1.0 / self.dim)

            # Scale unit vectors by radii and offset by the center
            center_tensor = torch.tensor(self.center, dtype=torch.float32, device=device)
            points = center_tensor + radii * unit_vectors
            return points

        else:  # NumPy implementation
            # Generate points from a standard Gaussian distribution
            gaussian_samples = np.random.normal(size=(num_points, self.dim))

            # Normalize to get uniformly distributed points on the sphere surface
            norms = np.linalg.norm(gaussian_samples, axis=1, keepdims=True)
            unit_vectors = gaussian_samples / np.where(norms < 1e-12, 1e-12, norms)

            # Sample radii to ensure uniform distribution within the n-ball volume
            u = np.random.rand(num_points, 1)
            radii = self.radius * np.power(u, 1.0 / self.dim)

            # Scale unit vectors by radii and offset by the center
            points = self.center + radii * unit_vectors
            return points

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Constraint: r² - ||x - c||² (positive inside circle)."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)
        center = translator.to_format(self.center)

        # Handle both shape conventions: (batch, dim) and (dim,)
        if x.ndim == 1:
            if len(x) != self.dim:
                raise ValueError(f"Expected {self.dim}D point, got {len(x)}D")
            diff = x - center
        elif x.ndim == 2:
            # Standard convention: (batch_size, dim)
            diff = x - center
        else:
            raise ValueError(f"Input must be 1D or 2D array, got {x.ndim}D")

        # Compute distance squared using appropriate operations
        if x.ndim == 1:
            dist_sq = translator.sum(diff * diff)
        else:
            dist_sq = translator.sum(diff * diff, dim=1)
        return self.radius**2 - dist_sq

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Check intersection with hyperrectangular region."""
        region_lower = center - radius
        region_upper = center + radius

        # Find closest point on rectangle to circle center
        closest_point = np.clip(self.center, region_lower, region_upper)
        distance_sq = np.sum((self.center - closest_point) ** 2)

        return distance_sq <= self.radius**2

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Check if circle contains entire hyperrectangle."""
        # Find the corner farthest from circle center
        corner_offsets = np.where(center >= self.center, radius, -radius)
        farthest_corner = center + corner_offsets
        distance_sq = np.sum((farthest_corner - self.center) ** 2)
        return distance_sq <= self.radius**2

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Check intersection with simplicial region."""
        # Check if any vertex is inside
        vertex_diffs = vertices - self.center[np.newaxis, :]
        vertex_dist_sq = np.sum(vertex_diffs * vertex_diffs, axis=1)
        if np.any(vertex_dist_sq <= self.radius**2):
            return True

        # Check if center is inside simplex
        temp_simplex = SimplicialRegion(vertices)
        if temp_simplex.contains_point(self.center):
            return True

        n_vertices = len(vertices)

        if self.dim == 2 and n_vertices >= 2:
            # 2D case: check edges
            for i in range(n_vertices):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % n_vertices]
                if self._circle_intersects_line_segment(v1, v2):
                    return True
        else:
            # Higher dimensions: use closest point method
            closest_point = closest_point_on_simplex(vertices, self.center)
            distance_sq = np.sum((closest_point - self.center) ** 2)
            return distance_sq <= self.radius**2

        return False

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """Check if circle contains entire simplex."""
        vertex_diffs = vertices - self.center[np.newaxis, :]
        vertex_dist_sq = np.sum(vertex_diffs * vertex_diffs, axis=1)
        return np.all(vertex_dist_sq <= self.radius**2)

    def _circle_intersects_line_segment(self, v1: np.ndarray, v2: np.ndarray) -> bool:
        """Check if circle intersects line segment."""
        edge_vec = v2 - v1
        edge_len_sq = np.dot(edge_vec, edge_vec)

        if edge_len_sq < 1e-12:  # Degenerate edge
            return np.sum((v1 - self.center) ** 2) <= self.radius**2

        t = np.clip(np.dot(self.center - v1, edge_vec) / edge_len_sq, 0, 1)
        closest_on_edge = v1 + t * edge_vec

        return np.sum((closest_on_edge - self.center) ** 2) <= self.radius**2


class BoxDomain(Domain):
    """Rectangular/box domain."""

    def __init__(self, bounds: List[List[float]]):
        if not bounds:
            raise ValueError("Bounds cannot be empty")

        for i, (low, high) in enumerate(bounds):
            if low >= high:
                raise ValueError(f"Invalid bounds at dimension {i}: {low} >= {high}")

        self.bounds = bounds
        self.low_bounds = np.array([bound[0] for bound in bounds], dtype=np.float64)
        self.high_bounds = np.array([bound[1] for bound in bounds], dtype=np.float64)
        super().__init__(len(bounds))

    def corners(self):
        # Generate all corners of the box
        dim = self.dim
        for i in range(2**dim):
            corner = np.zeros(dim, dtype=np.float64)
            for d in range(dim):
                if (i >> d) & 1:
                    corner[d] = self.high_bounds[d]
                else:
                    corner[d] = self.low_bounds[d]

            yield corner

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are inside box."""
        # For contains, we typically don't need translator operations
        # but we should handle both tensor types
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)
        low_bounds = translator.to_format(self.low_bounds)
        high_bounds = translator.to_format(self.high_bounds)

        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected points with dimensions {self.dim}, got {x.shape[-1]}")

        return translator.all((x >= low_bounds) & (x <= high_bounds), dim=-1)

    def volume(self) -> float:
        """Calculate the volume of the hyperrectangle."""
        return np.prod(self.high_bounds - self.low_bounds)

    def sample_points(self, num_points: int, device=None, use_torch=False, **kwargs):
        """Sample points uniformly from the box."""
        if num_points <= 0:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        if use_torch:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            low_tensor = torch.tensor(self.low_bounds, dtype=torch.float32, device=device)
            high_tensor = torch.tensor(self.high_bounds, dtype=torch.float32, device=device)

            sampler = Uniform(low_tensor, high_tensor)
            return sampler.sample(sample_shape=torch.Size([num_points]))
        else:
            return np.random.uniform(low=self.low_bounds, high=self.high_bounds, size=(num_points, self.dim))

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Constraint computation: minimum distance to all boundaries."""
        # Convert input to appropriate type based on translator

        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)
        low_bounds = translator.to_format(self.low_bounds)
        high_bounds = translator.to_format(self.high_bounds)

        # Handle both shape conventions: (batch, dim) and (dim,)
        if x.ndim == 2:
            # Standard convention: (batch_size, dim)
            pass  # Already correct
        elif x.ndim == 1:
            if len(x) != self.dim:
                raise ValueError(f"Expected {self.dim}D point, got {len(x)}D")
        else:
            raise ValueError(f"Input must be 1D or 2D array, got {x.ndim}D")

        # Translator case: handle both single point and batch
        if x.ndim == 1:
            distances = translator.cat([x - low_bounds, high_bounds - x])
            return translator.min(distances)
        else:
            dist_to_low = x - low_bounds
            dist_to_high = high_bounds - x
            distances = translator.cat([dist_to_low, dist_to_high], dim=1)
            return translator.min(distances, dim=1)

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Check intersection with hyperrectangular region."""
        region_lower = center - radius
        region_upper = center + radius

        # Two boxes intersect if they overlap in all dimensions
        return np.all(np.maximum(region_lower, self.low_bounds) <= np.minimum(region_upper, self.high_bounds))

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Check if box contains entire hyperrectangle."""
        region_lower = center - radius
        region_upper = center + radius

        return np.all(region_lower >= self.low_bounds) and np.all(region_upper <= self.high_bounds)

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Check intersection with simplicial region using SAT."""
        # Quick checks first
        # Check if any vertex is inside box
        vertex_in_box = np.all((vertices >= self.low_bounds[np.newaxis, :]) & (vertices <= self.high_bounds[np.newaxis, :]), axis=1)
        if np.any(vertex_in_box):
            return True

        # Check if any box corner is inside simplex
        box_corners = get_box_corners(self.low_bounds, self.high_bounds)
        temp_simplex = SimplicialRegion(vertices)
        for corner in box_corners:
            if temp_simplex.contains_point(corner):
                return True

        # SAT test
        return sat_box_simplex_test(self.low_bounds, self.high_bounds, vertices)

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """Check if box contains entire simplex."""
        return np.all((vertices >= self.low_bounds[np.newaxis, :]) & (vertices <= self.high_bounds[np.newaxis, :]))


class CompositeDomain(Domain):
    """Base class for composite domains with shared functionality."""

    def __init__(self, domains: List[Domain]):
        if not domains:
            raise ValueError("Composite domain requires at least one sub-domain")

        # Validate dimension consistency
        dim = domains[0].dim
        for i, domain in enumerate(domains[1:], 1):
            if domain.dim != dim:
                raise ValueError(f"All domains must have the same dimension. " f"Domain 0 has dim {dim}, but domain {i} has dim {domain.dim}")

        self.domains = domains
        super().__init__(dim)


class UnionDomain(CompositeDomain):
    """Union of multiple domains."""

    def __init__(self, domains, known_separated=False):
        self.known_separated = known_separated
        super().__init__(domains)

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are in any of the domains."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)

        contains = [domain.contains(x, translator) for domain in self.domains]
        contains = translator.stack(contains, dim=-1)

        return translator.any(contains, dim=-1)

    def sample_points(self, num_points: int, device=None, use_torch=False, **kwargs):
        """Sample points from the union, distributed proportionally to sub-domain volumes."""
        if num_points <= 0:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        # Calculate volumes and proportions
        volumes = np.array([d.volume() for d in self.domains])
        total_volume = np.sum(volumes)

        if total_volume > 1e-9:  # Use proportional sampling
            proportions = volumes / total_volume
            points_per_domain = (proportions * num_points).astype(int)
            remainder = num_points - np.sum(points_per_domain)
            # Distribute remainder to largest domains
            for i in np.argsort(-volumes)[:remainder]:
                points_per_domain[i] += 1
        else:  # Fallback to equal distribution if volumes are zero or invalid
            base_points = num_points // len(self.domains)
            remainder = num_points % len(self.domains)
            points_per_domain = [base_points + (1 if i < remainder else 0) for i in range(len(self.domains))]

        # Sample from each domain and collect results
        all_points = []
        for i, domain in enumerate(self.domains):
            n_points = points_per_domain[i]
            if n_points > 0:
                domain_points = domain.sample_points(n_points, device=device, use_torch=use_torch, **kwargs)
                if domain_points.shape[0] > 0:
                    all_points.append(domain_points)

        if not all_points:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        # Combine and return
        if use_torch:
            return torch.cat(all_points, dim=0)
        else:
            return np.vstack(all_points)

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Maximum constraint for union."""
        if translator is None:
            translator = NumpyTranslator()

        constraints = [domain.constraint(x, translator) for domain in self.domains]
        constraints = translator.stack(constraints, dim=-1)

        return translator.max(constraints, dim=-1)

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Union intersects if any subdomain intersects."""
        return any(domain.intersects_hyperrect(center, radius) for domain in self.domains)

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """
        Union contains if any subdomain completely contains the hyperrect.

        Raises NotImplementedError if the union domains intersect/touch and the hyperrect
        is not fully contained in a single domain (ambiguous case).
        """
        # Check which domains contain or intersect the hyperrect
        containing_domains = [i for i, domain in enumerate(self.domains) if domain.contains_hyperrect(center, radius)]

        # If any domain fully contains it, we can return True safely
        if containing_domains:
            return True

        # Check which domains intersect the hyperrect
        intersecting_domains = [i for i, domain in enumerate(self.domains) if domain.intersects_hyperrect(center, radius)]

        # If no domain intersects, clearly False
        if not intersecting_domains:
            return False

        # If multiple domains intersect but none contains it completely,
        # we need to check if the union domains themselves overlap
        if not self.known_separated and len(intersecting_domains) > 1:
            # Check if any pair of intersecting domains overlap with each other
            for i in range(len(intersecting_domains)):
                for j in range(i + 1, len(intersecting_domains)):
                    domain_i = self.domains[intersecting_domains[i]]
                    domain_j = self.domains[intersecting_domains[j]]

                    if isinstance(domain_i, CircleDomain) and isinstance(domain_j, CircleDomain):
                        center_dist_sq = np.sum((domain_i.center - domain_j.center) ** 2)
                        radius_sum = domain_i.radius + domain_j.radius
                        if center_dist_sq < radius_sum**2:
                            raise NotImplementedError(
                                f"UnionDomain.contains_simplex: Cannot determine containment when "
                                f"multiple union domains intersect the simplex but none fully contains it. "
                                f"This case is ambiguous when union domains may overlap."
                            )
                    else:
                        # Try to detect if domains overlap (this is conservative)
                        # For now, raise error when multiple domains intersect the region
                        raise NotImplementedError(
                            f"UnionDomain.contains_simplex: Cannot determine containment when "
                            f"multiple union domains intersect the simplex but none fully contains it. "
                            f"This case is ambiguous when union domains may overlap."
                        )

        # Single domain intersects but doesn't contain - clearly False
        return False

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Union intersects if any subdomain intersects."""
        return any(domain.intersects_simplex(vertices) for domain in self.domains)

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """
        Union contains if any subdomain completely contains the simplex.

        Raises NotImplementedError if the union domains intersect/touch and the simplex
        is not fully contained in a single domain (ambiguous case).
        """
        # Check which domains contain or intersect the simplex
        containing_domains = [i for i, domain in enumerate(self.domains) if domain.contains_simplex(vertices)]

        # If any domain fully contains it, we can return True safely
        if containing_domains:
            return True

        # Check which domains intersect the simplex
        intersecting_domains = [i for i, domain in enumerate(self.domains) if domain.intersects_simplex(vertices)]

        # If no domain intersects, clearly False
        if not intersecting_domains:
            return False

        # If multiple domains intersect but none contains it completely,
        # we need to check if the union domains themselves overlap
        if not self.known_separated and len(intersecting_domains) > 1:
            # Check if any pair of intersecting domains overlap with each other
            for i in range(len(intersecting_domains)):
                for j in range(i + 1, len(intersecting_domains)):
                    domain_i = self.domains[intersecting_domains[i]]
                    domain_j = self.domains[intersecting_domains[j]]

                    if isinstance(domain_i, CircleDomain) and isinstance(domain_j, CircleDomain):
                        center_dist_sq = np.sum((domain_i.center - domain_j.center) ** 2)
                        radius_sum = domain_i.radius + domain_j.radius
                        if center_dist_sq < radius_sum**2:
                            raise NotImplementedError(
                                f"UnionDomain.contains_simplex: Cannot determine containment when "
                                f"multiple union domains intersect the simplex but none fully contains it. "
                                f"This case is ambiguous when union domains may overlap."
                            )
                    else:
                        # Try to detect if domains overlap (this is conservative)
                        # For now, raise error when multiple domains intersect the region
                        raise NotImplementedError(
                            f"UnionDomain.contains_simplex: Cannot determine containment when "
                            f"multiple union domains intersect the simplex but none fully contains it. "
                            f"This case is ambiguous when union domains may overlap."
                        )

        # Single domain intersects but doesn't contain - clearly False
        return False


class IntersectionDomain(CompositeDomain):
    """Intersection of multiple domains."""

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are in all domains."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)
        contains = [domain.contains(x, translator) for domain in self.domains]
        contains = translator.stack(contains, dim=-1)

        return translator.all(contains, dim=-1)

    def sample_points(self, num_points: int, max_attempts_multiplier: int = 20, device=None, use_torch=False, **kwargs):
        """Sample points from the intersection using efficient rejection sampling."""
        if num_points <= 0:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        # Smart Base Sampler: Pick the domain with the smallest volume to maximize acceptance rate.
        try:
            base_domain = min(self.domains, key=lambda d: d.volume())
        except (ValueError, TypeError):  # Fallback if volume is not defined
            base_domain = self.domains[0]

        collected_points = []
        max_attempts = num_points * max_attempts_multiplier
        attempts = 0
        # Use a dynamic batch size for efficiency
        sample_batch_size = min(2048, max(512, num_points * 5))

        if use_torch:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            translator = TorchTranslator(device=device)

            while len(collected_points) < num_points and attempts < max_attempts:
                candidates = base_domain.sample_points(sample_batch_size, device=device, use_torch=True, **kwargs)
                attempts += candidates.shape[0]
                if candidates.shape[0] == 0:
                    continue

                # GPU-native check to avoid .cpu().numpy() transfers in the loop
                valid_mask = self.constraint(candidates.T, translator=translator) >= 0
                valid_points = candidates[valid_mask]

                if valid_points.shape[0] > 0:
                    needed = num_points - sum(p.shape[0] for p in collected_points)
                    collected_points.append(valid_points[:needed])

            if not collected_points:
                print(f"Warning: Could not sample any points for intersection after {attempts} attempts.")
                return torch.empty((0, self.dim), device=device)

            result = torch.cat(collected_points, dim=0)
        else:  # NumPy implementation
            while len(collected_points) < num_points and attempts < max_attempts:
                candidates = base_domain.sample_points(sample_batch_size, use_torch=False, **kwargs)
                attempts += candidates.shape[0]
                if candidates.shape[0] == 0:
                    continue

                valid_mask = self.contains(candidates)
                valid_points = candidates[valid_mask]

                if valid_points.shape[0] > 0:
                    needed = num_points - sum(p.shape[0] for p in collected_points)
                    collected_points.append(valid_points[:needed])

            if not collected_points:
                print(f"Warning: Could not sample any points for intersection after {attempts} attempts.")
                return np.empty((0, self.dim))

            result = np.vstack(collected_points)

        if result.shape[0] < num_points:
            print(f"Warning: Only sampled {result.shape[0]}/{num_points} points for intersection.")

        return result

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Smooth minimum constraint for intersection."""
        if translator is None:
            translator = NumpyTranslator()

        constraints = [domain.constraint(x, translator) for domain in self.domains]
        constraints = translator.stack(constraints, dim=-1)

        return translator.min(constraints, dim=-1)

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Intersection intersects if all subdomains intersect."""
        raise NotImplementedError("IntersectionDomain.intersects_hyperrect is not implemented.")

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Intersection contains if all subdomains contain."""
        return all(domain.contains_hyperrect(center, radius) for domain in self.domains)

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Intersection intersects if all subdomains intersect."""
        raise NotImplementedError("IntersectionDomain.intersects_simplex is not implemented.")

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """Intersection contains if all subdomains contain."""
        return all(domain.contains_simplex(vertices) for domain in self.domains)


class ProductDomain(Domain):
    """Cartesian product of multiple domains."""

    def __init__(self, domains: List[Domain]):
        if not domains:
            raise ValueError("Product domain requires at least one sub-domain")

        total_dim = sum(domain.dim for domain in domains)
        self.domains = domains
        super().__init__(total_dim)

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are in the product domain."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)

        contains = []
        offset = 0
        for domain in self.domains:
            dim = domain.dim
            contains.append(domain.contains(x[..., offset : offset + dim], translator))
            offset += dim

        contains = translator.stack(contains, dim=-1)
        return translator.all(contains, dim=-1)

    def sample_points(self, num_points: int, device=None, use_torch=False, **kwargs):
        """Sample points from the product domain."""
        if num_points <= 0:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        samples_list = []
        for domain in self.domains:
            domain_samples = domain.sample_points(num_points, device=device, use_torch=use_torch, **kwargs)
            samples_list.append(domain_samples)

        if use_torch:
            return torch.cat(samples_list, dim=1)
        else:
            return np.hstack(samples_list)

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Minimum constraint across all subdomains."""
        if translator is None:
            translator = NumpyTranslator()

        constraints = []
        offset = 0
        for domain in self.domains:
            dim = domain.dim
            domain_constraint = domain.constraint(x[:, offset : offset + dim], translator)
            constraints.append(domain_constraint)
            offset += dim

        constraints = translator.stack(constraints, dim=-1)
        return translator.min(constraints, dim=-1)

    def volume(self) -> float:
        """Calculate the volume of the product domain."""
        vol = 1.0
        for domain in self.domains:
            vol *= domain.volume()
        return vol

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        offset = 0
        for domain in self.domains:
            dim = domain.dim
            if not domain.intersects_hyperrect(center[offset : offset + dim], radius[offset : offset + dim]):
                return False
            offset += dim
        return True

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        offset = 0
        for domain in self.domains:
            dim = domain.dim
            if not domain.contains_hyperrect(center[offset : offset + dim], radius[offset : offset + dim]):
                return False
            offset += dim
        return True

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        offset = 0
        for domain in self.domains:
            dim = domain.dim
            if dim == 1:
                # 1D case: quick check using min/max
                v_min = np.min(vertices[:, offset : offset + dim])
                v_max = np.max(vertices[:, offset : offset + dim])
                if not domain.intersects_hyperrect(center=np.array([(v_min + v_max) / 2]), radius=np.array([(v_max - v_min) / 2])):
                    return False
            else:
                # Project vertices to the subdomain's dimensions
                projected_vertices = vertices[:, offset : offset + dim]

                # Triangulation to convert the projected polygon into simplices
                triangulation = Delaunay(projected_vertices)

                # Check intersection
                intersects = [domain.intersects_simplex(projected_vertices[triangle]) for triangle in triangulation.simplices]
                if not any(intersects):
                    return False
            offset += dim
        return True

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        offset = 0
        for domain in self.domains:
            dim = domain.dim
            if dim == 1:
                # 1D case: quick check using min/max
                v_min = np.min(vertices[:, offset : offset + dim])
                v_max = np.max(vertices[:, offset : offset + dim])
                if not domain.contains_hyperrect(center=np.array([(v_min + v_max) / 2]), radius=np.array([(v_max - v_min) / 2])):
                    return False
            else:
                # Project vertices to the subdomain's dimensions
                projected_vertices = vertices[:, offset : offset + dim]

                # Triangulation to convert the projected polygon into simplices
                triangulation = Delaunay(projected_vertices)

                # Check containment
                contains = [domain.contains_simplex(projected_vertices[triangle]) for triangle in triangulation.simplices]
                if not all(contains):
                    return False
            offset += dim
        return True


class ComplementDomain(Domain):
    """Complement of a domain (within specified bounds)."""

    def __init__(self, domain: Domain, bounds: List[List[float]]):
        if not bounds:
            raise ValueError("Bounds cannot be empty")

        for i, (low, high) in enumerate(bounds):
            if low >= high:
                raise ValueError(f"Invalid bounds at dimension {i}: {low} >= {high}")

        if domain.dim != len(bounds):
            raise ValueError(f"Domain dimension ({domain.dim}) must match bounds dimension ({len(bounds)})")

        self.domain = domain
        self.bounds = bounds
        self.bounding_box = BoxDomain(bounds)
        super().__init__(len(bounds))

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are in complement (inside bounds but outside domain)."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)

        # Ensure consistent array handling
        in_bounds = self.bounding_box.contains(x, translator)
        in_domain = self.domain.contains(x, translator)

        return in_bounds & ~in_domain

    def volume(self) -> float:
        """
        Calculate the volume of the complement domain.

        This is an approximation: volume(bounds) - volume(domain).
        Note: This may not be exact for complex domains where the excluded
        domain extends outside the bounds, but provides a reasonable estimate
        for sampling purposes.

        Returns:
            Approximate volume of the complement domain
        """
        bounding_box_volume = self.bounding_box.volume()

        # Try to get domain volume if available
        try:
            domain_volume = self.domain.volume()
            # The complement volume is at most the bounding box volume
            # (in case domain extends beyond bounds)
            return max(0.0, bounding_box_volume - domain_volume)
        except (AttributeError, NotImplementedError):
            # If domain doesn't have volume(), use bounding box volume as estimate
            # This is conservative and ensures sampling still works
            return bounding_box_volume

    def sample_points(self, num_points: int, max_attempts_multiplier: int = 15, device=None, use_torch=False, **kwargs):
        """Sample points from the complement using efficient rejection sampling."""
        if num_points <= 0:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        collected_points = []
        max_attempts = num_points * max_attempts_multiplier
        attempts = 0
        sample_batch_size = min(2048, max(512, num_points * 3))

        if use_torch:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            translator = TorchTranslator(device=device)

            while len(collected_points) < num_points and attempts < max_attempts:
                # Base sampler is always the bounding box
                candidates = self.bounding_box.sample_points(sample_batch_size, device=device, use_torch=True, **kwargs)
                attempts += candidates.shape[0]
                if candidates.shape[0] == 0:
                    continue

                # GPU-native check to avoid .cpu().numpy() transfers in the loop
                valid_mask = self.constraint(candidates, translator=translator) >= 0
                valid_points = candidates[valid_mask]

                if valid_points.shape[0] > 0:
                    needed = num_points - sum(p.shape[0] for p in collected_points)
                    collected_points.append(valid_points[:needed])

            if not collected_points:
                print(f"Warning: Could not sample any points for complement after {attempts} attempts.")
                return torch.empty((0, self.dim), device=device)

            result = torch.cat(collected_points, dim=0)
        else:  # NumPy implementation
            while len(collected_points) < num_points and attempts < max_attempts:
                candidates = self.bounding_box.sample_points(sample_batch_size, use_torch=False, **kwargs)
                attempts += candidates.shape[0]
                if candidates.shape[0] == 0:
                    continue

                valid_mask = self.contains(candidates)
                valid_points = candidates[valid_mask]

                if valid_points.shape[0] > 0:
                    needed = num_points - sum(p.shape[0] for p in collected_points)
                    collected_points.append(valid_points[:needed])

            if not collected_points:
                print(f"Warning: Could not sample any points for complement after {attempts} attempts.")
                return np.empty((0, self.dim))

            result = np.vstack(collected_points)

        if result.shape[0] < num_points:
            print(f"Warning: Only sampled {result.shape[0]}/{num_points} points for complement.")

        return result

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Vectorized constraint computation for complement."""
        # Complement constraint: positive when outside domain AND inside bounds
        domain_constraint = -self.domain.constraint(x, translator)  # Negate to flip
        bounds_constraint = self.bounding_box.constraint(x, translator)

        # Must satisfy both: outside domain AND inside bounds (intersection)
        if translator is None:
            if isinstance(domain_constraint, np.ndarray) and domain_constraint.ndim > 0:
                return np.minimum(domain_constraint, bounds_constraint)
            else:
                return min(domain_constraint, bounds_constraint)
        else:
            # Use smooth minimum for differentiability
            if not hasattr(translator, "sqrt"):
                raise ValueError("Translator must support sqrt() operation for smooth minimum")

            diff = domain_constraint - bounds_constraint
            epsilon = 0.1
            return (domain_constraint + bounds_constraint - translator.sqrt(diff * diff + epsilon**2)) / 2

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Complement intersects if region intersects bounds but not completely contained in domain."""
        if not self.bounding_box.intersects_hyperrect(center, radius):
            return False
        return not self.domain.contains_hyperrect(center, radius)

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Complement contains if region is inside bounds and doesn't intersect domain."""
        return (
            self.bounding_box.contains_hyperrect(center, radius)
            and not self.domain.intersects_hyperrect(center, radius)
            and not self.domain.contains_hyperrect(center, radius)
        )

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Complement intersects if simplex intersects bounds but not completely contained in domain."""
        if not self.bounding_box.intersects_simplex(vertices):
            return False
        return not self.domain.contains_simplex(vertices)

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """Complement contains if simplex is inside bounds and doesn't intersect domain."""
        return self.bounding_box.contains_simplex(vertices) and not self.domain.intersects_simplex(vertices) and not self.domain.contains_simplex(vertices)


class BoxExteriorDomain(Domain):
    def __init__(self, bounds: List[List[float]]):
        if not bounds:
            raise ValueError("Bounds cannot be empty")

        for i, (low, high) in enumerate(bounds):
            if low >= high:
                raise ValueError(f"Invalid bounds at dimension {i}: {low} >= {high}")

        self.bounds = bounds
        self.bounding_box = BoxDomain(bounds)
        super().__init__(len(bounds))

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are outside the box."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)
        return ~self.bounding_box.contains(x, translator)

    def volume(self) -> float:
        """Volume is infinite for box exterior."""
        return float("inf")

    def sample_points(self, num_points: int, device=None, use_torch=False, margin: float = 1.0, **kwargs):
        raise NotImplementedError("Sampling from BoxExteriorDomain is not implemented due to infinite volume.")

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Constraint computation: negative distance to box boundaries."""
        if translator is None:
            translator = NumpyTranslator()

        box_constraint = self.bounding_box.constraint(x, translator)
        return -box_constraint  # Negate to represent outside distance

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Box exterior intersects if hyperrect extends beyond box bounds."""
        region_lower = center - radius
        region_upper = center + radius

        for dim in range(self.dim):
            box_low, box_high = self.bounds[dim]
            if region_upper[dim] < box_low or region_lower[dim] > box_high:
                return True  # Region is completely outside box in this dimension
        return False  # Region is fully inside box in all dimensions

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Box exterior contains hyperrect if it is completely outside box bounds."""
        region_lower = center - radius
        region_upper = center + radius

        for dim in range(self.dim):
            box_low, box_high = self.bounds[dim]
            if not (region_upper[dim] < box_low or region_lower[dim] > box_high):
                return False  # Region overlaps or is inside box in this dimension
        return True  # Region is completely outside box in all dimensions

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Box exterior intersects if any vertex is outside box bounds."""
        for vertex in vertices:
            if self.contains(vertex):
                return True
        return False

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """Box exterior contains simplex if all vertices are outside box bounds."""
        for vertex in vertices:
            if not self.contains(vertex):
                return False
        return True


class SetMinusDomain(CompositeDomain):
    """
    Set minus domains.

    Unchecked assumption: minus_domain is fully contained within plus_domain.
    """

    def __init__(self, plus_domain: Domain, minus_domain: Domain):
        self.plus_domain = plus_domain
        self.minus_domain = minus_domain
        super().__init__([plus_domain, minus_domain])

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are in complement (inside bounds but outside domain)."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)

        in_plus_domain = self.plus_domain.contains(x, translator)
        in_minus_domain = self.minus_domain.contains(x, translator)

        return in_plus_domain & ~in_minus_domain

    def volume(self) -> float:
        """
        Calculate the volume of the complement domain.

        This is an approximation: volume(bounds) - volume(domain).
        Note: This may not be exact for complex domains where the excluded
        domain extends outside the bounds, but provides a reasonable estimate
        for sampling purposes.

        Returns:
            Approximate volume of the complement domain
        """
        return self.plus_domain.volume() - self.minus_domain.volume()

    def sample_points(self, num_points: int, device=None, use_torch=False, **kwargs):
        points = self.plus_domain.sample_points(2 * num_points, device=device, use_torch=use_torch, **kwargs)

        translator = TorchTranslator(device=device) if use_torch else NumpyTranslator()
        mask = self.contains(points, translator=translator)
        valid_points = points[mask]
        return valid_points[:num_points]

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """Vectorized constraint computation for complement."""
        # Complement constraint: positive when outside domain AND inside bounds
        plus_domain_constraint = self.plus_domain.constraint(x, translator)  # Negate to flip
        minus_domain_constraint = -self.minus_domain.constraint(x, translator)

        # Must satisfy both: inside plus_domain AND outside minus_domain (intersection)
        if translator is None:
            if isinstance(plus_domain_constraint, np.ndarray) and plus_domain_constraint.ndim > 0:
                return np.minimum(plus_domain_constraint, minus_domain_constraint)
            else:
                return min(plus_domain_constraint, minus_domain_constraint)
        else:
            # Use smooth minimum for differentiability
            if not hasattr(translator, "sqrt"):
                raise ValueError("Translator must support sqrt() operation for smooth minimum")

            diff = plus_domain_constraint - minus_domain_constraint
            epsilon = 0.1
            return (plus_domain_constraint + minus_domain_constraint - translator.sqrt(diff * diff + epsilon**2)) / 2

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Set minus intersects if region intersects bounds but not completely contained in domain."""
        return self.plus_domain_constraint.intersects_hyperrect(center, radius) and not self.minus_domain.intersects_hyperrect(center, radius)

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """Set minus contains if region is inside bounds and doesn't intersect domain."""
        return self.contains_hyperrect(center, radius) and not self.minus_domain.intersects_hyperrect(center, radius)

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Set minus intersects if simplex intersects bounds but not completely contained in domain."""
        return self.plus_domain.intersects_simplex(vertices) and not self.minus_domain.intersects_simplex(vertices)

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """Set minus contains if simplex is inside bounds and doesn't intersect domain."""
        return self.plus_domain.contains_simplex(vertices) and not self.minus_domain.intersects_simplex(vertices)


class ApproachConeDomain(Domain):
    """
    Approach-cone domain in cylindrical r–z subspace embedded in an N-D state.

    Constraint: h(x) = r - |z| * tan(theta_max) <= 0 (inside cone)

    The cone constraint is h(x) <= 0, which means:
    - Points with r <= |z| * tan(theta_max) are INSIDE the cone (safe)
    - Points with r > |z| * tan(theta_max) are OUTSIDE the cone (unsafe)

    To make this a closed, bounded domain for sampling, we require additional bounds
    on the r and z dimensions.

    Args:
        dim: total state dimension
        r_index: index of radial coordinate r in the state
        z_index: index of out-of-plane coordinate z in the state
        theta_max_rad: half-angle of the cone (in radians)
        r_bounds: [r_min, r_max] bounds for radial coordinate (required for sampling)
        z_bounds: [z_min, z_max] bounds for axial coordinate (required for sampling)
        symmetric_z: when True, uses |z| (default True)
    """

    def __init__(
        self,
        dim: int,
        r_index: int,
        z_index: int,
        theta_max_rad: float,
        r_bounds: Optional[List[float]] = None,
        z_bounds: Optional[List[float]] = None,
        symmetric_z: bool = True,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not (0 <= r_index < dim) or not (0 <= z_index < dim):
            raise ValueError("r_index and z_index must be valid state dimensions")

        self._dim_full = dim
        self.r_index = int(r_index)
        self.z_index = int(z_index)
        self.theta_max_rad = float(theta_max_rad)
        self.tan_theta = np.tan(self.theta_max_rad)
        self.symmetric_z = bool(symmetric_z)

        # Store bounds for sampling and volume calculation
        self.r_bounds = r_bounds if r_bounds is not None else [0.0, 1.0]
        self.z_bounds = z_bounds if z_bounds is not None else [-1.0, 1.0]

        if self.r_bounds[0] < 0:
            raise ValueError("r_bounds must be non-negative (radial coordinate)")
        if self.r_bounds[0] >= self.r_bounds[1]:
            raise ValueError("r_bounds must satisfy r_min < r_max")
        if self.z_bounds[0] >= self.z_bounds[1]:
            raise ValueError("z_bounds must satisfy z_min < z_max")

        super().__init__(dim)

    def volume(self) -> float:
        """
        Calculate approximate volume of the cone domain.

        This is a conservative estimate assuming the cone is fully contained
        within the r-z bounds, and other dimensions are treated as having unit extent.

        For a cone in 2D (r-z plane), the volume is approximately:
        V ≈ (1/3) * π * r_max^2 * |z_max - z_min|

        For embedded dimensions, we multiply by the "volume" of other dimensions (assumed 1).
        """
        r_max = self.r_bounds[1]
        z_span = self.z_bounds[1] - self.z_bounds[0]

        # Approximate volume as a cone section
        # This is conservative and used primarily for sampling proportions
        cone_volume_2d = (1.0 / 3.0) * np.pi * (r_max**2) * z_span

        # For embedded dimensions, assume unit extent
        # The actual volume would need integration over the cone constraint
        return cone_volume_2d

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point(s) are inside the approach cone."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected points with dimensions {self.dim}, got {x.shape[-1]}")

        r = x[..., self.r_index]
        z = x[..., self.z_index]

        if self.symmetric_z:
            z_term = translator.abs(z)
        else:
            z_term = z

        # Cone constraint: r <= |z| * tan(theta_max) means INSIDE
        h = r - z_term * self.tan_theta
        return h <= 0

    def sample_points(self, num_points: int, device=None, use_torch=False, **kwargs) -> np.ndarray:
        """
        Sample points uniformly from the approach cone within the specified bounds.
        Uses rejection sampling to ensure points satisfy the cone constraint.
        """
        if num_points <= 0:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        r_min, r_max = self.r_bounds
        z_min, z_max = self.z_bounds

        collected_points = []
        max_attempts = num_points * 50  # Allow more attempts for complex geometry
        attempts = 0
        batch_size = min(2048, max(512, num_points * 10))

        if use_torch:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            while len(collected_points) < num_points and attempts < max_attempts:
                # Sample from bounding box
                r_samples = torch.empty(batch_size, device=device).uniform_(r_min, r_max)
                z_samples = torch.empty(batch_size, device=device).uniform_(z_min, z_max)

                # Check cone constraint
                z_term = torch.abs(z_samples) if self.symmetric_z else z_samples
                valid_mask = r_samples <= z_term * self.tan_theta

                # Create full state vectors (zeros for other dimensions)
                valid_r = r_samples[valid_mask]
                valid_z = z_samples[valid_mask]

                if valid_r.shape[0] > 0:
                    pts = torch.zeros((valid_r.shape[0], self.dim), device=device)
                    pts[:, self.r_index] = valid_r
                    pts[:, self.z_index] = valid_z
                    collected_points.append(pts)

                attempts += batch_size

            if not collected_points:
                print(f"Warning: Could not sample any points from approach cone after {attempts} attempts.")
                return torch.empty((0, self.dim), device=device)

            result = torch.cat(collected_points, dim=0)[:num_points]
            return result

        else:  # NumPy implementation
            while len(collected_points) < num_points and attempts < max_attempts:
                # Sample from bounding box
                r_samples = np.random.uniform(r_min, r_max, size=batch_size)
                z_samples = np.random.uniform(z_min, z_max, size=batch_size)

                # Check cone constraint
                z_term = np.abs(z_samples) if self.symmetric_z else z_samples
                valid_mask = r_samples <= z_term * self.tan_theta

                # Create full state vectors (zeros for other dimensions)
                valid_r = r_samples[valid_mask]
                valid_z = z_samples[valid_mask]

                if valid_r.shape[0] > 0:
                    pts = np.zeros((valid_r.shape[0], self.dim), dtype=np.float64)
                    pts[:, self.r_index] = valid_r
                    pts[:, self.z_index] = valid_z
                    collected_points.append(pts)

                attempts += batch_size

            if not collected_points:
                print(f"Warning: Could not sample any points from approach cone after {attempts} attempts.")
                return np.empty((0, self.dim))

            result = np.vstack(collected_points)[:num_points]
            return result

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """
        Constraint: r - |z| * tan(theta_max) (negative inside cone, positive outside).

        For CBF convention where h(x) >= 0 means safe, we need to negate this
        to get: |z| * tan(theta_max) - r >= 0 when inside cone.

        However, looking at the safe_set usage, the ApproachConeDomain defines
        the safe region, so we want positive values inside the cone.
        """
        if translator is None:
            translator = NumpyTranslator()
        x = translator.to_format(x)

        if x.ndim == 1 and len(x) != self.dim:
            raise ValueError(f"Expected {self.dim}D point, got {len(x)}D")
        elif x.ndim > 2:
            raise ValueError(f"Input must be 1D or 2D array, got {x.ndim}D")

        r = x[..., self.r_index]
        z = x[..., self.z_index]
        z_term = translator.abs(z) if self.symmetric_z else z

        # Return positive inside cone (safe), negative outside cone (unsafe)
        # h(x) = |z| * tan(theta) - r
        # When r is small (close to axis), h is positive (safe)
        # When r is large (far from axis), h is negative (unsafe)
        return z_term * self.tan_theta - r

    def intersects_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """
        Check if cone intersects with hyperrectangular region.

        Exact check in r–z plane for rectangle [r_min,r_max] x [z_min,z_max]:
        Intersects iff there exists a point (r,z) in the rectangle such that
        r <= |z| * tan(theta_max).

        This is true if r_min <= max(|z_min|, |z_max|) * tan(theta_max).
        """
        r_min = center[self.r_index] - radius[self.r_index]
        r_max = center[self.r_index] + radius[self.r_index]
        z_min = center[self.z_index] - radius[self.z_index]
        z_max = center[self.z_index] + radius[self.z_index]

        # Clamp r_min to be non-negative (r is radial distance)
        r_min = max(0.0, r_min)

        if self.symmetric_z:
            # Maximum |z| in the interval
            max_abs_z = max(abs(z_min), abs(z_max))
        else:
            # For non-symmetric, use z directly
            max_abs_z = z_max

        # Cone boundary at this z: r_cone = |z| * tan(theta)
        r_cone_at_max_z = max_abs_z * self.tan_theta

        # Intersects if the rectangle's r-range overlaps with [0, r_cone_at_max_z]
        return r_min <= r_cone_at_max_z

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """
        Check if cone completely contains the hyperrectangular region.

        Rectangle fully contained iff all corners satisfy r <= |z| * tan(theta_max).
        For a rectangle in r-z plane, the critical point is at maximum r.
        """
        r_min = center[self.r_index] - radius[self.r_index]
        r_max = center[self.r_index] + radius[self.r_index]
        z_min = center[self.z_index] - radius[self.z_index]
        z_max = center[self.z_index] + radius[self.z_index]

        # Clamp r_min to be non-negative
        r_min = max(0.0, r_min)

        if self.symmetric_z:
            # Minimum |z| in the interval (closest to origin)
            if z_min <= 0.0 <= z_max:
                min_abs_z = 0.0
            else:
                min_abs_z = min(abs(z_min), abs(z_max))
        else:
            # For non-symmetric, use minimum z
            min_abs_z = z_min

        # At the minimum |z|, what is the maximum allowed r?
        r_cone_at_min_z = min_abs_z * self.tan_theta

        # Contained if maximum r is within cone at minimum |z|
        return r_max <= r_cone_at_min_z

    def intersects_simplex(self, vertices: np.ndarray) -> bool:
        """Check intersection with simplicial region (conservative check)."""
        # Conservative check: any vertex inside domain
        for v in vertices:
            if self.contains(v):
                return True
        return False

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """Check if cone contains entire simplex (conservative check)."""
        # Conservative check: all vertices inside domain
        for v in vertices:
            if not self.contains(v):
                return False
        return True


def parse_domain_definition(domain_def: Dict, input_domain: List[List[float]]) -> Domain:
    """
    Parse a domain definition dictionary into a Domain object.

    Args:
        domain_def: Dictionary defining the domain
        input_domain: Overall input domain bounds

    Returns:
        Domain object
    """
    domain_type = domain_def.get("type")

    if domain_type == "circle":
        return CircleDomain(domain_def["center"], domain_def["radius"])

    elif domain_type == "circle_exterior":
        circle_domain = CircleDomain(domain_def["center"], domain_def["radius"])
        return ComplementDomain(circle_domain, input_domain)

    elif domain_type == "box":
        return BoxDomain(domain_def["bounds"])

    elif domain_type == "approach_cone":
        # Support degrees or radians
        if "theta_max_deg" in domain_def:
            theta_max_rad = np.deg2rad(domain_def["theta_max_deg"])
        elif "theta_max_rad" in domain_def:
            theta_max_rad = float(domain_def["theta_max_rad"])
        else:
            raise ValueError("approach_cone requires 'theta_max_deg' or 'theta_max_rad'")

        r_index = int(domain_def.get("r_index", 0))
        z_index = int(domain_def.get("z_index", 2))
        symmetric_z = bool(domain_def.get("symmetric_z", True))

        # Extract r and z bounds from input_domain
        r_bounds = domain_def.get("r_bounds", input_domain[r_index])
        z_bounds = domain_def.get("z_bounds", input_domain[z_index])

        return ApproachConeDomain(
            dim=len(input_domain), r_index=r_index, z_index=z_index, theta_max_rad=theta_max_rad, r_bounds=r_bounds, z_bounds=z_bounds, symmetric_z=symmetric_z
        )
    elif domain_type == "union":
        if "regions" in domain_def:
            sub_domains = [parse_domain_definition(region, input_domain) for region in domain_def["regions"]]
        elif "obstacles" in domain_def:
            sub_domains = [parse_domain_definition({"type": "circle", **obs}, input_domain) for obs in domain_def["obstacles"]]
        else:
            raise ValueError("Union domain must specify 'regions' or 'obstacles'")
        return UnionDomain(sub_domains)

    elif domain_type == "intersection":
        if "regions" in domain_def:
            sub_domains = [parse_domain_definition(region, input_domain) for region in domain_def["regions"]]
        else:
            raise ValueError("Intersection domain must specify 'regions'")
        return IntersectionDomain(sub_domains)

    elif domain_type == "complement":
        base_domain = parse_domain_definition(domain_def["of"], input_domain)
        return ComplementDomain(base_domain, input_domain)

    else:
        raise ValueError(f"Unknown domain type: {domain_type}")


def visualize_domain_2d(domain: Domain, bounds: List[List[float]], resolution: int = 100, ax=None):
    """
    Visualize a 2D domain with improved performance.

    Args:
        domain: Domain to visualize
        bounds: Plotting bounds [[x_min, x_max], [y_min, y_max]]
        resolution: Grid resolution for visualization
        ax: Matplotlib axes (optional)

    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Create grid
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate domain membership using vectorized operations
        points = np.column_stack([X.ravel(), Y.ravel()])
        contained = domain.contains(points)
        Z = contained.reshape(X.shape).astype(float)

        # Plot
        ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=["lightblue"], alpha=0.7)
        ax.contour(X, Y, Z, levels=[0.5], colors=["blue"], linewidths=2)

        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        return ax

    except ImportError:
        print("Matplotlib not available for visualization")
        return None


def unsafe_region(sample, dynamics, require_complete_containment=False) -> bool:
    """
    Sound check if a sample region overlaps with the unsafe region defined by the dynamics model.

    Args:
        sample: CertificationRegion object (HyperrectangularRegion or SimplicialRegion)
        dynamics: The dynamics model containing the unsafe_domain property.
        require_complete_containment: If True, only return True if the entire sample is contained
                                    in the unsafe region. If False, return True if any part of
                                    the sample intersects the unsafe region.

    Returns:
        bool: True if the sample overlaps with (or is contained in) the unsafe region, False otherwise.
    """
    # Get the unsafe domain from dynamics
    unsafe_domain = dynamics.unsafe_domain
    if unsafe_domain is None:
        return False

    # Ensure we have a Domain object
    if not isinstance(unsafe_domain, Domain):
        raise ValueError(f"unsafe_domain must be a Domain object, got {type(unsafe_domain)}")

    # Use the unified interface from Domain class
    if require_complete_containment:
        return unsafe_domain.contains_region(sample)
    else:
        return unsafe_domain.intersects_region(sample)
