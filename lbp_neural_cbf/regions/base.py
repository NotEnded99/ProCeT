"""
Abstract base classes for unified region and mesh handling.

This module defines the core abstractions for certification regions and meshes,
providing a unified interface for both hyperrectangular and simplicial approaches.
"""

import abc
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np


class AbstractRegion(abc.ABC):
    """
    Abstract base class for all certification regions.

    This unified class combines geometric operations with verification-specific functionality,
    eliminating the need for separate primitive and region hierarchies.
    """

    def __init__(
        self,
        output_dim: int = None,
        nonlin_dependencies: List[int] = None,
    ):
        # Verification-specific attributes
        self.output_dim = output_dim

        if nonlin_dependencies is None:
            nonlin_dependencies = []
        self.nonlin_dependencies = nonlin_dependencies

        # Compute geometric properties (implemented by subclasses)
        self.centroid = self._compute_centroid()
        self.volume = self._compute_volume()

    # Pure geometric methods (essential for all regions)
    @abc.abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the region."""
        pass

    @abc.abstractmethod
    def _compute_centroid(self) -> np.ndarray:
        """Compute the centroid/center of the region."""
        pass

    @abc.abstractmethod
    def _compute_volume(self) -> float:
        """Compute the volume/measure of the region."""
        pass

    @abc.abstractmethod
    def contains_point(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a point is inside the region."""
        pass

    @abc.abstractmethod
    def sample_uniform(self, n_samples: int = 1) -> np.ndarray:
        """Sample points uniformly from the region."""
        pass

    @abc.abstractmethod
    def get_bounds(self) -> np.ndarray:
        """Get bounding box as (min, max) pairs for each dimension."""
        pass

    # Verification-specific methods
    @abc.abstractmethod
    def split(self, split_criterion=None, taylor_approximation=None, dynamics=None, timeout=False) -> List["AbstractRegion"]:
        """Split the region into smaller regions."""
        pass

    @abc.abstractmethod
    def _determine_split_dimension(self, taylor_approximation, dynamics, timeout=False):
        """
        Determine how to split the region based on approximation error.

        Args:
            taylor_approximation: Taylor approximation function
            dynamics: True dynamics function
            timeout: Whether this is a timeout case

        Returns:
            Dictionary with split parameters, or None if no split is recommended.
            The dictionary format allows different region types to specify different
            split strategies while maintaining a unified interface.
        """
        pass

    # Unified interface methods (combine geometric and verification functionality)
    def get_center(self) -> np.ndarray:
        """Get the center point of the region."""
        return self.centroid

    def lebesguemeasure(self) -> float:
        """Calculate the size/volume of the region."""
        return self.volume

    def incrementsplitdim(self):
        """Move to the next dimension for splitting."""
        self.last_split_dim = (self.last_split_dim + 1) % len(self.split_dims)

    @property
    def center(self) -> np.ndarray:
        """Compatibility property for existing code."""
        return self.get_center()

    @abc.abstractmethod
    def intersects_domain(self, domain) -> bool:
        """Check if this region intersects with a domain."""
        pass

    @abc.abstractmethod
    def contained_in_domain(self, domain) -> bool:
        """Check if this region is completely contained in a domain."""
        pass


class AbstractMesh(abc.ABC):
    """
    Abstract base class for mesh implementations.

    This defines the interface for managing collections of regions that form
    a mesh over a domain, supporting both hyperrectangular and simplicial approaches.
    """

    def __init__(self, domain_bounds):
        """
        Initialize a mesh over a given domain.

        Args:
            domain_bounds: List of (min, max) bounds for each dimension, or a domain object with .bounds
        """
        # Accept either a domain object or a bounds list
        if hasattr(domain_bounds, "bounds"):
            self.domain_bounds = domain_bounds.bounds
        else:
            self.domain_bounds = domain_bounds
        self.dim = len(self.domain_bounds)

    @abc.abstractmethod
    def get_regions(self, output_dim: int, nonlin_dependencies: Optional[List[Tuple[int, bool]]] = None) -> List[AbstractRegion]:
        """
        Get all regions in the mesh for a specific output dimension.

        Args:
            output_dim: Output dimension for verification
            nonlin_dependencies: Optional nonlinear dependencies for the regions

        Returns:
            List of regions
        """
        pass

    @abc.abstractmethod
    def total_volume(self) -> float:
        """Calculate total volume covered by the mesh."""
        pass

    @abc.abstractmethod
    def find_region_containing_point(self, point: np.ndarray) -> Optional[int]:
        """
        Find which region contains a given point.

        Args:
            point: Point to locate

        Returns:
            Index of the region containing the point, or None if not found
        """
        pass

    def __len__(self):
        """Return the number of regions in the mesh."""
        return len(self.get_regions(0))  # Use output_dim=0 for counting

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dim}D, {len(self)} regions)"


class AbstractRegionGenerator(abc.ABC):
    """
    Abstract base class for generating verification regions.

    This defines the interface for creating regions from dynamics models,
    supporting different mesh types and generation strategies.
    """

    @abc.abstractmethod
    def generate_regions(
        self,
        dynamics_model,
        nonlin_dependencies_func: Optional[Callable[[int], Optional[List[Tuple[int, bool]]]]] = None,
    ) -> List[AbstractRegion]:
        """
        Generate regions for verification.

        Args:
            dynamics_model: Dynamics model containing domain and parameters
            nonlin_dependencies_func: Optional function that takes output_dim and returns
                                    list of (input_idx, is_nonlin) tuples for backward compatibility

        Returns:
            List of regions for verification
        """
        pass

    @abc.abstractmethod
    def create_mesh(self, dynamics_model) -> AbstractMesh:
        """
        Create a mesh for the given dynamics model.

        Args:
            dynamics_model: Dynamics model containing domain and parameters

        Returns:
            Mesh instance
        """
        pass
