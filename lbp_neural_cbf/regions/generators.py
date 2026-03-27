"""
Unified region generators for verification.

This module provides the RegionGenerator interface and implementations
for creating verification regions from dynamics models.
"""

from typing import List

from .base import AbstractRegion, AbstractRegionGenerator
from .hyperrectangular import HyperrectangularRegionGenerator
from .simplicial import SimplicialRegionGenerator

# Export the main generators
__all__ = ["RegionGenerator", "HyperrectangularRegionGenerator", "SimplicialRegionGenerator", "create_region_generator"]

# Alias for the abstract base class
RegionGenerator = AbstractRegionGenerator


def create_region_generator(region_type: str = "hyperrectangular") -> AbstractRegionGenerator:
    """
    Factory function to create region generators.

    Args:
        region_type: Type of regions to generate ("hyperrectangular" or "simplicial")

    Returns:
        Appropriate region generator instance

    Raises:
        ValueError: If region_type is not supported
    """
    if region_type.lower() in ["hyperrectangular", "hyperrect", "grid"]:
        return HyperrectangularRegionGenerator()
    elif region_type.lower() in ["simplicial", "simplex", "delaunay"]:
        return SimplicialRegionGenerator()
    else:
        raise ValueError(f"Unsupported region type: {region_type}. " f"Supported types: 'hyperrectangular', 'simplicial'")


# Convenience functions for backward compatibility
def generate_hyperrectangular_regions(dynamics_model) -> List[AbstractRegion]:
    """Generate hyperrectangular regions for verification."""
    generator = HyperrectangularRegionGenerator()
    return generator.generate_regions(dynamics_model)


def generate_simplicial_regions(dynamics_model) -> List[AbstractRegion]:
    """Generate simplicial regions for verification."""
    generator = SimplicialRegionGenerator()
    return generator.generate_regions(dynamics_model)
