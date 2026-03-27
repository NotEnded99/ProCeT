"""
Unified regions package for certified neural approximations.

This package provides a unified interface for different types of certification regions,
including hyperrectangular and simplicial regions, with their corresponding meshes
and generators.

The main abstractions are organized in a layered architecture:

1. **Region Classes** (AbstractRegion, HyperrectangularRegion, SimplicialRegion):
   - Unified geometric and certification regions that directly handle both geometric
     operations and verification-specific functionality
   - Handle fundamental operations: volume calculation, point containment, uniform sampling,
     and adaptive splitting based on approximation quality
   - Track output dimensions and splitting history for verification workflows
   - Combine pure geometry with neural network verification needs in a single interface

2. **Mesh Classes** (AbstractMesh, HyperrectangularMesh, SimplicialMesh):
   - Manage collections of regions that tessellate a domain
   - Provide spatial indexing and neighbor relationships
   - Handle domain decomposition strategies (uniform grids vs. adaptive triangulations)
   - Support queries like finding regions containing specific points
   - Enable efficient traversal and refinement of the verification space

3. **RegionGenerator Classes** (AbstractRegionGenerator, HyperrectangularRegionGenerator, SimplicialRegionGenerator):
   - Create initial meshes and regions from dynamics model specifications
   - Implement different meshing strategies (structured grids, Delaunay triangulation)
   - Generate verification regions tailored to specific dynamics and approximation needs
   - Provide factory methods for creating appropriate region types based on problem characteristics

Concrete implementations:
- HyperrectangularRegion/HyperrectangularMesh: Grid-based approach for structured domains
- SimplicialRegion/SimplicialMesh: Delaunay triangulation approach for complex geometries

Note: The primitive classes (Hyperrectangle, Simplex) have been merged into their corresponding
region classes to eliminate architectural redundancy and simplify the interface.
"""

# Import abstract base classes
from .base import AbstractMesh, AbstractRegion, AbstractRegionGenerator

# Import unified generators
from .generators import RegionGenerator, create_region_generator, generate_hyperrectangular_regions, generate_simplicial_regions

# Import concrete implementations
from .hyperrectangular import HyperrectangularMesh, HyperrectangularRegion, HyperrectangularRegionGenerator
from .simplicial import SimplicialMesh, SimplicialRegion, SimplicialRegionGenerator

# Backward compatibility aliases
CertificationRegion = HyperrectangularRegion

__all__ = [
    # Abstract base classes
    "AbstractRegion",
    "AbstractMesh",
    "AbstractRegionGenerator",
    # Hyperrectangular implementations
    "HyperrectangularRegion",
    "HyperrectangularMesh",
    "HyperrectangularRegionGenerator",
    # Simplicial implementations
    "SimplicialRegion",
    "SimplicialMesh",
    "SimplicialRegionGenerator",
    # Unified interface
    "RegionGenerator",
    "create_region_generator",
    "generate_hyperrectangular_regions",
    "generate_simplicial_regions",
    # Backward compatibility
    "CertificationRegion",
]
