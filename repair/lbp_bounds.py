"""
LBP Lower Bound Computation for ICGAR

This module provides methods for computing LBP lower bounds and their gradients
for simplicial regions, which are used by ICGAR to compute the certificate manifold.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.linearization.crown import CrownLinearization
from lbp_neural_cbf.regions.simplicial import SimplicialRegion


class LBPLowerBoundComputer:
    """
    Computes LBP lower bounds and gradients for simplicial regions.

    This is used to:
    1. Compute h̲_v(θ) for verified simplices (for manifold definition)
    2. Compute gradients ∇_θ h̲_v(θ) (for tangent space)
    3. Evaluate loss on failed simplices during repair
    """

    def __init__(self, model, device=None, dtype=torch.float64):
        """
        Initialize the LBP bounds computer.

        Args:
            model: BarrierNN neural network
            device: torch device (cuda/cpu)
            dtype: torch data type
        """
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Store network structure for gradient computation
        self._cache_network_structure()

    def _cache_network_structure(self):
        """Cache network structure for efficient gradient computation."""
        self.layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.layers.append({
                    'name': name,
                    'module': module,
                    'weight': module.weight,
                    'bias': module.bias,
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'activation': self._get_activation_type(name)
                })

    def _get_activation_type(self, layer_name):
        """Determine activation type following a linear layer."""
        # This is a heuristic - actual activation depends on network structure
        # For ICGAR, we typically use ReLU networks
        return 'relu'  # Default to ReLU

    def compute_lower_bound(self, region):
        """
        Compute the LBP lower bound h̲ over a simplicial region.

        Args:
            region: SimplicialRegion

        Returns:
            lower_bound: scalar lower bound value
        """
        # For simplicial regions, evaluate at all vertices and take minimum
        vertices = torch.tensor(region.vertices, device=self.device, dtype=self.dtype)

        # Evaluate network at all vertices
        with torch.no_grad():
            outputs = self.model(vertices)  # Shape: (n_vertices, 1)

        # Lower bound is minimum over all vertices
        lower_bound = outputs.min().item()

        return lower_bound

    def compute_upper_bound(self, region):
        """
        Compute the LBP upper bound over a simplicial region.

        Args:
            region: SimplicialRegion

        Returns:
            upper_bound: scalar upper bound value
        """
        # For simplicial regions, evaluate at all vertices and take maximum
        vertices = torch.tensor(region.vertices, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model(vertices)

        # Upper bound is maximum over all vertices
        upper_bound = outputs.max().item()

        return upper_bound

    def compute_bounds(self, region):
        """
        Compute both lower and upper bounds.

        Args:
            region: SimplicialRegion

        Returns:
            (lower_bound, upper_bound): tuple of scalars
        """
        vertices = torch.tensor(region.vertices, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model(vertices)

        lower_bound = outputs.min().item()
        upper_bound = outputs.max().item()

        return lower_bound, upper_bound

    def compute_gradient_of_lower_bound(self, region):
        """
        Compute gradient of LBP lower bound w.r.t. parameters: ∇_θ h̲

        This identifies which vertex achieves the minimum and computes gradient through it.

        Args:
            region: SimplicialRegion

        Returns:
            gradient: vector of size |θ| (flattened parameters)
        """
        vertices = torch.tensor(region.vertices, device=self.device, dtype=self.dtype)
        vertices.requires_grad_(False)  # Input points don't need grad

        # Evaluate network at all vertices
        outputs = self.model(vertices)  # Shape: (n_vertices, 1)

        # Find which vertex achieves the minimum
        min_idx = torch.argmin(outputs).item()
        min_vertex = vertices[min_idx:min_idx+1]

        # Compute gradient of output w.r.t. parameters for the minimizing vertex
        min_vertex.requires_grad_(True)
        output = self.model(min_vertex)

        # Compute gradient (backward pass)
        self.model.zero_grad()
        output.backward()

        # Collect gradients from all parameters
        gradients = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.flatten().detach())

        # Flatten into single vector
        gradient = torch.cat(gradients).cpu()

        return gradient.numpy()

    def compute_gradients_batch(self, regions):
        """
        Compute gradients for multiple regions in batch.

        Args:
            regions: list of SimplicialRegion objects

        Returns:
            gradients: numpy array of shape (n_regions, |θ|)
        """
        n_regions = len(regions)
        n_params = sum(p.numel() for p in self.model.parameters())

        gradients = np.zeros((n_regions, n_params))

        for i, region in enumerate(regions):
            gradients[i] = self.compute_gradient_of_lower_bound(region)

        return gradients

    def compute_loss_on_regions(self, regions, reg_lambda=0.0, initial_params=None):
        """
        Compute repair loss on a set of regions.

        Loss = Σ_{region} [ -h̲(region) ]_+  + λ||θ - θ₀||²

        Args:
            regions: list of SimplicialRegion objects (typically failed regions)
            reg_lambda: L2 regularization weight
            initial_params: Initial parameter values (for regularization)

        Returns:
            loss: scalar loss value
        """
        # Hinge loss on regions
        hinge_loss = 0.0
        for region in regions:
            lower_bound = self.compute_lower_bound(region)
            hinge_loss += max(0, -lower_bound)

        # L2 regularization
        reg_loss = 0.0
        if reg_lambda > 0.0 and initial_params is not None:
            current_params = self._get_flattened_params()
            reg_loss = reg_lambda * np.sum((current_params - initial_params) ** 2)

        total_loss = hinge_loss + reg_loss

        return total_loss

    def _get_flattened_params(self):
        """Get flattened parameter vector."""
        params = []
        for param in self.model.parameters():
            params.append(param.detach().cpu().flatten().numpy())
        return np.concatenate(params)

    def set_flattened_params(self, flat_params):
        """Set parameters from flattened vector."""
        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                param_flat = flat_params[idx:idx+numel]
                param.data = torch.tensor(
                    param_flat.reshape(param.shape),
                    device=self.device,
                    dtype=self.dtype
                )
                idx += numel


class LBPBoundsComputer(LBPLowerBoundComputer):
    """
    Alias for backward compatibility.

    LBPBoundsComputer provides the same functionality as LBPLowerBoundComputer
    but with a more general name that could extend to upper bounds as well.
    """
    pass


def compute_lbp_bounds_for_simplex(model, simplex_vertices, device=None):
    """
    Convenience function to compute LBP bounds for a simplex.

    Args:
        model: BarrierNN
        simplex_vertices: array of shape (n_vertices, n_dims)
        device: torch device

    Returns:
        (lower_bound, upper_bound): tuple of scalars
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices = torch.tensor(simplex_vertices, device=device)

    with torch.no_grad():
        outputs = model(vertices)

    lower_bound = outputs.min().item()
    upper_bound = outputs.max().item()

    return lower_bound, upper_bound
