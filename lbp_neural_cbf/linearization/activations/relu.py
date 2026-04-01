"""
ReLU activation relaxation for CROWN-based verification.

This module implements linear relaxations for the ReLU activation function
and its derivative (step function).
"""

from typing import Tuple

import torch

from .activation_relaxations import ActivationRelaxation


class ReLUActivationRelaxation(ActivationRelaxation):
    """Concrete implementation of activation relaxation for the ReLU function."""

    def relax_activation(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the linear relaxation for ReLU.
        """
        # Initialize output tensors
        alpha_L = torch.zeros_like(lb)
        beta_L = torch.zeros_like(lb)
        alpha_U = torch.zeros_like(lb)
        beta_U = torch.zeros_like(lb)

        # Define all masks upfront
        unstable_mask = (lb < 0) & (ub > 0)
        active_mask = lb >= 0
        # inactive_mask = ub <= 0  # Not needed, zeros already set

        # Pre-compute values for unstable case
        alpha_U_unstable = ub / torch.clamp(ub - lb, min=1e-12)
        beta_U_unstable = -lb * alpha_U_unstable

        # Apply bounds using torch.where (vectorized, no branching)
        # Case 1: Active neurons (y >= 0) -> ReLU(y) = y
        alpha_L = torch.where(active_mask, torch.ones_like(alpha_L), alpha_L)
        alpha_U = torch.where(active_mask, torch.ones_like(alpha_U), alpha_U)

        # Case 2: Inactive neurons (y <= 0) -> ReLU(y) = 0
        # Already initialized to zeros, no action needed

        # Case 3: Unstable neurons -> Upper bound is secant line
        alpha_U = torch.where(unstable_mask, alpha_U_unstable, alpha_U)
        beta_U = torch.where(unstable_mask, beta_U_unstable, beta_U)

        return alpha_L, beta_L, alpha_U, beta_U

    # def relax_activation_derivative(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Computes the linear relaxation for the derivative of ReLU (a step function).

    #     Returns γ_L, δ_L, γ_U, δ_U such that:
    #     γ_L * y + δ_L <= σ'(y) <= γ_U * y + δ_U

    #     For ReLU derivative:
    #     - Active neurons (y >= 0): σ'(y) = 1
    #     - Inactive neurons (y <= 0): σ'(y) = 0
    #     - Unstable neurons (lb < 0 < ub): σ'(y) ∈ [0, 1]
    #     """
    #     gamma_L = torch.zeros_like(lb)
    #     delta_L = torch.zeros_like(lb)
    #     gamma_U = torch.zeros_like(lb)
    #     delta_U = torch.zeros_like(lb)

    #     # Masks for the three cases
    #     unstable_mask = (lb < 0) & (ub > 0)
    #     active_mask = lb >= 0

    #     # Case 1: Neuron is always active (y >= 0) -> ReLU'(y) = 1
    #     # We want: 1 <= σ'(y) <= 1, so we use γ*y + δ = 1
    #     # Since we want a constant bound, we use γ=0, δ=1 to get 0*y + 1 = 1
    #     # For active neurons, we can use γ=0, δ=1 to get a constant 1
    #     gamma_L[active_mask] = 0.0
    #     delta_L[active_mask] = 1.0  # This gives the constant bound 0*y + 1 = 1
    #     gamma_U[active_mask] = 0.0
    #     delta_U[active_mask] = 1.0

    #     # Case 2: Neuron is always inactive (y <= 0) -> ReLU'(y) = 0
    #     # We want: 0 <= σ'(y) <= 0, achieved with γ=0, δ=0
    #     # gamma_L and delta_L are already 0 for inactive neurons

    #     # Case 3: Neuron is unstable -> ReLU'(y) is in [0, 1]
    #     # Lower bound: 0 <= σ'(y), achieved with γ_L=0, δ_L=0 (gives 0*y + 0 = 0)
    #     # Upper bound: σ'(y) <= 1, achieved with γ_U=0, δ_U=1 (gives 0*y + 1 = 1)
    #     gamma_U[unstable_mask] = 0.0
    #     delta_U[unstable_mask] = 1.0

    #     return gamma_L, delta_L, gamma_U, delta_U

    def relax_activation_derivative(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the linear relaxation for the derivative of ReLU (a step function).
        """
        # γ (gamma) is ALWAYS 0 for all three cases in ReLU derivative relaxation
        gamma_L = torch.zeros_like(lb)
        # delta_L = torch.zeros_like(lb)
        gamma_U = torch.zeros_like(lb)
        # delta_U = torch.zeros_like(lb)
        
        # Masks for the cases
        unstable_mask = (lb < 0) & (ub > 0)
        active_mask = lb >= 0
        
        # We can directly cast boolean masks to the target dtype. 
        # This is a safe, out-of-place operation that maintains the compute graph.
        
        # δ_L is 1.0 only for strictly active neurons, 0.0 otherwise.
        delta_L = active_mask.to(dtype=lb.dtype)
        
        # δ_U is 1.0 for active AND unstable neurons (i.e., anytime ub > 0 and it's not strictly inactive)
        # Using logical OR (|) for masks
        delta_U = (active_mask | unstable_mask).to(dtype=ub.dtype)

        return gamma_L, delta_L, gamma_U, delta_U

    def apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the ReLU activation function."""
        return torch.relu(y)

    def apply_activation_derivative(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the ReLU derivative (step function)."""
        return (y > 0).to(dtype=y.dtype)
