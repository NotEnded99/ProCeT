"""
LeakyReLU activation relaxation for CROWN-based verification.

This module implements linear relaxations for the LeakyReLU activation function
and its derivative.
"""

from typing import Tuple

import torch

from .activation_relaxations import ActivationRelaxation


class LeakyReLUActivationRelaxation(ActivationRelaxation):
    """
    LeakyReLU activation relaxation implementation.

    LeakyReLU is defined as max(x, α*x) where α is the negative slope parameter.
    For the standard LeakyReLU, α = 0.01.
    """

    def __init__(self, negative_slope: float = 0.01):
        """
        Initialize the LeakyReLU relaxation.

        Args:
            negative_slope: The negative slope parameter α
        """
        self.negative_slope = negative_slope

    def relax_activation(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes linear relaxation for LeakyReLU(y) = max(y, α*y).

        LeakyReLU is piecewise linear with a "kink" at y=0:
        - For y >= 0: LeakyReLU(y) = y
        - For y < 0: LeakyReLU(y) = α*y

        This vectorized implementation uses torch.where for better GPU parallelism.

        Args:
            lb: Lower bounds of pre-activation
            ub: Upper bounds of pre-activation

        Returns:
            Tuple of (alpha_L, beta_L, alpha_U, beta_U) for linear bounds
        """
        # Initialize output tensors
        alpha_L = torch.zeros_like(lb)
        beta_L = torch.zeros_like(lb)
        alpha_U = torch.zeros_like(lb)
        beta_U = torch.zeros_like(lb)

        # Define all masks upfront
        unstable_mask = (lb < 0) & (ub > 0)
        active_mask = lb >= 0
        negative_mask = ub <= 0

        # Pre-compute values for unstable case
        # Upper bound: Line connecting (lb, α*lb) and (ub, ub)
        alpha_U_unstable = (ub - self.negative_slope * lb) / torch.clamp(ub - lb, min=1e-12)
        beta_U_unstable = ub - alpha_U_unstable * ub

        # Apply bounds using torch.where (vectorized, no branching)
        # Case 1: Active region (y >= 0) -> LeakyReLU(y) = y
        alpha_L = torch.where(active_mask, torch.ones_like(alpha_L), alpha_L)
        alpha_U = torch.where(active_mask, torch.ones_like(alpha_U), alpha_U)

        # Case 2: Negative region (y <= 0) -> LeakyReLU(y) = α*y
        alpha_L = torch.where(negative_mask, torch.full_like(alpha_L, self.negative_slope), alpha_L)
        alpha_U = torch.where(negative_mask, torch.full_like(alpha_U, self.negative_slope), alpha_U)

        # Case 3: Unstable region -> Lower bound y = α*x, Upper bound = secant
        alpha_L = torch.where(unstable_mask, torch.full_like(alpha_L, self.negative_slope), alpha_L)
        # beta_L already 0
        alpha_U = torch.where(unstable_mask, alpha_U_unstable, alpha_U)
        beta_U = torch.where(unstable_mask, beta_U_unstable, beta_U)

        return alpha_L, beta_L, alpha_U, beta_U

    def relax_activation_derivative(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes linear relaxation for LeakyReLU'(y).

        The derivative of LeakyReLU is a step function:
        - For y > 0: LeakyReLU'(y) = 1
        - For y < 0: LeakyReLU'(y) = α
        - At y = 0: undefined (we can choose either value)

        Args:
            lb: Lower bounds of pre-activation
            ub: Upper bounds of pre-activation

        Returns:
            Tuple of (gamma_L, delta_L, gamma_U, delta_U) for derivative bounds
        """
        gamma_L = torch.zeros_like(lb)
        delta_L = torch.zeros_like(lb)
        gamma_U = torch.zeros_like(lb)
        delta_U = torch.zeros_like(lb)

        # Masks for the three cases
        unstable_mask = (lb < 0) & (ub > 0)
        active_mask = lb >= 0
        negative_mask = ub <= 0

        # Case 1: Always active (y >= 0) -> LeakyReLU'(y) = 1
        gamma_L[active_mask] = 0.0
        delta_L[active_mask] = 1.0
        gamma_U[active_mask] = 0.0
        delta_U[active_mask] = 1.0

        # Case 2: Always in negative region (y <= 0) -> LeakyReLU'(y) = α
        gamma_L[negative_mask] = 0.0
        delta_L[negative_mask] = self.negative_slope
        gamma_U[negative_mask] = 0.0
        delta_U[negative_mask] = self.negative_slope

        # Case 3: Unstable -> LeakyReLU'(y) is in [α, 1]
        # Lower bound: α <= LeakyReLU'(y)
        gamma_L[unstable_mask] = 0.0
        delta_L[unstable_mask] = self.negative_slope

        # Upper bound: LeakyReLU'(y) <= 1
        gamma_U[unstable_mask] = 0.0
        delta_U[unstable_mask] = 1.0

        return gamma_L, delta_L, gamma_U, delta_U

    def apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the LeakyReLU activation function."""
        return torch.nn.functional.leaky_relu(y, negative_slope=self.negative_slope)

    def apply_activation_derivative(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the LeakyReLU derivative."""
        return torch.where(y > 0, 1.0, self.negative_slope)
