"""
Base classes for activation function relaxations in CROWN-based verification.

This module defines the abstract base class and enums for implementing
linear relaxations of activation functions and their derivatives.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class ActivationRelaxation(ABC):
    """
    Abstract base class for defining linear relaxations for activation functions
    and their derivatives, required for CROWN-based verification.
    """

    @abstractmethod
    def relax_activation(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the linear relaxation for the activation function σ(y).

        For a given input range [lb, ub], finds α_L, β_L, α_U, β_U such that:
        α_L * y + β_L <= σ(y) <= α_U * y + β_U

        Args:
            lb: Lower bounds of the pre-activation (y).
            ub: Upper bounds of the pre-activation (y).

        Returns:
            A tuple (alpha_L, beta_L, alpha_U, beta_U).
        """
        pass

    @abstractmethod
    def relax_activation_derivative(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the linear relaxation for the activation's derivative σ'(y).

        For a given input range [lb, ub], finds γ_L, δ_L, γ_U, δ_U such that:
        γ_L * y + δ_L <= σ'(y) <= γ_U * y + δ_U

        Args:
            lb: Lower bounds of the pre-activation (y).
            ub: Upper bounds of the pre-activation (y).

        Returns:
            A tuple (gamma_L, delta_L, gamma_U, delta_U).
        """
        pass

    @abstractmethod
    def apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the activation function σ(y) to the input tensor.

        Args:
            y: Pre-activation values.

        Returns:
            σ(y): Post-activation values.
        """
        pass

    @abstractmethod
    def apply_activation_derivative(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the activation derivative σ'(y) to the input tensor.

        Args:
            y: Pre-activation values.

        Returns:
            σ'(y): Derivative values.
        """
        pass
