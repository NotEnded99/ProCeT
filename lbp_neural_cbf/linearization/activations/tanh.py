"""
Tanh activation relaxation for CROWN-based verification.

This module implements linear relaxations for the tanh activation function
and its derivative.
"""

from typing import List, Tuple

import numpy as np
import torch

from .activation_relaxations import ActivationRelaxation


class TanhActivationRelaxation(ActivationRelaxation):
    """
    Tanh activation relaxation implementation.

    This class provides linear relaxations for the tanh function and its derivative
    using the analytical methods described in the bounding formulas.
    """

    def __init__(self):
        """
        Initialize the Tanh relaxation.
        """
        # Inflection point of tanh'(x)=sech²(x), where its concavity changes.
        self.x_inf_deriv = torch.atanh(1.0 / torch.sqrt(torch.tensor(3.0, dtype=torch.float64))).item()  # ≈ 0.6585

    def relax_activation(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes linear relaxation for tanh(y) - fully vectorized.

        For tanh, we need to handle different cases:
        - Convex region (y < 0): Lower bound is tangent, upper bound connects endpoints
        - Concave region (y > 0): Lower bound connects endpoints, upper bound is tangent
        - Mixed region: Bounds are parallel to the secant, tangent at specific points.

        Args:
            lb: Lower bounds of pre-activation
            ub: Upper bounds of pre-activation

        Returns:
            Tuple of (alpha_L, beta_L, alpha_U, beta_U) for linear bounds
        """
        # Use the vectorized _compute_tanh_bounds which now handles both scalar and tensor inputs
        alpha_L, beta_L, alpha_U, beta_U = self._compute_tanh_bounds(lb, ub)
        return alpha_L, beta_L, alpha_U, beta_U

    def relax_activation_derivative(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes linear relaxation for tanh'(y) = 1 - tanh²(y).

        The derivative of tanh has more complex convexity properties than tanh itself.

        Args:
            lb: Lower bounds of pre-activation
            ub: Upper bounds of pre-activation

        Returns:
            Tuple of (gamma_L, delta_L, gamma_U, delta_U) for derivative bounds
        """
        # Compute derivative relaxation for this interval
        gamma_L, delta_L, gamma_U, delta_U = self._compute_tanh_derivative_bounds(lb, ub)
        return gamma_L, delta_L, gamma_U, delta_U

    def _compute_tanh_bounds(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mathematically rigorous linear bounds for tanh over interval [lb, ub].
        """
        # Ensure inputs are tensors
        if not isinstance(lb, torch.Tensor):
            lb = torch.tensor(lb)
        if not isinstance(ub, torch.Tensor):
            ub = torch.tensor(ub)

        # Get original shape for output
        original_shape = lb.shape
        is_scalar = lb.dim() == 0

        # Flatten for processing
        lb_flat = lb.flatten()
        ub_flat = ub.flatten()

        # Compute for all elements at once
        y_l = torch.tanh(lb_flat)
        y_u = torch.tanh(ub_flat)

        # Initialize output tensors
        alpha_L = torch.zeros_like(lb_flat)
        beta_L = torch.zeros_like(lb_flat)
        alpha_U = torch.zeros_like(lb_flat)
        beta_U = torch.zeros_like(lb_flat)

        # Handle zero-width intervals (point evaluations)
        zero_width = torch.abs(ub_flat - lb_flat) < 1e-5
        if zero_width.any():
            slope_point = 1 - y_l[zero_width] ** 2
            intercept_point = y_l[zero_width] - slope_point * lb_flat[zero_width]
            alpha_L[zero_width] = slope_point
            beta_L[zero_width] = intercept_point
            alpha_U[zero_width] = slope_point
            beta_U[zero_width] = intercept_point

        # Process non-zero-width intervals
        non_zero = ~zero_width
        if not non_zero.any():
            return (alpha_L.reshape(original_shape), beta_L.reshape(original_shape), alpha_U.reshape(original_shape), beta_U.reshape(original_shape))

        # Compute secant slope for non-zero intervals
        secant_slope = torch.where(non_zero, (y_u - y_l) / torch.clamp(ub_flat - lb_flat, min=1e-6), torch.zeros_like(lb_flat))

        # === Define all masks upfront ===
        small_slope = non_zero & (torch.abs(secant_slope) < 1e-8)
        regular = non_zero & ~small_slope

        # Convexity regions
        convex = regular & (ub_flat <= 0)
        concave = regular & (lb_flat >= 0)
        mixed = regular & (lb_flat < 0) & (ub_flat > 0)

        # Near-saturation in convex/concave regions
        # Use 1e-3 threshold: tangent-based bounds become unsound in near-saturated regions
        near_sat_convex = convex & (secant_slope < 1e-3)
        near_sat_concave = concave & (secant_slope < 1e-3)

        # Valid (non-saturated) cases
        non_sat_convex = convex & ~near_sat_convex
        non_sat_concave = concave & ~near_sat_concave

        # === Compute values for all cases (even if mask is empty) ===
        # Pre-compute common values
        mid = (lb_flat + ub_flat) / 2
        tangent_slope_mid = 1 - torch.tanh(mid) ** 2
        beta_mid = torch.tanh(mid) - tangent_slope_mid * mid

        beta_secant = y_u - secant_slope * ub_flat

        mixed_slope_U = torch.min(1 - y_u**2, secant_slope)
        mixed_beta_U = y_u - mixed_slope_U * ub_flat
        mixed_slope_L = torch.min(1 - y_l**2, secant_slope)
        mixed_beta_L = y_l - mixed_slope_L * lb_flat

        # === Apply bounds using torch.where (no branching on .any()) ===
        # Combine all near-saturation cases
        all_near_sat = small_slope | near_sat_convex | near_sat_concave

        # Near-saturation: constant horizontal bounds
        alpha_L = torch.where(all_near_sat, torch.zeros_like(alpha_L), alpha_L)
        alpha_U = torch.where(all_near_sat, torch.zeros_like(alpha_U), alpha_U)
        beta_L = torch.where(all_near_sat, y_l, beta_L)
        beta_U = torch.where(all_near_sat, y_u, beta_U)

        # Convex region: lower=tangent at midpoint, upper=secant
        alpha_L = torch.where(non_sat_convex, tangent_slope_mid, alpha_L)
        beta_L = torch.where(non_sat_convex, beta_mid, beta_L)
        alpha_U = torch.where(non_sat_convex, secant_slope, alpha_U)
        beta_U = torch.where(non_sat_convex, beta_secant, beta_U)

        # Concave region: lower=secant, upper=tangent  at midpoint
        alpha_L = torch.where(non_sat_concave, secant_slope, alpha_L)
        beta_L = torch.where(non_sat_concave, beta_secant, beta_L)
        alpha_U = torch.where(non_sat_concave, tangent_slope_mid, alpha_U)
        beta_U = torch.where(non_sat_concave, beta_mid, beta_U)

        # Mixed region: lower=tangent at lower, upper=tangent at upper
        alpha_L = torch.where(mixed, mixed_slope_L, alpha_L)
        beta_L = torch.where(mixed, mixed_beta_L, beta_L)
        alpha_U = torch.where(mixed, mixed_slope_U, alpha_U)
        beta_U = torch.where(mixed, mixed_beta_U, beta_U)

        # Reshape to original shape
        alpha_L = alpha_L.reshape(original_shape)
        beta_L = beta_L.reshape(original_shape)
        alpha_U = alpha_U.reshape(original_shape)
        beta_U = beta_U.reshape(original_shape)

        # Return scalars if input was scalar
        if is_scalar:
            return alpha_L.item(), beta_L.item(), alpha_U.item(), beta_U.item()

        return alpha_L, beta_L, alpha_U, beta_U

    def _compute_tanh_derivative_bounds(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma_L = torch.zeros_like(lb)
        delta_L = torch.zeros_like(lb)
        gamma_U = torch.zeros_like(lb)
        delta_U = torch.zeros_like(lb)

        # Handle zero width case (point evaluation)
        isclose_mask = torch.isclose(lb, ub, atol=1e-12)
        if isclose_mask.any():
            lb_isclose = lb[isclose_mask]
            # For a point interval, both bounds should be the exact derivative value
            y = 1 - torch.tanh(lb_isclose) ** 2
            # The slope should be the second derivative at that point
            # For tanh: tanh''(x) = -2*tanh(x)*(1 - tanh²(x))
            tanh_val = torch.tanh(lb_isclose)
            second_deriv = -2 * tanh_val * y
            intercept = y - second_deriv * lb_isclose

            gamma_L[isclose_mask] = second_deriv
            delta_L[isclose_mask] = intercept
            gamma_U[isclose_mask] = second_deriv
            delta_U[isclose_mask] = intercept

        already_handled = isclose_mask

        # Process non-zero width intervals
        y_l = 1 - torch.tanh(lb.to(torch.float64)) ** 2
        y_u = 1 - torch.tanh(ub.to(torch.float64)) ** 2
        # Use torch.where to avoid division by zero/near-zero for already handled cases
        secant_slope = (y_u - y_l) / torch.clamp(ub - lb, min=1e-12)
        x_inf = self.x_inf_deriv

        # Shape: (size of lb, num_roots) or (batch, size of lb, num_roots) where num_roots = 3, and valid_roots_mask: (size of lb, num_roots) or (batch, size of lb, num_roots) bool
        # to indicate which entries contain valid roots (e.g. for cases with only 1 real root).
        tangent_cubic_roots, valid_roots_mask = self._solve_cubic_for_tangent(secant_slope)
        # Clamp to avoid gradient explosion in atanh backward: d/dx atanh(x) = 1/(1-x²)
        # Without clamp, x near ±1 causes inf gradient -> NaN after mul by 0
        # [ORIGINAL] tangent_points_x = torch.atanh(tangent_cubic_roots)
        eps = 1e-8
        tangent_points_x = torch.atanh(tangent_cubic_roots.clamp(-1 + eps, 1 - eps))  

        in_range = (tangent_points_x > lb.unsqueeze(-1)) & (tangent_points_x < ub.unsqueeze(-1)) & valid_roots_mask

        # First column with true value in each row, or in_range.size(-1) if none found
        first_in_range = torch.where(in_range, torch.arange(in_range.size(-1), device=in_range.device), in_range.size(-1)).min(dim=-1).values

        # Gather does not support negative indices. So replace in_range.size(-1) with 0 temporarily.
        # We'll filter using a where condition later, so that the inappropriate values are not used.
        first_in_range_aux = torch.where(first_in_range < in_range.size(-1), first_in_range, 0)

        # Select first_in_range points, or midpoint if none found
        # (size of lb,) or (batch, size of lb)
        x_tan = torch.where(
            first_in_range != in_range.size(-1), torch.gather(tangent_points_x, -1, first_in_range_aux.unsqueeze(-1)).squeeze(-1), (lb + ub) / 2
        )
        # We will use this x_tan only in Case A and Case B where there is exactly one tangent point in range
        # For Case C, we will use all tangent points in range to compute min/max residuals

        # Case A: Purely concave
        concave_mask = (lb >= -x_inf) & (ub <= x_inf) & (~already_handled)
        if concave_mask.any():
            m_lower = secant_slope[concave_mask]
            b_lower = y_l[concave_mask] - m_lower * lb[concave_mask]
            x_tan_concave = x_tan[concave_mask]
            m_upper = secant_slope[concave_mask]
            b_upper = (1 - torch.tanh(x_tan_concave) ** 2) - m_upper * x_tan_concave

            gamma_L[concave_mask] = m_lower.to(gamma_L.dtype)
            delta_L[concave_mask] = b_lower.to(delta_L.dtype)
            gamma_U[concave_mask] = m_upper.to(gamma_U.dtype)
            delta_U[concave_mask] = b_upper.to(delta_U.dtype)

        already_handled = already_handled | concave_mask

        # Case B: Purely convex
        convex_mask = ((lb >= x_inf) | (ub <= -x_inf)) & (~already_handled)
        if convex_mask.any():
            m_upper = secant_slope[convex_mask]
            b_upper = y_l[convex_mask] - m_upper * lb[convex_mask]
            x_tan_convex = x_tan[convex_mask]
            m_lower = secant_slope[convex_mask]
            b_lower = (1 - torch.tanh(x_tan_convex) ** 2) - m_lower * x_tan_convex

            gamma_L[convex_mask] = m_lower.to(gamma_L.dtype)
            delta_L[convex_mask] = b_lower.to(delta_L.dtype)
            gamma_U[convex_mask] = m_upper.to(gamma_U.dtype)
            delta_U[convex_mask] = b_upper.to(delta_U.dtype)

        already_handled = already_handled | convex_mask

        # Case C: Mixed concavity
        mixed_mask = ~already_handled
        if mixed_mask.any():
            m = secant_slope[mixed_mask]

            candidate_points = torch.cat(
                (torch.stack((lb[mixed_mask], ub[mixed_mask]), dim=-1), tangent_points_x[mixed_mask]), dim=-1
            )  # Shape: (num_mixed, 2 + num_roots)

            valid_candidates = torch.cat(
                (
                    torch.stack((torch.ones_like(lb[mixed_mask], dtype=torch.bool), torch.ones_like(ub[mixed_mask], dtype=torch.bool)), dim=-1),
                    in_range[mixed_mask],
                ),
                dim=-1,
            )  # Shape: (num_mixed, 2 + num_roots)

            residuals = (1 - torch.tanh(candidate_points) ** 2) - m.unsqueeze(-1) * candidate_points

            # Mask out invalid candidates by setting their residuals to +inf/-inf
            min_residuals = torch.where(valid_candidates, residuals, torch.tensor(float("inf"), device=residuals.device))
            max_residuals = torch.where(valid_candidates, residuals, torch.tensor(float("-inf"), device=residuals.device))

            min_resid = min_residuals.min(dim=-1).values
            max_resid = max_residuals.max(dim=-1).values

            gamma_L[mixed_mask] = m.to(gamma_L.dtype)
            delta_L[mixed_mask] = min_resid.to(delta_L.dtype)
            gamma_U[mixed_mask] = m.to(gamma_U.dtype)
            delta_U[mixed_mask] = max_resid.to(delta_U.dtype)

        return gamma_L, delta_L, gamma_U, delta_U

    # --- Utility Functions ---
    def apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the tanh activation function."""
        return torch.tanh(y)

    def apply_activation_derivative(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the tanh derivative: sech²(y) = 1 - tanh²(y)."""
        tanh_y = torch.tanh(y)
        return 1 - tanh_y * tanh_y

    def _solve_cubic_for_tangent(self, m: torch.Tensor) -> List[torch.Tensor]:
        """
        Solves a batch of depressed cubic equations 2t³ - 2t - m = 0 for t, where t = tanh(x).

        This implementation uses the analytical method (Cardano's method) to find the
        roots in a vectorized manner, allowing it to run entirely within PyTorch on a specified device (CPU or GPU).
        It only returns real roots that fall within the valid range for tanh, i.e., (-1, 1).

        Args:
            m (torch.Tensor): A 1D tensor of constant terms from the equations.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - roots (torch.Tensor): A tensor of shape (m.size(0), 3) with all real roots.
            - valid_roots (torch.Tensor): A boolean mask of the same shape indicating
                                        which roots are in the range (-1, 1).
        """
        # Convert to the standard depressed cubic form: t³ + pt + q = 0
        # by dividing by 2: t³ - t - m/2 = 0
        p = -1.0
        q = -m / 2.0

        # Calculate the discriminant of the cubic equation: Δ = (q/2)² + (p/3)³
        delta = (q / 2.0) ** 2 + (p / 3.0) ** 3

        # Store up to 3 roots per equation
        # Pick a value outside the valid tanh range as default
        if m.ndim == 1:
            roots = torch.full((m.size(0), 3), 2.0, device=m.device, dtype=m.dtype)
        elif m.ndim == 2:
            roots = torch.full((m.size(0), m.size(1), 3), 2.0, device=m.device, dtype=m.dtype)
        else:
            raise ValueError("Input tensor m must be 1D or 2D.")

        # Get machine epsilon for the tensor's dtype for robust floating-point comparisons
        epsilon = torch.finfo(m.dtype).eps

        # Case 1: Three distinct real roots
        # Use Viète's trigonometric solution.
        # Argument for acos must be clamped to [-1, 1] for numerical stability
        negative_delta_mask = delta < -epsilon

        # [ORIGINAL] sqrt_factor = np.sqrt(-p / 3.0)
        # np.sqrt returns float64 which causes dtype mismatch with float32 tensors
        # sqrt_factor = torch.sqrt(torch.tensor(-p / 3.0, dtype=m.dtype))
        # [ORIGINAL] sqrt_factor = np.sqrt(-p / 3.0)
        # np.sqrt returns float64 which causes dtype mismatch with float32 tensors
        # sqrt_factor = torch.sqrt(torch.tensor(-p / 3.0, dtype=m.dtype))
        sqrt_factor = np.sqrt(-p / 3.0)
        acos_arg = (3.0 * q[negative_delta_mask]) / (2.0 * p * sqrt_factor)
        acos_arg_clamped = torch.clamp(acos_arg, -1.0, 1.0)

        angle = torch.acos(acos_arg_clamped) / 3.0

        scale = 2.0 * sqrt_factor

        root1 = scale * torch.cos(angle)
        root2 = scale * torch.cos(angle - 2.0 * torch.pi / 3.0)
        root3 = scale * torch.cos(angle + 2.0 * torch.pi / 3.0)

        if m.ndim == 1:
            roots[negative_delta_mask, 0] = root1
            roots[negative_delta_mask, 1] = root2
            roots[negative_delta_mask, 2] = root3
        elif m.ndim == 2:
            i, j = negative_delta_mask.nonzero(as_tuple=True)
            roots[i, j, 0] = root1
            roots[i, j, 1] = root2
            roots[i, j, 2] = root3

        # Case 2: One real root (and two complex conjugate roots if delta > 0)
        # Or three real roots with at least two equal if delta = 0.
        else_mask = delta >= -epsilon

        # Clamp to zero to handle delta being slightly negative due to precision issues.
        sqrt_delta = torch.sqrt(torch.clamp(delta[else_mask], min=0.0))

        u_cubed = -q[else_mask] / 2.0 + sqrt_delta
        v_cubed = -q[else_mask] / 2.0 - sqrt_delta

        # Compute cube root handling the sign correctly for real cube roots
        # torch.cbrt doesn't exist, so use sign * abs^(1/3)
        u = torch.sign(u_cubed) * torch.pow(torch.abs(u_cubed), 1.0 / 3.0)
        v = torch.sign(v_cubed) * torch.pow(torch.abs(v_cubed), 1.0 / 3.0)

        root1 = u + v
        if m.ndim == 1:
            roots[else_mask, 0] = root1
        elif m.ndim == 2:
            i, j = else_mask.nonzero(as_tuple=True)
            roots[i, j, 0] = root1

        # Case 2b: Three real roots with at least two equal if delta = 0
        delta_zero_mask = else_mask & (torch.abs(delta) < epsilon)

        # Only compute root23 for the subset where delta_zero_mask is True
        # Need to extract the corresponding u values from the else_mask subset
        if delta_zero_mask.any():
            root23 = -u[delta_zero_mask] / 2.0
            if m.ndim == 1:
                roots[delta_zero_mask, 1] = root23
                roots[delta_zero_mask, 2] = root23
            elif m.ndim == 2:
                i, j = delta_zero_mask.nonzero(as_tuple=True)
                roots[i, j, 1] = root23
                roots[i, j, 2] = root23

        # Filter for roots within the valid tanh range (-1, 1)
        valid_roots = (roots > -1.0) & (roots < 1.0)

        return roots, valid_roots
