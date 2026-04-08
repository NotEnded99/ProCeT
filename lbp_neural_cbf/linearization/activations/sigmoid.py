"""
Sigmoid activation relaxation for CROWN-based verification.

This module implements linear relaxations for the sigmoid activation function
and its derivative using rigorous analytical formulas.
"""

from typing import Tuple

import torch

from .activation_relaxations import ActivationRelaxation


class SigmoidActivationRelaxation(ActivationRelaxation):
    """
    Sigmoid activation relaxation implementation using analytical formulas.

    Sigmoid is defined as σ(x) = 1/(1 + exp(-x)).
    It's S-shaped with an inflection point at x=0, convex for x < 0, concave for x > 0.

    The derivative σ'(x) is a bell-shaped curve with inflection points at x ≈ ±1.317.
    It's concave between these points and convex elsewhere.
    """

    def __init__(self):
        """
        Initialize the Sigmoid relaxation.
        """
        # x_inf_deriv is the inflection point for the sigmoid derivative (float64 for precision)
        sqrt_3 = torch.sqrt(torch.tensor(3.0, dtype=torch.float64))
        self.x_inf_deriv = torch.log((3 + sqrt_3) / (3 - sqrt_3))

    def relax_activation(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes linear relaxation for sigmoid(y) = 1/(1 + exp(-y)).

        Args:
            lb: Lower bounds of pre-activation
            ub: Upper bounds of pre-activation

        Returns:
            Tuple of (alpha_L, beta_L, alpha_U, beta_U) for linear bounds
        """
        # Use the vectorized _compute_sigmoid_bounds
        alpha_L, beta_L, alpha_U, beta_U = self._compute_sigmoid_bounds(lb, ub)
        return alpha_L, beta_L, alpha_U, beta_U

    def relax_activation_derivative(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes linear relaxation for σ'(y) = σ(y) * (1 - σ(y)) in a fully vectorized fashion.

        Args:
            lb: Lower bounds of pre-activation
            ub: Upper bounds of pre-activation

        Returns:
            Tuple of (gamma_L, delta_L, gamma_U, delta_U) for derivative bounds
        """
        if not isinstance(lb, torch.Tensor):
            lb = torch.tensor(lb, device=lb.device, dtype=lb.dtype)
        if not isinstance(ub, torch.Tensor):
            ub = torch.tensor(ub, device=lb.device, dtype=lb.dtype)

        original_shape = lb.shape
        is_scalar = lb.dim() == 0

        gamma_L = torch.zeros_like(lb)
        delta_L = torch.zeros_like(lb)
        gamma_U = torch.zeros_like(lb)
        delta_U = torch.zeros_like(lb)

        # Handle point intervals directly
        isclose_mask = torch.isclose(lb, ub, atol=1e-12)
        if isclose_mask.any():
            lb_point = lb[isclose_mask].to(torch.float64)
            sigmoid_val = torch.sigmoid(lb_point)
            y_point = sigmoid_val * (1 - sigmoid_val)
            second_deriv = y_point * (1 - 2 * sigmoid_val)
            intercept = y_point - second_deriv * lb_point

            gamma_L[isclose_mask] = second_deriv.to(gamma_L.dtype)
            delta_L[isclose_mask] = intercept.to(delta_L.dtype)
            gamma_U[isclose_mask] = second_deriv.to(gamma_U.dtype)
            delta_U[isclose_mask] = intercept.to(delta_U.dtype)

        if torch.all(isclose_mask):
            if is_scalar:
                return gamma_L.item(), delta_L.item(), gamma_U.item(), delta_U.item()
            return (gamma_L.reshape(original_shape), delta_L.reshape(original_shape), gamma_U.reshape(original_shape), delta_U.reshape(original_shape))

        # Cast to float64 for numerically stable calculations
        lb64 = lb.to(torch.float64)
        ub64 = ub.to(torch.float64)

        y_l = self.apply_activation_derivative(lb64)
        y_u = self.apply_activation_derivative(ub64)

        secant_slope = torch.zeros_like(lb64)
        non_point_mask = ~isclose_mask
        if non_point_mask.any():
            denom = torch.clamp(ub64 - lb64, min=1e-12)
            secant_values = (y_u - y_l) / denom
            secant_slope = torch.where(non_point_mask, secant_values, secant_slope)
        # Clamp to physical Lipschitz bound: |d/dy[σ'(y)]| ≤ 1/6 for sigmoid'
        # This prevents NaN from catastrophic cancellation in sqrt discriminant
        secant_slope = torch.clamp(secant_slope, min=-1.0/6.0, max=1.0/6.0)

        x_inf = torch.as_tensor(self.x_inf_deriv, dtype=lb64.dtype, device=lb64.device)

        concave_mask = (lb64 >= -x_inf) & (ub64 <= x_inf) & non_point_mask
        convex_mask = ((ub64 <= -x_inf) | (lb64 >= x_inf)) & non_point_mask & ~concave_mask
        mixed_mask = non_point_mask & ~(concave_mask | convex_mask)

        # Pre-compute tangent roots and logits for every element
        tangent_roots, valid_tangent_roots = self._solve_cubic_for_tangent(secant_slope)
        logit_tangent_roots = torch.zeros_like(tangent_roots)
        if valid_tangent_roots.any():
            safe_roots = torch.clamp(tangent_roots[valid_tangent_roots], min=1e-12, max=1 - 1e-12)
            logit_tangent_roots[valid_tangent_roots] = torch.log(safe_roots) - torch.log1p(-safe_roots)

        num_roots = tangent_roots.shape[-1]
        root_indices = torch.arange(num_roots, device=lb.device)

        # Concave region: secant lower, tangent upper
        if concave_mask.any():
            m_lower = secant_slope[concave_mask]
            lb_c = lb64[concave_mask]
            ub_c = ub64[concave_mask]
            y_l_c = y_l[concave_mask]
            y_u_c = y_u[concave_mask]

            b_lower = y_l_c - m_lower * lb_c

            valid_roots_c = valid_tangent_roots[concave_mask]
            logit_roots_c = logit_tangent_roots[concave_mask]

            in_bounds = valid_roots_c & (logit_roots_c > lb_c.unsqueeze(-1)) & (logit_roots_c < ub_c.unsqueeze(-1))
            first_in_range = torch.where(in_bounds, root_indices, num_roots).min(dim=-1).values
            has_tangent = first_in_range != num_roots
            safe_indices = torch.where(has_tangent, first_in_range, torch.zeros_like(first_in_range))
            tangent_x = torch.gather(logit_roots_c, -1, safe_indices.unsqueeze(-1)).squeeze(-1)

            tangent_y = self.apply_activation_derivative(tangent_x)
            b_upper = torch.where(
                has_tangent,
                tangent_y - m_lower * tangent_x,
                y_u_c - m_lower * ub_c,
            )

            gamma_L[concave_mask] = m_lower.to(gamma_L.dtype)
            delta_L[concave_mask] = b_lower.to(delta_L.dtype)
            gamma_U[concave_mask] = m_lower.to(gamma_U.dtype)
            delta_U[concave_mask] = b_upper.to(delta_U.dtype)

        # Convex region: tangent lower, secant upper
        if convex_mask.any():
            m_upper = secant_slope[convex_mask]
            lb_v = lb64[convex_mask]
            ub_v = ub64[convex_mask]
            y_l_v = y_l[convex_mask]
            y_u_v = y_u[convex_mask]

            b_upper = y_u_v - m_upper * ub_v

            valid_roots_v = valid_tangent_roots[convex_mask]
            logit_roots_v = logit_tangent_roots[convex_mask]

            in_bounds = valid_roots_v & (logit_roots_v > lb_v.unsqueeze(-1)) & (logit_roots_v < ub_v.unsqueeze(-1))
            first_in_range = torch.where(in_bounds, root_indices, num_roots).min(dim=-1).values
            has_tangent = first_in_range != num_roots
            safe_indices = torch.where(has_tangent, first_in_range, torch.zeros_like(first_in_range))
            tangent_x = torch.gather(logit_roots_v, -1, safe_indices.unsqueeze(-1)).squeeze(-1)

            tangent_y = self.apply_activation_derivative(tangent_x)
            b_lower = torch.where(
                has_tangent,
                tangent_y - m_upper * tangent_x,
                y_l_v - m_upper * lb_v,
            )

            gamma_L[convex_mask] = m_upper.to(gamma_L.dtype)
            delta_L[convex_mask] = b_lower.to(delta_L.dtype)
            gamma_U[convex_mask] = m_upper.to(gamma_U.dtype)
            delta_U[convex_mask] = b_upper.to(delta_U.dtype)

        # Mixed region: slope = secant, intercepts from residual extrema
        if mixed_mask.any():
            m_m = secant_slope[mixed_mask]
            lb_m = lb64[mixed_mask]
            ub_m = ub64[mixed_mask]

            valid_roots_m = valid_tangent_roots[mixed_mask]
            logit_roots_m = logit_tangent_roots[mixed_mask]

            in_bounds = valid_roots_m & (logit_roots_m > lb_m.unsqueeze(-1)) & (logit_roots_m < ub_m.unsqueeze(-1))
            x_candidates = torch.where(in_bounds, logit_roots_m, torch.zeros_like(logit_roots_m))

            endpoint_stack = torch.stack((lb_m, ub_m), dim=-1)
            candidate_points = torch.cat((endpoint_stack, x_candidates), dim=-1)

            endpoints_mask = torch.ones(endpoint_stack.shape, dtype=torch.bool, device=lb.device)
            valid_candidates = torch.cat((endpoints_mask, in_bounds), dim=-1)

            residuals = self.apply_activation_derivative(candidate_points) - m_m.unsqueeze(-1) * candidate_points

            pos_inf = torch.tensor(float("inf"), dtype=residuals.dtype, device=residuals.device)
            neg_inf = torch.tensor(float("-inf"), dtype=residuals.dtype, device=residuals.device)

            min_residuals = torch.where(valid_candidates, residuals, pos_inf).min(dim=-1).values
            max_residuals = torch.where(valid_candidates, residuals, neg_inf).max(dim=-1).values

            gamma_L[mixed_mask] = m_m.to(gamma_L.dtype)
            delta_L[mixed_mask] = min_residuals.to(delta_L.dtype)
            gamma_U[mixed_mask] = m_m.to(gamma_U.dtype)
            delta_U[mixed_mask] = max_residuals.to(delta_U.dtype)

        gamma_L = gamma_L.reshape(original_shape)
        delta_L = delta_L.reshape(original_shape)
        gamma_U = gamma_U.reshape(original_shape)
        delta_U = delta_U.reshape(original_shape)

        if is_scalar:
            return gamma_L.item(), delta_L.item(), gamma_U.item(), delta_U.item()
        return gamma_L, delta_L, gamma_U, delta_U

    def _compute_sigmoid_bounds(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mathematically rigorous optimal linear bounds for sigmoid over interval [lb, ub].
        Now supports both scalar and vectorized inputs using torch.where approach.
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

        # Compute sigmoid values
        y_l = torch.sigmoid(lb_flat)
        y_u = torch.sigmoid(ub_flat)

        # Initialize output tensors
        alpha_L = torch.zeros_like(lb_flat)
        beta_L = torch.zeros_like(lb_flat)
        alpha_U = torch.zeros_like(lb_flat)
        beta_U = torch.zeros_like(lb_flat)

        # Handle zero-width intervals (point evaluations)
        zero_width = torch.abs(ub_flat - lb_flat) < 1e-12
        if zero_width.any():
            slope_point = self.apply_activation_derivative(lb_flat[zero_width])
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
        # IMPORTANT: Clamp secant_slope to [0, 0.25] to prevent NaN from catastrophic
        # cancellation when lb ≈ ub (degenerate simplices). When x1 ≈ x2,
        # sigmoid(x2) - sigmoid(x1) suffers catastrophic cancellation, making
        # secant_slope appear >> 0.25, which makes discriminant = 1 - 4*m < 0
        # and sqrt(negative) = NaN. The true derivative of sigmoid is always in [0, 0.25].
        secant_slope_raw = (y_u - y_l) / torch.clamp(ub_flat - lb_flat, min=1e-12)
        secant_slope = torch.where(non_zero, secant_slope_raw, torch.zeros_like(lb_flat))
        # Clamp to physical bounds of sigmoid derivative [0, 0.25]
        secant_slope = torch.clamp(secant_slope, min=0.0, max=0.25)

        # === Define all masks upfront ===
        concave = non_zero & (lb_flat >= 0)  # Case A: Concave region (0 ≤ lb < ub)
        convex = non_zero & (ub_flat <= 0)  # Case B: Convex region (lb < ub ≤ 0)
        mixed = non_zero & (lb_flat < 0) & (ub_flat > 0)  # Case C: Mixed region

        # === Pre-compute values for all cases ===
        # Concave region values
        sigmoid_deriv_lb = self.apply_activation_derivative(lb_flat)
        alpha_U_concave = sigmoid_deriv_lb
        beta_U_concave = y_l - sigmoid_deriv_lb * lb_flat
        alpha_L_concave = secant_slope
        beta_L_concave = y_l - secant_slope * lb_flat

        # Convex region values
        sigmoid_deriv_ub = self.apply_activation_derivative(ub_flat)
        alpha_U_convex = secant_slope
        beta_U_convex = y_l - secant_slope * lb_flat
        alpha_L_convex = sigmoid_deriv_ub
        beta_L_convex = y_u - sigmoid_deriv_ub * ub_flat

        # Mixed region values - compute tangent points
        # Discriminant for quadratic y^2 - y + m = 0
        discriminant = 1 - 4 * secant_slope
        valid_discriminant = discriminant > 0

        # When discriminant <= 0, tangent points are at inflection (x=0)
        d = torch.sqrt(torch.clamp(discriminant, min=0))
        y_lambda = (1 - d) / 2  # lower tangent (convex part)
        y_mu = (1 + d) / 2  # upper tangent (concave part)

        # Compute x coordinates (avoid log(0) by clamping)
        lambda_x = -torch.log(torch.clamp(1 / y_lambda - 1, min=1e-12))
        mu_x = -lambda_x

        # Lower bound for mixed region
        use_tangent_lower = mixed & valid_discriminant & (lambda_x > lb_flat)
        alpha_L_mixed_tangent = secant_slope
        beta_L_mixed_tangent = y_lambda - secant_slope * lambda_x
        alpha_L_mixed_secant = secant_slope
        beta_L_mixed_secant = y_l - secant_slope * lb_flat

        # Upper bound for mixed region
        use_tangent_upper = mixed & valid_discriminant & (mu_x < ub_flat)
        alpha_U_mixed_tangent = secant_slope
        beta_U_mixed_tangent = y_mu - secant_slope * mu_x
        alpha_U_mixed_secant = secant_slope
        beta_U_mixed_secant = y_u - secant_slope * ub_flat

        # === Apply bounds using torch.where (no branching on .any()) ===
        # Concave region
        alpha_L = torch.where(concave, alpha_L_concave, alpha_L)
        beta_L = torch.where(concave, beta_L_concave, beta_L)
        alpha_U = torch.where(concave, alpha_U_concave, alpha_U)
        beta_U = torch.where(concave, beta_U_concave, beta_U)

        # Convex region
        alpha_L = torch.where(convex, alpha_L_convex, alpha_L)
        beta_L = torch.where(convex, beta_L_convex, beta_L)
        alpha_U = torch.where(convex, alpha_U_convex, alpha_U)
        beta_U = torch.where(convex, beta_U_convex, beta_U)

        # Mixed region - lower bound
        alpha_L = torch.where(use_tangent_lower, alpha_L_mixed_tangent, alpha_L)
        beta_L = torch.where(use_tangent_lower, beta_L_mixed_tangent, beta_L)
        # Use secant for lower bound when tangent not applicable
        use_secant_lower = mixed & ~use_tangent_lower
        alpha_L = torch.where(use_secant_lower, alpha_L_mixed_secant, alpha_L)
        beta_L = torch.where(use_secant_lower, beta_L_mixed_secant, beta_L)

        # Mixed region - upper bound
        alpha_U = torch.where(use_tangent_upper, alpha_U_mixed_tangent, alpha_U)
        beta_U = torch.where(use_tangent_upper, beta_U_mixed_tangent, beta_U)
        # Use secant for upper bound when tangent not applicable
        use_secant_upper = mixed & ~use_tangent_upper
        alpha_U = torch.where(use_secant_upper, alpha_U_mixed_secant, alpha_U)
        beta_U = torch.where(use_secant_upper, beta_U_mixed_secant, beta_U)

        # Reshape to original shape
        alpha_L = alpha_L.reshape(original_shape)
        beta_L = beta_L.reshape(original_shape)
        alpha_U = alpha_U.reshape(original_shape)
        beta_U = beta_U.reshape(original_shape)

        # Return scalars if input was scalar
        if is_scalar:
            return alpha_L.item(), beta_L.item(), alpha_U.item(), beta_U.item()

        return alpha_L, beta_L, alpha_U, beta_U

    # --- Utility Functions ---
    def apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the sigmoid activation function."""
        return torch.sigmoid(y)

    def apply_activation_derivative(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the sigmoid derivative: σ'(y) = σ(y) * (1 - σ(y))."""
        sigmoid_y = torch.sigmoid(y)
        return sigmoid_y * (1 - sigmoid_y)

    def _solve_cubic_for_tangent(self, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve 2y³ - 3y² + y - m = 0 for y = σ(x) in a vectorized manner.

        Args:
            m: Tensor of secant slopes.

        Returns:
            Tuple (roots, valid_mask) where each has shape m.shape + (3,).
        """
        m64 = m.to(torch.float64)
        original_shape = m64.shape
        m_flat = m64.reshape(-1)
        num_items = m_flat.numel()

        roots = torch.full((num_items, 3), 2.0, dtype=m64.dtype, device=m64.device)
        valid_mask = torch.zeros((num_items, 3), dtype=torch.bool, device=m64.device)

        p = torch.full_like(m_flat, -0.25)
        q = -m_flat / 2.0

        delta = (q / 2.0) ** 2 + (p / 3.0) ** 3
        eps = torch.finfo(m_flat.dtype).eps * 10

        negative_delta = delta < -eps
        non_negative_delta = ~negative_delta

        if negative_delta.any():
            p_neg = p[negative_delta]
            q_neg = q[negative_delta]

            sqrt_factor = torch.sqrt(-p_neg / 3.0)
            scale = 2.0 * sqrt_factor

            acos_arg = torch.clamp((3.0 * q_neg) / (2.0 * p_neg * sqrt_factor), -1.0, 1.0)
            angle = torch.acos(acos_arg) / 3.0

            t1 = scale * torch.cos(angle)
            t2 = scale * torch.cos(angle - 2.0 * torch.pi / 3.0)
            t3 = scale * torch.cos(angle + 2.0 * torch.pi / 3.0)

            shift = 0.5
            y1 = t1 + shift
            y2 = t2 + shift
            y3 = t3 + shift

            rows = torch.nonzero(negative_delta, as_tuple=False).squeeze(-1)
            roots[rows, 0] = y1
            roots[rows, 1] = y2
            roots[rows, 2] = y3

            valid_mask[rows, 0] = (y1 > 0) & (y1 < 1)
            valid_mask[rows, 1] = (y2 > 0) & (y2 < 1)
            valid_mask[rows, 2] = (y3 > 0) & (y3 < 1)

        if non_negative_delta.any():
            q_pos = q[non_negative_delta]
            delta_pos = torch.clamp(delta[non_negative_delta], min=0.0)

            sqrt_delta = torch.sqrt(delta_pos)
            u_cubed = -q_pos / 2.0 + sqrt_delta
            v_cubed = -q_pos / 2.0 - sqrt_delta

            u = torch.sign(u_cubed) * torch.pow(torch.abs(u_cubed), 1.0 / 3.0)
            v = torch.sign(v_cubed) * torch.pow(torch.abs(v_cubed), 1.0 / 3.0)

            t = u + v
            shift = 0.5
            y = t + shift

            rows = torch.nonzero(non_negative_delta, as_tuple=False).squeeze(-1)
            roots[rows, 0] = y
            valid_mask[rows, 0] = (y > 0) & (y < 1)

        roots = roots.reshape(*original_shape, 3)
        valid_mask = valid_mask.reshape(*original_shape, 3)
        return roots, valid_mask
