import numpy as np

from ..translators import NumpyTranslator


class CertifiedFirstOrderTaylorExpansion:
    """
    Represents a first-order Taylor expansion with certified error bounds.

    For a function f: R^n -> R^m, the Taylor expansion around point c is:
    f(x) ≈ f(c) + ∇f(c)(x - c) + R(x)

    where R(x) is the remainder term bounded by interval arithmetic.

    Attributes:
        expansion_point (np.ndarray): Point c around which the expansion is computed
        domain (tuple): (lower, upper) bounds defining the domain of validity
        linear_approximation (tuple): (Jacobian, constant) where:
            - Jacobian: ∇f(c) matrix of partial derivatives
            - constant: f(c) function value at expansion point
        remainder (tuple): (lower, upper) certified bounds on the remainder term
        simplex_vertices (np.ndarray): Vertices defining the simplex domain (if applicable)
        is_simplex (bool): Whether this Taylor expansion is over a simplicial domain
    """

    def __init__(self, expansion_point, domain, linear_approximation=None, remainder=None, numeric_translator=None):
        """
        Initialize a certified first-order Taylor expansion.

        Args:
            expansion_point (np.ndarray): Center point of the expansion
            domain (tuple): (lower_bounds, upper_bounds) defining valid input region
            linear_approximation (tuple): (Jacobian_matrix, function_value_at_center)
            remainder (tuple): (remainder_lower_bounds, remainder_upper_bounds)
        """
        self.expansion_point = expansion_point
        self.domain = domain
        # Initialize simplex-related attributes
        self.simplex_vertices = None
        self.is_simplex = False

        if linear_approximation is None:
            # Default f(x) = x
            self.linear_approximation = (np.eye(expansion_point.size), expansion_point)
            self.remainder = (np.zeros_like(expansion_point), np.zeros_like(expansion_point))
        else:
            self.linear_approximation = linear_approximation
            self.remainder = remainder

        if numeric_translator is None:
            numeric_translator = NumpyTranslator()
        self.translator = numeric_translator

    def _are_domains_compatible(self, other):
        """Check if two domains are compatible for operations (allowing mixed types)."""
        if not isinstance(other, CertifiedFirstOrderTaylorExpansion):
            return False

        # For mixed operations, we allow different domain types
        # The operation will not preserve simplex properties
        if self.is_simplex != other.is_simplex:
            return True  # Mixed operations are allowed

        return self._domains_equal(other)  # Same type operations require exact equality

    def _domains_equal(self, other):
        """Helper method to check if two domains are equal."""
        if not isinstance(other, CertifiedFirstOrderTaylorExpansion):
            return False

        # If one is simplex and other is not, they're not equal
        if self.is_simplex != other.is_simplex:
            return False

        if self.is_simplex:
            # Both are simplexes - compare vertices
            if self.simplex_vertices is None or other.simplex_vertices is None:
                return self.simplex_vertices is other.simplex_vertices
            return self.translator.allclose(self.simplex_vertices, other.simplex_vertices)
        else:
            # Both are hyperrectangles - compare bounds
            if isinstance(self.domain, tuple) and isinstance(other.domain, tuple):
                return self.translator.allclose(self.domain[0], other.domain[0]) and self.translator.allclose(self.domain[1], other.domain[1])
            else:
                # Fallback to direct comparison for other cases
                try:
                    if isinstance(self.domain, tuple) and hasattr(self.domain[0], "__len__"):
                        return self.translator.allclose(self.domain[0], other.domain[0]) and self.translator.allclose(self.domain[1], other.domain[1])
                    else:
                        return self.domain == other.domain
                except (TypeError, ValueError):
                    return False

    def _propagate_simplex_info(self, result, other=None):
        """Helper method to propagate simplex information to result."""
        if other is None:
            # Unary operation
            result.simplex_vertices = self.simplex_vertices
            result.is_simplex = self.is_simplex
        elif isinstance(other, CertifiedFirstOrderTaylorExpansion):
            # Binary operation with another Taylor expansion
            if self.is_simplex and other.is_simplex:
                # Both are simplices - they should have the same vertices
                if (
                    self.simplex_vertices is not None
                    and other.simplex_vertices is not None
                    and self.translator.allclose(self.simplex_vertices, other.simplex_vertices)
                ):
                    result.simplex_vertices = self.simplex_vertices
                    result.is_simplex = True
                else:
                    # Different simplex domains - lose simplex properties
                    result.simplex_vertices = None
                    result.is_simplex = False
            elif self.is_simplex and not other.is_simplex:
                # Mixed operation: simplex + hyperrectangle - lose simplex properties
                result.simplex_vertices = None
                result.is_simplex = False
            elif not self.is_simplex and other.is_simplex:
                # Mixed operation: hyperrectangle + simplex - lose simplex properties
                result.simplex_vertices = None
                result.is_simplex = False
            else:
                # Both are hyperrectangles - no simplex properties
                result.simplex_vertices = None
                result.is_simplex = False
        else:
            # Binary operation with scalar
            result.simplex_vertices = self.simplex_vertices
            result.is_simplex = self.is_simplex

    def _compute_function_composition_remainder(self, second_derivative_bounds):
        """
        Compute tight Lagrange remainder bounds for element-wise f(g(x)) using
        a simplex-aware Bernstein pipeline for the quadratic term (g(x)-g(c))^2,
        with a safe rectangular fallback.

        For each output component, the second-order Lagrange remainder is:
            R_loc(x) = 0.5 * f''(η(x)) * (g(x) - g(c))^2
        We bound K(x) = (g(x) - g(c))^2 tightly:
          - On a simplex: bound (J·dx)^2 via exact degree-2 Bernstein coeffs of the
            product of two linear forms; handle cross terms 2R·(J·dx) using exact
            linear ranges; and R^2 via interval arithmetic.
          - On a rectangle: fall back to interval bounds on S(x)=L(x)+R and then S^2.

        Args:
            second_derivative_bounds: (M_min, M_max) bounds on f''(y) over the range of g(x).
                Each is an array-like of shape (m,), where m is the output dimension.

        Returns:
            (remainder_lower, remainder_upper): arrays of shape (m,).
        """
        M_min_f_double_prime, M_max_f_double_prime = second_derivative_bounds

        # Unpack g's linearization and remainder
        J_g_c = self.linear_approximation[0]  # shape (m, n)
        R_g_lower, R_g_upper = self.remainder  # shape (m,)

        scalar_output = J_g_c.ndim == self.expansion_point.ndim
        if scalar_output:
            # Scalar output: add singleton dimension for correct matmul
            J_g_c = self.translator.unsqueeze(J_g_c, dim=-2)

        # Compute bounds for K(x) = (g(x) - g(c))^2 per output component
        if self.is_simplex and (self.simplex_vertices is not None):
            # Step 1: Transform to barycentric coordinates implicitly via vertex evaluation
            # Displacements from center to vertices
            dx_vertices = self.simplex_vertices - self.translator.unsqueeze(self.expansion_point, dim=-2)  # [V, n]

            # Linear form values at vertices: L(v) = J_g_c · (v - c)
            L_at_vertices = J_g_c @ self.translator.transpose(dx_vertices)  # [m, V]

            # Step 3: Bernstein coefficients for quadratic product of linears
            # Bound (J·dx)^2 over simplex using exact degree-2 Bernstein coefficients
            L2_low, L2_high = _bernstein_bounds_product_of_linears_over_simplex(L_at_vertices, L_at_vertices, self.translator)  # shape (m,)

            # Exact linear ranges for L over simplex (achieved at vertices)
            L_min = self.translator.min(L_at_vertices, dim=-1)
            L_max = self.translator.max(L_at_vertices, dim=-1)

            if scalar_output:
                # Remove added singleton dimension
                L2_low = self.translator.squeeze(L2_low, dim=-1)
                L2_high = self.translator.squeeze(L2_high, dim=-1)
                L_min = self.translator.squeeze(L_min, dim=-1)
                L_max = self.translator.squeeze(L_max, dim=-1)

            # Cross term: 2 * R * L, R in [R_low, R_high], L in [L_min, L_max]
            cross_candidates = self.translator.stack(
                [
                    2.0 * R_g_lower * L_min,
                    2.0 * R_g_lower * L_max,
                    2.0 * R_g_upper * L_min,
                    2.0 * R_g_upper * L_max,
                ],
                dim=-1,
            )
            cross_low = self.translator.min(cross_candidates, dim=-1)
            cross_high = self.translator.max(cross_candidates, dim=-1)

            # Constant term: R^2 for R in [R_low, R_high]
            zero_in_R = (R_g_lower <= 0.0) & (R_g_lower >= 0.0)
            R2_low = self.translator.where(zero_in_R, 0.0, self.translator.minimum(R_g_lower**2, R_g_lower**2))
            R2_high = self.translator.maximum(R_g_lower**2, R_g_lower**2)

            # Decomposition bound
            K_lower = L2_low + cross_low + R2_low
            K_upper = L2_high + cross_high + R2_high
        else:
            # Fallback on rectangles: interval bounds on S = L + R, then S^2
            dx_low, dx_high = self._get_dx_bounds()
            # Interval bounds for L = J_g_c * dx
            L_low, L_high = _mat_interval_vec_mul(J_g_c, dx_low, dx_high, self.translator)

            if scalar_output:
                # Remove added singleton dimension
                L_low = self.translator.squeeze(L_low, dim=-1)
                L_high = self.translator.squeeze(L_high, dim=-1)

            # Interval for S = L + R
            S_min = L_low + R_g_lower
            S_max = L_high + R_g_upper
            # Interval for S^2
            K_lower = self.translator.where(S_min * S_max <= 0.0, 0.0, self.translator.minimum(S_min**2, S_max**2))
            K_upper = self.translator.maximum(S_min**2, S_max**2)

        # Step 4: Bound the local error using f'' bounds and K bounds

        # Interval product [M_min, M_max] * [K_lower, K_upper]
        prod_terms = self.translator.stack(
            [
                M_min_f_double_prime * K_lower,
                M_min_f_double_prime * K_upper,
                M_max_f_double_prime * K_lower,
                M_max_f_double_prime * K_upper,
            ],
            dim=-1,
        )
        local_error_min = 0.5 * self.translator.min(prod_terms, dim=-1)
        local_error_max = 0.5 * self.translator.max(prod_terms, dim=-1)

        return local_error_min, local_error_max

    def __add__(self, other):
        """
        Addition operation for Taylor expansions.

        For f(x) = g(x) + h(x):
        - Linear parts add: ∇f = ∇g + ∇h
        - Constants add: f(c) = g(c) + h(c)
        - Remainders add: R_f = R_g + R_h

        Args:
            other: Another CertifiedFirstOrderTaylorExpansion or scalar

        Returns:
            CertifiedFirstOrderTaylorExpansion: Result of addition
        """
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            # Ensure compatible expansion points and domains
            assert self.translator.allclose(self.expansion_point, other.expansion_point)
            assert self._are_domains_compatible(other)

            # Properly add the linear approximation tuples element-wise
            new_jacobian = self.linear_approximation[0] + other.linear_approximation[0]
            new_constant = self.linear_approximation[1] + other.linear_approximation[1]
            new_linear_approximation = (new_jacobian, new_constant)

            result = CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=new_linear_approximation,
                remainder=(self.remainder[0] + other.remainder[0], self.remainder[1] + other.remainder[1]),
                numeric_translator=self.translator,
            )

            # Propagate simplex information
            self._propagate_simplex_info(result, other)
            return result
        elif isinstance(other, (int, float)):
            # Adding a scalar only affects the constant term
            result = CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(self.linear_approximation[0], self.linear_approximation[1] + other),
                remainder=self.remainder,
                numeric_translator=self.translator,
            )
            # Propagate simplex information
            self._propagate_simplex_info(result, other)
            return result
        else:
            raise ValueError("Unsupported type for addition")

    def __radd__(self, other):
        """Right addition (scalar + TaylorExpansion)."""
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction operation for Taylor expansions.

        For f(x) = g(x) - h(x):
        - Linear parts subtract: ∇f = ∇g - ∇h
        - Constants subtract: f(c) = g(c) - h(c)
        - Remainders subtract with interval arithmetic: R_f = R_g - R_h

        Note: For interval subtraction [a,b] - [c,d] = [a-d, b-c]
        """
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            assert self.translator.allclose(self.expansion_point, other.expansion_point)
            assert self._are_domains_compatible(other)

            # Properly subtract the linear approximation tuples element-wise
            new_jacobian = self.linear_approximation[0] - other.linear_approximation[0]
            new_constant = self.linear_approximation[1] - other.linear_approximation[1]
            new_linear_approximation = (new_jacobian, new_constant)

            result = CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=new_linear_approximation,
                # Interval subtraction: [a,b] - [c,d] = [a-d, b-c]
                remainder=(self.remainder[0] - other.remainder[1], self.remainder[1] - other.remainder[0]),
                numeric_translator=self.translator,
            )
            self._propagate_simplex_info(result, other)
            return result
        elif isinstance(other, (int, float)):
            # Subtracting a scalar only affects the constant term
            result = CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(self.linear_approximation[0], self.linear_approximation[1] - other),
                remainder=self.remainder,
                numeric_translator=self.translator,
            )
            self._propagate_simplex_info(result, other)
            return result
        else:
            raise ValueError("Unsupported type for subtraction")

    def __rsub__(self, other):
        """Right subtraction (scalar - TaylorExpansion)."""
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            assert np.allclose(self.expansion_point, other.expansion_point)
            assert self._are_domains_compatible(other)

            # Properly subtract the linear approximation tuples element-wise
            new_jacobian = other.linear_approximation[0] - self.linear_approximation[0]
            new_constant = other.linear_approximation[1] - self.linear_approximation[1]
            new_linear_approximation = (new_jacobian, new_constant)

            result = CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=new_linear_approximation,
                remainder=(other.remainder[0] - self.remainder[1], other.remainder[1] - self.remainder[0]),
                numeric_translator=self.translator,
            )
            self._propagate_simplex_info(result, other)
            return result
        elif isinstance(other, (int, float)):
            result = CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(-self.linear_approximation[0], other - self.linear_approximation[1]),
                # When subtracting from scalar: c - [a,b] = [c-b, c-a]
                remainder=(-self.remainder[1], -self.remainder[0]),
                numeric_translator=self.translator,
            )
            self._propagate_simplex_info(result, other)
            return result
        else:
            raise ValueError("Unsupported type for subtraction")

    def __mul__(self, other):
        """
        Multiplication operation for Taylor expansions.

        For f(x) = g(x) * h(x), using product rule:
        f(x) = g(c)h(c) + [g(c)∇h(c) + h(c)∇g(c)](x-c) + higher_order_terms

        The remainder includes:
        1. Propagated remainders: g(c)*R_h + h(c)*R_g
        2. Higher-order terms from (∇g·dx)(∇h·dx)
        3. Cross terms: R_g * R_h
        4. Cross terms: R_g * J_h(x-c) and R_h * J_g(x-c)

        For simplicial domains, uses dependency-aware bounds to reduce overestimation.
        For rectangular domains, uses interval arithmetic over the bounding box.
        """
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            # Taylor expansion multiplication using product rule
            assert self.translator.allclose(self.expansion_point, other.expansion_point), "Expansion points must match for TE * TE"
            assert self._are_domains_compatible(other), "Domains must be compatible for TE * TE"

            # Extract components for self (g)
            jacobian_self, const_self = self.linear_approximation
            remainder_self_low, remainder_self_high = self.remainder

            # Extract components for other (h)
            jacobian_other, const_other = other.linear_approximation
            remainder_other_low, remainder_other_high = other.remainder

            # Product rule for new constant: f(c) = g(c) * h(c)
            new_const = const_self * const_other

            # Element-wise case: each output component is independent
            new_jacobian = self.translator.unsqueeze(const_self, dim=-1) * jacobian_other + self.translator.unsqueeze(const_other, dim=-1) * jacobian_self

            # Remainder computation - use safe interval arithmetic approach for all domain types
            # This ensures correctness for non-linear remainder terms that may have extrema
            # in the interior of the domain, not just at vertices
            final_remainder_low, final_remainder_high = self._compute_multiplication_remainder(
                other,
                jacobian_self,
                jacobian_other,
                const_self,
                const_other,
                remainder_self_low,
                remainder_self_high,
                remainder_other_low,
                remainder_other_high,
            )

            result = CertifiedFirstOrderTaylorExpansion(
                self.expansion_point,
                self.domain,
                (new_jacobian, new_const),
                (final_remainder_low, final_remainder_high),
                self.translator,
            )
            self._propagate_simplex_info(result, other)
            return result
        elif isinstance(other, (int, float, np.number)):
            # Scalar multiplication: scales all terms
            new_df_c = self.linear_approximation[0] * other
            new_f_c = self.linear_approximation[1] * other
            # For interval [a,b] * c: if c≥0 then [ac,bc], if c<0 then [bc,ac]
            if other >= 0:
                new_remainder = (self.remainder[0] * other, self.remainder[1] * other)
            else:
                new_remainder = (self.remainder[1] * other, self.remainder[0] * other)
            result = CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(new_df_c, new_f_c),
                remainder=new_remainder,
                numeric_translator=self.translator,
            )
            self._propagate_simplex_info(result, other)
            return result
        else:
            return NotImplemented

    def _compute_multiplication_remainder(
        self,
        other,
        jacobian_self,
        jacobian_other,
        const_self,
        const_other,
        remainder_self_low,
        remainder_self_high,
        remainder_other_low,
        remainder_other_high,
    ):
        """
        Compute multiplication remainder.

        For rectangular domains, uses safe interval arithmetic (conservative).
        For simplicial domains shared by both operands, uses Bernstein-basis
        bounds over the simplex for the dependency-sensitive polynomial terms:
        - HOT: (J_g·dx) * (J_h·dx) using exact degree-2 Bernstein coefficients
        - Cross linear terms: R_g * (J_h·dx) and R_h * (J_g·dx) using exact
          linear ranges over the simplex (vertex enumeration)
        Remaining interval-only terms are handled via interval arithmetic.
        """
        # Common interval-style terms (valid for both domain types)
        # Term: const_self * Remainder_other
        const_self_times_rem_other_low = self.translator.where(const_self >= 0, const_self * remainder_other_low, const_self * remainder_other_high)
        const_self_times_rem_other_high = self.translator.where(const_self >= 0, const_self * remainder_other_high, const_self * remainder_other_low)

        # Term: const_other * Remainder_self
        const_other_times_rem_self_low = self.translator.where(const_other >= 0, const_other * remainder_self_low, const_other * remainder_self_high)
        const_other_times_rem_self_high = self.translator.where(const_other >= 0, const_other * remainder_self_high, const_other * remainder_self_low)

        # Term: Remainder_self * Remainder_other (interval product)
        rem_self_times_rem_other_products = self.translator.stack(
            [
                remainder_self_low * remainder_other_low,
                remainder_self_low * remainder_other_high,
                remainder_self_high * remainder_other_low,
                remainder_self_high * remainder_other_high,
            ],
            dim=-1,
        )
        rem_self_times_rem_other_low = self.translator.min(rem_self_times_rem_other_products, dim=-1)
        rem_self_times_rem_other_high = self.translator.max(rem_self_times_rem_other_products, dim=-1)

        scalar_output = jacobian_self.ndim == self.expansion_point.ndim
        if scalar_output:
            # Scalar output: add singleton dimension for correct matmul
            jacobian_self = self.translator.unsqueeze(jacobian_self, dim=-2)
            jacobian_other = self.translator.unsqueeze(jacobian_other, dim=-2)

        # Use simplex-aware bounds when both operands share the same simplex
        if (
            self.is_simplex
            and other.is_simplex
            and (self.simplex_vertices is not None)
            and (other.simplex_vertices is not None)
            and self.translator.allclose(self.simplex_vertices, other.simplex_vertices)
        ):
            # Displacements from expansion point to simplex vertices
            dx_vertices = self.simplex_vertices - self.translator.unsqueeze(self.expansion_point, dim=-2)  # [V, n]
            # Values of linear forms at vertices for each output component (m)
            # Shapes: [V, m]
            j_self_at_vertices = jacobian_self @ self.translator.transpose(dx_vertices)
            j_other_at_vertices = jacobian_other @ self.translator.transpose(dx_vertices)

            # Exact linear ranges over simplex (achieved at vertices)
            j_self_lin_min = self.translator.min(j_self_at_vertices, dim=-1)
            j_self_lin_max = self.translator.max(j_self_at_vertices, dim=-1)
            j_other_lin_min = self.translator.min(j_other_at_vertices, dim=-1)
            j_other_lin_max = self.translator.max(j_other_at_vertices, dim=-1)

            # HOT term: use product-of-linears Bernstein bounds over the simplex (any dim)
            hot_jdx_jdx_low, hot_jdx_jdx_high = _bernstein_bounds_product_of_linears_over_simplex(j_self_at_vertices, j_other_at_vertices, self.translator)

            if scalar_output:
                # Remove added singleton dimension
                j_self_lin_min = self.translator.squeeze(j_self_lin_min, dim=-1)
                j_self_lin_max = self.translator.squeeze(j_self_lin_max, dim=-1)
                j_other_lin_min = self.translator.squeeze(j_other_lin_min, dim=-1)
                j_other_lin_max = self.translator.squeeze(j_other_lin_max, dim=-1)
                hot_jdx_jdx_low = self.translator.squeeze(hot_jdx_jdx_low, dim=-1)
                hot_jdx_jdx_high = self.translator.squeeze(hot_jdx_jdx_high, dim=-1)

            # Cross terms using exact linear ranges
            # rem_self * (J_other · dx)
            rem_self_times_j_other_dx_products = self.translator.stack(
                [
                    remainder_self_low * j_other_lin_min,
                    remainder_self_low * j_other_lin_max,
                    remainder_self_high * j_other_lin_min,
                    remainder_self_high * j_other_lin_max,
                ],
                dim=-1,
            )
            rem_self_times_j_other_dx_low = self.translator.min(rem_self_times_j_other_dx_products, dim=-1)
            rem_self_times_j_other_dx_high = self.translator.max(rem_self_times_j_other_dx_products, dim=-1)

            # rem_other * (J_self · dx)
            rem_other_times_j_self_dx_products = self.translator.stack(
                [
                    remainder_other_low * j_self_lin_min,
                    remainder_other_low * j_self_lin_max,
                    remainder_other_high * j_self_lin_min,
                    remainder_other_high * j_self_lin_max,
                ],
                dim=-1,
            )
            rem_other_times_j_self_dx_low = self.translator.min(rem_other_times_j_self_dx_products, dim=-1)
            rem_other_times_j_self_dx_high = self.translator.max(rem_other_times_j_self_dx_products, dim=-1)
        else:
            # Fallback: rectangular interval bounds using dx bounding box
            dx_low, dx_high = self._get_dx_bounds()

            # Compute interval bounds for Jacobian_self * dx
            j_self_times_dx_low, j_self_times_dx_high = _mat_interval_vec_mul(jacobian_self, dx_low, dx_high, self.translator)

            # Compute interval bounds for Jacobian_other * dx
            j_other_times_dx_low, j_other_times_dx_high = _mat_interval_vec_mul(jacobian_other, dx_low, dx_high, self.translator)

            if scalar_output:
                # Remove added singleton dimension
                j_self_times_dx_low = self.translator.squeeze(j_self_times_dx_low, dim=-1)
                j_self_times_dx_high = self.translator.squeeze(j_self_times_dx_high, dim=-1)
                j_other_times_dx_low = self.translator.squeeze(j_other_times_dx_low, dim=-1)
                j_other_times_dx_high = self.translator.squeeze(j_other_times_dx_high, dim=-1)

            # Interval multiplication for (Jacobian_self · dx) * (Jacobian_other · dx)
            hot_jdx_jdx_products = self.translator.stack(
                [
                    j_self_times_dx_low * j_other_times_dx_low,
                    j_self_times_dx_low * j_other_times_dx_high,
                    j_self_times_dx_high * j_other_times_dx_low,
                    j_self_times_dx_high * j_other_times_dx_high,
                ],
                dim=-1,
            )
            hot_jdx_jdx_low = self.translator.min(hot_jdx_jdx_products, dim=-1)
            hot_jdx_jdx_high = self.translator.max(hot_jdx_jdx_products, dim=-1)

            # Term: Remainder_self * (Jacobian_other · dx)
            rem_self_times_j_other_dx_products = self.translator.stack(
                [
                    remainder_self_low * j_other_times_dx_low,
                    remainder_self_low * j_other_times_dx_high,
                    remainder_self_high * j_other_times_dx_low,
                    remainder_self_high * j_other_times_dx_high,
                ],
                dim=-1,
            )
            rem_self_times_j_other_dx_low = self.translator.min(rem_self_times_j_other_dx_products, dim=-1)
            rem_self_times_j_other_dx_high = self.translator.max(rem_self_times_j_other_dx_products, dim=-1)

            # Term: Remainder_other * (Jacobian_self · dx)
            rem_other_times_j_self_dx_products = self.translator.stack(
                [
                    remainder_other_low * j_self_times_dx_low,
                    remainder_other_low * j_self_times_dx_high,
                    remainder_other_high * j_self_times_dx_low,
                    remainder_other_high * j_self_times_dx_high,
                ],
                dim=-1,
            )
            rem_other_times_j_self_dx_low = self.translator.min(rem_other_times_j_self_dx_products, dim=-1)
            rem_other_times_j_self_dx_high = self.translator.max(rem_other_times_j_self_dx_products, dim=-1)
        # Combine all remainder terms
        final_remainder_low = (
            const_self_times_rem_other_low
            + const_other_times_rem_self_low
            + rem_self_times_rem_other_low
            + hot_jdx_jdx_low
            + rem_self_times_j_other_dx_low
            + rem_other_times_j_self_dx_low
        )
        final_remainder_high = (
            const_self_times_rem_other_high
            + const_other_times_rem_self_high
            + rem_self_times_rem_other_high
            + hot_jdx_jdx_high
            + rem_self_times_j_other_dx_high
            + rem_other_times_j_self_dx_high
        )

        return final_remainder_low, final_remainder_high

    def __rmul__(self, other):
        """Right multiplication (scalar * TaylorExpansion)."""
        if isinstance(other, (int, float, np.number)):
            # Multiplication is commutative for scalars
            return self.__mul__(other)
        else:
            return NotImplemented

    def _reciprocal(self) -> "CertifiedFirstOrderTaylorExpansion":
        """
        Compute the reciprocal 1/f(x) of a Taylor expansion.

        For g(y) = 1/y where y = f(x):
        g(y₀) = 1/y₀
        g'(y₀) = -1/y₀²

        The Taylor expansion becomes:
        1/f(x) ≈ 1/f(c) - (1/f(c)²)∇f(c)(x-c) + remainder

        Raises:
            ValueError: If the range of f(x) contains zero
        """
        y0 = self.linear_approximation[1]  # f(c)
        J_a = self.linear_approximation[0]  # ∇f(c)
        R_a_lower, R_a_upper = self.remainder

        # Check that expansion point is not zero
        if self.translator.any(self.translator.abs(y0) < 1e-14):
            raise ValueError("Reciprocal: expansion point cannot be zero or very close to zero")

        # Check that zero is not in the range to avoid division by zero
        a_range_lower, a_range_upper = self.range()
        if self.translator.any((a_range_lower <= 0) & (a_range_upper >= 0)):
            raise ValueError("Reciprocal of a Taylor expansion whose range contains zero is undefined")

        # g(y₀) = 1/y₀
        new_const = 1.0 / y0
        # g'(y₀) = -1/y₀²
        grad_g_y0 = -1.0 / (y0**2)

        # Chain rule: ∇(1/f) = g'(f(c))∇f(c)
        # Ensure proper broadcasting for the Jacobian calculation
        new_J = self.translator.unsqueeze(grad_g_y0, dim=-1) * J_a

        # Propagate remainder through the derivative
        prop_rem_term1 = grad_g_y0 * R_a_lower
        prop_rem_term2 = grad_g_y0 * R_a_upper
        propagated_rem_lower = self.translator.minimum(prop_rem_term1, prop_rem_term2)
        propagated_rem_upper = self.translator.maximum(prop_rem_term1, prop_rem_term2)

        # Second-order remainder: g''(y) = 2/y³, so Lagrange remainder involves max |2/η³|
        coeff_f_double_prime = 2.0
        exponent_f_double_prime = -3
        # Compute actual min and max of the second derivative 2/y^3 over the range of y
        M_min_g_double_prime = min_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (a_range_lower, a_range_upper), self.translator)
        M_max_g_double_prime = max_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (a_range_lower, a_range_upper), self.translator)

        second_derivative_bounds = (M_min_g_double_prime, M_max_g_double_prime)
        local_error_magnitude_min, local_error_magnitude_max = self._compute_function_composition_remainder(second_derivative_bounds)

        # Combine propagated remainder with local Lagrange error
        final_rem_lower = propagated_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_rem_upper + local_error_magnitude_max

        # Apply monotonic bounds tightening for reciprocal function
        # 1/x is monotonically decreasing for x > 0 and x < 0 separately
        reciprocal_at_boundaries = (1.0 / a_range_lower, 1.0 / a_range_upper)  # Note: order matches domain order

        # Create temporary Taylor expansion to use the monotonic tightening helper
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain, (new_J, new_const), (final_rem_lower, final_rem_upper), self.translator
        )

        clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, reciprocal_at_boundaries, is_increasing=False)

        # Intersect the Taylor bounds with the monotonic bounds
        final_rem_lower = self.translator.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = self.translator.minimum(final_rem_upper, clip_rem_upper)

        return self._create_result_with_simplex_info(
            source_expansion=self,
            expansion_point=self.expansion_point,
            domain=self.domain,
            linear_approximation=(new_J, new_const),
            remainder=(final_rem_lower, final_rem_upper),
        )

    def __truediv__(self, other):
        """Division: self / other = self * (1/other)."""
        if isinstance(other, (int, float, np.number)):
            if other == 0:
                raise ZeroDivisionError("Division by zero scalar.")
            return self * (1.0 / other)
        elif isinstance(other, CertifiedFirstOrderTaylorExpansion):
            return self * other._reciprocal()
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """Right division: other / self = other * (1/self)."""
        if isinstance(other, (int, float, np.number)):
            return other * self._reciprocal()
        else:
            return NotImplemented

    def __getitem__(self, key):
        """
        Index into a Taylor expansion to extract specific components.

        Args:
            key (int, slice, Tuple[Ellipsis, int/slice]): Index or slice of the components to extract

        Returns:
            CertifiedFirstOrderTaylorExpansion: Single-component or multi-component Taylor expansion
        """
        if isinstance(key, tuple) and Ellipsis in key:
            # Left-pad with full slices to not index into input dimensions
            dim_diff = self.linear_approximation[0].ndim - self.linear_approximation[1].ndim
            jacobian_key = (*key, *[slice(None) for _ in range(dim_diff)])
        else:
            jacobian_key = key

        new_df_c = self.linear_approximation[0][jacobian_key]
        new_f_c = self.linear_approximation[1][key]
        new_remainder_lower = self.remainder[0][key]
        new_remainder_upper = self.remainder[1][key]
        return self._create_result_with_simplex_info(
            source_expansion=self,
            expansion_point=self.expansion_point,
            domain=self.domain,
            linear_approximation=(new_df_c, new_f_c),
            remainder=(new_remainder_lower, new_remainder_upper),
        )

    def range(self):
        """
        Compute the range (lower and upper bounds) of the Taylor expansion over its domain.

        The range is computed as:
        f(x) = f(c) + ∇f(c)(x-c) + R(x)

        For simplicial domains, we solve linear programming problems to get tight bounds.
        For rectangular domains, we use interval arithmetic.

        Returns:
            tuple: (lower_bounds, upper_bounds) arrays of shape matching the output dimension
        """
        if self.is_simplex and self.simplex_vertices is not None:
            return self._range_simplex()
        else:
            return self._range_rectangular()

    def _range_rectangular(self):
        """Range computation for rectangular domains using interval arithmetic."""
        J = self.linear_approximation[0]  # Jacobian ∇f(c)
        fc = self.linear_approximation[1]  # f(c)
        c = self.expansion_point
        domain_lower, domain_upper = self.domain
        R_lower, R_upper = self.remainder

        # Convert to affine form: f(x) = A*x + b where A = J, b = fc - J*c
        b_affine = fc - self.translator.matrix_vector(J, c)
        A_affine = J

        # Split positive and negative parts for interval arithmetic
        A_affine_pos = self.translator.clamp(A_affine, min=0)
        A_affine_neg = self.translator.clamp(A_affine, max=0)

        # Compute bounds of affine part using interval arithmetic
        affine_range_lower = self.translator.matrix_vector(A_affine_pos, domain_lower) + self.translator.matrix_vector(A_affine_neg, domain_upper) + b_affine
        affine_range_upper = self.translator.matrix_vector(A_affine_pos, domain_upper) + self.translator.matrix_vector(A_affine_neg, domain_lower) + b_affine

        # Add remainder bounds to get total range
        total_range_lower = affine_range_lower + R_lower
        total_range_upper = affine_range_upper + R_upper

        # Sanity check: lower bounds should not exceed upper bounds
        assert self.translator.all(
            total_range_lower <= total_range_upper + 1e-9
        ), f"Lower bound > upper bound in range calculation: {total_range_lower} vs {total_range_upper}"

        return (total_range_lower, total_range_upper)

    def _get_dx_bounds(self):
        """
        Get tight bounds for dx = x - expansion_point over the domain.

        For simplicial domains, computes exact bounds using vertex enumeration.
        For rectangular domains, uses interval arithmetic.

        Returns:
            tuple: (dx_lower, dx_upper) tight bounds for displacement from expansion point
        """
        if self.is_simplex and self.simplex_vertices is not None:
            # For simplicial domains, compute exact bounds using vertices
            displacements = self.simplex_vertices - self.expansion_point
            dx_lower = self.translator.min(displacements, dim=0)
            dx_upper = self.translator.max(displacements, dim=0)
            return dx_lower, dx_upper
        else:
            # For rectangular domains, use interval arithmetic
            dx_lower = self.domain[0] - self.expansion_point
            dx_upper = self.domain[1] - self.expansion_point
            return dx_lower, dx_upper

    def _range_simplex(self):
        """Range computation for simplicial domains using linear programming."""
        J = self.linear_approximation[0]  # Jacobian ∇f(c)
        fc = self.linear_approximation[1]  # f(c)
        c = self.expansion_point
        R_lower, R_upper = self.remainder

        scalar_output = c.ndim == J.ndim
        if scalar_output:
            # Single batch case: add batch dimension for consistent processing
            J = self.translator.unsqueeze(J, dim=-2)
            fc = self.translator.unsqueeze(fc, dim=-1)

        # diff_dim = J.ndim - fc.ndim
        # for _ in range(diff_dim - 1):
        #     c = self.translator.unsqueeze(c, dim=-2)

        # Convert to affine form: f(x) = A*x + b where A = J, b = fc - J*c
        b_affine = fc - self.translator.matrix_vector(J, c)
        A_affine = J

        # Minimize and maximize the linear function over the simplex
        min_val = self._optimize_linear_over_simplex(A_affine, minimize=True)
        max_val = self._optimize_linear_over_simplex(A_affine, minimize=False)

        if scalar_output:
            # Remove added batch dimension
            b_affine = self.translator.squeeze(b_affine, dim=-1)
            min_val = self.translator.squeeze(min_val, dim=-1)
            max_val = self.translator.squeeze(max_val, dim=-1)

        affine_range_lower = min_val + b_affine
        affine_range_upper = max_val + b_affine

        # Add remainder bounds to get total range
        total_range_lower = affine_range_lower + R_lower
        total_range_upper = affine_range_upper + R_upper

        return (total_range_lower, total_range_upper)

    def _optimize_linear_over_simplex(self, objective, minimize=True):
        """
        Optimize a linear function over the simplex using vertex enumeration.

        For a linear function c^T x over a simplex, the optimum is always at a vertex.
        """
        values = objective @ self.translator.transpose(self.simplex_vertices)

        if minimize:
            return self.translator.min(values, dim=-1)
        else:
            return self.translator.max(values, dim=-1)

    def __neg__(self):
        """
        Negation operation for Taylor expansions.

        Returns:
            CertifiedFirstOrderTaylorExpansion: Negated Taylor expansion
        """
        result = CertifiedFirstOrderTaylorExpansion(
            expansion_point=self.expansion_point,
            domain=self.domain,
            linear_approximation=(-self.linear_approximation[0], -self.linear_approximation[1]),
            remainder=(-self.remainder[1], -self.remainder[0]),
            numeric_translator=self.translator,
        )
        self._propagate_simplex_info(result)
        return result

    def _create_result_with_simplex_info(self, source_expansion, expansion_point, domain, linear_approximation, remainder):
        """Helper method to create a result Taylor expansion and propagate simplex information."""
        result = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain,
            linear_approximation=linear_approximation,
            remainder=remainder,
            numeric_translator=self.translator,
        )

        # Propagate simplex information from source
        result.simplex_vertices = source_expansion.simplex_vertices
        result.is_simplex = source_expansion.is_simplex

        return result


class TaylorTranslator:
    def __init__(self, numeric_translator=None):
        if numeric_translator is None:
            numeric_translator = NumpyTranslator()
        self.translator = numeric_translator

    def eye_like(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Create an identity matrix matching the input Taylor expansion's input dimension.
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: np.ndarray of shape [n, n]
        """
        assert a.linear_approximation[0].ndim in [2, 3], "Jacobian must be 2D or 3D array"

        eye_dim = a.linear_approximation[0].shape[-2]
        orig_dim = a.linear_approximation[0].shape[-1]

        lin = self.translator.zeros((eye_dim, eye_dim, orig_dim))
        const = self.translator.eye(eye_dim)
        rem = self.translator.zeros((eye_dim, eye_dim))
        if a.linear_approximation[0].ndim == 3:
            batch_size = a.linear_approximation[0].shape[0]
            lin = self.translator.expand(self.translator.unsqueeze(lin, dim=0), batch_size, dim=0)  # Add batch dim
            const = self.translator.expand(self.translator.unsqueeze(const, dim=0), batch_size, dim=0)  # Add batch dim
            rem = self.translator.expand(self.translator.unsqueeze(rem, dim=0), batch_size, dim=0)  # Add batch dim

        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(lin, const),
            remainder=(rem, rem),
        )

    def zeros_like(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Create a zero Taylor expansion matching the input Taylor expansion's shape.
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        lin = self.translator.zeros_like(a.linear_approximation[0])
        const = self.translator.zeros_like(a.linear_approximation[1])
        rem = self.translator.zeros_like(a.remainder[0])
        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(lin, const),
            remainder=(rem, rem),
        )

    def ones_like(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Create a ones Taylor expansion matching the input Taylor expansion's shape.
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        lin = self.translator.zeros_like(a.linear_approximation[0])
        const = self.translator.ones_like(a.linear_approximation[1])
        rem = self.translator.zeros_like(a.remainder[0])
        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(lin, const),
            remainder=(rem, rem),
        )

    def matrix_vector(self, a, b: CertifiedFirstOrderTaylorExpansion):
        """
        Matrix-vector multiplication
        :param a: np.ndarray of floats [n, m]
        :param b: Taylor model of size [m]
        :return: CertifiedFirstOrderTaylorExpansion
        """
        linear_term = a @ b.linear_approximation[0]
        constant_term = a @ b.linear_approximation[1]

        new_linear = (linear_term, constant_term)

        remainder_lower, remainder_upper = b.remainder
        # Use proper interval arithmetic for matrix-vector multiplication
        # This correctly handles the signs of matrix elements
        remainder1, remainder2 = _mat_interval_vec_mul(a, remainder_lower, remainder_upper, self.translator)
        new_remainder = (remainder1, remainder2)

        result = b._create_result_with_simplex_info(
            source_expansion=b,
            expansion_point=b.expansion_point,
            domain=b.domain,
            linear_approximation=new_linear,
            remainder=new_remainder,
        )

        return result

    def sin(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Element-wise sine
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]  # Center of expansion for y
        f_y0_val = self.translator.sin(y0)  # sin(y0)
        grad_f_y0 = self.translator.cos(y0)  # cos(y0), derivative of sin(y) at y0

        linear_term_jacobian = self.translator.unsqueeze(grad_f_y0, dim=-1) * a.linear_approximation[0]
        domain_low, domain_high = a.range()

        # Check if any point x = k*2pi + pi/2 (where sin(x) = +1) is within [a, b]
        # This is true if there is an integer k such that:
        # a <= k*pi + pi/2 <= b  =>  (a - pi/2)/pi <= k <= (b - pi/2)/pi
        # So, we check if floor((b - pi/2)/pi) >= ceil((a - pi/2)/pi)

        # Trough of -sin (sin=1) at x = 2kπ + π/2
        k_lower_bound = self.translator.ceil((domain_low - 3 * np.pi / 2) / (2 * np.pi))
        k_upper_bound = self.translator.floor((domain_high - 3 * np.pi / 2) / (2 * np.pi))
        contains_trough = k_lower_bound <= k_upper_bound

        # If no crest, the maximum value of |sin(x)| is at the endpoints
        M_lagrange_max = self.translator.maximum(-self.translator.sin(domain_low), -self.translator.sin(domain_high))
        M_lagrange_max = self.translator.where(contains_trough, 1.0, M_lagrange_max)  # If contains trough, max is 1.0
        M_lagrange_max = self.translator.clamp(M_lagrange_max, min=0.0)  # Ensure non-negative max

        max_abs_y_minus_y0 = self.translator.maximum(self.translator.abs(domain_low - y0), self.translator.abs(domain_high - y0))
        local_error_magnitude_max = (M_lagrange_max / 2) * max_abs_y_minus_y0**2

        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = self.translator.ceil((domain_low - np.pi / 2) / (2 * np.pi))
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = self.translator.floor((domain_high - np.pi / 2) / (2 * np.pi))
        contains_crest = k_lower_bound <= k_upper_bound

        # If no trough, the minimum value of |sin(x)| is at the endpoints
        M_lagrange_min = self.translator.minimum(-self.translator.sin(domain_low), -self.translator.sin(domain_high))
        M_lagrange_min = self.translator.where(contains_crest, -1.0, M_lagrange_min)  # # If contains crest, min is -1.0
        M_lagrange_min = self.translator.clamp(M_lagrange_min, max=0.0)  # Ensure non-negative max

        max_abs_y_minus_y0 = self.translator.maximum(self.translator.abs(domain_low - y0), self.translator.abs(domain_high - y0))
        local_error_magnitude_min = (M_lagrange_min / 2) * max_abs_y_minus_y0**2

        second_derivative_bounds = (M_lagrange_min, M_lagrange_max)
        local_error_magnitude_min, local_error_magnitude_max = a._compute_function_composition_remainder(second_derivative_bounds)

        # Propagate remainder through the derivative
        prop_rem_lower_y, prop_rem_upper_y = a.remainder
        term1_rem = grad_f_y0 * prop_rem_lower_y
        term2_rem = grad_f_y0 * prop_rem_upper_y

        propagated_taylor_rem_lower = self.translator.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = self.translator.maximum(term1_rem, term2_rem)

        # --- Final summation (applies to all elements) ---
        final_rem_lower = propagated_taylor_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude_max

        # Apply global bounds tightening for sin function (range [-1, 1])
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point, a.domain, (linear_term_jacobian, f_y0_val), (final_rem_lower, final_rem_upper), self.translator
        )

        clip_rem_lower, clip_rem_upper = apply_global_bounds_tightening(temp_expansion, -1.0, 1.0)

        # Intersect the Taylor bounds with the global range bounds
        final_rem_lower = self.translator.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = self.translator.minimum(final_rem_upper, clip_rem_upper)

        remainder = (final_rem_lower, final_rem_upper)

        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(linear_term_jacobian, f_y0_val),
            remainder=remainder,
        )

    def cos(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Element-wise cosine
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]  # Center of expansion for y
        f_y0_val = self.translator.cos(y0)  # cos(y0)
        grad_cos_y0 = -self.translator.sin(y0)  # -sin(y0), derivative of cos(y) at y0

        linear_term_jacobian = self.translator.unsqueeze(grad_cos_y0, dim=-1) * a.linear_approximation[0]
        domain_low, domain_high = a.range()

        # Check if any point x = k*2pi (where cos(x) = +1) is within [a, b]
        # This is true if there is an integer k such that:
        # a <= k*2pi <= b  =>  (a)/2pi <= k <= (b)/2pi

        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = self.translator.ceil((domain_low - np.pi) / (2 * np.pi))
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = self.translator.floor((domain_high - np.pi) / (2 * np.pi))
        contains_trough = k_lower_bound <= k_upper_bound

        # If no crest, the maximum value of -cos(x) is at the endpoints
        M_lagrange_max = self.translator.maximum(-self.translator.cos(domain_low), -self.translator.cos(domain_high))
        M_lagrange_max = self.translator.where(contains_trough, 1.0, M_lagrange_max)  # If contains trough, max is 1.0
        M_lagrange_max = self.translator.clamp(M_lagrange_max, min=0.0)  # Ensure non-negative max

        max_abs_y_minus_y0 = self.translator.maximum(self.translator.abs(domain_low - y0), self.translator.abs(domain_high - y0))
        local_error_magnitude_max = (M_lagrange_max / 2) * max_abs_y_minus_y0**2

        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = self.translator.ceil(domain_low / (2 * np.pi))  # Leave np.pi as both torch and numpy support it
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = self.translator.floor(domain_high / (2 * np.pi))
        contains_crest = k_lower_bound <= k_upper_bound

        # If no trough, the minimum value of -cos(x) is at the endpoints
        M_lagrange_min = self.translator.minimum(-self.translator.cos(domain_low), -self.translator.cos(domain_high))
        M_lagrange_min = self.translator.where(contains_crest, -1.0, M_lagrange_min)  # # If contains crest, min is -1.0
        M_lagrange_min = self.translator.clamp(M_lagrange_min, max=0.0)  # Ensure non-negative max

        max_abs_y_minus_y0 = self.translator.maximum(self.translator.abs(domain_low - y0), self.translator.abs(domain_high - y0))
        local_error_magnitude_min = (M_lagrange_min / 2) * max_abs_y_minus_y0**2

        second_derivative_bounds = (M_lagrange_min, M_lagrange_max)
        local_error_magnitude_min, local_error_magnitude_max = a._compute_function_composition_remainder(second_derivative_bounds)

        # Propagate remainder through the derivative
        prop_rem_lower_y, prop_rem_upper_y = a.remainder
        term1_rem = grad_cos_y0 * prop_rem_lower_y
        term2_rem = grad_cos_y0 * prop_rem_upper_y

        propagated_taylor_rem_lower = self.translator.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = self.translator.maximum(term1_rem, term2_rem)

        # --- Final summation (applies to all elements) ---
        final_rem_lower = propagated_taylor_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude_max

        # Apply global bounds tightening for cos function (range [-1, 1])
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point, a.domain, (linear_term_jacobian, f_y0_val), (final_rem_lower, final_rem_upper), self.translator
        )

        clip_rem_lower, clip_rem_upper = apply_global_bounds_tightening(temp_expansion, -1.0, 1.0)

        # Intersect the Taylor bounds with the global range bounds
        final_rem_lower = self.translator.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = self.translator.minimum(final_rem_upper, clip_rem_upper)

        remainder = (final_rem_lower, final_rem_upper)

        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(linear_term_jacobian, f_y0_val),
            remainder=remainder,
        )

    def exp(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Element-wise exponential
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]
        exp_y0 = self.translator.exp(y0)

        linear_term = self.translator.unsqueeze(exp_y0, dim=-1) * a.linear_approximation[0]

        # For exp(x), f''(x) = exp(x)
        # Compute bounds on the second derivative over the range
        range_lower, range_upper = a.range()

        # exp''(x) = exp(x), bounded by [exp(range_lower), exp(range_upper)]
        M_min_f_double_prime = self.translator.exp(range_lower)
        M_max_f_double_prime = self.translator.exp(range_upper)

        second_derivative_bounds = (M_min_f_double_prime, M_max_f_double_prime)
        local_error_magnitude_min, local_error_magnitude_max = a._compute_function_composition_remainder(second_derivative_bounds)

        # Propagate remainder through the derivative
        remainder1, remainder2 = exp_y0 * a.remainder[0], exp_y0 * a.remainder[1]
        propagated_rem_lower = self.translator.minimum(remainder1, remainder2)
        propagated_rem_upper = self.translator.maximum(remainder1, remainder2)

        # Combine propagated and local remainders
        final_rem_lower = propagated_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_rem_upper + local_error_magnitude_max

        remainder = (final_rem_lower, final_rem_upper)

        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(linear_term, exp_y0),
            remainder=remainder,
        )

    def log(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Element-wise logarithm
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]

        # Check domain validity first - both expansion point and range must be positive
        if self.translator.any(y0 <= 0):
            raise ValueError("Logarithm domain error: expansion point must be positive")

        # Check domain validity for the entire range
        lower, upper = a.range()
        if self.translator.any(lower <= 0):
            raise ValueError("Logarithm domain error: entire range must be positive")

        log_y0 = self.translator.log(y0)
        grad_y0 = 1 / y0

        linear_term = self.translator.unsqueeze(grad_y0, dim=-1) * a.linear_approximation[0]

        # For log(x), f''(x) = -1/x^2
        # Compute bounds on the second derivative over the range
        # Since the range is guaranteed to be positive, this is safe
        M_min_f_double_prime = -1.0 / (upper**2)  # Maximum of -1/x^2 (least negative)
        M_max_f_double_prime = -1.0 / (lower**2)  # Minimum of -1/x^2 (most negative)

        # Ensure valid bounds
        M_max_f_double_prime = self.translator.clamp(M_max_f_double_prime, max=0)  # TODO: is this correct? max is negative and min is positive?
        M_min_f_double_prime = self.translator.clamp(M_min_f_double_prime, min=0)

        second_derivative_bounds = (M_min_f_double_prime, M_max_f_double_prime)
        local_error_magnitude_min, local_error_magnitude_max = a._compute_function_composition_remainder(second_derivative_bounds)

        # Propagate the remainder through the derivative
        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        propagated_rem_lower = self.translator.minimum(remainder1, remainder2)
        propagated_rem_upper = self.translator.maximum(remainder1, remainder2)

        # Combine propagated and local remainders
        final_rem_lower = propagated_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_rem_upper + local_error_magnitude_max

        # Apply monotonic bounds tightening for log function
        # log is monotonically increasing, so use boundary values as global bounds
        log_at_boundaries = (self.translator.log(lower), self.translator.log(upper))

        # Create temporary Taylor expansion to use the monotonic tightening helper
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point, a.domain, (linear_term, log_y0), (final_rem_lower, final_rem_upper), self.translator
        )

        clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, log_at_boundaries, is_increasing=True)

        # Intersect the Taylor bounds with the monotonic bounds
        final_rem_lower = self.translator.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = self.translator.minimum(final_rem_upper, clip_rem_upper)

        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(linear_term, log_y0),
            remainder=(final_rem_lower, final_rem_upper),
        )

    def sqrt(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Element-wise square root
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]

        # --- FIX 1: Check expansion point ---
        # The derivative 1/(2*sqrt(y)) is undefined at y=0.
        # The expansion point y0 must be strictly positive.
        if self.translator.any(y0 <= 1e-14):
            raise ValueError("Square root expansion point y0 must be strictly positive.")

        # --- Domain Check ---
        range_lower, range_upper = a.range()
        if self.translator.any(range_lower < 0):
            raise ValueError("Square root domain error: entire range must be non-negative")

        # --- Calculations (now safe due to check) ---
        sqrt_y0 = self.translator.sqrt(y0)
        grad_y0 = 0.5 / sqrt_y0
        linear_term = self.translator.unsqueeze(grad_y0, dim=-1) * a.linear_approximation[0]

        # --- Remainder Calculation ---

        # Strategy 1: Use Concavity (Always Valid)
        # Sqrt is concave, so f(y) <= f(y0) + f'(y0)(y-y0).
        # The linear approximation always overestimates.
        # The local remainder is always <= 0.
        linear_at_lower = sqrt_y0 + grad_y0 * (range_lower - y0)
        linear_at_upper = sqrt_y0 + grad_y0 * (range_upper - y0)

        true_at_lower = self.translator.sqrt(range_lower)
        true_at_upper = self.translator.sqrt(range_upper)

        remainder_at_lower = true_at_lower - linear_at_lower
        remainder_at_upper = true_at_upper - linear_at_upper

        # --- FIX 2: Correct concavity bounds ---
        concavity_rem_lower = self.translator.minimum(remainder_at_lower, remainder_at_upper)
        concavity_rem_upper = self.translator.zeros_like(concavity_rem_lower)  # Upper bound is 0

        # Strategy 2: Lagrange Remainder (Simplex-Aware)
        # f''(x) = -0.25 * x^(-1.5)
        # This is only computable if range_lower > 0 (i.e., not at zero)

        # We need to check this per-element for vectorization
        is_safe_for_lagrange = range_lower > 1e-14

        # Initialize remainder bounds with the "always valid" concavity bounds
        local_error_magnitude_min = concavity_rem_lower
        local_error_magnitude_max = concavity_rem_upper  # This is 0.0

        # --- Compute Lagrange where safe ---
        if self.translator.any(is_safe_for_lagrange):
            coeff_f_double_prime = -0.25
            exponent_f_double_prime = -1.5

            # Get safe sub-intervals
            safe_range_lower = range_lower[is_safe_for_lagrange]
            safe_range_upper = range_upper[is_safe_for_lagrange]

            # Initialize with -inf / 0 (f'' is always negative)
            M_min_f_double_prime = self.translator.full_like(range_lower, -np.inf)
            M_max_f_double_prime = self.translator.full_like(range_lower, 0.0)

            # Compute bounds only for the safe elements
            M_min_safe = min_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (safe_range_lower, safe_range_upper), self.translator)
            M_max_safe = max_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (safe_range_lower, safe_range_upper), self.translator)

            M_min_f_double_prime[is_safe_for_lagrange] = M_min_safe
            M_max_f_double_prime[is_safe_for_lagrange] = M_max_safe

            second_derivative_bounds = (M_min_f_double_prime, M_max_f_double_prime)

            # This call IS simplex-aware.
            # For unsafe elements, M_min is -inf, so this will
            # (correctly) produce [-inf, inf] for those elements.
            lagrange_error_min, lagrange_error_max = a._compute_function_composition_remainder(second_derivative_bounds)

            # Intersect the bounds.
            # For unsafe elements:
            # max(concavity_min, -inf) = concavity_min
            # min(0, +inf) = 0
            # This logic correctly selects the concavity bound for
            # unsafe elements and the tighter intersection for safe ones.
            local_error_magnitude_min = self.translator.maximum(concavity_rem_lower, lagrange_error_min)
            local_error_magnitude_max = self.translator.minimum(concavity_rem_upper, lagrange_error_max)

        # Propagate the remainder through the derivative (now safe)
        prop_rem_lower, prop_rem_upper = a.remainder
        term1_rem = grad_y0 * prop_rem_lower
        term2_rem = grad_y0 * prop_rem_upper

        propagated_rem_lower = self.translator.minimum(term1_rem, term2_rem)
        propagated_rem_upper = self.translator.maximum(term1_rem, term2_rem)

        # Combine propagated and local remainders
        final_rem_lower = propagated_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_rem_upper + local_error_magnitude_max

        # Apply monotonic bounds tightening for sqrt function
        # This helper is not simplex-aware, but is still a valid (if
        # slightly loose on simplex) tightening step.
        sqrt_at_boundaries = (self.translator.sqrt(range_lower), self.translator.sqrt(range_upper))

        # Create temporary Taylor expansion to use the monotonic tightening helper
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point, a.domain, (linear_term, sqrt_y0), (final_rem_lower, final_rem_upper), self.translator
        )

        clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, sqrt_at_boundaries, is_increasing=True)

        # Intersect the Taylor bounds with the monotonic bounds
        final_rem_lower = self.translator.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = self.translator.minimum(final_rem_upper, clip_rem_upper)

        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(linear_term, sqrt_y0),
            remainder=(final_rem_lower, final_rem_upper),
        )

    def cbrt(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Element-wise cube root

        Uses a hybrid remainder strategy:
        1. Tries the (tight) simplex-aware Lagrange remainder if domain is safe (doesn't contain 0).
        2. Computes a (safe) fallback bound by checking extrema at endpoints and
           the critical point y = -y0.
        3. Intersects these bounds to get the tightest finite local remainder.

        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]

        # --- FIX 1: Check expansion point (Still necessary) ---
        if self.translator.any(self.translator.abs(y0) < 1e-14):
            raise ValueError("Cube root expansion point y0 must be non-zero (derivative is infinite).")

        cbrt_y0 = self.translator.cbrt(y0)
        grad_y0 = 1 / (3 * self.translator.pow(cbrt_y0, 2))
        linear_term = self.translator.unsqueeze(grad_y0, dim=-1) * a.linear_approximation[0]

        range_lower, range_upper = a.range()

        # --- STRATEGY 1: Lagrange Remainder (Simplex-Aware, but fails at 0) ---

        # We can only use Lagrange if the domain [l, u] does not contain 0.
        is_safe_for_lagrange = (range_lower > 0) | (range_upper < 0)

        # Initialize with [-inf, +inf] for unsafe elements
        lagrange_error_min = self.translator.full_like(y0, -np.inf)
        lagrange_error_max = self.translator.full_like(y0, np.inf)

        if self.translator.any(is_safe_for_lagrange):
            safe_l = range_lower[is_safe_for_lagrange]
            safe_u = range_upper[is_safe_for_lagrange]

            coeff_f_double_prime = -2.0 / 9.0
            exponent_f_double_prime = -5.0 / 3.0

            M_min_safe = min_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (safe_l, safe_u), self.translator)
            M_max_safe = max_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (safe_l, safe_u), self.translator)

            # Create bounds array just for safe elements to pass to compute_remainder
            # (We only need to create a temporary 'a' object if we want to run
            # simplex-aware compute_remainder on a subset, which is complex.
            # Instead, we run on all, using -inf/inf for unsafe ones)
            M_min_f_double_prime = self.translator.full_like(y0, -np.inf)
            M_max_f_double_prime = self.translator.full_like(y0, np.inf)
            M_min_f_double_prime[is_safe_for_lagrange] = M_min_safe
            M_max_f_double_prime[is_safe_for_lagrange] = M_max_safe

            second_derivative_bounds = (M_min_f_double_prime, M_max_f_double_prime)

            # This call is simplex-aware.
            # It will return finite bounds for safe elements
            # and [-inf, +inf] for unsafe elements.
            lagrange_error_min, lagrange_error_max = a._compute_function_composition_remainder(second_derivative_bounds)

        # --- STRATEGY 2: Fallback Bound (Checking Extrema) ---
        # This is a *finite* bound for *all* elements.
        # Extrema of R(y) = f(y) - L(y) are at y=endpoints and y = -y0.

        # 1. At endpoints
        # Compute linear approximation at endpoints
        linear_at_lower = cbrt_y0 + grad_y0 * (range_lower - y0)
        linear_at_upper = cbrt_y0 + grad_y0 * (range_upper - y0)

        # Handle cube root for all real numbers (no domain restriction needed)
        true_at_lower = self.translator.cbrt(range_lower)
        true_at_upper = self.translator.cbrt(range_upper)

        # Remainder = true_value - linear_approximation
        remainder_at_lower = true_at_lower - linear_at_lower
        remainder_at_upper = true_at_upper - linear_at_upper

        # The remainder bounds are always non-positive due to concavity
        remainder_lower = self.translator.minimum(remainder_at_lower, remainder_at_upper)
        remainder_upper = self.translator.maximum(remainder_at_lower, remainder_at_upper)

        # Ensure upper bound is never positive (due to concavity)
        remainder_upper = self.translator.clamp(remainder_upper, min=0.0)

        # --- COMBINE BOUNDS ---
        # Intersect the two strategies to get the tightest certified bound
        local_error_magnitude_min = self.translator.maximum(lagrange_error_min, remainder_lower)
        local_error_magnitude_max = self.translator.minimum(lagrange_error_max, remainder_upper)

        # Propagate the remainder through the derivative
        prop_rem_lower, prop_rem_upper = a.remainder
        term1_rem = grad_y0 * prop_rem_lower
        term2_rem = grad_y0 * prop_rem_upper

        propagated_rem_lower = self.translator.minimum(term1_rem, term2_rem)
        propagated_rem_upper = self.translator.maximum(term1_rem, term2_rem)

        final_rem_lower = propagated_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_rem_upper + local_error_magnitude_max

        # --- FINAL TIGHTENING (MONOTONICITY) ---
        cbrt_at_boundaries = (self.translator.cbrt(range_lower), self.translator.cbrt(range_upper))
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point, a.domain, (linear_term, cbrt_y0), (final_rem_lower, final_rem_upper), self.translator
        )
        clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, cbrt_at_boundaries, is_increasing=True)

        final_rem_lower = self.translator.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = self.translator.minimum(final_rem_upper, clip_rem_upper)

        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(linear_term, cbrt_y0),
            remainder=(final_rem_lower, final_rem_upper),
        )

    def pow(self, a, exponent_b):
        """
        Element-wise power
        :param a: CertifiedFirstOrderTaylorExpansion
        :param exponent_b: integer exponent
        :return: CertifiedFirstOrderTaylorExpansion
        """
        assert isinstance(exponent_b, int), "Exponent must be an integer"

        # Handle trivial cases
        if exponent_b == 0:
            # f(y) = 1, f'(y) = 0
            f_y0 = self.translator.ones_like(a.linear_approximation[1])
            grad_f_y0 = self.translator.zeros_like(a.linear_approximation[1])
            linear_term_jacobian = self.translator.zeros_like(a.linear_approximation[0])
            local_error_magnitude_min = 0.0
            local_error_magnitude_max = 0.0
            remainder = (local_error_magnitude_min * self.translator.ones_like(f_y0), local_error_magnitude_max * self.translator.ones_like(f_y0))
            return CertifiedFirstOrderTaylorExpansion(a.expansion_point, a.domain, (linear_term_jacobian, f_y0), remainder, self.translator)

        elif exponent_b == 1:
            # f(y) = y, f'(y) = 1
            return a

        y0 = a.linear_approximation[1]

        # Handle special cases for negative exponents
        if exponent_b < 0:
            # For negative exponents, we need to ensure y0 and range don't contain 0
            range_lower, range_upper = a.range()
            if self.translator.any(self.translator.abs(y0) < 1e-14):
                raise ValueError(f"Power with negative exponent {exponent_b}: expansion point cannot be zero")
            if self.translator.any((range_lower <= 0) & (range_upper >= 0)):
                raise ValueError(f"Power with negative exponent {exponent_b}: range cannot contain zero")

        # For other exponents, proceed with normal computation
        f_y0 = self.translator.pow(y0, exponent_b)
        grad_f_y0 = exponent_b * self.translator.pow(y0, exponent_b - 1)

        linear_term_jacobian = self.translator.unsqueeze(grad_f_y0, dim=-1) * a.linear_approximation[0]

        range_of_y = a.range()
        coeff_f_double_prime = exponent_b * (exponent_b - 1)
        exponent_f_double_prime = exponent_b - 2
        M_lagrange_max = max_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, range_of_y, self.translator)
        M_lagrange_min = min_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, range_of_y, self.translator)

        # Use proper remainder bound computation for function composition f(g(x)) = g(x)^exponent_b
        second_derivative_bounds = (M_lagrange_min, M_lagrange_max)
        local_error_magnitude_min, local_error_magnitude_max = a._compute_function_composition_remainder(second_derivative_bounds)

        prop_rem_lower_y, prop_rem_upper_y = a.remainder

        term1_rem = grad_f_y0 * prop_rem_lower_y
        term2_rem = grad_f_y0 * prop_rem_upper_y

        propagated_taylor_rem_lower = self.translator.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = self.translator.maximum(term1_rem, term2_rem)

        final_rem_lower = propagated_taylor_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude_max

        # Apply monotonic bounds tightening for specific power functions
        if exponent_b > 0:
            # For positive exponents, x^n is monotonically increasing for x > 0
            if self.translator.all(range_of_y[0] > 0):  # All values in range are positive
                power_at_boundaries = (self.translator.pow(range_of_y[0], exponent_b), self.translator.pow(range_of_y[1], exponent_b))

                # Create temporary Taylor expansion to use the monotonic tightening helper
                temp_expansion = CertifiedFirstOrderTaylorExpansion(
                    a.expansion_point, a.domain, (linear_term_jacobian, f_y0), (final_rem_lower, final_rem_upper), self.translator
                )

                clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, power_at_boundaries, is_increasing=True)

                # Intersect the Taylor bounds with the monotonic bounds
                final_rem_lower = self.translator.maximum(final_rem_lower, clip_rem_lower)
                final_rem_upper = self.translator.minimum(final_rem_upper, clip_rem_upper)
        elif exponent_b < 0:
            # For negative exponents, x^n is monotonically decreasing for x > 0 (same as 1/x^|n|)
            if self.translator.all(range_of_y[0] > 0):  # All values in range are positive
                power_at_boundaries = (self.translator.pow(range_of_y[0], exponent_b), self.translator.pow(range_of_y[1], exponent_b))

                # Create temporary Taylor expansion to use the monotonic tightening helper
                temp_expansion = CertifiedFirstOrderTaylorExpansion(
                    a.expansion_point, a.domain, (linear_term_jacobian, f_y0), (final_rem_lower, final_rem_upper), self.translator
                )

                clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, power_at_boundaries, is_increasing=False)

                # Intersect the Taylor bounds with the monotonic bounds
                final_rem_lower = self.translator.maximum(final_rem_lower, clip_rem_lower)
                final_rem_upper = self.translator.minimum(final_rem_upper, clip_rem_upper)

        remainder = (final_rem_lower, final_rem_upper)

        return a._create_result_with_simplex_info(
            source_expansion=a,
            expansion_point=a.expansion_point,
            domain=a.domain,
            linear_approximation=(linear_term_jacobian, f_y0),
            remainder=remainder,
        )

    def cat(self, xs, dim=0):
        """
        Cat a list of certified taylor expansions along a given dimension

        :param a: list of certified taylor expansions

        :return: CertifiedFirstOrderTaylorExpansion
        """
        assert len(xs) > 0
        # Expansion points must match
        for i, x in enumerate(xs):
            assert self.translator.allclose(
                x.expansion_point, xs[0].expansion_point
            ), f"Expansion point mismatch at index {i}: {x.expansion_point} != {xs[0].expansion_point}"
        # Domains must match (use rectangular bounds for quick check)
        if x.is_simplex:
            # For simplicial domains, check if vertices match
            for i, x in enumerate(xs):
                assert (
                    x.is_simplex
                    and xs[0].is_simplex
                    and x.simplex_vertices is not None
                    and xs[0].simplex_vertices is not None
                    and self.translator.allclose(x.simplex_vertices, xs[0].simplex_vertices)
                ), f"Simplex vertices mismatch at index {i}"
        else:
            for i, x in enumerate(xs):
                assert self.translator.allclose(x.domain[0], xs[0].domain[0]) and self.translator.allclose(
                    x.domain[1], xs[0].domain[1]
                ), f"Domain mismatch at index {i}: {x.domain} != {xs[0].domain}"

        if dim < 0:
            cat_dim = dim - x.domain[0].ndim
        else:
            cat_dim = dim

        # Concatenate linear parts and remainders
        new_J = self.translator.cat([x.linear_approximation[0] for x in xs], dim=cat_dim)
        new_fc = self.translator.cat([x.linear_approximation[1] for x in xs], dim=dim)
        new_Rl = self.translator.cat([x.remainder[0] for x in xs], dim=dim)
        new_Rh = self.translator.cat([x.remainder[1] for x in xs], dim=dim)

        result = xs[0]._create_result_with_simplex_info(
            source_expansion=xs[0],
            expansion_point=xs[0].expansion_point,
            domain=xs[0].domain,
            linear_approximation=(new_J, new_fc),
            remainder=(new_Rl, new_Rh),
        )
        return result

    def stack(self, xs, dim=0):
        """
        Stack a list of certified taylor expansions along a given dimension
        :param xs: list of certified taylor expansions
        :return: CertifiedFirstOrderTaylorExpansion
        """
        assert len(xs) > 0
        # Expansion points must match
        for i, x in enumerate(xs):
            assert self.translator.allclose(
                x.expansion_point, xs[0].expansion_point
            ), f"Expansion point mismatch at index {i}: {x.expansion_point} != {xs[0].expansion_point}"
        # Domains must match (use rectangular bounds for quick check)
        if x.is_simplex:
            # For simplicial domains, check if vertices match
            for i, x in enumerate(xs):
                assert (
                    x.is_simplex
                    and xs[0].is_simplex
                    and x.simplex_vertices is not None
                    and xs[0].simplex_vertices is not None
                    and self.translator.allclose(x.simplex_vertices, xs[0].simplex_vertices)
                ), f"Simplex vertices mismatch at index {i}"
        else:
            for i, x in enumerate(xs):
                assert self.translator.allclose(x.domain[0], xs[0].domain[0]) and self.translator.allclose(
                    x.domain[1], xs[0].domain[1]
                ), f"Domain mismatch at index {i}: {x.domain} != {xs[0].domain}"

        if dim < 0:
            diff_dim = x.linear_approximation[0].ndim - x.linear_approximation[1].ndim
            jacobian_dim = dim - diff_dim
        else:
            jacobian_dim = dim

        # Concatenate linear parts and remainders
        new_J = self.translator.stack([x.linear_approximation[0] for x in xs], dim=jacobian_dim)
        new_fc = self.translator.stack([x.linear_approximation[1] for x in xs], dim=dim)
        new_Rl = self.translator.stack([x.remainder[0] for x in xs], dim=dim)
        new_Rh = self.translator.stack([x.remainder[1] for x in xs], dim=dim)

        result = xs[0]._create_result_with_simplex_info(
            source_expansion=xs[0],
            expansion_point=xs[0].expansion_point,
            domain=xs[0].domain,
            linear_approximation=(new_J, new_fc),
            remainder=(new_Rl, new_Rh),
        )
        return result

    def unsqueeze(self, x, dim=0):
        """
        Stack a list of certified taylor expansions along a given dimension
        :param xs: list of certified taylor expansions
        :return: CertifiedFirstOrderTaylorExpansion
        """
        if dim < 0:
            diff_dim = x.linear_approximation[0].ndim - x.linear_approximation[1].ndim
            jacobian_dim = dim - diff_dim
        else:
            jacobian_dim = dim

        # Concatenate linear parts and remainders
        new_J = self.translator.unsqueeze(x.linear_approximation[0], dim=jacobian_dim)
        new_fc = self.translator.unsqueeze(x.linear_approximation[1], dim=dim)
        new_Rl = self.translator.unsqueeze(x.remainder[0], dim=dim)
        new_Rh = self.translator.unsqueeze(x.remainder[1], dim=dim)

        result = x._create_result_with_simplex_info(
            source_expansion=x,
            expansion_point=x.expansion_point,
            domain=x.domain,
            linear_approximation=(new_J, new_fc),
            remainder=(new_Rl, new_Rh),
        )
        return result

    def to_format(self, point, lower, upper):
        """
        Initialize the computation of a certified first-order Taylor expansion with
        the trivial Taylor expansion of f(x) = x, the identity; with a given
        expansion point and domain, and a linear approximation of the identity.
        The remainder is set to zero.
        :param point: expansion point
        :param lower: lower bounds of domain
        :param upper: upper bounds of domain
        :return: CertifiedFirstOrderTaylorExpansion
        """

        # for f(x) = x, the Taylor expansion is c + (x - c) \oplus R where R = 0
        # The Jacobian should be the identity matrix of appropriate size
        result = CertifiedFirstOrderTaylorExpansion(
            expansion_point=point,
            domain=(lower, upper),
            linear_approximation=(self.translator.eye(point.shape[0]), point),
            remainder=(self.translator.zeros(point.shape[0]), self.translator.zeros(point.shape[0])),
            numeric_translator=self.translator,
        )

        # No simplex information for to_format (rectangular domain)
        return result

    def to_format_simplex(self, expansion_point, simplex_vertices):
        """
        Initialize the computation of a certified first-order Taylor expansion with
        the trivial Taylor expansion of f(x) = x, the identity; with a given
        expansion point and simplicial domain defined by vertices.
        The remainder is set to zero.

        :param expansion_point: center point for the Taylor expansion
        :param simplex_vertices: numpy array of vertices defining the simplex domain
        :return: CertifiedFirstOrderTaylorExpansion
        """
        # Ensure inputs are the correct array type
        expansion_point = self.translator.to_format(expansion_point)
        simplex_vertices = self.translator.to_format(simplex_vertices)

        # Validate dimensions
        if len(simplex_vertices.shape) == 1:
            # Handle 1D case where vertices might be passed as 1D array
            simplex_vertices = simplex_vertices.reshape(-1, 1)

        expected_dim = expansion_point.shape[-1]
        if simplex_vertices.shape[-1] != expected_dim:
            raise ValueError(f"Simplex vertices dimension {simplex_vertices.shape[-1]} " f"does not match expansion point dimension {expected_dim}")

        # For f(x) = x, the Taylor expansion is c + (x - c) + R where R = 0
        # The Jacobian should be the identity matrix of appropriate size

        # Compute bounding box of the simplex for compatibility
        domain_lower = self.translator.min(simplex_vertices, dim=-2)
        domain_upper = self.translator.max(simplex_vertices, dim=-2)
        domain = (domain_lower, domain_upper)

        lin = self.translator.eye(expansion_point.shape[-1])
        if expansion_point.ndim == 2:
            batch_size = expansion_point.shape[0]
            lin = self.translator.expand(self.translator.unsqueeze(lin, dim=0), batch_size, dim=0)

        rem = self.translator.zeros_like(expansion_point)

        result = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain,  # Bounding box of simplex for compatibility
            linear_approximation=(lin, expansion_point),
            remainder=(rem, rem),
            numeric_translator=self.translator,
        )

        # Store the exact simplex vertices for precise geometric operations
        result.simplex_vertices = simplex_vertices
        result.is_simplex = True

        return result


def max_monomial_vectorized(c, n, intervals, translator):
    """
    Bound the maximum value of a univariate monomial f(x) = c * x^n over multiple intervals.
    :param c: Coefficient of the monomial (scalar or array-like of shape (m,)).
    :param n: Degree of the monomial (scalar or array-like of shape (m,)).
    :param intervals: tuple(np.ndarray, np.ndarray), both of shape (m,).
    :return: np.ndarray of shape (m,).
    """
    a, b = intervals
    c = translator.to_format(c)
    n = translator.to_format(n)

    # Evaluate endpoints robustly (supports negative endpoints for k/3 exponents)
    f_a = c * translator.pow(a, n)
    f_b = c * translator.pow(b, n)

    max_values = translator.maximum(f_a, f_b)

    # If 0 is in [a, b], also consider f(0). If n < 0, function diverges -> max is +inf
    zero_in_interval = (a <= 0) & (b >= 0)
    if translator.any(zero_in_interval):
        neg_exp_mask = n < 0
        nonzero_c_mask = c != 0
        # Where both zero_in_interval and neg exponent, set +inf conservatively
        both_mask = zero_in_interval & neg_exp_mask & nonzero_c_mask
        if translator.any(both_mask):
            # Broadcast-aware assignment
            max_values = translator.where(both_mask, np.inf, max_values)  # Leave np.inf as both torch and numpy support it
        # Otherwise, include f(0)
        if translator.any((~neg_exp_mask) & zero_in_interval):
            # For non-negative exponents, f(0) = c * 0^n
            # Special cases: 0^0 is typically treated as 1, but for n=0 we have c*1=c
            # Use safe evaluation to avoid np.power(0.0, 0) warnings
            f_zero = translator.where(n == 0, c, 0.0)  # n=0 gives f(0)=c, n>0 gives f(0)=0
            max_values = translator.where(zero_in_interval & ((~neg_exp_mask) | (~nonzero_c_mask)), translator.maximum(max_values, f_zero), max_values)

    return max_values


def min_monomial_vectorized(c, n, intervals, translator):
    """
    Bound the minimum value of a univariate monomial f(x) = c * x^n over multiple intervals.
    :param c: Coefficient of the monomial (scalar or array-like of shape (m,)).
    :param n: Degree of the monomial (scalar or array-like of shape (m,)).
    :param intervals: tuple(np.ndarray, np.ndarray), both of shape (m,).
    :return: np.ndarray of shape (m,).
    """
    a, b = intervals
    c = translator.to_format(c)
    n = translator.to_format(n)

    # Evaluate endpoints robustly (supports negative endpoints for k/3 exponents)
    f_a = c * translator.pow(a, n)
    f_b = c * translator.pow(b, n)

    min_values = translator.minimum(f_a, f_b)

    # If 0 is in [a, b], also consider f(0). If n < 0, function diverges -> min is -inf
    zero_in_interval = (a <= 0) & (b >= 0)
    if translator.any(zero_in_interval):
        neg_exp_mask = n < 0
        nonzero_c_mask = c != 0
        both_mask = zero_in_interval & neg_exp_mask & nonzero_c_mask
        if translator.any(both_mask):
            min_values = translator.where(both_mask, -np.inf, min_values)  # Leave np.inf as both torch and numpy support it
        if translator.any((~neg_exp_mask) & zero_in_interval):
            # For non-negative exponents, f(0) = c * 0^n
            # Special cases: 0^0 is typically treated as 1, but for n=0 we have c*1=c
            # Use safe evaluation to avoid np.power(0.0, 0) warnings
            f_zero = translator.where(n == 0, c, 0.0)  # n=0 gives f(0)=c, n>0 gives f(0)=0
            min_values = translator.where(zero_in_interval & ((~neg_exp_mask) | (~nonzero_c_mask)), translator.minimum(min_values, f_zero), min_values)

    return min_values


def apply_global_bounds_tightening(taylor_expansion, global_lower_bound, global_upper_bound):
    """
    Apply post-processing step to tighten remainder bounds for functions with known global ranges.

    For functions f(y) with known bounds f(y) ∈ [L, U], we can clip the remainder R_f(y)
    to the domain:
    R_f(y) ∈ [L - f_L(y), U - f_L(y)] ⊆ [L - max_y f_L(y), U - min_y f_L(y)]

    where f_L(y) is the linear approximation part of the Taylor expansion.

    Args:
        taylor_expansion (CertifiedFirstOrderTaylorExpansion): The Taylor expansion to tighten
        global_lower_bound (float): Known global lower bound L for the function
        global_upper_bound (float): Known global upper bound U for the function

    Returns:
        tuple: (tightened_remainder_lower, tightened_remainder_upper)
    """
    # Extract linear approximation components
    linear_jacobian, f_y0_val = taylor_expansion.linear_approximation
    translator = taylor_expansion.translator

    # Get domain bounds
    x_domain_low, x_domain_high = taylor_expansion.domain
    x_0 = taylor_expansion.expansion_point

    # Find the interval for (x - x_0)
    delta_x_low = x_domain_low - x_0
    delta_x_high = x_domain_high - x_0

    # Compute range of f_L(x) using interval arithmetic (center-radius form)
    delta_x_center = (delta_x_low + delta_x_high) / 2.0
    delta_x_radius = (delta_x_high - delta_x_low) / 2.0

    scalar_output = linear_jacobian.ndim == delta_x_center.ndim
    if scalar_output:
        # Special case: single output with batch dimension
        linear_jacobian = translator.unsqueeze(linear_jacobian, dim=-2)  # [B, 1, D]

    # Range of J_f @ (x - x_0) = (J_f @ center) +/- (|J_f| @ radius)
    f_L_center_offset = translator.matrix_vector(linear_jacobian, delta_x_center)
    f_L_radius = translator.matrix_vector(translator.abs(linear_jacobian), delta_x_radius)

    if scalar_output:
        f_L_center_offset = translator.squeeze(f_L_center_offset, dim=-1)  # [B]
        f_L_radius = translator.squeeze(f_L_radius, dim=-1)  # [B]

    # Total range of f_L(x) = f_0 + range(J_f @ (x - x_0))
    f_L_min = (f_y0_val + f_L_center_offset) - f_L_radius
    f_L_max = (f_y0_val + f_L_center_offset) + f_L_radius

    # Compute the new remainder bounds implied by the global range
    # R_f(x) >= global_lower_bound - f_L(x)
    # The constant lower bound must be <= the minimum of the right side:
    # R_min' <= min_x(global_lower_bound - f_L(x)) = global_lower_bound - max_x(f_L(x))
    clip_rem_lower = global_lower_bound - f_L_max

    # R_f(x) <= global_upper_bound - f_L(x)
    # The constant upper bound must be >= the maximum of the right side:
    # R_max' >= max_x(global_upper_bound - f_L(x)) = global_upper_bound - min_x(f_L(x))
    clip_rem_upper = global_upper_bound - f_L_min

    return clip_rem_lower, clip_rem_upper


def apply_monotonic_bounds_tightening(taylor_expansion, domain_range, is_increasing=True):
    """
    Apply post-processing step to tighten remainder bounds for monotonic functions.

    For monotonic functions, we can use the fact that extreme values occur at boundaries.
    We compute the actual function values at the boundaries and use them as global bounds.

    Args:
        taylor_expansion (CertifiedFirstOrderTaylorExpansion): The Taylor expansion to tighten
        domain_range (tuple): (f(domain_low), f(domain_high)) - function values at domain boundaries
        is_increasing (bool): True if function is monotonically increasing, False if decreasing

    Returns:
        tuple: (tightened_remainder_lower, tightened_remainder_upper)
    """
    f_at_low, f_at_high = domain_range

    if is_increasing:
        # For increasing functions: min = f(domain_low), max = f(domain_high)
        global_lower = f_at_low
        global_upper = f_at_high
    else:
        # For decreasing functions: min = f(domain_high), max = f(domain_low)
        global_lower = f_at_high
        global_upper = f_at_low

    return apply_global_bounds_tightening(taylor_expansion, global_lower, global_upper)


# Helper function for TE * TE multiplication remainder calculation
def _mat_interval_vec_mul(M, v_low, v_high, translator):
    M_pos = translator.clamp(M, min=0)
    M_neg = translator.clamp(M, max=0)
    res_low = translator.matrix_vector(M_pos, v_low) + translator.matrix_vector(M_neg, v_high)
    res_high = translator.matrix_vector(M_pos, v_high) + translator.matrix_vector(M_neg, v_low)
    return res_low, res_high


def _bernstein_bounds_product_of_linears_over_simplex(L1_at_vertices: np.ndarray, L2_at_vertices: np.ndarray, translator):
    """
    Tight bounds for the product of two linear polynomials over a simplex, per-output.

    ... (docstring remains the same) ...
    """
    assert L1_at_vertices.shape == L2_at_vertices.shape  # [*, V]

    # Vertex coefficients: b_{2e_i}
    # Edge coefficients: b_{e_i+e_j} for i<j

    coeffs = 0.5 * (
        translator.unsqueeze(L1_at_vertices, -1) * translator.unsqueeze(L2_at_vertices, -2)  # [*, V, 1] * [*, 1, V]
        + translator.unsqueeze(L1_at_vertices, -2) * translator.unsqueeze(L2_at_vertices, -1)  # [*, 1, V] * [*, V, 1]
    )  # [*, V, V]
    coeffs = coeffs.reshape((*coeffs.shape[:-2], -1))  # [*, V*(V-1)/2] including diagonal (corresponding to vertex coeffs)

    low = translator.min(coeffs, dim=-1)
    high = translator.max(coeffs, dim=-1)

    return low, high
