import numpy as np

from ..certification_results import AugmentedSample
from ..regions import HyperrectangularRegion, SimplicialRegion
from ..translators import TaylorTranslator


def first_order_certified_taylor_expansion(dynamics, expansion_point, delta, numeric_translator=None):
    """
    A 1st-order Taylor expansion including residual (certified) of a function around a point.

    This is computed using the TaylorTranslator implementation.

    :param dynamics: An object representing the dynamics to be expanded.
    :param expansion_point: The point around which to expand the dynamics.
    :param delta: The (hyperrectangular) radius of the expansion.
    :return: CertifiedFirstOrderTaylorExpansion object representing f(x) = f(c) + ∇f(c)(x-c) + R(x)
    """
    translator = TaylorTranslator(numeric_translator=numeric_translator)

    # Define the domain bounds
    lower_bounds = expansion_point - delta
    upper_bounds = expansion_point + delta

    # Create the initial Taylor expansion for the identity function f(x) = x
    x = translator.to_format(expansion_point, lower_bounds, upper_bounds)

    # Compute the dynamics using the Taylor translator
    y = dynamics.compute_dynamics(x, translator)

    return y


def first_order_certified_taylor_expansion_simplex(dynamics, expansion_point, simplex_vertices, numeric_translator=None):
    """
    A 1st-order Taylor expansion including residual (certified) of a function around a point
    for a simplicial domain.

    This is computed using the TaylorTranslator implementation with simplicial bounds.

    :param dynamics: An object representing the dynamics to be expanded.
    :param expansion_point: The point around which to expand the dynamics.
    :param simplex_vertices: The vertices of the simplex defining the domain.
    :return: CertifiedFirstOrderTaylorExpansion object representing f(x) = f(c) + ∇f(c)(x-c) + R(x)
    """
    translator = TaylorTranslator(numeric_translator=numeric_translator)

    # Create the initial Taylor expansion for the identity function f(x) = x with simplicial domain
    x = translator.to_format_simplex(expansion_point, simplex_vertices)

    # Compute the dynamics using the Taylor translator
    y = dynamics.compute_dynamics(x, translator)

    return y


class TaylorLinearization:
    """
    Taylor linearization using the Python taylor translator implementation.
    """

    def __init__(self, dynamics, numeric_translator=None):
        self.dynamics = dynamics
        self.translator = numeric_translator

    def linearize(self, samples):
        """
        Linearizes a batch of samples using Taylor expansion.

        :param samples: List of samples to linearize
        :return: List of AugmentedSample objects with linearization information
        """
        return [self.linearize_sample(sample) for sample in samples]

    def linearize_sample(self, sample):
        """
        Linearize a single sample using certified Taylor expansion.

        Supports both hyperrectangular and simplicial regions.

        :param sample: Sample object (HyperrectangularRegion, SimplicialRegion, or AugmentedSample)
        :return: AugmentedSample with linearization bounds
        """
        # Handle AugmentedSample wrapper - extract the actual region
        actual_region = sample.region if hasattr(sample, "region") else sample

        # Determine region type and compute Taylor expansion accordingly
        if isinstance(actual_region, SimplicialRegion):
            # SimplicialRegion - use simplicial Taylor expansion
            vertices = self.translator.to_format(actual_region.vertices)
            center = self.translator.to_format(actual_region.centroid)  # Use centroid as expansion point

            taylor_expansion = first_order_certified_taylor_expansion_simplex(self.dynamics, center, vertices, self.translator)
        elif isinstance(actual_region, HyperrectangularRegion):
            # HyperrectangularRegion - use hyperrectangular Taylor expansion
            center = self.translator.to_format(actual_region.centroid)
            radius = self.translator.to_format(actual_region.radius_vec)

            taylor_expansion = first_order_certified_taylor_expansion(self.dynamics, center, radius, self.numeric_translator)
        else:
            # Unsupported region type
            raise TypeError(f"Unsupported region type: {type(actual_region)}. " f"Expected SimplicialRegion or HyperrectangularRegion.")

        # Extract components for the specific output dimension
        output_idx = actual_region.output_dim

        # Get the Jacobian (gradient) and function value at center
        jacobian, f_c = taylor_expansion.linear_approximation
        remainder_lower, remainder_upper = taylor_expansion.remainder
        expansion_point = taylor_expansion.expansion_point

        # Extract values for the specific output dimension
        if jacobian.ndim > 1 and output_idx is not None:
            # Multi-dimensional output
            df_c = jacobian[output_idx]  # Gradient for this output dimension
            f_c_val = f_c[output_idx]  # Function value for this output dimension
            r_lower = remainder_lower[output_idx]  # Lower remainder bound
            r_upper = remainder_upper[output_idx]  # Upper remainder bound
        elif jacobian.ndim > 1:
            # All output dimensions
            df_c = jacobian
            f_c_val = f_c
            r_lower = remainder_lower
            r_upper = remainder_upper
        else:
            # Single-dimensional output
            df_c = jacobian.flatten()
            f_c_val = f_c.item() if hasattr(f_c, "item") else f_c
            r_lower = remainder_lower.item() if hasattr(remainder_lower, "item") else remainder_lower
            r_upper = remainder_upper.item() if hasattr(remainder_upper, "item") else remainder_upper

        # Construct affine bounds: f(x) ≈ f(c) + ∇f(c)·(x - c) + R
        # In affine form: A·x + b where A = ∇f(c), b = f(c) - ∇f(c)·c + R

        for _ in range(df_c.ndim - 3):
            expansion_point = self.translator.unsqueeze(expansion_point, dim=-2)

        # Upper bound: A_upper·x + b_upper
        A_upper = df_c
        b_upper = f_c_val - self.translator.matrix_vector(df_c, expansion_point) + r_upper

        # Lower bound: A_lower·x + b_lower
        A_lower = df_c
        b_lower = f_c_val - self.translator.matrix_vector(df_c, expansion_point) + r_lower

        # Maximum gap between upper and lower bounds
        max_gap = r_upper - r_lower

        return AugmentedSample.from_certification_region(actual_region, ((A_lower, b_lower), (A_upper, b_upper), max_gap))

    def get_taylor_expansion(self, expansion_point, delta):
        """
        Get the full Taylor expansion object for analysis.

        :param expansion_point: Point around which to expand
        :param delta: Radius of expansion domain
        :return: CertifiedFirstOrderTaylorExpansion object
        """
        return first_order_certified_taylor_expansion(self.dynamics, expansion_point, delta)

    def evaluate_at_point(self, point):
        """
        Evaluate the dynamics at a specific point.

        :param point: Point at which to evaluate
        :return: Function value at the point
        """
        # Create a trivial expansion at the point
        delta = np.zeros_like(point)
        expansion = first_order_certified_taylor_expansion(self.dynamics, point, delta)

        # Return the function value (constant term)
        return expansion.linear_approximation[1]

    def get_jacobian_at_point(self, point):
        """
        Get the Jacobian matrix at a specific point.

        :param point: Point at which to compute Jacobian
        :return: Jacobian matrix
        """
        # Create a small expansion around the point
        delta = np.full_like(point, 1e-8)
        expansion = first_order_certified_taylor_expansion(self.dynamics, point, delta)

        # Return the Jacobian (linear term)
        return expansion.linear_approximation[0]
