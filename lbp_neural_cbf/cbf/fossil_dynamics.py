from typing import Union

import numpy as np
import torch

from ..translators import NumpyTranslator, TorchTranslator
from .cbf_dynamics import CBFDynamicalSystem
from .domain import BoxDomain, BoxExteriorDomain, CircleDomain, ComplementDomain, Domain, ProductDomain, SetMinusDomain, UnionDomain, parse_domain_definition


class Barrier1UnsafeDomain(Domain):
    """
    Defines the unsafe domain for the Barrier 1 system:
    Unsafe set: x1 + x2^2 <= 0
    """

    def __init__(self, left_bound=-float("inf")):
        super().__init__(2)  # 2D system

        self.left_bound = left_bound  # Left bound for x1 to avoid infinite volume issues

    def contains(self, x: np.ndarray, translator=None) -> Union[torch.Tensor, np.ndarray]:
        """Check if point x is in the unsafe set."""
        if translator is None:
            translator = NumpyTranslator()

        x = translator.to_format(x)

        x1, x2 = x[..., 0], x[..., 1]
        return (x1 + x2**2 <= 0.0) & (x1 >= self.left_bound)

    def constraint(self, x: np.ndarray, translator=None) -> np.ndarray:
        """
        Constraint computation: h(x) = -x1 - x2^2 (positive inside set).

        WARN: Doesn't work for x1 < left_bound.
        """
        if translator is None:
            translator = NumpyTranslator()

        x1, x2 = x[..., 0], x[..., 1]
        return -x1 - translator.pow(x2, 2)

    def volume(self) -> float:
        """Volume of the unsafe set (infinite for this unbounded set)."""
        if self.left_bound == -float("inf"):
            return float("inf")
        else:
            # Volume of the left-pointing parabolic region x1 + x2^2 <= 0 with x1 >= left_bound
            # Equivalent to area under curve x2 = -x1^2 + left_bound in the positive region,
            # which is from -sqrt(-left_bound) to sqrt(-left_bound).
            # This definite integral evaluates to (4/3)(-left_bound)^(3/2).
            return (4 / 3) * (-self.left_bound) ** 1.5

    def sample_points(self, num_points: int, device=None, use_torch=False, **kwargs):
        """
        Sample points uniformly from the unsafe set.

        Note: Since the unsafe set is unbounded, we restrict sampling to x1 >= left_bound.

        Args:
            num_points: Number of points to sample
            device: PyTorch device (cuda/cpu) for tensor generation
            use_torch: If True, return PyTorch tensor; if False, return NumPy array
        Returns:
            Tensor or array of sampled points with shape [num_points, 2]
        """
        if num_points <= 0:
            if use_torch:
                return torch.empty((0, self.dim), device=device)
            return np.empty((0, self.dim))

        if self.left_bound == -float("inf"):
            raise ValueError("Cannot sample uniformly from unbounded unsafe set (without left_bound).")

        # Create box domain for sampling
        x1_min = self.left_bound
        x1_max = 0.0  # Since x1 + x2^2 <= 0 implies x1 <= 0
        x2_bound = np.sqrt(-self.left_bound)  # Max |x2| when x1 = left_bound
        x2_min = -x2_bound
        x2_max = x2_bound

        box_domain = BoxDomain([[x1_min, x1_max], [x2_min, x2_max]])
        samples = box_domain.sample_points(num_points * 10, device=device, use_torch=use_torch)  # Oversample

        # Filter samples to only those in the unsafe set
        x1, x2 = samples[:, 0], samples[:, 1]
        mask = x1 + x2**2 <= 0.0
        samples = samples[mask]

        return samples[:num_points]  # Return only the requested number of points

    def intersects_hyperrect(self, center, radius):
        """
        Check if the unsafe set intersects with a given hyperrectangular region.

        Args:
            center: Center of the hyperrectangle [c1, c2]
            radius: Half-widths of the hyperrectangle [r1, r2]
        Returns:
            True if intersects, False otherwise
        """
        c1, c2 = center[0], center[1]
        r1, r2 = radius[0], radius[1]

        # Find closest point in hyperrectangle to the unsafe set boundary
        closest_x1 = np.clip(-(c2**2), c1 - r1, c1 + r1)
        closest_x2 = np.clip(0.0, c2 - r2, c2 + r2)

        # Check if this point is in the unsafe set
        return (closest_x1 + closest_x2**2) <= 0.0

    def contains_hyperrect(self, center, radius):
        """
        Check if the hyperrectangular region is entirely contained within the unsafe set.

        Args:
            center: Center of the hyperrectangle [c1, c2]
            radius: Half-widths of the hyperrectangle [r1, r2]
        Returns:
            True if contained, False otherwise
        """
        c1, c2 = center[0], center[1]
        r1, r2 = radius[0], radius[1]

        # Check all corners of the hyperrectangle
        corners = [
            (c1 - r1, c2 - r2),
            (c1 - r1, c2 + r2),
            (c1 + r1, c2 - r2),
            (c1 + r1, c2 + r2),
        ]

        for x1, x2 in corners:
            if (x1 + x2**2) > 0.0:
                return False  # Found a corner outside unsafe set

        return True  # All corners inside unsafe set

    def intersects_simplex(self, vertices):
        """
        Check if the unsafe set intersects with a given simplicial region.

        Args:
            vertices: Vertices of the simplex, shape [num_vertices, 2]
        Returns:
            True if intersects, False otherwise
        """
        # closest point on simplex to unsafe set boundary - edge by edge
        num_vertices = vertices.shape[0]
        for i in range(num_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % num_vertices]
            edge_vec = v2 - v1

            # Project unsafe set boundary point onto edge
            t = -(v1[0] + v1[1] ** 2) * np.dot(edge_vec, np.array([1, 2 * v1[1]])) / np.dot(edge_vec, edge_vec)
            t = np.clip(t, 0.0, 1.0)
            closest_point = v1 + t * edge_vec

            # Check if closest point is in unsafe set
            if (closest_point[0] + closest_point[1] ** 2) <= 0.0:
                return True  # Intersection found

        return False  # No intersection found

    def contains_simplex(self, vertices):
        """
        Check if the simplicial region is entirely contained within the unsafe set.

        Args:
            vertices: Vertices of the simplex, shape [num_vertices, 2]
        Returns:
           True if contained, False otherwise
        """
        for v in vertices:
            if (v[0] + v[1] ** 2) > 0.0:
                return False  # Found a vertex outside unsafe set
        return True  # All vertices inside unsafe set


# Darboux System in https://arxiv.org/pdf/2310.09360
class Barrier1System(CBFDynamicalSystem):
    """
    Barrier 1 dynamical system (Darboux) without control for testing CBF verification.

    Dynamics: dx/dt = f(x) where:
    - f(x) = [x2 + 2 * x1 * x2, -x1 - x2**2 + 2 * x1**2] (nonlinear drift)
    """

    def __init__(self, alpha=1.0, safe_set=None):
        super().__init__()
        self.system_name = "barr1"
        self.input_dim = 2
        self.output_dim = 2
        self.control_dim = 0
        self.alpha = alpha

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-2, 2], [-2, 2]])

        # Parse unsafe set into domain object
        self.unsafe_set_interior = Barrier1UnsafeDomain(left_bound=-2)
        self.unsafe_set_exterior = BoxExteriorDomain(self.input_domain.bounds)
        self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        # Define safe set (if provided, otherwise it's the complement of unsafe within domain)
        if safe_set is not None:
            self.safe_set_def = safe_set
            self.safe_set = parse_domain_definition(safe_set, self.input_domain.bounds)
        else:
            # Safe set is the complement of unsafe set within the input domain
            self.safe_set = ComplementDomain(self.unsafe_set_interior, self.input_domain.bounds)

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [128, 256, 128]  # Hidden layer sizes for neural network
        # self.activation_fnc = "Tanh"
        self.activation_fnc = "Relu"

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5])  # Reasonable grid spacing for 2D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            u: Should be None since control_dim=0
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 2]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)
        return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [x2 + 2 * x1 * x2, -x1 - x2**2 + 2 * x1**2]

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [2] or [batch_size, 2]
        """
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = x2 + 2 * x1 * x2
        dx2 = -x1 - translator.pow(x2, 2) + 2 * translator.pow(x1, 2)
        return translator.stack([dx1, dx2], dim=-1)

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the domain object to compute the constraint
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """
        Property to expose the unsafe domain.

        Returns:
            Domain: The unsafe domain object.
        """
        return self.unsafe_set_interior


class Barrier2System(CBFDynamicalSystem):
    """
    Barrier 2 dynamical system without control for testing CBF verification.

    Dynamics: dx/dt = f(x) where:
    - f(x) = [exp(-x) + y - 1, -((sin(x)) ** 2)] (nonlinear drift)
    """

    def __init__(self, alpha=1.0, safe_set=None):
        super().__init__()
        self.system_name = "barr2"
        self.input_dim = 2
        self.output_dim = 2
        self.control_dim = 0
        self.alpha = alpha

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-2, 2], [-2, 2]])

        # Parse unsafe set into domain object
        self.unsafe_set_interior = CircleDomain(center=[0.7, -0.7], radius=0.3)
        self.unsafe_set_exterior = BoxExteriorDomain(self.input_domain.bounds)
        self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        # Define safe set (if provided, otherwise it's the complement of unsafe within domain)
        if safe_set is not None:
            self.safe_set_def = safe_set
            self.safe_set = parse_domain_definition(safe_set, self.input_domain.bounds)
        else:
            # Safe set is the complement of unsafe set within the input domain
            self.safe_set = ComplementDomain(self.unsafe_set_interior, self.input_domain.bounds)

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [64, 64]  # Hidden layer sizes for neural network

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5])  # Reasonable grid spacing for 2D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            u: Should be None since control_dim=0
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 2]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)
        return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [exp(-x) + y - 1, -(sin(x) ** 2)]

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [2] or [batch_size, 2]
        """
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = translator.exp(-x1) + x2 - 1
        dx2 = -translator.pow(translator.sin(x1), 2)
        return translator.stack([dx1, dx2], dim=-1)

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the domain object to compute the constraint
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """
        Property to expose the unsafe domain.

        Returns:
            Domain: The unsafe domain object.
        """
        return self.unsafe_set_interior


class LShapeDomain(UnionDomain):
    """Union of multiple domains."""

    def __init__(self, rect1: BoxDomain, rect2: BoxDomain):
        assert isinstance(rect1, BoxDomain)
        assert isinstance(rect2, BoxDomain)

        # Check that rect1 and rect2 share one corner - otherwise they don't form an L-shape
        for v1 in rect1.corners():
            for v2 in rect2.corners():
                if np.allclose(v1, v2):
                    break
            else:
                continue
            break
        else:
            raise ValueError("LShapeDomain: rect1 and rect2 must share one corner point.")

        domains = [rect1, rect2]
        super().__init__(domains)

        assert self.dim == 2, "LShapeDomain only supports 2D domains."

    def contains_hyperrect(self, center: np.ndarray, radius: np.ndarray) -> bool:
        """
        Union contains if any of the two rects completely contains the hyperrect.

        Raises NotImplementedError if the union domains intersect/touch and the hyperrect
        is not fully contained in a single domain (ambiguous case).
        """
        return any(domain.contains_hyperrect(center, radius) for domain in self.domains)

    def contains_simplex(self, vertices: np.ndarray) -> bool:
        """
        Union contains if any subdomain completely contains the simplex.
        """
        # If any domain fully contains it, we can return True safely
        if any(domain.contains_simplex(vertices) for domain in self.domains):
            return True

        # If any vertex is outside all domains, we can return False safely
        for v in vertices:
            if not any(domain.contains(v) for domain in self.domains):
                return False

        # Traverse each edge of the simplex and check if it crosses domain boundaries of rect1
        # If it does, check if the crossing point is contained in rect2
        rect1 = self.domains[0]

        for i in range(vertices.shape[0]):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % vertices.shape[0]]

            if rect1.contains(v1) and rect1.contains(v2):
                # Edge fully in rect1
                continue

            edge_vec = v2 - v1

            # Check for intersection with rect1 edges
            for dim in [0, 1]:
                for bound in rect1.bounds[dim]:
                    if edge_vec[dim] == 0:
                        continue  # Parallel to this edge

                    t = (bound - v1[dim]) / edge_vec[dim]
                    if 0.0 <= t <= 1.0:
                        intersection_point = v1 + t * edge_vec
                        # Check if intersection point is in any domain
                        if not any(domain.contains(intersection_point) for domain in self.domains):
                            return False  # Intersection point outside all domains

        return True  # No violations found

    def volume(self) -> float:
        """Volume of the L-shape domain."""
        return sum(domain.volume() for domain in self.domains)  # Not exactly, but close enough for our purposes


class Barrier3System(CBFDynamicalSystem):
    """
    Barrier 3 dynamical system without control for testing CBF verification.

    Dynamics: dx/dt = f(x) where:
    - f(x) = [y, -x - y + 1 / 3 * x**3] (nonlinear drift)
    """

    def __init__(self, alpha=1.0, safe_set=None):
        super().__init__()
        self.system_name = "barr3"
        self.input_dim = 2
        self.output_dim = 2
        self.control_dim = 0
        self.alpha = alpha

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-3, 2.5], [-2, 1]])

        # Parse unsafe set into domain object
        self.unsafe_set_interior = UnionDomain(
            [CircleDomain(center=[-1, -1], radius=0.4), LShapeDomain(BoxDomain([[0.4, 0.6], [0.1, 0.5]]), BoxDomain([[0.4, 0.8], [0.1, 0.3]]))],
            known_separated=True,
        )
        self.unsafe_set_exterior = BoxExteriorDomain(self.input_domain.bounds)
        self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        # Define safe set (if provided, otherwise it's the complement of unsafe within domain)
        if safe_set is not None:
            self.safe_set_def = safe_set
            self.safe_set = parse_domain_definition(safe_set, self.input_domain)
        else:
            # Safe set is the complement of unsafe set within the input domain
            self.safe_set = ComplementDomain(self.unsafe_set_interior, self.input_domain.bounds)

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [64, 64]  # Hidden layer sizes for neural network

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5])  # Reasonable grid spacing for 2D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            u: Should be None since control_dim=0
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 2]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)
        return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [y, -x - y + 1 / 3 * x**3]

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [2] or [batch_size, 2]
        """
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = x2
        dx2 = -x1 - x2 + (1 / 3) * translator.pow(x1, 3)
        return translator.stack([dx1, dx2], dim=-1)

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the domain object to compute the constraint
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """
        Property to expose the unsafe domain.

        Returns:
            Domain: The unsafe domain object.
        """
        return self.unsafe_set_interior


# Obstacle avoidance system from https://arxiv.org/pdf/2305.16241.pdf
class Barrier4System(CBFDynamicalSystem):
    """
    Barrier 4 dynamical system without control for testing CBF verification.

    Dynamics: dx/dt = f(x) where:
    - f(x) = [v * sin(phi), v * cos(phi), -sin(phi) + 3 * (x * sin(phi) + y * cos(phi)) / (0.5 + x**2 + y**2)] (nonlinear drift)
    - v = 1.0 (constant speed)
    """

    def __init__(self, alpha=1.0, safe_set=None):
        super().__init__()
        self.system_name = "barr4"
        self.input_dim = 3
        self.output_dim = 3
        self.control_dim = 0
        self.alpha = alpha

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-2, 2], [-2, 2], [-1.57, 1.57]])

        # Parse unsafe set into domain object
        self.unsafe_set_interior = ProductDomain([CircleDomain(center=[0.0, 0.0], radius=0.2), BoxDomain([[-1.57, 1.57]])])
        self.unsafe_set_exterior = BoxExteriorDomain(self.input_domain.bounds)
        self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        # Define safe set (if provided, otherwise it's the complement of unsafe within domain)
        if safe_set is not None:
            self.safe_set_def = safe_set
            self.safe_set = parse_domain_definition(safe_set, self.input_domain)
        else:
            # Safe set is the complement of unsafe set within the input domain
            self.safe_set = ComplementDomain(self.unsafe_set_interior, self.input_domain.bounds)

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [64, 64]  # Hidden layer sizes for neural network

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5, 0.5])  # Reasonable grid spacing for 2D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x, y, phi] with shape [3] or [batch_size, 3]
            u: Should be None since control_dim=0
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 3]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)
        return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [v * sin(phi), v * cos(phi), -sin(phi) + 3 * (x * sin(phi) + y * cos(phi)) / (0.5 + x**2 + y**2)]

        Args:
            x: State [x, y, phi] with shape [3] or [batch_size, 3]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [3] or [batch_size, 3]
        """
        x, y, phi = x[..., 0], x[..., 1], x[..., 2]
        dx = 1.0 * translator.sin(phi)
        dy = 1.0 * translator.cos(phi)
        dphi = -translator.sin(phi) + 3 * (x * translator.sin(phi) + y * translator.cos(phi)) / (0.5 + translator.pow(x, 2) + translator.pow(y, 2))
        return translator.stack([dx, dy, dphi], dim=-1)

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 3]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the domain object to compute the constraint
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """
        Property to expose the unsafe domain.

        Returns:
            Domain: The unsafe domain object.
        """
        return self.unsafe_set_interior


class HighOrd2System(CBFDynamicalSystem):
    """
    Higher order 2 dynamical system without control for testing CBF verification.

    Dynamics: dx/dt = f(x) + g(x)u where:
    - f(x) = [x2, -1.62212*x1  -2.206*x2] (nonlinear drift)
    - g(x) = [[0], [1]] (control input matrix)
    - u ∈ [-2 * 2.5, 2 * 2.5] (bounded control input)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.system_name = "hiord2"
        self.input_dim = 2
        self.output_dim = 2
        self.control_dim = 1
        self.alpha = alpha

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-2.5, 2.5], [-2.5, 2.5]])
        self.circle_domain = CircleDomain(center=[0.0, 0.0], radius=2.0)

        # Parse unsafe set into domain object
        self.unsafe_set_interior = ComplementDomain(self.circle_domain, self.input_domain.bounds)
        # self.unsafe_set_interior = CircleDomain(center=[-2.0, -2.0], radius=0.4)
        # self.unsafe_set_exterior = ComplementDomain(self.circle_domain, self.input_domain.bounds)
        # self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        self.safe_set = self.circle_domain

        self.control_bounds = 9.5703
        self.u_min = np.array([-self.control_bounds])
        self.u_max = np.array([self.control_bounds])

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [64, 64]  # Hidden layer sizes for neural network

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5])  # Reasonable grid spacing for 4D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            u: Control input [u1] with shape [1] or [batch_size, 1] (optional)
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 2]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)

        if u is not None:
            g_x = self.compute_g(x, translator)
            g_u = translator.matrix_vector(g_x, u)
            return f_x + g_u
        else:
            return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [x2, -1.62212*x1  -2.206*x2]

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [2] or [batch_size, 2]
        """
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = x2
        dx2 = -1.62212 * x1 - 2.206 * x2
        return translator.stack([dx1, dx2], dim=-1)

    def compute_g(self, x, translator):
        """
        Compute control input matrix g(x).

        For this system, g(x) = [[0], [1]].

        Args:
            x: State [x1, x2] with shape [batch_size, 2] or [2]
            translator: Mathematical operations translator

        Returns:
            Control matrix with shape [batch_size, 1, 2] for batched inputs or [1, 2] for single inputs
            For TaylorTranslator: returns TaylorExpansion for state-dependent elements
        """
        x2 = x[..., 1]
        zero = translator.zeros_like(x2)
        one = translator.ones_like(x2)

        g_x = translator.unsqueeze(translator.stack([zero, one], dim=-1), dim=-2)

        return g_x

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 4]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the domain object to compute the constraint
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """
        Property to expose the unsafe domain.

        Returns:
            Domain: The unsafe domain object.
        """
        return self.unsafe_set_interior


class HighOrd4System(CBFDynamicalSystem):
    """
    Higher order 4 dynamical system without control for testing CBF verification.

    Dynamics: dx/dt = f(x) + g(x)u where:
    - f(x) = [x2, x3, x4, -5.10018*x1  -10.8641*x2  -9.91781*x3  -4.6946*x4] (nonlinear drift)
    - g(x) = [[0], [0], [0], [1]] (control input matrix)
    - u ∈ [-3980 * 2.5, 3980 * 2.5] (bounded control input)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.system_name = "hiord4"
        self.input_dim = 4
        self.output_dim = 4
        self.control_dim = 1
        self.alpha = alpha

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]])
        self.circle_domain = CircleDomain(center=[0.0, 0.0, 0.0, 0.0], radius=2.0)

        # Parse unsafe set into domain object
        self.unsafe_set_interior = ComplementDomain(self.circle_domain, self.input_domain.bounds)
        # self.unsafe_set_interior = CircleDomain(center=[-2.0, -2.0, -2.0, -2.0], radius=0.4)
        # self.unsafe_set_exterior = ComplementDomain(self.circle_domain, self.input_domain.bounds)
        # self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        self.safe_set = self.circle_domain

        self.control_bounds = 76.441725
        self.u_min = np.array([-self.control_bounds])
        self.u_max = np.array([self.control_bounds])

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [64, 64]  # Hidden layer sizes for neural network

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5, 0.5, 0.5])  # Reasonable grid spacing for 4D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2, x3, x4] with shape [4] or [batch_size, 4]
            u: Should be None since control_dim=0
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 4]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)

        if u is not None:
            g_x = self.compute_g(x, translator)
            g_u = translator.matrix_vector(g_x, u)
            return f_x + g_u
        else:
            return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [x2, x3, x4, -5.10018*x1  -10.8641*x2  -9.91781*x3  -4.6946*x4]

        Args:
            x: State [x1, x2, x3, x4] with shape [4] or [batch_size, 4]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [4] or [batch_size, 4]
        """
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        dx1, dx2, dx3 = x2, x3, x4
        dx4 = -5.10018 * x1 - 10.8641 * x2 - 9.91781 * x3 - 4.6946 * x4
        return translator.stack([dx1, dx2, dx3, dx4], dim=-1)

    def compute_g(self, x, translator):
        """
        Compute control input matrix g(x).

        For this system, g(x) = [[0], [0], [0], [1]].

        Args:
            x: State [x1, x2, x3, x4] with shape [batch_size, 4] or [4]
            translator: Mathematical operations translator

        Returns:
            Control matrix with shape [batch_size, 1, 4] for batched inputs or [1, 4] for single inputs
            For TaylorTranslator: returns TaylorExpansion for state-dependent elements
        """
        x4 = x[..., 3]
        zero = translator.zeros_like(x4)
        one = translator.ones_like(x4)

        g_x = translator.unsqueeze(translator.stack([zero, zero, zero, one], dim=-1), dim=-2)

        return g_x

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 4]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the domain object to compute the constraint
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """
        Property to expose the unsafe domain.

        Returns:
            Domain: The unsafe domain object.
        """
        return self.unsafe_set_interior


class HighOrd6System(CBFDynamicalSystem):
    """
    Higher order 6 dynamical system with affine control for testing CBF verification.

    Dynamics: dx/dt = f(x) + g(x)u where:
    - f(x) = [x2, x3, x4, x5, x6, -26.8334*x1  -72.4487*x2  -88.9126*x3  -64.016*x4  -28.8069*x5  -7.7778*x6] (drift)
    - g(x) = [[0], [0], [0], [0], [0], [1]] (control input matrix)
    - u ∈ [-800 * 2.5, 800 * 2.5] (bounded control input)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.system_name = "hiord6"
        self.input_dim = 6
        self.output_dim = 6
        self.control_dim = 0
        self.alpha = alpha

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]])
        self.circle_domain = CircleDomain(center=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], radius=2.0)

        # Parse unsafe set into domain object
        self.unsafe_set_interior = ComplementDomain(self.circle_domain, self.input_domain.bounds)
        # self.unsafe_set_interior = CircleDomain(center=[-2.0, -2.0, -2.0, -2.0], radius=0.4)
        # self.unsafe_set_exterior = ComplementDomain(self.circle_domain, self.input_domain.bounds)
        # self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        self.safe_set = self.circle_domain

        self.control_bounds = 721.9885  # Max control based on state bounds
        self.u_min = np.array([-self.control_bounds])
        self.u_max = np.array([self.control_bounds])

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [64, 64]  # Hidden layer sizes for neural network

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Reasonable grid spacing for 6D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2, x3, x4, x5, x6] with shape [6] or [batch_size, 6]
            u: Control input [u1] with shape [1] or [batch_size, 1] (optional)
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 6]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)

        if u is not None:
            g_x = self.compute_g(x, translator)
            g_u = translator.matrix_vector(g_x, u)
            return f_x + g_u
        else:
            return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [x2, x3, x4, x5, x6, -26.8334*x1  -72.4487*x2  -88.9126*x3  -64.016*x4  -28.8069*x5  -7.7778*x6]

        Args:
            x: State [x1, x2, x3, x4, x5, x6] with shape [6] or [batch_size, 6]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [6] or [batch_size, 6]
        """
        x1, x2, x3, x4, x5, x6 = x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4], x[..., 5]
        dx1, dx2, dx3, dx4, dx5 = x2, x3, x4, x5, x6
        dx6 = -26.8334 * x1 - 72.4487 * x2 - 88.9126 * x3 - 64.016 * x4 - 28.8069 * x5 - 7.7778 * x6
        return translator.stack([dx1, dx2, dx3, dx4, dx5, dx6], dim=-1)

    def compute_g(self, x, translator):
        """
        Compute control input matrix g(x).

        For this system, g(x) = [[0], [0], [0], [0], [0], [1]].

        Args:
            x: State [x1, x2, x3, x4, x5, x6] with shape [batch_size, 6] or [6]
            translator: Mathematical operations translator

        Returns:
            Control matrix with shape [batch_size, 6, 1] for batched inputs or [6, 1] for single inputs
            For TaylorTranslator: returns TaylorExpansion for state-dependent elements
        """
        x6 = x[..., 5]
        zero = translator.zeros_like(x6)
        one = translator.ones_like(x6)

        g_x = translator.unsqueeze(translator.stack([zero, zero, zero, zero, zero, one], dim=-1), dim=-2)

        return g_x

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 6]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the domain object to compute the constraint
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """
        Property to expose the unsafe domain.

        Returns:
            Domain: The unsafe domain object.
        """
        return self.unsafe_set_interior


class HighOrd8System(CBFDynamicalSystem):
    """
    Higher order 8 dynamical system with affine control for testing CBF verification.

    Dynamics: dx/dt = f(x) + g(x)u where:
    - f(x) = [x2, x3, x4, x5, x6, x7, x8, -198.776*x1  -630.116*x2  -932.147*x3  -844.257*x4  -514.673*x5  -217.537*x6  -62.6909*x7  -11.3562*x8] (drift)
    - g(x) = [[0], [0], [0], [0], [0], [0], [0], [1]] (control input matrix)
    - u ∈ [-20 * 2.5, 20 * 2.5] (bounded control input)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.system_name = "hiord8"
        self.input_dim = 8
        self.output_dim = 8
        self.control_dim = 1
        self.alpha = alpha

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]])
        self.circle_domain = CircleDomain(center=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], radius=2.0)

        # Parse unsafe set into domain object
        self.unsafe_set_interior = ComplementDomain(self.circle_domain, self.input_domain.bounds)
        # self.unsafe_set_interior = CircleDomain(center=[-2.0, -2.0, -2.0, -2.0], radius=0.4)
        # self.unsafe_set_exterior = ComplementDomain(self.circle_domain, self.input_domain.bounds)
        # self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        self.safe_set = self.circle_domain

        self.control_bounds = 20 * self.input_domain.bounds[-1][1]  # Max control based on state bounds
        self.u_min = np.array([-self.control_bounds])
        self.u_max = np.array([self.control_bounds])

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [64, 64]  # Hidden layer sizes for neural network

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Reasonable grid spacing for 8D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2, x3, x4, x5, x6, x7, x8] with shape [8] or [batch_size, 8]
            u: Control input [u1] with shape [1] or [batch_size, 1] (optional)
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 8]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)

        if u is not None:
            g_x = self.compute_g(x, translator)
            g_u = translator.matrix_vector(g_x, u)
            return f_x + g_u
        else:
            return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [x2, x3, x4, x5, x6, x7, x8, -198.776*x1  -630.116*x2  -932.147*x3  -844.257*x4  -514.673*x5  -217.537*x6  -62.6909*x7  -11.3562*x8]

        Args:
            x: State [x1, x2, x3, x4, x5, x6, x7, x8] with shape [8] or [batch_size, 8]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [8] or [batch_size, 8]
        """
        x1, x2, x3, x4, x5, x6, x7, x8 = x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4], x[..., 5], x[..., 6], x[..., 7]
        dx1, dx2, dx3, dx4, dx5, dx6, dx7 = x2, x3, x4, x5, x6, x7, x8
        dx8 = -198.776 * x1 - 630.116 * x2 - 932.147 * x3 - 844.257 * x4 - 514.673 * x5 - 217.537 * x6 - 62.6909 * x7 - 11.3562 * x8
        return translator.stack([dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8], dim=-1)

    def compute_g(self, x, translator):
        """
        Compute control input matrix g(x).

        For this system, g(x) = [[0], [0], [0], [0], [0], [0], [0], [1]].

        Args:
            x: State [x1, x2, x3, x4, x5, x6, x7, x8] with shape [batch_size, 8] or [8]
            translator: Mathematical operations translator

        Returns:
            Control matrix with shape [batch_size, 8, 1] for batched inputs or [8, 1] for single inputs
            For TaylorTranslator: returns TaylorExpansion for state-dependent elements
        """
        x8 = x[..., 7]
        zero = translator.zeros_like(x8)
        one = translator.ones_like(x8)

        g_x = translator.unsqueeze(translator.stack([zero, zero, zero, zero, zero, zero, zero, one], dim=-1), dim=-2)

        return g_x

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 8]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the domain object to compute the constraint
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """
        Property to expose the unsafe domain.

        Returns:
            Domain: The unsafe domain object.
        """
        return self.unsafe_set_interior
