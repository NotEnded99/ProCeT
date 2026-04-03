from abc import abstractmethod
from typing import List, Optional

import numpy as np

from ..dynamics import DynamicalSystem
from ..translators import NumpyTranslator, TorchTranslator
from .domain import ApproachConeDomain, BoxDomain, BoxExteriorDomain, ComplementDomain, UnionDomain, parse_domain_definition


class CBFDynamicalSystem(DynamicalSystem):
    """
    Base class for dynamical systems with control barrier function properties.

    DYNAMICS METHODS OVERVIEW:
    -------------------------
    This class provides three methods for computing dynamics, each serving a specific purpose:

    1. compute_dynamics(x, translator, u=None):
       - Full dynamics including control: dx/dt = f(x) + g(x)u
       - Used for: Simulation, evaluation with specific control inputs
       - For autonomous systems (control_dim=0): This is the only dynamics method needed
       - For controlled systems (control_dim>0): Should use compute_f and compute_g internally

    2. compute_f(x, translator):
       - Drift dynamics (uncontrolled): f(x)
       - Used for: CBF training, CBF verification (drift term in Lie derivative)
       - Required for: Systems with control_dim > 0
       - For autonomous systems: Defaults to calling compute_dynamics(x, translator, u=None)

    3. compute_g(x, translator):
       - Control input matrix: g(x)
       - Used for: CBF training, CBF verification (control term optimization)
       - Required for: Systems with control_dim > 0
       - Returns: [state_dim, control_dim] for constant g(x), or [state_dim, control_dim, batch_size] for state-dependent g(x)

    AFFINE CONTROL SYSTEMS:
    ----------------------
    For systems with affine control (control_dim > 0), the dynamics follow:
        dx/dt = f(x) + g(x)u
    where:
        - f(x) is the drift dynamics (uncontrolled)
        - g(x) is the control input matrix
        - u is the control input with bounds u_min <= u <= u_max

    Subclasses must implement:
        - compute_dynamics(x, translator, u): Full dynamics (should use compute_f and compute_g)
        - compute_f(x, translator): Drift dynamics f(x) [REQUIRED if control_dim > 0]
        - compute_g(x, translator): Control matrix g(x) [REQUIRED if control_dim > 0]
        - safe_set_constraint(x, translator): Safe set constraint h(x) >= 0
    """

    def __init__(self) -> None:
        super().__init__()
        self.safe_set: Optional[object] = None  # Definition of the safe set
        self.unsafe_set: Optional[object] = None  # Definition of the unsafe set
        self.control_dim: Optional[int] = None  # Dimension of control input
        self.alpha: Optional[float] = None  # Class K function parameter for CBF condition
        self.u_min: Optional[np.ndarray] = None  # Minimum control bounds
        self.u_max: Optional[np.ndarray] = None  # Maximum control bounds
        self.hidden_sizes = [64, 64, 8]  # Default hidden layer sizes for neural network
        # self.activation_fnc = "Tanh"  # Default activation function for neural network
        self.activation_fnc = "Relu"  

    def __call__(self, x, u=None, translator=None):
        """
        Compute the dynamics for the system with optional control input.

        Args:
            x: The state tensor with shape [input_dim, batch_size]
            u: The control input tensor with shape [control_dim, batch_size] (optional)
            translator: The translator for mathematical operations

        Returns:
            The derivatives of the system with shape [output_dim, batch_size]
        """
        if translator is None:
            if isinstance(x, np.ndarray):
                # Use NumpyTranslator if x is a NumPy array
                translator = NumpyTranslator()
            else:
                translator = TorchTranslator()

        return self.compute_dynamics(x, translator, u)

    @abstractmethod
    def compute_dynamics(self, x, translator, u=None):
        """
        Compute the controlled dynamics dx/dt = f(x) + g(x)u.

        For affine control systems (control_dim > 0), this should use:
            return compute_f(x, translator) + compute_g(x, translator) @ u

        Args:
            x: State tensor with shape [state_dim, batch_size]
            translator: Mathematical operations translator
            u: Control input tensor with shape [control_dim, batch_size] (optional)

        Returns:
            State derivatives with shape [state_dim, batch_size]
        """
        pass

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        This method should be implemented for affine control systems (control_dim > 0).
        It represents the uncontrolled dynamics.

        Args:
            x: State tensor with shape [state_dim, batch_size]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [state_dim, batch_size]

        Raises:
            NotImplementedError: If not implemented for a controlled system
        """
        if self.control_dim > 0:
            raise NotImplementedError(
                f"{self.__class__.__name__} has control_dim={self.control_dim} but does not implement compute_f(). "
                "Affine control systems must implement compute_f() for verification."
            )
        # For uncontrolled systems, f(x) = compute_dynamics(x, translator, u=None)
        return self.compute_dynamics(x, translator, u=None)

    def compute_g(self, x, translator):
        """
        Compute control input matrix g(x).

        This method should be implemented for affine control systems (control_dim > 0).
        For affine systems: dx/dt = f(x) + g(x)u

        Args:
            x: State tensor with shape [state_dim, batch_size]
            translator: Mathematical operations translator

        Returns:
            Control matrix with shape [state_dim, control_dim] or [state_dim, control_dim, batch_size]

        Raises:
            NotImplementedError: If not implemented for a controlled system
        """
        if self.control_dim > 0:
            raise NotImplementedError(
                f"{self.__class__.__name__} has control_dim={self.control_dim} but does not implement compute_g(). "
                "Affine control systems must implement compute_g() for verification."
            )
        # For uncontrolled systems, there is no g(x)
        return None

    @abstractmethod
    def safe_set_constraint(self, x, translator):
        """
        Define the safe set constraint h(x) >= 0.

        Args:
            x: State tensor with shape [state_dim, batch_size]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [1, batch_size]
        """
        pass

    def alpha_function(self, h, translator=None):
        """
        Class K function α(h) for CBF condition.
        Default implementation: α(h) = self.alpha * h

        Args:
            h: Barrier function values
            translator: Mathematical operations translator

        Returns:
            α(h) values
        """
        if self.alpha is None:
            self.alpha = 1.0
        return self.alpha * h

    def cbf_condition(self, x, u_bounds, h, grad_h, translator):
        """
        Compute the CBF condition over the range of control input bounds.

        Args:
            x: State vector
            u_bounds: Tuple (u_min, u_max) representing control input bounds
            h: Barrier function values
            grad_h: Gradient of the barrier function
            translator: Mathematical operations translator

        Returns:
            Tuple (cbf_min, cbf_max) representing the minimum and maximum values of the CBF condition.
        """
        u_min, u_max = u_bounds

        # Compute dynamics for minimum and maximum control inputs
        f_min = self.compute_dynamics(x, translator=translator, u=u_min)
        f_max = self.compute_dynamics(x, translator=translator, u=u_max)

        # Compute CBF condition for minimum and maximum dynamics
        cbf_min = translator.matrix_vector(grad_h, f_min) + self.alpha_function(h, translator)
        cbf_max = translator.matrix_vector(grad_h, f_max) + self.alpha_function(h, translator)

        return cbf_min, cbf_max


class Simple2DSystem(CBFDynamicalSystem):
    """
    Simple 2D dynamical system with affine control for testing CBF verification.

    Dynamics: dx/dt = f(x) + g(x)u where:
    - f(x) = [-x1*x2, -x2^2] (nonlinear drift)
    - g(x) = I (identity, direct actuation)
    - u ∈ [-0.5, 0.5]^2 (bounded control input)
    """

    def __init__(self, alpha=1.0, control_bounds=0.5, unsafe_set=None, safe_set=None):
        super().__init__()
        self.system_name = "simple_2d"
        self.input_dim = 2
        self.output_dim = 2
        self.control_dim = 2
        self.alpha = alpha

        # Control bounds: u ∈ [-control_bounds, control_bounds]^2
        self.control_bounds = control_bounds
        self.u_min = np.array([-control_bounds, -control_bounds])
        self.u_max = np.array([control_bounds, control_bounds])

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-3, 3], [-2, 2]])

        # Define unsafe set (obstacles to avoid)
        if unsafe_set is None:
            # Default: single circular obstacle
            # self.unsafe_set_def = {"type": "circle", "center": [1.5, 0.0], "radius": 0.3}
            self.unsafe_set_def = {
                                "type": "union",
                                "regions": [
                                    {"type": "circle", "center": [1.5, 0.0], "radius": 0.3},   # 原有的
                                    {"type": "circle", "center": [-1.0, -0.5], "radius": 0.6}, # 新增的
                                ]
                            }
        else:
            self.unsafe_set_def = unsafe_set

        # Parse unsafe set into domain object
        self.unsafe_set_interior = parse_domain_definition(self.unsafe_set_def, self.input_domain)
        self.unsafe_set_exterior = BoxExteriorDomain(self.input_domain.bounds)
        self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        # Define safe set (if provided, otherwise it's the complement of unsafe within domain)
        if safe_set is not None:
            self.safe_set_def = safe_set
            self.safe_set = parse_domain_definition(safe_set, self.input_domain)
        else:
            # Safe set is the complement of unsafe set within the input domain
            self.safe_set_def = {"type": "complement", "of": self.unsafe_set_def, "within_domain": True}
            self.safe_set = ComplementDomain(self.unsafe_set_interior, self.input_domain.bounds)

        # Default translator for verification
        self.translator = NumpyTranslator()

        # self.hidden_sizes = [64, 64, 8]  # Hidden layer sizes for neural network
        self.hidden_sizes = [32, 64, 32]  # Hidden layer sizes for neural network

        # Delta for region generation (half-width of hyperrectangles)
        self.delta = np.array([0.5, 0.5])  # Reasonable grid spacing for 2D system

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            u: Control input [u1, u2] with shape [2] or [batch_size, 2] (optional)
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [batch_size, 2]
        """
        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)

        # Add control term g(x)u if provided
        if u is not None:
            g_x = self.compute_g(x, translator)
            g_u = translator.matrix_vector(g_x, u)
            return f_x + g_u
        else:
            return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [-x1*x2, -x2^2]

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [2] or [batch_size, 2]
        """
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = -x1 * x2
        dx2 = -translator.pow(x2, 2)
        return translator.stack([dx1, dx2], dim=-1)

    def compute_g(self, x, translator):
        """
        Compute control input matrix g(x).

        For this system, g(x) = I (identity matrix, constant).

        Args:
            x: State [x1, x2] with shape [2] or [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Control matrix with shape [2, 2] or [batch_size, 2, 2]
            For TaylorTranslator: returns constant value (not state-dependent)
        """
        return translator.eye_like(x)

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.
        This represents the actual safe region that the system must stay within.

        Args:
            x: State tensor with shape [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [ batch_size] where positive means safe
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


class StateDependentControl2DSystem(CBFDynamicalSystem):
    """
    2D dynamical system with state-dependent control matrix for testing CBF verification.

    Dynamics: dx/dt = f(x) + g(x)u where:
    - f(x) = [-x1*x2, -x2^2] (nonlinear drift)
    - g(x) = [[1, x2], [0, 1]] (state-dependent control matrix)
    - u ∈ [-0.5, 0.5]^2 (bounded control input)

    This tests the general case where g(x) varies with state.
    """

    def __init__(self, alpha=1.0, control_bounds=0.5, unsafe_set=None, safe_set=None):
        super().__init__()
        self.system_name = "state_dependent_2d"
        self.input_dim = 2
        self.output_dim = 2
        self.control_dim = 2
        self.alpha = alpha

        # Control bounds: u ∈ [-control_bounds, control_bounds]^2
        self.control_bounds = control_bounds
        self.u_min = np.array([-control_bounds, -control_bounds])
        self.u_max = np.array([control_bounds, control_bounds])

        # Input domain for grid generation and verification
        self.input_domain = BoxDomain([[-3, 3], [-2, 2]])

        # Define unsafe set (obstacles to avoid) - same as Simple2D for consistency
        if unsafe_set is None:
            self.unsafe_set_def = {"type": "circle", "center": [0.0, 0.0], "radius": 0.5}
        else:
            self.unsafe_set_def = unsafe_set

        # Parse unsafe set into domain object
        self.unsafe_set_interior = parse_domain_definition(self.unsafe_set_def, self.input_domain)
        self.unsafe_set_exterior = BoxExteriorDomain(self.input_domain.bounds)
        self.unsafe_set = UnionDomain([self.unsafe_set_interior, self.unsafe_set_exterior])

        # Define safe set
        if safe_set is not None:
            self.safe_set = parse_domain_definition(safe_set, self.input_domain)
        else:
            # Safe set is complement of unsafe within domain
            self.safe_set = ComplementDomain(self.unsafe_set_interior, self.input_domain.bounds)

        # Default translator for verification
        self.translator = NumpyTranslator()

        self.hidden_sizes = [64, 64, 8]  # Hidden layer sizes for neural network

        # Delta for region generation
        self.delta = np.array([0.5, 0.5])

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Uses compute_f and compute_g for consistency with verification.

        Args:
            x: State [x1, x2] with shape [2, batch_size]
            u: Control input [u1, u2] with shape [2, batch_size] (optional)
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [2, batch_size]
        """
        if translator is None:
            if isinstance(x, np.ndarray):
                translator = NumpyTranslator()
            else:
                translator = TorchTranslator()

        # Compute drift term f(x)
        f_x = self.compute_f(x, translator)

        # Add control term g(x)u if provided
        if u is not None:
            g_x = self.compute_g(x, translator)
            # g_x @ u
            g_u = translator.matrix_vector(g_x, u)
            return f_x + g_u
        else:
            return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control.

        f(x) = [-x1*x2, -x2^2]

        Args:
            x: State [x1, x2] with shape [batch_size, 2] or [2]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [batch_size, 2] or [2]
        """
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = -x1 * x2
        dx2 = -translator.pow(x2, 2)
        return translator.stack([dx1, dx2], dim=-1)

    def compute_g(self, x, translator):
        """
        Compute control input matrix g(x).

        For this system, g(x) = [[1, x2], [0, 1]] (state-dependent).

        Args:
            x: State [x1, x2] with shape [batch_size, 2] or [2]
            translator: Mathematical operations translator

        Returns:
            Control matrix with shape [batch_size, 2, 2] for batched inputs or [2, 2] for single inputs
            For TaylorTranslator: returns TaylorExpansion for state-dependent elements
        """
        x2 = x[..., 1]
        zero = translator.zeros_like(x2)
        one = translator.ones_like(x2)

        first_col = translator.stack([one, zero], dim=-1)
        sec_col = translator.stack([x2, zero], dim=-1)
        g_x = translator.stack([first_col, sec_col], dim=-2)  # Shape: [batch_size, 2, 2] or [2, 2]

        return g_x

    def safe_set_constraint(self, x, translator):
        """
        True safe set constraint using the domain objects.

        Args:
            x: State tensor with shape [batch_size, 2]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        h = self.safe_set.constraint(x, translator)
        return h

    @property
    def unsafe_domain(self):
        """Property to expose the unsafe domain."""
        return self.unsafe_set_interior


class RendezvousDockingSystem(CBFDynamicalSystem):
    """
    6D Spacecraft rendezvous and docking dynamics (Clohessy-Wiltshire) in cylindrical coordinates.

    State x = [r, phi, z, v_r, v_phi, v_z]
        r: radial distance from docking axis (km)
        phi: azimuthal angle around docking axis (rad)
        z: axial distance along docking axis (km)
        v_r: radial velocity (km/s)
        v_phi: tangential velocity (km/s)
        v_z: axial velocity (km/s)

    Control u = [u_r, u_phi, u_z] (cylindrical thrust components, N)

    The control input matrix g(x) converts thrust to acceleration accounting for:
    - Variable mass (thrust/mass)
    - Coordinate transformation effects
    """

    def __init__(
        self,
        alpha: float = 1.0,
        orbital_rate: float = 0.001,  # rad/s for LEO
        r_bounds: Optional[List[float]] = None,
        phi_bounds: Optional[List[float]] = None,
        z_bounds: Optional[List[float]] = None,
        vr_bounds: Optional[List[float]] = None,
        vphi_bounds: Optional[List[float]] = None,
        vz_bounds: Optional[List[float]] = None,
        m_bounds: Optional[List[float]] = None,
        thrust_limits: Optional[List[float]] = None,
        theta_max_deg: float = 15.0,
        fuel_consumption_coeff: float = 0.0001,  # kg/(N*s)
        mass: float = 20.0,  # kg
    ):
        super().__init__()

        self.system_name = "spacecraft_rendezvous"
        self.input_dim = 6
        self.output_dim = 6
        self.control_dim = 3
        self.alpha = alpha

        self.mass = mass

        # Clohessy-Wiltshire orbital rate
        self.n = orbital_rate

        # Fuel consumption coefficient
        self.fuel_alpha = fuel_consumption_coeff

        # Approach cone half-angle in radians
        self.theta_max_rad = np.deg2rad(theta_max_deg)

        # Operating domain in cylindrical coordinates
        r_low, r_high = r_bounds if r_bounds is not None else (0.01, 2.0)
        phi_low, phi_high = phi_bounds if phi_bounds is not None else (-np.pi, np.pi)
        z_low, z_high = z_bounds if z_bounds is not None else (-1.0, 1.0)
        vr_low, vr_high = vr_bounds if vr_bounds is not None else (-0.05, 0.05)
        vphi_low, vphi_high = vphi_bounds if vphi_bounds is not None else (-0.05, 0.05)
        vz_low, vz_high = vz_bounds if vz_bounds is not None else (-0.05, 0.05)

        self.input_domain = BoxDomain(
            [
                [r_low, r_high],
                [phi_low, phi_high],
                [z_low, z_high],
                [vr_low, vr_high],
                [vphi_low, vphi_high],
                [vz_low, vz_high],
            ]
        )

        # Safe set: approach cone constraint
        self.safe_set = ApproachConeDomain(
            dim=self.input_dim,
            r_index=0,
            z_index=2,
            theta_max_rad=self.theta_max_rad,
            symmetric_z=True,
        )

        self.unsafe_set_interior = ComplementDomain(self.safe_set, self.input_domain.bounds)

        # Control bounds (thrust limits in N)
        thrust_limits = thrust_limits if thrust_limits is not None else [10.0, 10.0, 10.0]
        self.u_min = -np.array(thrust_limits)
        self.u_max = np.array(thrust_limits)

        # Default translator for verification
        self.translator = NumpyTranslator()

        # Network and region parameters
        self.hidden_sizes = [128, 128]
        self.delta = np.array([0.5, np.pi / 4, 0.5, 0.1, 0.1, 0.1])

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u.
        """
        f_x = self.compute_f(x, translator)

        if u is not None:
            g_x = self.compute_g(x, translator)
            g_u = translator.matrix_vector(g_x, u)
            return f_x + g_u

        return f_x

    def compute_f(self, x, translator):
        """
        Clohessy-Wiltshire drift dynamics in cylindrical coordinates (no control).

        Drift equations:
        dr/dt = v_r
        dphi/dt = v_phi / r
        dz/dt = v_z
        dv_r/dt = (v_phi^2)/r + 3n^2*r*cos^2(phi) + 2n*v_phi
        dv_phi/dt = -(v_r*v_phi)/r + 3n^2*r*sin(phi)*cos(phi) - 2n*v_r
        dv_z/dt = -n^2*z
        """
        r = x[..., 0]
        phi = x[..., 1]
        z = x[..., 2]
        v_r = x[..., 3]
        v_phi = x[..., 4]
        v_z = x[..., 5]

        n = self.n
        cos_phi = translator.cos(phi)
        sin_phi = translator.sin(phi)

        # Add small epsilon to avoid division by zero at r=0
        epsilon = 1e-6
        r_safe = r + epsilon

        # Kinematic equations
        dr = v_r
        dphi = v_phi / r_safe
        dz = v_z

        # Dynamic equations (Clohessy-Wiltshire in cylindrical coordinates)
        # Centrifugal force
        centrifugal = (v_phi * v_phi) / r_safe

        # Coriolis and gravity gradient terms
        dv_r = centrifugal + 3 * (n**2) * r * translator.pow(cos_phi, 2) + 2 * n * v_phi
        dv_phi = -(v_r * v_phi) / r_safe + 3 * (n**2) * r * sin_phi * cos_phi - 2 * n * v_r
        dv_z = -(n**2) * z

        return translator.stack([dr, dphi, dz, dv_r, dv_phi, dv_z], dim=-1)

    def compute_g(self, x, translator):
        """
        Control influence matrix: maps thrust [u_r, u_phi, u_z] to state derivatives.

        The control affects:
        1. Acceleration in cylindrical coordinates (thrust/mass)
        2. Fuel consumption (proportional to thrust magnitude)

        Returns:
            g(x) with shape [..., 7, 3] where:
            - First 3 rows are zeros (kinematic equations not directly controlled)
            - Rows 3-5 are acceleration terms (thrust/mass)
        """
        r = x[..., 0]

        zero = translator.zeros_like(r)
        inv_m = translator.ones_like(r) / self.mass

        # Control matrix columns for [u_r, u_phi, u_z]
        # Each column shows how that control affects all 7 states

        # u_r affects: dv_r
        col_ur = translator.stack(
            [
                zero,  # dr/dt
                zero,  # dphi/dt
                zero,  # dz/dt
                inv_m,  # dv_r/dt = u_r/m
                zero,  # dv_phi/dt
                zero,  # dv_z/dt
            ],
            dim=-1,
        )

        # u_phi affects: dv_phi
        col_uphi = translator.stack(
            [
                zero,  # dr/dt
                zero,  # dphi/dt
                zero,  # dz/dt
                zero,  # dv_r/dt
                inv_m,  # dv_phi/dt = u_phi/m
                zero,  # dv_z/dt
            ],
            dim=-1,
        )

        # u_z affects: dv_z
        col_uz = translator.stack(
            [
                zero,  # dr/dt
                zero,  # dphi/dt
                zero,  # dz/dt
                zero,  # dv_r/dt
                zero,  # dv_phi/dt
                inv_m,  # dv_z/dt = u_z/m
            ],
            dim=-1,
        )

        # Stack columns to form g(x) matrix
        g_x = translator.stack([col_ur, col_uphi, col_uz], dim=-2)

        return g_x

    def safe_set_constraint(self, x, translator):
        """Use the ApproachConeDomain for the safe-set constraint."""
        return self.safe_set.constraint(x, translator)

    @property
    def unsafe_domain(self):
        return self.unsafe_set_interior


class CartPoleSystem(CBFDynamicalSystem):
    """
    Cart-pole system with affine control for CBF verification.

    State x = [x, x_dot, theta, theta_dot]
        x: cart position (m)
        x_dot: cart velocity (m/s)
        theta: pole angle from vertical (rad), positive is counter-clockwise
        theta_dot: pole angular velocity (rad/s)

    Control u = [F] (horizontal force applied to cart, N)

    Dynamics from paper:
        θ̈ = [g*sin(θ) + cos(θ)*(-F - m*ℓ*θ̇²*sin(θ) + μ_c*sgn(ẋ))/(M+m) - μ_p*θ̇/(m*ℓ)] /
            [ℓ*(4/3 - m*cos²(θ)/(M+m))]

        ẍ = [F + m*ℓ*(θ̇²*sin(θ) - θ̈*cos(θ)) - μ_c*sgn(ẋ)] / (M+m)

    Affine form: dx/dt = f(x) + g(x)u where:
    - f(x) contains the nonlinear drift dynamics (with F=0)
    - g(x) maps the control force to state derivatives
    - u ∈ [u_min, u_max] (bounded control input)

    Parameters:
        m_cart (M): mass of the cart (kg)
        m_pole (m): mass of the pole (kg)
        length (ℓ): length of the pole (m)
        gravity (g): gravitational acceleration (m/s^2)
        mu_c: cart friction coefficient ( we ignore cart fricition in this implementation)
        mu_p: pole friction coefficient
    """

    def __init__(
        self,
        alpha: float = 1.0,
        m_cart: float = 1.0,
        m_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.81,
        mu_c: float = 0.0,  # Cart friction coefficient
        mu_p: float = 0.01,  # Pole friction coefficient
        control_bounds: float = 10.0,
        x_bounds: Optional[List[float]] = None,
        x_dot_bounds: Optional[List[float]] = None,
        theta_bounds: Optional[List[float]] = None,
        theta_dot_bounds: Optional[List[float]] = None,
        x_safe_bounds: Optional[List[float]] = None,
    ):
        super().__init__()

        self.system_name = "cart_pole"
        self.input_dim = 4
        self.output_dim = 4
        self.control_dim = 1
        self.alpha = alpha

        # Physical parameters
        self.m_cart = m_cart  # M in paper
        self.m_pole = m_pole  # m in paper
        self.length = length  # ℓ in paper
        self.gravity = gravity  # g in paper
        self.mu_c = mu_c  # Cart friction coefficient
        self.mu_p = mu_p  # Pole friction coefficient
        self.total_mass = m_cart + m_pole

        # Control bounds: u ∈ [-control_bounds, control_bounds]
        self.u_min = np.array([-control_bounds])
        self.u_max = np.array([control_bounds])

        # State space bounds
        x_low, x_high = x_bounds if x_bounds is not None else [-2.4, 2.4]
        x_dot_low, x_dot_high = x_dot_bounds if x_dot_bounds is not None else [-3.0, 3.0]
        theta_low, theta_high = theta_bounds if theta_bounds is not None else [-np.pi / 6, np.pi / 6]
        theta_dot_low, theta_dot_high = theta_dot_bounds if theta_dot_bounds is not None else [-2.0, 2.0]

        self.input_domain = BoxDomain(
            [
                [x_low, x_high],
                [x_dot_low, x_dot_high],
                [theta_low, theta_high],
                [theta_dot_low, theta_dot_high],
            ]
        )

        # Safe set: constrain cart position x within bounds
        if x_safe_bounds is None:
            x_safe_bounds = [-2.0, 2.0]  # Slightly tighter than state space

        self.x_safe_min = x_safe_bounds[0]
        self.x_safe_max = x_safe_bounds[1]

        # Safe set: cart position must stay within [x_safe_min, x_safe_max]
        # The safe set is defined on the full state space but only constrains x
        safe_bounds = [
            [self.x_safe_min, self.x_safe_max],  # x constraint
            [x_dot_low, x_dot_high],  # x_dot unconstrained (within state space)
            [theta_low, theta_high],  # theta unconstrained (within state space)
            [theta_dot_low, theta_dot_high],  # theta_dot unconstrained (within state space)
        ]
        self.safe_set = BoxDomain(safe_bounds)

        # Unsafe set: outside the safe position bounds (complement of safe set within input domain)
        self.unsafe_set_interior = ComplementDomain(self.safe_set, self.input_domain.bounds)

        # Default translator
        self.translator = NumpyTranslator()

        # Network parameters
        # self.hidden_sizes = [64, 64]
        self.hidden_sizes = [32, 64, 32]

        # Delta for region generation
        self.delta = np.array([0.3, 0.5, np.pi / 12, 0.3])

    def compute_dynamics(self, x, translator, u=None):
        """
        Affine control system: dx/dt = f(x) + g(x)u

        Args:
            x: State [x, x_dot, theta, theta_dot] with shape [4, batch_size]
            u: Control input [F] with shape [1, batch_size] (optional)
            translator: Mathematical operations translator

        Returns:
            State derivatives with shape [4, batch_size]
        """
        f_x = self.compute_f(x, translator)

        if u is not None:
            g_x = self.compute_g(x, translator)
            g_u = translator.matrix_vector(g_x, u)
            return f_x + g_u

        return f_x

    def compute_f(self, x, translator):
        """
        Compute drift dynamics f(x) without control (u=0).

        From the paper:
        θ̈ = [g*sin(θ) + cos(θ)*(-m*ℓ*θ̇²*sin(θ) + μ_c*sgn(ẋ))/(M+m) - μ_p*θ̇/(m*ℓ)] /
            [ℓ*(4/3 - m*cos²(θ)/(M+m))]

        ẍ = [m*ℓ*(θ̇²*sin(θ) - θ̈*cos(θ)) - μ_c*sgn(ẋ)] / (M+m)

        Note: When F=0, we solve these coupled equations.

        Args:
            x: State with shape [batch_size, 4] or [4]
            translator: Mathematical operations translator

        Returns:
            Drift term with shape [batch_size, 4] or [4]
        """
        # Extract state variables
        x_pos = x[..., 0]
        x_dot = x[..., 1]
        theta = x[..., 2]
        theta_dot = x[..., 3]

        # Precompute trig functions
        sin_theta = translator.sin(theta)
        cos_theta = translator.cos(theta)
        cos_theta_sq = translator.pow(cos_theta, 2)

        # Paper notation: M = m_cart, m = m_pole, ℓ = length, g = gravity
        M = self.m_cart
        m = self.m_pole
        ell = self.length
        g = self.gravity
        # Ignore  friction force between the cart and the track, include (i.e. mu_c = 0)
        mu_p = self.mu_p

        # Compute θ̈ (theta acceleration) with F=0
        # Numerator: g*sin(θ) + cos(θ)*(-m*ℓ*θ̇²*sin(θ) + μ_c*sgn(ẋ))/(M+m) - μ_p*θ̇/(m*ℓ)
        theta_ddot_num = g * sin_theta + cos_theta * (-m * ell * translator.pow(theta_dot, 2) * sin_theta) / (M + m) - mu_p * theta_dot / (m * ell)

        # Denominator: ℓ*(4/3 - m*cos²(θ)/(M+m))
        theta_ddot_denom = ell * (4.0 / 3.0 - m * cos_theta_sq / (M + m))

        theta_ddot = theta_ddot_num / theta_ddot_denom

        # Compute ẍ (cart acceleration) with F=0
        # ẍ = [m*ℓ*(θ̇²*sin(θ) - θ̈*cos(θ)) - μ_c*sgn(ẋ)] / (M+m)
        x_ddot = (m * ell * (translator.pow(theta_dot, 2) * sin_theta - theta_ddot * cos_theta)) / (M + m)

        # Stack the derivatives
        return translator.stack([x_dot, x_ddot, theta_dot, theta_ddot], dim=-1)

    def compute_g(self, x, translator):
        """
        Compute control input matrix g(x).

        The control force F affects both cart and pole accelerations.
        From the paper, with control F:

        θ̈ = [g*sin(θ) + cos(θ)*(-F - m*ℓ*θ̇²*sin(θ) + μ_c*sgn(ẋ))/(M+m) - μ_p*θ̇/(m*ℓ)] /
            [ℓ*(4/3 - m*cos²(θ)/(M+m))]

        ẍ = [F + m*ℓ*(θ̇²*sin(θ) - θ̈*cos(θ)) - μ_c*sgn(ẋ)] / (M+m)

        Control influence is extracted by taking ∂/∂F of these equations.

        Args:
            x: State with shape [batch_size, 4] or [4]
            translator: Mathematical operations translator

        Returns:
            Control matrix with shape [batch_size, 4, 1] or [4, 1]
        """
        # Extract state variables
        theta = x[..., 2]

        # Precompute trig functions
        cos_theta = translator.cos(theta)
        cos_theta_sq = translator.pow(cos_theta, 2)

        # Paper notation
        M = self.m_cart
        m = self.m_pole
        ell = self.length

        # Create zero elements
        zero = translator.zeros_like(theta)

        # Denominator for θ̈: ℓ*(4/3 - m*cos²(θ)/(M+m))
        theta_ddot_denom = ell * (4.0 / 3.0 - m * cos_theta_sq / (M + m))

        # Control influence on θ̈: ∂θ̈/∂F = -cos(θ)/[(M+m)*ℓ*(4/3 - m*cos²(θ)/(M+m))]
        g_theta_dot = -cos_theta / ((M + m) * theta_ddot_denom)

        # Control influence on ẍ: ∂ẍ/∂F = [1 - m*ℓ*cos(θ)*∂θ̈/∂F] / (M+m)
        # = [1 - m*ℓ*cos(θ)*(-cos(θ)/[(M+m)*ℓ*(4/3 - m*cos²(θ)/(M+m))])] / (M+m)
        # = [1 + m*cos²(θ)/[(M+m)*(4/3 - m*cos²(θ)/(M+m))]] / (M+m)
        g_x_dot = (translator.ones_like(theta) - m * ell * cos_theta * g_theta_dot) / (M + m)

        # Stack into control matrix: [0, g_x_dot, 0, g_theta_dot]^T
        g_col = translator.stack([zero, g_x_dot, zero, g_theta_dot], dim=-1)

        # Reshape to [..., 4, 1] to represent a column vector
        return translator.unsqueeze(g_col, -1)

    def safe_set_constraint(self, x, translator):
        """
        Safe set constraint: cart position must stay within bounds.

        The constraint only depends on the cart position x (first state variable),
        but operates on the full 4D state vector.

        h(x) = min(x - x_min, x_max - x) >= 0

        where x is the cart position (x[..., 0]).

        Args:
            x: State tensor with shape [batch_size, 4]
            translator: Mathematical operations translator

        Returns:
            Constraint values with shape [batch_size] where positive means safe
        """
        # Use the BoxDomain constraint which handles the full state
        # The safe_set BoxDomain only constrains x, other dimensions are unconstrained
        h = self.safe_set.constraint(x, translator)

        return h

    @property
    def unsafe_domain(self):
        """Property to expose the unsafe domain."""
        return self.unsafe_set_interior
