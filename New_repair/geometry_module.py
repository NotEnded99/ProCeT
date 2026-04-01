"""
Geometry Module for CBF Verification - LBP Engine and Jacobian Computation

This module provides core functionality for converting state space simplices
into parameter space constraint matrices for neural CBF verification.

Core Functions:
1. compute_simplex_bound: Use LBP forward propagation to compute bounds
2. compute_jacobian_matrix: Compute Jacobian matrix of constraint constraint w.r.t. network parameters
"""
# Import components from original verification code
import sys
from pathlib import Path
cwd = str(Path.cwd())
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import torch
import torch.nn.functional as F
from typing import List, Tuple, Union
import numpy as np
from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization
from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.cbf.cbf_dynamics import CBFDynamicalSystem
from lbp_neural_cbf.regions.simplicial import SimplicialRegion
from lbp_neural_cbf.linearization.taylor import TaylorLinearization
from lbp_neural_cbf.translators import TorchTranslator


def compute_simplex_bound(model, simplex, dynamics_model, region_type='safe', device='cpu'):
    """
    Execute LBP forward propagation with McCormick relaxation, compute and return
    corresponding scalar bound based on region type.

    Args:
        model: Current neural network B_theta (BarrierNN or torch.nn.Module)
        simplex: Simplex object (SimplicialRegion)
        dynamics_model: Dynamical system (CBFDynamicalSystem)
        region_type: String, 'safe' or 'unsafe'
        device: Computation device ('cpu' or 'cuda')

    Returns:
        bound_val: Scalar Tensor representing bound value for corresponding region
            - For 'unsafe' region: returns upper bound of h(x) (for verifying h_max < 0)
            - For 'safe' region: returns lower bound of CBF Lie derivative condition
              (for verifying min_L >= 0)

    Raises:
        ValueError: If region_type is not 'safe' or 'unsafe'
    """
    if region_type not in ['safe', 'unsafe']:
        raise ValueError(f"region_type must be 'safe' or 'unsafe', got '{region_type}'")

    # Ensure model is on correct device
    if hasattr(model, 'to'):
        model = model.to(device)

    # Set model to eval mode
    if hasattr(model, 'eval'):
        model.eval()

    # Create Crown linearizer
    # Note: CrownPartialLinearization requires dtype parameter
    dtype = torch.float32
    network_linearizer = CrownPartialLinearization(model, dtype=dtype)

    # Move model to specified device (if needed)
    network_linearizer.device = torch.device(device)

    # Compute LBP forward propagation
    # Note: compute_network_bounds needs a batch as a list
    network_linearizer.compute_network_bounds([simplex])

    # Get network output bounds h(x)
    h_min, h_max = network_linearizer.get_network_output_bounds(sample_idx=0)

    if region_type == 'unsafe':
        """
        Unsafe region verification condition: h_max < 0

        For truly unsafe regions (obstacle interior), we need to verify:
        h(x) is negative over entire region, i.e., upper bound is less than zero
        """
        # Return upper bound (we need h_max < 0 to satisfy condition)
        bound_val = torch.tensor(h_max, device=device, dtype=dtype, requires_grad=False)

    elif region_type == 'safe':
        """
        Safe region verification condition: Lower bound of Lie derivative forward invariance condition >= 0

        CBF condition: min_L >= 0
        where min_L is the lower bound of L_f h + sup_u L_g h * u + alpha(h)
        """
        # Compute Jacobian bounds (for computing Lie derivative)
        network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=0)

        # Get Jacobian bounds J(x) = ∇h/∂x
        A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()

        # Convert to format needed by CBF verification code
        J_affine_L = (A_L, b_L)
        J_affine_U = (A_U, b_U)

        # Get network linear bounds h(x)
        (h_A_L, h_a_L), (h_A_U, h_a_U) = network_linearizer.get_network_linear_bounds()

        # Compute dynamics bounds (using Taylor expansion)
        # Note: _compute_dynamics_bounds_taylor needs batch
        translator = TorchTranslator(device=torch.device(device))
        taylor_linearizer = TaylorLinearization(dynamics_model, numeric_translator=translator)

        # Compute drift term f(x) bounds
        f_linearization = taylor_linearizer.linearize_sample(simplex)
        (f_A_L, f_b_L), (f_A_U, f_b_U), _ = f_linearization.first_order_model
        f_affine_L = (f_A_L, f_b_L)
        f_affine_U = (f_A_U, f_b_U)

        # Compute McCormick product lower bound for drift term
        # Term: J(x) · f(x)
        eta_drift = 0.5
        drift_L = _compute_mccormick_product_lower_bound(
            J_affine_L, J_affine_U, f_affine_L, f_affine_U,
            simplex, eta=eta_drift, device=device, dtype=dtype
        )

        # Sum over state dimensions
        drift_L_sum = drift_L.sum() if hasattr(drift_L, 'sum') else torch.tensor(drift_L.sum(), device=device, dtype=dtype)

        # Compute class-K term bound: alpha(h(x))
        # Use lower bound of h to compute lower bound of alpha(h)
        alpha_A_L = dynamics_model.alpha_function(h_A_L[..., 0, :])
        alpha_a_L = dynamics_model.alpha_function(h_a_L[..., 0])

        alpha_L_bound = alpha_A_L.sum() if hasattr(alpha_A_L, 'sum') else torch.tensor(alpha_A_L.sum(), device=device, dtype=dtype)
        alpha_a_L_bound = alpha_a_L if isinstance(alpha_a_L, torch.Tensor) else torch.tensor(alpha_a_L, device=device, dtype=dtype)

        alpha_L_total = alpha_L_bound + alpha_a_L_bound

        # Handle control term (if any)
        n = dynamics_model.input_dim
        m = dynamics_model.control_dim

        control_L_sum = torch.tensor(0.0, device=device, dtype=dtype)

        if m > 0:
            # Compute control term g(x) bounds
            class GDynamics:
                def __init__(self, original_dynamics):
                    self.original_dynamics = original_dynamics
                    self.input_dim = original_dynamics.input_dim
                def compute_dynamics(self, x, translator):
                    return self.original_dynamics.compute_g(x, translator)

            g_dynamics = GDynamics(dynamics_model)
            g_linearizer = TaylorLinearization(g_dynamics, numeric_translator=translator)
            g_linearization = g_linearizer.linearize_sample(simplex)
            (g_A_L, g_b_L), (g_A_U, g_b_U), _ = g_linearization.first_order_model
            g_affine_L = (g_A_L, g_b_L)
            g_affine_U = (g_A_U, g_b_U)

            # Compute McCormick product lower bound for control term
            # Term: v(x) = J(x) · g(x)
            eta_control = 0.5
            v_L = _compute_mccormick_product_lower_bound(
                J_affine_L, J_affine_U, g_affine_L, g_affine_U,
                simplex, eta=eta_control, device=device, dtype=dtype
            )

            # Lower and upper bounds of v(x)
            v_L_min = _get_affine_function_bounds((v_L[0], v_L[1]), simplex, device=device, dtype=dtype)[0]
            v_L_max = _get_affine_function_bounds((v_L[0], v_L[1]), simplex, device=device, dtype=dtype)[1]

            # Compute sup_u v(x) · u
            u_min = torch.tensor(dynamics_model.u_min, device=device, dtype=dtype)
            u_max = torch.tensor(dynamics_model.u_max, device=device, dtype=dtype)

            # For each control dimension, find optimal u
            v_L_pos = torch.where(v_L_min >= 0, v_L_max, v_L_min)
            v_L_neg = torch.where(v_L_max <= 0, v_L_min, v_L_max)

            # Positive terms use u_max, negative terms use u_min
            control_L_sum = (v_L_pos * u_max).sum() + (v_L_neg * u_min).sum()

        # Final CBF condition lower bound: min_L = drift + control + alpha(h)
        min_L = drift_L_sum + control_L_sum + alpha_L_total

        bound_val = min_L.detach()  # Detach from computation graph

    return bound_val


def compute_jacobian_matrix(model, V_safe, V_unsafe, dynamics_model, device='cpu'):
    """
    Compute Jacobian matrix J of all verified region constraint bounds w.r.t. network parameters theta.

    Args:
        model: Neural network (BarrierNN or torch.nn.Module)
        V_safe: Set of safe simplices (goal is to ensure Lie derivative lower bound >= 0)
        V_unsafe: Set of obstacle simplices (goal is to ensure network output upper bound < 0)
        dynamics_model: Dynamical system (used for safe region CBF condition computation)
        device: Computation device ('cpu' or 'cuda')

    Returns:
        J: 2D Tensor of shape (m, |theta|)
           where m = |V_safe| + |V_unsafe|
           |theta| is the total number of network parameters

    Raises:
        ValueError: If model has no parameters
    """
    # Ensure model is on correct device
    if hasattr(model, 'to'):
        model = model.to(device)

    # Set model to eval mode
    if hasattr(model, 'eval'):
        model.eval()

    # Get all network parameters (in order)
    params_dict = dict(model.named_parameters())
    param_names = list(params_dict.keys())
    total_params = sum(p.numel() for p in params_dict.values())

    if total_params == 0:
        raise ValueError("Model has no parameters")

    # Create wrapper function for computing constraint bound for single simplex
    def _compute_constraint_for_simplex(simplex, region='safe'):
        """
        Compute constraint bound value for single simplex

        Args:
            simplex: Simplex object
            region: 'safe' or 'unsafe'

        Returns:
            constraint_value: Scalar Tensor
        """
        with torch.enable_grad():  # Ensure gradient is available
            return compute_simplex_bound(model, simplex, dynamics_model, region_type=region, device=device)

    # Prepare all simplices
    all_regions = [(simplex, 'safe') for simplex in V_safe] + [(simplex, 'unsafe') for simplex in V_unsafe]
    m = len(all_regions)

    # Method 1: Use torch.func.jacrev to compute Jacobian row by row
    # Note: torch.func.jacrev requires PyTorch 2.0+
    # We create a functional API to compute all constraints at once

    # Define functional call
    def _constraint_function(params_flat, simplexes_data, region_types_data, model_arch, dynamics):
        """
        Functional constraint computation for Jacobian computation

        Args:
            params_flat: Flattened parameter vector
            simplexes_data: Simplex vertices data
            region_types_data: Region type markers
            model_arch: Model architecture information
            dynamics: Dynamical system

        Returns:
            constraints: Constraint value vector
        """
        # Load parameters into temporary model
        idx = 0
        for name, param in model.named_parameters():
            numel = param.numel()
            param.data.copy_(params_flat[idx:idx+numel].reshape(param.shape))
            idx += numel

        # Compute each constraint
        constraints = []
        for simplex_data, region_type in zip(simplexes_data, region_types_data):
            # Reconstruct simplex object from data
            vertices = simplex_data.numpy() if hasattr(simplex_data, 'numpy') else simplex_data
            simplex = SimplicialRegion(vertices)

            # Compute constraint
            constraint = compute_simplex_bound(model_arch, simplex, dynamics, region_type=region, device=device)
            constraints.append(constraint)

        return torch.stack(constraints)

    # Prepare data
    simplexes_data = [torch.tensor(simplex.vertices, device=device, dtype=torch.float32) for simplex, _ in all_regions]
    region_types_data = [region_type for _, region_type in all_regions]

    # Get model initial parameters
    initial_params = torch.cat([p.flatten() for p in model.parameters()])

    # Create functional model call
    try:
        # Use functorch or torch.func
        if hasattr(torch, 'func') and hasattr(torch.func, 'jacrev'):
            # PyTorch 2.0+
            constraint_func = lambda params: _constraint_function(
                params, simplexes_data, region_types_data, model, dynamics_model
            )

            # Compute Jacobian matrix
            J = torch.func.jacrev(constraint_func)(initial_params)

        else:
            # Fallback method: use autograd row by row
            J_rows = []

            for simplex, region_type in all_regions:
                # Clear gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = torch.zeros_like(param)
                    else:
                        param.grad.zero_()

                # Compute constraint and backpropagate
                constraint = compute_simplex_bound(model, simplex, dynamics_model, region_type=region, device=device)

                # Backpropagate
                constraint.backward()

                # Collect gradient
                grad_vec = torch.cat([p.grad.flatten() for p in model.parameters()])
                J_rows.append(grad_vec)

            J = torch.stack(J_rows)

    except Exception as e:
        # If jacrev is not available, use numerical differentiation
        print(f"Warning: jacrev not available, using numerical differentiation. Error: {e}")
        J = _compute_jacobian_numerical(model, all_regions, dynamics_model, device=device, eps=1e-5)

    return J


def _compute_mccormick_product_lower_bound(affine1_L, affine1_U, affine2_L, affine2_U, simplex, eta=0.5, device='cpu', dtype=torch.float32):
    """
    Compute McCormick lower bound for product: (affine1) · (affine2)

    Args:
        affine1_L, affine1_U: First affine function's (A, b) lower and upper bounds
        affine2_L, affine2_U: Second affine function's (A, b) lower and upper bounds
        simplex: Simplex object
        eta: McCormick relaxation parameter (0 to 1)
        device: Computation device
        dtype: Data type

    Returns:
        (M, c): Lower bound affine function's coefficient and constant
    """
    # Get interval bounds
    y1_min, y1_max = _get_affine_function_bounds(affine1_L, simplex, affine1_U, device=device, dtype=dtype)
    y2_min, y2_max = _get_affine_function_bounds(affine2_L, simplex, affine2_U, device=device, dtype=dtype)

    (A1_L, b1_L), (A1_U, b1_U) = affine1_L, affine1_U
    (A2_L, b2_L), (A2_U, b2_U) = affine2_L, affine2_U

    # McCormick lower bound formula
    C1 = eta * y1_min + (1 - eta) * y1_max
    C2 = eta * y2_min + (1 - eta) * y2_max
    const_part = -(eta * y1_min * y2_min + (1 - eta) * y1_max * y2_max)

    C1_pos = torch.clamp(C1, min=0)
    C1_neg = C1 - C1_pos
    C2_pos = torch.clamp(C2, min=0)
    C2_neg = C2 - C2_pos

    # Lower bound affine coefficient
    M = C1_pos.unsqueeze(-1) * A2_L + C1_neg.unsqueeze(-1) * A2_U + C2_pos.unsqueeze(-1) * A1_L + C2_neg.unsqueeze(-1) * A1_U
    c = C1_pos * b2_L + C1_neg * b2_U + C2_pos * b1_L + C2_neg * b1_U + const_part

    return M, c


def _get_affine_function_bounds(affine_L, simplex, affine_U=None, device='cpu', dtype=torch.float32):
    """
    Compute min/max of affine function over simplex

    Args:
        affine_L: Lower bound (A, b)
        simplex: Simplex object
        affine_U: Upper bound (A, b), if None then use affine_L
        device: Computation device
        dtype: Data type

    Returns:
        (lower, upper): Minimum and maximum values
    """
    (A, b) = affine_L

    # For simplex, evaluate at all vertices to find min/max
    vertices = torch.tensor(simplex.vertices, device=device, dtype=dtype)

    # Evaluate lower bound function at all vertices
    values_L = torch.matmul(vertices, A) + b
    lower = torch.min(values_L)

    if affine_U is None:
        upper = torch.max(values_L)
    else:
        (A_U, b_U) = affine_U
        values_U = torch.matmul(vertices, A_U) + b_U
        upper = torch.max(values_U)

    return lower, upper


def _compute_jacobian_numerical(model, regions_with_types, dynamics_model, device='cpu', eps=1e-5):
    """
    Compute Jacobian matrix using numerical finite difference (fallback method)

    Args:
        model: Neural network
        regions_with_types: List of (rarex, region_type) tuples
        dynamics_model: Dynamical system
        device: Computation device
        eps: Numerical difference step size

    Returns:
        J: Jacobian matrix (m, |theta|)
    """
    # Get total parameters count
    total_params = sum(p.numel() for p in model.parameters())

    # Save current parameters
    original_params = [p.data.clone() for p in model.parameters()]

    # Compute all constraints under current parameters
    def _compute_all_constraints(params):
        idx = 0
        for param, original in zip(model.parameters(), original_params):
            param.data.copy_(params[idx:idx+param.numel()].reshape(param.shape))
            idx += param.numel()

        values = []
        for simplex, region_type in regions_with_types:
            val = compute_simplex_bound(model, simplex, dynamics_model, region_type=region, device=device)
            values.append(val.item())
        return torch.tensor(values, device=device, dtype=torch.float32)

    J_rows = torch.zeros(len(regions_with_types), total_params, device=device, dtype=torch.float32)

    # Compute finite difference for each parameter
    current_values = _compute_all_constraints(torch.cat([p.flatten() for p in model.parameters()]))

    for param_idx in range(total_params):
        # Create perturbation
        params_perturbed = torch.cat([p.flatten() for p in model.parameters()]).clone()
        params_perturbed[param_idx] += eps

        # Compute constraint values after perturbation
        perturbed_values = _compute_all_constraints(params_perturbed)

        # Numerical gradient
        J_rows[:, param_idx] = (perturbed_values - current_values) / eps

    # Restore original parameters
    for param, original in zip(model.parameters(), original_params):
        param.data.copy_(original)

    return J_rows


def test_geometry_module():
    """
    Test geometry module functionality

    Returns:
        dict: Test results
    """
    import sys
    from pathlib import Path
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    from lbp_neural_cbf.cbf.fossil_dynamics import Barrier3System
    from lbp_neural_cbf.cbf.network import BarrierNN

    # Create test system
    dynamics_model = Barrier3System(alpha=1.0)

    # Load test model
    model_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.pth"

    try:
        model = BarrierNN(
            input_size=dynamics_model.input_dim,
            hidden_sizes=dynamics_model.hidden_sizes,
            device='cpu'
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()

        # Create test simplex
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        test_simplex = SimplicialRegion(vertices)

        # Test compute_simplex_bound
        print("\nTest 1: compute_simplex_bound")
        print("-" * 40)

        # Test unsafe region
        unsafe_bound = compute_simplex_bound(model, test_simplex, dynamics_model, region_type='unsafe', device='cpu')
        print(f"Unsafe region bound (h_max): {unsafe_bound.item():.6f}")
        print(f"Verification condition: h_max < 0 ? {unsafe_bound.item() < 0}")

        # Test safe region
        safe_bound = compute_simplex_bound(model, test_simplex, dynamics_model, region_type='safe', device='cpu')
        print(f"\nSafe region bound (min_L): {safe_bound.item():.6f}")
        print(f"Verification condition: min_L >= 0 ? {safe_bound.item() >= 0}")

        # Test compute_jacobian_matrix
        print("\nTest 2: compute_jacobian_matrix")
        print("-" * 40)

        # Create test simplex sets
        V_safe = [test_simplex]
        V_unsafe = []

        try:
            J = compute_jacobian_matrix(model, V_safe, V_unsafe, dynamics_model, device='cpu')
            print(f"Jacobian matrix shape: {J.shape}")
            print(f"Total parameters: {J.shape[1]}")
            print(f"Total constraints: {J.shape[0]}")

            # Print partial Jacobian values
            print("\nFirst 5x5 Jacobian values:")
            print(J[:5, :5])

            return {
                'success': True,
                'unsafe_bound': unsafe_bound.item(),
                'safe_bound': safe_bound.item(),
                'jacobian_shape': J.shape,
                'jacobian_sample': J[:5, :5].tolist()
            }

        except Exception as e:
            print(f"Jacobian computation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'unsafe_bound': unsafe_bound.item(),
                'safe_bound': safe_bound.item()
            }

    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f"Model loading failed: {e}"
        }


if __name__ == '__main__':
    # Run tests
    results = test_geometry_module()

    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    if results.get('success', False):
        print("All core functionality tests passed!")
        print(f"  - Unsafe region bound: {results['unsafe_bound']:.6f}")
        print(f"  - Safe region bound: {results['safe_bound']:.6f}")
        print(f"  - Jacobian matrix shape: {results['jacobian_shape']}")
    else:
        print("Test failed!")
        if 'error' in results:
            print(f"  Error: {results['error']}")
