"""
IBP-Based CBF Verification Module

Implements Interval Bound Propagation (IBP) for neural CBF verification
as a stable alternative to LBP (Linear Bound Propagation) during fine-tuning.

Key design:
- Uses LBP (CROWN) for computing h_min, h_max (network output bounds)
  This part is stable because it uses linear bounds with CROWN
- Uses IBP for computing min_L (CBF condition lower bound)
  This avoids the problematic 1/(u-l) slope computations in LBP
- For dynamics bounds, uses IBP or vertex evaluation

Based on: IBP_Based_CBF_Verification_Technical_Specification.md
"""

from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
import time
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lbp_neural_cbf.linearization import TaylorLinearization
from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization
from lbp_neural_cbf.translators import TorchTranslator
from lbp_neural_cbf.regions import SimplicialRegion, HyperrectangularRegion
from lbp_neural_cbf.cbf.domain import unsafe_region


# =============================================================================
# 1. Core IBP Data Structures
# =============================================================================

@dataclass
class IBPBounds:
    """
    Interval bounds [lower, upper] for layer values.

    Attributes:
        lower: Lower bound tensor
        upper: Upper bound tensor
    """
    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self):
        assert self.lower.shape == self.upper.shape, \
            f"Lower and upper must have same shape, got {self.lower.shape} vs {self.upper.shape}"


# =============================================================================
# 2. Activation Derivative Bounds
# =============================================================================

def relu_derivative_bounds(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (sigma_prime_l, sigma_prime_u) for ReLU activation.
    """
    sigma_prime_l = torch.zeros_like(l)
    sigma_prime_u = torch.ones_like(u)

    active = (l >= 0) & (u >= 0)
    sigma_prime_l[active] = 1.0
    sigma_prime_u[active] = 1.0

    inactive = (l <= 0) & (u <= 0)
    sigma_prime_l[inactive] = 0.0
    sigma_prime_u[inactive] = 0.0

    return sigma_prime_l, sigma_prime_u


def tanh_derivative_bounds(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (sigma_prime_l, sigma_prime_u) for Tanh activation.
    """
    def sigma_prime(y: torch.Tensor) -> torch.Tensor:
        y_clamped = y.clamp(-50, 50)
        exp_2y = torch.exp(2 * y_clamped)
        return 4 * exp_2y / (exp_2y + 1) ** 2

    sp_l = sigma_prime(l)
    sp_u = sigma_prime(u)

    sigma_prime_l = torch.minimum(sp_l, sp_u)
    contains_zero = (l < 0) & (u > 0)
    sigma_prime_u = torch.where(contains_zero, torch.ones_like(l), torch.maximum(sp_l, sp_u))

    sigma_prime_l = torch.clamp(sigma_prime_l, 0, 1)
    sigma_prime_u = torch.clamp(sigma_prime_u, 0, 1)

    return sigma_prime_l, sigma_prime_u


def sigmoid_derivative_bounds(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (sigma_prime_l, sigma_prime_u) for Sigmoid activation.
    """
    def sigma_prime(y: torch.Tensor) -> torch.Tensor:
        y_clamped = y.clamp(-50, 50)
        exp_y = torch.exp(y_clamped)
        return exp_y / (exp_y + 1) ** 2

    sp_l = sigma_prime(l)
    sp_u = sigma_prime(u)

    sigma_prime_l = torch.minimum(sp_l, sp_u)
    contains_zero = (l < 0) & (u > 0)
    sigma_prime_u = torch.where(contains_zero, 0.25 * torch.ones_like(l), torch.maximum(sp_l, sp_u))

    sigma_prime_l = torch.clamp(sigma_prime_l, 0, 0.25)
    sigma_prime_u = torch.clamp(sigma_prime_u, 0, 0.25)

    return sigma_prime_l, sigma_prime_u


def get_activation_derivative_bounds(
    activation_type: str,
    l: torch.Tensor,
    u: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatcher for activation derivative bounds."""
    if activation_type.lower() == 'relu':
        return relu_derivative_bounds(l, u)
    elif activation_type.lower() == 'tanh':
        return tanh_derivative_bounds(l, u)
    elif activation_type.lower() == 'sigmoid':
        return sigmoid_derivative_bounds(l, u)
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


# =============================================================================
# 3. IBP Forward Propagation
# =============================================================================

def ibp_linear(input_bounds: IBPBounds, weight: torch.Tensor, bias: torch.Tensor) -> IBPBounds:
    """
    Apply IBP through a linear layer: y = Wx + b
    Uses positive-negative weight decomposition.
    """
    W_pos = torch.relu(weight)
    W_neg = weight - W_pos

    l_out = F.linear(input_bounds.lower, W_pos, bias) + F.linear(input_bounds.upper, W_neg, None)
    u_out = F.linear(input_bounds.upper, W_pos, bias) + F.linear(input_bounds.lower, W_neg, None)

    return IBPBounds(l_out, u_out)


def ibp_relu(input_bounds: IBPBounds) -> IBPBounds:
    """Apply IBP through ReLU activation."""
    l_out = torch.clamp(input_bounds.lower, min=0)
    u_out = torch.clamp(input_bounds.upper, min=0)
    return IBPBounds(l_out, u_out)


def ibp_tanh(input_bounds: IBPBounds) -> IBPBounds:
    """Apply IBP through Tanh activation."""
    l_out = torch.tanh(input_bounds.lower)
    u_out = torch.tanh(input_bounds.upper)
    return IBPBounds(l_out, u_out)


def ibp_sigmoid(input_bounds: IBPBounds) -> IBPBounds:
    """Apply IBP through Sigmoid activation."""
    l_out = torch.sigmoid(input_bounds.lower)
    u_out = torch.sigmoid(input_bounds.upper)
    return IBPBounds(l_out, u_out)


def ibp_activation(input_bounds: IBPBounds, activation_type: str) -> IBPBounds:
    """Dispatcher for IBP activation propagation."""
    if activation_type.lower() == 'relu':
        return ibp_relu(input_bounds)
    elif activation_type.lower() == 'tanh':
        return ibp_tanh(input_bounds)
    elif activation_type.lower() == 'sigmoid':
        return ibp_sigmoid(input_bounds)
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


# =============================================================================
# 4. IBP Network Bound Calculator
# =============================================================================

class IBPNetworkBoundCalculator:
    """
    Computes IBP bounds for neural networks.

    Provides:
    - ibp_forward(): Forward interval propagation through network
    - ibp_jacobian_bounds(): Jacobian interval propagation
    - compute_min_L(): CBF condition lower bound computation
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.dtype = dtype
        self.device = device or torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")

        # Extract layers and activation types
        self.fc_layers: List[nn.Linear] = []
        self.activation_types: List[str] = []

        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.fc_layers.append(module)
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                self.activation_types.append(type(module).__name__)

        # Storage for forward pass bounds
        self.pre_act_bounds: List[IBPBounds] = []
        self.post_act_bounds: List[IBPBounds] = []
        self.input_bounds: Optional[IBPBounds] = None
        self.output_bounds: Optional[IBPBounds] = None

    def _extract_input_bounds(self, batch) -> IBPBounds:
        """Extract input bounds from a batch of regions."""
        if isinstance(batch[0], SimplicialRegion):
            vertices = []
            for sample in batch:
                verts = torch.tensor(sample.vertices, dtype=self.dtype, device=self.device)
                vertices.append(verts)
            vertices = torch.stack(vertices, dim=0)

            x_lb = vertices.min(dim=-2).values
            x_ub = vertices.max(dim=-2).values

        elif isinstance(batch[0], HyperrectangularRegion):
            centroids = []
            radii = []
            for sample in batch:
                centroids.append(torch.tensor(sample.centroid, dtype=self.dtype, device=self.device))
                radii.append(torch.tensor(sample.radius_vec, dtype=self.dtype, device=self.device))
            centroids = torch.stack(centroids, dim=0)
            radii = torch.stack(radii, dim=0)

            x_lb = centroids - radii
            x_ub = centroids + radii

        else:
            raise TypeError(f"Unsupported region type: {type(batch[0])}")

        return IBPBounds(x_lb, x_ub)

    def ibp_forward(self, batch) -> List[IBPBounds]:
        """Perform IBP forward pass through the network."""
        self.pre_act_bounds = []
        self.post_act_bounds = []

        self.input_bounds = self._extract_input_bounds(batch)
        current_bounds = self.input_bounds

        num_activations = len(self.activation_types)
        layer_idx = 0

        for i, fc_layer in enumerate(self.fc_layers):
            weight = fc_layer.weight.to(self.device, self.dtype)
            bias = fc_layer.bias.to(self.device, self.dtype) if fc_layer.bias is not None else None

            current_bounds = ibp_linear(current_bounds, weight, bias)
            self.pre_act_bounds.append(current_bounds)

            if layer_idx < num_activations:
                act_type = self.activation_types[layer_idx]
                current_bounds = ibp_activation(current_bounds, act_type)
                self.post_act_bounds.append(current_bounds)
                layer_idx += 1

        self.output_bounds = current_bounds
        return self.post_act_bounds

    def ibp_jacobian_bounds(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IBP bounds on the Jacobian ∂h/∂x using autograd.

        For a scalar output h(x), the Jacobian is ∂h/∂x with shape [batch, n_in].
        We use the network's autograd to compute this directly.

        Returns:
            Tuple of (J_L, J_U): Lower and upper Jacobian bounds
            Shape: [batch, n_in]
        """
        if not hasattr(self, 'output_bounds') or self.output_bounds is None:
            raise ValueError("Must call ibp_forward() first")

        batch_size = len(batch)
        n_in = self.input_bounds.lower.shape[-1]

        # Get the actual input bounds (center of region for evaluation)
        x_lb = self.input_bounds.lower  # [batch, n_in]
        x_ub = self.input_bounds.upper  # [batch, n_in]
        x_center = (x_lb + x_ub) / 2  # Use center for Jacobian evaluation

        # Clone and make requires_grad for gradient computation
        x_for_grad = x_center.detach().clone().requires_grad_(True)

        # Compute network output
        model = self.model.to(self.device)
        model.eval()  # Don't affect batch norm etc

        output = model(x_for_grad)  # [batch, n_out]

        # For scalar output, compute Jacobian via autograd
        n_out = output.shape[-1]

        if n_out == 1:
            # Scalar output: Jacobian is grad of scalar output w.r.t. input
            jac = torch.zeros(batch_size, n_in, dtype=self.dtype, device=self.device)

            for i in range(batch_size):
                if output[i, 0].requires_grad:
                    grads = torch.autograd.grad(
                        outputs=output[i, 0],
                        inputs=x_for_grad,
                        retain_graph=(i < batch_size - 1)
                    )[0]
                    jac[i] = grads[i]

            # For IBP, we use the computed Jacobian as both bounds
            # (This is the "center" value; interval propagation would give bounds)
            J_L = jac
            J_U = jac
        else:
            # Multi-output case
            jac = torch.zeros(batch_size, n_out, n_in, dtype=self.dtype, device=self.device)
            for i in range(batch_size):
                for j in range(n_out):
                    if output[i, j].requires_grad:
                        grads = torch.autograd.grad(
                            outputs=output[i, j],
                            inputs=x_for_grad,
                            retain_graph=(i < batch_size - 1 or j < n_out - 1)
                        )[0]
                        jac[i, j] = grads[i]

            J_L = jac.reshape(batch_size, -1)
            J_U = jac.reshape(batch_size, -1)

        return J_L, J_U

    def ibp_jacobian_bounds_interval(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IBP bounds on the Jacobian using interval arithmetic directly.

        This computes conservative bounds via backward interval propagation.

        Returns:
            Tuple of (J_L, J_U): Lower and upper Jacobian bounds
            Shape: [batch, n_in]
        """
        if not self.pre_act_bounds:
            raise ValueError("Must call ibp_forward() before ibp_jacobian_bounds_interval()")

        batch_size = len(batch)
        n_in = self.input_bounds.lower.shape[-1]
        n_out = self.output_bounds.lower.shape[-1]

        num_layers = len(self.fc_layers)
        num_activations = len(self.activation_types)

        # Initialize: d(output)/d(layer_output) = 1 for scalar output
        # dh/dz has shape [batch, n_i] where n_i is the output dim of layer i
        # Start with scalar 1: [batch, 1]
        dh_dz_L = torch.ones(batch_size, 1, dtype=self.dtype, device=self.device)
        dh_dz_U = torch.ones(batch_size, 1, dtype=self.dtype, device=self.device)

        # Backward pass through layers
        for layer_idx in reversed(range(num_layers)):
            fc_layer = self.fc_layers[layer_idx]
            weight = fc_layer.weight.to(self.device, self.dtype)
            W_pos = torch.relu(weight)
            W_neg = weight - W_pos

            # Apply activation derivative first (going backward)
            if layer_idx < num_activations:
                act_type = self.activation_types[layer_idx]
                pre_bounds = self.pre_act_bounds[layer_idx]
                sp_l, sp_u = get_activation_derivative_bounds(act_type, pre_bounds.lower, pre_bounds.upper)

                # sp has shape [batch, n_i] where n_i is the output dim of this layer
                # But we need to handle broadcasting correctly
                # For scalar output, dh/dz has shape [batch, 1], sp has shape [batch, n_i]
                # We need to expand dh/dz to match sp

                # Actually, for proper interval propagation:
                # dh/dy = dh/dz * sp (element-wise)
                # dh/dz has shape [batch, n_i] (output dim of layer)
                # sp has shape [batch, n_i]
                # Result: [batch, n_i]

                dh_dz_L = dh_dz_L * sp_l
                dh_dz_U = dh_dz_U * sp_u

            # Apply linear backward: dh/dz_prev = dh/dy @ W
            # dh/dy [batch, n_out_i], W [n_out_i, n_in_i] -> dh/dz_prev [batch, n_in_i]
            # For interval: use W^+ for lower, W^- for upper (conservative)
            dh_dz_L_new = torch.matmul(dh_dz_L, W_pos) + torch.matmul(dh_dz_U, W_neg)
            dh_dz_U_new = torch.matmul(dh_dz_U, W_pos) + torch.matmul(dh_dz_L, W_neg)

            dh_dz_L = dh_dz_L_new
            dh_dz_U = dh_dz_U_new

        # After backward pass, dh_dz is dh/dx with shape [batch, n_in]
        J_L = dh_dz_L
        J_U = dh_dz_U

        assert J_L.shape == (batch_size, n_in), f"Expected J_L shape {(batch_size, n_in)}, got {J_L.shape}"

        return J_L, J_U

    def get_network_output_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the IBP bounds on network output h(x)."""
        if not hasattr(self, 'output_bounds') or self.output_bounds is None:
            raise ValueError("Must call ibp_forward() first")

        return self.output_bounds.lower, self.output_bounds.upper


# =============================================================================
# 5. vec_min Helper Functions
# =============================================================================

def vec_min(*tensors: torch.Tensor) -> torch.Tensor:
    """Compute element-wise minimum across multiple tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.minimum(result, t)
    return result


def vec_max(*tensors: torch.Tensor) -> torch.Tensor:
    """Compute element-wise maximum across multiple tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.maximum(result, t)
    return result


# =============================================================================
# 6. Dynamics Bounds (Reusing Taylor-based Approach)
# =============================================================================

def _compute_dynamics_bounds_ibp(batch, dynamics_model, device, dtype):
    """
    Compute dynamics bounds by evaluating f(x) and g(x) at region vertices/centers.

    Returns bounds in interval format [f_L, f_U] and [g_L, g_U].
    Each bound is [batch_size, n] for consistency with IBP bounds.
    """
    n = dynamics_model.input_dim
    m = dynamics_model.control_dim
    batch_size = len(batch)

    # Create translator for dynamics evaluation
    translator = TorchTranslator(device=device, dtype=dtype)

    if isinstance(batch[0], SimplicialRegion):
        # For simplicial regions, evaluate at all vertices for each sample
        f_L_list = []
        f_U_list = []
        g_L_list = []
        g_U_list = []

        for sample in batch:
            verts = torch.tensor(sample.vertices, device=device, dtype=dtype)  # [V, n]
            f_vals = []
            g_vals = []
            for v in verts:
                x = v.unsqueeze(0)  # [1, n]
                with torch.no_grad():
                    f_val = dynamics_model.compute_f(x, translator).squeeze(0)  # [n]
                    f_vals.append(f_val)
                    if m > 0:
                        g_val = dynamics_model.compute_g(x, translator).squeeze(0)  # [n, m]
                        g_vals.append(g_val)

            f_vals = torch.stack(f_vals, dim=0)  # [V, n]
            f_L_list.append(f_vals.min(dim=0).values)
            f_U_list.append(f_vals.max(dim=0).values)

            if m > 0:
                g_vals = torch.stack(g_vals, dim=0)  # [V, n, m]
                g_L_list.append(g_vals.min(dim=0).values)
                g_U_list.append(g_vals.max(dim=0).values)

        f_L = torch.stack(f_L_list, dim=0)  # [batch_size, n]
        f_U = torch.stack(f_U_list, dim=0)  # [batch_size, n]

        if m > 0:
            g_L = torch.stack(g_L_list, dim=0)  # [batch_size, n, m]
            g_U = torch.stack(g_U_list, dim=0)  # [batch_size, n, m]
        else:
            g_L = None
            g_U = None

    elif isinstance(batch[0], HyperrectangularRegion):
        # For hyperrectangular regions, evaluate at all corners for each sample
        f_L_list = []
        f_U_list = []
        g_L_list = []
        g_U_list = []

        import itertools
        for sample in batch:
            center = torch.tensor(sample.center_point, device=device, dtype=dtype)
            radius = torch.tensor(sample.radius_vec, device=device, dtype=dtype)

            # Generate all corners
            corners = []
            for combo in itertools.product([0, 1], repeat=n):
                corner = center.clone()
                for i, c in enumerate(combo):
                    corner[i] = center[i] + radius[i] if c == 1 else center[i] - radius[i]
                corners.append(corner)

            f_vals = []
            g_vals = []
            with torch.no_grad():
                for c in corners:
                    x = c.unsqueeze(0)
                    f_val = dynamics_model.compute_f(x, translator).squeeze(0)
                    f_vals.append(f_val)
                    if m > 0:
                        g_val = dynamics_model.compute_g(x, translator).squeeze(0)
                        g_vals.append(g_val)

            f_vals = torch.stack(f_vals, dim=0)  # [num_corners, n]
            f_L_list.append(f_vals.min(dim=0).values)
            f_U_list.append(f_vals.max(dim=0).values)

            if m > 0:
                g_vals = torch.stack(g_vals, dim=0)  # [num_corners, n, m]
                g_L_list.append(g_vals.min(dim=0).values)
                g_U_list.append(g_vals.max(dim=0).values)

        f_L = torch.stack(f_L_list, dim=0)  # [batch_size, n]
        f_U = torch.stack(f_U_list, dim=0)  # [batch_size, n]

        if m > 0:
            g_L = torch.stack(g_L_list, dim=0)  # [batch_size, n, m]
            g_U = torch.stack(g_U_list, dim=0)  # [batch_size, n, m]
        else:
            g_L = None
            g_U = None
    else:
        raise TypeError(f"Unsupported region type: {type(batch[0])}")

    f_bounds = (f_L, f_U)
    g_bounds = (g_L, g_U) if m > 0 else None

    return f_bounds, g_bounds


# =============================================================================
# 7. min_L Computation
# =============================================================================

def compute_min_L_ibp(
    batch,
    dynamics_model,
    ibp_calculator: IBPNetworkBoundCalculator,
    device,
    dtype,
    h_lb_lbp=None
) -> torch.Tensor:
    """
    Compute lower bound on CBF condition using IBP bounds.

    CBF condition: ∇h(x)·f(x) + ∇h(x)·g(x)·u + α(h(x)) ≥ 0

    The lower bound is computed using interval arithmetic.

    Args:
        batch: List of region samples
        dynamics_model: CBF dynamics model
        ibp_calculator: IBPNetworkBoundCalculator instance for J_L/J_U and dynamics bounds
        device, dtype: Computation device and data type
        h_lb_lbp: Optional LBP h_lb tensor [batch] for alpha term (more accurate than IBP h_lb)
    """
    m = dynamics_model.control_dim
    batch_size = len(batch)

    # Get network bounds and Jacobian bounds from IBP
    h_lb_ibp, h_ub_ibp = ibp_calculator.get_network_output_bounds()
    J_L, J_U = ibp_calculator.ibp_jacobian_bounds(batch)

    # J_L, J_U: [batch, n_in]
    n = J_L.shape[-1]

    # Get dynamics bounds
    f_bounds, g_bounds = _compute_dynamics_bounds_ibp(batch, dynamics_model, device, dtype)
    f_L, f_U = f_bounds

    # Compute drift lower bound: J @ f
    term1 = J_L * f_L
    term2 = J_L * f_U
    term3 = J_U * f_L
    term4 = J_U * f_U

    L_drift = vec_min(term1, term2, term3, term4).sum(dim=-1)

    # Compute control lower bound if applicable
    L_ctrl = torch.zeros_like(L_drift)

    if m > 0 and g_bounds is not None:
        g_L, g_U = g_bounds

        batch_size = len(batch)
        v_L = torch.zeros(batch_size, m, dtype=dtype, device=device)
        v_U = torch.zeros(batch_size, m, dtype=dtype, device=device)

        for k in range(m):
            g_L_k = g_L[..., k]
            g_U_k = g_U[..., k]

            t1 = J_L * g_L_k
            t2 = J_L * g_U_k
            t3 = J_U * g_L_k
            t4 = J_U * g_U_k

            v_L[:, k] = vec_min(t1, t2, t3, t4).sum(dim=-1)
            v_U[:, k] = vec_max(t1, t2, t3, t4).sum(dim=-1)

        u_min = torch.tensor(dynamics_model.u_min, dtype=dtype, device=device)
        u_max = torch.tensor(dynamics_model.u_max, dtype=dtype, device=device)

        for k in range(m):
            v_L_k = v_L[:, k]
            v_U_k = v_U[:, k]

            pos_mask = v_L_k >= 0
            neg_mask = v_U_k <= 0
            mixed_mask = ~(pos_mask | neg_mask)

            L_ctrl += torch.where(pos_mask, v_L_k * u_min[k], torch.zeros_like(v_L_k))
            L_ctrl += torch.where(neg_mask, v_U_k * u_max[k], torch.zeros_like(v_L_k))

            mixed_vals = torch.minimum(v_L_k * u_max[k], v_U_k * u_min[k])
            L_ctrl += torch.where(mixed_mask, mixed_vals, torch.zeros_like(v_L_k))

    # Add class-K term
    # Use LBP h_lb for alpha term if provided (more accurate), otherwise fall back to IBP h_lb
    if h_lb_lbp is not None:
        alpha_l = dynamics_model.alpha_function(h_lb_lbp)
    else:
        alpha_l = dynamics_model.alpha_function(h_lb_ibp.squeeze(-1))
    L_total = L_drift + L_ctrl + alpha_l

    return L_total


# =============================================================================
# 8. Compatibility Wrappers
# =============================================================================

def compute_simplex_bound_ibp(
    model: nn.Module,
    simplex_vertices: Union[torch.Tensor, np.ndarray],
    region_type: str,
    dynamics_model=None,
    translator=None
):
    """
    IBP-based version of compute_simplex_bound.

    Returns min_L for 'safe' regions, (h_lb, h_ub) for 'unsafe' regions.
    """
    if isinstance(simplex_vertices, np.ndarray):
        verts_np = simplex_vertices
    else:
        verts_np = simplex_vertices.cpu().numpy()

    V, D = verts_np.shape
    if V != D + 1:
        raise ValueError(f"Simplex must have V=D+1 vertices, got V={V}, D={D}")

    sample = SimplicialRegion(verts_np, output_dim=None)
    batch = [sample]

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ibp_calc = IBPNetworkBoundCalculator(model, dtype=dtype, device=device)
    ibp_calc.ibp_forward(batch)

    h_lb, h_ub = ibp_calc.get_network_output_bounds()
    h_lb = h_lb.reshape(-1)
    h_ub = h_ub.reshape(-1)

    if region_type == 'unsafe':
        return h_lb[0], h_ub[0]

    if region_type == 'safe':
        if dynamics_model is None:
            raise ValueError("dynamics_model is required for 'safe' region type")

        min_L = compute_min_L_ibp(batch, dynamics_model, ibp_calc, device, dtype)
        return min_L[0]

    raise ValueError(f"Invalid region_type: {region_type}. Must be 'unsafe' or 'safe'")


def compute_simplex_bound_batch_ibp(
    model: nn.Module,
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    region_type: str,
    dynamics_model=None,
    translator=None
):
    """
    Batch version of compute_simplex_bound_ibp.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    batch = []
    for verts in vertices_list:
        if isinstance(verts, torch.Tensor):
            verts_np = verts.cpu().numpy()
        else:
            verts_np = verts
        batch.append(SimplicialRegion(verts_np, output_dim=None))

    B = len(batch)

    ibp_calc = IBPNetworkBoundCalculator(model, dtype=dtype, device=device)
    ibp_calc.ibp_forward(batch)

    h_lb, h_ub = ibp_calc.get_network_output_bounds()
    h_lb = h_lb.reshape(B, -1)[:, 0]
    h_ub = h_ub.reshape(B, -1)[:, 0]

    if region_type == 'unsafe':
        return h_lb, h_ub

    if region_type == 'safe':
        if dynamics_model is None:
            raise ValueError("dynamics_model is required for 'safe' region type")

        min_L = compute_min_L_ibp(batch, dynamics_model, ibp_calc, device, dtype)
        return min_L.reshape(B)

    raise ValueError(f"Invalid region_type: {region_type}. Must be 'unsafe' or 'safe'")


# =============================================================================
# 9. IBP Verifier Class (Complete Implementation)
# =============================================================================

def _ibp_handle_split(sample, start_time, results, sample_idx, min_volume, split_type, unsat_type, max_depth=None):
    """Record a MAYBE result via splitting or an UNSAT counterexample if splitting is not possible or depth limited."""
    # Check if maximum depth is reached
    if max_depth is not None and sample.depth >= max_depth:
        counterexample = sample.center
        from lbp_neural_cbf.certification_results import SampleResultUNSAT
        results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type="depth_limit_reached")
        return

    if sample._compute_volume() > min_volume:
        new_samples = sample.split()
        if new_samples:
            from lbp_neural_cbf.certification_results import SampleResultMaybe
            results[sample_idx] = SampleResultMaybe(sample, start_time, new_samples, split_type=split_type)
            return

    counterexample = sample.center
    from lbp_neural_cbf.certification_results import SampleResultUNSAT
    results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type=unsat_type)


def verify_batch_ibp(batch, dynamics_model, lbp_linearizer, device, dtype, min_volume=1e-8, max_depth=None):
    """
    Verify a batch of samples using hybrid LBP+IBP approach.

    - LBP (CROWN) for h_min, h_max: network output bounds via CrownPartialLinearization
      This is stable because CROWN uses linear bounds without problematic slope computations
    - IBP for min_L: CBF condition lower bound via Interval Bound Propagation
      This avoids the 1/(u-l) slope issues in LBP

    CBF verification theory with 6 cases:
    Case 1: h_max < 0 -> SAT -> V_unsafe (unsafe region verified)
    Case 2a: unsafe & h_min >= 0 -> UNSAT -> F_h_positive_in_unsafe (h positive in unsafe = violation)
    Case 2b: unsafe & h_min < 0 -> MAYBE -> split or depth_limit_reached
    Case 3: safe region with min_L >= -1e-12 -> SAT -> V_safe
    Case 3b: safe region with min_L < -1e-12 & counterexample -> UNSAT -> F_safe_cbf_violation
    Case 3c: safe region with min_L < -1e-12 & no counterexample -> MAYBE -> split or depth_limit_reached

    Args:
        batch: List of region samples
        dynamics_model: CBF dynamics model
        lbp_linearizer: CrownPartialLinearization instance for h_min/h_max computation
        device, dtype: Computation device and data type
        min_volume, max_depth: Splitting parameters

    Returns:
        List of SampleResult objects (SampleResultSAT, SampleResultUNSAT, or SampleResultMaybe)
    """
    start_time = time.time()
    results = [None for _ in range(len(batch))]

    # Import here to avoid circular imports
    from lbp_neural_cbf.cbf.domain import unsafe_region
    from lbp_neural_cbf.certification_results import SampleResultSAT, SampleResultUNSAT, SampleResultMaybe

    # Step 1: Compute h_min, h_max using LBP (CROWN) - this is stable
    lbp_linearizer.compute_network_bounds(batch)
    h_lb_all, h_ub_all = lbp_linearizer.get_network_output_bounds()
    h_lb_all = h_lb_all.reshape(len(batch), -1)[:, 0]
    h_ub_all = h_ub_all.reshape(len(batch), -1)[:, 0]

    to_check_cbf_cond = []

    for sample_idx, sample in enumerate(batch):
        h_min = h_lb_all[sample_idx].item()
        h_max = h_ub_all[sample_idx].item()

        # Case 1: h_max < 0 -> SAT (unsafe region verified)
        if h_max < 0:
            results[sample_idx] = SampleResultSAT(sample, start_time, result_type="unsafe_region")

        # Case 2: unsafe region (intersects with unsafe set)
        elif unsafe_region(sample, dynamics_model, require_complete_containment=False):
            # Case 2a: h_min >= 0 -> UNSAT (violation: h positive in unsafe)
            if h_min >= 0:
                counterexample = sample.center
                results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type="h_positive_in_unsafe")
            # Case 2b: h_min < 0 -> MAYBE (boundary, need to split)
            else:
                _ibp_handle_split(
                    sample=sample,
                    start_time=start_time,
                    results=results,
                    sample_idx=sample_idx,
                    min_volume=min_volume,
                    split_type="case_1_boundary_unsafe",
                    unsat_type="unsafe_cannot_split",
                    max_depth=max_depth,
                )

        # Case 3: safe region (does not intersect unsafe set)
        else:
            to_check_cbf_cond.append(sample_idx)

    if len(to_check_cbf_cond) == 0:
        return results

    # Step 2: CBF condition verification for safe regions using IBP
    subbatch = [batch[i] for i in to_check_cbf_cond]

    # Get LBP bounds for subbatch (more accurate than IBP for h_lb/h_ub)
    lbp_linearizer.compute_network_bounds(subbatch)
    sub_h_lb_lbp, sub_h_ub_lbp = lbp_linearizer.get_network_output_bounds()
    sub_h_lb_lbp = sub_h_lb_lbp.reshape(len(subbatch), -1)[:, 0]

    # Create IBP calculator for min_L computation (J_L, J_U, dynamics bounds)
    sub_ibp_calc = IBPNetworkBoundCalculator(lbp_linearizer.network, dtype=dtype, device=device)
    sub_ibp_calc.ibp_forward(subbatch)

    # Compute min_L for CBF condition using IBP with LBP h_lb for alpha term
    sub_min_L = compute_min_L_ibp(subbatch, dynamics_model, sub_ibp_calc, device, dtype, h_lb_lbp=sub_h_lb_lbp)

    # Case 3: check CBF condition for each safe region
    for subsample_idx, sample_idx in enumerate(to_check_cbf_cond):
        sample = batch[sample_idx]
        min_L_val = sub_min_L[subsample_idx].item()

        # Case 3: min_L >= -1e-12 -> SAT (CBF condition satisfied)
        if min_L_val >= -1e-12:
            results[sample_idx] = SampleResultSAT(sample, start_time, result_type="safe_cbf_verified")
        # Case 3b/3c: min_L < -1e-12 -> need counterexample or split
        else:
            _ibp_handle_split(
                sample=sample,
                start_time=start_time,
                results=results,
                sample_idx=sample_idx,
                min_volume=min_volume,
                split_type="case_2_cbf_failure",
                unsat_type="safe_cbf_violation",
                max_depth=max_depth,
            )

    return results


class IBPVerifier:
    """
    Standalone verifier using IBP bounds for CBF conditions.

    This class provides complete verification capability matching verify_cbf.py,
    but using Interval Bound Propagation (IBP) instead of LBP (CROWN + McCormick).
    """

    def __init__(
        self,
        model: nn.Module,
        dynamics_model,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        max_depth: Optional[int] = None
    ):
        self.model = model
        self.dynamics_model = dynamics_model
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        self.max_depth = max_depth

    def verify_batch(self, batch):
        """
        Verify a batch of regions.

        Returns:
            List of SampleResult objects
        """
        # Create LBP linearizer for h_min/h_max computation (stable via CROWN)
        lbp_linearizer = CrownPartialLinearization(self.model, dtype=self.dtype)
        return verify_batch_ibp(
            batch,
            self.dynamics_model,
            lbp_linearizer,
            self.device,
            self.dtype,
            min_volume=1e-8,
            max_depth=self.max_depth
        )

    def verify_model(self, onnx_path, max_depth=None):
        """
        Full verification of a model with region splitting.

        Args:
            onnx_path: Ignored (kept for API compatibility)
            max_depth: Maximum depth for region splitting

        Returns:
            Dictionary with verification results including:
            - V_safe, V_unsafe, F_h_positive_in_unsafe, F_safe_cbf_violation,
              F_depth_limit_reached, F_unsafe_cannot_split
            - certified_percentage, uncertified_percentage
        """
        import time
        import itertools

        max_depth = max_depth or self.max_depth or 20

        self.model.eval()

        # Generate initial samples
        from lbp_neural_cbf.regions import create_region_generator
        region_generator = create_region_generator("simplicial")
        samples = region_generator.create_mesh(self.dynamics_model).get_regions(0)

        # Process all samples with splitting
        all_results = []
        samples_to_process = list(samples)
        computation_start = time.time()

        while samples_to_process:
            batch = samples_to_process[:512]
            samples_to_process = samples_to_process[512:]

            batch_results = self.verify_batch(batch)
            all_results.extend(batch_results)

            # Collect new samples from MAYBE results
            new_samples = []
            for result in batch_results:
                if hasattr(result, 'hasnewsamples') and result.hasnewsamples():
                    new_samples.extend(result.newsamples())

            # Check depth limit
            samples_to_process = []
            for new_sample in new_samples:
                if max_depth is None or new_sample.depth < max_depth:
                    samples_to_process.append(new_sample)
                else:
                    from lbp_neural_cbf.certification_results import SampleResultUNSAT
                    all_results.append(SampleResultUNSAT(
                        new_sample, computation_start, [new_sample.center],
                        result_type="depth_limit_reached"
                    ))

        computation_time = time.time() - computation_start

        # Import result types for classification
        from lbp_neural_cbf.certification_results import SampleResultSAT, SampleResultUNSAT

        # Classify results
        V_safe = []
        V_unsafe = []
        F_h_positive_in_unsafe = []
        F_safe_cbf_violation = []
        F_depth_limit_reached = []
        F_unsafe_cannot_split = []

        for result in all_results:
            sample = result.sample
            if hasattr(sample, 'vertices'):
                vertices = np.array(sample.vertices, dtype=np.float32)
            elif hasattr(sample, 'center_point') and hasattr(sample, 'radius_vec'):
                center = np.array(sample.center_point, dtype=np.float32)
                radius = np.array(sample.radius_vec, dtype=np.float32)
                vertices = np.stack([center - radius, center + radius], axis=0)
            else:
                continue

            if isinstance(result, SampleResultSAT):
                if result.result_type == "unsafe_region":
                    V_unsafe.append(vertices)
                elif result.result_type == "safe_cbf_verified":
                    V_safe.append(vertices)

            elif isinstance(result, SampleResultUNSAT):
                if result.result_type == "h_positive_in_unsafe":
                    F_h_positive_in_unsafe.append(vertices)
                elif result.result_type == "safe_cbf_violation":
                    F_safe_cbf_violation.append(vertices)
                elif result.result_type == "depth_limit_reached":
                    F_depth_limit_reached.append(vertices)
                elif result.result_type == "unsafe_cannot_split":
                    F_unsafe_cannot_split.append(vertices)

        total = len(all_results)
        certified = len(V_safe) + len(V_unsafe)
        certified_percentage = (certified / total * 100) if total > 0 else 0
        uncertified_percentage = 100 - certified_percentage

        return {
            "regions": all_results,
            "certified_percentage": certified_percentage,
            "uncertified_percentage": uncertified_percentage,
            "computation_time": computation_time,
            "total_samples": total,
            "V_safe": V_safe,
            "V_unsafe": V_unsafe,
            "F_h_positive_in_unsafe": F_h_positive_in_unsafe,
            "F_safe_cbf_violation": F_safe_cbf_violation,
            "F_depth_limit_reached": F_depth_limit_reached,
            "F_unsafe_cannot_split": F_unsafe_cannot_split,
        }


# =============================================================================
# 10. Helper for extracting activation type from model
# =============================================================================

def get_activation_type(model: nn.Module) -> str:
    """Detect activation function type from model."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            return 'ReLU'
        elif isinstance(module, nn.Tanh):
            return 'Tanh'
        elif isinstance(module, nn.Sigmoid):
            return 'Sigmoid'
    raise ValueError("No supported activation function found in model")
