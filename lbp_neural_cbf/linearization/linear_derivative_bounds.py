import numpy as np
import torch
import torch.nn.functional as F

from ..certification_results import AugmentedSample
from ..regions import HyperrectangularRegion, SimplicialRegion

from .activations import ActivationRelaxation, LeakyReLUActivationRelaxation, ReLUActivationRelaxation, SigmoidActivationRelaxation, TanhActivationRelaxation


class CrownPartialLinearization:

    def __init__(self, network, activation_relaxation: ActivationRelaxation = None, dtype=torch.float32):
        """
        Initialize CROWN linearization for computing partial derivative bounds.

        :param network: Neural network (torch.nn.Sequential or similar)
        :param activation_relaxation: An object providing relaxation methods for activations.
                                      If None, will be automatically detected from the network.
        :param debug: If True, enable debug output.
        """
        self.network = network
        self.device = torch.device("cuda:0" if next(network.parameters()).is_cuda else "cpu")
        self.dtype = dtype
        self.fc_layers = self._extract_linear_layers()

        if activation_relaxation is None:
            self.activation_relaxation = self._detect_activation_relaxation()
        else:
            self.activation_relaxation = activation_relaxation

        self.forward_bounds = {}
        self.derivative_bounds = {}

    def _extract_linear_layers(self):
        """Extract linear layers from a PyTorch network for verification."""
        layers = []

        # General PyTorch model using .modules() is the most robust way
        for layer in self.network.modules():
            if isinstance(layer, torch.nn.Linear):
                layers.append(layer)

        return layers

    def _detect_activation_relaxation(self) -> ActivationRelaxation:
        """
        Automatically detect the activation function used in the network and return
        the appropriate activation relaxation object.
        """
        activation_types = set()
        all_modules = list(self.network.modules())[1:]  # Exclude the top-level container

        for i, module in enumerate(all_modules):
            # Skip the last layer as it has no activation
            if isinstance(module, torch.nn.Linear) and i == len(all_modules) - 1:
                continue

            if isinstance(module, torch.nn.ReLU):
                activation_types.add("relu")
            elif isinstance(module, torch.nn.Tanh):
                activation_types.add("tanh")
            elif isinstance(module, torch.nn.Sigmoid):
                activation_types.add("sigmoid")
            elif isinstance(module, torch.nn.LeakyReLU):
                activation_types.add("leaky_relu")

        if not activation_types:
            raise ValueError("No supported activation function detected.")
        if len(activation_types) > 1:
            raise ValueError(f"Multiple activation types detected: {activation_types}.")

        activation_type = activation_types.pop()

        if activation_type == "relu":
            return ReLUActivationRelaxation()
        if activation_type == "tanh":
            return TanhActivationRelaxation()
        if activation_type == "sigmoid":
            return SigmoidActivationRelaxation()
        if activation_type == "leaky_relu":
            negative_slope = next((m.negative_slope for m in all_modules if isinstance(m, torch.nn.LeakyReLU)), 0.01)
            return LeakyReLUActivationRelaxation(negative_slope=negative_slope)

        raise ValueError(f"Unsupported activation type: {activation_type}")

    def _compute_network_bounds(self, batch):
        """
        Internal method to compute bounds for pre-activation (y) and post-activation (z) values for each layer.
        """
        self.forward_bounds = {}

        if isinstance(batch[0], HyperrectangularRegion):
            centroids = [torch.tensor(sample.centroid, dtype=self.dtype, device=self.device) for sample in batch]
            center = torch.stack(centroids)
            radiuses = [torch.tensor(sample.radius_vec, dtype=self.dtype, device=self.device) for sample in batch]
            radius = torch.stack(radiuses)
            self.x_ub, self.x_lb = center + radius, center - radius
        elif isinstance(batch[0], SimplicialRegion):
            # For debugging purposes, we can also compute the min/max from vertices
            # bounds_list = [sample.get_bounds() for sample in batch]
            # mins = [torch.tensor(b[0], dtype=self.dtype, device=self.device) for b in bounds_list]
            # min = torch.stack(mins, dim=0)
            # maxs = [torch.tensor(b[1], dtype=self.dtype, device=self.device) for b in bounds_list]
            # max = torch.stack(maxs, dim=0)
            # center = (min + max) / 2.0
            # radius = (max - min) / 2.0
            vertices = []
            for sample in batch:
                if isinstance(sample, SimplicialRegion):
                    vertices.append(torch.tensor(sample.vertices, dtype=self.dtype, device=self.device))
            vertices = torch.stack(vertices)
            self.x_ub, self.x_lb = vertices.max(dim=-2).values, vertices.min(dim=-2).values
        else:
            raise TypeError(f"Unsupported region type: {type(batch[0])}.")

        input_dim = self.fc_layers[0].in_features
        A_L = A_U = torch.eye(input_dim, device=self.device, dtype=self.dtype)
        a_L = a_U = torch.zeros(input_dim, device=self.device, dtype=self.dtype)

        for i, layer in enumerate(self.fc_layers):
            W, b = layer.weight, layer.bias

            W_pos = F.relu(W)
            W_neg = W - W_pos  # More efficient than a second clamp/relu

            A_y_L = W_pos @ A_L + W_neg @ A_U
            a_y_L = (W_pos @ a_L.unsqueeze(-1)).squeeze(-1) + (W_neg @ a_U.unsqueeze(-1)).squeeze(-1) + b
            A_y_U = W_pos @ A_U + W_neg @ A_L
            a_y_U = (W_pos @ a_U.unsqueeze(-1)).squeeze(-1) + (W_neg @ a_L.unsqueeze(-1)).squeeze(-1) + b

            if isinstance(batch[0], HyperrectangularRegion):
                y_lb = ((A_y_L @ center.unsqueeze(-1)).squeeze(-1) + a_y_L) - (torch.abs(A_y_L) @ radius.unsqueeze(-1)).squeeze(-1)
                y_ub = ((A_y_U @ center.unsqueeze(-1)).squeeze(-1) + a_y_U) + (torch.abs(A_y_U) @ radius.unsqueeze(-1)).squeeze(-1)
            elif isinstance(batch[0], SimplicialRegion):
                vertex_lb = A_y_L @ vertices.transpose(-2, -1)
                vertex_ub = A_y_U @ vertices.transpose(-2, -1)

                prj_vertex_lb = vertex_lb + a_y_L.unsqueeze(-1)
                prj_vertex_ub = vertex_ub + a_y_U.unsqueeze(-1)

                y_lb = vertex_lb.min(dim=-1).values + a_y_L
                y_ub = vertex_ub.max(dim=-1).values + a_y_U
            else:
                raise TypeError(f"Unsupported region type: {type(batch[0])}.")

            self.forward_bounds[f"layer_{i}_pre_act_bounds"] = {
                "lb": y_lb,
                "ub": y_ub,
                "A_L": A_y_L,
                "a_L": a_y_L,
                "A_U": A_y_U,
                "a_U": a_y_U,
                "prj_vertex_lb": prj_vertex_lb if isinstance(batch[0], SimplicialRegion) else None,
                "prj_vertex_ub": prj_vertex_ub if isinstance(batch[0], SimplicialRegion) else None,
            }

            if i == len(self.fc_layers) - 1:
                A_L, a_L, A_U, a_U = A_y_L, a_y_L, A_y_U, a_y_U
                current_lb, current_ub = y_lb, y_ub
            else:
                alpha_L, beta_L, alpha_U, beta_U = self.activation_relaxation.relax_activation(y_lb, y_ub)
                alpha_L_pos = F.relu(alpha_L)
                alpha_L_neg = alpha_L - alpha_L_pos
                alpha_U_pos = F.relu(alpha_U)
                alpha_U_neg = alpha_U - alpha_U_pos

                A_L = alpha_L_pos.unsqueeze(-1) * A_y_L + alpha_L_neg.unsqueeze(-1) * A_y_U
                a_L = alpha_L_pos * a_y_L + alpha_L_neg * a_y_U + beta_L
                A_U = alpha_U_pos.unsqueeze(-1) * A_y_U + alpha_U_neg.unsqueeze(-1) * A_y_L
                a_U = alpha_U_pos * a_y_U + alpha_U_neg * a_y_L + beta_U

                if isinstance(batch[0], HyperrectangularRegion):
                    current_lb = ((A_L @ center.unsqueeze(-1)).squeeze(-1) + a_L) - (torch.abs(A_L) @ radius.unsqueeze(-1)).squeeze(-1)
                    current_ub = ((A_U @ center.unsqueeze(-1)).squeeze(-1) + a_U) + (torch.abs(A_U) @ radius.unsqueeze(-1)).squeeze(-1)
                elif isinstance(batch[0], SimplicialRegion):
                    vertex_lb = A_L @ vertices.transpose(-2, -1)
                    vertex_ub = A_U @ vertices.transpose(-2, -1)

                    prj_vertex_lb = vertex_lb + a_L.unsqueeze(-1)
                    prj_vertex_ub = vertex_ub + a_U.unsqueeze(-1)

                    current_lb = vertex_lb.min(dim=-1).values + a_L
                    current_ub = vertex_ub.max(dim=-1).values + a_U
                else:
                    raise TypeError(f"Unsupported region type: {type(batch[0])}.")

            # Store final bounds for the post-activation of layer i
            self.forward_bounds[f"layer_{i}_post_act_bounds"] = {
                "lb": current_lb,
                "ub": current_ub,
                "A_L": A_L,
                "a_L": a_L,
                "A_U": A_U,
                "a_U": a_U,
                "prj_vertex_lb": prj_vertex_lb if isinstance(batch[0], SimplicialRegion) else None,
                "prj_vertex_ub": prj_vertex_ub if isinstance(batch[0], SimplicialRegion) else None,
            }

    def compute_network_bounds(self, batch):
        self._compute_network_bounds(batch)

    def keep_indices(self, indices, include_partial_deriv_bounds=False):
        """Keep only the specified indices in the stored bounds."""
        if not self.forward_bounds:
            raise ValueError("No network bounds computed. Call compute_network_bounds() first.")

        for key in self.forward_bounds:
            for bound_key in ["lb", "ub", "a_L", "a_U"]:
                if self.forward_bounds[key][bound_key].ndim == 2:
                    self.forward_bounds[key][bound_key] = self.forward_bounds[key][bound_key][indices]

            for bound_key in ["A_L", "A_U", "prj_vertex_lb", "prj_vertex_ub"]:
                if self.forward_bounds[key][bound_key].ndim == 3:
                    self.forward_bounds[key][bound_key] = self.forward_bounds[key][bound_key][indices]

        if include_partial_deriv_bounds and self.derivative_bounds:
            for bound_key in ["A_L", "A_U", "b_L", "b_U"]:
                self.derivative_bounds[bound_key] = self.derivative_bounds[bound_key][indices]

    def get_network_output_bounds(self, sample_idx=None):
        if not self.forward_bounds:
            raise ValueError("No network bounds computed. Call compute_network_bounds() first.")

        final_layer_idx = len(self.fc_layers) - 1
        final_bounds = self.forward_bounds[f"layer_{final_layer_idx}_pre_act_bounds"]

        if sample_idx is None:
            return final_bounds["lb"], final_bounds["ub"]

        return final_bounds["lb"][sample_idx].item(), final_bounds["ub"][sample_idx].item()

    def get_network_output_bounds_with_grad(self, sample_idx=None):
        """
        返回网络输出边界，保留梯度追踪（用于修复算法的反向传播）。

        与 get_network_output_bounds 的区别：
        - sample_idx 为 None 时：直接返回存储的 tensor（已保留梯度）
        - sample_idx 不为 None 时：返回切片后的 tensor（而非 .item() 的标量）

        返回:
            (h_lb, h_ub): 均为 torch.Tensor, requires_grad=True
        """
        if not self.forward_bounds:
            raise ValueError("No network bounds computed. Call compute_network_bounds() first.")

        final_layer_idx = len(self.fc_layers) - 1
        final_bounds = self.forward_bounds[f"layer_{final_layer_idx}_pre_act_bounds"]

        if sample_idx is None:
            return final_bounds["lb"], final_bounds["ub"]

        # 不使用 .item()，直接切片 tensor，clone 副本 + requires_grad_(True) 成为叶子节点
        h_lb = final_bounds["lb"][sample_idx].clone().requires_grad_(True)
        h_ub = final_bounds["ub"][sample_idx].clone().requires_grad_(True)
        return h_lb, h_ub

    def get_network_linear_bounds(self, sample_idx=None):
        if not self.forward_bounds:
            raise ValueError("No network bounds computed. Call compute_network_bounds() first.")

        final_layer_idx = len(self.fc_layers) - 1
        final_bounds = self.forward_bounds[f"layer_{final_layer_idx}_pre_act_bounds"]

        A_L, a_L = final_bounds["A_L"], final_bounds["a_L"]
        A_U, a_U = final_bounds["A_U"], final_bounds["a_U"]

        if sample_idx is None:
            return (A_L, a_L), (A_U, a_U)

        return (A_L[sample_idx], a_L[sample_idx]), (A_U[sample_idx], a_U[sample_idx])

    def clear_bounds(self):
        """Clear stored forward bounds and sample tracking."""
        self.forward_bounds = {}
        self.derivative_bounds = {}

    def get_partial_derivative_bounds(self, sample_idx=None):
        if not self.derivative_bounds:
            raise ValueError("No derivative bounds computed. Call compute_partial_derivative_bounds() first.")

        if sample_idx is None:
            return self.derivative_bounds["A_L"], self.derivative_bounds["b_L"], self.derivative_bounds["A_U"], self.derivative_bounds["b_U"]

        return (
            self.derivative_bounds["A_L"][sample_idx],
            self.derivative_bounds["b_L"][sample_idx],
            self.derivative_bounds["A_U"][sample_idx],
            self.derivative_bounds["b_U"][sample_idx],
        )

    def compute_partial_derivative_bounds(self, input_idx, output_idx=None):
        L = len(self.fc_layers)

        A_L_running, b_L_running, A_U_running, b_U_running = self._get_jacobian_bounds_for_layer(L)
        # DEBUG
        # print(f"    DEBUG: After _get_jacobian_bounds_for_layer: A_L_running={A_L_running.shape}, b_L_running={b_L_running.shape}")
        A_L_running, b_L_running, A_U_running, b_U_running = (
            A_L_running.unsqueeze(0),
            b_L_running.unsqueeze(0),
            A_U_running.unsqueeze(0),
            b_U_running.unsqueeze(0),
        )  # Add batch dimension
        # DEBUG
        # print(f"    DEBUG: After unsqueeze: A_L_running={A_L_running.shape}, b_L_running={b_L_running.shape}")

        for i in range(L - 1, 0, -1):
            Lambda_L, lambda_L, Lambda_U, lambda_U = self._get_jacobian_bounds_for_layer(i)

            # 2. Get bounds for the common variable, y_i
            pre_act_bounds = self.forward_bounds[f"layer_{i-1}_pre_act_bounds"]

            # 3. Bound the matrix product M^(i) = M^(i+1) * J^(i).
            # The result, A/b_new, will be bounds for M^(i) w.r.t y_i.
            A_L_new, b_L_new, A_U_new, b_U_new = self._vectorized_mccormick_product(
                (A_L_running, b_L_running, A_U_running, b_U_running),
                (Lambda_L, lambda_L, Lambda_U, lambda_U),
                pre_act_bounds,
            )

            # 4. Propagate the new product bounds to be functions of y_{i-1} (or x)
            # This prepares A/b_running for the next iteration of the loop.
            A_L_running, b_L_running, A_U_running, b_U_running = self._propagate_bounds_one_layer(i, A_L_new, b_L_new, A_U_new, b_U_new)

        A_L_final, b_L_final, A_U_final, b_U_final = A_L_running, b_L_running, A_U_running, b_U_running

        if input_idx is None and output_idx is None:
            A_L, b_L, A_U, b_U = A_L_final, b_L_final, A_U_final, b_U_final
        elif input_idx is None:
            A_L, b_L, A_U, b_U = A_L_final[:, output_idx], b_L_final[:, output_idx], A_U_final[:, output_idx], b_U_final[:, output_idx]
        elif output_idx is None:
            A_L, b_L, A_U, b_U = A_L_final[:, :, input_idx], b_L_final[:, input_idx], A_U_final[:, :, input_idx], b_U_final[:, input_idx]
        else:
            A_L, b_L, A_U, b_U = (
                A_L_final[:, output_idx, input_idx],
                b_L_final[:, output_idx, input_idx],
                A_U_final[:, output_idx, input_idx],
                b_U_final[:, output_idx, input_idx],
            )

        self.derivative_bounds = {
            "A_L": A_L,
            "b_L": b_L,
            "A_U": A_U,
            "b_U": b_U,
        }
        # DEBUG
        # print(f"    DEBUG get_partial_deriv: A_L={A_L.shape}, b_L={b_L.shape}")

    def _get_jacobian_bounds_for_layer(self, i):
        W_i = self.fc_layers[i - 1].weight
        n_out_i, n_in_i = W_i.shape

        if i == len(self.fc_layers):  # Final layer is linear
            zeros = torch.zeros((n_out_i, n_in_i, n_in_i), device=self.device, dtype=self.dtype)
            return zeros, W_i, zeros, W_i

        y_i_lb = self.forward_bounds[f"layer_{i-1}_pre_act_bounds"]["lb"]
        y_i_ub = self.forward_bounds[f"layer_{i-1}_pre_act_bounds"]["ub"]

        S_L, s_L, S_U, s_U = self.activation_relaxation.relax_activation_derivative(y_i_lb, y_i_ub)

        W_i_pos = F.relu(W_i)
        W_i_neg = W_i - W_i_pos

        # Create the (p, k) terms that will form the diagonal
        term_L = W_i_pos * S_L.unsqueeze(-1) + W_i_neg * S_U.unsqueeze(-1)
        term_U = W_i_pos * S_U.unsqueeze(-1) + W_i_neg * S_L.unsqueeze(-1)

        # (b, p, k) -> transpose -> (b, k, p) -> diag_embed -> (b, k, p, p) -> permute -> (b, p, k, p)
        Lambda_L = torch.diag_embed(term_L.transpose(-2, -1)).permute(0, 2, 1, 3)
        Lambda_U = torch.diag_embed(term_U.transpose(-2, -1)).permute(0, 2, 1, 3)

        lambda_L = W_i_pos * s_L.unsqueeze(-1) + W_i_neg * s_U.unsqueeze(-1)
        lambda_U = W_i_pos * s_U.unsqueeze(-1) + W_i_neg * s_L.unsqueeze(-1)

        return Lambda_L, lambda_L, Lambda_U, lambda_U

    def _propagate_bounds_one_layer(self, i, A_L_in, b_L_in, A_U_in, b_U_in):
        # Base case: propagate to input `x`
        if i <= 1:  # The original code had i > 1, this should be i <= 1 for base case
            W_0, b_0 = self.fc_layers[0].weight, self.fc_layers[0].bias
            K_L = K_U = W_0.unsqueeze(0)
            k_L = k_U = b_0.unsqueeze(0)
        # Recursive step: propagate to previous layer's pre-activation `y_{i-1}`
        else:
            W_i, b_i = self.fc_layers[i - 1].weight, self.fc_layers[i - 1].bias
            y_prev_lb = self.forward_bounds[f"layer_{i-2}_pre_act_bounds"]["lb"]
            y_prev_ub = self.forward_bounds[f"layer_{i-2}_pre_act_bounds"]["ub"]
            G_L, g_L, G_U, g_U = self.activation_relaxation.relax_activation(y_prev_lb, y_prev_ub)  # TODO: Rely on stored bounds (need to store them first)

            W_i_pos = F.relu(W_i)
            W_i_neg = W_i - W_i_pos

            K_L = W_i_pos * G_L.unsqueeze(-2) + W_i_neg * G_U.unsqueeze(-2)
            k_L = (W_i_pos @ g_L.unsqueeze(-1)).squeeze(-1) + (W_i_neg @ g_U.unsqueeze(-1)).squeeze(-1) + b_i
            K_U = W_i_pos * G_U.unsqueeze(-2) + W_i_neg * G_L.unsqueeze(-2)
            k_U = (W_i_pos @ g_U.unsqueeze(-1)).squeeze(-1) + (W_i_neg @ g_L.unsqueeze(-1)).squeeze(-1) + b_i

        A_L_in_pos = F.relu(A_L_in)
        A_L_in_neg = A_L_in - A_L_in_pos
        A_U_in_pos = F.relu(A_U_in)
        A_U_in_neg = A_U_in - A_U_in_pos

        Pi_L = A_L_in_pos @ K_L.unsqueeze(1) + A_L_in_neg @ K_U.unsqueeze(1)
        pi_L = (A_L_in_pos @ k_L.unsqueeze(-2).unsqueeze(-1)).squeeze(-1) + (A_L_in_neg @ k_U.unsqueeze(-2).unsqueeze(-1)).squeeze(-1) + b_L_in
        Pi_U = A_U_in_pos @ K_U.unsqueeze(1) + A_U_in_neg @ K_L.unsqueeze(1)
        pi_U = (A_U_in_pos @ k_U.unsqueeze(-2).unsqueeze(-1)).squeeze(-1) + (A_U_in_neg @ k_L.unsqueeze(-2).unsqueeze(-1)).squeeze(-1) + b_U_in

        return Pi_L, pi_L, Pi_U, pi_U

    def _vectorized_mccormick_product(self, M_old_bounds, J_local_bounds, pre_act_bounds, eta=0.5, nu=0.5):
        """
        Computes vectorized McCormick relaxations for the product of two matrix-valued affine bounds.

        Implements the mathematical formula:
        (J^(i+1))_{jp}(J^(i))_{pk} >=
            (eta*J^(i+1)_L + (1-eta)*J^(i+1)_U)_{jp}^+ * (Lambda_L * y + lambda_L)_{pk} +
            (eta*J^(i+1)_L + (1-eta)*J^(i+1)_U)_{jp}^- * (Lambda_U * y + lambda_U)_{pk} +
            (Pi_L * y + pi_L)_{jp} * (eta*J^(i)_L + (1-eta)*J^(i)_U)_{pk}^+ +
            (Pi_U * y + pi_U)_{jp} * (eta*J^(i)_L + (1-eta)*J^(i)_U)_{pk}^- -
            eta*J^(i+1)_L*J^(i)_L - (1-eta)*J^(i+1)_U*J^(i)_U
        """
        Pi_L, pi_L, Pi_U, pi_U = M_old_bounds
        Lambda_L, lambda_L, Lambda_U, lambda_U = J_local_bounds

        if pre_act_bounds["prj_vertex_lb"] is not None:
            #   Pi_*:   [b, j, p, m]
            #   pi_*:   [b, j, p]
            #   Lambda_*:[b, p, k, m]
            #   lambda_*:[b, p, k]
            #   prj_vertex_lb/ub: [b, m, V]  (V = number of vertices stored for lb/ub; could vary)
            #
            prj_vertex_lb = pre_act_bounds["prj_vertex_lb"]
            prj_vertex_ub = pre_act_bounds["prj_vertex_ub"]

            # Evaluate Pi_L and Pi_U at all vertices:
            # einsum 'bjpm,bmv->bjpv' -> result [b, j, p, V]
            Pi_L_pos = F.relu(Pi_L)
            Pi_L_neg = Pi_L - Pi_L_pos
            Pi_U_pos = F.relu(Pi_U)
            Pi_U_neg = Pi_U - Pi_U_pos
            J_Pi_L_v = (
                torch.einsum("bjpm,bmv->bjpv", Pi_L_pos, prj_vertex_lb) + torch.einsum("bjpm,bmv->bjpv", Pi_L_neg, prj_vertex_ub) + pi_L.unsqueeze(-1)
            )  # [b, j, p, V]
            J_Pi_U_v = (
                torch.einsum("bjpm,bmv->bjpv", Pi_U_pos, prj_vertex_ub) + torch.einsum("bjpm,bmv->bjpv", Pi_U_neg, prj_vertex_lb) + pi_U.unsqueeze(-1)
            )  # [b, j, p, V]

            # Lower/upper across vertices
            J_Pi_L, _ = torch.min(J_Pi_L_v, dim=-1)  # [b, j, p]
            J_Pi_U, _ = torch.max(J_Pi_U_v, dim=-1)  # [b, j, p]

            # Evaluate Lambda_L / Lambda_U at vertices: einsum 'bpkm,bmv->bpkv' -> [b, p, k, V]
            Lambda_L_pos = F.relu(Lambda_L)
            Lambda_L_neg = Lambda_L - Lambda_L_pos
            Lambda_U_pos = F.relu(Lambda_U)
            Lambda_U_neg = Lambda_U - Lambda_U_pos
            J_Lambda_L_v = (
                torch.einsum("bpkm,bmv->bpkv", Lambda_L_pos, prj_vertex_lb)
                + torch.einsum("bpkm,bmv->bpkv", Lambda_L_neg, prj_vertex_ub)
                + lambda_L.unsqueeze(-1)
            )  # [b, p, k, V]
            J_Lambda_U_v = (
                torch.einsum("bpkm,bmv->bpkv", Lambda_U_pos, prj_vertex_ub)
                + torch.einsum("bpkm,bmv->bpkv", Lambda_U_neg, prj_vertex_lb)
                + lambda_U.unsqueeze(-1)
            )  # [b, p, k, V]

            # Lower/upper across vertices for Lambda
            J_Lambda_L, _ = torch.min(J_Lambda_L_v, dim=-1)  # [b, p, k]
            J_Lambda_U, _ = torch.max(J_Lambda_U_v, dim=-1)  # [b, p, k]
        else:
            y_L = pre_act_bounds["lb"]
            y_U = pre_act_bounds["ub"]
            center, radius = (y_U + y_L) / 2.0, (y_U - y_L) / 2.0
            J_Pi_L = (Pi_L @ center.unsqueeze(-2).unsqueeze(-1)).squeeze(-1) + pi_L - (torch.abs(Pi_L) @ radius.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
            J_Pi_U = (Pi_U @ center.unsqueeze(-2).unsqueeze(-1)).squeeze(-1) + pi_U + (torch.abs(Pi_U) @ radius.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
            J_Lambda_L = (
                (Lambda_L @ center.unsqueeze(-2).unsqueeze(-1)).squeeze(-1) + lambda_L - (torch.abs(Lambda_L) @ radius.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
            )
            J_Lambda_U = (
                (Lambda_U @ center.unsqueeze(-2).unsqueeze(-1)).squeeze(-1) + lambda_U + (torch.abs(Lambda_U) @ radius.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
            )

        # --- Lower Bound ---
        # eta_J_Pi and eta_J_Lambda have shape [j, p] and [p, k] respectively
        eta_J_Pi = eta * J_Pi_L + (1 - eta) * J_Pi_U  # [j, p]
        eta_J_Lambda = eta * J_Lambda_L + (1 - eta) * J_Lambda_U  # [p, k]

        eta_J_Pi_pos = F.relu(eta_J_Pi)
        eta_J_Pi_neg = eta_J_Pi - eta_J_Pi_pos
        eta_J_Lambda_pos = F.relu(eta_J_Lambda)
        eta_J_Lambda_neg = eta_J_Lambda - eta_J_Lambda_pos

        # Note: eta_J_Pi is [b, j, p], eta_J_Lambda is [b, p, k], Lambda is [b, p, k, m], Pi is [b, j, p, m]
        A_L_new = (
            torch.einsum("bjp,bpkm->bjkm", eta_J_Pi_pos, Lambda_L)
            + torch.einsum("bjp,bpkm->bjkm", eta_J_Pi_neg, Lambda_U)
            + torch.einsum("bpk,bjpm->bjkm", eta_J_Lambda_pos, Pi_L)
            + torch.einsum("bpk,bjpm->bjkm", eta_J_Lambda_neg, Pi_U)
        )

        b_L_new = (
            eta_J_Pi_pos @ lambda_L
            + eta_J_Pi_neg @ lambda_U
            + pi_L @ eta_J_Lambda_pos  # pi_L is [b, j, p], eta_J_Lambda_pos is [b, p, k] -> result [b, j, k]
            + pi_U @ eta_J_Lambda_neg
            - (eta * (J_Pi_L @ J_Lambda_L) + (1 - eta) * (J_Pi_U @ J_Lambda_U))
        )

        # --- Upper Bound ---
        nu_J_Pi = nu * J_Pi_U + (1 - nu) * J_Pi_L  # [b, j, p]
        nu_J_Lambda = nu * J_Lambda_L + (1 - nu) * J_Lambda_U  # [b, p, k]

        nu_J_Pi_pos, nu_J_Pi_neg = F.relu(nu_J_Pi), nu_J_Pi - F.relu(nu_J_Pi)
        nu_J_Lambda_pos, nu_J_Lambda_neg = F.relu(nu_J_Lambda), nu_J_Lambda - F.relu(nu_J_Lambda)

        A_U_new = (
            torch.einsum("bjp,bpkm->bjkm", nu_J_Pi_pos, Lambda_U)
            + torch.einsum("bjp,bpkm->bjkm", nu_J_Pi_neg, Lambda_L)
            + torch.einsum("bpk,bjpm->bjkm", nu_J_Lambda_pos, Pi_U)
            + torch.einsum("bpk,bjpm->bjkm", nu_J_Lambda_neg, Pi_L)
        )

        b_U_new = (
            nu_J_Pi_pos @ lambda_U
            + nu_J_Pi_neg @ lambda_L
            + pi_U @ nu_J_Lambda_pos  # pi_U is [b, j, p], nu_J_Lambda_pos is [b, p, k] -> result [b, j, k]
            + pi_L @ nu_J_Lambda_neg
            - (nu * (J_Pi_U @ J_Lambda_L) + (1 - nu) * (J_Pi_L @ J_Lambda_U))
        )

        return A_L_new, b_L_new, A_U_new, b_U_new
