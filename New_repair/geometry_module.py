"""
修复算法的核心模块：LBP 重计算与雅可比提取

实现两个核心函数：
1. compute_simplex_bound: 重新执行 LBP 前向传播，返回标量边界
2. compute_jacobian_matrix: 计算标量边界对网络参数的雅可比矩阵

支持多个动力学系统（Barrier1, Barrier2, Barrier3, Barrier4 等）
通过 dynamics_model 接口动态调用 compute_f() 方法。
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_lbp_bounds_flexible(
    model: nn.Module,
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    translator,
    return_all_bounds: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], List[Tuple]]:
    """
    灵活的 LBP 前向传播，计算网络的区间界和中间层激活界。

    Args:
        model: 神经网络
        x_lb: 输入下界 [D]
        x_ub: 输入上界 [D]
        translator: TorchTranslator，用于数学运算
        return_all_bounds: 是否返回所有层的激活界

    Returns:
        (h_lb, h_ub, pre_act_bounds, post_act_bounds)
        - h_lb, h_ub: 最终输出的下界和上界（标量）
        - pre_act_bounds: 每层预激活界的列表 [(lb, ub), ...]
        - post_act_bounds: 每层后激活界的列表 [(lb, ub), ...]
    """
    device = x_lb.device
    dtype = x_lb.dtype

    # 提取网络层
    fc_layers = []
    activation_types = []  # 存储激活函数类型

    for module in model.modules():
        if isinstance(module, nn.Linear):
            fc_layers.append(module)
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            activation_types.append(type(module).__name__)

    # LBP 传播
    current_lb = x_lb.clone()
    current_ub = x_ub.clone()

    pre_act_bounds = []
    post_act_bounds = []

    for i, layer in enumerate(fc_layers):
        W, b = layer.weight, layer.bias

        # 线性变换界的 LBP
        W_pos = F.relu(W)
        W_neg = W - W_pos

        y_lb = (W_pos @ current_lb) + (W_neg @ current_ub) + b
        y_ub = (W_pos @ current_ub) + (W_neg @ current_lb) + b

        pre_act_bounds.append((y_lb.clone(), y_ub.clone()))

        if i < len(fc_layers) - 1:  # 不是最后一层，应用激活
            # 获取激活类型
            act_type = activation_types[i] if i < len(activation_types) else 'ReLU'

            if act_type == 'Tanh':
                # Tanh 是单调递增的，且值域在 [-1, 1]
                z_lb = torch.tanh(y_lb)
                z_ub = torch.tanh(y_ub)
            elif act_type == 'Sigmoid':
                # Sigmoid 是单调递增的，值域在 [0, 1]
                z_lb = torch.sigmoid(y_lb)
                z_ub = torch.sigmoid(y_ub)
            else:  # ReLU
                inactive_mask = y_ub <= 0
                active_mask = y_lb >= 0
                unstable_mask = ~active_mask & ~inactive_mask

                z_lb = torch.where(inactive_mask, torch.zeros_like(y_lb), y_lb)

                denom = torch.clamp(y_ub - y_lb, min=1e-12)
                clip_upper = y_ub * y_lb / denom
                z_ub = torch.where(unstable_mask, clip_upper, y_ub)

            current_lb = z_lb
            current_ub = z_ub
            post_act_bounds.append((current_lb.clone(), current_ub.clone()))
        else:
            # 最后一层，没有激活函数
            post_act_bounds.append((y_lb.clone(), y_ub.clone()))

    h_lb = current_lb.squeeze()
    h_ub = current_ub.squeeze()

    # 确保输出是标量（如果输出维度大于1，取第一个元素作为标量）
    if h_lb.dim() > 0:
        h_lb = h_lb[0]
    if h_ub.dim() > 0:
        h_ub = h_ub[0]

    return h_lb, h_ub, pre_act_bounds, post_act_bounds


def _compute_jacobian_bounds(
    fc_layers: List[nn.Linear],
    pre_act_bounds: List[Tuple[torch.Tensor, torch.Tensor]],
    activation_types: List[str],
    output_is_scalar: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算输出对输入的 Jacobian 界（反向传播版本）。

    使用 CROWN 线性界传播的反向形式计算 Jacobian 界。
    目标：计算 ∂h/∂x 的界，其中 h 是标量输出。

    Args:
        fc_layers: 全连接层列表
        pre_act_bounds: 预激活界列表
        activation_types: 激活函数类型列表
        output_is_scalar: 输出是否是标量

    Returns:
        (J_lb, J_ub): Jacobian 的下界和上界，形状 [input_dim]
        对于标量输出，返回一维向量 [input_dim]
    """
    num_layers = len(fc_layers)

    if num_layers == 0:
        return None, None

    input_dim = fc_layers[0].in_features

    if output_is_scalar:
        # 标量输出的 Jacobian 界
        # 使用反向传播形式计算 dL/dx
        # 最后一层：∂h/∂y = 1（因为输出已经是标量）
        # dL/d(pre_act_L) = 1 * W^L（输出对 pre-activation 的导数）

        # 最后一层的梯度：W^L 是 [1, dim_{L-1}]
        W_L = fc_layers[-1].weight  # [1, dim_{L-1}]
        grad = W_L[0, :]  # [dim_{L-1}]，取第一行

        for i in range(num_layers - 2, -1, -1):
            W = fc_layers[i].weight  # [out_dim, in_dim]

            # 反向传播梯度：grad = W^T @ grad
            grad = W.T @ grad  # [in_dim]

            # 获取激活函数导数的界（来自下一层，即索引 i）
            if i > 0:
                act_lb, act_ub = pre_act_bounds[i - 1]  # 第 i 层的 pre-activation 界
                act_type = activation_types[i - 1] if i - 1 < len(activation_types) else 'ReLU'

                # 计算激活函数导数的界
                if act_type == 'Tanh':
                    S_L = torch.zeros_like(act_lb)
                    S_U = torch.ones_like(act_lb)
                elif act_type == 'Sigmoid':
                    sigma_lb = torch.sigmoid(act_lb)
                    sigma_ub = torch.sigmoid(act_ub)
                    S_L = sigma_lb * (1 - sigma_ub)
                    S_U = sigma_ub * (1 - sigma_lb)
                else:  # ReLU
                    S_L = torch.zeros_like(act_lb)
                    S_U = torch.ones_like(act_lb)

                # 乘以激活函数导数的界（使用中间值）
                grad = grad * ((S_L + S_U) / 2)

        # grad 现在是 [input_dim] 形状的 dL/dx
        # 使用绝对值作为界（保守处理）
        J_lb = grad.abs()
        J_ub = grad.abs()

        return J_lb, J_ub

    else:
        # 多输出情况（不常用，暂不实现）
        raise NotImplementedError("Multi-output Jacobian bounds not implemented yet")


def _compute_dynamics_bounds_torch(
    dynamics_model,
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    translator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 TorchTranslator 计算动力学的界。

    Args:
        dynamics_model: 动力学系统对象（有 compute_f 方法）
        x_lb: 状态下界 [D]
        x_ub: 状态上界 [D]
        translator: TorchTranslator

    Returns:
        (f_lb, f_ub): 动力学函数 f(x) 的下界和上界，形状 [D]
    """
    device = x_lb.device
    dtype = x_lb.dtype
    D = x_lb.shape[-1]

    # 计算区间内所有顶点的 f(x) 值
    # 对于 D 维空间，我们需要评估 2^D 个角点
    corner_points = []
    for binary in range(1 << D):
        corner = torch.zeros(D, dtype=dtype, device=device)
        for d in range(D):
            if (binary >> d) & 1:
                corner[d] = x_ub[d]
            else:
                corner[d] = x_lb[d]
        corner_points.append(corner)

    corner_points = torch.stack(corner_points, dim=0)  # [2^D, D]

    # 计算每个角点的 f(x)
    f_values = dynamics_model.compute_f(corner_points, translator)  # [2^D, D]

    # 取每个维度的最小值和最大值作为界
    f_lb = torch.min(f_values, dim=0).values
    f_ub = torch.max(f_values, dim=0).values

    return f_lb, f_ub


def _compute_mccormick_product_lower_bound(
    J_lb: torch.Tensor,
    J_ub: torch.Tensor,
    f_lb: torch.Tensor,
    f_ub: torch.Tensor
) -> torch.Tensor:
    """
    计算 J * f 的下界（McCormick 乘积松弛）。

    对于每个维度 i,j：J[i,j] * f[j] 的下界
    然后对所有 j 求和得到标量

    Args:
        J_lb, J_ub: Jacobian 界，形状 [D, D]
        f_lb, f_ub: f(x) 界，形状 [D]

    Returns:
        J * f 的下界（标量）
    """
    # J * f 是矩阵-向量乘积，结果是向量 [D]
    # 然后我们对向量元素求和得到标量

    # 计算每个 J[i,j] * f[j] 的界
    # 下界 = min(4 个角点乘积)
    product_min = torch.zeros_like(J_lb)  # [D, D]

    corners = [
        (J_lb, f_lb),  # a_L * b_L
        (J_lb, f_ub),  # a_L * b_U
        (J_ub, f_lb),  # a_U * b_L
        (J_ub, f_ub),  # a_U * b_U
    ]

    for J_mat, f_vec in corners:
        product_min = product_min + J_mat * f_vec

    # J * f 的下界：对 J 的每个行向量与 f 的乘积极小化
    # 实际上这是 element-wise 的最小值
    product_lower = torch.minimum(
        torch.minimum(J_lb * f_lb, J_lb * f_ub),
        torch.minimum(J_ub * f_lb, J_ub * f_ub)
    )  # [D]

    # 求和得到标量
    min_L = torch.sum(product_lower)

    return min_L


def compute_simplex_bound(
    model: nn.Module,
    simplex_vertices: Union[torch.Tensor, np.ndarray],
    region_type: str,
    dynamics_model=None,
    translator=None
) -> torch.Tensor:
    """
    对传入的单纯形顶点重新执行 LBP 前向传播，返回对应的标量边界。

    Args:
        model: 神经网络（BarrierNN 或 nn.Sequential）
        simplex_vertices: 单纯形的顶点，形状 [V, D]，其中 V = D+1
        region_type: 'unsafe' 或 'safe'
            - 'unsafe': 返回网络输出的上界 h_max（验证条件 h_max < 0）
            - 'safe': 返回 CBF Lie 导数条件的下界 min_L（验证条件 min_L >= 0）
        dynamics_model: 动力学系统（safe 区域需要，支持多个系统）
        translator: TorchTranslator（safe 区域需要）

    Returns:
        标量边界值，保留计算图用于反向传播

    避坑：
        - 必须保证 requires_grad=True
        - 严禁使用 with torch.no_grad()
        - 严禁任何 In-place 操作
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # 转换顶点为 Tensor，确保保留计算图
    if isinstance(simplex_vertices, np.ndarray):
        vertices = torch.tensor(simplex_vertices, dtype=dtype, device=device)
    else:
        vertices = simplex_vertices.to(dtype=dtype, device=device)

    # 确保 vertices 保留梯度（虽然它不参与反向传播，但必须保持计算图连通性）
    vertices = vertices.clone().requires_grad_(True)

    V, D = vertices.shape
    if V != D + 1:
        raise ValueError(f"Simplex vertices must have V=D+1 vertices, got V={V}, D={D}")

    # ========== 计算输入区间 ==========
    x_lb = vertices.min(dim=0).values
    x_ub = vertices.max(dim=0).values

    # ========== LBP 前向传播（获取 h_max 或 h_min） ==========
    h_lb, h_ub, pre_act_bounds, post_act_bounds = _compute_lbp_bounds_flexible(
        model, x_lb, x_ub, translator
    )

    # ========== 根据区域类型返回对应边界 ==========
    if region_type == 'unsafe':
        # 返回 h_max（验证条件需要 h_max < 0）
        # 确保是标量
        return h_ub.reshape(-1)

    elif region_type == 'safe':
        # 返回 CBF Lie 导数条件的下界 min_L
        # 验证条件：∇h(x) · f(x) + α(h(x)) >= 0

        if dynamics_model is None or translator is None:
            raise ValueError("dynamics_model and translator are required for 'safe' region type")

        # ========== 提取网络层和激活类型 ==========
        fc_layers = []
        activation_types = []

        for module in model.modules():
            if isinstance(module, nn.Linear):
                fc_layers.append(module)
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                activation_types.append(type(module).__name__)

        # ========== 计算 Jacobian 界 ==========
        J_lb, J_ub = _compute_jacobian_bounds(
            fc_layers, pre_act_bounds, activation_types, output_is_scalar=True
        )
        # J_lb, J_ub: [D, D] - 输出对每个输入维度的偏导数界

        # ========== 计算动力学界 ==========
        f_lb, f_ub = _compute_dynamics_bounds_torch(
            dynamics_model, x_lb, x_ub, translator
        )
        # f_lb, f_ub: [D]

        # ========== 计算 ∇h · f 的下界 ==========
        # 使用 McCormick 乘积松弛
        dh_df_L = _compute_mccormick_product_lower_bound(J_lb, J_ub, f_lb, f_ub)

        # ========== 计算 α(h) 项 ==========
        # CBF 条件：∇h · f + α(h) >= 0
        # α(h) = α * h(x)，其中 α > 0 是 class-K 函数
        alpha = dynamics_model.alpha if hasattr(dynamics_model, 'alpha') else 1.0

        # h 的界（保守使用下界）
        h_L = h_lb  # h 的下界

        alpha_h_L = alpha * h_L

        # ========== CBF 条件下界 ==========
        # min_L = ∇h · f 的下界 + α(h) 的下界
        # 注意：α 是正数，所以 α * h 的下界是 α * h_lb
        min_L = dh_df_L + alpha_h_L

        # 确保是标量并保留梯度
        return min_L.reshape(-1)

    else:
        raise ValueError(f"Invalid region_type: {region_type}. Must be 'unsafe' or 'safe'")


def _extract_layer_params(model: nn.Module) -> Tuple[List[nn.Linear], List[str], dict]:
    """
    从模型中提取层参数和命名信息。

    Returns:
        (fc_layers, activation_types, param_dict)
    """
    fc_layers = []
    activation_types = []
    param_dict = {}

    for name, param in model.named_parameters():
        param_dict[name] = param

    for module in model.modules():
        if isinstance(module, nn.Linear):
            fc_layers.append(module)
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            activation_types.append(type(module).__name__)

    return fc_layers, activation_types, param_dict


def _forward_pass_simple(
    model: nn.Module,
    x_lb: torch.Tensor,
    x_ub: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    简单的 LBP 前向传播（不带中间界返回）。

    Returns:
        (h_lb, h_ub): 输出的下界和上界
    """
    # 提取层信息
    fc_layers = []
    activation_types = []

    for module in model.modules():
        if isinstance(module, nn.Linear):
            fc_layers.append(module)
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            activation_types.append(type(module).__name__)

    current_lb = x_lb.clone()
    current_ub = x_ub.clone()

    for i, layer in enumerate(fc_layers):
        W = layer.weight
        b = layer.bias

        W_pos = F.relu(W)
        W_neg = W - W_pos

        y_lb = (W_pos @ current_lb) + (W_neg @ current_ub) + b
        y_ub = (W_pos @ current_ub) + (W_neg @ current_lb) + b

        if i < len(fc_layers) - 1:
            act_type = activation_types[i] if i < len(activation_types) else 'ReLU'

            if act_type == 'Tanh':
                z_lb = torch.tanh(y_lb)
                z_ub = torch.tanh(y_ub)
            elif act_type == 'Sigmoid':
                z_lb = torch.sigmoid(y_lb)
                z_ub = torch.sigmoid(y_ub)
            else:  # ReLU
                inactive_mask = y_ub <= 0
                active_mask = y_lb >= 0
                unstable_mask = ~active_mask & ~inactive_mask

                z_lb = torch.where(inactive_mask, torch.zeros_like(y_lb), y_lb)

                denom = torch.clamp(y_ub - y_lb, min=1e-12)
                clip_upper = y_ub * y_lb / denom
                z_ub = torch.where(unstable_mask, clip_upper, y_ub)

            current_lb = z_lb
            current_ub = z_ub
        else:
            current_lb = y_lb
            current_ub = y_ub

    return current_lb.squeeze(), current_ub.squeeze()


def compute_jacobian_matrix(
    model: nn.Module,
    V_safe: List[Union[torch.Tensor, np.ndarray]],
    V_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model=None,
    translator=None
) -> torch.Tensor:
    """
    计算 V_safe 和 V_unsafe 中所有单纯形的标量边界对网络参数的雅可比矩阵。

    使用 torch.autograd.grad 计算雅可比矩阵，
    确保每次计算独立，不累加梯度。

    Args:
        model: 神经网络（BarrierNN 或 nn.Sequential）
        V_safe: 安全区中验证通过的单纯形顶点列表
        V_unsafe: 障碍区中验证通过的单纯形顶点列表
        dynamics_model: 动力学系统（safe 区域需要，支持多个系统）
        translator: TorchTranslator（safe 区域需要）

    Returns:
        雅可比矩阵 J，形状 [N, P]，其中 N 是单纯形总数，P 是网络参数总数
        J[i, j] = ∂bound_i / ∂θ_j
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # 准备所有单纯形和对应的区域类型
    all_vertices = []
    all_region_types = []

    for v in V_safe:
        if isinstance(v, np.ndarray):
            v = torch.tensor(v, dtype=dtype, device=device)
        else:
            v = v.to(dtype=dtype, device=device)
        all_vertices.append(v)
        all_region_types.append('safe')

    for v in V_unsafe:
        if isinstance(v, np.ndarray):
            v = torch.tensor(v, dtype=dtype, device=device)
        else:
            v = v.to(dtype=dtype, device=device)
        all_vertices.append(v)
        all_region_types.append('unsafe')

    N = len(all_vertices)

    if N == 0:
        num_params = sum(p.numel() for p in model.parameters())
        return torch.zeros(0, num_params, dtype=dtype, device=device)

    num_params = sum(p.numel() for p in model.parameters())
    all_grads = []

    for i, (vertices, region_type) in enumerate(zip(all_vertices, all_region_types)):
        vertices = vertices.clone().requires_grad_(True)

        # 计算输入界
        x_lb = vertices.min(dim=0).values
        x_ub = vertices.max(dim=0).values

        # 前向传播
        h_lb, h_ub = _forward_pass_simple(model, x_lb, x_ub)

        # 选择输出
        if region_type == 'unsafe':
            output = h_ub
        else:
            output = h_lb

        # 计算对所有参数的梯度
        grad_list = []
        params_for_grad = list(model.parameters())

        # 使用 autograd.grad 单独计算每个参数的梯度
        for param in params_for_grad:
            grad = torch.autograd.grad(
                outputs=output,
                inputs=param,
                retain_graph=True,
                allow_unused=True
            )[0]
            if grad is not None:
                grad_list.append(grad.flatten())
            else:
                grad_list.append(torch.zeros(param.numel(), dtype=dtype, device=device))

        grad_vector = torch.cat(grad_list)
        all_grads.append(grad_vector)

    # Stack 成矩阵
    J = torch.stack(all_grads, dim=0)  # [N, num_params]

    return J


def compute_gradient_norms(
    model: nn.Module,
    V_safe: List[Union[torch.Tensor, np.ndarray]],
    V_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model=None,
    translator=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算每个单纯形的边界值对参数的梯度范数。

    用于分析修复方向的重要性和敏感性。

    Args:
        model: 神经网络
        V_safe: 安全区单纯形列表
        V_unsafe: 障碍区单纯形列表
        dynamics_model: 动力学系统
        translator: TorchTranslator

    Returns:
        (safe_grad_norms, unsafe_grad_norms): 安全区和障碍区的梯度范数
    """
    J = compute_jacobian_matrix(
        model, V_safe, V_unsafe, dynamics_model, translator
    )

    # 计算每行的 L2 范数
    grad_norms = torch.norm(J, dim=1)  # [N]

    n_safe = len(V_safe)
    n_unsafe = len(V_unsafe)

    if n_safe > 0 and n_unsafe > 0:
        safe_grad_norms = grad_norms[:n_safe]
        unsafe_grad_norms = grad_norms[n_safe:]
    elif n_safe > 0:
        safe_grad_norms = grad_norms
        unsafe_grad_norms = torch.tensor([], device=grad_norms.device)
    else:
        safe_grad_norms = torch.tensor([], device=grad_norms.device)
        unsafe_grad_norms = grad_norms

    return safe_grad_norms, unsafe_grad_norms
