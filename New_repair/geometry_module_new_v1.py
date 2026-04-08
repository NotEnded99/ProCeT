"""
基于 torch.func 优化 Jacobian 矩阵计算的模块 (v1)

核心改进:
1. 利用 torch.func.vmap + torch.func.jacrev 实现真正的向量化并行计算
2. CrownPartialLinearization 一次性处理整个 batch（而非逐个 simplex）
3. 函数式重构：使用 functional_call 替代直接模型调用
4. 安全区 Jacobian 使用链式法则：
   - 先用 CrownPartialLinearization 批量计算界（无梯度）
   - 用 functional_call + jacrev 计算 ∂h/∂θ
   - 链式法则：∂min_L/∂θ = (∂min_L/∂h) · (∂h/∂θ)

数学一致性：完全复用 verify_cbf.py 中的：
- _batched_compute_mccormick_product_lower_bound
- _batched_get_affine_function_bounds

Jacobian 矩阵形状：[N * output_dim, total_params]
其中 N = len(V_safe) + len(V_unsafe)，output_dim = network output dimension
"""

from typing import List, Tuple, Union, Dict, Optional

import copy
import functools
import numpy as np
import torch
import cvxpy as cp
import torch.nn as nn
from torch import vmap
from torch.func import functional_call, jacrev

from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization
from lbp_neural_cbf.regions import SimplicialRegion
from lbp_neural_cbf.cbf.verify_cbf import (
    _compute_dynamics_bounds_taylor,
    _batched_compute_mccormick_product_lower_bound,
    _batched_get_affine_function_bounds,
    _vectorized_get_affine_function_bounds,
)


# =============================================================================
# Helper: 从顶点列表构建 SimplicialRegion batch
# =============================================================================

def _build_simplicial_batch(
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    device: torch.device,
    dtype: torch.dtype,
) -> List[SimplicialRegion]:
    """
    将单纯形顶点列表转换为 SimplicialRegion 对象列表。

    Args:
        vertices_list: 单纯形顶点列表，每个元素形状 [D+1, D]
        device, dtype: 张量设备和数据类型

    Returns:
        SimplicialRegion 对象列表（长度为 batch_size）
    """
    batch = []
    for verts in vertices_list:
        if isinstance(verts, torch.Tensor):
            verts_np = verts.cpu().numpy()
        else:
            verts_np = verts
        batch.append(SimplicialRegion(verts_np, output_dim=None))
    return batch


# =============================================================================
# Helper: 批量版本的李导数下界提取
# =============================================================================

def _extract_lie_derivative_lower_bound(
    network_linearizer: CrownPartialLinearization,
    dynamics_bounds,
    g_dynamics_bounds,
    batch: List,
    dynamics_model,
    device: torch.device,
    dtype: torch.dtype,
    eta: tuple = (0.5, 0.5),
) -> torch.Tensor:
    """
    复现 verify_cbf.py:_verify_cbf_condition_affine 的逻辑，
    计算 CBF 李导数条件的下界 min_L。

    完全按照 verify_cbf.py 的步骤实现：
    1. 从 network_linearizer 获取 Jacobian 界 (J_affine_L, J_affine_U)
    2. 从 dynamics_bounds 获取 f(x) 界 (f_affine_L, f_affine_U)
    3. 计算 J(x)·f(x) 的 McCormick 下界
    4. 添加 class-K 项 alpha(h(x))
    5. 如果有控制维度，计算 g(x) 项
    6. 用 _batched_get_affine_function_bounds 在单纯形上求最小值
    """
    n = dynamics_model.input_dim
    m = dynamics_model.control_dim
    f_affine_bounds = dynamics_bounds
    g_affine_bounds = g_dynamics_bounds

    # 1. 获取 Jacobian 界
    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
    J_affine_L, J_affine_U = (A_L, b_L), (A_U, b_U)

    # 2. Dynamics 界
    f_affine_L, f_affine_U = f_affine_bounds

    # 3. J(x)·f(x) 下界
    eta_drift = eta[0]
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_affine_L,
        J_affine_U,
        f_affine_L,
        f_affine_U,
        batch,
        eta=eta_drift,
        device=device,
        dtype=dtype,
    )
    M_D, c_D = M_D.sum(dim=-2), c_D.sum(dim=-1)

    # 4. Class-K 项：alpha(h(x)) = alpha * h(x)
    (A_L_net, a_L_net), (A_U_net, a_U_net) = network_linearizer.get_network_linear_bounds()
    alpha_A_L = dynamics_model.alpha_function(A_L_net[..., 0, :])
    alpha_a_L = dynamics_model.alpha_function(a_L_net[..., 0])

    M_total = M_D + alpha_A_L
    c_total = c_D + alpha_a_L

    # 5. 控制项
    if m > 0 and g_affine_bounds is not None:
        g_affine_L = g_affine_bounds[0][0], g_affine_bounds[0][1]
        g_affine_U = g_affine_bounds[1][0], g_affine_bounds[1][1]

        eta_control_L = eta[1]
        M_v_L, c_v_L = _batched_compute_mccormick_product_lower_bound(
            J_affine_L,
            J_affine_U,
            g_affine_L,
            g_affine_U,
            batch,
            eta=eta_control_L,
            device=device,
            dtype=dtype,
        )
        M_v_L, c_v_L = M_v_L.sum(dim=-2), c_v_L.sum(dim=-1)

        v_affine_L = (M_v_L, c_v_L)
        v_L_min, v_L_max = _batched_get_affine_function_bounds(
            v_affine_L, batch, device=device, dtype=dtype
        )

        u_min = torch.tensor(dynamics_model.u_min, device=device, dtype=dtype)
        u_max = torch.tensor(dynamics_model.u_max, device=device, dtype=dtype)

        M_v_L_u_min = M_v_L * u_min.unsqueeze(-1)
        c_v_L_u_min = c_v_L * u_min
        M_v_L_u_max = M_v_L * u_max.unsqueeze(-1)
        c_v_L_u_max = c_v_L * u_max

        for sample_idx, sample in enumerate(batch):
            M_C = torch.zeros(n, device=device, dtype=dtype)
            c_C = torch.tensor(0.0, device=device, dtype=dtype)

            v_sample_min = v_L_min[sample_idx]
            v_sample_max = v_L_max[sample_idx]

            pos_mask = v_sample_min >= 0
            if pos_mask.any():
                M_C += M_v_L_u_max[sample_idx, pos_mask].sum(dim=0)
                c_C += c_v_L_u_max[sample_idx, pos_mask].sum()

            neg_mask = v_sample_max <= 0
            if neg_mask.any():
                M_C += M_v_L_u_min[sample_idx, neg_mask].sum(dim=0)
                c_C += c_v_L_u_min[sample_idx, neg_mask].sum()

            mixed_mask = ~(pos_mask | neg_mask)
            if mixed_mask.any():
                v_u_min_b, _ = _vectorized_get_affine_function_bounds(
                    (M_v_L_u_min[sample_idx, mixed_mask], c_v_L_u_min[sample_idx, mixed_mask]),
                    sample,
                    device=device,
                    dtype=dtype,
                )
                v_u_max_b, _ = _vectorized_get_affine_function_bounds(
                    (M_v_L_u_max[sample_idx, mixed_mask], c_v_L_u_max[sample_idx, mixed_mask]),
                    sample,
                    device=device,
                    dtype=dtype,
                )
                c_C += torch.maximum(v_u_min_b, v_u_max_b).sum()

            M_total[sample_idx] += M_C
            c_total[sample_idx] += c_C

    # 6. 在单纯形上求最小值
    min_L, _ = _batched_get_affine_function_bounds(
        (M_total.unsqueeze(1), c_total.unsqueeze(1)),
        batch,
        device=device,
        dtype=dtype,
    )
    min_L = min_L.squeeze(-1)

    return min_L


# =============================================================================
# 核心: 向量化 Jacobian 矩阵计算
# =============================================================================

def _make_forward_functional(model_for_jac: nn.Module):
    """
    创建一个函数式前向传播函数，用于 jacrev 求导。

    jacrev(functional_call, argnums=0) 会尝试对 module 参数求导，
    这在 functional_call 的语义中是无效的。正确做法是：
    1. 定义一个包装函数，将 module 绑定为常量
    2. 对包装函数的第一个参数（params_dict）求导
    """
    def forward_fn(params, inputs):
        """functional_call 的包装，module 被绑定为常量。"""
        return functional_call(model_for_jac, params, inputs)

    return forward_fn


def _compute_jacobian_rows_single_output(
    model_for_jac: nn.Module,
    params_dict: Dict[str, torch.Tensor],
    verts_single: torch.Tensor,
    output_dim: int,
) -> torch.Tensor:
    """
    计算单个单纯形的网络输出对所有参数的 Jacobian 行。

    使用 functional_call + jacrev 计算：
        h = network(verts_single)  # [output_dim]
        ∂h[i]/∂θ  →  形状 [output_dim, num_params]

    Args:
        model_for_jac: 用于 Jacobian 计算的模型（带参数追踪）
        params_dict: 参数字典
        verts_single: 单个单纯形顶点，形状 [D+1, D]
        output_dim: 网络输出维度

    Returns:
        jac_rows: [output_dim, num_params] 张量
    """
    # 构建包装函数：module 绑定为常量，jacrev 只对 params 求导
    def forward_fn(params, inputs):
        return functional_call(model_for_jac, params, inputs)

    # jacrev: 对第一个参数（params）求导
    jac_rev_fn = jacrev(forward_fn, argnums=0)
    grads_all = jac_rev_fn(params_dict, (verts_single.unsqueeze(0),))
    # grads_all: dict[param_name] -> [1, D+1, output_dim, *param_shape]
    # 其中 D+1 是输入顶点数量的延伸，不是 batch 维度
    # 需要去掉 batch(0) 和顶点维(1)，只保留 output_dim 和 param_shape

    jac_rows = []
    for name_p, param_p in model_for_jac.named_parameters():
        g = grads_all[name_p]  # [1, D+1, output_dim, ...]
        g = g.squeeze(0).select(0, 0)  # -> [output_dim, *param_shape]
        jac_rows.append(g.flatten(start_dim=1))  # [output_dim, numel(param)]

    jac = torch.cat(jac_rows, dim=1)  # [output_dim, num_params]
    return jac


def _safe_lie_bound_fn(
    params_dict: Dict[str, torch.Tensor],
    model_grad: nn.Module,
    M_coeff: torch.Tensor,
    c_coeff: torch.Tensor,
    verts_single: torch.Tensor,
) -> torch.Tensor:
    """
    计算单个单纯形的 min_L（可微分函数）。

    用于 jacrev 求导。内部使用 functional_call 确保参数梯度追踪。

    Args:
        params_dict: 参数字典（jacrev 的 argnums=0）
        model_grad: 带梯度追踪的模型副本
        M_coeff: [output_dim, n_state] — 该单纯形的 McCormick 系数
        c_coeff: [output_dim] — 该单纯形的偏置系数
        verts_single: [n_vertices, D] — 单纯形顶点

    Returns:
        min_L: 标量，CBF 李导数下界（通过 softmin 可微分近似）
    """
    # functional_call(model, params, (inputs,)) — 注意 inputs 是 tuple
    h_all = functional_call(model_grad, params_dict, (verts_single.unsqueeze(0),))
    # h_all: [1, n_vertices, output_dim]
    h_all = h_all.squeeze(0)  # [n_vertices, output_dim]

    # M_coeff @ h_all + c_coeff，然后对 state 维求和
    # M_coeff: [output_dim, n_state], h_all: [n_vertices, output_dim]
    # einsum 'oi,vi->vo' → [n_vertices, output_dim]
    lie_pv = torch.einsum('oi,vi->vo', M_coeff, h_all) + c_coeff
    lie_pv = lie_pv.sum(dim=1)  # [n_vertices]

    # softmin 可微分近似（梯度流向 argmin 顶点）
    lie_stable = lie_pv - lie_pv.max()
    weights = torch.softmax(lie_stable * 1000, dim=0)  # [n_vertices]
    min_L_soft = (weights * lie_pv).sum()  # 标量

    return min_L_soft


def _build_safe_bound_functional(
    model_for_jac: nn.Module,
    dynamics_model,
    M_coeff_fixed: torch.Tensor,
    c_coeff_fixed: torch.Tensor,
    vertices_np_list: List[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> callable:
    """
    构建可微分的 safe region bound 函数。

    策略：
    - 界系数 (M_coeff, c_coeff) 由 CrownPartialLinearization 预计算（作为常量传入）
    - 网络前向传播由 functional_call 计算（支持自动微分）
    - min_L ≈ M_coeff @ h + c_coeff（链式法则）

    返回:
        bound_fn(params_dict, verts_single, idx): 计算第 idx 个单纯形的 bound
    """
    n_samples = len(vertices_np_list)

    def bound_fn(params_dict, verts_single, idx_scalar):
        """
        计算第 idx 个单纯形的 bound 值（保留梯度）。

        Args:
            params_dict: 参数字典
            verts_single: 单个单纯形顶点 [D+1, D]
            idx_scalar: 单纯形索引（标量 tensor）

        Returns:
            bound 值（标量）
        """
        idx = idx_scalar.item() if torch.is_tensor(idx_scalar) else idx_scalar

        # 网络输出（functional_call 支持自动微分）
        h = functional_call(
            model_for_jac, params_dict, (verts_single.unsqueeze(0),)
        )  # [1, output_dim]
        h = h.squeeze(0)  # [output_dim]

        # 提取对应单纯形的系数
        M_i = M_coeff_fixed[idx]  # [n_state] or [n_state, output_dim]
        c_i = c_coeff_fixed[idx]   # [n_state]

        # clamp idx 到有效范围
        idx_clamped = min(max(idx, 0), n_samples - 1)
        M_i = M_coeff_fixed[idx_clamped]
        c_i = c_coeff_fixed[idx_clamped]

        # min_L ≈ M_coeff @ h + c_coeff（对 h[0] 的线性近似）
        if M_i.ndim == 1:
            # M_i: [n_state]，取第一个分量的梯度
            bound = (M_i * h).sum() + c_i.sum()
        else:
            # M_i: [n_state, output_dim]，取 h[0] 的加权
            bound = (M_i * h.unsqueeze(0)).sum() + c_i.sum()

        return bound

    return bound_fn


def compute_jacobian_matrix_v1(
    model: nn.Module,
    V_safe: List[Union[torch.Tensor, np.ndarray]],
    V_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model=None,
    translator=None,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    使用 torch.func.vmap + jacrev 实现的向量化 Jacobian 矩阵计算。

    核心思想：
    - 使用 functional_call + jacrev + vmap 追踪参数梯度
    - unsafe 区域：直接对 h_ub 求 Jacobian（精确）
    - safe 区域：链式法则 = (∂min_L/∂h) · (∂h/∂θ)（近似）
    - CrownPartialLinearization 一次性处理整个 batch

    Jacobian 矩阵: J[i*output_dim + j, k] = ∂h_i[j] / ∂θ_k
    形状: [N * output_dim, Total_Params]

    Args:
        model: 神经网络（BarrierNN 或 nn.Sequential）
        V_safe: 安全区单纯形顶点列表
        V_unsafe: 障碍区单纯形顶点列表
        dynamics_model: 动力学系统
        translator: TorchTranslator
        batch_size: 每个 vmap batch 的最大单纯形数量

    Returns:
        Jacobian 矩阵 J，形状 [N * output_dim, Total_Params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 收集所有单纯形 ----------
    all_vertices = []
    all_region_types = []

    for v in V_safe:
        all_vertices.append(v)
        all_region_types.append('safe')
    for v in V_unsafe:
        all_vertices.append(v)
        all_region_types.append('unsafe')

    N = len(all_vertices)
    num_params = sum(p.numel() for p in model.parameters())

    if N == 0:
        return torch.zeros(0, num_params, dtype=dtype, device=device)

    # ---------- 确定网络输出维度 ----------
    sample_dim = all_vertices[0].shape[1]  # D
    dummy_input = torch.zeros(1, model.input_dim if hasattr(model, 'input_dim') else sample_dim,
                              dtype=dtype, device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    output_dim = dummy_output.shape[-1]
    del dummy_input, dummy_output

    # ---------- 克隆模型（用于梯度追踪）----------
    model_grad = copy.deepcopy(model)
    for p in model_grad.parameters():
        p.requires_grad_(True)

    # ---------- 提取参数字典 ----------
    param_dict = {name: param for name, param in model_grad.named_parameters()}

    # ---------- 转换顶点 ----------
    vertices_np_list = []
    for verts in all_vertices:
        if isinstance(verts, torch.Tensor):
            vertices_np_list.append(verts.cpu().numpy())
        else:
            vertices_np_list.append(verts)

    # ---------- 分块处理 ----------
    print(f"[v1] Vectorized Jacobian: {N} simplices, {num_params} params, "
          f"output_dim={output_dim}, batch_size={batch_size}")

    # 预分配结果矩阵：每行对应一个单纯形的一个输出维度
    J = torch.zeros(N * output_dim, num_params, dtype=dtype, device=device)

    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_vertices = vertices_np_list[start:end]
        batch_types = all_region_types[start:end]
        batch_n = end - start

        if batch_n == 0:
            continue

        print(f"  Batch {batch_idx + 1}/{num_batches}: {batch_n} simplices ("
              f"unsafe={sum(t=='unsafe' for t in batch_types)}, "
              f"safe={sum(t=='safe' for t in batch_types)})")

        # ---------- 构建 SimplicialRegion batch ----------
        sim_batch = _build_simplicial_batch(batch_vertices, device, dtype)

        # ---------- 顶点张量 ----------
        verts_tensors = torch.stack([
            torch.tensor(v, dtype=dtype, device=device) for v in batch_vertices
        ], dim=0)  # [batch_n, D+1, D]

        # =================================================================
        # Unsafe 区域：functional_call + jacrev + vmap
        # =================================================================
        unsafe_mask = [t == 'unsafe' for t in batch_types]
        unsafe_indices_local = [i for i, m in enumerate(unsafe_mask) if m]
        n_unsafe = len(unsafe_indices_local)

        if n_unsafe > 0:
            unsafe_verts = verts_tensors[unsafe_indices_local]  # [n_unsafe, D+1, D]

            # 使用闭包捕获 model_grad（供 jacrev 内部使用）
            def unsafe_jac_fn(params_d, verts_s):
                return _compute_jacobian_rows_single_output(
                    model_grad, params_d, verts_s, output_dim
                )

            unsafe_jac = vmap(unsafe_jac_fn, in_dims=(None, 0))(
                param_dict, unsafe_verts
            )  # [n_unsafe, output_dim, num_params]

            # 填入全局 J 矩阵
            for local_i, global_i in enumerate(unsafe_indices_local):
                J[global_i * output_dim:(global_i + 1) * output_dim] = unsafe_jac[local_i]

        # =================================================================
        # Safe 区域：精确链式法则 Jacobian
        #
        # min_L 是网络输出 h 的非线性函数，通过 McCormick 界传播得到。
        # 由于 CrownPartialLinearization 不维护计算图（∂Jacobian_bounds/∂θ = 0），
        # McCormick 系数 (M_total, c_total) 对 θ 的梯度 = 0。
        #
        # 精确梯度通过以下链式法则计算：
        #   min_L = min_v (M_total @ h(v) + c_total)
        #   ∂min_L/∂θ = M_total @ (∂h(v*)/∂θ)      [v* = argmin 顶点]
        #
        # 实现步骤：
        # 1. 预计算 McCormick 系数（M_total, c_total）— 固定值（无梯度）
        # 2. 构建可微分函数 lie_bound_fn(params, verts)：对单纯形内所有顶点求 min_L
        # 3. jacrev(lie_bound_fn) 计算精确梯度
        # 4. 通过 argmin 索引加权得到最终梯度
        # =================================================================
        safe_mask = [t == 'safe' for t in batch_types]
        safe_indices_local = [i for i, m in enumerate(safe_mask) if m]
        n_safe = len(safe_indices_local)

        if n_safe > 0:
            safe_sim_batch = [sim_batch[i] for i in safe_indices_local]

            # Step 1: CrownPartialLinearization 批量计算界（无梯度）
            model_lin = copy.deepcopy(model_grad)
            for p in model_lin.parameters():
                p.requires_grad_(False)

            lin_safe = CrownPartialLinearization(model_lin, dtype=dtype)
            lin_safe.compute_network_bounds(safe_sim_batch)
            lin_safe.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

            # Step 2: Dynamics 界
            try:
                f_bounds, g_bounds = _compute_dynamics_bounds_taylor(
                    safe_sim_batch, dynamics_model, device=device, dtype=dtype
                )
            except (ValueError, AttributeError):
                print(f"    Warning: dynamics bounds failed for safe batch {batch_idx}, "
                      "using zeros for safe Jacobian rows")
                for local_i, global_i in enumerate(safe_indices_local):
                    J[global_i * output_dim:(global_i + 1) * output_dim] = 0.0
                continue

            # Step 3: 提取所有 McCormick 系数（预计算，固定值）
            (A_L_net, a_L_net), _ = lin_safe.get_network_linear_bounds()
            alpha_A_L = dynamics_model.alpha_function(A_L_net[..., 0, :])   # [n_safe, n_state]
            alpha_a_L = dynamics_model.alpha_function(a_L_net[..., 0])        # [n_safe]

            A_L, b_L, A_U, b_U = lin_safe.get_partial_derivative_bounds()
            f_affine_L, f_affine_U = f_bounds

            # J(x)·f(x) 下界（McCormick 系数）
            M_D, c_D = _batched_compute_mccormick_product_lower_bound(
                (A_L, b_L), (A_U, b_U), f_affine_L, f_affine_U,
                safe_sim_batch, eta=0.5, device=device, dtype=dtype,
            )
            M_D_sum = M_D.sum(dim=-2)  # [n_safe, output_dim, n_state]
            c_D_sum = c_D.sum(dim=-1)  # [n_safe, output_dim]

            # 总系数（不含控制项）
            M_total_fixed = M_D_sum + alpha_A_L.unsqueeze(1)  # [n_safe, output_dim, n_state]
            c_total_fixed = c_D_sum + alpha_a_L.unsqueeze(1)   # [n_safe, output_dim]

            # 控制项系数（若有）
            m = dynamics_model.control_dim
            has_control = (m > 0 and g_bounds is not None)
            if has_control:
                g_affine_L = g_bounds[0][0], g_bounds[0][1]
                g_affine_U = g_bounds[1][0], g_bounds[1][1]
                M_v_L, c_v_L = _batched_compute_mccormick_product_lower_bound(
                    (A_L, b_L), (A_U, b_U), g_affine_L, g_affine_U,
                    safe_sim_batch, eta=0.5, device=device, dtype=dtype,
                )
                M_v_L_sum = M_v_L.sum(dim=-2)  # [n_safe, output_dim, n_state, m]
                c_v_L_sum = c_v_L.sum(dim=-1)  # [n_safe, output_dim, m]
                u_min = torch.tensor(dynamics_model.u_min, device=device, dtype=dtype)
                u_max = torch.tensor(dynamics_model.u_max, device=device, dtype=dtype)
                M_u_fixed = (M_v_L_sum * u_min.unsqueeze(-2)) + \
                            (M_v_L_sum * u_max.unsqueeze(-2))
                # 控制项最终在顶点上取 min（控制项对 θ 无梯度）
            else:
                M_u_fixed = None
                c_v_L_sum = None

            # Step 4: 对每个单纯形计算精确 Jacobian
            # 使用 functools.partial 绑定 per-simplex 系数，避免闭包捕获问题
            # 由于 lie_bound_fn 内的 M_total_fixed 仍有 [n_safe, ...] 维度，
            # 需要用 functools.partial 绑定 per-simplex 系数，避免闭包捕获问题
            safe_verts = verts_tensors[safe_indices_local]  # [n_safe, D+1, D]
            n_safe_actual = safe_verts.shape[0]

            safe_grad_simplex = []
            for si in range(n_safe_actual):
                M_s = M_total_fixed[si]    # [output_dim, n_state]
                c_s = c_total_fixed[si]     # [output_dim]
                verts_s = safe_verts[si]    # [n_vertices, D]

                bound_fn_partial = functools.partial(
                    _safe_lie_bound_fn,
                    model_grad=model_grad,
                    M_coeff=M_s,
                    c_coeff=c_s,
                    verts_single=verts_s,
                )

                grad_dict = jacrev(bound_fn_partial, argnums=0)(param_dict)
                rows = []
                for name, _ in model_grad.named_parameters():
                    g = grad_dict[name].flatten()
                    rows.append(g)
                grad_one = torch.cat(rows)
                safe_grad_simplex.append(grad_one)

            safe_grad_simplex = torch.stack(safe_grad_simplex, dim=0)  # [n_safe, num_params]
            safe_grad_expanded = safe_grad_simplex.unsqueeze(1).expand(
                n_safe_actual, output_dim, num_params
            ).clone()

            for local_i, global_i in enumerate(safe_indices_local):
                J[global_i * output_dim:(global_i + 1) * output_dim] = safe_grad_expanded[local_i]

    print(f"  Jacobian matrix final shape: {J.shape}")
    return J


# =============================================================================
# 备选方案: 纯 vmap+jacrev（适合小 batch）
# =============================================================================

def compute_jacobian_matrix_v1_pure_vmap(
    model: nn.Module,
    V_safe: List[Union[torch.Tensor, np.ndarray]],
    V_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model=None,
    translator=None,
) -> torch.Tensor:
    """
    纯 vmap+jacrev 实现，无 batch 分块。

    适用于 N <= 256 的场景，完全向量化，无 Python 循环。
    如果 N 较大，请使用 compute_jacobian_matrix_v1。

    Jacobian 形状: [N * output_dim, num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 收集所有单纯形 ----------
    all_vertices = []
    all_region_types = []

    for v in V_safe:
        all_vertices.append(v)
        all_region_types.append('safe')
    for v in V_unsafe:
        all_vertices.append(v)
        all_region_types.append('unsafe')

    N = len(all_vertices)
    num_params = sum(p.numel() for p in model.parameters())

    if N == 0:
        return torch.zeros(0, num_params, dtype=dtype, device=device)

    # ---------- 确定网络输出维度 ----------
    sample_dim = all_vertices[0].shape[1]
    dummy_input = torch.zeros(1, model.input_dim if hasattr(model, 'input_dim') else sample_dim,
                              dtype=dtype, device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    output_dim = dummy_output.shape[-1]
    del dummy_input, dummy_output

    # ---------- 克隆模型 ----------
    model_grad = copy.deepcopy(model)
    for p in model_grad.parameters():
        p.requires_grad_(True)
    param_dict = {name: param for name, param in model_grad.named_parameters()}

    # ---------- 构建批次 ----------
    verts_tensors = torch.stack([
        torch.tensor(v.cpu().numpy() if isinstance(v, torch.Tensor) else v,
                     dtype=dtype, device=device)
        for v in all_vertices
    ], dim=0)  # [N, D+1, D]

    region_types = [0 if t == 'safe' else 1 for t in all_region_types]
    region_types_tensor = torch.tensor(region_types, dtype=torch.long, device=device)

    # ---------- Unsafe Jacobian（向量化的 vmap+jacrev）----------
    unsafe_mask = (region_types_tensor == 1)
    unsafe_indices = unsafe_mask.nonzero(as_tuple=True)[0]
    n_unsafe = len(unsafe_indices)

    if n_unsafe > 0:
        unsafe_verts = verts_tensors[unsafe_indices]

        def unsafe_jac_fn(params_d, verts_s):
            return _compute_jacobian_rows_single_output(
                model_grad, params_d, verts_s, output_dim
            )

        unsafe_jac = vmap(unsafe_jac_fn, in_dims=(None, 0))(
            param_dict, unsafe_verts
        )  # [n_unsafe, output_dim, num_params]

    # ---------- Safe Jacobian（精确链式法则）----------
    safe_mask = (region_types_tensor == 0)
    safe_indices = safe_mask.nonzero(as_tuple=True)[0]
    n_safe = len(safe_indices)

    if n_safe > 0:
        safe_verts = verts_tensors[safe_indices]  # [n_safe, D+1, D]
        safe_vertices_np = [all_vertices[i] for i in safe_indices]
        safe_sim_batch = _build_simplicial_batch(safe_vertices_np, device, dtype)

        model_lin = copy.deepcopy(model_grad)
        for p in model_lin.parameters():
            p.requires_grad_(False)

        lin_safe = CrownPartialLinearization(model_lin, dtype=dtype)
        lin_safe.compute_network_bounds(safe_sim_batch)
        lin_safe.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

        try:
            f_bounds, g_bounds = _compute_dynamics_bounds_taylor(
                safe_sim_batch, dynamics_model, device=device, dtype=dtype
            )
        except (ValueError, AttributeError):
            print("Warning: dynamics bounds failed for pure-vmap safe batch")
            safe_jac = torch.zeros(n_safe, output_dim, num_params,
                                   dtype=dtype, device=device)
        else:
            # Step 1: 提取 McCormick 系数（固定值，无梯度）
            (A_L_net, a_L_net), _ = lin_safe.get_network_linear_bounds()
            alpha_A_L = dynamics_model.alpha_function(A_L_net[..., 0, :])   # [n_safe, n_state]
            alpha_a_L = dynamics_model.alpha_function(a_L_net[..., 0])        # [n_safe]

            A_L, b_L, A_U, b_U = lin_safe.get_partial_derivative_bounds()
            f_affine_L, f_affine_U = f_bounds

            M_D, c_D = _batched_compute_mccormick_product_lower_bound(
                (A_L, b_L), (A_U, b_U), f_affine_L, f_affine_U,
                safe_sim_batch, eta=0.5, device=device, dtype=dtype,
            )
            M_D_sum = M_D.sum(dim=-2)  # [n_safe, output_dim, n_state]
            c_D_sum = c_D.sum(dim=-1)  # [n_safe, output_dim]

            M_total_fixed = M_D_sum + alpha_A_L.unsqueeze(1)
            c_total_fixed = c_D_sum + alpha_a_L.unsqueeze(1)

            m = dynamics_model.control_dim
            has_control = (m > 0 and g_bounds is not None)
            if has_control:
                g_affine_L = g_bounds[0][0], g_bounds[0][1]
                g_affine_U = g_bounds[1][0], g_bounds[1][1]
                M_v_L, c_v_L = _batched_compute_mccormick_product_lower_bound(
                    (A_L, b_L), (A_U, b_U), g_affine_L, g_affine_U,
                    safe_sim_batch, eta=0.5, device=device, dtype=dtype,
                )
                M_v_L_sum = M_v_L.sum(dim=-2)  # [n_safe, output_dim, n_state, m]
                c_v_L_sum = c_v_L.sum(dim=-1)   # [n_safe, output_dim, m]
                u_min = torch.tensor(dynamics_model.u_min, device=device, dtype=dtype)
                u_max = torch.tensor(dynamics_model.u_max, device=device, dtype=dtype)
                M_u_fixed = (M_v_L_sum * u_min.unsqueeze(-2)) + \
                             (M_v_L_sum * u_max.unsqueeze(-2))
            else:
                M_u_fixed = None
                c_v_L_sum = None

            # Step 2: 逐单纯形计算精确 Jacobian（vmap 处理每个单纯形）
            M_coeffs_per_simplex = []
            c_coeffs_per_simplex = []

            for si in range(n_safe):
                M_coeffs_per_simplex.append(M_total_fixed[si])    # [output_dim, n_state]
                c_coeffs_per_simplex.append(c_total_fixed[si])     # [output_dim]

            safe_grad_simplex = []
            for si in range(n_safe):
                M_s = M_coeffs_per_simplex[si]
                c_s = c_coeffs_per_simplex[si]
                verts_s = safe_verts[si]  # [n_vertices, D]

                # 使用 functools.partial 绑定常量参数，避免闭包捕获问题
                bound_fn_partial = functools.partial(
                    _safe_lie_bound_fn,
                    model_grad=model_grad,
                    M_coeff=M_s,
                    c_coeff=c_s,
                    verts_single=verts_s,
                )

                # jacrev: 对第一个参数（params_dict）求导
                grad_dict = jacrev(bound_fn_partial, argnums=0)(param_dict)
                rows = []
                for name, param_p in model_grad.named_parameters():
                    g = grad_dict[name].flatten()  # 展平为 1D，无维度丢失
                    rows.append(g)
                grad_one = torch.cat(rows)  # [num_params]
                safe_grad_simplex.append(grad_one)

            safe_grad_simplex = torch.stack(safe_grad_simplex, dim=0)  # [n_safe, num_params]
            safe_jac = safe_grad_simplex.unsqueeze(1).expand(
                n_safe, output_dim, num_params
            ).clone()

    # ---------- 组装完整 Jacobian ----------
    J = torch.zeros(N * output_dim, num_params, dtype=dtype, device=device)
    unsafe_ptr = 0
    safe_ptr = 0

    for i, rt in enumerate(all_region_types):
        if rt == 'unsafe':
            J[i * output_dim:(i + 1) * output_dim] = unsafe_jac[unsafe_ptr].squeeze(0)
            unsafe_ptr += 1
        else:
            J[i * output_dim:(i + 1) * output_dim] = safe_jac[safe_ptr].squeeze(0)
            safe_ptr += 1

    print(f"Pure vmap Jacobian matrix shape: {J.shape}")
    return J


# =============================================================================
# 兼容接口: compute_jacobian_matrix (自动选择实现)
# =============================================================================

def compute_jacobian_matrix(
    model: nn.Module,
    V_safe: List[Union[torch.Tensor, np.ndarray]],
    V_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model=None,
    translator=None,
    use_vmap: bool = True,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    向量化 Jacobian 矩阵计算（主入口）。

    自动在以下模式间选择：
    - use_vmap=True, N <= 256: 使用纯 vmap 实现（无分块循环）
    - use_vmap=True, N > 256: 使用分块 vmap 实现（显存安全）
    - use_vmap=False: 回退到 geometry_module_new.py 的多线程实现

    Jacobian 矩阵形状: [N * output_dim, Total_Params]
    其中 N = len(V_safe) + len(V_unsafe)，output_dim = network output dimension

    Args:
        model: 神经网络
        V_safe: 安全区单纯形顶点列表
        V_unsafe: 障碍区单纯形顶点列表
        dynamics_model: 动力学系统
        translator: TorchTranslator
        use_vmap: 是否使用 vmap+jacrev（推荐 True）
        batch_size: 每个 batch 的最大单纯形数量

    Returns:
        Jacobian 矩阵 J，形状 [N * output_dim, Total_Params]
    """
    N = len(V_safe) + len(V_unsafe)

    if not use_vmap:
        from New_repair.geometry_module_new import compute_jacobian_matrix as compute_jacobian_matrix_threaded
        return compute_jacobian_matrix_threaded(
            model, V_safe, V_unsafe, dynamics_model, translator
        )

    if N <= 256:
        print(f"[v1] Using pure vmap+jacrev (N={N} <= 256)")
        return compute_jacobian_matrix_v1_pure_vmap(
            model, V_safe, V_unsafe, dynamics_model, translator
        )
    else:
        print(f"[v1] Using batched vmap+jacrev (N={N} > 256, batch_size={batch_size})")
        return compute_jacobian_matrix_v1(
            model, V_safe, V_unsafe, dynamics_model, translator,
            batch_size=batch_size,
        )


# =============================================================================
# 辅助函数
# =============================================================================

def copy_model_with_grad(model: nn.Module) -> nn.Module:
    """克隆模型，并将所有参数的 requires_grad 设为 True。"""
    cloned = copy.deepcopy(model)
    for p in cloned.parameters():
        p.requires_grad_(True)
    return cloned
