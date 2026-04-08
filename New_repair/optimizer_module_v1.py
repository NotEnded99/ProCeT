"""
修复算法优化模块 (v1): Batch 损失计算 + 切向空间提取 + 参数更新

核心改进（相比 optimizer_module.py）:
1. compute_repair_loss_and_grad_batch: 使用 CrownPartialLinearization 批量计算
   - unsafe 区域: 一次性计算所有 F_h_positive_in_unsafe 的 h_lb
   - safe 区域: 一次性计算所有 F_safe/violation/depth/unsafe_split 的 min_L
   - 避免逐个单纯形的 for 循环，利用 GPU/向量化的批量计算优势

完全复用:
- extract_tangent_space: SVD 切向空间提取（无变化）
- project_and_update: 隐式投影参数更新（无变化）
- repair_iteration: 流程编排（调用 v1 版本）
"""

from typing import List, Tuple, Union, Dict, Optional
import cvxpy as cp

import copy
import torch
import torch.nn as nn
import numpy as np

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

def _vertices_to_simplicial_batch(
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    device: torch.device,
    dtype: torch.dtype,
) -> List[SimplicialRegion]:
    """将顶点列表转换为 SimplicialRegion 列表。"""
    batch = []
    for verts in vertices_list:
        if isinstance(verts, torch.Tensor):
            verts_np = verts.cpu().numpy()
        else:
            verts_np = verts
        batch.append(SimplicialRegion(verts_np, output_dim=None))
    return batch


# =============================================================================
# Helper: 批量提取李导数下界（与 geometry_module_new.py 逻辑完全一致）
# =============================================================================

def _extract_lie_derivative_lower_bound_batch(
    network_linearizer: CrownPartialLinearization,
    dynamics_bounds,
    g_dynamics_bounds,
    batch: List[SimplicialRegion],
    dynamics_model,
    device: torch.device,
    dtype: torch.dtype,
    eta: tuple = (0.5, 0.5),
) -> torch.Tensor:
    """
    批量版本的李导数下界提取，复现 geometry_module_new:_extract_lie_derivative_lower_bound。
    输入 batch 可包含任意数量单纯形，输出 min_L 形状 [batch_size, n_state]。
    """
    n = dynamics_model.input_dim
    m = dynamics_model.control_dim
    f_affine_bounds = dynamics_bounds
    g_affine_bounds = g_dynamics_bounds

    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
    J_affine_L, J_affine_U = (A_L, b_L), (A_U, b_U)

    f_affine_L, f_affine_U = f_affine_bounds

    eta_drift = eta[0]
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_affine_L, J_affine_U,
        f_affine_L, f_affine_U,
        batch, eta=eta_drift, device=device, dtype=dtype,
    )
    M_D, c_D = M_D.sum(dim=-2), c_D.sum(dim=-1)

    (A_L_net, a_L_net), (A_U_net, a_U_net) = network_linearizer.get_network_linear_bounds()
    alpha_A_L = dynamics_model.alpha_function(A_L_net[..., 0, :])
    alpha_a_L = dynamics_model.alpha_function(a_L_net[..., 0])

    M_total = M_D + alpha_A_L
    c_total = c_D + alpha_a_L

    if m > 0 and g_affine_bounds is not None:
        g_affine_L = g_affine_bounds[0][0], g_affine_bounds[0][1]
        g_affine_U = g_affine_bounds[1][0], g_affine_bounds[1][1]

        eta_control_L = eta[1]
        M_v_L, c_v_L = _batched_compute_mccormick_product_lower_bound(
            J_affine_L, J_affine_U,
            g_affine_L, g_affine_U,
            batch, eta=eta_control_L, device=device, dtype=dtype,
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
                    sample, device=device, dtype=dtype,
                )
                v_u_max_b, _ = _vectorized_get_affine_function_bounds(
                    (M_v_L_u_max[sample_idx, mixed_mask], c_v_L_u_max[sample_idx, mixed_mask]),
                    sample, device=device, dtype=dtype,
                )
                c_C += torch.maximum(v_u_min_b, v_u_max_b).sum()

            M_total[sample_idx] += M_C
            c_total[sample_idx] += c_C

    min_L, _ = _batched_get_affine_function_bounds(
        (M_total.unsqueeze(1), c_total.unsqueeze(1)),
        batch, device=device, dtype=dtype,
    )
    min_L = min_L.squeeze(-1)
    return min_L


# =============================================================================
# 核心: 批量损失计算
# =============================================================================

def compute_repair_loss_and_grad_batch(
    model: nn.Module,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    tolerance: float = -1e-12,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量计算 Hinge Loss 及其梯度。

    优化策略：
    - unsafe 区域：一次性构建 batch，计算所有单纯形的 h_lb
    - safe 区域：一次性构建 batch，计算所有单纯形的 min_L
    - 所有违规项的损失通过向量化的 clamp + relu 合并
    - 单次 backward 计算所有梯度

    Args:
        model: 神经网络
        F_h_positive_in_unsafe: 障碍区内 h(x)>=0 的单纯形列表
        F_safe_cbf_violation: 安全区 CBF 违规的单纯形列表
        F_depth_limit_reached: 达到深度的单纯形列表
        F_unsafe_cannot_split: 无法细分的单纯形列表
        dynamics_model: 动力学系统
        translator: TorchTranslator
        tolerance: CBF 条件容忍度
        verbose: 是否打印调试信息

    Returns:
        (total_loss, g_raw): 总损失和展平梯度向量 [num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    total_failure = (len(F_h_positive_in_unsafe) + len(F_safe_cbf_violation) +
                     len(F_depth_limit_reached) + len(F_unsafe_cannot_split))

    num_params = sum(p.numel() for p in model.parameters())

    if total_failure == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), \
               torch.zeros(num_params, dtype=dtype, device=device)

    # ---- 克隆模型用于构建计算图 ----
    model_grad = copy.deepcopy(model)
    for p in model_grad.parameters():
        p.requires_grad_(True)

    # =================================================================
    # Part A: unsafe 区域批量计算 (F_h_positive_in_unsafe)
    # =================================================================
    unsafe_h_lb_list = []  # 每个元素形状 [n_out]

    if len(F_h_positive_in_unsafe) > 0:
        unsafe_batch = _vertices_to_simplicial_batch(F_h_positive_in_unsafe, device, dtype)

        lin_unsafe = CrownPartialLinearization(model_grad, dtype=dtype)
        lin_unsafe.compute_network_bounds(unsafe_batch)

        h_lb_batch, h_ub_batch = lin_unsafe.get_network_output_bounds()
        # h_lb_batch: [num_unsafe, output_dim]
        unsafe_h_lb_list = [h_lb_batch[i] for i in range(len(unsafe_batch))]

        if verbose:
            print(f"  Unsafe batch: {len(unsafe_batch)} simplices, "
                  f"h_lb range: [{h_lb_batch.min().item():.4f}, {h_lb_batch.max().item():.4f}]")

    # =================================================================
    # Part B: safe 区域批量计算 (三个列表合并)
    # =================================================================
    safe_vertices_list = F_safe_cbf_violation + F_depth_limit_reached + F_unsafe_cannot_split

    safe_min_L_list = []  # 每个元素形状 [n_state]

    if len(safe_vertices_list) > 0:
        safe_batch = _vertices_to_simplicial_batch(safe_vertices_list, device, dtype)

        lin_safe = CrownPartialLinearization(model_grad, dtype=dtype)
        lin_safe.compute_network_bounds(safe_batch)
        lin_safe.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

        try:
            f_bounds, g_bounds = _compute_dynamics_bounds_taylor(
                safe_batch, dynamics_model, device=device, dtype=dtype
            )
        except ValueError:
            # Dynamics 边界计算失败，跳过（返回 0 损失）
            model_grad.zero_grad()
            return torch.tensor(0.0, dtype=dtype, device=device), \
                   torch.zeros(num_params, dtype=dtype, device=device)

        min_L_batch = _extract_lie_derivative_lower_bound_batch(
            network_linearizer=lin_safe,
            dynamics_bounds=f_bounds,
            g_dynamics_bounds=g_bounds,
            batch=safe_batch,
            dynamics_model=dynamics_model,
            device=device,
            dtype=dtype,
        )
        # min_L_batch: [num_safe_total, n_state]
        safe_min_L_list = [min_L_batch[i] for i in range(len(safe_batch))]

        if verbose:
            print(f"  Safe batch: {len(safe_batch)} simplices (violation={len(F_safe_cbf_violation)}, "
                  f"depth={len(F_depth_limit_reached)}, unsplit={len(F_unsafe_cannot_split)}), "
                  f"min_L range: [{min_L_batch.min().item():.4f}, {min_L_batch.max().item():.4f}]")

    # =================================================================
    # Part C: 构建损失项（向量化的 clamp + relu）
    # =================================================================
    # unsafe: clamp(h_lb - 0, min=0) → 惩罚 h_lb > 0
    unsafe_loss_terms = []
    for h_lb in unsafe_h_lb_list:
        # 障碍区中，h(x) < 0 才安全。惩罚 h_lb > 0
        loss_term = torch.clamp(h_lb, min=0.0)  # h_lb > 0 → positive loss
        unsafe_loss_terms.append(loss_term.sum())

    # safe: clamp(tolerance - min_L, min=0) → 惩罚 min_L < tolerance
    safe_loss_terms = []
    tol_tensor = torch.tensor(tolerance, dtype=dtype, device=device)

    for min_L in safe_min_L_list:
        # 安全区内，min_L >= tolerance 才满足 CBF 条件。惩罚 min_L < tolerance
        loss_term = torch.clamp(tol_tensor - min_L, min=0.0)
        safe_loss_terms.append(loss_term.sum())

    # 合并所有损失项
    all_loss_terms = unsafe_loss_terms + safe_loss_terms

    if len(all_loss_terms) == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), \
               torch.zeros(num_params, dtype=dtype, device=device)

    # 堆叠并求和（自动过滤掉 0 值）
    loss_tensor = torch.stack(all_loss_terms)  # [num_active_terms]
    total_loss = loss_tensor.sum()

    if verbose:
        num_active = (loss_tensor > 0).sum().item()
        print(f"  Active loss terms: {num_active}/{len(all_loss_terms)}, "
              f"total loss: {total_loss.item():.6f}")

    # =================================================================
    # Part D: 反向传播并提取梯度
    # =================================================================
    model_grad.zero_grad()
    total_loss.backward()

    # 展平并拼接梯度
    grad_list = []
    for p in model_grad.parameters():
        if p.requires_grad:
            if p.grad is not None:
                grad_list.append(p.grad.flatten())
            else:
                grad_list.append(torch.zeros(p.numel(), dtype=dtype, device=device))

    if len(grad_list) == 0:
        g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    else:
        g_raw = torch.cat(grad_list)

    if torch.isnan(g_raw).any():
        print("  警告: 梯度中存在 NaN，已替换为 0。")
        g_raw = torch.nan_to_num(g_raw, nan=0.0)

    if verbose:
        print(f"  梯度范数: {g_raw.norm().item():.6f}, 非零元素: {(g_raw != 0).sum().item()}")

    return total_loss, g_raw


# =============================================================================
# 切向空间提取 & 投影更新（与 optimizer_module.py 完全一致）
# =============================================================================

def extract_tangent_space(
    J: torch.Tensor,
    k_rank: int = 500
) -> Tuple[torch.Tensor, int]:
    """
    对雅可比矩阵进行截断 SVD，提取法向空间的正交基。
    """
    N, P = J.shape

    k_effective = min(k_rank, N)

    if k_effective <= 0:
        return torch.empty(P, 0, device=J.device, dtype=J.dtype), 0

    try:
        U, S, V = torch.svd_lowrank(J, q=k_effective)
        V_k = V

        singular_values = S.cpu().numpy()
        max_sv = singular_values[0] if len(singular_values) > 0 else 0
        tol = 0.01 * max_sv
        actual_rank = int(np.sum(singular_values > tol))
        actual_rank = min(actual_rank, k_effective)
        actual_rank = max(actual_rank, 1)

        if actual_rank < k_effective:
            V_k = V_k[:, :actual_rank]

        return V_k, actual_rank

    except Exception as e:
        print(f"  警告: SVD 失败 ({e})，使用 QR 分解作为备选")
        Q, R = torch.linalg.qr(J.T)
        k_actual = min(k_rank, R.shape[0])
        V_k = Q[:, :k_actual]
        return V_k, k_actual


def project_and_update(
    model: nn.Module,
    g_raw: torch.Tensor,
    V_k: torch.Tensor,
    lr: float = 1e-3,
    alpha: float = 0.0,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    执行隐式投影和参数更新。
    """
    device = g_raw.device
    dtype = g_raw.dtype

    P = g_raw.shape[0]

    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    if g_raw.shape[0] != num_params:
        raise ValueError(f"梯度维度不匹配: g_raw {g_raw.shape[0]} vs 参数 {num_params}")

    if V_k.shape[1] > 0:
        coeffs = V_k.T @ g_raw
        g_perp = V_k @ coeffs
        g_parallel = g_raw - g_perp
    else:
        g_parallel = g_raw.clone()
        g_perp = torch.zeros_like(g_raw)

    g_update = g_parallel + alpha * g_perp

    grad_norm = g_raw.norm().item()

    if verbose:
        perp_norm = g_perp.norm().item()
        parallel_norm = g_parallel.norm().item()
        update_norm = g_update.norm().item()
        print(f"  |g_raw|: {grad_norm:.6f}")
        print(f"  |g_perp|: {perp_norm:.6f}")
        print(f"  |g_parallel|: {parallel_norm:.6f}")
        print(f"  |g_update|: {update_norm:.6f}")

    theta_old = torch.nn.utils.parameters_to_vector(params)
    theta_new = theta_old - lr * g_update
    torch.nn.utils.vector_to_parameters(theta_new, params)

    return grad_norm, g_update.norm().item()


# =============================================================================
# 流程编排
# =============================================================================

def repair_iteration_batch(
    model: nn.Module,
    J: torch.Tensor,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    k_rank: int = 500,
    lr: float = 1e-3,
    alpha: float = 0.0,
    tolerance: float = -1e-12,
    verbose: bool = False
) -> Tuple[float, float, int]:
    """
    单次修复迭代（v1 批量版本）。

    流程:
    1. SVD 提取切向空间
    2. 批量计算损失和梯度
    3. 投影并更新参数
    """
    if verbose:
        print(f"[1] 提取切向空间 (k_rank={k_rank})...")

    V_k, k_effective = extract_tangent_space(J, k_rank)

    if verbose:
        print(f"    实际 rank: {k_effective}")

    if verbose:
        print(f"[2] 批量计算损失和梯度...")

    loss, g_raw = compute_repair_loss_and_grad_batch(
        model,
        F_h_positive_in_unsafe,
        F_safe_cbf_violation,
        F_depth_limit_reached,
        F_unsafe_cannot_split,
        dynamics_model,
        translator,
        tolerance=tolerance,
        verbose=verbose,
    )

    if verbose:
        print(f"    损失值: {loss.item():.6f}")
        print(f"    梯度范数: {g_raw.norm().item():.6f}")

    if verbose:
        print(f"[3] 投影并更新参数 (lr={lr}, alpha={alpha})...")

    grad_norm, update_norm = project_and_update(
        model, g_raw, V_k,
        lr=lr, alpha=alpha,
        verbose=verbose,
    )

    return loss.item(), grad_norm, k_effective


def repair_loop_batch(
    model: nn.Module,
    J: torch.Tensor,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    max_iters: int = 20,
    target_loss: float = 1e-6,
    grad_tol: float = 1e-6,
    k_rank: int = 500,
    lr: float = 1e-3,
    alpha: float = 0.0,
    tolerance: float = -1e-12,
    verbose: bool = False,
) -> Tuple[float, float, int]:
    """
    多轮修复循环（v1 批量版本）。
    """
    print(f"\n{'=' * 50}")
    print(f"Batch 修复循环 (Max Iters: {max_iters}, Target Loss: {target_loss})")
    print(f"{'=' * 50}")

    history = []

    for i in range(max_iters):
        if verbose:
            print(f"\n>>> [Iter {i+1}/{max_iters}]")

        loss, grad_norm, k_effective = repair_iteration_batch(
            model=model,
            J=J,
            F_h_positive_in_unsafe=F_h_positive_in_unsafe,
            F_safe_cbf_violation=F_safe_cbf_violation,
            F_depth_limit_reached=F_depth_limit_reached,
            F_unsafe_cannot_split=F_unsafe_cannot_split,
            dynamics_model=dynamics_model,
            translator=translator,
            k_rank=k_rank,
            lr=lr,
            alpha=alpha,
            tolerance=tolerance,
            verbose=verbose,
        )

        history.append({
            "iter": i + 1,
            "loss": loss,
            "grad_norm": grad_norm,
            "k_effective": k_effective,
        })

        if verbose:
            print(f"    Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f} | Rank: {k_effective}")
        else:
            print(f"  Iter {i+1:02d}/{max_iters} | Loss: {loss:.4f} | "
                  f"Grad Norm: {grad_norm:.4f} | Rank: {k_effective}")

        if loss <= target_loss:
            print(f"\n  修复成功！Loss 已降至 {loss:.8f} (<= {target_loss})")
            break

        if grad_norm < grad_tol:
            print(f"\n  提前停止！梯度范数 ({grad_norm:.8f}) 过小 (< {grad_tol})")
            break

    print(f"\n  修复循环结束。共 {len(history)} 轮。最终 Loss: {history[-1]['loss']:.6f}")
    return loss, grad_norm, k_effective
