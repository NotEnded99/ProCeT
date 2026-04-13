"""
Randomized Smoothing (RS) Jacobian 估计模块 v4 (优化版)

核心公式：
    J_RS ≈ (1/(N * sigma^2)) * sum_i[psi_invar(theta_old + epsilon_i) * epsilon_i]
    其中 epsilon_i ~ N(0, sigma^2 I)

核心思想：
    - 不直接计算 ∂psi/∂theta（需要经过激活函数的反向传播，梯度爆炸）
    - 使用重参数化技巧：theta' = theta + epsilon，对 epsilon 求和
    - psi 只需要前向传播，不需要任何反向传播

并行化策略（共享 epsilon 版本）：
    - 预采样 N 个 epsilon（所有单纯形共用同一组 epsilon）
    - 对每个 epsilon_i：
        1. 扰动参数：theta_i = theta_old + epsilon_i
        2. Crown 批量处理所有单纯形（利用批量能力）
        3. 得到所有单纯形的 psi 值
        4. 累积贡献
    - 效率：B 个单纯形 × N 个 epsilon，只需要 N 次 Crown 调用（每次批量 B 个单纯形）
      而不是 B × N 次
"""

from typing import List, Tuple, Union

import copy
import numpy as np
import torch
import torch.nn as nn

from New_repair.geometry_module_new import (
    compute_simplex_bound,
    compute_simplex_bound_batch,
)


def compute_jacobian_rs(
    model: nn.Module,
    V_safe: List[Union[torch.Tensor, np.ndarray]],
    V_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    N: int = 100,
    sigma: float = 0.01,
) -> torch.Tensor:
    """
    使用随机平滑估计 Jacobian（高效共享 epsilon 版本）。

    核心优化：所有单纯形共享同一组 N 个 epsilon 噪声。
    对每个 epsilon_i，批量处理所有单纯形（Crown 支持单纯形批量）。

    效率对比：
        旧版本（串行）：B 单纯形 × N epsilon = B×N 次前向
        新版本（共享）：N epsilon × 1次批量 = N 次前向（每次批量 B 个单纯形）

    梯度公式：
        J_RS[s] = (1/(N * sigma^2)) * sum_i[psi_s(theta + eps_i) * eps_i]

    其中 psi_s 是单纯形 s 的 CBF 条件下界。

    Args:
        model: 神经网络
        V_safe: 已验证安全区单纯形列表
        V_unsafe: 已验证障碍区单纯形列表
        dynamics_model: 动力学系统（safe 区域需要）
        translator: TorchTranslator
        N: 采样次数（每个单纯形使用同一组 N 个 epsilon）
        sigma: 噪声标准差

    Returns:
        J_RS: 形状 [N_simplices, num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 1. 收集并分离单纯形 ----------
    V_safe_list = list(V_safe) if V_safe else []
    V_unsafe_list = list(V_unsafe) if V_unsafe else []

    n_safe = len(V_safe_list)
    n_unsafe = len(V_unsafe_list)
    n_simplices = n_safe + n_unsafe

    num_params = sum(p.numel() for p in model.parameters())

    if n_simplices == 0:
        return torch.zeros(0, num_params, dtype=dtype, device=device)

    print(f"  [RS Jacobian] n_simplices={n_simplices} (safe={n_safe}, unsafe={n_unsafe})")
    print(f"  [RS Jacobian] num_params={num_params}, N={N}, sigma={sigma}")
    print(f"  [RS Jacobian] 策略: 共享 epsilon 噪声，批量处理所有单纯形")

    # ---------- 2. 预采样 N 个 epsilon（所有单纯形共用）----------
    # eps_samples: [N, num_params]
    eps_samples = torch.randn(N, num_params, dtype=dtype, device=device) * sigma

    # 保存原始参数
    original_state = copy.deepcopy(model.state_dict())
    theta_old = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    # ---------- 3. 初始化累积器 ----------
    # J_RS[s] = (1/(N*sigma^2)) * sum_i[psi_s(theta+eps_i) * eps_i]
    # accumulator[s] = sum_i[psi_s(theta+eps_i) * eps_i]
    J_accumulator = torch.zeros(n_simplices, num_params, dtype=dtype, device=device)
    valid_counts = torch.zeros(n_simplices, dtype=torch.int32)

    # ---------- 4. 对每个 epsilon 执行批量计算 ----------
    for eps_idx in range(N):
        eps_i = eps_samples[eps_idx]  # [num_params]

        # 4a. 扰动参数
        theta_i = theta_old + eps_i
        torch.nn.utils.vector_to_parameters(theta_i, model.parameters())

        # 4b. 批量计算所有 safe 单纯形的 psi（min_L）
        if n_safe > 0:
            with torch.no_grad():
                min_L_all = compute_simplex_bound_batch(
                    model, V_safe_list, 'safe',
                    dynamics_model=dynamics_model, translator=translator
                )  # [n_safe]
                # 检查有效性
                valid_mask = torch.isfinite(min_L_all) & ~torch.isnan(min_L_all)
                # 累积到前 n_safe 行
                for s_idx in range(n_safe):
                    if valid_mask[s_idx]:
                        J_accumulator[s_idx].add_(eps_i * min_L_all[s_idx].detach())
                        valid_counts[s_idx] += 1

        # 4c. 批量计算所有 unsafe 单纯形的 psi（h_ub）
        if n_unsafe > 0:
            with torch.no_grad():
                _, h_ub_all = compute_simplex_bound_batch(
                    model, V_unsafe_list, 'unsafe',
                    dynamics_model=None, translator=None
                )  # [n_unsafe]
                # 检查有效性
                valid_mask = torch.isfinite(h_ub_all) & ~torch.isnan(h_ub_all)
                # 累积到后 n_unsafe 行
                for s_idx in range(n_unsafe):
                    if valid_mask[s_idx]:
                        J_accumulator[n_safe + s_idx].add_(eps_i * h_ub_all[s_idx].detach())
                        valid_counts[n_safe + s_idx] += 1

        # 进度打印
        if (eps_idx + 1) % 20 == 0 or eps_idx + 1 == N:
            print(f"    RS epsilon 进度: {eps_idx + 1}/{N}")

    # ---------- 5. 恢复原始参数 ----------
    model.load_state_dict(original_state)

    # ---------- 6. 计算最终 Jacobian ----------
    # J_RS = accumulator / (N * sigma^2)
    # 处理有效计数为 0 的行
    sigma_sq = sigma * sigma
    J_RS = torch.zeros_like(J_accumulator)
    for s in range(n_simplices):
        if valid_counts[s] > 0:
            J_RS[s] = J_accumulator[s] / (valid_counts[s].item() * sigma_sq)
        # else: 保持全零（无效单纯形）

    print(f"  [RS Jacobian] 完成，有效单纯形: {(valid_counts > 0).sum().item()}/{n_simplices}")
    return J_RS


def compute_jacobian_rs_sequential(
    model: nn.Module,
    V_safe: List[Union[torch.Tensor, np.ndarray]],
    V_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    N: int = 100,
    sigma: float = 0.01,
) -> torch.Tensor:
    """
    串行版 RS Jacobian（每个单纯形独立处理）。

    与 compute_jacobian_rs 的区别：
    - 不共享 epsilon，每个单纯形独立采样 N 个 epsilon
    - 用于对比验证

    Args:
        同 compute_jacobian_rs

    Returns:
        J_RS: 形状 [N_simplices, num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 1. 收集所有单纯形 ----------
    all_vertices = []
    all_region_types = []
    for v in V_safe:
        all_vertices.append(v)
        all_region_types.append('safe')
    for v in V_unsafe:
        all_vertices.append(v)
        all_region_types.append('unsafe')

    n_simplices = len(all_vertices)
    num_params = sum(p.numel() for p in model.parameters())

    if n_simplices == 0:
        return torch.zeros(0, num_params, dtype=dtype, device=device)

    print(f"  [RS Jacobian Seq] n_simplices={n_simplices}, N={N}, sigma={sigma}")

    # 保存原始参数
    original_state = copy.deepcopy(model.state_dict())
    theta_old = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    # 结果矩阵
    J_RS = torch.zeros(n_simplices, num_params, dtype=dtype, device=device)

    for s_idx in range(n_simplices):
        vertices = all_vertices[s_idx]
        region_type = all_region_types[s_idx]

        accumulator = torch.zeros(num_params, dtype=dtype, device=device)
        valid_count = 0

        for _ in range(N):
            eps_i = torch.randn(num_params, dtype=dtype, device=device) * sigma

            # 扰动参数
            theta_i = theta_old + eps_i
            torch.nn.utils.vector_to_parameters(theta_i, model.parameters())

            # 前向计算 psi
            with torch.no_grad():
                if region_type == 'unsafe':
                    _, h_ub = compute_simplex_bound(
                        model, vertices, 'unsafe',
                        dynamics_model=None, translator=None
                    )
                    psi_val = h_ub.squeeze().detach()
                else:
                    min_L = compute_simplex_bound(
                        model, vertices, 'safe',
                        dynamics_model=dynamics_model, translator=translator
                    )
                    psi_val = min_L.squeeze().detach()

            if torch.isfinite(psi_val) and not torch.isnan(psi_val):
                accumulator.add_(eps_i * psi_val)
                valid_count += 1

        # 恢复原始参数
        torch.nn.utils.vector_to_parameters(theta_old.clone(), model.parameters())

        if valid_count > 0:
            J_RS[s_idx] = accumulator / (valid_count * sigma * sigma)

        if (s_idx + 1) % 100 == 0 or s_idx + 1 == n_simplices:
            print(f"    Progress: {s_idx + 1}/{n_simplices}")

    model.load_state_dict(original_state)
    return J_RS


