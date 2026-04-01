"""
修复算法的优化模块：损失计算、切向空间提取、参数更新

实现三个核心函数：
1. compute_repair_loss_and_grad: 计算 Hinge Loss 并提取梯度
2. extract_tangent_space: 提取法向空间的正交基
3. project_and_update: 隐式投影和参数更新
"""

from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
import numpy as np


def compute_repair_loss_and_grad(
    model: nn.Module,
    F_safe: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    tolerance: float = -1e-12,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    针对 UNSAT 集合计算 Hinge Loss 并提取梯度。

    Args:
        model: 神经网络
        F_safe: 安全区中验证失败的单纯形列表
        F_unsafe: 障碍区中验证失败的单纯形列表
        dynamics_model: 动力学系统
        translator: TorchTranslator
        tolerance: 容差阈值，默认 -1e-12
        verbose: 是否打印详细信息

    Returns:
        (total_loss, g_raw): 总损失和展平后的梯度向量
        - total_loss: 标量损失
        - g_raw: 一维梯度向量，形状 [P]

    避坑:
        - total_loss.backward() 后使用 parameters_to_vector 提取梯度
        - 严禁使用 torch.no_grad()
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if len(F_safe) == 0 and len(F_unsafe) == 0:
        # 没有需要修复的区域，返回零损失和零梯度
        num_params = sum(p.numel() for p in model.parameters())
        return torch.tensor(0.0, dtype=dtype, device=device), \
               torch.zeros(num_params, dtype=dtype, device=device)

    # 导入 geometry_module 中的函数
    from New_repair.geometry_module import compute_simplex_bound

    total_loss = torch.tensor(0.0, dtype=dtype, device=device)
    num_terms = 0

    # ========== 处理安全区中的 UNSAT 区域 ==========
    # 安全区的目标是强迫下界 >= tolerance
    # Loss = sum(ReLU(tolerance - min_L))

    for i, vertices in enumerate(F_safe):
        # 计算 CBF 条件下界 (min_L)
        # 对于 safe 区域，我们希望 min_L >= tolerance
        min_L = compute_simplex_bound(
            model, vertices, 'safe',
            dynamics_model=dynamics_model,
            translator=translator
        )

        # Hinge Loss: max(0, tolerance - min_L)
        loss_term = torch.clamp(tolerance - min_L, min=0.0)

        if loss_term > 0:
            total_loss = total_loss + loss_term
            num_terms += 1

        if verbose and i == 0:
            print(f"  F_safe[0] min_L: {min_L.item():.6f}, loss_term: {loss_term.item():.6f}")

    # ========== 处理障碍区中的 UNSAT 区域 ==========
    # 障碍区的目标：h_max < 0（网络应该输出负值）
    # Loss = sum(ReLU(h_max - tolerance))，其中 tolerance 应该为 0

    for i, vertices in enumerate(F_unsafe):
        # 计算网络输出的上界 (h_max)
        h_max = compute_simplex_bound(
            model, vertices, 'unsafe',
            dynamics_model=None,
            translator=translator
        )

        # Hinge Loss: max(0, h_max - 0) = max(0, h_max)
        # 我们希望 h_max < 0，所以对于 h_max > 0 的区域施加惩罚
        loss_term = torch.clamp(h_max - 0.0, min=0.0)

        if loss_term > 0:
            total_loss = total_loss + loss_term
            num_terms += 1

        if verbose and i == 0:
            print(f"  F_unsafe[0] h_max: {h_max.item():.6f}, loss_term: {loss_term.item():.6f}")

    if verbose:
        print(f"  总损失项数: {num_terms}, 总损失: {total_loss.item():.6f}")

    # ========== 反向传播并提取梯度 ==========
    # 清空之前的梯度
    model.zero_grad()

    # 反向传播
    total_loss.backward()

    # 使用 parameters_to_vector 提取并展平梯度
    # 注意：必须包含所有 requires_grad=True 的参数，即使 grad 为 None
    grad_list = []
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is not None:
                grad_list.append(p.grad)
            else:
                # 如果 grad 为 None，用零填充
                grad_list.append(torch.zeros_like(p))

    # 处理没有梯度的情况
    if len(grad_list) == 0:
        num_params = sum(p.numel() for p in model.parameters())
        g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    else:
        g_raw = torch.nn.utils.parameters_to_vector(grad_list)

    return total_loss, g_raw


def extract_tangent_space(
    J: torch.Tensor,
    k_rank: int = 500
) -> Tuple[torch.Tensor, int]:
    """
    对雅可比矩阵进行截断 SVD，提取法向空间的正交基。

    Args:
        J: 雅可比矩阵，形状 [N, P]
        k_rank: 期望的 rank 上限

    Returns:
        (V_k, k_effective): 正交基矩阵和实际 rank
        - V_k: 正交基矩阵，形状 [P, k_effective]
        - k_effective: 实际使用的 rank

    避坑:
        - 动态计算 k_effective = min(k_rank, J.shape[0])
        - 确保 m < k_rank 时 SVD 不会崩溃
    """
    N, P = J.shape

    # 动态计算有效的 rank
    k_effective = min(k_rank, N)

    if k_effective <= 0:
        # 雅可比矩阵为空或 N=0
        return torch.empty(P, 0, device=J.device, dtype=J.dtype), 0

    # 使用截断 SVD
    # J = U @ S @ V^T
    # V 是 [P, k_effective] 的正交矩阵
    try:
        # torch.svd_lowrank 返回 (U, S, V)，其中 V 是右奇异向量
        U, S, V = torch.svd_lowrank(J, q=k_effective)

        # V 是 [P, k_effective] 的正交基
        V_k = V

        # 计算实际的有效 rank（基于奇异值）
        # 保留奇异值 > 1% 的成分
        singular_values = S.cpu().numpy()
        max_sv = singular_values[0] if len(singular_values) > 0 else 0

        # 计算有效 rank（奇异值 > 1% 最大值）
        tol = 0.01 * max_sv
        actual_rank = int(np.sum(singular_values > tol))

        # 确保不超过 k_effective
        actual_rank = min(actual_rank, k_effective)
        actual_rank = max(actual_rank, 1)  # 至少保留 1 维

        # 如果实际 rank 小于 k_effective，截断 V_k
        if actual_rank < k_effective:
            V_k = V_k[:, :actual_rank]

        return V_k, actual_rank

    except Exception as e:
        # 如果 SVD 失败，使用备选方案
        print(f"  警告: SVD 失败 ({e})，使用 QR 分解作为备选")
        # 使用 QR 分解作为备选
        Q, R = torch.linalg.qr(J.T)  # Q 是 [P, min(P,N)] 的正交基
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

    将梯度投影到法向空间的正交补（即切向空间），然后更新参数。

    Args:
        model: 神经网络
        g_raw: 原始梯度向量，形状 [P]
        V_k: 法向空间的正交基，形状 [P, k_effective]
        lr: 学习率
        alpha: 投影系数（0 表示完全投影到切向空间）

    Returns:
        (grad_norm, update_norm): 原始梯度范数和更新向量范数

    避坑:
        - 绝对不能构造 |theta| x |theta| 的投影矩阵
        - 必须利用 V_k 的正交性隐式计算
    """
    device = g_raw.device
    dtype = g_raw.dtype

    P = g_raw.shape[0]

    # 获取原始参数的视图
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    # 确保 g_raw 长度匹配
    if g_raw.shape[0] != num_params:
        raise ValueError(f"梯度维度不匹配: g_raw {g_raw.shape[0]} vs 参数 {num_params}")

    # ========== 隐式投影计算 ==========
    # 利用 V_k 的正交性计算投影系数
    # g_raw = g_parallel + g_perp
    # 其中 g_perp = V_k @ (V_k^T @ g_raw) 是法向分量
    # g_parallel = g_raw - g_perp 是切向分量

    if V_k.shape[1] > 0:
        # 计算 V_k^T @ g_raw：[k_effective]
        coeffs = V_k.T @ g_raw  # [k_effective]

        # 法向分量：g_perp = V_k @ coeffs：[P]
        g_perp = V_k @ coeffs

        # 切向分量：g_parallel = g_raw - g_perp
        g_parallel = g_raw - g_perp
    else:
        # 没有法向空间，g_parallel = g_raw
        g_parallel = g_raw.clone()
        g_perp = torch.zeros_like(g_raw)

    # ========== 混合投影 ==========
    # g_update = g_parallel + alpha * g_perp
    # alpha = 0: 完全投影到切向空间（梯度下降在切向空间）
    # alpha = 1: 不投影（原始梯度下降）
    g_update = g_parallel + alpha * g_perp

    if verbose:
        grad_norm = g_raw.norm().item()
        perp_norm = g_perp.norm().item()
        parallel_norm = g_parallel.norm().item()
        update_norm = g_update.norm().item()
        print(f"  |g_raw|: {grad_norm:.6f}")
        print(f"  |g_perp|: {perp_norm:.6f}")
        print(f"  |g_parallel|: {parallel_norm:.6f}")
        print(f"  |g_update|: {update_norm:.6f}")

    # ========== 参数更新 ==========
    # theta_new = theta_old - lr * g_update

    # 获取原始参数值
    theta_old = torch.nn.utils.parameters_to_vector(params)

    # 计算更新
    theta_new = theta_old - lr * g_update

    # 将更新后的参数写回模型
    torch.nn.utils.vector_to_parameters(theta_new, params)

    return g_raw.norm().item(), g_update.norm().item()


def repair_iteration(
    model: nn.Module,
    J: torch.Tensor,
    F_safe: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    k_rank: int = 500,
    lr: float = 1e-3,
    alpha: float = 0.0,
    tolerance: float = -1e-12,
    verbose: bool = False
) -> Tuple[float, float, int]:
    """
    执行一次完整的修复迭代。

    整合 compute_repair_loss_and_grad, extract_tangent_space, project_and_update。

    Args:
        model: 神经网络
        J: 雅可比矩阵，形状 [N, P]
        F_safe: 安全区中验证失败的单纯形
        F_unsafe: 障碍区中验证失败的单纯形
        dynamics_model: 动力学系统
        translator: TorchTranslator
        k_rank: SVD rank 上限
        lr: 学习率
        alpha: 投影系数
        tolerance: 容差阈值
        verbose: 是否打印详细信息

    Returns:
        (loss, grad_norm, k_effective): 损失值、梯度范数、实际使用的 rank
    """
    if verbose:
        print("\n" + "=" * 50)
        print("修复迭代")
        print("=" * 50)

    # Step 1: 提取切向空间
    if verbose:
        print(f"[1] 提取切向空间 (k_rank={k_rank})...")

    V_k, k_effective = extract_tangent_space(J, k_rank)

    if verbose:
        print(f"    实际 rank: {k_effective}")

    # Step 2: 计算损失和梯度
    if verbose:
        print(f"[2] 计算损失和梯度...")

    loss, g_raw = compute_repair_loss_and_grad(
        model, F_safe, F_unsafe,
        dynamics_model, translator,
        tolerance=tolerance,
        verbose=verbose
    )

    if verbose:
        print(f"    损失值: {loss.item():.6f}")
        print(f"    梯度范数: {g_raw.norm().item():.6f}")

    # Step 3: 投影并更新
    if verbose:
        print(f"[3] 投影并更新参数 (lr={lr}, alpha={alpha})...")

    grad_norm, update_norm = project_and_update(
        model, g_raw, V_k,
        lr=lr, alpha=alpha,
        verbose=verbose
    )

    return loss.item(), grad_norm, k_effective
