"""
修复算法的优化模块：损失计算、切向空间提取、参数更新

实现三个核心函数：
1. compute_repair_loss_and_grad: 计算 Hinge Loss 并提取梯度
2. extract_tangent_space: 提取法向空间的正交基
3. project_and_update: 隐式投影和参数更新
"""

from typing import List, Tuple, Union, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp


def compute_repair_loss_and_grad(
    model: nn.Module,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    tolerance: float = -1e-12,
    verbose: bool = False,
    grad_clip_norm: float = 10.0,
    normalize_loss: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    total_failure = (len(F_h_positive_in_unsafe) + len(F_safe_cbf_violation) +
                     len(F_depth_limit_reached) + len(F_unsafe_cannot_split))

    num_params = sum(p.numel() for p in model.parameters())

    if total_failure == 0:
        return (torch.tensor(0.0, dtype=dtype, device=device),
                torch.zeros(num_params, dtype=dtype, device=device))

    from New_repair.geometry_module_new import compute_simplex_bound

    # ---------- 阶段一：前向计算，收集 valid term ----------
    _valid_terms: List[torch.Tensor] = []  # loss_term 列表
    _nan_terms: List[Tuple] = []            # (loss_tag, idx, reason)

    # ---------- F_h_positive_in_unsafe ----------
    for i, vertices in enumerate(F_h_positive_in_unsafe):
        h_lb, h_ub = compute_simplex_bound(
            model, vertices, 'unsafe',
            dynamics_model=None, translator=translator
        )
        loss_term = torch.clamp(h_lb - 0.0, min=0.0)
        if loss_term > 0:
            if torch.isnan(loss_term).any() or torch.isnan(h_ub).any():
                _nan_terms.append(('F_h', i, 'forward_nan'))
            else:
                _valid_terms.append(loss_term)
        if verbose and i == 0:
            print(f"  F_h[0] h_lb={h_lb.item():.6f}, loss={loss_term.item():.6f}")

    # ---------- F_safe_cbf_violation ----------
    for i, vertices in enumerate(F_safe_cbf_violation):
        min_L = compute_simplex_bound(
            model, vertices, 'safe',
            dynamics_model=dynamics_model, translator=translator
        )
        loss_term = torch.clamp(tolerance - min_L, min=0.0)
        if loss_term > 0:
            if torch.isnan(loss_term).any() or torch.isnan(min_L).any():
                _nan_terms.append(('F_safe', i, 'forward_nan'))
            else:
                _valid_terms.append(loss_term)
        if verbose and i == 0:
            print(f"  F_safe[0] min_L={min_L.item():.6f}, loss={loss_term.item():.6f}")

    # ---------- F_depth_limit_reached ----------
    for i, vertices in enumerate(F_depth_limit_reached):
        min_L = compute_simplex_bound(
            model, vertices, 'safe',
            dynamics_model=dynamics_model, translator=translator
        )
        loss_term = torch.clamp(tolerance - min_L, min=0.0)
        if loss_term > 0:
            if torch.isnan(loss_term).any() or torch.isnan(min_L).any():
                _nan_terms.append(('F_depth', i, 'forward_nan'))
            else:
                _valid_terms.append(loss_term)
        if verbose and i == 0:
            print(f"  F_depth[0] min_L={min_L.item():.6f}, loss={loss_term.item():.6f}")

    # ---------- F_unsafe_cannot_split ----------
    for i, vertices in enumerate(F_unsafe_cannot_split):
        min_L = compute_simplex_bound(
            model, vertices, 'safe',
            dynamics_model=dynamics_model, translator=translator
        )
        loss_term = torch.clamp(tolerance - min_L, min=0.0)
        if loss_term > 0:
            if torch.isnan(loss_term).any() or torch.isnan(min_L).any():
                _nan_terms.append(('F_unsafe', i, 'forward_nan'))
            else:
                _valid_terms.append(loss_term)
        if verbose and i == 0:
            print(f"  F_unsafe[0] min_L={min_L.item():.6f}, loss={loss_term.item():.6f}")

    # ---------- 报告统计 ----------
    n_valid = len(_valid_terms)
    n_nan = len(_nan_terms)
    if n_nan > 0:
        print(f"  [警告] {n_nan}/{total_failure} 个 simplex 前向计算产生 NaN, 已跳过: "
              f"{_nan_terms[:5]}{'...' if len(_nan_terms) > 5 else ''}")

    if n_valid == 0:
        return (torch.tensor(0.0, dtype=dtype, device=device),
                torch.zeros(num_params, dtype=dtype, device=device))

    if verbose:
        loss_tag_counts = {}
        for t in _valid_terms:
            pass  # 暂不追踪 tag，保持简洁
        print(f"  有效 loss 项: {n_valid}/{total_failure}")

    total_loss = sum(_valid_terms)

    if verbose:
        print(f"  total_loss (raw sum): {total_loss.item():.6f}")


    model.zero_grad()
    total_loss.backward()

    g_raw = torch.cat([
        p.grad.flatten() if p.grad is not None
        else torch.zeros(p.numel(), dtype=dtype, device=device)
        for p in model.parameters()
    ])

    return total_loss, g_raw


def compute_repair_loss_and_grad_perterm(
    model: nn.Module,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    tolerance: float = -1e-12,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    total_failure = (len(F_h_positive_in_unsafe) + len(F_safe_cbf_violation) +
                     len(F_depth_limit_reached) + len(F_unsafe_cannot_split))

    if total_failure == 0:
        num_params = sum(p.numel() for p in model.parameters())
        return torch.tensor(0.0, dtype=dtype, device=device), \
               torch.zeros(num_params, dtype=dtype, device=device)

    # 导入 geometry_module 中的函数
    from New_repair.geometry_module_new import compute_simplex_bound

    total_loss = torch.tensor(0.0, dtype=dtype, device=device)
    num_terms = 0

    # # ========== 处理 F_h_positive_in_unsafe ==========
    # 障碍区内 h(x) >= 0 的违规：惩罚 h_lb > 0
    # 使用 region_type='unsafe'，dynamics_model=None
    for i, vertices in enumerate(F_h_positive_in_unsafe):
        h_lb, h_ub = compute_simplex_bound(
            model, vertices, 'unsafe',
            dynamics_model=None,
            translator=translator
        )
        # 惩罚 h_lb > 0（safe 区域的 h 应该 < 0）
        loss_term = torch.clamp(h_lb - 0.0, min=0.0)
        if loss_term > 0:
            total_loss = total_loss + loss_term
            num_terms += 1

        if verbose and i == 0:
            print(f"  F_h_positive_in_unsafe[0] h_lb: {h_lb.item():.6f}, loss: {loss_term.item():.6f}")

    # ========== 处理 F_safe_cbf_violation ==========
    # 安全区内 CBF 条件违规：惩罚 min_L < tolerance
    for i, vertices in enumerate(F_safe_cbf_violation):
        min_L = compute_simplex_bound(
            model, vertices, 'safe',
            dynamics_model=dynamics_model,
            translator=translator
        )
        loss_term = torch.clamp(tolerance - min_L, min=0.0)

        if loss_term > 0:
            total_loss = total_loss + loss_term
            num_terms += 1

        if verbose and i == 0:
            print(f"  F_safe_cbf_violation[0] min_L: {min_L.item():.6f}, loss: {loss_term.item():.6f}")

    # ========== 处理 F_depth_limit_reached ==========
    # 达到最大分裂深度：惩罚 min_L < tolerance
    for i, vertices in enumerate(F_depth_limit_reached):
        min_L = compute_simplex_bound(
            model, vertices, 'safe',
            dynamics_model=dynamics_model,
            translator=translator
        )
        loss_term = torch.clamp(tolerance - min_L, min=0.0)
        if loss_term > 0:
            total_loss = total_loss + loss_term
            num_terms += 1

        if verbose and i == 0:
            print(f"  F_depth_limit_reached[0] min_L: {min_L.item():.6f}, loss: {loss_term.item():.6f}")

    # ========== 处理 F_unsafe_cannot_split ==========
    # 障碍区无法继续细分：惩罚 min_L < tolerance
    for i, vertices in enumerate(F_unsafe_cannot_split):
        min_L = compute_simplex_bound(
            model, vertices, 'safe',
            dynamics_model=dynamics_model,
            translator=translator
        )
        loss_term = torch.clamp(tolerance - min_L, min=0.0)

        if loss_term > 0:
            total_loss = total_loss + loss_term
            num_terms += 1

        if verbose and i == 0:
            print(f"  F_unsafe_cannot_split[0] min_L: {min_L.item():.6f}, loss: {loss_term.item():.6f}")

    if verbose:
        print(f"  总损失项数: {num_terms}, 总损失: {total_loss.item():.6f}")

    # ========== 反向传播并提取梯度 ==========
    # 使用 per-term backward 替代 total_loss.backward() 来避免累积图的数值不稳定性。
    # total_loss.backward() 在大量 loss_term 累积时可能产生 NaN（迭代间传播的数值误差），
    # 而 per-term backward 每个 simplex 独立计算，保持数值稳定。
    from New_repair.geometry_module_new import compute_simplex_bound as csb

    num_params = sum(p.numel() for p in model.parameters())
    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    actual_loss_sum = torch.tensor(0.0, dtype=dtype, device=device)
    nan_terms = []

    # F_h terms
    for i, vertices in enumerate(F_h_positive_in_unsafe):
        h_lb, h_ub = csb(model, vertices, 'unsafe', dynamics_model=None, translator=translator)
        loss_term = torch.clamp(h_lb - 0.0, min=0.0)
        if loss_term > 0:
            actual_loss_sum = actual_loss_sum + loss_term
            model.zero_grad()
            loss_term.backward(retain_graph=False)
            grad_this = torch.cat([p.grad.flatten() if p.grad is not None
                                   else torch.zeros(p.numel(), dtype=dtype, device=device)
                                   for p in model.parameters()])
            if grad_this.isnan().any():
                nan_terms.append(('F_h', i, grad_this.isnan().sum().item()))
                g_raw.add_(torch.nan_to_num(grad_this, nan=0.0))
            else:
                g_raw.add_(grad_this)

    # F_safe terms
    for i, vertices in enumerate(F_safe_cbf_violation):
        min_L = csb(model, vertices, 'safe', dynamics_model=dynamics_model, translator=translator)
        loss_term = torch.clamp(tolerance - min_L, min=0.0)
        if loss_term > 0:
            actual_loss_sum = actual_loss_sum + loss_term
            model.zero_grad()
            loss_term.backward(retain_graph=False)
            grad_this = torch.cat([p.grad.flatten() if p.grad is not None
                                   else torch.zeros(p.numel(), dtype=dtype, device=device)
                                   for p in model.parameters()])
            if grad_this.isnan().any():
                nan_terms.append(('F_safe', i, grad_this.isnan().sum().item()))
                g_raw.add_(torch.nan_to_num(grad_this, nan=0.0))
            else:
                g_raw.add_(grad_this)

    # F_depth terms
    for i, vertices in enumerate(F_depth_limit_reached):
        min_L = csb(model, vertices, 'safe', dynamics_model=dynamics_model, translator=translator)
        loss_term = torch.clamp(tolerance - min_L, min=0.0)
        if loss_term > 0:
            actual_loss_sum = actual_loss_sum + loss_term
            model.zero_grad()
            loss_term.backward(retain_graph=False)
            grad_this = torch.cat([p.grad.flatten() if p.grad is not None
                                   else torch.zeros(p.numel(), dtype=dtype, device=device)
                                   for p in model.parameters()])
            if grad_this.isnan().any():
                nan_terms.append(('F_depth', i, grad_this.isnan().sum().item()))
                g_raw.add_(torch.nan_to_num(grad_this, nan=0.0))
            else:
                g_raw.add_(grad_this)

    # F_unsafe_split terms
    for i, vertices in enumerate(F_unsafe_cannot_split):
        min_L = csb(model, vertices, 'safe', dynamics_model=dynamics_model, translator=translator)
        loss_term = torch.clamp(tolerance - min_L, min=0.0)
        if loss_term > 0:
            actual_loss_sum = actual_loss_sum + loss_term
            model.zero_grad()
            loss_term.backward(retain_graph=False)
            grad_this = torch.cat([p.grad.flatten() if p.grad is not None
                                   else torch.zeros(p.numel(), dtype=dtype, device=device)
                                   for p in model.parameters()])
            if grad_this.isnan().any():
                nan_terms.append(('F_unsafe_split', i, grad_this.isnan().sum().item()))
                g_raw.add_(torch.nan_to_num(grad_this, nan=0.0))
            else:
                g_raw.add_(grad_this)

    if nan_terms:
        print(f"  警告: {len(nan_terms)} 个 simplex 产生 NaN 梯度: {nan_terms[:3]}")

    return actual_loss_sum, g_raw



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

    # if verbose:
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
    print(f"  |theta_old|: {theta_old.norm().item():.6f}")
    print("theta_old", theta_old[0:100]) # 打印前100个参数值以检查更新前状态

    # 计算更新
    theta_new = theta_old - lr * g_update

    # 将更新后的参数写回模型
    torch.nn.utils.vector_to_parameters(theta_new, params)

    print(f"  |theta_new|: {theta_new.norm().item():.6f}")
    print("theta_new", theta_new[0:100]) # 打印前100个参数值以检查更新

    return g_raw.norm().item(), g_update.norm().item()


def repair_iteration(
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


    if verbose:
        print(f"[1] 提取切向空间 (k_rank={k_rank})...")

    V_k, k_effective = extract_tangent_space(J, k_rank)

    if verbose:
        print(f"    实际 rank: {k_effective}")

    # Step 2: 计算损失和梯度
    if verbose:
        print(f"[2] 计算损失和梯度...")

    loss, g_raw = compute_repair_loss_and_grad(
        model,
        F_h_positive_in_unsafe,
        F_safe_cbf_violation,
        F_depth_limit_reached,
        F_unsafe_cannot_split,
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


def repair_new_iteration(
    model: nn.Module,
    J: torch.Tensor,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    lr: float = 1e-3,           # 注意: 这里的 lr 变成了绝对步长 (因为梯度会被归一化)
    tolerance: float = -1e-12,
    verbose: bool = False
) -> Tuple[float, float, int]:
    """
    执行单次约束修复迭代 (基于 QP 对偶投影)
    """

    if verbose:
        print(f"[1] 计算修复损失和原始梯度...")

    # 避坑提醒：请确保在 compute_repair_loss_and_grad 内部，
    # 针对不同失败区域的 Loss 是【求平均(Mean)】而不是【求和(Sum)】
    loss, g_raw = compute_repair_loss_and_grad(
        model,
        F_h_positive_in_unsafe,
        F_safe_cbf_violation,
        F_depth_limit_reached,
        F_unsafe_cannot_split,
        dynamics_model, 
        translator,
        tolerance=tolerance,
        verbose=verbose
    )

    if verbose:
        print(f"    损失值 (Loss): {loss.item():.6f}")
        print(f"    原始梯度范数 (|g_raw|): {g_raw.norm().item():.6f}")

    # ==========================================
    # Step 2: 求解 QP 约束并更新参数
    # ==========================================
    if verbose:
        print(f"[2] 执行 QP 对偶约束投影并更新参数 (lr={lr})...")

    # 调用我们上一轮编写的 qp_project_and_update 函数
    g_raw_norm, update_norm, active_constraints = qp_project_and_update(
        model=model,
        g_raw=g_raw,
        J_verified=J,
        lr=lr,
        verbose=verbose
    )

    # 返回值说明:
    # 1. 损失值
    # 2. 最终实际更新的向量范数 (确保步子没有迈得太大)
    # 3. 活跃约束数量 (代替了原来的 k_effective，表示有多少个已验证区域真正受到了威胁)
    return loss.item(), update_norm, active_constraints


def repair_loop(
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
    verbose: bool = False
) -> List[Dict[str, float]]:
    """
    执行多次迭代修复，直到满足终止条件或达到最大迭代次数。

    Returns:
        history: 包含每次迭代损失、梯度等记录的列表
    """
    print(f"\n{'=' * 50}")
    print(f"🚀 开始多轮修复 (Max Iters: {max_iters}, Target Loss: {target_loss})")
    print(f"{'=' * 50}")

    history = []

    for i in range(max_iters):
        if verbose:
            print(f"\n>>> [Iter {i+1}/{max_iters}] 开始执行...")

        # 1. 执行单次修复
        loss, grad_norm, k_effective = repair_iteration(
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
            verbose=verbose
        )

        # 2. 记录当前迭代的状态
        history.append({
            "iter": i + 1,
            "loss": loss,
            "grad_norm": grad_norm,
            "k_effective": k_effective
        })

        if verbose:
            # 详细模式：保留你原本的多行格式，但加上轮次标题以防眼花
            print(f"    [Iter {i+1}/{max_iters}] 结果:")
            print(f"      损失: {loss:.4f}")
            print(f"      梯度范数: {grad_norm:.4f}")
            print(f"      实际 rank: {k_effective}")
            print("-" * 30) # 加一条分割线让视觉更清晰
        else:
            # 精简模式：浓缩在一行，适合监控长时间运行的训练
            print(f"  Iter {i+1:02d}/{max_iters} | Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f} | Rank: {k_effective}")
        # ========== 3. 判断提前终止条件 ==========
        
        # 条件 A: 损失降到目标值以下（通常为 0，代表所有惩罚项消失，修复成功）
        if loss <= target_loss:
            print(f"\n✅ 修复成功！Loss 已降至 {loss:.8f} (<= {target_loss})，所有失败区域均已满足条件。")
            break

        # 条件 B: 梯度范数过小（网络可能陷入局部最优，或者在正交空间内无法继续下降）
        if grad_norm < grad_tol:
            print(f"\n⚠️ 提前停止！梯度范数 ({grad_norm:.8f}) 过小 (< {grad_tol})，继续训练可能无明显收益。")
            break

    print(f"\n🎯 修复循环结束。共执行 {len(history)} 轮。最终 Loss: {history[-1]['loss']:.6f}")
    # return history
    return loss, grad_norm, k_effective



def qp_project_and_update(
    model: nn.Module,
    g_raw: torch.Tensor,
    J_verified: torch.Tensor,
    lr: float = 1e-3,
    verbose: bool = False
) -> Tuple[float, float, float]:
    """
    使用二次规划 (QP) 求解拉格朗日对偶问题，计算满足安全约束的参数更新方向，并更新模型。

    Args:
        model: 神经网络
        g_raw: 修复失败区域产生的原始梯度，形状 [P]
        J_verified: 濒危已验证区域的雅可比矩阵，形状 [N, P]
        lr: 学习率 (由于方向被归一化，lr 将直接作为绝对步长)
        verbose: 是否打印调试信息

    Returns:
        (g_raw_norm, g_update_norm, active_constraints): 
        原始梯度范数，最终更新范数，起作用的安全约束数量(lambda > 0 的个数)
    """
    P = g_raw.shape[0]
    N = J_verified.shape[0] if J_verified is not None else 0

    # 获取原始参数的视图
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    if P != num_params:
        raise ValueError(f"梯度维度不匹配: g_raw {P} vs 参数 {num_params}")

    # ==========================================
    # 1. 极端情况防御：如果没有需要防御的已验证区域
    # ==========================================
    if N == 0:
        if verbose: print("  提示: 没有活动的安全约束，执行标准归一化梯度下降。")
        g_norm = g_raw.norm() + 1e-8
        g_update = g_raw / g_norm  # 只保留方向
        theta_old = torch.nn.utils.parameters_to_vector(params)
        theta_new = theta_old - lr * g_update
        torch.nn.utils.vector_to_parameters(theta_new, params)
        return g_norm.item(), g_update.norm().item(), 0

    # ==========================================
    # 2. 核心：强制 L2 归一化 (消除 LBP 梯度爆炸)
    # ==========================================
    epsilon = 1e-8
    
    # 将原始梯度化为单位向量
    g_raw_norm = g_raw.norm()
    # g_hat = g_raw / (g_raw_norm + epsilon)
    
    # g_raw_norm = g_hat.norm()
    g_hat = g_raw

    # 将雅可比矩阵的**每一行**分别化为单位向量
    J_norms = torch.norm(J_verified, dim=1, keepdim=True)
    J_hat = J_verified / (J_norms + epsilon)

    # ==========================================
    # 3. 构建并求解 QP (在 CPU 上使用 cvxpy)
    # ==========================================
    # 将 Tensor 转为 numpy 供 cvxpy 使用
    J_np = J_hat.detach().cpu().numpy()  # [N, P]
    g_np = g_hat.detach().cpu().numpy()  # [P]

    # 定义未知数 lambda，长度为 N
    lam = cp.Variable(N, nonneg=True)

    # 目标函数: min 0.5 * || J^T * lambda - g_hat ||^2
    # 注意矩阵乘法: J_np.T 是 [P, N], lam 是 [N]
    residual = J_np.T @ lam - g_np
    objective = cp.Minimize(0.5 * cp.sum_squares(residual))
    prob = cp.Problem(objective)

    try:
        # 使用 OSQP 求解器，对于 QP 问题速度极快
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"QP 求解器未能找到最优解，状态: {prob.status}")
            
    except Exception as e:
        print(f"  警告: QP 求解失败 ({e})，降级为截断原梯度。")
        # 降级方案：如果不幸失败，强行砍掉原梯度大小，防止爆炸
        lam_value = np.zeros(N)
    else:
        lam_value = lam.value

    
    # 将求出的 lambda 转换回 GPU Tensor
    lam_star = torch.tensor(lam_value, dtype=g_raw.dtype, device=g_raw.device)

    # ==========================================
    # 4. 合成最终安全的更新方向
    # ==========================================
    # d = g_hat - J_hat^T * lambda_star
    # d_update 将严格保证 J_hat * d_update <= 0
    g_update = g_hat - (J_hat.T @ lam_star)

    # ==========================================
    # 5. 参数更新
    # ==========================================
    theta_old = torch.nn.utils.parameters_to_vector(params)
    theta_new = theta_old - lr * g_update
    torch.nn.utils.vector_to_parameters(theta_new, params)

    # 统计信息
    print( "lam_value", lam_value) # 打印 lambda 的值以检查哪些约束被激活
    active_constraints = int(np.sum(lam_value > 1e-4))  # 统计起了实际阻挡作用的墙的数量
    update_norm = g_update.norm().item()

    if verbose:
        print(f"  |g_raw| (原始大小): {g_raw_norm.item():.2e} (已被归一化抛弃)")
        print(f"  |g_update| (最终方向长度): {update_norm:.4f}")
        print(f"  活跃安全约束数量: {active_constraints} / {N}")
        print(f"  |theta_new|: {theta_new.norm().item():.6f}")

    return g_raw_norm.item(), update_norm, active_constraints



def inner_loop_repair(
    model: nn.Module,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    num_inner_steps: int = 10,
    batch_ratio: float = 0.2,
    lr: float = 1e-3,
    tolerance: float = -1e-12,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
    seed: int = None,
    # ---- 新增: 已验证区域和损失权重 ----
    V_safe: List[Union[torch.Tensor, np.ndarray]] = None,
    V_unsafe: List[Union[torch.Tensor, np.ndarray]] = None,
    lambda_penalty: float = 1.0,
    lambda_stability: float = 0.1,
    lambda_barrier: float = 0.1,
    gamma_safe: float = 0.1,
    gamma_unsafe: float = 0.1,
    verified_batch_ratio: float = 1.0,
) -> List[Dict[str, float]]:
    """
    内循环 Mini-batch 修复策略（组合损失函数版本）。

    总损失函数：
        L_total = λ1 * L_penalty + λ2 * L_stability + λ3 * L_barrier

    1. L_penalty: 处理验证失败区域（原有 Hinge Loss，防止违规）
       - 障碍区违规: clamp(h_lb, min=0)^2
       - 安全区违规: clamp(tolerance - min_L, min=0)^2

    2. L_stability: 针对已验证安全区 V_safe（正向引导 Lie 导数）
       L_stability = mean[ max(0, γ_safe - min_L)^2 ]
       迫使 min_L > γ_safe，防止退化成平坦函数

    3. L_barrier: 针对已验证障碍区 V_unsafe（强化屏障远离边界）
       L_barrier = mean[ max(0, h_ub + γ_unsafe)^2 ]
       迫使 h(x) < -γ_unsafe，远离安全边界

    Args:
        V_safe: 已验证安全区（用于 L_stability）
        V_unsafe: 已验证障碍区（用于 L_barrier）
        lambda_penalty: 惩罚项权重
        lambda_stability: 稳定性项权重
        lambda_barrier: 屏障强化项权重
        gamma_safe: 安全区 Lie 导数裕度（要求 min_L > γ_safe）
        gamma_unsafe: 障碍区屏障裕度（要求 h_ub < -γ_unsafe）
        verified_batch_ratio: 已验证区域的采样比例
    """
    import random as random_module

    # 设置随机种子
    if seed is not None:
        random_module.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 1. 准备失败区域 ----------
    F_all = (
        list(F_h_positive_in_unsafe) +
        list(F_safe_cbf_violation) +
        list(F_depth_limit_reached) +
        list(F_unsafe_cannot_split)
    )
    total_failures = len(F_all)
    n_h = len(F_h_positive_in_unsafe)
    n_safe = len(F_safe_cbf_violation)
    n_depth = len(F_depth_limit_reached)

    V_safe_list = list(V_safe) if V_safe is not None else []
    V_unsafe_list = list(V_unsafe) if V_unsafe is not None else []

    if total_failures == 0 and not V_safe_list and not V_unsafe_list:
        print("  [内循环] 没有失败区域也没有已验证区域，跳过。")
        return []

    batch_size = max(1, int(total_failures * batch_ratio))
    print(f"  [内循环] 失败区域: {total_failures}(batch={batch_size}), "
          f"V_safe={len(V_safe_list)}, V_unsafe={len(V_unsafe_list)}, "
          f"内迭代={num_inner_steps}")
    print(f"  [内循环] 权重: λ_penalty={lambda_penalty}, λ_stab={lambda_stability}, "
          f"λ_barr={lambda_barrier}, γ_safe={gamma_safe}, γ_unsafe={gamma_unsafe}")

    from New_repair.geometry_module_new import compute_simplex_bound_batch

    inner_history = []

    # ---------- 2. 内循环迭代 ----------
    for inner_step in range(num_inner_steps):
        all_terms = []  # (weight, tensor, name) 三元组

        # ---- 2.1 L_penalty: 失败区域（批量计算）----
        if total_failures > 0:
            if batch_ratio < 1.0 and total_failures > batch_size:
                indices = random_module.sample(range(total_failures), batch_size)
            else:
                indices = list(range(total_failures))

            # 按原始索引分离各类型
            idx_h = [i for i in indices if i < n_h]
            idx_safe = [i - n_h for i in indices if n_h <= i < n_h + n_safe]
            idx_depth = [i - n_h - n_safe for i in indices
                         if n_h + n_safe <= i < n_h + n_safe + n_depth]
            idx_unsafe = [i - n_h - n_safe - n_depth for i in indices
                          if i >= n_h + n_safe + n_depth]

            F_bh = [F_h_positive_in_unsafe[i] for i in idx_h] if idx_h else []
            F_bs = [F_safe_cbf_violation[i] for i in idx_safe] if idx_safe else []
            F_bd = [F_depth_limit_reached[i] for i in idx_depth] if idx_depth else []
            F_bu = [F_unsafe_cannot_split[i] for i in idx_unsafe] if idx_unsafe else []

            # F_h_positive_in_unsafe: clamp(h_lb, 0)^2，批量计算
            if F_bh:
                h_lb_all, _ = compute_simplex_bound_batch(
                    model, F_bh, 'unsafe', dynamics_model=None, translator=translator)
                penalty_terms = torch.clamp(h_lb_all, min=0.0) ** 2
                penalty_terms = penalty_terms[torch.isfinite(penalty_terms) & (penalty_terms > 0)]
                if penalty_terms.numel() > 0:
                    L_penalty = penalty_terms.mean()
                    all_terms.append((lambda_penalty, L_penalty, 'L_penalty'))

            # F_safe_cbf_violation + F_depth_limit_reached + F_unsafe_cannot_split: 批量计算
            F_safe_batch = F_bs + F_bd + F_bu
            if F_safe_batch:
                min_L_all = compute_simplex_bound_batch(
                    model, F_safe_batch, 'safe', dynamics_model=dynamics_model, translator=translator)
                safe_terms = torch.clamp(tolerance - min_L_all, min=0.0) ** 2
                safe_terms = safe_terms[torch.isfinite(safe_terms) & (safe_terms > 0)]
                if safe_terms.numel() > 0:
                    L_safe_penalty = safe_terms.mean()
                    # 合并到 L_penalty 或单独添加（这里合并保持原有结构）
                    if any(name == 'L_penalty' for _, _, name in all_terms):
                        # 与已有的 L_penalty 合并
                        for i, (w, t, n) in enumerate(all_terms):
                            if n == 'L_penalty':
                                all_terms[i] = (w, t + L_safe_penalty, n)
                                break
                    else:
                        all_terms.append((lambda_penalty, L_safe_penalty, 'L_penalty'))

        # ---- 2.2 L_stability: 已验证安全区 V_safe（批量计算）----
        if V_safe_list:
            v_batch = max(1, int(len(V_safe_list) * verified_batch_ratio))
            v_idx = random_module.sample(range(len(V_safe_list)), min(v_batch, len(V_safe_list)))
            V_batch = [V_safe_list[i] for i in v_idx]
            min_L_all = compute_simplex_bound_batch(
                model, V_batch, 'safe', dynamics_model=dynamics_model, translator=translator)
            stability_terms = torch.clamp(gamma_safe - min_L_all, min=0.0) ** 2
            stability_terms = stability_terms[torch.isfinite(stability_terms)]
            if stability_terms.numel() > 0:
                L_stability = stability_terms.mean()
                all_terms.append((lambda_stability, L_stability, 'L_stability'))

        # ---- 2.3 L_barrier: 已验证障碍区 V_unsafe（批量计算）----
        if V_unsafe_list:
            v_batch = max(1, int(len(V_unsafe_list) * verified_batch_ratio))
            v_idx = random_module.sample(range(len(V_unsafe_list)), min(v_batch, len(V_unsafe_list)))
            V_batch_unsafe = [V_unsafe_list[i] for i in v_idx]
            _, h_ub_all = compute_simplex_bound_batch(
                model, V_batch_unsafe, 'unsafe', dynamics_model=None, translator=translator)
            # 目标: h_ub < -γ_unsafe → max(0, h_ub + γ_unsafe)^2
            barrier_terms = torch.clamp(h_ub_all + gamma_unsafe, min=0.0) ** 2
            barrier_terms = barrier_terms[torch.isfinite(barrier_terms)]
            if barrier_terms.numel() > 0:
                L_barrier = barrier_terms.mean()
                all_terms.append((lambda_barrier, L_barrier, 'L_barrier'))

        # 无有效项则跳过
        if len(all_terms) == 0:
            if verbose:
                print(f"    [内步 {inner_step+1}] 无有效 loss，跳过。")
            inner_history.append({
                'step': inner_step + 1,
                'loss': 0.0,
                'L_penalty': 0.0,
                'L_stability': 0.0,
                'L_barrier': 0.0,
                'grad_norm': 0.0,
                'update_norm': 0.0,
            })
            continue

        # ---- 2.4 合并损失并反向传播 ----
        total_loss = sum(w * t for (w, t, _) in all_terms)
        model.zero_grad()
        total_loss.backward()

        # 提取梯度并裁剪
        g_raw = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        g_raw = torch.nan_to_num(g_raw, nan=0.0, posinf=0.0, neginf=0.0)

        grad_norm = g_raw.norm().item()
        if grad_norm > grad_clip_norm:
            g_clipped = g_raw * (grad_clip_norm / grad_norm)
        else:
            g_clipped = g_raw

        # 参数更新
        params = [p for p in model.parameters() if p.requires_grad]
        theta_old = torch.nn.utils.parameters_to_vector(params)
        theta_new = theta_old - lr * g_clipped
        torch.nn.utils.vector_to_parameters(theta_new, params)
        update_norm = (theta_new - theta_old).norm().item()

        # ---- 2.5 记录 ----
        loss_vals = {name: t.detach().item() for (_, t, name) in all_terms}
        if verbose:
            loss_str = ", ".join(f"{n}={v:.6f}" for n, v in loss_vals.items())
            print(f"    [内步 {inner_step+1}/{num_inner_steps}] "
                  f"total={total_loss.item():.6f} ({loss_str}), "
                  f"|g|={grad_norm:.4f}, |Δθ|={update_norm:.6f}")

        inner_history.append({
            'step': inner_step + 1,
            'loss': total_loss.item(),
            'L_penalty': loss_vals.get('L_penalty', 0.0),
            'L_stability': loss_vals.get('L_stability', 0.0),
            'L_barrier': loss_vals.get('L_barrier', 0.0),
            'grad_norm': grad_norm,
            'update_norm': update_norm,
        })

    return inner_history

















# def inner_loop_repair(
#     model: nn.Module,
#     F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
#     F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
#     F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
#     F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
#     dynamics_model,
#     translator,
#     num_inner_steps: int = 10,
#     batch_ratio: float = 0.2,
#     lr: float = 1e-3,
#     tolerance: float = -1e-12,
#     grad_clip_norm: float = 10.0,
#     verbose: bool = False,
#     seed: int = None,
# ) -> List[Dict[str, float]]:


#     import random as random_module

#     # 设置随机种子
#     if seed is not None:
#         random_module.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#     device = next(model.parameters()).device
#     dtype = next(model.parameters()).dtype

#     # ---------- 1. 合并所有失败区域 ----------
#     F_all = (
#         list(F_h_positive_in_unsafe) +
#         list(F_safe_cbf_violation) +
#         list(F_depth_limit_reached) +
#         list(F_unsafe_cannot_split)
#     )
#     total_failures = len(F_all)

#     if total_failures == 0:
#         print("  [内循环] 没有需要修复的失败区域，跳过。")
#         return []

#     num_params = sum(p.numel() for p in model.parameters())
#     batch_size = max(1, int(total_failures * batch_ratio))

#     print(f"  [内循环] 总失败区域: {total_failures}, 批次大小: {batch_size} ({batch_ratio*100:.0f}%), 内迭代: {num_inner_steps}")

#     # 延迟导入 geometry_module
#     from New_repair.geometry_module_new import compute_simplex_bound

#     inner_history = []

#     # ---------- 2. 内循环迭代 ----------
#     for inner_step in range(num_inner_steps):
#         # 2.1 Mini-batch 随机采样
#         if batch_ratio < 1.0 and total_failures > batch_size:
#             indices = random_module.sample(range(total_failures), batch_size)
#             F_batch = [F_all[i] for i in indices]
#         else:
#             F_batch = F_all
#             indices = list(range(total_failures))

#         # 分离 batch 中的各类型
#         n_h = len(F_h_positive_in_unsafe)
#         n_safe = len(F_safe_cbf_violation)
#         n_depth = len(F_depth_limit_reached)

#         batch_h = [v for v in F_batch if any(np.allclose(v, h) for h in F_h_positive_in_unsafe)] if F_h_positive_in_unsafe else []
#         batch_safe = [v for v in F_batch if any(np.allclose(v, s) for s in F_safe_cbf_violation)] if F_safe_cbf_violation else []
#         batch_depth = [v for v in F_batch if any(np.allclose(v, d) for d in F_depth_limit_reached)] if F_depth_limit_reached else []
#         batch_unsafe_split = [v for v in F_batch if v not in batch_h and v not in batch_safe and v not in batch_depth] if F_unsafe_cannot_split else []

#         # 简单方式：直接按原始索引划分
#         idx_h = [i for i in indices if i < n_h]
#         idx_safe = [i - n_h for i in indices if n_h <= i < n_h + n_safe]
#         idx_depth = [i - n_h - n_safe for i in indices if n_h + n_safe <= i < n_h + n_safe + n_depth]
#         idx_unsafe = [i - n_h - n_safe - n_depth for i in indices if i >= n_h + n_safe + n_depth]

#         F_batch_h = [F_h_positive_in_unsafe[i] for i in idx_h] if idx_h else []
#         F_batch_safe = [F_safe_cbf_violation[i] for i in idx_safe] if idx_safe else []
#         F_batch_depth = [F_depth_limit_reached[i] for i in idx_depth] if idx_depth else []
#         F_batch_unsafe = [F_unsafe_cannot_split[i] for i in idx_unsafe] if idx_unsafe else []

#         actual_batch_size = len(F_batch_h) + len(F_batch_safe) + len(F_batch_depth) + len(F_batch_unsafe)

#         if actual_batch_size == 0:
#             if verbose:
#                 print(f"    [内步 {inner_step+1}] 批次为空，跳过。")
#             continue

#         # 2.2 前向计算，收集有效 loss 项
#         _valid_terms = []

#         # F_h_positive_in_unsafe
#         for vertices in F_batch_h:
#             h_lb, h_ub = compute_simplex_bound(
#                 model, vertices, 'unsafe',
#                 dynamics_model=None, translator=translator
#             )
#             loss_term = torch.clamp(h_lb - 0.0, min=0.0)
#             if loss_term > 0 and not (torch.isnan(loss_term).any() or torch.isnan(h_ub).any()):
#                 _valid_terms.append(loss_term)

#         # F_safe_cbf_violation
#         for vertices in F_batch_safe:
#             min_L = compute_simplex_bound(
#                 model, vertices, 'safe',
#                 dynamics_model=dynamics_model, translator=translator
#             )
#             loss_term = torch.clamp(tolerance - min_L, min=0.0)
#             if loss_term > 0 and not torch.isnan(loss_term).any():
#                 _valid_terms.append(loss_term)

#         # F_depth_limit_reached
#         for vertices in F_batch_depth:
#             min_L = compute_simplex_bound(
#                 model, vertices, 'safe',
#                 dynamics_model=dynamics_model, translator=translator
#             )
#             loss_term = torch.clamp(tolerance - min_L, min=0.0)
#             if loss_term > 0 and not torch.isnan(loss_term).any():
#                 _valid_terms.append(loss_term)

#         # F_unsafe_cannot_split
#         for vertices in F_batch_unsafe:
#             min_L = compute_simplex_bound(
#                 model, vertices, 'safe',
#                 dynamics_model=dynamics_model, translator=translator
#             )
#             loss_term = torch.clamp(tolerance - min_L, min=0.0)
#             if loss_term > 0 and not torch.isnan(loss_term).any():
#                 _valid_terms.append(loss_term)

#         # 无有效项则跳过
#         if len(_valid_terms) == 0:
#             if verbose:
#                 print(f"    [内步 {inner_step+1}] 无有效 loss 项 (batch_size={actual_batch_size})，跳过。")
#             inner_history.append({
#                 'step': inner_step + 1,
#                 'loss': 0.0,
#                 'batch_size': actual_batch_size,
#                 'grad_norm': 0.0,
#                 'update_norm': 0.0,
#             })
#             continue

#         # 2.3 反向传播
#         total_loss = sum(_valid_terms)
#         model.zero_grad()
#         total_loss.backward()

#         # 2.4 提取梯度并裁剪
#         g_raw = torch.cat([
#             p.grad.flatten() if p.grad is not None
#             else torch.zeros(p.numel(), dtype=dtype, device=device)
#             for p in model.parameters()
#         ])
#         g_raw = torch.nan_to_num(g_raw, nan=0.0)

#         grad_norm = g_raw.norm().item()
#         if grad_norm > grad_clip_norm:
#             g_clipped = g_raw * (grad_clip_norm / grad_norm)
#             if verbose:
#                 print(f"    [内步 {inner_step+1}] 梯度裁剪: {grad_norm:.4f} -> {grad_clip_norm:.4f}")
#         else:
#             g_clipped = g_raw

#         # 2.5 参数更新: theta = theta - lr * g_clipped
#         params = [p for p in model.parameters() if p.requires_grad]
#         theta_old = torch.nn.utils.parameters_to_vector(params)
#         theta_new = theta_old - lr * g_clipped
#         torch.nn.utils.vector_to_parameters(theta_new, params)

#         update_norm = (theta_new - theta_old).norm().item()

#         # 2.6 记录
#         if verbose:
#             print(f"    [内步 {inner_step+1}/{num_inner_steps}] "
#                   f"loss={total_loss.item():.6f}, batch={actual_batch_size}, "
#                   f"|g|={grad_norm:.6f}, |update|={update_norm:.6f}")

#         inner_history.append({
#             'step': inner_step + 1,
#             'loss': total_loss.item(),
#             'batch_size': actual_batch_size,
#             'grad_norm': grad_norm,
#             'update_norm': update_norm,
#         })

#     return inner_history


def inner_loop_repair_with_qp(
    model: nn.Module,
    J: torch.Tensor,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    num_inner_steps: int = 10,
    batch_ratio: float = 0.2,
    lr: float = 1e-3,
    tolerance: float = -1e-12,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
    seed: int = None,
    # ---- 已验证区域和组合损失权重 ----
    V_safe: List[Union[torch.Tensor, np.ndarray]] = None,
    V_unsafe: List[Union[torch.Tensor, np.ndarray]] = None,
    lambda_penalty: float = 1.0,
    lambda_stability: float = 0.1,
    lambda_barrier: float = 0.1,
    gamma_safe: float = 0.1,
    gamma_unsafe: float = 0.1,
    verified_batch_ratio: float = 0.1,
) -> List[Dict[str, float]]:
    """
    内循环 Mini-batch 修复策略（QP 投影版本）。

    外层（main_v1.py）负责计算 J 矩阵并传入，本函数固定 J 进行多次内循环梯度更新。
    总损失函数：
        L_total = λ1 * L_penalty + λ2 * L_stability + λ3 * L_barrier

    1. L_penalty: 处理验证失败区域（原有 Hinge Loss，防止违规）
       - 障碍区违规: clamp(h_lb, min=0)^2
       - 安全区违规: clamp(tolerance - min_L, min=0)^2

    2. L_stability: 针对已验证安全区 V_safe（正向引导 Lie 导数）
       L_stability = mean[ max(0, γ_safe - min_L)^2 ]
       迫使 min_L > γ_safe，防止退化成平坦函数

    3. L_barrier: 针对已验证障碍区 V_unsafe（强化屏障远离边界）
       L_barrier = mean[ max(0, h_ub + γ_unsafe)^2 ]
       迫使 h(x) < -γ_unsafe，远离安全边界

    梯度更新使用 QP 投影思想（qp_project_and_update）：
        - 将原始梯度 g_raw 归一化为单位向量 g_hat
        - 将 J 的每行归一化为单位向量 J_hat
        - 求解 QP: min 0.5 * || J^T * λ - g_hat ||^2,  s.t. λ >= 0
        - 最终更新方向: d = g_hat - J_hat^T * λ_star
        - 参数更新: θ_new = θ_old - lr * d

    Args:
        J: 外层传入的雅可比矩阵，形状 [N, P]（外层迭代中 J 保持不变）
        V_safe: 已验证安全区（用于 L_stability）
        V_unsafe: 已验证障碍区（用于 L_barrier）
        lambda_penalty: 惩罚项权重
        lambda_stability: 稳定性项权重
        lambda_barrier: 屏障强化项权重
        gamma_safe: 安全区 Lie 导数裕度（要求 min_L > γ_safe）
        gamma_unsafe: 障碍区屏障裕度（要求 h_ub < -γ_unsafe）
        verified_batch_ratio: 已验证区域的采样比例
    """
    import random as random_module

    # 设置随机种子
    if seed is not None:
        random_module.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 1. 准备区域列表 ----------
    F_all = (
        list(F_h_positive_in_unsafe) +
        list(F_safe_cbf_violation) +
        list(F_depth_limit_reached) +
        list(F_unsafe_cannot_split)
    )
    total_failures = len(F_all)
    n_h = len(F_h_positive_in_unsafe)
    n_safe = len(F_safe_cbf_violation)
    n_depth = len(F_depth_limit_reached)

    V_safe_list = list(V_safe) if V_safe is not None else []
    V_unsafe_list = list(V_unsafe) if V_unsafe is not None else []

    # J 矩阵信息
    J_rows = J.shape[0]
    num_params = J.shape[1]

    if total_failures == 0 and not V_safe_list and not V_unsafe_list:
        print("  [内循环QP] 没有失败区域也没有已验证区域，跳过。")
        return []

    batch_size = max(1, int(total_failures * batch_ratio))
    print(f"  [内循环QP] J.shape={J.shape}, 失败区域: {total_failures}(batch={batch_size}), "
          f"V_safe={len(V_safe_list)}, V_unsafe={len(V_unsafe_list)}, "
          f"内迭代={num_inner_steps}")
    print(f"  [内循环QP] 权重: λ_penalty={lambda_penalty}, λ_stab={lambda_stability}, "
          f"λ_barr={lambda_barrier}, γ_safe={gamma_safe}, γ_unsafe={gamma_unsafe}")

    from New_repair.geometry_module_new import compute_simplex_bound_batch

    # ---------- 2. QP 投影预计算（J 归一化） ----------
    epsilon = 1e-8
    J_norms = torch.norm(J, dim=1, keepdim=True)  # [N, 1]
    J_hat = J / (J_norms + epsilon)  # [N, P], 每行单位化

    inner_history = []

    # ---------- 3. 内循环迭代（固定 J） ----------
    for inner_step in range(num_inner_steps):
        all_terms = []  # (weight, tensor, name) 三元组

        # ---- 3.1 L_penalty: 失败区域（批量计算）----
        if total_failures > 0:
            if batch_ratio < 1.0 and total_failures > batch_size:
                indices = random_module.sample(range(total_failures), batch_size)
            else:
                indices = list(range(total_failures))

            # 按原始索引分离各类型
            idx_h = [i for i in indices if i < n_h]
            idx_safe = [i - n_h for i in indices if n_h <= i < n_h + n_safe]
            idx_depth = [i - n_h - n_safe for i in indices
                         if n_h + n_safe <= i < n_h + n_safe + n_depth]
            idx_unsafe = [i - n_h - n_safe - n_depth for i in indices
                          if i >= n_h + n_safe + n_depth]

            F_bh = [F_h_positive_in_unsafe[i] for i in idx_h] if idx_h else []
            F_bs = [F_safe_cbf_violation[i] for i in idx_safe] if idx_safe else []
            F_bd = [F_depth_limit_reached[i] for i in idx_depth] if idx_depth else []
            F_bu = [F_unsafe_cannot_split[i] for i in idx_unsafe] if idx_unsafe else []

            # F_h_positive_in_unsafe: clamp(h_lb, 0)^2
            if F_bh:
                h_lb_all, _ = compute_simplex_bound_batch(
                    model, F_bh, 'unsafe', dynamics_model=None, translator=translator)
                penalty_terms = torch.clamp(h_lb_all, min=0.0) ** 2
                penalty_terms = penalty_terms[torch.isfinite(penalty_terms) & (penalty_terms > 0)]
                if penalty_terms.numel() > 0:
                    L_penalty_h = penalty_terms.mean()
                    all_terms.append((lambda_penalty, L_penalty_h, 'L_penalty_h'))

            # F_safe_cbf_violation + F_depth_limit_reached + F_unsafe_cannot_split: clamp(tol - min_L, 0)^2
            F_safe_batch = F_bs + F_bd + F_bu
            if F_safe_batch:
                min_L_all = compute_simplex_bound_batch(
                    model, F_safe_batch, 'safe', dynamics_model=dynamics_model, translator=translator)
                safe_terms = torch.clamp(tolerance - min_L_all, min=0.0) ** 2
                safe_terms = safe_terms[torch.isfinite(safe_terms) & (safe_terms > 0)]
                if safe_terms.numel() > 0:
                    L_safe_penalty = safe_terms.mean()
                    all_terms.append((lambda_penalty, L_safe_penalty, 'L_penalty_safe'))

        # ---- 3.2 L_stability: 已验证安全区 V_safe ----
        if V_safe_list:
            v_batch = max(1, int(len(V_safe_list) * verified_batch_ratio))
            v_idx = random_module.sample(range(len(V_safe_list)), min(v_batch, len(V_safe_list)))
            V_batch = [V_safe_list[i] for i in v_idx]
            min_L_all = compute_simplex_bound_batch(
                model, V_batch, 'safe', dynamics_model=dynamics_model, translator=translator)
            stability_terms = torch.clamp(gamma_safe - min_L_all, min=0.0) ** 2
            stability_terms = stability_terms[torch.isfinite(stability_terms)]
            if stability_terms.numel() > 0:
                L_stability = stability_terms.mean()
                all_terms.append((lambda_stability, L_stability, 'L_stability'))

        # ---- 3.3 L_barrier: 已验证障碍区 V_unsafe ----
        if V_unsafe_list:
            v_batch = max(1, int(len(V_unsafe_list) * verified_batch_ratio))
            v_idx = random_module.sample(range(len(V_unsafe_list)), min(v_batch, len(V_unsafe_list)))
            V_batch_unsafe = [V_unsafe_list[i] for i in v_idx]
            _, h_ub_all = compute_simplex_bound_batch(
                model, V_batch_unsafe, 'unsafe', dynamics_model=None, translator=translator)
            # 目标: h_ub < -γ_unsafe → max(0, h_ub + γ_unsafe)^2
            barrier_terms = torch.clamp(h_ub_all + gamma_unsafe, min=0.0) ** 2
            barrier_terms = barrier_terms[torch.isfinite(barrier_terms)]
            if barrier_terms.numel() > 0:
                L_barrier = barrier_terms.mean()
                all_terms.append((lambda_barrier, L_barrier, 'L_barrier'))

        # 无有效项则跳过
        if len(all_terms) == 0:
            if verbose:
                print(f"    [内步 {inner_step+1}] 无有效 loss，跳过。")
            inner_history.append({
                'step': inner_step + 1,
                'loss': 0.0,
                'L_penalty': 0.0,
                'L_stability': 0.0,
                'L_barrier': 0.0,
                'grad_norm': 0.0,
                'update_norm': 0.0,
                'active_constraints': 0,
            })
            continue

        # ---- 3.4 合并损失并反向传播 ----
        total_loss = sum(w * t for (w, t, _) in all_terms)
        model.zero_grad()
        total_loss.backward()

        # 提取原始梯度
        g_raw = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        g_raw = torch.nan_to_num(g_raw, nan=0.0, posinf=0.0, neginf=0.0)

        grad_norm = g_raw.norm().item()
        if grad_norm < 1e-12:
            if verbose:
                print(f"    [内步 {inner_step+1}] 梯度过小，跳过更新。")
            loss_vals = {}
            for (_, t, name) in all_terms:
                v = t.detach().item()
                if name.startswith('L_penalty'):
                    loss_vals['L_penalty'] = loss_vals.get('L_penalty', 0.0) + v
                else:
                    loss_vals[name] = v
            inner_history.append({
                'step': inner_step + 1,
                'loss': total_loss.item(),
                'L_penalty': loss_vals.get('L_penalty', 0.0),
                'L_stability': loss_vals.get('L_stability', 0.0),
                'L_barrier': loss_vals.get('L_barrier', 0.0),
                'grad_norm': grad_norm,
                'update_norm': 0.0,
                'active_constraints': 0,
            })
            continue

        # ---- 3.5 QP 投影（复用 qp_project_and_update 的核心思想）----
        # 梯度归一化
        g_hat = g_raw / (grad_norm + epsilon)  # 单位向量

        # 构建 QP: min 0.5 * || J_hat^T * λ - g_hat ||^2,  s.t. λ >= 0
        J_np = J_hat.detach().cpu().numpy()   # [N, P]
        g_np = g_hat.detach().cpu().numpy()  # [P]

        N_j = J_np.shape[0]
        lam = cp.Variable(N_j, nonneg=True)
        residual = J_np.T @ lam - g_np
        objective = cp.Minimize(0.5 * cp.sum_squares(residual))
        prob = cp.Problem(objective)

        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"QP 求解器状态: {prob.status}")
            lam_value = lam.value
        except Exception as e:
            if verbose:
                print(f"    [内步 {inner_step+1}] QP 求解失败 ({e})，降级为梯度裁剪。")
            lam_value = np.zeros(N_j)

        lam_star = torch.tensor(lam_value, dtype=dtype, device=device)
        active_constraints = int(np.sum(lam_value > 1e-4))

        # 安全的更新方向: d = g_hat - J_hat^T * λ_star
        g_update = g_hat - (J_hat.T @ lam_star)

        # 可选梯度裁剪（作为安全上限）
        update_norm_raw = g_update.norm().item()
        if update_norm_raw > grad_clip_norm:
            g_update = g_update * (grad_clip_norm / update_norm_raw)

        # ---- 3.6 参数更新: θ_new = θ_old - lr * g_update ----
        params = [p for p in model.parameters() if p.requires_grad]
        theta_old = torch.nn.utils.parameters_to_vector(params)
        theta_new = theta_old - lr * g_update
        torch.nn.utils.vector_to_parameters(theta_new, params)
        update_norm = (theta_new - theta_old).norm().item()

        # ---- 3.7 记录 ----
        loss_vals = {}
        for (_, t, name) in all_terms:
            v = t.detach().item()
            if name.startswith('L_penalty'):
                loss_vals['L_penalty'] = loss_vals.get('L_penalty', 0.0) + v
            else:
                loss_vals[name] = v

        if verbose:
            loss_str = ", ".join(f"{n}={v:.6f}" for n, v in loss_vals.items())
            print(f"    [内步 {inner_step+1}/{num_inner_steps}] "
                  f"total={total_loss.item():.6f} ({loss_str}), "
                  f"|g|={grad_norm:.4f}, |d|={update_norm:.6f}, "
                  f"active={active_constraints}/{N_j}")

        inner_history.append({
            'step': inner_step + 1,
            'loss': total_loss.item(),
            'L_penalty': loss_vals.get('L_penalty', 0.0),
            'L_stability': loss_vals.get('L_stability', 0.0),
            'L_barrier': loss_vals.get('L_barrier', 0.0),
            'grad_norm': grad_norm,
            'update_norm': update_norm,
            'active_constraints': active_constraints,
        })

    return inner_history
