"""
优化器模块 v5: RS 损失梯度计算

核心改进（相比 v3/v4）：
    - 损失梯度使用随机平滑（RS）方法，替代 autograd.backward()
    - 与 RS Jacobian 完全对齐：都对 Crown 下界求梯度
    - 不使用任何 autograd.backward()，完全通过前向传播 + RS 公式估计梯度

RS 损失梯度公式：
    g_s = (1 / (N · σ²)) · Σ_{i=1}^{N} [ ψ_s(θ + εᵢ) · εᵢ ]
    其中 εᵢ ~ N(0, σ² I)

ψ_s(θ) 的定义：
    - safe 类型（CBF 违规）: ψ_s = softplus(cbf_margin - min_L)
    - unsafe 类型（h 违规）: ψ_s = softplus(h_lb + margin)

最终 g_raw = mean(g_s) over 所有 loss > 0 的 simplices。
"""

from typing import List, Tuple, Union, Dict, Optional

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from New_repair.geometry_module_new import (
    compute_simplex_bound,
    compute_simplex_bound_batch,
)


def compute_repair_loss_and_grad_rs(
    model: nn.Module,
    failed_safe_simplices: List,
    failed_unsafe_simplices: List,
    dynamics_model,
    translator,
    tolerance: float = -1e-12,
    margin: float = 0.0,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    rs_n: int = 100,
    rs_sigma: float = 0.01,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    """
    用随机平滑（RS）方法计算 failure 区域的损失梯度。

    高效实现：预采样 N 个 epsilon，批量处理所有单纯形。
    循环 rs_n 次（参数扰动次数），而非 n_failed 次（单纯形个数）。

    流程：
        1. 预采样 rs_n 个 epsilon（所有单纯形共用）
        2. 对每个 epsilon_i：
           - 扰动参数为 theta_old + eps_i
           - 批量计算所有 safe 单纯形的 psi = softplus(cbf_margin - min_L)
           - 批量计算所有 unsafe 单纯形的 psi = softplus(h_lb + margin)
           - 累积梯度贡献
        3. 对所有有效单纯形的 g_s 取平均
        4. 梯度裁剪后返回

    不使用任何 autograd.backward()，完全通过前向传播 + RS 公式估计梯度。

    Args:
        model: BarrierNN 网络
        failed_safe_simplices: CBF 违规区域的单纯形列表
            (F_safe_cbf_violation + F_depth_limit_reached + F_unsafe_cannot_split)
        failed_unsafe_simplices: 障碍区违规的单纯形列表
            (F_h_positive_in_unsafe)
        dynamics_model: 动力学系统
        translator: TorchTranslator
        tolerance: 数值 tolerance（只有 loss > tolerance 的单纯形才计入梯度）
        margin: h 值的容差（h 应该 <= 0）
        cbf_margin: CBF 条件的容差（cbf 应该 >= cbf_margin）
        beta: softplus 的 beta 参数
        rs_n: 随机平滑采样次数 N
        rs_sigma: 随机平滑噪声标准差 sigma
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出

    Returns:
        (total_loss, g_raw): 平均损失值和修复梯度 [num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    # ---------- 1. 分离 safe 和 unsafe 单纯形 ----------
    failed_safe_list = list(failed_safe_simplices) if failed_safe_simplices else []
    failed_unsafe_list = list(failed_unsafe_simplices) if failed_unsafe_simplices else []

    n_safe = len(failed_safe_list)
    n_unsafe = len(failed_unsafe_list)
    n_failed = n_safe + n_unsafe

    if n_failed == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    if verbose:
        print(f"  [RS Loss] n_failed={n_failed} (safe={n_safe}, unsafe={n_unsafe})")
        print(f"  [RS Loss] rs_n={rs_n}, rs_sigma={rs_sigma}, "
              f"margin={margin}, cbf_margin={cbf_margin}, beta={beta}")
        print(f"  [RS Loss] 策略: 共享 epsilon 噪声，批量处理所有单纯形")

    # ---------- 2. 预采样 N 个 epsilon（所有单纯形共用）----------
    eps_samples = torch.randn(rs_n, num_params, dtype=dtype, device=device) * rs_sigma

    # 保存原始参数
    original_state = copy.deepcopy(model.state_dict())
    theta_old = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    # ---------- 3. 初始化累积器 ----------
    # accumulator[s] = sum_i[psi_s(theta + eps_i) * eps_i] for each simplex s
    # shape: [n_failed, num_params]
    accumulators = torch.zeros(n_failed, num_params, dtype=dtype, device=device)
    valid_counts = torch.zeros(n_failed, dtype=torch.int32, device=device)  # 每个单纯形的有效 epsilon 计数
    loss_values = torch.zeros(n_failed, dtype=dtype, device=device)  # 每个单纯形的损失值

    # ---------- 4. 对每个 epsilon 执行批量计算 ----------
    for eps_idx in range(rs_n):
        eps_i = eps_samples[eps_idx]  # [num_params]

        # 4a. 扰动参数: theta_i = theta_old + eps_i
        theta_i = theta_old + eps_i
        torch.nn.utils.vector_to_parameters(theta_i, model.parameters())

        # 4b. 批量计算所有 safe 单纯形的 psi
        if n_safe > 0:
            with torch.no_grad():
                min_L_all = compute_simplex_bound_batch(
                    model, failed_safe_list, 'safe',
                    dynamics_model=dynamics_model, translator=translator
                )  # [n_safe]
                psi_safe = F.softplus(cbf_margin - min_L_all, beta=beta)  # [n_safe]
                valid_mask_safe = torch.isfinite(psi_safe) & ~torch.isnan(psi_safe)
                loss_values[:n_safe] = psi_safe.detach()
                # 累积梯度贡献: accumulators[s] += psi_s * eps_i
                for s_idx in range(n_safe):
                    if valid_mask_safe[s_idx]:
                        accumulators[s_idx].add_(eps_i * psi_safe[s_idx].detach())
                        valid_counts[s_idx] += 1

        # 4c. 批量计算所有 unsafe 单纯形的 psi
        if n_unsafe > 0:
            with torch.no_grad():
                h_lb_all, _ = compute_simplex_bound_batch(
                    model, failed_unsafe_list, 'unsafe',
                    dynamics_model=None, translator=None
                )  # [n_unsafe]
                psi_unsafe = F.softplus(h_lb_all + margin, beta=beta)  # [n_unsafe]
                valid_mask_unsafe = torch.isfinite(psi_unsafe) & ~torch.isnan(psi_unsafe)
                loss_values[n_safe:] = psi_unsafe.detach()
                # 累积梯度贡献: accumulators[s] += psi_s * eps_i
                for s_idx in range(n_unsafe):
                    if valid_mask_unsafe[s_idx]:
                        accumulators[n_safe + s_idx].add_(eps_i * psi_unsafe[s_idx].detach())
                        valid_counts[n_safe + s_idx] += 1

        # 进度打印
        if (eps_idx + 1) % 20 == 0 or eps_idx + 1 == rs_n:
            print(f"    RS epsilon 进度: {eps_idx + 1}/{rs_n}")

    # ---------- 5. 恢复原始参数 ----------
    model.load_state_dict(original_state)

    # ---------- 6. 计算平均梯度 ----------
    # 只计入 loss > tolerance 的单纯形
    sigma_sq = rs_sigma * rs_sigma
    valid_simplex_mask = (loss_values > tolerance) & (valid_counts > 0)
    valid_simplex_indices = valid_simplex_mask.nonzero(as_tuple=True)[0]
    n_valid = len(valid_simplex_indices)

    if n_valid == 0:
        if verbose:
            print(f"  [RS Loss] 没有有效的 failure simplices，返回零梯度")
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    # 每个单纯形的 RS 梯度: g_s = accumulators[s] / (valid_counts[s] * sigma^2)
    # g_raw = mean(g_s) over all valid simplices

    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_loss_sum = 0.0

    for s_idx in valid_simplex_indices:
        vc = valid_counts[s_idx].item()
        if vc > 0:
            g_s = accumulators[s_idx] / (vc * sigma_sq)  # 这个单纯形的 RS 梯度
            g_raw.add_(g_s)
            total_loss_sum += loss_values[s_idx].item()

    g_raw = g_raw / n_valid  # 对所有有效单纯形取平均
    total_loss = total_loss_sum / n_valid

    # ---------- 7. 梯度裁剪 ----------
    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)
        if verbose:
            print(f"  [RS Loss] 梯度裁剪: {grad_norm:.4f} -> {grad_clip_norm:.4f}")

    if verbose:
        print(f"  [RS Loss] total_loss={total_loss:.6f}, |g_raw|={g_raw.norm().item():.4f}, "
              f"valid_simplices={n_valid}/{n_failed}")

    return total_loss, g_raw


def qp_project_and_update(
    model: nn.Module,
    g_raw: torch.Tensor,
    J_verified: torch.Tensor,
    lr: float = 1e-3,
    verbose: bool = False,
) -> Tuple[float, float, int]:
    """
    QP 约束投影与参数更新（与 v3 相同）。

    求解 QP:
        min_λ 1/2 λ^T (J J^T) λ - (J g_hat)^T λ   s.t.  λ >= 0
    其中 g_hat = g_raw / |g_raw| 已归一化。

    Args:
        model: 神经网络
        g_raw: 原始梯度 [num_params]
        J_verified: 已验证区域的雅可比矩阵 [N, num_params]
        lr: 学习率
        verbose: 诊断输出

    Returns:
        (g_raw_norm, update_norm, n_violate_after)
    """
    import cvxpy as cp

    device = g_raw.device
    dtype = g_raw.dtype
    params = [p for p in model.parameters() if p.requires_grad]
    theta_old = torch.nn.utils.parameters_to_vector(params)

    g_norm = g_raw.norm().item()
    g_hat = g_raw / (g_norm + 1e-8)

    J_norms = torch.norm(J_verified, dim=1, keepdim=True)
    J_hat = J_verified / (J_norms + 1e-8)

    # ---- 诊断：更新前的约束违反情况 ----
    Jg_before = J_hat @ g_hat
    n_violate_before = int((Jg_before > 0).sum().item())

    # ---- 核心优化：在 GPU 上计算 Q 和 q ----
    Q_tensor = J_hat @ J_hat.T
    q_tensor = J_hat @ g_hat

    Q_np = Q_tensor.detach().cpu().numpy()
    q_np = q_tensor.detach().cpu().numpy()

    lam = cp.Variable(J_verified.shape[0], nonneg=True)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(lam, Q_np) - q_np.T @ lam))

    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        lam_value = lam.value if lam.value is not None else np.zeros(J_verified.shape[0])
    except Exception as e:
        if verbose:
            print(f"[QP Error] Solver failed: {e}")
        lam_value = np.zeros(J_verified.shape[0])

    lam_star = torch.tensor(lam_value, dtype=dtype, device=device)
    g_update = g_hat - (J_hat.T @ lam_star)
    update_norm = g_update.norm().item()

    # 实施参数更新
    theta_new = theta_old - lr * g_update
    torch.nn.utils.vector_to_parameters(theta_new, params)

    # ---- 诊断：更新后的约束违反情况 ----
    Jd_after = J_hat @ g_update
    n_violate_after = int((Jd_after > 0).sum().item())
    active = int(np.sum(lam_value > 1e-4))

    if verbose:
        print(f"  [QP] |g|={g_norm:.4f}, 初始违反={n_violate_before}/{J_verified.shape[0]}, "
              f"active(λ>1e-4)={active}, "
              f"投影后违反={n_violate_after}, Jd_after_max={Jd_after.max().item():.6f}")

    return g_norm, update_norm, n_violate_after
