"""
修复算法的优化模块 v4：随机平滑梯度估计 + QP 投影

核心改进：
    - compute_repair_loss_and_grad_rs: 用随机平滑替代 autograd 反向传播
    - qp_project_and_update: 保持不变（复用 v1 版本）

随机平滑梯度公式：
    g_i = (1/(N * sigma^2)) * sum_j[psi(x_j) * epsilon_j]
    其中 epsilon_j ~ N(0, sigma^2 I)
"""

from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
import copy


def compute_repair_loss_and_grad_rs(
    model: nn.Module,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    tolerance: float = -1e-12,
    # === 随机平滑参数 ===
    rs_n_samples: int = 50,
    rs_sigma: float = 0.01,
    rs_aggregated: str = "mean",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    用随机平滑版本替代原有的 autograd 梯度计算。

    策略：对 failure 区域的每个单纯形，分别用随机平滑计算其对参数的梯度，
    然后平均得到 g_raw。

    梯度估计公式（对单个单纯形 s）：
        g_s = (1/(N * sigma^2)) * sum_i[psi_s(theta + epsilon_i) * epsilon_i]

    其中 psi_s(theta) 是该单纯形的 CBF 条件下界（标量），epsilon_i ~ N(0, sigma^2 I)。

    最终 g_raw = mean(g_s) over all failure simplices with positive loss.

    注意：这里 psi 的计算仍然需要 compute_simplex_bound() 做前向传播，
    但不需要反向传播。

    Args:
        rs_n_samples: 每个单纯形的采样次数 N
        rs_sigma: 噪声标准差 sigma
        rs_aggregated: "mean" | "sum"，梯度聚合方式

    Returns:
        (total_loss, g_raw): 总损失和随机平滑梯度向量
    """
    from New_repair.geometry_module_new import compute_simplex_bound

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    total_failure = (
        len(F_h_positive_in_unsafe) +
        len(F_safe_cbf_violation) +
        len(F_depth_limit_reached) +
        len(F_unsafe_cannot_split)
    )
    num_params = sum(p.numel() for p in model.parameters())

    if total_failure == 0:
        return (
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.zeros(num_params, dtype=dtype, device=device)
        )

    # ---------- 1. 收集所有 failure 单纯形 ----------
    all_failure_vertices = []
    all_failure_types = []  # 'unsafe_h', 'safe'

    for v in F_h_positive_in_unsafe:
        all_failure_vertices.append(v)
        all_failure_types.append('unsafe_h')
    for v in F_safe_cbf_violation:
        all_failure_vertices.append(v)
        all_failure_types.append('safe')
    for v in F_depth_limit_reached:
        all_failure_vertices.append(v)
        all_failure_types.append('safe')
    for v in F_unsafe_cannot_split:
        all_failure_vertices.append(v)
        all_failure_types.append('safe')

    n_failure = len(all_failure_vertices)
    n_h = len(F_h_positive_in_unsafe)
    n_safe_total = n_failure - n_h

    # ---------- 2. 获取原始参数 ----------
    params = list(model.parameters())
    theta_old = torch.nn.utils.parameters_to_vector(params).detach().clone()

    # ---------- 3. 对每个 failure 单纯形计算 RS 梯度 ----------
    all_grads = []
    all_losses = []

    print(f"  [RS Grad] n_failure={n_failure}, N={rs_n_samples}, sigma={rs_sigma}")

    for idx in range(n_failure):
        vertices = all_failure_vertices[idx]
        ftype = all_failure_types[idx]

        # --- 3a. 计算当前单纯形的 loss 值（判断是否需要包含）---
        with torch.no_grad():
            if ftype == 'unsafe_h':
                h_lb, h_ub = compute_simplex_bound(
                    model, vertices, 'unsafe',
                    dynamics_model=None, translator=None
                )
                loss_val = torch.clamp(h_lb - 0.0, min=0.0).item()
            else:
                min_L = compute_simplex_bound(
                    model, vertices, 'safe',
                    dynamics_model=dynamics_model, translator=translator
                )
                loss_val = torch.clamp(tolerance - min_L, min=0.0).item()

        # 只对 loss > 0 的单纯形计算梯度
        if loss_val <= 0:
            continue

        # --- 3b. 随机平滑梯度估计 ---
        accumulator = torch.zeros(num_params, dtype=dtype, device=device)
        valid_count = 0

        for _ in range(rs_n_samples):
            # 采样 epsilon ~ N(0, sigma^2 I)
            eps_i = torch.randn(num_params, dtype=dtype, device=device) * rs_sigma

            # 扰动参数
            theta_i = theta_old + eps_i
            torch.nn.utils.vector_to_parameters(theta_i, params)

            # 前向传播计算 psi
            with torch.no_grad():
                if ftype == 'unsafe_h':
                    h_lb, h_ub = compute_simplex_bound(
                        model, vertices, 'unsafe',
                        dynamics_model=None, translator=None
                    )
                    psi_val = h_ub.squeeze()
                else:
                    min_L = compute_simplex_bound(
                        model, vertices, 'safe',
                        dynamics_model=dynamics_model, translator=translator
                    )
                    psi_val = min_L.squeeze()

            if torch.isfinite(psi_val) and not torch.isnan(psi_val):
                accumulator.add_(eps_i * psi_val)
                valid_count += 1

        # 恢复原始参数
        torch.nn.utils.vector_to_parameters(theta_old.clone(), params)

        if valid_count > 0:
            g_s = accumulator / (rs_n_samples * rs_sigma * rs_sigma)
        else:
            g_s = torch.zeros(num_params, dtype=dtype, device=device)

        all_grads.append(g_s)
        all_losses.append(loss_val)

        if verbose and idx == 0:
            print(f"    failure[0] loss={loss_val:.6f}, |g_rs|={g_s.norm().item():.6f}")

    # ---------- 4. 聚合梯度 ----------
    n_valid = len(all_grads)
    if n_valid == 0:
        return (
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.zeros(num_params, dtype=dtype, device=device)
        )

    if rs_aggregated == "mean":
        g_raw = torch.stack(all_grads, dim=0).mean(dim=0)
    else:  # sum
        g_raw = torch.stack(all_grads, dim=0).sum(dim=0)

    total_loss = torch.tensor(all_losses, dtype=dtype, device=device).mean()

    if verbose:
        print(f"  [RS Grad] n_valid={n_valid}/{n_failure}, total_loss={total_loss.item():.6f}, |g_raw|={g_raw.norm().item():.6f}")

    return total_loss, g_raw


def compute_repair_loss_and_grad_rs_per_simplex(
    model: nn.Module,
    F_h_positive_in_unsafe: List[Union[torch.Tensor, np.ndarray]],
    F_safe_cbf_violation: List[Union[torch.Tensor, np.ndarray]],
    F_depth_limit_reached: List[Union[torch.Tensor, np.ndarray]],
    F_unsafe_cannot_split: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    tolerance: float = -1e-12,
    rs_n_samples: int = 50,
    rs_sigma: float = 0.01,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    逐单纯形计算 RS 梯度（更细粒度版本，用于调试）。

    与 compute_repair_loss_and_grad_rs 的区别：
    - 每个单纯形的梯度单独反向一次（而不是 batch 后平均）
    - 返回 per-simplex 的梯度信息
    """
    from New_repair.geometry_module_new import compute_simplex_bound

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    total_failure = (
        len(F_h_positive_in_unsafe) +
        len(F_safe_cbf_violation) +
        len(F_depth_limit_reached) +
        len(F_unsafe_cannot_split)
    )
    num_params = sum(p.numel() for p in model.parameters())

    if total_failure == 0:
        return (
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.zeros(num_params, dtype=dtype, device=device)
        )

    all_failure_vertices = []
    all_failure_types = []

    for v in F_h_positive_in_unsafe:
        all_failure_vertices.append(v)
        all_failure_types.append('unsafe_h')
    for v in F_safe_cbf_violation:
        all_failure_vertices.append(v)
        all_failure_types.append('safe')
    for v in F_depth_limit_reached:
        all_failure_vertices.append(v)
        all_failure_types.append('safe')
    for v in F_unsafe_cannot_split:
        all_failure_vertices.append(v)
        all_failure_types.append('safe')

    params = list(model.parameters())
    theta_old = torch.nn.utils.parameters_to_vector(params).detach().clone()

    total_loss_tensor = torch.tensor(0.0, dtype=dtype, device=device)
    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    n_contrib = 0

    for idx in range(len(all_failure_vertices)):
        vertices = all_failure_vertices[idx]
        ftype = all_failure_types[idx]

        # 前向计算 loss
        with torch.no_grad():
            if ftype == 'unsafe_h':
                h_lb, h_ub = compute_simplex_bound(
                    model, vertices, 'unsafe',
                    dynamics_model=None, translator=None
                )
                loss_val = torch.clamp(h_lb - 0.0, min=0.0)
            else:
                min_L = compute_simplex_bound(
                    model, vertices, 'safe',
                    dynamics_model=dynamics_model, translator=translator
                )
                loss_val = torch.clamp(tolerance - min_L, min=0.0)

        if loss_val.item() <= 0:
            continue

        # RS 梯度
        accumulator = torch.zeros(num_params, dtype=dtype, device=device)
        valid_count = 0

        for _ in range(rs_n_samples):
            eps_i = torch.randn(num_params, dtype=dtype, device=device) * rs_sigma
            theta_i = theta_old + eps_i
            torch.nn.utils.vector_to_parameters(theta_i, params)

            with torch.no_grad():
                if ftype == 'unsafe_h':
                    _, h_ub = compute_simplex_bound(
                        model, vertices, 'unsafe',
                        dynamics_model=None, translator=None
                    )
                    psi_val = h_ub.squeeze()
                else:
                    min_L = compute_simplex_bound(
                        model, vertices, 'safe',
                        dynamics_model=dynamics_model, translator=translator
                    )
                    psi_val = min_L.squeeze()

            if torch.isfinite(psi_val) and not torch.isnan(psi_val):
                accumulator.add_(eps_i * psi_val)
                valid_count += 1

        torch.nn.utils.vector_to_parameters(theta_old.clone(), params)

        if valid_count > 0:
            g_s = accumulator / (rs_n_samples * rs_sigma * rs_sigma)
            total_loss_tensor = total_loss_tensor + loss_val
            g_raw.add_(g_s)
            n_contrib += 1

    if n_contrib > 0:
        g_raw.div_(n_contrib)
        total_loss_tensor.div_(n_contrib)

    if verbose:
        print(f"  [RS Per-Simplex] n_contrib={n_contrib}/{len(all_failure_vertices)}, loss={total_loss_tensor.item():.6f}")

    return total_loss_tensor, g_raw


# =============================================================================
# 复用 optimizer_module.py 中的 qp_project_and_update
# =============================================================================
from New_repair.optimizer_module import qp_project_and_update


def inner_loop_repair_with_pgd_rs(
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
    # ---- PGD 参数 ----
    pgd_steps: int = 20,
    pgd_lr: float = 0.1,
    # ---- 随机平滑参数 ----
    rs_n_samples: int = 50,
    rs_sigma: float = 0.01,
) -> List[Dict[str, float]]:
    """
    内循环 Mini-batch 修复策略（随机平滑梯度 + PGD QP 投影版本）。

    与 inner_loop_repair_with_pgd 的区别：
    - 用 compute_repair_loss_and_grad_rs 替代 autograd 反向传播
    - 其他逻辑（QP 投影、组合损失函数）完全相同

    总损失函数：
        L_total = λ1 * L_penalty + λ2 * L_stability + λ3 * L_barrier

    其中 L_penalty 使用随机平滑梯度估计。
    """
    import random as random_module
    from New_repair.geometry_module_new import compute_simplex_bound_batch

    if seed is not None:
        random_module.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 准备区域列表 ----------
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

    J_rows = J.shape[0]
    num_params = J.shape[1]

    if total_failures == 0 and not V_safe_list and not V_unsafe_list:
        if verbose:
            print("  [内循环PGD-RS] 没有失败区域也没有已验证区域，跳过。")
        return []

    batch_size = max(1, int(total_failures * batch_ratio))
    if verbose:
        print(f"  [内循环PGD-RS] J.shape={J.shape}, 失败: {total_failures}(batch={batch_size}), "
              f"V_safe={len(V_safe_list)}, V_unsafe={len(V_unsafe_list)}, "
              f"内迭代={num_inner_steps}, RS(N={rs_n_samples}, sigma={rs_sigma})")

    # ---------- 预计算 J 归一化 ----------
    epsilon = 1e-8
    J_norms = torch.norm(J, dim=1, keepdim=True)
    J_hat = J / (J_norms + epsilon)
    J_hat_np = J_hat.detach().cpu().numpy()

    inner_history = []

    # ---------- 内循环迭代 ----------
    for inner_step in range(num_inner_steps):
        all_terms = []

        # ---- L_penalty: 失败区域（使用随机平滑梯度）----
        if total_failures > 0:
            if batch_ratio < 1.0 and total_failures > batch_size:
                indices = random_module.sample(range(total_failures), batch_size)
            else:
                indices = list(range(total_failures))

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

            # F_h: clamp(h_lb, 0)^2
            if F_bh:
                h_lb_all, _ = compute_simplex_bound_batch(
                    model, F_bh, 'unsafe', dynamics_model=None, translator=translator)
                penalty_h = torch.clamp(h_lb_all, min=0.0) ** 2
                penalty_h = penalty_h[torch.isfinite(penalty_h) & (penalty_h > 0)]
                if penalty_h.numel() > 0:
                    all_terms.append((lambda_penalty, penalty_h.mean(), 'L_penalty_h'))

            # F_safe + F_depth + F_unsafe_split: clamp(tol - min_L, 0)^2
            F_safe_batch = F_bs + F_bd + F_bu
            if F_safe_batch:
                min_L_all = compute_simplex_bound_batch(
                    model, F_safe_batch, 'safe', dynamics_model=dynamics_model, translator=translator)
                safe_terms = torch.clamp(tolerance - min_L_all, min=0.0) ** 2
                safe_terms = safe_terms[torch.isfinite(safe_terms) & (safe_terms > 0)]
                if safe_terms.numel() > 0:
                    all_terms.append((lambda_penalty, safe_terms.mean(), 'L_penalty_safe'))

        # ---- L_stability: 已验证安全区 V_safe ----
        if V_safe_list:
            v_batch = max(1, int(len(V_safe_list) * verified_batch_ratio))
            v_idx = random_module.sample(range(len(V_safe_list)), min(v_batch, len(V_safe_list)))
            V_batch = [V_safe_list[i] for i in v_idx]
            min_L_all = compute_simplex_bound_batch(
                model, V_batch, 'safe', dynamics_model=dynamics_model, translator=translator)
            stability_terms = torch.clamp(gamma_safe - min_L_all, min=0.0) ** 2
            stability_terms = stability_terms[torch.isfinite(stability_terms)]
            if stability_terms.numel() > 0:
                all_terms.append((lambda_stability, stability_terms.mean(), 'L_stability'))

        # ---- L_barrier: 已验证障碍区 V_unsafe ----
        if V_unsafe_list:
            v_batch = max(1, int(len(V_unsafe_list) * verified_batch_ratio))
            v_idx = random_module.sample(range(len(V_unsafe_list)), min(v_batch, len(V_unsafe_list)))
            V_batch_unsafe = [V_unsafe_list[i] for i in v_idx]
            _, h_ub_all = compute_simplex_bound_batch(
                model, V_batch_unsafe, 'unsafe', dynamics_model=None, translator=translator)
            barrier_terms = torch.clamp(h_ub_all + gamma_unsafe, min=0.0) ** 2
            barrier_terms = barrier_terms[torch.isfinite(barrier_terms)]
            if barrier_terms.numel() > 0:
                all_terms.append((lambda_barrier, barrier_terms.mean(), 'L_barrier'))

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

        # ---- 合并损失（不使用 backward，手动前向+RS梯度）----
        weighted_sum = sum(w * t for (w, t, _) in all_terms)
        total_weight = sum(w for (w, _, _) in all_terms)
        total_loss = weighted_sum / total_weight

        # ---- 使用随机平滑计算 L_penalty 的梯度 ----
        # 从 all_terms 中提取 L_penalty 相关项的梯度
        # 注意：L_stability 和 L_barrier 仍然使用 autograd（它们来自已验证区域，梯度稳定）
        g_raw = torch.zeros(num_params, dtype=dtype, device=device)

        # L_penalty 梯度用 RS 估计
        loss_penalty_total = torch.tensor(0.0, dtype=dtype, device=device)
        n_penalty_terms = 0

        for (w, t, name) in all_terms:
            if name.startswith('L_penalty'):
                loss_penalty_total = loss_penalty_total + w * t
                n_penalty_terms += 1

        if n_penalty_terms > 0:
            # 对 L_penalty 使用 RS 梯度
            # 收集所有 failure simplices 的信息用于 RS
            F_all_batch = F_bh + F_bs + F_bd + F_bu if total_failures > 0 else []
            if F_all_batch:
                g_rs = compute_rs_gradient_for_batch(
                    model, F_all_batch, dynamics_model, translator,
                    tolerance, rs_n_samples, rs_sigma
                )
                if g_rs is not None:
                    g_raw.add_(g_rs)

        # L_stability 和 L_barrier 用 autograd
        autograd_terms = [(w, t) for (w, t, name) in all_terms if not name.startswith('L_penalty')]
        if autograd_terms:
            autograd_sum = sum(w * t for (w, t) in autograd_terms)
            model.zero_grad()
            autograd_sum.backward()
            g_autograd = torch.cat([
                p.grad.flatten() if p.grad is not None
                else torch.zeros(p.numel(), dtype=dtype, device=device)
                for p in model.parameters()
            ])
            g_autograd = torch.nan_to_num(g_autograd, nan=0.0, posinf=0.0, neginf=0.0)
            g_raw.add_(g_autograd)

        g_raw = torch.nan_to_num(g_raw, nan=0.0, posinf=0.0, neginf=0.0)

        grad_norm = g_raw.norm().item()
        if grad_norm < 1e-12:
            if verbose:
                print(f"    [内步 {inner_step+1}] 梯度过小，跳过更新。")
            inner_history.append({
                'step': inner_step + 1,
                'loss': total_loss.item(),
                'L_penalty': 0.0,
                'L_stability': 0.0,
                'L_barrier': 0.0,
                'grad_norm': grad_norm,
                'update_norm': 0.0,
                'active_constraints': 0,
            })
            continue

        # ---- PGD 求解 QP ----
        g_hat = g_raw / (grad_norm + epsilon)

        Jg = J_hat_np @ g_hat.cpu().numpy()
        JJT = J_hat_np @ J_hat_np.T

        lam = np.zeros(J_rows, dtype=np.float64)

        for pgd_iter in range(pgd_steps):
            grad_f = JJT @ lam - Jg
            lam = lam - pgd_lr * grad_f
            lam = np.maximum(lam, 0.0)

        lam_star = torch.tensor(lam, dtype=dtype, device=device)
        active_constraints = int(np.sum(lam > 1e-4))

        g_update = g_hat - (J_hat.T @ lam_star)

        update_norm_raw = g_update.norm().item()
        if update_norm_raw > grad_clip_norm:
            g_update = g_update * (grad_clip_norm / update_norm_raw)

        # ---- 参数更新 ----
        params = [p for p in model.parameters() if p.requires_grad]
        theta_old = torch.nn.utils.parameters_to_vector(params)
        theta_new = theta_old - lr * g_update
        torch.nn.utils.vector_to_parameters(theta_new, params)
        update_norm = (theta_new - theta_old).norm().item()

        # ---- 记录 ----
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
                  f"|g|={grad_norm:.4f}, |d|={update_norm:.6f}, active={active_constraints}/{J_rows}")

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


def compute_rs_gradient_for_batch(
    model: nn.Module,
    F_batch: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model,
    translator,
    tolerance: float,
    rs_n_samples: int,
    rs_sigma: float,
) -> torch.Tensor:
    """
    对一批 failure单纯形使用随机平滑计算聚合梯度。

    用于 inner_loop_repair_with_pgd_rs 中的 L_penalty 梯度计算。

    Args:
        F_batch: failure 单纯形列表
        tolerance: CBF 容差
        rs_n_samples: 采样次数
        rs_sigma: 噪声标准差

    Returns:
        g_rs: 聚合随机平滑梯度 [num_params]
    """
    from New_repair.geometry_module_new import compute_simplex_bound

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    if len(F_batch) == 0:
        return None

    params = list(model.parameters())
    theta_old = torch.nn.utils.parameters_to_vector(params).detach().clone()

    # 判断类型（假设 F_batch 内类型一致，这里简化处理）
    # 实际应用中需要传入类型信息
    g_accumulator = torch.zeros(num_params, dtype=dtype, device=device)
    n_contrib = 0

    for vertices in F_batch:
        # 前向计算 loss（判断是否需要包含）
        with torch.no_grad():
            # 假设是 safe 区域（大多数 failure 是 safe 类型）
            min_L = compute_simplex_bound(
                model, vertices, 'safe',
                dynamics_model=dynamics_model, translator=translator
            )
            loss_val = torch.clamp(tolerance - min_L, min=0.0).item()

        if loss_val <= 0:
            continue

        # RS 梯度
        accumulator = torch.zeros(num_params, dtype=dtype, device=device)
        valid_count = 0

        for _ in range(rs_n_samples):
            eps_i = torch.randn(num_params, dtype=dtype, device=device) * rs_sigma
            theta_i = theta_old + eps_i
            torch.nn.utils.vector_to_parameters(theta_i, params)

            with torch.no_grad():
                min_L = compute_simplex_bound(
                    model, vertices, 'safe',
                    dynamics_model=dynamics_model, translator=translator
                )
                psi_val = min_L.squeeze()

            if torch.isfinite(psi_val) and not torch.isnan(psi_val):
                accumulator.add_(eps_i * psi_val)
                valid_count += 1

        torch.nn.utils.vector_to_parameters(theta_old.clone(), params)

        if valid_count > 0:
            g_accumulator.add_(accumulator / (rs_n_samples * rs_sigma * rs_sigma))
            n_contrib += 1

    if n_contrib > 0:
        g_accumulator.div_(n_contrib)

    return g_accumulator
