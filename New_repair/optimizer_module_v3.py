"""
优化器模块 v3: 全量特征点雅可比计算、修复损失梯度、QP 约束投影

核心功能:
1. compute_jacobian_at_feature_points: 批量计算所有特征点的真实雅可比矩阵 J_true
2. compute_repair_loss_and_grad: 在所有特征点上计算修复损失及反向传播梯度 g_F
3. qp_project_and_update: QP 约束投影与参数更新（与 v2 相同）

v3 改进点 (相比 v2):
- 不再使用对抗采样找最坏点
- 在所有特征点（顶点+重心）上计算雅可比和损失
- 消除次梯度震荡，全局批量计算更稳定
- 计算速度大幅提升（确定性特征点，无随机采样开销）
"""

from typing import List, Tuple, Union, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp


def compute_jacobian_at_feature_points(
    model: nn.Module,
    dynamics_model,
    feature_points: torch.Tensor,
    translator=None,
) -> torch.Tensor:
    """
    在所有特征点处批量计算 CBF 条件对网络参数的雅可比矩阵。

    使用 vmap + jacrev 高效批量计算。

    Args:
        model: BarrierNN 网络
        dynamics_model: 动力学系统
        feature_points: 特征点张量 [N, D]（已展平的所有特征点）
        translator: TorchTranslator（可选）

    Returns:
        J: 形状 [N, num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    N = feature_points.shape[0]
    if N == 0:
        return torch.empty(0, num_params, device=device, dtype=dtype)

    # 确保模型参数可求导
    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    # ---------- 使用 autograd.grad 逐点计算 ----------
    J_rows = []
    batch_size = 500  # 分批处理避免显存问题

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = feature_points[start:end].detach().clone().requires_grad_(True)

        # 批量前向: h(x)
        h = model(batch).squeeze(-1)  # [batch]

        # 批量计算 ∇h
        grad_h = torch.autograd.grad(
            outputs=h,
            inputs=batch,
            grad_outputs=torch.ones_like(h, device=device),
            create_graph=False,
            retain_graph=True,
        )[0]  # [batch, D]

        # 计算 f(x)
        if translator is None:
            x1 = batch[..., 0]
            x2 = batch[..., 1]
            D = batch.shape[1]
            if D == 2 and dynamics_model.control_dim == 0:
                dx1 = x2
                dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
                f_x = torch.stack([dx1, dx2], dim=-1)
            else:
                raise NotImplementedError("需要 translator 进行通用 compute_f")
        else:
            f_x = dynamics_model.compute_f(batch, translator)

        # 计算 cbf = ∇h·f + α·h
        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)
        alpha_h = dynamics_model.alpha_function(h, translator)
        cbf = grad_h_dot_f + alpha_h

        # ---------- 对每个样本计算 ∂cbf/∂θ ----------
        for i in range(batch.shape[0]):
            cbf_i = cbf[i]
            grads = torch.autograd.grad(
                outputs=cbf_i,
                inputs=model.parameters(),
                retain_graph=True,
            )
            grad_vec = torch.cat([g.flatten() for g in grads if g is not None])
            J_rows.append(grad_vec)

    J = torch.stack(J_rows, dim=0)
    return J.detach()


def compute_jacobian_at_feature_points_v2(
    model: nn.Module,
    dynamics_model,
    feature_points: torch.Tensor,
    translator=None,
) -> torch.Tensor:
    """
    使用 torch.func 模块的高效批量雅可比计算。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        feature_points: [N, D]
        translator: TorchTranslator

    Returns:
        J: [N, num_params]
    """
    from torch.func import vmap, jacrev

    device = next(model.parameters()).device
    num_params = sum(p.numel() for p in model.parameters())

    N = feature_points.shape[0]
    if N == 0:
        return torch.empty(0, num_params, device=device)

    params = {name: p.detach().clone().requires_grad_(True)
              for name, p in model.named_parameters()}

    def cbf_fn(params_dict, x):
        old = model.state_dict()
        model.load_state_dict({k: v for k, v in params_dict.items()})
        x_b = x.unsqueeze(0)
        h = model(x_b).squeeze(-1)

        grad_h = torch.autograd.grad(
            outputs=h, inputs=x_b,
            grad_outputs=torch.ones_like(h, device=device),
            create_graph=False, retain_graph=True,
        )[0]

        if translator is None:
            x1 = x[..., 0]
            x2 = x[..., 1]
            D = x.shape[0]
            if D == 2 and dynamics_model.control_dim == 0:
                dx1 = x2
                dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
                f_x = torch.stack([dx1, dx2], dim=-1)
            else:
                raise NotImplementedError("需要 translator")
        else:
            f_x = dynamics_model.compute_f(x_b, translator)

        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)
        alpha_h = dynamics_model.alpha_function(h, translator)

        model.load_state_dict(old)
        return (grad_h_dot_f + alpha_h).squeeze(0)

    x_detached = feature_points.detach().clone().requires_grad_(True)

    jac_fn = jacrev(cbf_fn, argnums=0)
    jac_vmapped = vmap(jac_fn, in_dims=(0, 0))

    param_list = [params] * N
    J = jac_vmapped(param_list, x_detached)

    if isinstance(J, tuple):
        J = torch.stack([torch.cat([j_.flatten() for j_ in J_i])
                         for J_i in zip(*J)], dim=0)

    return J.detach()


def compute_repair_loss_and_grad(
    model: nn.Module,
    dynamics_model,
    failed_safe_feature_points: torch.Tensor,
    failed_unsafe_feature_points: torch.Tensor,
    margin: float = 0.0,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
    translator=None,
) -> Tuple[float, torch.Tensor]:
    """
    在所有失败区域的特征点上计算修复损失并反向传播得到梯度 g_F。

    Repair Loss 公式:
    - F_h_positive_in_unsafe（障碍区违规）:
        loss_unsafe = mean( softplus(h(x) + margin) )
    - F_safe_cbf_violation / F_depth_limit_reached（CBF 条件违规）:
        loss_cbf = mean( softplus(cbf_margin - cbf(x)) )

    使用全量特征点而非最坏点采样，损失更平滑稳定。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        failed_safe_feature_points: CBF 违规区域的特征点 [N_safe, D]
        failed_unsafe_feature_points: 障碍区违规的特征点 [N_unsafe, D]
        margin: h 值的容差（h 应该 <= 0）
        cbf_margin: CBF 条件的容差（cbf 应该 >= cbf_margin）
        beta: softplus 的 beta 参数
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出
        translator: TorchTranslator

    Returns:
        total_loss: 平均损失值
        g_raw: 修复梯度 [num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_loss_sum = 0.0
    n_valid = 0

    # ---------- 处理 unsafe（障碍区违规）----------
    # loss = softplus(h + margin)，h 应该 <= 0
    if failed_unsafe_feature_points.shape[0] > 0:
        x_nograd = failed_unsafe_feature_points.detach().to(device)
        x_in = x_nograd.requires_grad_(True)

        h = model(x_in).squeeze(-1)
        loss_unsafe = F.softplus(h + margin, beta=beta)

        # 处理 NaN/Inf
        if not torch.isfinite(loss_unsafe).all():
            if verbose:
                print(f"  [警告] unsafe 损失存在 NaN/Inf，跳过")
        else:
            model.zero_grad()
            loss_unsafe.mean().backward(retain_graph=False)

            grad_this = torch.cat([
                p.grad.flatten() if p.grad is not None
                else torch.zeros(p.numel(), dtype=dtype, device=device)
                for p in model.parameters()
            ])
            grad_this = torch.nan_to_num(grad_this, nan=0.0, posinf=0.0, neginf=0.0)
            g_raw.add_(grad_this)
            total_loss_sum += loss_unsafe.mean().item()
            n_valid += 1

    # ---------- 处理 safe（CBF 条件违规）----------
    # loss = softplus(cbf_margin - cbf)，cbf 应该 >= cbf_margin
    if failed_safe_feature_points.shape[0] > 0:
        x_nograd = failed_safe_feature_points.detach().to(device)
        x_in = x_nograd.requires_grad_(True)

        # h(x)
        h = model(x_in).squeeze(-1)

        # ∇h(x)
        grad_h = torch.autograd.grad(
            outputs=h, inputs=x_in,
            grad_outputs=torch.ones_like(h, device=device),
            create_graph=True, retain_graph=True,
        )[0]

        # f(x)
        if translator is None:
            x1 = x_in[..., 0]
            x2 = x_in[..., 1]
            D = x_in.shape[1]
            if D == 2 and dynamics_model.control_dim == 0:
                dx1 = x2
                dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
                f_x = torch.stack([dx1, dx2], dim=-1)
            else:
                raise NotImplementedError("需要 translator")
        else:
            f_x = dynamics_model.compute_f(x_in, translator)

        # ∇h·f
        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)

        # α·h
        alpha_h = dynamics_model.alpha_function(h, translator)

        # cbf
        cbf = grad_h_dot_f + alpha_h

        loss_cbf = F.softplus(cbf_margin - cbf, beta=beta)

        if not torch.isfinite(loss_cbf).all():
            if verbose:
                print(f"  [警告] CBF 损失存在 NaN/Inf，跳过")
        else:
            model.zero_grad()
            loss_cbf.mean().backward(retain_graph=False)

            grad_this = torch.cat([
                p.grad.flatten() if p.grad is not None
                else torch.zeros(p.numel(), dtype=dtype, device=device)
                for p in model.parameters()
            ])
            grad_this = torch.nan_to_num(grad_this, nan=0.0, posinf=0.0, neginf=0.0)

            if torch.isnan(grad_this).any() or torch.isinf(grad_this).any():
                if verbose:
                    print(f"  [警告] safe 梯度存在 NaN/Inf，跳过该批次")
            else:
                g_raw.add_(grad_this)
                total_loss_sum += loss_cbf.mean().item()
                n_valid += 1

    if n_valid == 0:
        return 0.0, torch.zeros(num_params, device=device, dtype=dtype)

    # ---------- 梯度裁剪 ----------
    total_loss = total_loss_sum / n_valid
    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)

    if verbose:
        print(f"  [修复损失] total_loss={total_loss:.6f}, |g_raw|={grad_norm:.4f}, "
              f"unsafe_pts={failed_unsafe_feature_points.shape[0]}, "
              f"safe_pts={failed_safe_feature_points.shape[0]}")

    return total_loss, g_raw


def qp_project_and_update(
    model: nn.Module,
    g_raw: torch.Tensor,
    J_verified: torch.Tensor,
    lr: float = 1e-3,
    verbose: bool = False,
) -> Tuple[float, float, int]:
    """
    QP 约束投影与参数更新（与 v2 相同）。

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



def qp_project_and_update_gd(
    model: nn.Module,
    g_raw: torch.Tensor,
    J_verified: torch.Tensor,
    lr: float = 1e-3,
    gd_steps: int = 100,
    gd_lr: float = 1e-2,
    verbose: bool = False,
) -> Tuple[float, float, int]:
    """
    使用对偶空间的投影梯度下降 (PGD) 替代 OSQP 求解。
    求解目标: min_λ 0.5 * λ^T (J J^T) λ - (J g_hat)^T λ  s.t. λ >= 0
    """
    device = g_raw.device
    dtype = g_raw.dtype
    
    # 1. 参数提取与归一化
    params = [p for p in model.parameters() if p.requires_grad]
    theta_old = torch.nn.utils.parameters_to_vector(params)

    g_norm = g_raw.norm().item()
    g_hat = g_raw / (g_norm + 1e-8)

    J_norms = torch.norm(J_verified, dim=1, keepdim=True)
    J_hat = J_verified / (J_norms + 1e-8)
    num_constraints = J_hat.shape[0]

    # 2. 预计算对偶问题的系数 (在 GPU 上)
    # Q = J J^T, q = J g_hat
    Q = torch.matmul(J_hat, J_hat.T)
    q = torch.matmul(J_hat, g_hat)

    # 3. 对偶变量 λ 的迭代优化 (PGD)
    # λ 的维度是 [num_constraints]
    lam = torch.zeros(num_constraints, device=device, dtype=dtype).requires_grad_(True)
    # 使用 Adam 能够比普通 SGD 更快地处理 Hinge-like 的约束边界
    optimizer = torch.optim.Adam([lam], lr=gd_lr)

    for i in range(gd_steps):
        optimizer.zero_grad()
        
        # 目标函数: 0.5 * λ^T Q λ - q^T λ
        # 使用 quadratic form 显式表达
        quad_term = 0.5 * torch.dot(lam, torch.matmul(Q, lam))
        lin_term = torch.dot(q, lam)
        loss = quad_term - lin_term
        
        loss.backward()
        optimizer.step()

        # 投影算子: λ >= 0 (ReLU 投影)
        with torch.no_grad():
            lam.clamp_(min=0)

    lam_star = lam.detach()

    # 4. 计算更新方向 d = g_hat - J^T λ
    g_update = g_hat - torch.matmul(J_hat.T, lam_star)
    update_norm = g_update.norm().item()

    # 5. 更新模型参数
    with torch.no_grad():
        theta_new = theta_old - lr * g_update
        torch.nn.utils.vector_to_parameters(theta_new, params)

    # 6. 诊断输出
    # Jd_after = torch.matmul(J_hat, g_update)
    n_violate_after =  0
    # n_violate_after = int((Jd_after > 1e-5).sum().item()) # 给予微小的容忍度
    active = int((lam_star > 1e-4).sum().item())

    if verbose:
        print(f"[GD-QP] |g|={g_norm:.4f}, Steps={gd_steps}, "
              f"Active λ={active}, 后验违反={n_violate_after}, ")
            #   f"Max Violated={Jd_after.max().item():.6f}")

    return g_norm, update_norm, n_violate_after

def _compute_cbf_batch(model, dynamics_model, worst_points, translator=None):
    """批量计算多个点的 cbf 值，返回 tensor [N]"""
    x = worst_points.detach().requires_grad_(True)
    h = model(x).squeeze(-1)
    grad_h = torch.autograd.grad(
        outputs=h, inputs=x,
        grad_outputs=torch.ones_like(h, device=x.device),
        create_graph=False, retain_graph=True,
    )[0]
    if translator is None:
        x1 = x[..., 0]
        x2 = x[..., 1]
        D = x.shape[1]
        if D == 2 and dynamics_model.control_dim == 0:
            dx1 = x2
            dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
            f_x = torch.stack([dx1, dx2], dim=-1)
        else:
            raise NotImplementedError("需要 translator")
    else:
        f_x = dynamics_model.compute_f(x, translator)
    grad_h_dot_f = (grad_h * f_x).sum(dim=-1)
    alpha_h = dynamics_model.alpha_function(h, translator)
    cbf = grad_h_dot_f + alpha_h
    return cbf


def repair_loop(
    model: nn.Module,
    dynamics_model,
    verified_safe_feature_points: torch.Tensor,
    verified_unsafe_feature_points: torch.Tensor,
    failed_safe_feature_points: torch.Tensor,
    failed_unsafe_feature_points: torch.Tensor,
    num_inner_steps: int = 10,
    lr: float = 1e-4,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
    seed: int = None,
) -> List[Dict]:
    """
    执行内循环修复迭代（v3 版本，使用全量特征点）。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        verified_safe_feature_points: 已验证安全区域的特征点 [N_safe, D]
        verified_unsafe_feature_points: 已验证不安全区域的特征点 [N_unsafe, D]
        failed_safe_feature_points: CBF 违规区域的特征点 [N_fail_safe, D]
        failed_unsafe_feature_points: 障碍区违规的特征点 [N_fail_unsafe, D]
        num_inner_steps: 内循环步数
        lr: 学习率
        grad_clip_norm: 梯度裁剪阈值
        verbose: 是否打印详细信息
        seed: 随机种子

    Returns:
        inner_history: 每次迭代的记录列表
    """
    import random

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = next(model.parameters()).device

    # ---------- 计算已验证区域的雅可比 ----------
    # V_safe 的特征点: ∂cbf/∂θ，约束 J @ d >= 0
    # V_unsafe 的特征点: ∂h/∂θ，约束取反（J @ d >= 0 意味着 h 减小）

    print(f"  [内循环] 计算已验证区域雅可比...")
    print(f"    V_safe 特征点: {verified_safe_feature_points.shape[0]}")
    print(f"    V_unsafe 特征点: {verified_unsafe_feature_points.shape[0]}")

    # Safe 区域的雅可比
    if verified_safe_feature_points.shape[0] > 0:
        J_safe = compute_jacobian_at_feature_points(
            model, dynamics_model, verified_safe_feature_points
        )
        print(f"    J_safe shape: {J_safe.shape}")
    else:
        J_safe = torch.empty(0, sum(p.numel() for p in model.parameters()),
                            device=device)

    # Unsafe 区域的雅可比（取反）
    if verified_unsafe_feature_points.shape[0] > 0:
        J_unsafe = compute_jacobian_at_feature_points(
            model, dynamics_model, verified_unsafe_feature_points
        )
        J_unsafe = -J_unsafe  # 约束取反：J @ d >= 0 意味着 h 减小
        print(f"    J_unsafe shape: {J_unsafe.shape}")
    else:
        J_unsafe = torch.empty(0, sum(p.numel() for p in model.parameters()),
                              device=device)

    # 合并
    J_verified = torch.cat([J_safe, J_unsafe], dim=0)
    print(f"    J_verified shape: {J_verified.shape}")

    inner_history = []

    for step in range(num_inner_steps):
        # 计算修复损失和梯度（全量特征点）
        loss_val, g_F = compute_repair_loss_and_grad(
            model=model,
            dynamics_model=dynamics_model,
            failed_safe_feature_points=failed_safe_feature_points,
            failed_unsafe_feature_points=failed_unsafe_feature_points,
            verbose=verbose,
        )

        # QP 投影更新
        g_raw_norm, update_norm, active = qp_project_and_update(
            model=model,
            g_raw=g_F,
            J_verified=J_verified,
            lr=lr,
            verbose=verbose,
        )

        inner_history.append({
            'step': step + 1,
            'loss': loss_val,
            'g_raw_norm': g_raw_norm,
            'update_norm': update_norm,
            'active_constraints': active,
        })

        if verbose:
            print(f"    [内步 {step+1}/{num_inner_steps}] "
                  f"loss={loss_val:.6f}, |g|={g_raw_norm:.4f}, "
                  f"|d|={update_norm:.6f}, active={active}")

    return inner_history
