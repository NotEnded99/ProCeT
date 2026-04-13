"""
优化器模块 v2: 雅可比矩阵计算、修复损失梯度、QP 约束投影

核心功能:
1. compute_jacobian_at_worst_points: 批量计算真实雅可比矩阵 J_true
2. compute_sampled_repair_loss_and_grad: 计算修复损失及反向传播梯度 g_F
3. qp_project_and_update: QP 约束投影与参数更新
"""

from typing import List, Tuple, Union, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp

from torch.func import vmap, jacrev


def compute_jacobian_at_worst_points_simple(
    model: nn.Module,
    dynamics_model,
    worst_points: torch.Tensor,
    translator=None,
) -> torch.Tensor:
    """
    在最坏点处逐点计算 CBF 条件对网络参数的雅可比矩阵。

    使用 torch.autograd.grad 对每个最坏点单独计算参数梯度，
    提取 ∂cbf/∂θ 的梯度向量。

    Args:
        model: BarrierNN 网络
        dynamics_model: 动力学系统
        worst_points: 最坏点张量 [N, D]
        translator: 未使用（保留接口兼容性）

    Returns:
        J: 形状 [N, num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    N = worst_points.shape[0]
    num_params = sum(p.numel() for p in model.parameters())

    if N == 0:
        return torch.empty(0, num_params, device=device, dtype=dtype)

    # 确保模型参数可求导
    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    J_rows = []

    for i in range(N):
        x_i = worst_points[i].detach().clone().to(device).requires_grad_(True)

        # ---------- 前向: h(x) ----------
        x_b = x_i.unsqueeze(0)  # [1, D]
        h = model(x_b).squeeze(-1)  # [1]

        # ---------- ∇x h(x) ----------
        grad_h = torch.autograd.grad(
            outputs=h,
            inputs=x_b,
            grad_outputs=torch.ones_like(h, device=device),
            create_graph=True,   # 需要 create_graph 以便后续对参数求导
            retain_graph=True,
        )[0]  # [1, D]

        # ---------- f(x) ----------
        if translator is None:
            raise NotImplementedError("Translator is required for dynamics_model.compute_f in safe case.")
        else:
            f_x = dynamics_model.compute_f(x_b, translator)  # [1, D]

        # ---------- ∇h·f ----------
        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)  # [1]

        # ---------- α·h ----------
        alpha_h = dynamics_model.alpha_function(h, translator)  # [1]

        # ---------- cbf scalar ----------
        cbf_i = (grad_h_dot_f + alpha_h).squeeze(0)  # scalar

        # ---------- ∂cbf/∂θ ----------
        grads = torch.autograd.grad(
            outputs=cbf_i,
            inputs=model.parameters(),
            retain_graph=False,
        )
        grad_vec = torch.cat([g.flatten() for g in grads if g is not None])
        J_rows.append(grad_vec)

    J = torch.stack(J_rows, dim=0)  # [N, P]
    return J.detach()


def compute_jacobian_at_worst_points_vmap(
    model: nn.Module,
    dynamics_model,
    worst_points: torch.Tensor,
    translator=None,
) -> torch.Tensor:
    """
    使用 vmap + jacrev 批量计算雅可比矩阵（高效版本）。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        worst_points: [N, D]
        translator: 未使用

    Returns:
        J: [N, num_params]
    """
    device = next(model.parameters()).device
    N = worst_points.shape[0]
    num_params = sum(p.numel() for p in model.parameters())

    if N == 0:
        return torch.empty(0, num_params, device=device)

    params = {name: p.detach().clone().requires_grad_(True)
              for name, p in model.named_parameters()}

    def cbf_fn(params, x):
        old = model.state_dict()
        model.load_state_dict({k: v for k, v in params.items()})
        x_b = x.unsqueeze(0)
        h = model(x_b).squeeze(-1)

        grad_h = torch.autograd.grad(
            outputs=h, inputs=x_b,
            grad_outputs=torch.ones_like(h, device=device),
            create_graph=False, retain_graph=True,
        )[0]

        if translator is None:
            raise NotImplementedError("Translator is required for dynamics_model.compute_f in safe case.")
        else:
            f_x = dynamics_model.compute_f(x_b, translator)

        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)
        alpha_h = dynamics_model.alpha_function(h, translator)

        model.load_state_dict(old)
        return (grad_h_dot_f + alpha_h).squeeze(0)

    x_detached = worst_points.detach().clone().requires_grad_(True)

    jac_fn = jacrev(cbf_fn, argnums=0)
    jac_vmapped = vmap(jac_fn, in_dims=(0, 0))

    param_list = [params] * N
    J = jac_vmapped(param_list, x_detached)  # [N, num_params]

    if isinstance(J, tuple):
        J = torch.stack([torch.cat([j_.flatten() for j_ in J_i]) for J_i in zip(*J)], dim=0)

    return J.detach()


def compute_sampled_repair_loss_and_grad(
    model: nn.Module,
    dynamics_model,
    failed_worst_points: Dict[str, List[Tuple[torch.Tensor, float]]],
    margin: float = 0.0,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
    translator=None,
) -> Tuple[float, torch.Tensor]:
    """
    计算采样修复损失并反向传播得到梯度 g_F。

    Repair Loss 公式:
    - F_h_positive_in_unsafe（障碍区违规）:
        loss_unsafe = mean( F.softplus(h(x) + margin, beta=beta) )
    - F_safe_cbf_violation / F_depth_limit_reached（CBF 条件违规）:
        loss_cbf = mean( F.softplus(cbf_margin - cbf_condition(x), beta=beta) )

    所有 loss 均使用 mean 方式计算，梯度与样本数量解耦。
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_loss_sum = 0.0
    n_valid = 0

    # ---------- 处理 unsafe（障碍区违规）----------
    # loss = softplus(h + margin)，h 应该 <= 0，margin 通常为 0
    unsafe_list = failed_worst_points.get('unsafe', [])
    unsafe_losses = []

    for x_worst, h_val in unsafe_list:
        x_nograd = x_worst.detach().to(device)
        x_in = x_nograd.unsqueeze(0).requires_grad_(True)  # [1, D]
        h = model(x_in).squeeze(-1)  # [1]
        loss_term = F.softplus(h + margin, beta=beta)  # [1]

        if not torch.isfinite(loss_term).all():
            continue

        unsafe_losses.append(loss_term)

    # 批量 mean backward
    if len(unsafe_losses) > 0:
        loss_unsafe = torch.stack(unsafe_losses).mean()
        model.zero_grad()
        loss_unsafe.backward(retain_graph=False)

        grad_this = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        grad_this = torch.nan_to_num(grad_this, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_this)
        total_loss_sum += loss_unsafe.item()
        n_valid += 1

    # ---------- 处理 safe（CBF 条件违规）----------
    # loss = softplus(cbf_margin - cbf)，cbf 应该 >= cbf_margin
    safe_list = failed_worst_points.get('safe', [])
    safe_losses = []

    for x_worst, cbf_val in safe_list:
        x_nograd = x_worst.detach().to(device)
        x_in = x_nograd.unsqueeze(0).requires_grad_(True)  # [1, D]

        # h(x)
        h = model(x_in).squeeze(-1)  # [1]

        # ∇h(x)
        grad_h = torch.autograd.grad(
            outputs=h, inputs=x_in,
            grad_outputs=torch.ones_like(h, device=device),
            create_graph=True, retain_graph=True,
        )[0]  # [1, D]

        # f(x)
        if translator is None:
            raise NotImplementedError("Translator is required for dynamics_model.compute_f in safe case.")
        else:
            f_x = dynamics_model.compute_f(x_in, translator)  # [1, D]

        # ∇h·f
        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)  # [1]

        # α·h
        alpha_h = dynamics_model.alpha_function(h, translator)  # [1]

        # cbf
        cbf = (grad_h_dot_f + alpha_h).squeeze(0)  # scalar

        loss_term = F.softplus(cbf_margin - cbf, beta=beta)

        if not torch.isfinite(loss_term).all():
            continue

        safe_losses.append(loss_term)

    # 批量 mean backward
    if len(safe_losses) > 0:
        loss_cbf = torch.stack(safe_losses).mean()
        model.zero_grad()
        loss_cbf.backward(retain_graph=False)

        grad_this = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        grad_this = torch.nan_to_num(grad_this, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.isnan(grad_this).any() or torch.isinf(grad_this).any():
            if verbose:
                print(f"  [警告] safe 梯度存在 NaN/Inf，跳过")
        else:
            g_raw.add_(grad_this)
            total_loss_sum += loss_cbf.item()
            n_valid += 1

    if n_valid == 0:
        return 0.0, torch.zeros(num_params, device=device, dtype=dtype)

    # ---------- 梯度裁剪 ----------
    total_loss = total_loss_sum / n_valid
    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)

    if verbose:
        print(f"  [修复损失] total_loss={total_loss:.6f}, "
              f"|g_raw|={grad_norm:.4f}, "
              f"unsafe_terms={len(unsafe_list)}, safe_terms={len(safe_list)}")

    return total_loss, g_raw


def line_search_safety_update(
    model: nn.Module,
    g_raw: torch.Tensor,
    J_verified: torch.Tensor,
    worst_points: torch.Tensor,
    dynamics_model,
    lr: float = 1e-3,
    max_backtrack: int = 15,
    verbose: bool = False,
    translator=None,
) -> Tuple[float, float, int]:
    """
    线搜索 + 安全检查的参数更新。

    1. 计算基线 cbf 值: cbf_old[x_i] = cbf(θ_old, x_i)，x_i = 已验证最坏点
    2. 取 d = g_raw / |g_raw|（归一化方向），步长 = lr * |g_raw|
    3. 检查 cbf(θ_old + α*d, x_i) >= cbf_old[x_i] - ε（容差）
    4. 若违反，回溯找到满足约束的最大 α

    安全约束: cbf 在已验证最坏点不降低（margin > 0 容差）
    """
    device = g_raw.device
    dtype = g_raw.dtype
    N = J_verified.shape[0]

    params = [p for p in model.parameters() if p.requires_grad]
    theta_old = torch.nn.utils.parameters_to_vector(params).detach().clone()

    g_norm = g_raw.norm().item()
    if g_norm < 1e-10:
        return 0.0, 0.0, 0

    d = g_raw / g_norm  # 单位方向

    # ---------- 预计算基线 cbf 值 ----------
    cbf_old = _compute_cbf_batch(model, dynamics_model, worst_points, translator).detach()
    cbf_old_np = cbf_old.cpu().numpy()

    # ---------- 检查初始方向是否满足约束（J @ d 近似） ----------
    Jd = (J_verified @ d).cpu().numpy()  # [N]
    n_violate_j = int(np.sum(Jd < 0))
    if verbose:
        print(f"  [线搜索] |d|={g_norm:.4f}, J@d<0 违反={n_violate_j}/{N}, "
              f"min_Jd={Jd.min():.4f}")

    # ---------- 如果 J @ d >= 0 全部满足，尝试完整步长 ----------
    if n_violate_j == 0:
        step_size = lr
        theta_new = theta_old + step_size * d
        torch.nn.utils.vector_to_parameters(theta_new, params)
        cbf_new = _compute_cbf_batch(model, dynamics_model, worst_points, translator).detach().cpu().numpy()
        delta_cbf = cbf_new - cbf_old_np
        n_violate = int(np.sum(delta_cbf < 0))

        if n_violate == 0:
            if verbose:
                print(f"  [线搜索] ✓ 方向安全 (J@d)，使用完整步长 α={step_size:.6f}")
            return g_norm, step_size, 0
        else:
            if verbose:
                print(f"  [线搜索] J@d 安全但 cbf 仍有违反: {n_violate}/{N}，进入回溯")

    # ---------- 回溯线搜索 ----------
    alpha = lr
    best_alpha = alpha

    for bt in range(max_backtrack):
        theta_new = theta_old + alpha * d
        torch.nn.utils.vector_to_parameters(theta_new, params)

        cbf_new = _compute_cbf_batch(model, dynamics_model, worst_points, translator).detach().cpu().numpy()
        delta_cbf = cbf_new - cbf_old_np
        n_violate = int(np.sum(delta_cbf < 0))
        min_delta = float(np.min(delta_cbf))

        if verbose and bt == 0:
            print(f"  [线搜索] α={alpha:.6f}, 违反={n_violate}/{N}, "
                  f"min_Δcbf={min_delta:.6f}")

        if n_violate == 0:
            if verbose:
                print(f"  [线搜索] ✓ 找到可行步长 α={alpha:.6f} (回溯{bt+1}次)")
            return g_norm, alpha, 0

        best_alpha = alpha
        alpha = alpha * 0.5

    # ---------- 最终保守步长 ----------
    theta_new = theta_old + alpha * d
    torch.nn.utils.vector_to_parameters(theta_new, params)
    cbf_final = _compute_cbf_batch(model, dynamics_model, worst_points, translator).detach().cpu().numpy()
    n_violate = int(np.sum(cbf_final - cbf_old_np < 0))

    if verbose:
        print(f"  [线搜索] 用保守步长 α={alpha:.6f}，违反={n_violate}/{N}")

    return g_norm, alpha, n_violate


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
        raise NotImplementedError("Translator is required for dynamics_model.compute_f in safe case.")
    else:
        f_x = dynamics_model.compute_f(x, translator)
    grad_h_dot_f = (grad_h * f_x).sum(dim=-1)
    alpha_h = dynamics_model.alpha_function(h, translator)
    cbf = grad_h_dot_f + alpha_h
    return cbf  # [N] tensor


# def qp_project_and_update(
#     model: nn.Module,
#     g_raw: torch.Tensor,
#     J_verified: torch.Tensor,
#     lr: float = 1e-3,
#     verbose: bool = False,
# ) -> Tuple[float, float, int]:
#     """
#     QP 约束投影与参数更新。

#     求解 QP:
#         min_λ 1/2 || J^T λ - g_hat ||^2   s.t.  λ >= 0

#     其中 g_hat = g_raw / |g_raw| 已归一化。

#     Args:
#         model: 神经网络
#         g_raw: 原始梯度 [num_params]
#         J_verified: 已验证区域的雅可比矩阵 [N, num_params]
#         lr: 学习率（定长步长）
#         verbose: 诊断输出

#     Returns:
#         (g_raw_norm, update_norm, n_violate)
#         - g_raw_norm: 原始梯度范数
#         - update_norm: 更新方向范数
#         - n_violate: 更新后 J @ d > 0 的约束数（J @ d < 0 则 cbf 增加）
#     """
#     device = g_raw.device
#     dtype = g_raw.dtype
#     params = [p for p in model.parameters() if p.requires_grad]
#     theta_old = torch.nn.utils.parameters_to_vector(params)

#     g_norm = g_raw.norm().item()
#     g_hat = g_raw / (g_norm + 1e-8)

#     J_norms = torch.norm(J_verified, dim=1, keepdim=True)
#     J_hat = J_verified / (J_norms + 1e-8)

#     # ---- 诊断：更新前的约束违反情况 ----
#     Jg_before = (J_hat @ g_hat).cpu().numpy()
#     n_violate_before = int(np.sum(Jg_before > 0))

#     J_np = J_hat.detach().cpu().numpy()
#     g_np = g_hat.detach().cpu().numpy()

#     lam = cp.Variable(J_verified.shape[0], nonneg=True)
#     residual = J_np.T @ lam - g_np
#     prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(residual)))
#     prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
#     lam_value = lam.value if lam.value is not None else np.zeros(J_verified.shape[0])

#     lam_star = torch.tensor(lam_value, dtype=dtype, device=device)
#     g_update = g_hat - (J_hat.T @ lam_star)

#     update_norm = g_update.norm().item()
#     # if update_norm > 1.0:
#     #     g_update = g_update * (1.0 / update_norm)
#     #     update_norm = 1.0

#     theta_new = theta_old - lr * g_update
#     torch.nn.utils.vector_to_parameters(theta_new, params)

#     # ---- 诊断：更新后的约束违反情况 ----
#     Jd_after = (J_hat @ g_update).cpu().numpy()
#     n_violate_after = int(np.sum(Jd_after > 0))

#     active = int(np.sum(lam_value > 1e-4))

#     if verbose:
#         print(f"  [QP] |g|={g_norm:.4f}, 初始违反={n_violate_before}/{J_verified.shape[0]}, "
#               f"active(λ>1e-4)={active}, "
#               f"投影后违反={n_violate_after}, Jd_after_max={Jd_after.max():.6f}")

#     return g_norm, update_norm, active

def qp_project_and_update(
    model: nn.Module,
    g_raw: torch.Tensor,
    J_verified: torch.Tensor,
    lr: float = 1e-3,
    verbose: bool = False,
) -> Tuple[float, float, int]:
    """
    QP 约束投影与参数更新。
    
    求解 QP:
        min_λ 1/2 λ^T (J J^T) λ - (J g_hat)^T λ   s.t.  λ >= 0
    其中 g_hat = g_raw / |g_raw| 已归一化。
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

    # ================= 核心优化区 =================
    # 1. 在 GPU 上利用 PyTorch 高效计算二次项 Q 和一次项 q
    Q_tensor = J_hat @ J_hat.T  # shape: [N, N]
    q_tensor = J_hat @ g_hat    # shape: [N]

    # 2. 将小尺寸矩阵转移到 CPU 供 CVXPY 求解
    Q_np = Q_tensor.detach().cpu().numpy()
    q_np = q_tensor.detach().cpu().numpy()

    # # 注入微小正定扰动，防止 J 不满秩导致的 OSQP 报错或数值不稳定
    # Q_np += np.eye(Q_np.shape[0]) * 1e-6 

    lam = cp.Variable(J_verified.shape[0], nonneg=True)
    # 使用二次规划的标准形式：1/2 x^T Q x - q^T x
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(lam, Q_np) - q_np.T @ lam))
    
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        lam_value = lam.value if lam.value is not None else np.zeros(J_verified.shape[0])
    except Exception as e:
        if verbose: print(f"[QP Error] Solver failed: {e}")
        lam_value = np.zeros(J_verified.shape[0])
    # ==============================================

    lam_star = torch.tensor(lam_value, dtype=dtype, device=device)
    
    # 根据 d = g_hat - J_hat^T λ 计算最终更新方向
    g_update = g_hat - (J_hat.T @ lam_star)
    update_norm = g_update.norm().item()

    # 实施参数更新 (梯度下降，减去更新方向)
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

    # 修正了返回值，与 Docstring 保持一致
    return g_norm, update_norm, n_violate_after


def _clone_model_with_grad(model: nn.Module) -> nn.Module:
    """克隆模型并设置 requires_grad=True"""
    cloned = model.state_dict()
    # 实际上不需要真的克隆，直接用原模型即可
    return model


def repair_loop(
    model: nn.Module,
    dynamics_model,
    safe_worst_points: torch.Tensor,
    unsafe_worst_points: torch.Tensor,
    cbf_worst_points: torch.Tensor,
    failed_worst_points: Dict[str, List[Tuple[torch.Tensor, float]]],
    num_inner_steps: int = 10,
    lr: float = 1e-4,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
    seed: int = None,
) -> List[Dict]:
    """
    执行内循环修复迭代。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        safe_worst_points: V_safe 的最坏点 [N_safe, D]
        unsafe_worst_points: F_h 的最坏点 [N_unsafe, D]
        cbf_worst_points: F_safe/F_depth 的最坏点 [N_cbf, D]
        failed_worst_points: dict，保存最坏点和对应的值
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

    # 计算已验证区域的雅可比
    print(f"  [内循环] 计算已验证区域雅可比: {safe_worst_points.shape[0]} 个点")
    J_verified = compute_jacobian_at_worst_points_simple(
        model, dynamics_model, safe_worst_points
    )
    print(f"  [内循环] J_verified shape: {J_verified.shape}")

    inner_history = []

    for step in range(num_inner_steps):
        # 计算修复损失和梯度
        loss_val, g_F = compute_sampled_repair_loss_and_grad(
            model=model,
            dynamics_model=dynamics_model,
            failed_worst_points=failed_worst_points,
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
