"""
优化器模块 v8: 使用上下界计算修复损失（替代采样方法）

核心改进:
1. 对于 Unsafe 区域 (F_h_positive_in_unsafe): 使用 LBP 上界 (h_ub)
   - LBP 上界更紧
   - 没有梯度问题（可以直接求导）

2. 对于 Safe 区域 (F_safe_cbf_violation, F_depth_limit_reached): 使用 IBP 下界 (min_L)
   - LBP 的 CBF 下界在 safe 区域可能有 NaN 问题
   - IBP 下界数值稳定

损失函数:
- Unsafe: loss_unsafe = softplus(h_ub + margin)，推动 h_ub <= 0
- Safe: loss_safe = softplus(cbf_margin - min_L)，推动 min_L >= cbf_margin
"""

from typing import List, Tuple
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Helper Functions (inlined from verify_cbf_ibp to avoid circular imports)
# =============================================================================

def vec_min(*tensors: torch.Tensor) -> torch.Tensor:
    """Compute element-wise minimum across multiple tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.minimum(result, t)
    return result


def vec_max(*tensors: torch.Tensor) -> torch.Tensor:
    """Compute element-wise maximum across multiple tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.maximum(result, t)
    return result


def _compute_dynamics_bounds_ibp(batch, dynamics_model, device, dtype):  # noqa: N805
    """
    Compute dynamics bounds by evaluating f(x) at region vertices/centers.
    Returns bounds in interval format [f_L, f_U] and [g_L, g_U].
    """
    from lbp_neural_cbf.translators import TorchTranslator

    n = dynamics_model.input_dim
    m = dynamics_model.control_dim

    translator = TorchTranslator(device=device, dtype=dtype)

    if isinstance(batch[0], type(batch[0])) and batch[0].__class__.__name__ == 'SimplicialRegion':
        from lbp_neural_cbf.regions import SimplicialRegion
        f_L_list = []
        f_U_list = []
        g_L_list = []
        g_U_list = []

        for sample in batch:
            verts = torch.tensor(sample.vertices, device=device, dtype=dtype)
            f_vals = []
            g_vals = []
            for v in verts:
                x = v.unsqueeze(0)
                with torch.no_grad():
                    f_val = dynamics_model.compute_f(x, translator).squeeze(0)
                    f_vals.append(f_val)
                    if m > 0:
                        g_val = dynamics_model.compute_g(x, translator).squeeze(0)
                        g_vals.append(g_val)

            f_vals = torch.stack(f_vals, dim=0)
            f_L_list.append(f_vals.min(dim=0).values)
            f_U_list.append(f_vals.max(dim=0).values)

            if m > 0:
                g_vals = torch.stack(g_vals, dim=0)
                g_L_list.append(g_vals.min(dim=0).values)
                g_U_list.append(g_vals.max(dim=0).values)

        f_L = torch.stack(f_L_list, dim=0)
        f_U = torch.stack(f_U_list, dim=0)

        if m > 0:
            g_L = torch.stack(g_L_list, dim=0)
            g_U = torch.stack(g_U_list, dim=0)
        else:
            g_L = None
            g_U = None
    else:
        raise TypeError(f"Unsupported region type: {type(batch[0])}")

    f_bounds = (f_L, f_U)
    g_bounds = (g_L, g_U) if m > 0 else None

    return f_bounds, g_bounds


def _compute_min_L_ibp(
    batch,
    dynamics_model,
    ibp_calculator,
    device,
    dtype,
    h_lb_lbp=None
) -> torch.Tensor:
    """
    Compute lower bound on CBF condition using IBP bounds.

    CBF condition: ∇h(x)·f(x) + ∇h(x)·g(x)·u + α(h(x)) ≥ 0
    """
    m = dynamics_model.control_dim

    # Get network bounds and Jacobian bounds from IBP
    h_lb_ibp, h_ub_ibp = ibp_calculator.get_network_output_bounds()
    J_L, J_U = ibp_calculator.ibp_jacobian_bounds(batch)

    n = J_L.shape[-1]

    # Get dynamics bounds
    f_bounds, g_bounds = _compute_dynamics_bounds_ibp(batch, dynamics_model, device, dtype)
    f_L, f_U = f_bounds

    # Compute drift lower bound: J @ f
    term1 = J_L * f_L
    term2 = J_L * f_U
    term3 = J_U * f_L
    term4 = J_U * f_U

    L_drift = vec_min(term1, term2, term3, term4).sum(dim=-1)

    # Compute control lower bound if applicable
    L_ctrl = torch.zeros_like(L_drift)

    if m > 0 and g_bounds is not None:
        g_L, g_U = g_bounds

        batch_size = len(batch)
        v_L = torch.zeros(batch_size, m, dtype=dtype, device=device)
        v_U = torch.zeros(batch_size, m, dtype=dtype, device=device)

        for k in range(m):
            g_L_k = g_L[..., k]
            g_U_k = g_U[..., k]

            t1 = J_L * g_L_k
            t2 = J_L * g_U_k
            t3 = J_U * g_L_k
            t4 = J_U * g_U_k

            v_L[:, k] = vec_min(t1, t2, t3, t4).sum(dim=-1)
            v_U[:, k] = vec_max(t1, t2, t3, t4).sum(dim=-1)

        u_min = torch.tensor(dynamics_model.u_min, dtype=dtype, device=device)
        u_max = torch.tensor(dynamics_model.u_max, dtype=dtype, device=device)

        for k in range(m):
            v_L_k = v_L[:, k]
            v_U_k = v_U[:, k]

            pos_mask = v_L_k >= 0
            neg_mask = v_U_k <= 0
            mixed_mask = ~(pos_mask | neg_mask)

            L_ctrl += torch.where(pos_mask, v_L_k * u_min[k], torch.zeros_like(v_L_k))
            L_ctrl += torch.where(neg_mask, v_U_k * u_max[k], torch.zeros_like(v_L_k))

            mixed_vals = torch.minimum(v_L_k * u_max[k], v_U_k * u_min[k])
            L_ctrl += torch.where(mixed_mask, mixed_vals, torch.zeros_like(v_L_k))

    # Add class-K term
    if h_lb_lbp is not None:
        alpha_l = dynamics_model.alpha_function(h_lb_lbp)
    else:
        alpha_l = dynamics_model.alpha_function(h_lb_ibp.squeeze(-1))
    L_total = L_drift + L_ctrl + alpha_l

    return L_total


# =============================================================================
# Main Loss Functions
# =============================================================================

def compute_repair_loss_and_grad_unsafe_lbp(
    model: nn.Module,
    dynamics_model,
    unsafe_simplices: List,
    lbp_linearizer,
    margin: float = 0.0,
    beta: float = 5.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    """
    使用 LBP 上界计算 Unsafe 区域违规的修复损失和梯度。

    对于 F_h_positive_in_unsafe 区域（障碍区中 h >= 0 的违规）:
    - 计算 h 的 LBP 上界 h_ub
    - loss = softplus(h_ub + margin)，推动 h_ub + margin <= 0

    LBP 上界是通过 CrownPartialLinearization 计算的，可以直接对网络参数求导。

    Args:
        model: BarrierNN 网络
        dynamics_model: 动力学系统
        unsafe_simplices: Unsafe 违规区域的单纯形列表
        lbp_linearizer: CrownPartialLinearization 实例
        margin: h 值的容差
        beta: softplus 的 beta 参数
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出

    Returns:
        total_loss: 平均损失值
        g_raw: 修复梯度 [num_params]
    """
    from lbp_neural_cbf.regions import SimplicialRegion

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    if len(unsafe_simplices) == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    # 转换为单纯形区域对象（SimplicialRegion 期望 numpy.ndarray）
    regions = []
    for verts in unsafe_simplices:
        if isinstance(verts, np.ndarray):
            verts_np = verts.astype(np.float32)
        else:
            verts_np = verts.detach().cpu().numpy().astype(np.float32)
        regions.append(SimplicialRegion(verts_np, output_dim=None))

    # 使用 LBP 计算 h 上界
    lbp_linearizer.compute_network_bounds(regions)
    h_lb_all, h_ub_all = lbp_linearizer.get_network_output_bounds()
    h_ub = h_ub_all.reshape(len(regions), -1)[:, 0]

    # 计算损失: softplus(h_ub + margin)
    loss_unsafe = F.softplus(h_ub + margin, beta=beta)

    if not torch.isfinite(loss_unsafe).all():
        if verbose:
            print(f"  [警告] unsafe LBP 损失存在 NaN/Inf，跳过")
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    # 梯度计算
    model.zero_grad()
    loss_unsafe.sum().backward()

    grad_this = torch.cat([
        p.grad.flatten() if p.grad is not None
        else torch.zeros(p.numel(), dtype=dtype, device=device)
        for p in model.parameters()
    ])
    grad_this = torch.nan_to_num(grad_this, nan=0.0, posinf=0.0, neginf=0.0)

    grad_norm = grad_this.norm().item()
    if grad_norm > grad_clip_norm:
        grad_this = grad_this * (grad_clip_norm / grad_norm)

    if verbose:
        print(f"  [Unsafe LBP] loss={loss_unsafe.mean().item():.6f}, |g|={grad_norm:.4f}, n={len(regions)}")

    return loss_unsafe.mean().item(), grad_this


def compute_repair_loss_and_grad_safe_ibp(
    model: nn.Module,
    dynamics_model,
    safe_simplices: List,
    lbp_linearizer,
    ibp_calculator,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    """
    使用 IBP 下界计算 Safe 区域 CBF 违规的修复损失和梯度。

    对于 F_safe_cbf_violation / F_depth_limit_reached 区域（CBF 条件违规）:
    - 计算 CBF 条件的 IBP 下界 min_L
    - alpha 项使用 LBP 的 h_lb (更准确)
    - loss = softplus(cbf_margin - min_L)，推动 min_L >= cbf_margin

    Args:
        model: BarrierNN 网络
        dynamics_model: 动力学系统
        safe_simplices: CBF 违规区域的单纯形列表
        lbp_linearizer: CrownPartialLinearization 实例，用于 alpha 项的 h_lb
        ibp_calculator: IBPNetworkBoundCalculator 实例，用于 min_L
        cbf_margin: CBF 条件的容差
        beta: softplus 的 beta 参数
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出

    Returns:
        total_loss: 平均损失值
        g_raw: 修复梯度 [num_params]
    """
    from lbp_neural_cbf.regions import SimplicialRegion

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    if len(safe_simplices) == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    # 转换为单纯形区域对象（SimplicialRegion 期望 numpy.ndarray）
    regions = []
    for verts in safe_simplices:
        if isinstance(verts, np.ndarray):
            verts_np = verts.astype(np.float32)
        else:
            verts_np = verts.detach().cpu().numpy().astype(np.float32)
        regions.append(SimplicialRegion(verts_np, output_dim=None))

    # Step 1: 使用 LBP 计算 h_lb，用于 alpha 项（更准确）
    lbp_linearizer.compute_network_bounds(regions)
    h_lb_lbp, h_ub_lbp = lbp_linearizer.get_network_output_bounds()
    h_lb_lbp = h_lb_lbp.reshape(len(regions), -1)[:, 0]

    # Step 2: 使用 IBP 计算 min_L (CBF 条件下界)
    ibp_calculator.ibp_forward(regions)

    # 计算 min_L，使用 LBP 的 h_lb 用于 alpha 项
    min_L = _compute_min_L_ibp(regions, dynamics_model, ibp_calculator, device, dtype, h_lb_lbp=h_lb_lbp)
    min_L = min_L.reshape(-1)

    # 计算损失: softplus(cbf_margin - min_L)
    loss_cbf = F.softplus(cbf_margin - min_L, beta=beta)

    if not torch.isfinite(loss_cbf).all():
        if verbose:
            print(f"  [警告] safe IBP 损失存在 NaN/Inf，跳过")
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    # 梯度计算
    model.zero_grad()
    loss_cbf.sum().backward()

    grad_this = torch.cat([
        p.grad.flatten() if p.grad is not None
        else torch.zeros(p.numel(), dtype=dtype, device=device)
        for p in model.parameters()
    ])
    grad_this = torch.nan_to_num(grad_this, nan=0.0, posinf=0.0, neginf=0.0)

    grad_norm = grad_this.norm().item()
    if grad_norm > grad_clip_norm:
        grad_this = grad_this * (grad_clip_norm / grad_norm)

    if verbose:
        print(f"  [Safe IBP] loss={loss_cbf.mean().item():.6f}, |g|={grad_norm:.4f}, n={len(regions)}")

    return loss_cbf.mean().item(), grad_this


def compute_repair_loss_and_grad_bounds(
    model: nn.Module,
    dynamics_model,
    safe_simplices: List,
    unsafe_simplices: List,
    lbp_linearizer,
    ibp_calculator,
    margin: float = 0.0,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    """
    综合修复损失计算：使用 LBP 上界处理 unsafe，IBP 下界处理 safe。

    - Unsafe: LBP 上界 (h_ub)，更紧且稳定
    - Safe: IBP 下界 (min_L)，避免 LBP 的梯度 NaN 问题

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        safe_simplices: CBF 违规区域的单纯形列表
        unsafe_simplices: 障碍区违规区域的单纯形列表
        lbp_linearizer: CrownPartialLinearization 实例
        ibp_calculator: IBPNetworkBoundCalculator 实例
        margin: h 值的容差（用于 unsafe）
        cbf_margin: CBF 条件的容差（用于 safe）
        beta: softplus 的 beta 参数
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出

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

    # 处理 Unsafe 区域 (使用 LBP 上界)
    if len(unsafe_simplices) > 0:
        loss_unsafe, grad_unsafe = compute_repair_loss_and_grad_unsafe_lbp(
            model, dynamics_model, unsafe_simplices, lbp_linearizer,
            margin=margin, beta=beta, grad_clip_norm=grad_clip_norm, verbose=verbose
        )
        if torch.isfinite(grad_unsafe).all():
            g_raw.add_(grad_unsafe)
            total_loss_sum += loss_unsafe * len(unsafe_simplices)
            n_valid += 1

    # 处理 Safe 区域 (使用 IBP 下界，alpha 项用 LBP)
    if len(safe_simplices) > 0:
        loss_safe, grad_safe = compute_repair_loss_and_grad_safe_ibp(
            model, dynamics_model, safe_simplices, lbp_linearizer, ibp_calculator,
            cbf_margin=cbf_margin, beta=beta, grad_clip_norm=grad_clip_norm, verbose=verbose
        )
        if torch.isfinite(grad_safe).all():
            g_raw.add_(grad_safe)
            total_loss_sum += loss_safe * len(safe_simplices)
            n_valid += 1

    if n_valid == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    total_loss = total_loss_sum / n_valid
    grad_norm = g_raw.norm().item()

    if verbose:
        print(f"  [修复损失 Bounds] total_loss={total_loss:.6f}, |g|={grad_norm:.4f}, "
              f"unsafe={len(unsafe_simplices)}, safe={len(safe_simplices)}")

    return total_loss, g_raw


def simple_gradient_update(
    model: nn.Module,
    grad: torch.Tensor,
    lr: float,
) -> float:
    """
    简单的梯度更新（不使用 QP 投影）。

    Args:
        model: 神经网络
        grad: 梯度向量 [num_params]
        lr: 学习率

    Returns:
        update_norm: 更新向量的范数
    """
    params = [p for p in model.parameters() if p.requires_grad]
    theta_old = torch.nn.utils.parameters_to_vector(params)

    theta_new = theta_old - lr * grad
    torch.nn.utils.vector_to_parameters(theta_new, params)

    update_norm = (theta_old - theta_new).norm().item()
    return update_norm
