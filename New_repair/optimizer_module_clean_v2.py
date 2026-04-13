"""
optimizer_module_clean_v2.py: 修复损失计算 (完整版)

包含 5 种损失函数（与 train_cbf.py 一致）:
1. L_safe: 安全集内 h(x) ≥ safe_margin, softplus(safe_margin - h)
2. L_unsafe: 不安全集内 h(x) ≤ -unsafe_margin, softplus(h + unsafe_margin)
3. L_unsafe_max: 额外惩罚 unsafe 区 h 值最大的 1% 点, top-1% softplus(h_topk + unsafe_margin)
4. L_cbf: CBF 前向不变性条件, softplus(cbf_margin - CBF_condition)
5. L_boundary: 边界点 h(x) ≈ 0, softplus(h_boundary + unsafe_margin)
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cbf_condition_simple(
    model: nn.Module,
    dynamics_model,
    x: torch.Tensor,
    translator=None,
) -> torch.Tensor:
    """
    计算 CBF 条件值。

    cbf(x) = ∇h·f(x) + sup_u[∇h·g(x)·u] + α·h(x)

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        x: 状态点 [batch, D], requires_grad=True
        translator: TorchTranslator

    Returns:
        cbf_values: [batch]
    """
    # h(x)
    h = model(x).squeeze(-1)  # [batch]

    # ∇h(x)
    grad_h = torch.autograd.grad(
        outputs=h,
        inputs=x,
        grad_outputs=torch.ones_like(h, device=x.device),
        create_graph=False,
        retain_graph=True,
    )[0]  # [batch, D]

    # f(x)
    if translator is None:
        x1 = x[..., 0]
        x2 = x[..., 1]
        D = x.shape[1]
        if D == 2 and dynamics_model.control_dim == 0:
            dx1 = x2
            dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
            f_x = torch.stack([dx1, dx2], dim=-1)
        else:
            raise NotImplementedError("需要 translator 进行通用 compute_f")
    else:
        f_x = dynamics_model.compute_f(x, translator)  # [batch, D]

    # ∇h·f
    grad_h_dot_f = (grad_h * f_x).sum(dim=-1)  # [batch]

    # sup_u 项（仅对有控制系统）
    if dynamics_model.control_dim > 0:
        if translator is None:
            raise ValueError("有控制系统需要提供 TorchTranslator")
        g_x = dynamics_model.compute_g(x, translator)  # [batch, D, m]
        u_min = torch.tensor(dynamics_model.u_min, device=x.device, dtype=x.dtype)
        u_max = torch.tensor(dynamics_model.u_max, device=x.device, dtype=x.dtype)
        grad_h_g = torch.einsum('bd,bdm->bm', grad_h, g_x)
        term_max = grad_h_g * u_max.unsqueeze(0)
        term_min = grad_h_g * u_min.unsqueeze(0)
        sup_u = torch.sum(torch.maximum(term_max, term_min), dim=-1)
    else:
        sup_u = torch.zeros_like(h)

    # α·h
    alpha_h = dynamics_model.alpha_function(h, translator)

    # CBF 条件
    cbf = grad_h_dot_f + sup_u + alpha_h
    return cbf


def compute_cbf_loss_modified(
    model: nn.Module,
    dynamics_model,
    safe_points: torch.Tensor,
    unsafe_points: torch.Tensor,
    failed_safe_points: torch.Tensor,
    boundary_points: torch.Tensor = None,
    translator=None,
    lambda_safe: float = 1.0,
    lambda_unsafe: float = 10.0,
    lambda_unsafe_max: float = 5.0,
    lambda_cbf: float = 5.0,
    lambda_boundary: float = 0.1,
    safe_margin: float = 0.01,
    unsafe_margin: float = 0.01,
    cbf_margin: float = 0.0,
    beta_softplus: float = 5.0,
    beta_safe: float = 100.0,
    beta_boundary: float = 100.0,
    verbose: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    计算完整的 CBF 修复损失（5 种 loss 组合）。

    Loss 公式:
    1. L_safe: softplus(safe_margin - h) in safe regions
    2. L_unsafe: softplus(h + unsafe_margin) in unsafe regions
    3. L_unsafe_max: top-1% softplus(h_topk + unsafe_margin)
    4. L_cbf: softplus(cbf_margin - CBF_condition) where h >= 0
    5. L_boundary: softplus(h_boundary + unsafe_margin) at boundary

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        safe_points: 安全集采样点 [N_safe, D]
        unsafe_points: 不安全集采样点 [N_unsafe, D]
        failed_safe_points: CBF 违规的安全区点 [N_fail, D]
        boundary_points: 边界采样点 [N_bndry, D]（可选）
        translator: TorchTranslator
        lambda_*: 各 loss 的权重
        safe_margin: 安全 margin (h >= safe_margin)
        unsafe_margin: 不安全 margin (h <= -unsafe_margin)
        cbf_margin: CBF 条件 margin
        beta_softplus: softplus beta 参数
        beta_safe: safe loss 的 beta
        beta_boundary: boundary loss 的 beta
        verbose: 诊断输出

    Returns:
        (total_loss, loss_dict)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ========== L_safe: 安全集 h(x) >= safe_margin ==========
    safe_loss = torch.tensor(0.0, device=device)
    if safe_points.shape[0] > 0:
        x_safe = safe_points.detach().clone().requires_grad_(True)
        h_safe = model(x_safe).squeeze(-1)
        safe_violations = F.softplus(safe_margin - h_safe, beta=beta_safe)
        safe_loss = torch.mean(safe_violations)

    # ========== L_unsafe: 不安全集 h(x) <= -unsafe_margin ==========
    unsafe_loss = torch.tensor(0.0, device=device)
    h_unsafe_max_loss = torch.tensor(0.0, device=device)

    if unsafe_points.shape[0] > 0:
        x_unsafe = unsafe_points.detach().clone().requires_grad_(True)
        h_unsafe = model(x_unsafe).squeeze(-1)

        # L_unsafe: mean softplus(h + unsafe_margin)
        unsafe_violations = F.softplus(h_unsafe + unsafe_margin, beta=beta_softplus)
        unsafe_loss = torch.mean(unsafe_violations)

        # L_unsafe_max: top-1% softplus(h_topk + unsafe_margin)
        n_unsafe = h_unsafe.shape[0]
        top_k = max(1, int(0.01 * n_unsafe))
        if top_k > 0:
            h_unsafe_topk, _ = torch.topk(h_unsafe, k=top_k)
            h_unsafe_max_violations = F.softplus(h_unsafe_topk + unsafe_margin, beta=beta_softplus)
            h_unsafe_max_loss = torch.mean(h_unsafe_max_violations)

    # ========== L_cbf: CBF 前向不变性条件 ==========
    cbf_loss = torch.tensor(0.0, device=device)

    if failed_safe_points.shape[0] > 0:
        x_fail = failed_safe_points.detach().clone().requires_grad_(True)

        # h(x)
        h_fail = model(x_fail).squeeze(-1)

        # ∇h(x)
        grad_h = torch.autograd.grad(
            outputs=h_fail,
            inputs=x_fail,
            grad_outputs=torch.ones_like(h_fail, device=device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # f(x)
        if translator is None:
            x1 = x_fail[..., 0]
            x2 = x_fail[..., 1]
            D = x_fail.shape[1]
            if D == 2 and dynamics_model.control_dim == 0:
                dx1 = x2
                dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
                f_x = torch.stack([dx1, dx2], dim=-1)
            else:
                raise NotImplementedError("需要 translator")
        else:
            f_x = dynamics_model.compute_f(x_fail, translator)

        # ∇h·f
        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)

        # sup_u 项
        if dynamics_model.control_dim > 0:
            if translator is None:
                raise NotImplementedError("有控制系统需要 translator")
            g_x = dynamics_model.compute_g(x_fail, translator)
            u_min = torch.tensor(dynamics_model.u_min, device=device, dtype=x_fail.dtype)
            u_max = torch.tensor(dynamics_model.u_max, device=device, dtype=x_fail.dtype)
            grad_h_g = torch.einsum('bd,bdm->bm', grad_h, g_x)
            control_term = torch.sum(
                torch.maximum(grad_h_g * u_max.unsqueeze(0), grad_h_g * u_min.unsqueeze(0)),
                dim=-1
            )
        else:
            control_term = torch.zeros_like(h_fail)

        # α·h
        alpha_h = dynamics_model.alpha_function(h_fail, translator)

        # CBF 条件
        cbf_condition = grad_h_dot_f + control_term + alpha_h

        # 仅在 h >= -unsafe_margin 区域惩罚（屏障正区域）
        h_positive_mask = h_fail >= -unsafe_margin
        if h_positive_mask.any():
            cbf_violations = F.softplus(cbf_margin - cbf_condition[h_positive_mask], beta=beta_softplus)
            cbf_loss = torch.mean(cbf_violations)

    # ========== L_boundary: 边界点 h(x) ≈ 0 ==========
    boundary_loss = torch.tensor(0.0, device=device)

    if boundary_points is not None and boundary_points.shape[0] > 0:
        x_bndry = boundary_points.detach().clone().requires_grad_(True)
        h_bndry = model(x_bndry).squeeze(-1)
        boundary_violations = F.softplus(h_bndry + unsafe_margin, beta=beta_boundary)
        boundary_loss = torch.mean(boundary_violations)

    # ========== 总损失 ==========
    total_loss = (
        lambda_safe * safe_loss +
        lambda_unsafe * unsafe_loss +
        lambda_unsafe_max * h_unsafe_max_loss +
        lambda_cbf * cbf_loss +
        lambda_boundary * boundary_loss
    )

    loss_dict = {
        'total_loss': total_loss.item(),
        'safe_loss': safe_loss.item(),
        'unsafe_loss': unsafe_loss.item(),
        'unsafe_max_loss': h_unsafe_max_loss.item(),
        'cbf_loss': cbf_loss.item(),
        'boundary_loss': boundary_loss.item(),
    }

    if verbose:
        print(f"  [损失分解] total={total_loss.item():.6f}, "
              f"safe={safe_loss.item():.6f}, unsafe={unsafe_loss.item():.6f}, "
              f"unsafe_max={h_unsafe_max_loss.item():.6f}, "
              f"cbf={cbf_loss.item():.6f}, boundary={boundary_loss.item():.6f}")

    return total_loss, loss_dict


def compute_repair_loss_and_grad_modified(
    model: nn.Module,
    dynamics_model,
    safe_worst_points: torch.Tensor,
    unsafe_worst_points: torch.Tensor,
    cbf_worst_points: torch.Tensor,
    translator=None,
    lambda_safe: float = 1.0,
    lambda_unsafe: float = 10.0,
    lambda_unsafe_max: float = 5.0,
    lambda_cbf: float = 5.0,
    lambda_boundary: float = 0.1,
    safe_margin: float = 0.01,
    unsafe_margin: float = 0.01,
    cbf_margin: float = 0.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor, Dict[str, float]]:
    """
    计算修复损失并反向传播得到梯度 g_F。

    与 compute_cbf_loss_modified 的区别：
    - 接受最坏点作为输入（而非批量采样点）
    - 自动处理梯度计算和参数更新

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        safe_worst_points: V_safe 最坏点 [N_safe, D]
        unsafe_worst_points: F_h 最坏点 [N_unsafe, D]
        cbf_worst_points: CBF 违规最坏点 [N_cbf, D]
        translator: TorchTranslator
        lambda_*: 各 loss 权重
        safe_margin, unsafe_margin, cbf_margin: 各 margin
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出

    Returns:
        (total_loss, g_raw, loss_dict)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    g_raw = torch.zeros(num_params, dtype=dtype, device=device)

    # ---------- L_safe ----------
    safe_loss = torch.tensor(0.0, device=device)
    if safe_worst_points.shape[0] > 0:
        x_safe = safe_worst_points.detach().clone().to(device).requires_grad_(True)
        h_safe = model(x_safe).squeeze(-1)
        safe_loss = F.softplus(safe_margin - h_safe, beta=100.0).mean()
        model.zero_grad()
        safe_loss.backward(retain_graph=True)
        grad_safe = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        grad_safe = torch.nan_to_num(grad_safe, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_safe)

    # ---------- L_unsafe ----------
    unsafe_loss = torch.tensor(0.0, device=device)
    unsafe_max_loss = torch.tensor(0.0, device=device)

    if unsafe_worst_points.shape[0] > 0:
        x_unsafe = unsafe_worst_points.detach().clone().to(device).requires_grad_(True)
        h_unsafe = model(x_unsafe).squeeze(-1)

        # L_unsafe
        unsafe_loss = F.softplus(h_unsafe + unsafe_margin, beta=5.0).mean()

        # L_unsafe_max: top 1%
        n_unsafe = h_unsafe.shape[0]
        top_k = max(1, int(0.01 * n_unsafe))
        if top_k > 0:
            h_unsafe_topk, _ = torch.topk(h_unsafe, k=top_k)
            unsafe_max_loss = F.softplus(h_unsafe_topk + unsafe_margin, beta=5.0).mean()

        model.zero_grad()
        (unsafe_loss + unsafe_max_loss).backward(retain_graph=True)
        grad_unsafe = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        grad_unsafe = torch.nan_to_num(grad_unsafe, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_unsafe)

    # ---------- L_cbf ----------
    cbf_loss = torch.tensor(0.0, device=device)

    if cbf_worst_points.shape[0] > 0:
        x_cbf = cbf_worst_points.detach().clone().to(device).requires_grad_(True)

        h_cbf = model(x_cbf).squeeze(-1)
        grad_h = torch.autograd.grad(
            outputs=h_cbf, inputs=x_cbf,
            grad_outputs=torch.ones_like(h_cbf, device=device),
            create_graph=True, retain_graph=True,
        )[0]

        if translator is None:
            x1 = x_cbf[..., 0]
            x2 = x_cbf[..., 1]
            D = x_cbf.shape[1]
            if D == 2 and dynamics_model.control_dim == 0:
                dx1 = x2
                dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
                f_x = torch.stack([dx1, dx2], dim=-1)
            else:
                raise NotImplementedError("需要 translator")
        else:
            f_x = dynamics_model.compute_f(x_cbf, translator)

        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)
        alpha_h = dynamics_model.alpha_function(h_cbf, translator)
        cbf_condition = grad_h_dot_f + alpha_h

        # h >= -unsafe_margin 区域
        h_pos_mask = h_cbf >= -unsafe_margin
        if h_pos_mask.any():
            cbf_loss = F.softplus(cbf_margin - cbf_condition[h_pos_mask], beta=5.0).mean()

        model.zero_grad()
        if cbf_loss.item() > 0:
            grad_cbf = torch.autograd.grad(
                outputs=cbf_loss,
                inputs=model.parameters(),
                retain_graph=False,
            )
            grad_cbf = torch.cat([g.flatten() for g in grad_cbf if g is not None])
            grad_cbf = torch.nan_to_num(grad_cbf, nan=0.0, posinf=0.0, neginf=0.0)
            if not (torch.isnan(grad_cbf).any() or torch.isinf(grad_cbf).any()):
                g_raw.add_(grad_cbf)

    # ---------- 总损失和梯度裁剪 ----------
    total_loss = (
        lambda_safe * safe_loss +
        lambda_unsafe * unsafe_loss +
        lambda_unsafe_max * unsafe_max_loss +
        lambda_cbf * cbf_loss
    )

    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)

    loss_dict = {
        'total_loss': total_loss.item(),
        'safe_loss': safe_loss.item(),
        'unsafe_loss': unsafe_loss.item(),
        'unsafe_max_loss': unsafe_max_loss.item(),
        'cbf_loss': cbf_loss.item(),
        'boundary_loss': 0.0,
    }

    if verbose:
        print(f"  [损失分解] total={total_loss.item():.6f}, "
              f"safe={safe_loss.item():.6f}, unsafe={unsafe_loss.item():.6f}, "
              f"unsafe_max={unsafe_max_loss.item():.6f}, cbf={cbf_loss.item():.6f}")

    return total_loss, g_raw, loss_dict


def compute_combined_loss_and_grad(
    model: nn.Module,
    dynamics_model,
    # 验证区域最坏点
    safe_worst_points: torch.Tensor,
    unsafe_worst_points: torch.Tensor,
    cbf_worst_points: torch.Tensor,
    # 训练过程动态采样点（可选）
    train_safe_points: torch.Tensor = None,
    train_unsafe_points: torch.Tensor = None,
    train_boundary_points: torch.Tensor = None,
    # translator
    translator=None,
    # 损失权重
    lambda_safe: float = 1.0,
    lambda_unsafe: float = 10.0,
    lambda_unsafe_max: float = 5.0,
    lambda_cbf: float = 5.0,
    lambda_boundary: float = 0.1,
    # 损失权重（训练部分）
    lambda_train_safe: float = 1.0,
    lambda_train_unsafe: float = 10.0,
    lambda_train_unsafe_max: float = 5.0,
    lambda_train_cbf: float = 5.0,
    lambda_train_boundary: float = 0.1,
    # margins
    safe_margin: float = 0.01,
    unsafe_margin: float = 0.01,
    cbf_margin: float = 0.0,
    # 梯度裁剪
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor, Dict[str, float]]:
    """
    组合损失计算：验证区域最坏点 loss + 训练过程动态采样 loss。

    两种 loss 的梯度合并后用于参数更新。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        # 验证区域最坏点
        safe_worst_points: V_safe 最坏点 [N_safe, D]
        unsafe_worst_points: F_h 最坏点 [N_unsafe, D]
        cbf_worst_points: CBF 违规最坏点 [N_cbf, D]
        # 训练过程动态采样点
        train_safe_points: 训练时采样的安全区点 [N_train_safe, D]
        train_unsafe_points: 训练时采样的不安全区点 [N_train_unsafe, D]
        train_boundary_points: 训练时采样的边界点 [N_train_bndry, D]
        # 各损失权重（验证部分）
        lambda_safe, lambda_unsafe, lambda_unsafe_max, lambda_cbf, lambda_boundary
        # 各损失权重（训练部分）
        lambda_train_safe, lambda_train_unsafe, lambda_train_unsafe_max,
        lambda_train_cbf, lambda_train_boundary
        # margins
        safe_margin, unsafe_margin, cbf_margin
        # 梯度裁剪
        grad_clip_norm
        verbose

    Returns:
        (total_loss, g_raw, loss_dict)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_loss_sum = 0.0
    n_loss_terms = 0

    loss_dict = {
        'total_loss': 0.0,
        # 验证部分
        'safe_loss': 0.0,
        'unsafe_loss': 0.0,
        'unsafe_max_loss': 0.0,
        'cbf_loss': 0.0,
        'boundary_loss': 0.0,
        # 训练部分
        'train_safe_loss': 0.0,
        'train_unsafe_loss': 0.0,
        'train_unsafe_max_loss': 0.0,
        'train_cbf_loss': 0.0,
        'train_boundary_loss': 0.0,
    }

    # ================================================================
    # PART 1: 验证区域最坏点损失（与 compute_repair_loss_and_grad_modified 相同）
    # ================================================================

    # ---------- L_safe (验证) ----------
    if safe_worst_points.shape[0] > 0:
        x_safe = safe_worst_points.detach().clone().to(device).requires_grad_(True)
        h_safe = model(x_safe).squeeze(-1)
        safe_loss = F.softplus(safe_margin - h_safe, beta=100.0).mean()

        safe_grads = torch.autograd.grad(
            outputs=safe_loss,
            inputs=model.parameters(),
            retain_graph=True,
        )
        grad_safe = torch.cat([g.flatten() for g in safe_grads if g is not None])
        grad_safe = torch.nan_to_num(grad_safe, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_safe)

        total_loss_sum += lambda_safe * safe_loss.item()
        loss_dict['safe_loss'] = safe_loss.item()
        n_loss_terms += 1

    # ---------- L_unsafe (验证) ----------
    if unsafe_worst_points.shape[0] > 0:
        x_unsafe = unsafe_worst_points.detach().clone().to(device).requires_grad_(True)
        h_unsafe = model(x_unsafe).squeeze(-1)

        unsafe_loss = F.softplus(h_unsafe + unsafe_margin, beta=5.0).mean()

        n_unsafe = h_unsafe.shape[0]
        top_k = max(1, int(0.01 * n_unsafe))
        if top_k > 0:
            h_unsafe_topk, _ = torch.topk(h_unsafe, k=top_k)
            unsafe_max_loss = F.softplus(h_unsafe_topk + unsafe_margin, beta=5.0).mean()
        else:
            unsafe_max_loss = torch.tensor(0.0, device=device)

        combined_unsafe_loss = unsafe_loss + unsafe_max_loss

        unsafe_grads = torch.autograd.grad(
            outputs=combined_unsafe_loss,
            inputs=model.parameters(),
            retain_graph=True,
        )
        grad_unsafe = torch.cat([g.flatten() for g in unsafe_grads if g is not None])
        grad_unsafe = torch.nan_to_num(grad_unsafe, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_unsafe)

        total_loss_sum += lambda_unsafe * unsafe_loss.item() + lambda_unsafe_max * unsafe_max_loss.item()
        loss_dict['unsafe_loss'] = unsafe_loss.item()
        loss_dict['unsafe_max_loss'] = unsafe_max_loss.item()
        n_loss_terms += 1

    # ---------- L_cbf (验证) ----------
    if cbf_worst_points.shape[0] > 0:
        x_cbf = cbf_worst_points.detach().clone().to(device).requires_grad_(True)

        h_cbf = model(x_cbf).squeeze(-1)
        grad_h = torch.autograd.grad(
            outputs=h_cbf, inputs=x_cbf,
            grad_outputs=torch.ones_like(h_cbf, device=device),
            create_graph=True, retain_graph=True,
        )[0]

        if translator is None:
            x1 = x_cbf[..., 0]
            x2 = x_cbf[..., 1]
            D = x_cbf.shape[1]
            if D == 2 and dynamics_model.control_dim == 0:
                dx1 = x2
                dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
                f_x = torch.stack([dx1, dx2], dim=-1)
            else:
                raise NotImplementedError("需要 translator")
        else:
            f_x = dynamics_model.compute_f(x_cbf, translator)

        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)
        alpha_h = dynamics_model.alpha_function(h_cbf, translator)
        cbf_condition = grad_h_dot_f + alpha_h

        h_pos_mask = h_cbf >= -unsafe_margin
        cbf_loss = torch.tensor(0.0, device=device)
        if h_pos_mask.any():
            cbf_loss = F.softplus(cbf_margin - cbf_condition[h_pos_mask], beta=5.0).mean()

        if cbf_loss.item() > 0:
            cbf_grads = torch.autograd.grad(
                outputs=cbf_loss,
                inputs=model.parameters(),
                retain_graph=False,
            )
            grad_cbf = torch.cat([g.flatten() for g in cbf_grads if g is not None])
            grad_cbf = torch.nan_to_num(grad_cbf, nan=0.0, posinf=0.0, neginf=0.0)
            if not (torch.isnan(grad_cbf).any() or torch.isinf(grad_cbf).any()):
                g_raw.add_(grad_cbf)

            total_loss_sum += lambda_cbf * cbf_loss.item()
            loss_dict['cbf_loss'] = cbf_loss.item()
            n_loss_terms += 1

    # ================================================================
    # PART 2: 训练过程动态采样损失
    # 使用 dynamics_model.safe_set.contains() 识别点所属区域
    # ================================================================

    # ---------- L_safe (训练) ----------
    if train_safe_points is not None and train_safe_points.shape[0] > 0:
        x_train_safe = train_safe_points.detach().clone().to(device).requires_grad_(True)
        h_train_safe = model(x_train_safe).squeeze(-1)
        train_safe_loss = F.softplus(safe_margin - h_train_safe, beta=100.0).mean()

        train_safe_grads = torch.autograd.grad(
            outputs=train_safe_loss,
            inputs=model.parameters(),
            retain_graph=True,
        )
        grad_train_safe = torch.cat([g.flatten() for g in train_safe_grads if g is not None])
        grad_train_safe = torch.nan_to_num(grad_train_safe, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_train_safe)

        total_loss_sum += lambda_train_safe * train_safe_loss.item()
        loss_dict['train_safe_loss'] = train_safe_loss.item()
        n_loss_terms += 1

    # ---------- L_unsafe (训练) ----------
    if train_unsafe_points is not None and train_unsafe_points.shape[0] > 0:
        x_train_unsafe = train_unsafe_points.detach().clone().to(device).requires_grad_(True)
        h_train_unsafe = model(x_train_unsafe).squeeze(-1)

        train_unsafe_loss = F.softplus(h_train_unsafe + unsafe_margin, beta=5.0).mean()

        n_train_unsafe = h_train_unsafe.shape[0]
        top_k = max(1, int(0.01 * n_train_unsafe))
        if top_k > 0:
            h_train_unsafe_topk, _ = torch.topk(h_train_unsafe, k=top_k)
            train_unsafe_max_loss = F.softplus(h_train_unsafe_topk + unsafe_margin, beta=5.0).mean()
        else:
            train_unsafe_max_loss = torch.tensor(0.0, device=device)

        combined_train_unsafe_loss = train_unsafe_loss + train_unsafe_max_loss

        train_unsafe_grads = torch.autograd.grad(
            outputs=combined_train_unsafe_loss,
            inputs=model.parameters(),
            retain_graph=True,
        )
        grad_train_unsafe = torch.cat([g.flatten() for g in train_unsafe_grads if g is not None])
        grad_train_unsafe = torch.nan_to_num(grad_train_unsafe, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_train_unsafe)

        total_loss_sum += lambda_train_unsafe * train_unsafe_loss.item() + lambda_train_unsafe_max * train_unsafe_max_loss.item()
        loss_dict['train_unsafe_loss'] = train_unsafe_loss.item()
        loss_dict['train_unsafe_max_loss'] = train_unsafe_max_loss.item()
        n_loss_terms += 1

    # ---------- L_cbf (训练) ----------
    if train_unsafe_points is not None and train_unsafe_points.shape[0] > 0:
        x_train_cbf = train_unsafe_points.detach().clone().to(device).requires_grad_(True)

        h_train_cbf = model(x_train_cbf).squeeze(-1)
        grad_h_train = torch.autograd.grad(
            outputs=h_train_cbf, inputs=x_train_cbf,
            grad_outputs=torch.ones_like(h_train_cbf, device=device),
            create_graph=True, retain_graph=True,
        )[0]

        if translator is None:
            x1 = x_train_cbf[..., 0]
            x2 = x_train_cbf[..., 1]
            D = x_train_cbf.shape[1]
            if D == 2 and dynamics_model.control_dim == 0:
                dx1 = x2
                dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
                f_x_train = torch.stack([dx1, dx2], dim=-1)
            else:
                f_x_train = None
        else:
            f_x_train = dynamics_model.compute_f(x_train_cbf, translator)

        if f_x_train is not None:
            grad_h_dot_f_train = (grad_h_train * f_x_train).sum(dim=-1)
            alpha_h_train = dynamics_model.alpha_function(h_train_cbf, translator)
            cbf_condition_train = grad_h_dot_f_train + alpha_h_train

            h_pos_mask_train = h_train_cbf >= -unsafe_margin
            train_cbf_loss = torch.tensor(0.0, device=device)
            if h_pos_mask_train.any():
                train_cbf_loss = F.softplus(cbf_margin - cbf_condition_train[h_pos_mask_train], beta=5.0).mean()

            if train_cbf_loss.item() > 0:
                train_cbf_grads = torch.autograd.grad(
                    outputs=train_cbf_loss,
                    inputs=model.parameters(),
                    retain_graph=False,
                )
                grad_train_cbf = torch.cat([g.flatten() for g in train_cbf_grads if g is not None])
                grad_train_cbf = torch.nan_to_num(grad_train_cbf, nan=0.0, posinf=0.0, neginf=0.0)
                if not (torch.isnan(grad_train_cbf).any() or torch.isinf(grad_train_cbf).any()):
                    g_raw.add_(grad_train_cbf)

                    total_loss_sum += lambda_train_cbf * train_cbf_loss.item()
                    loss_dict['train_cbf_loss'] = train_cbf_loss.item()
                    n_loss_terms += 1

    # ---------- L_boundary (训练) ----------
    if train_boundary_points is not None and train_boundary_points.shape[0] > 0:
        x_train_bndry = train_boundary_points.detach().clone().to(device).requires_grad_(True)
        h_train_bndry = model(x_train_bndry).squeeze(-1)
        train_boundary_loss = F.softplus(h_train_bndry + unsafe_margin, beta=100.0).mean()

        train_bndry_grads = torch.autograd.grad(
            outputs=train_boundary_loss,
            inputs=model.parameters(),
            retain_graph=False,
        )
        grad_train_bndry = torch.cat([g.flatten() for g in train_bndry_grads if g is not None])
        grad_train_bndry = torch.nan_to_num(grad_train_bndry, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_train_bndry)

        total_loss_sum += lambda_train_boundary * train_boundary_loss.item()
        loss_dict['train_boundary_loss'] = train_boundary_loss.item()
        n_loss_terms += 1

    # ================================================================
    # 梯度裁剪
    # ================================================================
    total_loss = total_loss_sum / max(n_loss_terms, 1)

    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)

    loss_dict['total_loss'] = total_loss

    if verbose:
        print(f"  [组合损失] total={total_loss:.6f}, "
              f"safe={loss_dict['safe_loss']:.6f}, unsafe={loss_dict['unsafe_loss']:.6f}, "
              f"unsafe_max={loss_dict['unsafe_max_loss']:.6f}, cbf={loss_dict['cbf_loss']:.6f}, "
              f"train_safe={loss_dict['train_safe_loss']:.6f}, "
              f"train_unsafe={loss_dict['train_unsafe_loss']:.6f}, "
              f"train_cbf={loss_dict['train_cbf_loss']:.6f}")

    return total_loss, g_raw, loss_dict
