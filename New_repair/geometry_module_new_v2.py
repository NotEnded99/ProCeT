"""
几何模块 v2: 对抗采样 (Adversarial Sampling) 与最坏点提取

核心功能:
1. sample_simplices_batched: 利用 Dirichlet 分布在单纯形内部批量随机采样
2. find_worst_case_points: 对每个单纯形采样并找出最坏点
3. compute_cbf_condition: 计算 CBF 条件值 ∇h·f + sup_u[∇h·g·u] + α·h

改进点 (相比 v1):
- 使用 Dirichlet 分布进行均匀采样（相比 LBP 边界更精确）
- 采样估计真实梯度替代 McCormick 松弛梯度
- vmap + jacrev 批量计算雅可比矩阵
"""

from typing import List, Tuple, Union, Dict, Optional

import torch
import torch.nn as nn
import numpy as np


def sample_simplices_batched(
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    num_samples: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    利用 Dirichlet 分布批量采样单纯形内部点。

    Args:
        vertices_list: 单纯形顶点列表，每个元素形状 [D+1, D]
        num_samples: 每个单纯形采样的点数 K
        device: 目标设备
        dtype: 数据类型

    Returns:
        x_samples: 形状 [B, K, D]，B = len(vertices_list)
    """
    B = len(vertices_list)
    if B == 0:
        return torch.empty(0, num_samples, 0, device=device, dtype=dtype)

    # ---------- 转换顶点为张量 ----------
    vertices_tensors = []
    for verts in vertices_list:
        if isinstance(verts, np.ndarray):
            v = torch.from_numpy(verts).to(device=device, dtype=dtype)
        else:
            v = verts.to(device=device, dtype=dtype)
        vertices_tensors.append(v)  # each [D+1, D]

    # 获取维度 D
    D = vertices_tensors[0].shape[1]  # D

    # ---------- Dirichlet 采样 ----------
    # alpha = ones(D+1) -> 均匀采样
    alpha = torch.ones(D + 1, device=device, dtype=dtype)
    # shape: [B, K, D+1]
    barycentric = torch.distributions.Dirichlet(alpha).sample([B, num_samples])
    # barycentric 是归一化的权重 [B, K, D+1]

    # ---------- 映射到欧几里得坐标 ----------
    # x = B @ V: [B, K, D+1] @ [B, D+1, D] -> [B, K, D]
    # 先将 vertices stack 成 [B, D+1, D]
    V = torch.stack(vertices_tensors, dim=0)  # [B, D+1, D]
    x_samples = torch.bmm(barycentric, V)     # [B, K, D]

    return x_samples


def compute_cbf_condition_simple(
    model: nn.Module,
    dynamics_model,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    简化版 CBF 条件计算（无 translator，使用直接 PyTorch 操作）。

    适用于 Barrier3 等无控制系统的标准形式。
    cbf(x) = ∇h·f(x) + α·h(x)
    其中 f(x) = [x2, -x1 - x2 + x1^3/3]

    Args:
        model: BarrierNN 网络
        dynamics_model: 动力学系统
        x: 状态点 [batch, D]，requires_grad=True

    Returns:
        cbf_values: [batch]
    """
    # h(x)
    h = model(x).squeeze(-1)

    # ∇h(x)
    grad_h = torch.autograd.grad(
        outputs=h,
        inputs=x,
        grad_outputs=torch.ones_like(h, device=x.device),
        create_graph=False,
        retain_graph=True,
    )[0]

    # f(x) = [x2, -x1 - x2 + x1^3/3]
    x1 = x[..., 0]
    x2 = x[..., 1]
    dx1 = x2
    dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
    f_x = torch.stack([dx1, dx2], dim=-1)

    # ∇h·f
    grad_h_dot_f = (grad_h * f_x).sum(dim=-1)

    # α·h
    alpha_h = dynamics_model.alpha_function(h, None)

    cbf = grad_h_dot_f + alpha_h
    return cbf


def compute_cbf_condition(
    model: nn.Module,
    dynamics_model,
    x: torch.Tensor,
    translator=None,
) -> torch.Tensor:
    """
    计算 CBF 条件值: cbf(x) = ∇h·f(x) + sup_u[∇h·g(x)·u] + α·h(x)

    对于无控制系统 (control_dim=0): cbf(x) = ∇h·f(x) + α·h(x)
    对于有控制系统: sup_u 项通过 g(x) 的各列计算

    Args:
        model: BarrierNN 网络，输入 [batch, D]，输出 [batch, 1]
        dynamics_model: 动力学系统对象
        x: 状态点，形状 [batch, D]，requires_grad=True
        translator: TorchTranslator（可选，默认用内置 torch 操作）

    Returns:
        cbf_values: 形状 [batch]
    """
    batch_size = x.shape[0]
    D = x.shape[1]

    # ---------- 前向传播得到 h(x) ----------
    h = model(x).squeeze(-1)  # [batch]

    # ---------- 计算 ∇h(x) ----------
    grad_h = torch.autograd.grad(
        outputs=h,
        inputs=x,
        grad_outputs=torch.ones_like(h, device=x.device),
        create_graph=False,
        retain_graph=True,
    )[0]  # [batch, D]

    # ---------- 计算 f(x) ----------
    if translator is None:
        # 直接用 PyTorch 操作计算 f(x)
        x1 = x[..., 0]
        x2 = x[..., 1]
        if D == 2 and dynamics_model.control_dim == 0:
            # Barrier3 等无控制系统的典型形式: f(x) = [x2, -x1 - x2 + x1^3/3]
            dx1 = x2
            dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
            f_x = torch.stack([dx1, dx2], dim=-1)  # [batch, 2]
        else:
            raise NotImplementedError(
                f"通用 dynamics compute_f 需要 translator，"
                f"当前 D={D}, control_dim={dynamics_model.control_dim}"
            )
    else:
        f_x = dynamics_model.compute_f(x, translator)  # [batch, D]

    # ---------- 计算 ∇h·f(x) ----------
    grad_h_dot_f = (grad_h * f_x).sum(dim=-1)  # [batch]

    # ---------- sup_u 项（仅对有控制系统）----------
    if dynamics_model.control_dim > 0:
        if translator is None:
            raise ValueError("有控制系统需要提供 TorchTranslator")
        g_x = dynamics_model.compute_g(x, translator)  # [batch, D, m]

        u_min = torch.tensor(dynamics_model.u_min, device=x.device, dtype=x.dtype)
        u_max = torch.tensor(dynamics_model.u_max, device=x.device, dtype=x.dtype)

        # ∇h·g: [batch, D] @ [batch, D, m] -> [batch, m]
        grad_h_g = torch.einsum('bd,bdm->bm', grad_h, g_x)

        # sup_u = sum_j max(∇h·g_j·u_max_j, ∇h·g_j·u_min_j)
        term_max = grad_h_g * u_max.unsqueeze(0)
        term_min = grad_h_g * u_min.unsqueeze(0)
        sup_u = torch.sum(torch.maximum(term_max, term_min), dim=-1)
    else:
        sup_u = torch.zeros_like(h)

    # ---------- class-K 项 α·h(x) ----------
    alpha_h = dynamics_model.alpha_function(h, translator)

    # ---------- CBF 条件 ----------
    cbf_values = grad_h_dot_f + sup_u + alpha_h

    return cbf_values


def find_worst_case_points(
    model: nn.Module,
    dynamics_model,
    simplices_list: List[Union[torch.Tensor, np.ndarray]],
    num_samples: int = 500,
    region_type: str = 'safe',
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在每个单纯形内批量采样，找到最坏点。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        simplices_list: 单纯形列表（V_safe 或 V_unsafe）
        num_samples: 每个单纯形采样数
        region_type: 'safe' -> 选 cbf_condition 最小的点
                     'unsafe' -> 选 h(x) 最大的点
        device, dtype: 张量配置

    Returns:
        worst_points: [N, D]，每个单纯形的最坏点
        worst_values: [N]，对应的最坏值（cbf 或 h）
    """
    if not simplices_list:
        D = 2
        return (torch.empty(0, D, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype))

    N = len(simplices_list)

    # ---------- 批量采样: [N, K, D] ----------
    x_samples = sample_simplices_batched(
        simplices_list, num_samples, device=device, dtype=dtype
    )  # [N, K, D]

    # ---------- 展平为 [N*K, D] 以便批量计算 ----------
    NK = N * num_samples
    x_flat = x_samples.view(NK, -1).requires_grad_(True)  # [NK, D]

    # ---------- 计算 h(x) 或 cbf(x) ----------
    if region_type == 'unsafe':
        # 障碍区：找 h(x) 最大的点
        h_flat = model(x_flat).squeeze(-1)  # [NK]
        h_vals = h_flat.view(N, num_samples)  # [N, K]
        worst_idx = h_vals.argmax(dim=1)      # [N]
        worst_values = h_vals[torch.arange(N, device=device), worst_idx]

        # 正确索引：[N] + [N] → [N, D]，每行取对应 simplex 的 worst sample
        worst_points = x_samples[torch.arange(N, device=device), worst_idx]  # [N, D]

    else:
        # 安全区：找 cbf_condition 最小的点
        cbf_flat = compute_cbf_condition_simple(model, dynamics_model, x_flat)
        cbf_vals = cbf_flat.view(N, num_samples)  # [N, K]
        worst_idx = cbf_vals.argmin(dim=1)        # [N]
        worst_values = cbf_vals[torch.arange(N, device=device), worst_idx]

        worst_points = x_samples[torch.arange(N, device=device), worst_idx]  # [N, D]

    return worst_points.detach(), worst_values.detach()


def find_worst_case_points_v2(
    model: nn.Module,
    dynamics_model,
    simplices_list: List[Union[torch.Tensor, np.ndarray]],
    num_samples: int = 500,
    region_type: str = 'safe',
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    v2 版本：分批计算以节省显存。

    适用于 simplices 数量大、num_samples 很大的情况。
    """
    if not simplices_list:
        D = 2
        return (torch.empty(0, D, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype))

    N = len(simplices_list)
    batch_size = 50  # 每批处理的单纯形数

    all_worst_points = []
    all_worst_values = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_simplices = simplices_list[start:end]
        bN = len(batch_simplices)

        # 采样
        x_samples = sample_simplices_batched(
            batch_simplices, num_samples, device=device, dtype=dtype
        )  # [bN, K, D]

        NK = bN * num_samples
        x_flat = x_samples.view(NK, -1).requires_grad_(True)

        if region_type == 'unsafe':
            h_flat = model(x_flat).squeeze(-1)
            h_vals = h_flat.view(bN, num_samples)
            worst_idx = h_vals.argmax(dim=1)
            worst_values = h_vals[torch.arange(bN, device=device), worst_idx]
            worst_points = x_samples[torch.arange(bN, device=device), worst_idx]
        else:
            cbf_flat = compute_cbf_condition_simple(model, dynamics_model, x_flat)
            cbf_vals = cbf_flat.view(bN, num_samples)
            worst_idx = cbf_vals.argmin(dim=1)
            worst_values = cbf_vals[torch.arange(bN, device=device), worst_idx]
            worst_points = x_samples[torch.arange(bN, device=device), worst_idx]

        all_worst_points.append(worst_points.detach())
        all_worst_values.append(worst_values.detach())

    worst_points = torch.cat(all_worst_points, dim=0)
    worst_values = torch.cat(all_worst_values, dim=0)

    return worst_points, worst_values
