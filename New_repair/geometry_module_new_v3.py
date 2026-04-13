"""
几何模块 v3: 确定性特征点提取 (Feature Points Extraction)

核心功能:
1. extract_feature_points: 从单纯形提取顶点 + 重心点（确定性，无随机采样）
2. extract_all_feature_points: 批量提取所有单纯形的特征点
3. compute_cbf_condition_at_points: 在给定特征点上计算 CBF 条件

v3 改进点 (相比 v2):
- 不再使用 Dirichlet 随机采样找最坏点
- 改用确定性特征点提取：每个单纯形的顶点 + 重心
- 消除次梯度震荡（Subgradient Chattering）
- 大幅提升计算速度（无随机采样开销）
"""

from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
import numpy as np


def extract_feature_points_from_simplex(vertices: torch.Tensor) -> torch.Tensor:
    """
    从单个单纯形提取特征点：顶点 + 重心。

    Args:
        vertices: 单纯形顶点，形状 [D+1, D]

    Returns:
        feature_points: 形状 [D+2, D]，包含 D+1 个顶点 + 1 个重心
    """
    D = vertices.shape[1]  # 空间维度

    # 顶点: [D+1, D]
    vertices_feat = vertices  # [D+1, D]

    # 重心: 所有顶点的均值
    centroid = vertices.mean(dim=0, keepdim=True)  # [1, D]

    # 合并: [D+2, D]
    feature_points = torch.cat([vertices_feat, centroid], dim=0)

    return feature_points


def extract_all_feature_points(
    simplices_list: List[Union[torch.Tensor, np.ndarray]],
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从所有单纯形批量提取特征点（顶点 + 重心）。

    Args:
        simplices_list: 单纯形列表，每个元素形状 [D+1, D]
        device: 目标设备
        dtype: 数据类型

    Returns:
        all_feature_points: 形状 [N, D+2, D]，N = 单纯形数量
        feature_mask: 标记每个点是顶点(0)还是重心(1)，形状 [N, D+2]
                      0 = 顶点, 1 = 重心
    """
    if not simplices_list:
        return (torch.empty(0, 0, 0, device=device, dtype=dtype),
                torch.empty(0, 0, device=device, dtype=torch.int32))

    # ---------- 转换顶点为张量 ----------
    vertices_tensors = []
    D = None
    for verts in simplices_list:
        if isinstance(verts, np.ndarray):
            v = torch.from_numpy(verts).to(device=device, dtype=dtype)
        else:
            v = verts.to(device=device, dtype=dtype)
        vertices_tensors.append(v)
        if D is None:
            D = v.shape[1]

    N = len(vertices_tensors)
    num_vertices = D + 1
    num_feature_points = D + 2  # 顶点 + 重心

    # ---------- 批量计算重心 ----------
    V = torch.stack(vertices_tensors, dim=0)  # [N, D+1, D]
    centroids = V.mean(dim=1, keepdim=True)   # [N, 1, D]

    # ---------- 合并顶点 + 重心 ----------
    all_feature_points = torch.cat([V, centroids], dim=1)  # [N, D+2, D]

    # ---------- 创建特征点掩码 ----------
    # 前 D+1 个是顶点 (label=0)，最后一个是重心 (label=1)
    feature_mask = torch.zeros(N, num_feature_points, dtype=torch.int32, device=device)
    feature_mask[:, num_vertices:] = 1  # 重心位置标记为 1

    return all_feature_points, feature_mask


def extract_vertices_only(
    simplices_list: List[Union[torch.Tensor, np.ndarray]],
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    仅提取所有单纯形的顶点（不含重心）。

    Args:
        simplices_list: 单纯形列表
        device, dtype: 张量配置

    Returns:
        all_vertices: 形状 [N*(D+1), D]
    """
    if not simplices_list:
        return torch.empty(0, 0, device=device, dtype=dtype)

    vertices_tensors = []
    for verts in simplices_list:
        if isinstance(verts, np.ndarray):
            v = torch.from_numpy(verts).to(device=device, dtype=dtype)
        else:
            v = verts.to(device=device, dtype=dtype)
        vertices_tensors.append(v)

    V = torch.stack(vertices_tensors, dim=0)  # [N, D+1, D]
    all_vertices = V.view(-1, V.shape[-1])    # [N*(D+1), D]

    return all_vertices


def compute_cbf_condition_at_points(
    model: nn.Module,
    dynamics_model,
    x: torch.Tensor,
    translator=None,
) -> torch.Tensor:
    """
    在给定特征点上计算 CBF 条件值。

    cbf(x) = ∇h·f(x) + α·h(x)
    （无控制系统的简化形式，如有控制系统则包含 sup_u 项）

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        x: 特征点，形状 [batch, D]，requires_grad=True
        translator: TorchTranslator（可选）

    Returns:
        cbf_values: 形状 [batch]
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
        # 直接用 PyTorch 操作计算 f(x)
        x1 = x[..., 0]
        x2 = x[..., 1]
        D = x.shape[1]
        if D == 2 and dynamics_model.control_dim == 0:
            # Barrier3 等无控制系统: f(x) = [x2, -x1 - x2 + x1^3/3]
            dx1 = x2
            dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
            f_x = torch.stack([dx1, dx2], dim=-1)
        else:
            raise NotImplementedError(
                f"通用 dynamics compute_f 需要 translator，"
                f"当前 D={D}, control_dim={dynamics_model.control_dim}"
            )
    else:
        f_x = dynamics_model.compute_f(x, translator)

    # ∇h·f
    grad_h_dot_f = (grad_h * f_x).sum(dim=-1)

    # α·h
    alpha_h = dynamics_model.alpha_function(h, translator)

    cbf = grad_h_dot_f + alpha_h
    return cbf


def compute_h_values_at_points(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    在给定点上计算 h(x) 值。

    Args:
        model: BarrierNN
        x: 特征点，形状 [batch, D]

    Returns:
        h_values: 形状 [batch]
    """
    h = model(x).squeeze(-1)
    return h


def compute_cbf_condition_batch(
    model: nn.Module,
    dynamics_model,
    all_feature_points: torch.Tensor,
    feature_mask: torch.Tensor,
    translator=None,
) -> Dict[str, torch.Tensor]:
    """
    批量计算所有特征点的 CBF 条件和 h 值。

    用于分析哪些单纯形/特征点需要修复。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        all_feature_points: 形状 [N, D+2, D]
        feature_mask: 标记顶点和重心
        translator: TorchTranslator

    Returns:
        Dict with keys:
            'cbf_vertices': 顶点处 CBF 值, shape [N, D+1]
            'cbf_centroids': 重心处 CBF 值, shape [N]
            'h_vertices': 顶点处 h 值, shape [N, D+1]
            'h_centroids': 重心处 h 值, shape [N]
            'worst_cbf_per_simplex': 每个单纯形最差 CBF, shape [N]
            'worst_h_per_simplex': 每个单纯形最差 h, shape [N]
    """
    N, num_fp, D = all_feature_points.shape
    num_vertices = D + 1

    # 展平为批量计算
    x_flat = all_feature_points.view(N * num_fp, D)  # [N*(D+2), D]

    # 计算 h(x)
    h_flat = compute_h_values_at_points(model, x_flat)  # [N*(D+2)]
    h_vals = h_flat.view(N, num_fp)  # [N, D+2]

    # 分离顶点和重心
    h_vertices = h_vals[:, :num_vertices]   # [N, D+1]
    h_centroids = h_vals[:, num_vertices]   # [N]

    # 计算 CBF 条件
    cbf_flat = compute_cbf_condition_at_points(
        model, dynamics_model, x_flat, translator
    )  # [N*(D+2)]
    cbf_vals = cbf_flat.view(N, num_fp)  # [N, D+2]

    cbf_vertices = cbf_vals[:, :num_vertices]   # [N, D+1]
    cbf_centroids = cbf_vals[:, num_vertices]   # [N]

    # 每个单纯形的最坏值
    worst_cbf_per_simplex = cbf_vals.min(dim=1).values  # [N]（最坏=最小）
    worst_h_per_simplex = h_vals.max(dim=1).values     # [N]（对 unsafe，h 应该最小）

    return {
        'cbf_vertices': cbf_vertices,
        'cbf_centroids': cbf_centroids,
        'h_vertices': h_vertices,
        'h_centroids': h_centroids,
        'worst_cbf_per_simplex': worst_cbf_per_simplex,
        'worst_h_per_simplex': worst_h_per_simplex,
    }


def find_failed_simplices(
    model: nn.Module,
    dynamics_model,
    simplices_list: List[Union[torch.Tensor, np.ndarray]],
    region_type: str = 'safe',
    cbf_threshold: float = 0.0,
    h_threshold: float = 0.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    translator=None,
) -> Tuple[List[int], Dict]:
    """
    找出哪些单纯形需要修复（CBF 违规或 h 违规）。

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        simplices_list: 单纯形列表
        region_type: 'safe' -> 检查 CBF 条件违规
                     'unsafe' -> 检查 h(x) > 0 违规
        cbf_threshold: CBF 阈值（小于等于视为违规）
        h_threshold: h 阈值（大于视为违规）
        device, dtype: 张量配置
        translator: TorchTranslator

    Returns:
        failed_indices: 需要修复的单纯形索引列表
        debug_info: 详细诊断信息
    """
    if not simplices_list:
        return [], {}

    all_feature_points, feature_mask = extract_all_feature_points(
        simplices_list, device=device, dtype=dtype
    )

    results = compute_cbf_condition_batch(
        model, dynamics_model, all_feature_points, feature_mask, translator
    )

    N = len(simplices_list)
    failed_indices = []
    debug_info = {
        'worst_cbf_values': [],
        'worst_h_values': [],
        'cbf_violations': [],
        'h_violations': [],
    }

    for i in range(N):
        if region_type == 'safe':
            # CBF 条件应该 >= 0
            worst_cbf = results['worst_cbf_per_simplex'][i].item()
            if worst_cbf < cbf_threshold:
                failed_indices.append(i)
                debug_info['cbf_violations'].append(worst_cbf)
        else:
            # h(x) 应该 <= 0（障碍函数）
            worst_h = results['worst_h_per_simplex'][i].item()
            if worst_h > h_threshold:
                failed_indices.append(i)
                debug_info['h_violations'].append(worst_h)

        debug_info['worst_cbf_values'].append(results['worst_cbf_per_simplex'][i].item())
        debug_info['worst_h_values'].append(results['worst_h_per_simplex'][i].item())

    return failed_indices, debug_info


def extract_feature_points_flat(
    simplices_list: List[Union[torch.Tensor, np.ndarray]],
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    提取所有特征点并展平为 2D 张量。

    Args:
        simplices_list: 单纯形列表，每个元素形状 [D+1, D]
        device, dtype: 张量配置

    Returns:
        feature_points_flat: 形状 [N*(D+2), D]
    """
    if not simplices_list:
        return torch.empty(0, 0, device=device, dtype=dtype)

    all_feature_points, _ = extract_all_feature_points(
        simplices_list, device=device, dtype=dtype
    )

    N, num_fp, D = all_feature_points.shape
    return all_feature_points.view(N * num_fp, D)


def separate_vertices_and_centroids(
    all_feature_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将特征点分离为顶点和重心。

    Args:
        all_feature_points: 形状 [N*(D+2), D] 或 [N, D+2, D]

    Returns:
        vertices: 顶点 [N*(D+1), D]
        centroids: 重心 [N, D]
    """
    if all_feature_points.dim() == 3:
        N, num_fp, D = all_feature_points.shape
        num_vertices = D + 1
        vertices = all_feature_points[:, :num_vertices, :].view(N * num_vertices, D)
        centroids = all_feature_points[:, num_vertices, :]
    else:
        raise ValueError("Expected 3D tensor, call extract_all_feature_points first")

    return vertices, centroids
