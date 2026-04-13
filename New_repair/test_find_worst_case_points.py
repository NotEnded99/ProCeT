"""
test_find_worst_case_points.py: 验证采样点是否在单纯形内部

测试目的:
1. 验证 Dirichlet 采样产生的点确实在单纯形内部
2. 验证 find_worst_case_points 采样出来的点满足单纯形约束
3. 几何直观验证：重心坐标非负且和为1

单纯形定义:
- D维单纯形由 D+1 个顶点 v_0, v_1, ..., v_D 组成
- 内部点 x 可以表示为: x = Σ λ_i * v_i, 其中 λ_i >= 0, Σ λ_i = 1
- λ = (λ_0, ..., λ_D) 即重心坐标
"""

import sys
import os
import torch
import numpy as np

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from New_repair.geometry_module_new_v2 import (
    sample_simplices_batched,
    find_worst_case_points,
)


def verify_point_in_simplex(points: torch.Tensor, vertices: torch.Tensor) -> tuple:
    """
    验证点是否在单纯形内部。

    Args:
        points: 待验证的点 [N, D]
        vertices: 单纯形顶点 [D+1, D]

    Returns:
        (is_inside, barycentric_coords)
        - is_inside: bool, 是否在单纯形内
        - barycentric: 重心坐标 [N, D+1], 所有值应 >= 0 且和为1
    """
    N, D = points.shape
    num_vertices = D + 1

    # 构建 Vandermonde 矩阵: V @ λ = x
    # V shape: [D, D+1] -> 转置后: [D+1, D]
    # 实际上是求解 V^T @ λ = x, 其中 V 是顶点矩阵

    # 使用线性回归求解重心坐标（因为 D+1 个顶点，D 个维度，可能过约束）
    # 更准确的方法：找到唯一的 λ 满足 Σλ_i=1, λ_i>=0, 且 x=Σλ_i*v_i

    # 简化为数值求解：构造约束矩阵
    V = vertices  # [D+1, D]
    I = torch.eye(num_vertices, device=vertices.device, dtype=vertices.dtype)

    # 添加约束 Σλ_i = 1
    # 求解: [V^T; 1^T] @ λ = [x; 1]
    # 即 A @ λ = b

    A = torch.cat([V.T, torch.ones(1, num_vertices, device=vertices.device, dtype=vertices.dtype)], dim=0)
    b = torch.cat([points.T, torch.ones(1, N, device=points.device, dtype=points.dtype)], dim=0)

    # 最小二乘求解（理论上精确）
    # λ = (A^T @ A)^{-1} @ A^T @ b
    try:
        ATA = A @ A.T
        ATA_inv = torch.inverse(ATA + 1e-8 * torch.eye(num_vertices, device=ATA.device, dtype=ATA.dtype))
        lambda_vec = A.T @ ATA_inv @ b  # [D+1, N]
        barycentric = lambda_vec.T  # [N, D+1]
    except Exception:
        return False, None

    # 验证
    is_non_negative = (barycentric >= -1e-6).all(dim=1)
    sum_close_to_one = torch.abs(barycentric.sum(dim=1) - 1.0) < 1e-6

    return (is_non_negative & sum_close_to_one).all().item(), barycentric


def test_sample_simplices_batched():
    """测试 Dirichlet 采样是否在单纯形内部"""
    print("=" * 70)
    print("测试 1: sample_simplices_batched - Dirichlet 采样验证")
    print("=" * 70)

    device = torch.device('cpu')
    dtype = torch.float32

    # 创建简单测试单纯形
    # 2D 单纯形（三角形）
    # 顶点: (0,0), (1,0), (0,1)
    triangle = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=dtype, device=device)

    # 创建多个单纯形列表
    simplices_list = [triangle, triangle + 1.0]  # 2个三角形

    num_samples = 1000
    x_samples = sample_simplices_batched(simplices_list, num_samples, device=device, dtype=dtype)

    print(f"采样形状: {x_samples.shape}")  # [B, K, D] = [2, 1000, 2]
    B, K, D = x_samples.shape
    num_vertices = D + 1

    # 验证每个采样点
    all_inside = True
    for b in range(B):
        simplex_verts = simplices_list[b]  # [D+1, D]
        points = x_samples[b]  # [K, D]

        is_inside, barycentric = verify_point_in_simplex(points, simplex_verts)

        print(f"\n单纯形 {b}:")
        print(f"  顶点: {simplex_verts.cpu().numpy()}")
        print(f"  采样点数: {K}")
        print(f"  重心坐标范围: [{barycentric.min().item():.4f}, {barycentric.max().item():.4f}]")
        print(f"  重心坐标和: {barycentric.sum(dim=1).mean().item():.6f} (应为1.0)")
        print(f"  全部在内部: {is_inside}")

        # 额外检查：非负性
        if not (barycentric >= 0).all():
            print(f"  [错误] 发现负的重心坐标!")
            all_inside = False
            neg_coords = (barycentric < 0).sum().item()
            print(f"  负坐标数量: {neg_coords} / {K * num_vertices}")

    print(f"\n{'='*50}")
    print(f"测试 1 结果: {'通过 ✓' if all_inside else '失败 ✗'}")
    return all_inside


def test_find_worst_case_points():
    """测试 find_worst_case_points 采样点是否在单纯形内部"""
    print("\n" + "=" * 70)
    print("测试 2: find_worst_case_points - 采样点验证")
    print("=" * 70)

    device = torch.device('cpu')
    dtype = torch.float32

    # 导入必要的模块
    import torch.nn as nn

    class SimpleBarrierNN(nn.Module):
        """简单测试网络"""
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.fc(x)

    model = SimpleBarrierNN()

    # 创建简单动力学模型（用于测试）
    class SimpleDynamics:
        def __init__(self):
            self.control_dim = 0
            self.input_dim = 2

        def alpha_function(self, h, translator):
            return h  # α(h) = h

    dynamics_model = SimpleDynamics()

    # 创建测试单纯形
    triangle1 = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=dtype, device=device)

    triangle2 = torch.tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [1.0, 1.0],
    ], dtype=dtype, device=device)

    simplices_list = [triangle1, triangle2]
    num_samples = 100

    # 测试 unsafe 模式（应该找 h 最大的点）
    unsafe_points, unsafe_values = find_worst_case_points(
        model=model,
        dynamics_model=dynamics_model,
        simplices_list=simplices_list,
        num_samples=num_samples,
        region_type='unsafe',
        device=device,
    )

    print(f"\nUnsafe 模式采样:")
    print(f"  采样点数: {unsafe_points.shape[0]}")
    print(f"  h 值范围: [{unsafe_values.min().item():.4f}, {unsafe_values.max().item():.4f}]")

    # 验证这些点是否在单纯形内
    all_inside_unsafe = True
    for i, simplex_idx in enumerate(range(len(simplices_list))):
        simplex_verts = simplices_list[simplex_idx]
        point = unsafe_points[i].unsqueeze(0)

        is_inside, barycentric = verify_point_in_simplex(point, simplex_verts)
        print(f"\n单纯形 {simplex_idx} 的最坏点:")
        print(f"  坐标: {point.cpu().numpy().flatten()}")
        print(f"  h 值: {unsafe_values[i].item():.4f}")
        print(f"  重心坐标: {barycentric.flatten().cpu().numpy()}")
        print(f"  在内部: {is_inside}")

        if not is_inside:
            all_inside_unsafe = False

    # 测试 safe 模式（应该找 cbf 最小的点）
    safe_points, safe_values = find_worst_case_points(
        model=model,
        dynamics_model=dynamics_model,
        simplices_list=simplices_list,
        num_samples=num_samples,
        region_type='safe',
        device=device,
    )

    print(f"\n{'='*50}")
    print(f"Safe 模式采样:")
    print(f"  采样点数: {safe_points.shape[0]}")
    print(f"  cbf 值范围: [{safe_values.min().item():.4f}, {safe_values.max().item():.4f}]")

    all_inside_safe = True
    for i, simplex_idx in enumerate(range(len(simplices_list))):
        simplex_verts = simplices_list[simplex_idx]
        point = safe_points[i].unsqueeze(0)

        is_inside, barycentric = verify_point_in_simplex(point, simplex_verts)
        print(f"\n单纯形 {simplex_idx} 的最坏点:")
        print(f"  坐标: {point.cpu().numpy().flatten()}")
        print(f"  cbf 值: {safe_values[i].item():.4f}")
        print(f"  重心坐标: {barycentric.flatten().cpu().numpy()}")
        print(f"  在内部: {is_inside}")

        if not is_inside:
            all_inside_safe = False

    print(f"\n{'='*50}")
    print(f"测试 2 结果: unsafe={'通过 ✓' if all_inside_unsafe else '失败 ✗'}, safe={'通过 ✓' if all_inside_safe else '失败 ✗'}")
    return all_inside_unsafe and all_inside_safe


def test_dirichlet_math():
    """数学验证：Dirichlet采样为何产生单纯形内的点"""
    print("\n" + "=" * 70)
    print("测试 3: Dirichlet 采样数学原理验证")
    print("=" * 70)

    device = torch.device('cpu')
    dtype = torch.float32

    # 2D 单纯形（三角形）
    D = 2
    num_vertices = D + 1  # = 3

    alpha = torch.ones(num_vertices, device=device, dtype=dtype)
    num_samples = 10000

    # Dirichlet 采样
    barycentric = torch.distributions.Dirichlet(alpha).sample([num_samples])

    print(f"\nDirichlet(alpha={alpha.tolist()}) 采样 {num_samples} 次:")
    print(f"  采样形状: {barycentric.shape}")  # [10000, 3]
    print(f"  每行和: min={barycentric.sum(dim=1).min().item():.6f}, "
          f"max={barycentric.sum(dim=1).max().item():.6f}")

    # 验证约束
    non_negative = (barycentric >= 0).all(dim=1).sum().item()
    sum_one = torch.abs(barycentric.sum(dim=1) - 1.0).max().item()

    print(f"  非负比例: {non_negative}/{num_samples} = {non_negative/num_samples*100:.2f}%")
    print(f"  最大偏差(从1.0): {sum_one:.8f}")

    # 映射到欧几里得空间
    triangle = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=dtype, device=device)

    V = triangle  # [3, 2]
    x_samples = torch.bmm(barycentric.unsqueeze(0), V.unsqueeze(0)).squeeze(0)  # [10000, 2]

    # 验证映射后的点是否在三角形内
    in_triangle = ((x_samples[:, 0] >= 0) & (x_samples[:, 1] >= 0) &
                   (x_samples[:, 0] + x_samples[:, 1] <= 1.0 + 1e-6)).sum().item()

    print(f"\n映射到三角形后:")
    print(f"  点坐标范围: x1=[{x_samples[:,0].min().item():.4f}, {x_samples[:,0].max().item():.4f}], "
          f"x2=[{x_samples[:,1].min().item():.4f}, {x_samples[:,1].max().item():.4f}]")
    print(f"  在三角形内: {in_triangle}/{num_samples} = {in_triangle/num_samples*100:.2f}%")

    passed = (sum_one < 1e-5) and (in_triangle == num_samples)
    print(f"\n测试 3 结果: {'通过 ✓' if passed else '失败 ✗'}")
    return passed


def main():
    print("\n" + "#" * 70)
    print("# 单纯形采样验证测试")
    print("#" * 70)

    results = []

    # 测试 1: Dirichlet 采样基础验证
    results.append(("sample_simplices_batched", test_sample_simplices_batched()))

    # 测试 2: find_worst_case_points 验证
    results.append(("find_worst_case_points", test_find_worst_case_points()))

    # 测试 3: Dirichlet 数学原理
    results.append(("Dirichlet 数学验证", test_dirichlet_math()))

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    all_passed = True
    for name, passed in results:
        status = "通过 ✓" if passed else "失败 ✗"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n{'='*50}")
    if all_passed:
        print("全部测试通过! 采样点在单纯形内部 ✓")
    else:
        print("存在测试失败! 需要检查代码 ✗")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
