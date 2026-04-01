"""
测试 geometry_module.py 的导入和基本功能
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from New_repair.geometry_module import compute_simplex_bound, compute_jacobian_matrix


def test_compute_simplex_bound():
    """测试 compute_simplex_bound 函数"""
    print("\n" + "="*60)
    print("TEST 1: compute_simplex_bound")
    print("="*60)

    # 创建一个简单的 2D 神经网络
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    model.eval()

    # 创建一个 2D 单纯形的顶点（三角形）
    simplex_vertices = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ], dtype=torch.float32)

    print(f"Simplex vertices shape: {simplex_vertices.shape}")

    # 测试 unsafe 区域
    try:
        bound = compute_simplex_bound(model, simplex_vertices, 'unsafe')
        print(f"Unsafe region bound (h_max): {bound.item():.6f}")
        print(f"Requires grad: {bound.requires_grad}")
        print("✅ compute_simplex_bound (unsafe) passed!")
    except Exception as e:
        print(f"❌ compute_simplex_bound (unsafe) failed: {e}")
        import traceback
        traceback.print_exc()

    # 测试 safe 区域（需要 dynamics_model 和 translator）
    try:
        from lbp_neural_cbf.cbf.fossil_dynamics import Barrier3System
        from lbp_neural_cbf.translators import TorchTranslator

        dynamics = Barrier3System(alpha=1.0)
        translator = TorchTranslator()

        bound = compute_simplex_bound(model, simplex_vertices, 'safe', dynamics, translator)
        print(f"Safe region bound (min_L): {bound.item():.6f}")
        print(f"Requires grad: {bound.requires_grad}")
        print("✅ compute_simplex_bound (safe) passed!")
    except Exception as e:
        print(f"❌ compute_simplex_bound (safe) failed: {e}")
        import traceback
        traceback.print_exc()


def test_compute_jacobian_matrix():
    """测试 compute_jacobian_matrix 函数"""
    print("\n" + "="*60)
    print("TEST 2: compute_jacobian_matrix")
    print("="*60)

    # 创建一个简单的 2D 神经网络
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    model.eval()

    # 创建一些单纯形
    V_safe = [
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32),
        torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.5, 2.0]], dtype=torch.float32)
    ]

    V_unsafe = [
        torch.tensor([[-1.0, -1.0], [0.0, -1.0], [-0.5, 0.0]], dtype=torch.float32),
    ]

    print(f"V_safe count: {len(V_safe)}")
    print(f"V_unsafe count: {len(V_unsafe)}")

    # 计算雅可比矩阵
    try:
        J = compute_jacobian_matrix(model, V_safe, V_unsafe)
        print(f"Jacobian matrix shape: {J.shape}")
        print(f"Number of simplices: {J.shape[0]}")
        print(f"Number of parameters: {J.shape[1]}")
        print("✅ compute_jacobian_matrix passed!")
    except Exception as e:
        print(f"❌ compute_jacobian_matrix failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing geometry_module.py...")
    print("="*60)

    test_compute_simplex_bound()
    test_compute_jacobian_matrix()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
