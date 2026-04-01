"""
测试 geometry_module.py 雅可比提取的正确性

流程:
1. 加载 barr3 系统的 CBF 神经网络
2. 读取 verified_regions_barr3.pt 获取 V_safe 和 V_unsafe
3. 调用 compute_jacobian_matrix 并打印结果
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np

from lbp_neural_cbf.cbf.fossil_dynamics import Barrier3System
from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.translators import TorchTranslator
from New_repair.geometry_module import compute_jacobian_matrix, compute_simplex_bound


def test_jacobian_extraction():
    """测试雅可比矩阵提取的正确性"""
    print("=" * 60)
    print("测试雅可比矩阵提取")
    print("=" * 60)

    # ========== 1. 加载动力学系统 ==========
    dynamics_model = Barrier3System(alpha=1.0)
    print(f"\n[1] 动力学系统: {dynamics_model.system_name}")
    print(f"    输入维度: {dynamics_model.input_dim}")
    print(f"    隐藏层大小: {dynamics_model.hidden_sizes}")

    # ========== 2. 加载神经网络 ==========
    device = torch.device('cpu')
    model_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.pth"

    print(f"\n[2] 加载神经网络: {model_path}")

    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # 统计参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    模型参数数量: {num_params}")
    print(f"    模型结构: {dynamics_model.input_dim} -> {dynamics_model.hidden_sizes} -> 1")

    # ========== 3. 读取验证区域 ==========
    regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}.pt"

    print(f"\n[3] 读取验证区域: {regions_path}")

    regions_data = torch.load(regions_path, map_location=device, weights_only=False)

    V_safe = regions_data['V_safe']
    V_unsafe = regions_data['V_unsafe']

    print(f"    V_safe 数量: {len(V_safe)}")
    print(f"    V_unsafe 数量: {len(V_unsafe)}")

    if len(V_safe) == 0 and len(V_unsafe) == 0:
        print("    警告: 没有找到任何验证区域!")
        return

    # 检查 V_safe 和 V_unsafe 的形状
    if len(V_safe) > 0:
        safe_shape = np.array(V_safe[0]).shape
        print(f"    V_safe[0] 形状: {safe_shape}")

    if len(V_unsafe) > 0:
        unsafe_shape = np.array(V_unsafe[0]).shape
        print(f"    V_unsafe[0] 形状: {unsafe_shape}")

    # ========== 4. 计算雅可比矩阵 ==========
    translator = TorchTranslator(device=device)

    print(f"\n[4] 计算雅可比矩阵...")

    J = compute_jacobian_matrix(
        model,
        V_safe,
        V_unsafe,
        dynamics_model=dynamics_model,
        translator=translator
    )

    print(f"    雅可比矩阵 J 形状: {J.shape}")
    print(f"    期望形状: ({len(V_safe) + len(V_unsafe)}, {num_params})")

    # 验证形状是否正确
    expected_shape = (len(V_safe) + len(V_unsafe), num_params)
    if J.shape == expected_shape:
        print("    ✓ 雅可比矩阵形状正确!")
    else:
        print(f"    ✗ 形状不匹配! 期望 {expected_shape}, 实际 {J.shape}")

    # ========== 5. 检查雅可比矩阵的性质 ==========
    print(f"\n[5] 检查雅可比矩阵的性质...")

    # 检查是否有 NaN 或 Inf
    has_nan = torch.isnan(J).any().item()
    has_inf = torch.isinf(J).any().item()

    print(f"    包含 NaN: {has_nan}")
    print(f"    包含 Inf: {has_inf}")

    if has_nan or has_inf:
        print("    ✗ 雅可比矩阵包含无效值!")
    else:
        print("    ✓ 雅可比矩阵无无效值")

    # 检查数值范围
    J_abs_max = torch.max(torch.abs(J)).item()
    J_mean = torch.mean(torch.abs(J)).item()
    print(f"    |J| 最大值: {J_abs_max:.6f}")
    print(f"    |J| 平均值: {J_mean:.6f}")

    # ========== 6. 测试 compute_simplex_bound ==========
    print(f"\n[6] 测试 compute_simplex_bound...")

    if len(V_safe) > 0:
        # 测试 safe 区域
        vertices_safe = V_safe[0]
        bound_safe = compute_simplex_bound(
            model, vertices_safe, 'safe',
            dynamics_model=dynamics_model,
            translator=translator
        )
        print(f"    V_safe[0] 边界值 (safe): {bound_safe.item():.6f}")
        print(f"    requires_grad: {bound_safe.requires_grad}")

    if len(V_unsafe) > 0:
        # 测试 unsafe 区域
        vertices_unsafe = V_unsafe[0]
        bound_unsafe = compute_simplex_bound(
            model, vertices_unsafe, 'unsafe',
            dynamics_model=dynamics_model,
            translator=translator
        )
        print(f"    V_unsafe[0] 边界值 (unsafe): {bound_unsafe.item():.6f}")
        print(f"    requires_grad: {bound_unsafe.requires_grad}")

    # ========== 7. 梯度反向传播测试 ==========
    print(f"\n[7] 梯度反向传播测试...")

    if len(V_safe) > 0:
        vertices_safe = V_safe[0]
        bound_safe = compute_simplex_bound(
            model, vertices_safe, 'safe',
            dynamics_model=dynamics_model,
            translator=translator
        )

        # 反向传播
        loss_safe = bound_safe.sum()
        loss_safe.backward()

        # 检查是否有梯度
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

        if grad_norms:
            print(f"    安全区边界梯度范数: {grad_norms[0]:.6f} (第一个参数)")
            print("    ✓ 梯度反向传播成功!")
        else:
            print("    ✗ 没有检测到梯度!")

        # 清空梯度
        model.zero_grad()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


def main():
    test_jacobian_extraction()


if __name__ == "__main__":
    main()
