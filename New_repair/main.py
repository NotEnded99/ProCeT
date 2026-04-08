"""
迭代修复脚本：加载 verified_regions.pt，执行修复-验证循环 5 次

流程:
1. 加载初始模型和验证区域
2. 重复 5 次:
   - 计算雅可比矩阵 J
   - 执行修复迭代
   - 将 PyTorch 模型转换为 ONNX
   - 运行 verify_cbf 验证
   - 保存验证结果
   - 输出验证通过率
   - 保存修复后的模型

用法:
    python3 New_repair/main.py --activation Tanh --system barr1
    python3 New_repair/main.py -a Tanh -s simple_2d
"""

import sys
import os
import random
import argparse
import numpy as np
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np

from lbp_neural_cbf.cbf.fossil_dynamics import (
    Barrier1System, Barrier2System, Barrier3System, Barrier4System
)
from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem
from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from lbp_neural_cbf.translators import TorchTranslator

from New_repair.geometry_module_new import compute_jacobian_matrix
from New_repair.optimizer_module import repair_iteration

# 支持的动力学系统映射
DYNAMICS_SYSTEMS = {
    'simple_2d': Simple2DSystem,
    'barr1': Barrier1System,
    'barr2': Barrier2System,
    'barr3': Barrier3System,
    'barr4': Barrier4System,
}

# 支持的激活函数
SUPPORTED_ACTIVATIONS = ['Relu', 'Tanh', 'Sigmoid']


def pytorch_to_onnx(model, onnx_path, input_dim=2):
    """
    将 PyTorch 模型转换为 ONNX 格式。

    Args:
        model: PyTorch 模型
        onnx_path: 输出 ONNX 文件路径
        input_dim: 输入维度
    """
    device = next(model.parameters()).device
    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(1, input_dim, device=device)

    # 导出为 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )


def verify_model(model_path, dynamics_model, max_depth=13):
    """
    验证模型并返回验证结果。

    Args:
        model_path: ONNX 模型路径
        dynamics_model: 动力学系统

    Returns:
        验证结果字典
    """
    results = verify_cbf(
        dynamics_model,
        model_path,
        visualize=False,
        use_gpu=False,
        batch_size=512,
        executor_type="single",
        region_type="simplicial",
        max_depth=max_depth,
    )
    return results


def calculate_pass_rate(results):
    """
    计算验证通过率。

    Args:
        results: verify_cbf 返回的结果

    Returns:
        (pass_rate, stats): 通过率和统计信息
    """
    # 从 verify_cbf 结果中提取信息
    certified_pct = results.get('certified_percentage', 0.0)
    uncertified_pct = results.get('uncertified_percentage', 0.0)
    total_samples = results.get('total_samples', 0)

    # 通过率 = certified_percentage
    pass_rate = certified_pct

    stats = {
        'total': total_samples,
        'certified_pct': certified_pct,
        'uncertified_pct': uncertified_pct,
        'pass_rate': pass_rate
    }

    return pass_rate, stats


def main():
    # ========== 0. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description='Neural CBF 迭代修复')
    parser.add_argument('--activation', '-a', type=str, required=True,
                        choices=SUPPORTED_ACTIVATIONS,
                        help='激活函数: Relu, Tanh, Sigmoid')
    parser.add_argument('--system', '-s', type=str, required=True,
                        choices=list(DYNAMICS_SYSTEMS.keys()),
                        help='动力学系统: simple_2d, barr1, barr2, barr3, barr4')
    args = parser.parse_args()

    activation = args.activation   # e.g. 'Relu'
    system_name_key = args.system  # e.g. 'barr1'

    # ========== 0. 固定随机数种子 ==========
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # 确保 CuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print(f"Neural CBF 迭代修复演示  [激活={activation}, 系统={system_name_key}]")
    print("=" * 70)

    num_iterations = 5
    max_depth = 8

    # ========== 1. 加载动力学系统 ==========
    dynamics_class = DYNAMICS_SYSTEMS[system_name_key]
    dynamics_model = dynamics_class(alpha=1.0)

    if activation is not None:
        valid_activations = ["Tanh", "Relu", "Sigmoid"]
        if activation not in valid_activations:
            raise ValueError(f"Invalid activation function: {activation}. Must be one of {valid_activations}")
        dynamics_model.activation_fnc = activation
        print(f"  Using activation function: {activation} (overriding default)")
    
    print(f"\n[1] 动力学系统: {dynamics_model.system_name}")
    print(f"    激活函数: {activation}")
    print(f"    输入维度: {dynamics_model.input_dim}")
    print(f"    隐藏层大小: {dynamics_model.hidden_sizes}")

    # ========== 2. 加载初始神经网络 ==========
    device = torch.device('cuda')
    model_dir = f"data/New_models_Hard_{activation}"
    model_path = f"{model_dir}/{dynamics_model.system_name}_cbf.pth"

    print(f"\n[2] 加载初始神经网络: {model_path}")

    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device,
        activation_fnc=activation
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"    参数数量: {num_params}")

    # ========== 2.1 计算初始模型验证通过率 ==========
    # print(f"\n[2.1] 计算初始模型验证通过率...")
    # onnx_path = f"{model_dir}/{dynamics_model.system_name}_cbf.onnx"
    # initial_results = verify_model(onnx_path, dynamics_model, max_depth=max_depth)
    # initial_pass_rate, initial_stats = calculate_pass_rate(initial_results)

    # print(f"\n    初始模型验证结果:")
    # print(f"    总样本数: {initial_stats['total']}")
    # print(f"    Certified: {initial_stats['certified_pct']:.2f}%")
    # print(f"    Uncertified: {initial_stats['uncertified_pct']:.2f}%")
    # print(f"    ★ 初始通过率: {initial_pass_rate:.2f}%")

    # ========== 3. 读取初始验证区域 ==========
    regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}.pt"

    print(f"\n[3] 读取初始验证区域: {regions_path}")

    regions_data = torch.load(regions_path, map_location=device, weights_only=False)

    V_safe_init = regions_data['V_safe']
    V_unsafe_init = regions_data['V_unsafe']
    F_h_positive_in_unsafe_init = regions_data['F_h_positive_in_unsafe']
    F_safe_cbf_violation_init = regions_data['F_safe_cbf_violation']
    F_depth_limit_reached_init = regions_data['F_depth_limit_reached']
    F_unsafe_cannot_split_init = regions_data['F_unsafe_cannot_split']
    initial_pass_rate = regions_data['Certified percentage']

    total_fail = (len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init) +
                  len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init))

    print(f"    初始 V_safe: {len(V_safe_init)}")
    print(f"    初始 V_unsafe: {len(V_unsafe_init)}")
    print(f"    初始 F_h_positive_in_unsafe: {len(F_h_positive_in_unsafe_init)}  <- 需要修复")
    print(f"    初始 F_safe_cbf_violation: {len(F_safe_cbf_violation_init)}  <- 需要修复")
    print(f"    初始 F_depth_limit_reached: {len(F_depth_limit_reached_init)}  <- 需要修复")
    print(f"    初始 F_unsafe_cannot_split: {len(F_unsafe_cannot_split_init)}  <- 需要修复")
    print(f"    总需修复区域数: {total_fail}")
    print(f"    初始验证通过率: {initial_pass_rate:.2f}%")

    # ========== 4. 设置 translator ==========
    translator = TorchTranslator(device=device)

    # ========== 5. 迭代修复 ==========


    # 用于跟踪迭代结果
    iteration_results = []

    print(f"\n{'='*70}")
    print(f"开始 {num_iterations} 次迭代修复")
    print(f"{'='*70}")

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"迭代 {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        # 5.1 计算雅可比矩阵
        print(f"\n[迭代 {iteration+1}.1] 计算雅可比矩阵...")

        J = compute_jacobian_matrix(
            model,
            V_safe_init,
            V_unsafe_init,
            dynamics_model=dynamics_model,
            translator=translator
        )

        print(f"    J 形状: {J.shape}")

        # 5.2 执行修复迭代
        print(f"\n[迭代 {iteration+1}.2] 执行修复迭代...")

        total_fail = (len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init) +
                      len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init))

        if total_fail == 0:
            print("    没有需要修复的区域!")
            loss = 0.0
            grad_norm = 0.0
            k_effective = 0
        else:
            loss, grad_norm, k_effective = repair_iteration(
                model,
                J,
                F_h_positive_in_unsafe_init,
                F_safe_cbf_violation_init,
                F_depth_limit_reached_init,
                F_unsafe_cannot_split_init,
                dynamics_model,
                translator,
                k_rank=500,
                lr=1e-1,
                alpha=0.0,
                tolerance=-1e-12,
                verbose=False
            )

            print(f"    损失: {loss:.4f}")
            print(f"    梯度范数: {grad_norm:.4f}")
            print(f"    实际 rank: {k_effective}")

        # 5.3 保存 PyTorch 模型
        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        print(f"\n[迭代 {iteration+1}.3] 保存 PyTorch 模型: {pytorch_save_path}")

        # 5.4 转换为 ONNX
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired.onnx"
        print(f"[迭代 {iteration+1}.4] 转换为 ONNX: {onnx_path}")
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

        # 5.5 运行验证
        print(f"\n[迭代 {iteration+1}.5] 运行验证...")
        results = verify_model(onnx_path, dynamics_model, max_depth=max_depth)

        # 5.6 计算通过率
        pass_rate, stats = calculate_pass_rate(results)

        print(f"\n[迭代 {iteration+1}.6] 验证结果:")
        print(f"    总样本数: {stats['total']}")
        print(f"    Certified: {stats['certified_pct']:.2f}%")
        print(f"    Uncertified: {stats['uncertified_pct']:.2f}%")
        print(f"    ★ 通过率 (Certified): {pass_rate:.2f}%")

        # 5.7 保存验证结果
        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired.pt"

        print(f"\n[迭代 {iteration+1}.7] 保存验证区域: {verified_regions_path}")

        # 从 results 中提取区域信息并保存
        regions_to_save = {
            'V_safe': results.get('V_safe', V_safe_init),
            'V_unsafe': results.get('V_unsafe', V_unsafe_init),
            'F_h_positive_in_unsafe': results.get('F_h_positive_in_unsafe', F_h_positive_in_unsafe_init),
            'F_safe_cbf_violation': results.get('F_safe_cbf_violation', F_safe_cbf_violation_init),
            'F_depth_limit_reached': results.get('F_depth_limit_reached', F_depth_limit_reached_init),
            'F_unsafe_cannot_split': results.get('F_unsafe_cannot_split', F_unsafe_cannot_split_init),
            'pass_rate': pass_rate,
            'stats': stats,
            'iteration': iteration + 1
        }

        torch.save(regions_to_save, verified_regions_path)

        # 5.8 重新读取更新后的验证区域（用于下一次迭代的修复）
        updated_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired.pt"
        print(f"[迭代 {iteration+1}.8] 重新读取验证区域: {updated_regions_path}")

        updated_data = torch.load(updated_regions_path, map_location=device, weights_only=False)
        V_safe_init = updated_data['V_safe']
        V_unsafe_init = updated_data['V_unsafe']
        F_h_positive_in_unsafe_init = updated_data['F_h_positive_in_unsafe']
        F_safe_cbf_violation_init = updated_data['F_safe_cbf_violation']
        F_depth_limit_reached_init = updated_data['F_depth_limit_reached']
        F_unsafe_cannot_split_init = updated_data['F_unsafe_cannot_split']

        print(f"    更新后 V_safe: {len(V_safe_init)}, V_unsafe: {len(V_unsafe_init)}")
        print(f"    更新后 F_h_positive: {len(F_h_positive_in_unsafe_init)}, "
              f"F_safe_violation: {len(F_safe_cbf_violation_init)}, "
              f"F_depth: {len(F_depth_limit_reached_init)}, "
              f"F_unsafe_split: {len(F_unsafe_cannot_split_init)}")

        # 保存迭代结果用于最终总结
        iteration_results.append({
            'iteration': iteration + 1,
            'loss': loss,
            'grad_norm': grad_norm,
            'rank': k_effective,
            'pass_rate': stats['certified_pct'],
            'f_h_positive': len(F_h_positive_in_unsafe_init),
            'f_safe_violation': len(F_safe_cbf_violation_init),
            'f_depth': len(F_depth_limit_reached_init),
            'f_unsafe_split': len(F_unsafe_cannot_split_init),
        })

    # ========== 6. 最终总结 ==========
    print(f"\n{'='*70}")
    print("迭代修复完成 - 总结")
    print(f"{'='*70}")

    # 打印迭代结果表格
    print(f"\n┌{'─'*8}┬{'─'*15}┬{'─'*15}┬{'─'*8}┬{'─'*12}┐")
    print(f"│ {'迭代':^6} │ {'损失':^13} │ {'梯度范数':^13} │ {'Rank':^6} │ {'通过率':^10} │")
    print(f"├{'─'*8}┼{'─'*15}┼{'─'*15}┼{'─'*8}┼{'─'*12}┤")
    for r in iteration_results:
        print(f"│ {r['iteration']:^6} │ {r['loss']:>13.2f} │ {r['grad_norm']:>13.2f} │ {r['rank']:^6} │ {r['pass_rate']:>10.2f}% │")
    print(f"└{'─'*8}┴{'─'*15}┴{'─'*15}┴{'─'*8}┴{'─'*12}┘")

    # 分析趋势
    model_initial_rate = initial_pass_rate
    final_rate = iteration_results[-1]['pass_rate']
    improvement = final_rate - model_initial_rate

    print(f"\n模型初始通过率: {model_initial_rate:.2f}%")
    print(f"最终通过率: {final_rate:.2f}%")
    print(f"变化: {improvement:+.2f}%")

    if improvement > 0:
        print("✓ 修复有效：通过率提升!")
    elif improvement < 0:
        print("✗ 修复效果负向：通过率下降")
    else:
        print("- 修复效果持平")

    print("\n" + "=" * 70)
    print("演示结束")
    print("=" * 70)


if __name__ == "__main__":
    main()
