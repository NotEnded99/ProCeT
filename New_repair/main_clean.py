"""
直接用下界的梯度下降方法来修复
"""

import sys
import os
import random
import argparse
import json
from datetime import datetime
import numpy as np
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lbp_neural_cbf.cbf.fossil_dynamics import (
    Barrier1System, Barrier2System, Barrier3System, Barrier4System
)
from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem
from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from lbp_neural_cbf.translators import TorchTranslator

from New_repair.optimizer_module import compute_repair_loss_and_grad, inner_loop_repair

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
        save_verification_regions = False,
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
    certified_percentage = certified_pct

    stats = {
        'total': total_samples,
        'certified_pct': certified_pct,
        'uncertified_pct': uncertified_pct,
        'certified_percentage': certified_percentage
    }

    return certified_percentage, stats


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

    num_iterations = 10
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


        # 5.2 执行内循环 Mini-batch 修复
        print(f"\n[迭代 {iteration+1}.1] 执行内循环 Mini-batch 修复...")

        total_fail = len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init) +\
                      len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init)

        print("total_fail", total_fail)
        if total_fail == 0:
            print("    没有需要修复的区域!")
            loss = 0.0
        else:
            # 内循环参数
            num_inner_steps = 10   # 内循环迭代次数
            batch_ratio = 0.2      # 每次采样比例
            lr = 1e-3

            inner_history = inner_loop_repair(
                model=model,
                F_h_positive_in_unsafe=F_h_positive_in_unsafe_init,
                F_safe_cbf_violation=F_safe_cbf_violation_init,
                F_depth_limit_reached=F_depth_limit_reached_init,
                F_unsafe_cannot_split=F_unsafe_cannot_split_init,
                dynamics_model=dynamics_model,
                translator=translator,
                num_inner_steps=num_inner_steps,
                batch_ratio=batch_ratio,
                lr=lr,
                verbose=True,
                seed=42 + iteration,  # 每次外迭代用不同种子，保证可复现
            )

            # 取内循环最后一次的 loss 作为本轮指标
            if inner_history:
                loss = inner_history[-1]['loss']
                print(f"    内循环完成: {len(inner_history)} 步, 最终 loss={loss:.6f}")
            else:
                loss = 0.0


        # 5.3 保存 PyTorch 模型
        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_clean_repaired.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        print(f"\n[迭代 {iteration+1}.3] 保存 PyTorch 模型: {pytorch_save_path}")

        # 5.4 转换为 ONNX
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_clean_repaired.onnx"
        print(f"[迭代 {iteration+1}.4] 转换为 ONNX: {onnx_path}")
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

        # 5.5 运行验证
        print(f"\n[迭代 {iteration+1}.5] 运行验证...")
        results = verify_model(onnx_path, dynamics_model, max_depth=max_depth)

        # print(results)

        # 5.6 计算通过率
        certified_percentage, stats = calculate_pass_rate(results)

        print(f"\n[迭代 {iteration+1}.6] 验证结果:")
        print(f"    总样本数: {stats['total']}")
        print(f"    Certified: {stats['certified_pct']:.2f}%")
        print(f"    Uncertified: {stats['uncertified_pct']:.2f}%")
        print(f"    ★ 通过率 (Certified): {certified_percentage:.2f}%")

        # 5.7 保存验证结果
        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_clean_repaired.pt"

        print(f"\n[迭代 {iteration+1}.7] 保存验证区域: {verified_regions_path}")

        # 从 results 中提取区域信息并保存
        regions_to_save = {
            'V_safe': results.get('V_safe', V_safe_init),
            'V_unsafe': results.get('V_unsafe', V_unsafe_init),
            'F_h_positive_in_unsafe': results.get('F_h_positive_in_unsafe', F_h_positive_in_unsafe_init),
            'F_safe_cbf_violation': results.get('F_safe_cbf_violation', F_safe_cbf_violation_init),
            'F_depth_limit_reached': results.get('F_depth_limit_reached', F_depth_limit_reached_init),
            'F_unsafe_cannot_split': results.get('F_unsafe_cannot_split', F_unsafe_cannot_split_init),
            'certified_percentage': certified_percentage,
            'stats': stats,
            'iteration': iteration + 1
        }

        torch.save(regions_to_save, verified_regions_path)

        # 5.8 重新读取更新后的验证区域（用于下一次迭代的修复）
        updated_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_clean_repaired.pt"
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
            'certified_percentage': certified_percentage,
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
    print(f"│ {'迭代':^6} │ {'损失':^13} │  {'通过率':^10} │")
    print(f"├{'─'*8}┼{'─'*15}┼{'─'*15}┼{'─'*8}┼{'─'*12}┤")
    for r in iteration_results:
        print(f"│ {r['iteration']:^6} │ {r['loss']:>13.2f} │ {r['certified_percentage']:>10.2f}% │")
    print(f"└{'─'*8}┴{'─'*15}┴{'─'*15}┴{'─'*8}┴{'─'*12}┘")

    # 分析趋势
    model_initial_rate = initial_pass_rate
    final_rate = iteration_results[-1]['certified_percentage']
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

    # ========== 7. 保存单次运行结果到 nr_results_clean/ ==========
    import json
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_clean")
    os.makedirs(results_dir, exist_ok=True)

    # 保存为 JSON（便于后续读取汇总）
    run_result = {
        'system': system_name_key,
        'activation': activation,
        'initial_pass_rate': model_initial_rate,
        'final_pass_rate': final_rate,
        'improvement': improvement,
        'num_iterations': num_iterations,
        'max_depth': max_depth,
        'initial_regions': {
            'V_safe': len(V_safe_init) if 'V_safe_init' in dir() else 0,
            'V_unsafe': len(V_unsafe_init) if 'V_unsafe_init' in dir() else 0,
            'F_h_positive_in_unsafe': len(F_h_positive_in_unsafe_init),
            'F_safe_cbf_violation': len(F_safe_cbf_violation_init),
            'F_depth_limit_reached': len(F_depth_limit_reached_init),
            'F_unsafe_cannot_split': len(F_unsafe_cannot_split_init),
        },
        'final_regions': {
            'V_safe': iteration_results[-1].get('f_h_positive', 0),
            'V_unsafe': iteration_results[-1].get('f_safe_violation', 0),
            'F_h_positive_in_unsafe': iteration_results[-1].get('f_h_positive', 0),
            'F_safe_cbf_violation': iteration_results[-1].get('f_safe_violation', 0),
            'F_depth_limit_reached': iteration_results[-1].get('f_depth', 0),
            'F_unsafe_cannot_split': iteration_results[-1].get('f_unsafe_split', 0),
        },
        'iteration_results': iteration_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    print(f"\n[7] 单次结果已保存: {result_file}")

    print("\n" + "=" * 70)
    print("演示结束")
    print("=" * 70)


if __name__ == "__main__":
    main()
