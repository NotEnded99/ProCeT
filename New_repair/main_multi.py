"""
多系统迭代修复脚本：支持 Barrier1-4 所有系统的修复-验证循环

流程:
1. 遍历所有指定的动力学系统
2. 对每个系统执行修复-验证循环
3. 保存所有结果并汇总对比

用法:
    python3 New_repair/main_multi.py
"""

import sys
import os
import random
import numpy as np
import torch
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lbp_neural_cbf.cbf.fossil_dynamics import (
    Barrier1System, Barrier2System, Barrier3System, Barrier4System
)
from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from lbp_neural_cbf.translators import TorchTranslator

from New_repair.geometry_module import compute_jacobian_matrix
from New_repair.optimizer_module import repair_iteration, repair_loop


# 支持的动力学系统映射
DYNAMICS_SYSTEMS = {
    # 'Barrier1': Barrier1System,
    'Barrier2': Barrier2System,
    'Barrier3': Barrier3System,
    'Barrier4': Barrier4System,
}


def pytorch_to_onnx(model, onnx_path, input_dim=2):
    """将 PyTorch 模型转换为 ONNX 格式。"""
    device = next(model.parameters()).device
    model.eval()
    dummy_input = torch.randn(1, input_dim, device=device)
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
    """验证模型并返回验证结果。"""
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
    """计算验证通过率。"""
    certified_pct = results.get('certified_percentage', 0.0)
    uncertified_pct = results.get('uncertified_percentage', 0.0)
    total_samples = results.get('total_samples', 0)

    pass_rate = certified_pct
    stats = {
        'total': total_samples,
        'certified_pct': certified_pct,
        'uncertified_pct': uncertified_pct,
        'pass_rate': pass_rate
    }
    return pass_rate, stats


def run_single_system_experiment(system_name, dynamics_class, num_iterations=5, max_depth=13):
    """
    对单个动力学系统执行完整修复实验。

    Args:
        system_name: 系统名称 (e.g., 'Barrier1')
        dynamics_class: 动力学系统类
        num_iterations: 迭代次数
        max_depth: 验证最大深度

    Returns:
        实验结果字典
    """
    # ========== 固定随机数种子 ==========
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'#'*70}")
    print(f"# {system_name} 实验开始")
    print(f"{'#'*70}")

    # ========== 1. 加载动力学系统 ==========
    dynamics_model = dynamics_class(alpha=1.0)
    print(f"\n[1] 动力学系统: {dynamics_model.system_name}")
    print(f"    输入维度: {dynamics_model.input_dim}")
    print(f"    隐藏层大小: {dynamics_model.hidden_sizes}")

    # ========== 2. 加载初始神经网络 ==========
    device = torch.device('cpu')
    model_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.pth"

    print(f"\n[2] 加载初始神经网络: {model_path}")

    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"    参数数量: {num_params}")

    # ========== 2.1 计算初始模型验证通过率 ==========
    print(f"\n[2.1] 计算初始模型验证通过率...")
    onnx_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.onnx"
    initial_results = verify_model(onnx_path, dynamics_model, max_depth=max_depth)
    initial_pass_rate, initial_stats = calculate_pass_rate(initial_results)

    print(f"\n    初始模型验证结果:")
    print(f"    总样本数: {initial_stats['total']}")
    print(f"    Certified: {initial_stats['certified_pct']:.2f}%")
    print(f"    Uncertified: {initial_stats['uncertified_pct']:.2f}%")
    print(f"    ★ 初始通过率: {initial_pass_rate:.2f}%")

    # ========== 3. 读取初始验证区域 ==========
    regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}.pt"

    print(f"\n[3] 读取初始验证区域: {regions_path}")

    regions_data = torch.load(regions_path, map_location=device, weights_only=False)

    V_safe_init = regions_data['V_safe']
    V_unsafe_init = regions_data['V_unsafe']
    F_safe_init = regions_data['F_safe']
    F_unsafe_init = regions_data['F_unsafe']

    print(f"    初始 V_safe: {len(V_safe_init)}")
    print(f"    初始 V_unsafe: {len(V_unsafe_init)}")
    print(f"    初始 F_safe: {len(F_safe_init)}  <- 需要修复")
    print(f"    初始 F_unsafe: {len(F_unsafe_init)}  <- 需要修复")

    # ========== 4. 设置 translator ==========
    translator = TorchTranslator(device=device)

    # ========== 5. 迭代修复 ==========
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

        if len(F_safe_init) == 0 and len(F_unsafe_init) == 0:
            print("    没有需要修复的区域!")
            loss = 0.0
            grad_norm = 0.0
            k_effective = 0
        else:
            # loss, grad_norm, k_effective = repair_iteration(
            #     model,
            #     J,
            #     F_safe_init,
            #     F_unsafe_init,
            #     dynamics_model,
            #     translator,
            #     k_rank=500,
            #     lr=1e-3,
            #     alpha=0.0,
            #     tolerance=-1e-12,
            #     verbose=False
            # )

            # print(f"    损失: {loss:.4f}")
            # print(f"    梯度范数: {grad_norm:.4f}")
            # print(f"    实际 rank: {k_effective}")

            loss, grad_norm, k_effective = repair_loop(
                    model=model,
                    J=J,
                    F_safe=F_safe_init,
                    F_unsafe=F_unsafe_init,
                    dynamics_model=dynamics_model,
                    translator=translator,
                    max_iters=10,       # 最多跑 10 轮
                    lr=1e-4,            # 可以根据迭代情况微调学习率
                    verbose=False       # 如果设为 True 会打印极其详细的单步信息
                )
            
        # 5.3 保存 PyTorch 模型
        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_cbf_repaired_iter{iteration+1}.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        print(f"\n[迭代 {iteration+1}.3] 保存 PyTorch 模型: {pytorch_save_path}")

        # 5.4 转换为 ONNX
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_cbf_repaired_iter{iteration+1}.onnx"
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
        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_iter{iteration+1}.pt"

        print(f"\n[迭代 {iteration+1}.7] 保存验证区域: {verified_regions_path}")

        regions_to_save = {
            'V_safe': results.get('V_safe', V_safe_init),
            'V_unsafe': results.get('V_unsafe', V_unsafe_init),
            'F_safe': results.get('F_safe', F_safe_init),
            'F_unsafe': results.get('F_unsafe', F_unsafe_init),
            'pass_rate': pass_rate,
            'stats': stats,
            'iteration': iteration + 1
        }

        torch.save(regions_to_save, verified_regions_path)

        # 5.8 重新读取更新后的验证区域
        updated_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}.pt"
        print(f"[迭代 {iteration+1}.8] 重新读取验证区域: {updated_regions_path}")

        updated_data = torch.load(updated_regions_path, map_location=device, weights_only=False)
        V_safe_init = updated_data['V_safe']
        V_unsafe_init = updated_data['V_unsafe']
        F_safe_init = updated_data['F_safe']
        F_unsafe_init = updated_data['F_unsafe']

        print(f"    更新后 V_safe: {len(V_safe_init)}, V_unsafe: {len(V_unsafe_init)}")
        print(f"    更新后 F_safe: {len(F_safe_init)}, F_unsafe: {len(F_unsafe_init)}")

        # 保存迭代结果
        iteration_results.append({
            'iteration': iteration + 1,
            'loss': loss,
            'grad_norm': grad_norm,
            'rank': k_effective,
            'pass_rate': stats['certified_pct'],
            'v_safe': len(V_safe_init),
            'v_unsafe': len(V_unsafe_init),
            'f_safe': len(F_safe_init),
            'f_unsafe': len(F_unsafe_init),
        })

    # ========== 6. 最终总结 ==========
    print(f"\n{'='*70}")
    print(f"{system_name} 迭代修复完成 - 总结")
    print(f"{'='*70}")

    # 打印迭代结果表格
    print(f"\n┌{'─'*8}┬{'─'*15}┬{'─'*15}┬{'─'*8}┬{'─'*12}┐")
    print(f"│ {'迭代':^6} │ {'损失':^13} │ {'梯度范数':^13} │ {'Rank':^6} │ {'通过率':^10} │")
    print(f"├{'─'*8}┼{'─'*15}┼{'─'*15}┼{'─'*8}┼{'─'*12}┤")
    for r in iteration_results:
        print(f"│ {r['iteration']:^6} │ {r['loss']:>13.2f} │ {r['grad_norm']:>13.2f} │ {r['rank']:^6} │ {r['pass_rate']:>10.2f}% │")
    print(f"└{'─'*8}┴{'─'*15}┴{'─'*15}┴{'─'*8}┴{'─'*12}┘")

    final_rate = iteration_results[-1]['pass_rate'] if iteration_results else 0
    improvement = final_rate - initial_pass_rate

    print(f"\n模型初始通过率: {initial_pass_rate:.2f}%")
    print(f"最终通过率: {final_rate:.2f}%")
    print(f"变化: {improvement:+.2f}%")

    if improvement > 0:
        print("✓ 修复有效：通过率提升!")
    elif improvement < 0:
        print("✗ 修复效果负向：通过率下降")
    else:
        print("- 修复效果持平")

    # 返回实验结果
    return {
        'system_name': system_name,
        'initial_pass_rate': initial_pass_rate,
        'final_pass_rate': final_rate,
        'improvement': improvement,
        'iteration_results': iteration_results,
    }


def main():
    print("=" * 70)
    print("Neural CBF 多系统迭代修复演示 (Barrier1-4)")
    print("=" * 70)

    # 实验配置
    num_iterations = 2
    max_depth = 13

    # 所有系统列表
    systems_to_run = list(DYNAMICS_SYSTEMS.keys())

    print(f"\n将运行以下系统: {systems_to_run}")
    print(f"迭代次数: {num_iterations}")
    print(f"最大验证深度: {max_depth}")

    # 存储所有系统结果
    all_results = []

    # 遍历每个系统执行实验
    for system_name in systems_to_run:
        dynamics_class = DYNAMICS_SYSTEMS[system_name]
        result = run_single_system_experiment(
            system_name, dynamics_class, num_iterations, max_depth
        )
        all_results.append(result)

    # ========== 汇总所有系统结果 ==========
    print(f"\n{'#'*70}")
    print(f"# 所有系统实验汇总")
    print(f"{'#'*70}")

    # 打印汇总表格
    print(f"\n┌{'─'*20}┬{'─'*15}┬{'─'*15}┬{'─'*15}┐")
    print(f"│ {'系统':^18} │ {'初始通过率':^13} │ {'最终通过率':^13} │ {'变化':^13} │")
    print(f"├{'─'*20}┼{'─'*15}┼{'─'*15}┼{'─'*15}┤")
    for r in all_results:
        print(f"│ {r['system_name']:^18} │ {r['initial_pass_rate']:>12.2f}% │ {r['final_pass_rate']:>12.2f}% │ {r['improvement']:>+12.2f}% │")
    print(f"└{'─'*20}┴{'─'*15}┴{'─'*15}┴{'─'*15}┘")

    # 计算平均提升
    avg_improvement = sum(r['improvement'] for r in all_results) / len(all_results)
    print(f"\n平均通过率变化: {avg_improvement:+.2f}%")

    # 保存汇总结果
    summary_path = "New_repair/regions/experiment_summary.pt"
    torch.save({
        'all_results': all_results,
        'timestamp': datetime.now().isoformat(),
    }, summary_path)
    print(f"\n汇总结果已保存: {summary_path}")

    print("\n" + "=" * 70)
    print("所有实验完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
