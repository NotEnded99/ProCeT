"""
Neural CBF 迭代修复 v4：Randomized Smoothing Jacobian + v3 损失计算

核心改进：
    - J_verified: 使用随机平滑（共享 epsilon 版本）计算
    - 损失计算: 使用 v3 的特征点方法（顶点+重心）替代 RS 梯度估计
    - QP 投影: 复用 v3 的 qp_project_and_update

公式（J_RS）：
    J_RS ≈ (1/(N * sigma^2)) * sum_i[psi(theta + epsilon_i) * epsilon_i]
    epsilon_i ~ N(0, sigma^2 I)
"""

import sys
import os
import random
import argparse
import numpy as np
import torch
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import time

from lbp_neural_cbf.cbf.fossil_dynamics import (
    Barrier1System, Barrier2System, Barrier3System, Barrier4System
)
from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem
from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from lbp_neural_cbf.translators import TorchTranslator

# v4: RS Jacobian（共享 epsilon 版本）
from New_repair.geometry_module_new_v4 import compute_jacobian_rs

# v3: 特征点提取和损失计算
from New_repair.geometry_module_new_v3 import (
    extract_all_feature_points,
)
from New_repair.optimizer_module_v3 import (
    compute_repair_loss_and_grad,
    qp_project_and_update_gd,
    qp_project_and_update
)


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
    certified_pct = results.get('certified_percentage', 0.0)
    uncertified_pct = results.get('uncertified_percentage', 0.0)
    total_samples = results.get('total_samples', 0)
    return certified_pct, {
        'total': total_samples,
        'certified_pct': certified_pct,
        'uncertified_pct': uncertified_pct,
        'certified_percentage': certified_pct
    }


def extract_feature_points_from_regions(
    simplices_list: list,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    从单纯形列表中提取所有特征点（顶点+重心）。

    Args:
        simplices_list: 单纯形列表
        device: 目标设备
        dtype: 数据类型

    Returns:
        feature_points: 形状 [N*(D+2), D]，所有单纯形的特征点展平
    """
    if not simplices_list:
        return torch.empty(0, 2, device=device, dtype=dtype)

    all_feature_points, _ = extract_all_feature_points(
        simplices_list, device=device, dtype=dtype
    )
    N, num_fp, D = all_feature_points.shape
    return all_feature_points.view(N * num_fp, D)


def main():
    # ========== 0. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description='Neural CBF 迭代修复 v4 (RS Jacobian + v3 损失)')
    parser.add_argument('--activation', '-a', type=str, required=True,
                        choices=SUPPORTED_ACTIVATIONS,
                        help='激活函数: Relu, Tanh, Sigmoid')
    parser.add_argument('--system', '-s', type=str, required=True,
                        choices=list(DYNAMICS_SYSTEMS.keys()),
                        help='动力学系统: simple_2d, barr1, barr2, barr3, barr4')
    # RS Jacobian 参数
    parser.add_argument('--rs-n', type=int, default=100,
                        help='随机平滑采样次数 N (default: 100)')
    parser.add_argument('--rs-sigma', type=float, default=0.01,
                        help='随机平滑噪声标准差 sigma (default: 0.01)')
    # 内循环参数
    parser.add_argument('--num-inner-steps', type=int, default=5,
                        help='内循环步数 (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (default: 1e-4)')
    parser.add_argument('--max-depth', type=int, default=12,
                        help='验证最大深度 (default: 12)')
    args = parser.parse_args()

    activation = args.activation
    system_name_key = args.system
    rs_n = args.rs_n
    rs_sigma = args.rs_sigma
    num_inner_steps = args.num_inner_steps
    lr = args.lr
    max_depth = args.max_depth

    # ========== 0. 固定随机数种子 ==========
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print(f"Neural CBF 迭代修复 v4 [RS Jacobian + v3 损失]")
    print(f"  激活={activation}, 系统={system_name_key}")
    print(f"  RS 参数: N={rs_n}, sigma={rs_sigma}")
    print(f"  内循环: {num_inner_steps} 步, lr={lr}")
    print("=" * 70)

    num_iterations = 10

    # ========== 1. 加载动力学系统 ==========
    dynamics_class = DYNAMICS_SYSTEMS[system_name_key]
    dynamics_model = dynamics_class(alpha=1.0)

    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Invalid activation: {activation}")
    dynamics_model.activation_fnc = activation

    print(f"\n[1] 动力学系统: {dynamics_model.system_name}")
    print(f"    激活函数: {activation}")
    print(f"    输入维度: {dynamics_model.input_dim}")
    print(f"    隐藏层大小: {dynamics_model.hidden_sizes}")

    # ========== 2. 加载初始神经网络 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    print(f"    初始 F_h_positive_in_unsafe: {len(F_h_positive_in_unsafe_init)}")
    print(f"    初始 F_safe_cbf_violation: {len(F_safe_cbf_violation_init)}")
    print(f"    初始 F_depth_limit_reached: {len(F_depth_limit_reached_init)}")
    print(f"    初始 F_unsafe_cannot_split: {len(F_unsafe_cannot_split_init)}")
    print(f"    总需修复区域数: {total_fail}")
    print(f"    初始验证通过率: {initial_pass_rate:.2f}%")

    # ========== 4. 设置 translator ==========
    translator = TorchTranslator(device=device)

    # ========== 5. 迭代修复 ==========
    iteration_results = []

    print(f"\n{'='*70}")
    print(f"开始 {num_iterations} 次迭代修复 (v4: RS Jacobian + v3 损失)")
    print(f"{'='*70}")

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"迭代 {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        total_fail = (len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init) +
                      len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init))

        print(f"  失败区域总数: {total_fail}, V_safe: {len(V_safe_init)}, V_unsafe: {len(V_unsafe_init)}")

        # 如果没有失败区域，提前终止
        if total_fail == 0:
            print(f"\n  === 所有区域已验证通过！提前终止 ===")
            certified_percentage = 100.0
            stats = {
                'total': 0,
                'certified_pct': 100.0,
                'uncertified_pct': 0.0,
                'certified_percentage': 100.0
            }

            pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v4.pth"
            torch.save(model.state_dict(), pytorch_save_path)
            print(f"\n[迭代 {iteration+1}] 保存 PyTorch 模型: {pytorch_save_path}")

            onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v4.onnx"
            pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)
            print(f"[迭代 {iteration+1}] 转换为 ONNX: {onnx_path}")

            verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired_v4.pt"
            regions_to_save = {
                'V_safe': V_safe_init,
                'V_unsafe': V_unsafe_init,
                'F_h_positive_in_unsafe': F_h_positive_in_unsafe_init,
                'F_safe_cbf_violation': F_safe_cbf_violation_init,
                'F_depth_limit_reached': F_depth_limit_reached_init,
                'F_unsafe_cannot_split': F_unsafe_cannot_split_init,
                'Certified percentage': certified_percentage,
                'stats': stats,
                'iteration': iteration + 1
            }
            torch.save(regions_to_save, verified_regions_path)
            print(f"[迭代 {iteration+1}] 保存验证区域: {verified_regions_path}")

            iteration_results.append({
                'iteration': iteration + 1,
                'loss': 0.0,
                'update_norm': 0.0,
                'certified_percentage': certified_percentage,
                'f_h_positive': 0,
                'f_safe_violation': 0,
                'f_depth': 0,
                'f_unsafe_split': 0,
            })

            for rem_iter in range(iteration + 1, num_iterations):
                iteration_results.append({
                    'iteration': rem_iter + 1,
                    'loss': 0.0,
                    'update_norm': 0.0,
                    'certified_percentage': 100.0,
                    'f_h_positive': 0,
                    'f_safe_violation': 0,
                    'f_depth': 0,
                    'f_unsafe_split': 0,
                })
            print(f"\n  修复完成，提前终止。")
            print(f"  ★ 最终通过率: {certified_percentage:.2f}%")
            break

        # ========== 5.1 提取特征点（v3 方法：顶点+重心）==========
        print(f"\n[迭代 {iteration+1}.1] 提取特征点（顶点+重心）...")

        # V_safe 特征点
        verified_safe_feature_points = extract_feature_points_from_regions(
            V_safe_init, device=device
        )
        print(f"    V_safe 特征点: {verified_safe_feature_points.shape[0]} 个")

        # V_unsafe 特征点
        verified_unsafe_feature_points = extract_feature_points_from_regions(
            list(V_unsafe_init), device=device
        )
        print(f"    V_unsafe 特征点: {verified_unsafe_feature_points.shape[0]} 个")

        # F_h_positive_in_unsafe 特征点
        failed_unsafe_feature_points = extract_feature_points_from_regions(
            F_h_positive_in_unsafe_init, device=device
        )
        print(f"    F_h 违规特征点: {failed_unsafe_feature_points.shape[0]} 个")

        # CBF 违规区域特征点
        failed_safe_simplices = (
            list(F_safe_cbf_violation_init) +
            list(F_depth_limit_reached_init) +
            list(F_unsafe_cannot_split_init)
        )
        failed_safe_feature_points = extract_feature_points_from_regions(
            failed_safe_simplices, device=device
        )
        print(f"    CBF 违规特征点: {failed_safe_feature_points.shape[0]} 个")

        # ========== 5.2 计算 RS Jacobian（J_verified）==========
        print(f"\n[迭代 {iteration+1}.2] 计算随机平滑 Jacobian (N={rs_n}, sigma={rs_sigma})...")

        J = compute_jacobian_rs(
            model,
            V_safe_init,
            V_unsafe_init,
            dynamics_model=dynamics_model,
            translator=translator,
            N=rs_n,
            sigma=rs_sigma,
        )
        print(f"    J_RS 形状: {J.shape}")

        # ========== 5.3 内循环修复（v3 损失计算 + QP 投影）==========
        print(f"\n[迭代 {iteration+1}.3] 内循环修复 ({num_inner_steps} 步)...")

        inner_history = []
        for inner_step in range(num_inner_steps):
            # 计算修复损失（v3 方法：特征点 + autograd）
            t0 = time.perf_counter()
            loss_val, g_F = compute_repair_loss_and_grad(
                model=model,
                dynamics_model=dynamics_model,
                failed_safe_feature_points=failed_safe_feature_points,
                failed_unsafe_feature_points=failed_unsafe_feature_points,
                margin=0.1,
                cbf_margin=0.0,
                beta=5.0,
                grad_clip_norm=10.0,
                verbose=False,
                translator=translator,
            )
            t1 = time.perf_counter()

            # QP 投影更新
            g_raw_norm, update_norm, active = qp_project_and_update_gd(
                model=model,
                g_raw=g_F,
                J_verified=J,
                lr=lr,
                verbose=False,
            )
            t2 = time.perf_counter()

            inner_history.append({
                'step': inner_step + 1,
                'loss': loss_val,
                'g_raw_norm': g_raw_norm,
                'update_norm': update_norm,
                'active_constraints': active,
                't_loss': t1 - t0,
                't_qp': t2 - t1,
            })

            if inner_step == num_inner_steps - 1 or (inner_step + 1) % 5 == 0:
                print(f"    [内步 {inner_step+1}/{num_inner_steps}] "
                      f"loss={loss_val:.6f}, |g|={g_raw_norm:.4f}, "
                      f"|d|={update_norm:.6f}, active={active}, "
                      f"t_loss={t1-t0:.3f}s, t_qp={t2-t1:.3f}s")

        # ========== 5.4 保存 PyTorch 模型 ==========
        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v4.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        print(f"\n[迭代 {iteration+1}.4] 保存 PyTorch 模型: {pytorch_save_path}")

        # ========== 5.5 转换为 ONNX ==========
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v4.onnx"
        print(f"[迭代 {iteration+1}.5] 转换为 ONNX: {onnx_path}")
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

        # ========== 5.6 运行验证 ==========
        print(f"\n[迭代 {iteration+1}.6] 运行验证...")
        results = verify_model(onnx_path, dynamics_model, max_depth=max_depth)

        # ========== 5.7 计算通过率 ==========
        certified_percentage, stats = calculate_pass_rate(results)

        print(f"\n[迭代 {iteration+1}.7] 验证结果:")
        print(f"    总样本数: {stats['total']}")
        print(f"    Certified: {stats['certified_pct']:.2f}%")
        print(f"    Uncertified: {stats['uncertified_pct']:.2f}%")
        print(f"    ★ 通过率 (Certified): {certified_percentage:.2f}%")

        # ========== 5.8 保存验证结果 ==========
        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired_v4.pt"
        print(f"\n[迭代 {iteration+1}.8] 保存验证区域: {verified_regions_path}")

        regions_to_save = {
            'V_safe': results.get('V_safe', V_safe_init),
            'V_unsafe': results.get('V_unsafe', V_unsafe_init),
            'F_h_positive_in_unsafe': results.get('F_h_positive_in_unsafe', F_h_positive_in_unsafe_init),
            'F_safe_cbf_violation': results.get('F_safe_cbf_violation', F_safe_cbf_violation_init),
            'F_depth_limit_reached': results.get('F_depth_limit_reached', F_depth_limit_reached_init),
            'F_unsafe_cannot_split': results.get('F_unsafe_cannot_split', F_unsafe_cannot_split_init),
            'Certified percentage': certified_percentage,
            'stats': stats,
            'iteration': iteration + 1
        }

        torch.save(regions_to_save, verified_regions_path)

        # ========== 5.9 重新读取更新后的验证区域 ==========
        updated_regions_path = verified_regions_path
        print(f"[迭代 {iteration+1}.9] 重新读取验证区域: {updated_regions_path}")

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

        # 保存迭代结果
        iteration_results.append({
            'iteration': iteration + 1,
            'loss': inner_history[-1]['loss'] if inner_history else 0.0,
            'update_norm': inner_history[-1]['update_norm'] if inner_history else 0.0,
            'certified_percentage': stats['certified_percentage'],
            'f_h_positive': len(F_h_positive_in_unsafe_init),
            'f_safe_violation': len(F_safe_cbf_violation_init),
            'f_depth': len(F_depth_limit_reached_init),
            'f_unsafe_split': len(F_unsafe_cannot_split_init),
        })

    # ========== 6. 最终总结 ==========
    print(f"\n{'='*70}")
    print("迭代修复完成 - 总结 (v4: RS Jacobian + v3 损失)")
    print(f"{'='*70}")

    print(f"\n┌{'─'*8}┬{'─'*15}┬{'─'*15}┬{'─'*8}┐")
    print(f"│ {'迭代':^6} │ {'损失':^13} │ {'更新范数':^13} │ {'通过率':^10} │")
    print(f"├{'─'*8}┼{'─'*15}┼{'─'*15}┼{'─'*8}┤")
    for r in iteration_results:
        print(f"│ {r['iteration']:^6} │ {r['loss']:>13.4f} │ {r['update_norm']:>13.4f} │ {r['certified_percentage']:>10.2f}% │")
    print(f"└{'─'*8}┴{'─'*15}┴{'─'*15}┴{'─'*8}┘")

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

    # ========== 7. 保存结果到 JSON ==========
    import json
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_v4")
    os.makedirs(results_dir, exist_ok=True)

    run_result = {
        'system': system_name_key,
        'activation': activation,
        'method': 'RS Jacobian + v3 Feature Point Loss',
        'rs_n': rs_n,
        'rs_sigma': rs_sigma,
        'num_inner_steps': num_inner_steps,
        'lr': lr,
        'max_depth': max_depth,
        'initial_pass_rate': model_initial_rate,
        'final_pass_rate': final_rate,
        'improvement': improvement,
        'num_iterations': num_iterations,
        'initial_regions': {
            'V_safe': len(V_safe_init) if 'V_safe_init' in dir() else 0,
            'V_unsafe': len(V_unsafe_init) if 'V_unsafe_init' in dir() else 0,
        },
        'iteration_results': iteration_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v4.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    print(f"\n[7] 单次结果已保存: {result_file}")

    print("\n" + "=" * 70)
    print("演示结束")
    print("=" * 70)


if __name__ == "__main__":
    main()
