"""
main_v2.py: Neural CBF 迭代修复 v2

核心逻辑: 对抗采样 (Adversarial Sampling) + 采样估计真实梯度 


相比 v1 的改进:
- 不再使用 LBP/McCormick 松弛的梯度（保守且有梯度爆炸问题）
- 在单纯形内用 Dirichlet 分布采样，提取对抗最坏点
- 用真实 autograd 梯度代替 LBP 近似
- 线搜索安全检查确保不破坏已验证区域
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

from New_repair.geometry_module_new_v2 import (
    sample_simplices_batched,
    find_worst_case_points,
    compute_cbf_condition_simple,
)
from New_repair.optimizer_module_v2 import (
    compute_jacobian_at_worst_points_simple,
    compute_sampled_repair_loss_and_grad,
    qp_project_and_update,
)


# 支持的动力学系统映射
DYNAMICS_SYSTEMS = {
    'simple_2d': Simple2DSystem,
    'barr1': Barrier1System,
    'barr2': Barrier2System,
    'barr3': Barrier3System,
    'barr4': Barrier4System,
}

SUPPORTED_ACTIVATIONS = ['Relu', 'Tanh', 'Sigmoid']


def pytorch_to_onnx(model, onnx_path, input_dim=2):
    """导出 PyTorch 模型为 ONNX"""
    device = next(model.parameters()).device
    model.eval()
    dummy_input = torch.randn(1, input_dim, device=device)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )


def verify_model(model_path, dynamics_model, max_depth=10):
    """调用 verify_cbf 验证模型"""
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
    """从验证结果中计算通过率"""
    certified_pct = results.get('certified_percentage', 0.0)
    stats = {
        'total': results.get('total_samples', 0),
        'certified_pct': certified_pct,
        'uncertified_pct': results.get('uncertified_percentage', 0.0),
        'certified_percentage': certified_pct,
    }
    return certified_pct, stats


def main():
    # ========== 0. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description='Neural CBF 迭代修复 v2')
    parser.add_argument('--activation', '-a', type=str, required=True,
                        choices=SUPPORTED_ACTIVATIONS,
                        help='激活函数: Relu, Tanh, Sigmoid')
    parser.add_argument('--system', '-s', type=str, required=True,
                        choices=list(DYNAMICS_SYSTEMS.keys()),
                        help='动力学系统: simple_2d, barr1, barr2, barr3, barr4')
    parser.add_argument('--iterations', '-i', type=int, default=10,
                        help='迭代次数（默认 10）')
    parser.add_argument('--num_samples', '-k', type=int, default=50,
                        help='每个单纯形采样数（默认 500）')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='内循环步数（默认 10）')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率（默认 1e-4）')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='验证最大深度（默认 10）')
    args = parser.parse_args()

    activation = args.activation
    system_name_key = args.system
    num_iterations = args.iterations
    num_samples = args.num_samples
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
    print(f"Neural CBF 迭代修复 v2 [激活={activation}, 系统={system_name_key}]")
    print(f"  采样数/单纯形: {num_samples}, 内迭代: {num_inner_steps}, lr: {lr}")
    print("=" * 70)

    # ========== 1. 加载动力学系统 ==========
    dynamics_class = DYNAMICS_SYSTEMS[system_name_key]
    dynamics_model = dynamics_class(alpha=1.0)

    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Invalid activation: {activation}")
    dynamics_model.activation_fnc = activation

    print(f"\n[1] 动力学系统: {dynamics_model.system_name}")
    print(f"    激活函数: {activation}")
    print(f"    输入维度: {dynamics_model.input_dim}")
    print(f"    控制维度: {dynamics_model.control_dim}")

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

    print(f"    V_safe: {len(V_safe_init)}, V_unsafe: {len(V_unsafe_init)}")
    print(f"    F_h_positive_in_unsafe: {len(F_h_positive_in_unsafe_init)}")
    print(f"    F_safe_cbf_violation: {len(F_safe_cbf_violation_init)}")
    print(f"    F_depth_limit_reached: {len(F_depth_limit_reached_init)}")
    print(f"    F_unsafe_cannot_split: {len(F_unsafe_cannot_split_init)}")
    print(f"    总需修复区域数: {total_fail}")
    print(f"    初始验证通过率: {initial_pass_rate:.2f}%")

    # ========== 4. 初始化 translator ==========
    translator = TorchTranslator(device=device)

    # ========== 5. 迭代修复 ==========
    iteration_results = []

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"迭代 {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        total_fail = (len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init) +
                      len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init))

        print(f"  失败区域总数: {total_fail}")

        # 提前终止：无失败区域
        if total_fail == 0:
            print(f"\n  === 所有区域已验证通过，提前终止 ===")
            certified_percentage = 100.0
            stats = {'total': 0, 'certified_pct': 100.0, 'uncertified_pct': 0.0,
                     'certified_percentage': 100.0}

            _save_artifacts(model, dynamics_model, iteration, certified_percentage,
                           stats, V_safe_init, V_unsafe_init,
                           F_h_positive_in_unsafe_init, F_safe_cbf_violation_init,
                           F_depth_limit_reached_init, F_unsafe_cannot_split_init)

            iteration_results.append({
                'iteration': iteration + 1, 'loss': 0.0, 'update_norm': 0.0,
                'certified_percentage': 100.0,
                'f_h_positive': 0, 'f_safe_violation': 0,
                'f_depth': 0, 'f_unsafe_split': 0,
            })

            for rem in range(iteration + 1, num_iterations):
                iteration_results.append({
                    'iteration': rem + 1, 'loss': 0.0, 'update_norm': 0.0,
                    'certified_percentage': 100.0,
                    'f_h_positive': 0, 'f_safe_violation': 0,
                    'f_depth': 0, 'f_unsafe_split': 0,
                })
            break

        # ---------- 5.1 找最坏点 ----------
        print(f"\n[迭代 {iteration+1}.1] 采样并找最坏点 (num_samples={num_samples})...")

        # V_safe -> safe_worst（cbf_condition 最小的点）
        safe_worst_points, safe_worst_values = find_worst_case_points(
            model=model,
            dynamics_model=dynamics_model,
            simplices_list=V_safe_init,
            num_samples=num_samples,
            region_type='safe',
            device=device,
        )
        print(f"    V_safe 最坏点: {safe_worst_points.shape[0]} 个")

        # F_h_positive_in_unsafe -> unsafe_worst（h(x) 最大的点，用于修复损失）
        unsafe_worst_points, unsafe_worst_values = find_worst_case_points(
            model=model,
            dynamics_model=dynamics_model,
            simplices_list=F_h_positive_in_unsafe_init,
            num_samples=num_samples,
            region_type='unsafe',
            device=device,
        )

        print(f"    F_h 最坏点: {unsafe_worst_points.shape[0]} 个")

        # V_unsafe_init -> unsafe_jac_points（用于 J_unsafe 计算，所有 unsafe 区域）
        unsafe_jac_points, _ = find_worst_case_points(
            model=model,
            dynamics_model=dynamics_model,
            simplices_list=list(V_unsafe_init),
            num_samples=num_samples,
            region_type='unsafe',
            device=device,
        )
        print(f"    V_unsafe J_采样点: {unsafe_jac_points.shape[0]} 个")

        # F_safe_cbf_violation + F_depth_limit_reached + F_unsafe_cannot_split -> cbf_worst
        failed_safe_simplices = (
            list(F_safe_cbf_violation_init) +
            list(F_depth_limit_reached_init) +
            list(F_unsafe_cannot_split_init)
        )
        cbf_worst_points, cbf_worst_values = find_worst_case_points(
            model=model,
            dynamics_model=dynamics_model,
            simplices_list=failed_safe_simplices,
            num_samples=num_samples,
            region_type='safe',
            device=device,
        )
        print(f"    CBF 违规最坏点: {cbf_worst_points.shape[0]} 个")

        # ---------- 5.2 计算已验证区域的真实雅可比 ----------
        # V_safe: ∂cbf/∂θ，QP 约束方向不变（J @ d >= 0：cbf 不减小）
        # V_unsafe: ∂h/∂θ，但约束方向取反（因为要的是 h 减小，J_unsafe @ d >= 0 => -∂h/∂θ @ d >= 0 => ∂h/∂θ @ d <= 0）
        print(f"\n[迭代 {iteration+1}.2] 计算真实雅可比矩阵...")
        J_safe = compute_jacobian_at_worst_points_simple(
            model=model,
            dynamics_model=dynamics_model,
            worst_points=safe_worst_points,
            translator=translator,
        )
        print(f"    J_safe shape: {J_safe.shape}")

        if unsafe_jac_points.shape[0] > 0:
            J_unsafe = compute_jacobian_at_worst_points_simple(
                model=model,
                dynamics_model=dynamics_model,
                worst_points=unsafe_jac_points,
                translator=translator,
            )
            # unsafe 点约束方向取反：J_unsafe @ d >= 0 等价于 ∂h/∂θ @ d <= 0（h 减小）
            J_unsafe = -J_unsafe
            J_true = torch.cat([J_safe, J_unsafe], dim=0)
            print(f"    J_unsafe shape: {J_unsafe.shape}, 合并后: {J_true.shape}")
        else:
            J_true = J_safe
            print(f"    J_true shape: {J_true.shape}")

        # ---------- 5.3 构造失败最坏点字典 ----------
        failed_worst_points = {
            'unsafe': list(zip(unsafe_worst_points, unsafe_worst_values.tolist())),
            'safe': list(zip(cbf_worst_points, cbf_worst_values.tolist())),
        }

        # ---------- 5.4 内循环修复 ----------
        print(f"\n[迭代 {iteration+1}.3] 内循环修复 ({num_inner_steps} 步)...")

        inner_history = []
        for inner_step in range(num_inner_steps):
            # 计算修复损失
            t0 = time.perf_counter()
            loss_val, g_F = compute_sampled_repair_loss_and_grad(
                model=model,
                dynamics_model=dynamics_model,
                failed_worst_points=failed_worst_points,
                verbose=False,
                translator=translator,
                margin = 0.1,
            )
            t1 = time.perf_counter()

            # QP 投影更新
            t2 = time.perf_counter()
            g_raw_norm, update_norm, active = qp_project_and_update(
                model=model,
                g_raw=g_F,
                J_verified=J_true,
                lr=lr,
                verbose=False,
            )
            t3 = time.perf_counter()

            print(f"    内步 {inner_step+1}/{num_inner_steps}: ", t2-t1, t3-t2)
            inner_history.append({
                'step': inner_step + 1,
                'loss': loss_val,
                'g_raw_norm': g_raw_norm,
                'update_norm': update_norm,
                'active_constraints': active,
                't_loss': t1 - t0,
                't_qp': t3 - t2,
            })

            if inner_step == num_inner_steps - 1 or (inner_step + 1) % 5 == 0:
                print(f"    [内步 {inner_step+1}/{num_inner_steps}] "
                      f"loss={loss_val:.6f}, |g|={g_raw_norm:.4f}, "
                      f"|d|={update_norm:.6f}, active={active}, "
                      f"t_loss={t1-t0:.3f}s, t_qp={t3-t2:.3f}s")

        # ---------- 5.5 保存模型 ----------
        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v2.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        print(f"\n[迭代 {iteration+1}.4] 保存 PyTorch 模型: {pytorch_save_path}")

        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v2.onnx"
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)
        print(f"[迭代 {iteration+1}.5] 转换为 ONNX: {onnx_path}")

        # ---------- 5.6 验证 ----------
        print(f"\n[迭代 {iteration+1}.6] 运行验证...")
        results = verify_model(onnx_path, dynamics_model, max_depth=max_depth)

        certified_percentage, stats = calculate_pass_rate(results)

        print(f"\n[迭代 {iteration+1}.7] 验证结果:")
        print(f"    总样本数: {stats['total']}")
        print(f"    Certified: {stats['certified_pct']:.2f}%")
        print(f"    ★ 通过率: {certified_percentage:.2f}%")

        # ---------- 5.7 保存验证结果 ----------
        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired_v2.pt"

        regions_to_save = {
            'V_safe': results.get('V_safe', V_safe_init),
            'V_unsafe': results.get('V_unsafe', V_unsafe_init),
            'F_h_positive_in_unsafe': results.get('F_h_positive_in_unsafe', F_h_positive_in_unsafe_init),
            'F_safe_cbf_violation': results.get('F_safe_cbf_violation', F_safe_cbf_violation_init),
            'F_depth_limit_reached': results.get('F_depth_limit_reached', F_depth_limit_reached_init),
            'F_unsafe_cannot_split': results.get('F_unsafe_cannot_split', F_unsafe_cannot_split_init),
            'Certified percentage': certified_percentage,
            'stats': stats,
            'iteration': iteration + 1,
        }
        torch.save(regions_to_save, verified_regions_path)
        print(f"[迭代 {iteration+1}.8] 保存验证区域: {verified_regions_path}")

        # ---------- 5.8 更新区域用于下次迭代 ----------
        updated_regions_path = verified_regions_path
        updated_data = torch.load(updated_regions_path, map_location=device, weights_only=False)

        V_safe_init = updated_data['V_safe']
        V_unsafe_init = updated_data['V_unsafe']
        F_h_positive_in_unsafe_init = updated_data['F_h_positive_in_unsafe']
        F_safe_cbf_violation_init = updated_data['F_safe_cbf_violation']
        F_depth_limit_reached_init = updated_data['F_depth_limit_reached']
        F_unsafe_cannot_split_init = updated_data['F_unsafe_cannot_split']

        print(f"    更新后 F_h: {len(F_h_positive_in_unsafe_init)}, "
              f"F_safe: {len(F_safe_cbf_violation_init)}, "
              f"F_depth: {len(F_depth_limit_reached_init)}, "
              f"F_unsafe_split: {len(F_unsafe_cannot_split_init)}")

        # ---------- 记录 ----------
        iteration_results.append({
            'iteration': iteration + 1,
            'loss': inner_history[-1]['loss'] if inner_history else 0.0,
            'update_norm': inner_history[-1]['update_norm'] if inner_history else 0.0,
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

    # ========== 7. 保存运行结果 ==========
    import json
    results_dir = "New_repair/nr_results_v2"
    os.makedirs(results_dir, exist_ok=True)

    run_result = {
        'system': system_name_key,
        'activation': activation,
        'initial_pass_rate': model_initial_rate,
        'final_pass_rate': final_rate,
        'improvement': improvement,
        'num_iterations': num_iterations,
        'num_samples': num_samples,
        'num_inner_steps': num_inner_steps,
        'lr': lr,
        'max_depth': max_depth,
        'iteration_results': iteration_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v2.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    print(f"\n[7] 运行结果已保存: {result_file}")
    print("\n" + "=" * 70)
    print("演示结束")
    print("=" * 70)


def _save_artifacts(model, dynamics_model, iteration, certified_percentage, stats,
                    V_safe, V_unsafe, F_h, F_safe, F_depth, F_unsafe):
    """保存模型和验证结果（提前终止时使用）"""
    pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{dynamics_model.activation_fnc}_cbf_repaired_v2.pth"
    torch.save(model.state_dict(), pytorch_save_path)

    onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{dynamics_model.activation_fnc}_cbf_repaired_v2.onnx"
    pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

    verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{dynamics_model.activation_fnc}_repaired_v2.pt"
    regions_to_save = {
        'V_safe': V_safe,
        'V_unsafe': V_unsafe,
        'F_h_positive_in_unsafe': F_h,
        'F_safe_cbf_violation': F_safe,
        'F_depth_limit_reached': F_depth,
        'F_unsafe_cannot_split': F_unsafe,
        'Certified percentage': certified_percentage,
        'stats': stats,
        'iteration': iteration + 1,
    }
    torch.save(regions_to_save, verified_regions_path)


if __name__ == "__main__":
    main()
