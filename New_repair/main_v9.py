"""
Neural CBF 迭代修复 v9：基于调和平均综合通过率 + Top-N Unsafe 选择

核心改进 (相对于v8):
    - Jacobian 计算时，对 V_unsafe 也使用 Top-N 选择
    - V_safe: 选择 h 下界最小的 N 个（最接近边界）
    - V_unsafe: 选择 h 上界最大的 N 个（最危险/最接近违规）

其他内容与 main_v8.py 完全一致。
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

# v4: RS Jacobian (混合版：V_safe用RS，V_unsafe用精确torch Jacobian)
from New_repair.geometry_module_new_v4 import compute_jacobian_rs, compute_jacobian_rs_new

# v3: 特征点提取和损失计算
from New_repair.geometry_module_new_v3 import extract_all_feature_points
from New_repair.geometry_module_new import compute_simplex_bound_batch
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


def compute_simplex_volume(simplex):
    """
    计算n维单纯形的体积（面积）
    公式: Volume = (1/n!) * |det(v1-v0, v2-v0, ..., vn-v0)|
    """
    num_vertices = simplex.shape[0]
    n = simplex.shape[1]

    if n == 0:
        return 0.0

    if num_vertices != n + 1:
        raise ValueError(f"Invalid simplex shape: expected [n+1, n], got {simplex.shape}")

    origin = simplex[0]
    vectors = simplex[1:] - origin
    det = np.linalg.det(vectors)
    volume = abs(det) / np.math.factorial(n)

    return volume


def compute_total_volume(simplices_list):
    if not simplices_list:
        return 0.0
    return sum(compute_simplex_volume(s) for s in simplices_list)


def compute_safety_metrics_v8(
    V_safe, V_unsafe, F_h_positive_in_unsafe, F_safe_cbf_violation,
    F_depth_limit_reached_unsafe, F_depth_limit_reached_safe, F_unsafe_cannot_split,
):
    """v8版本：计算基于调和平均的综合安全指标"""
    volume_v_safe = compute_total_volume(V_safe)
    volume_v_unsafe = compute_total_volume(V_unsafe)
    volume_f_h = compute_total_volume(F_h_positive_in_unsafe)
    volume_f_safe_violation = compute_total_volume(F_safe_cbf_violation)
    volume_f_depth_unsafe = compute_total_volume(F_depth_limit_reached_unsafe)
    volume_f_depth_safe = compute_total_volume(F_depth_limit_reached_safe)
    volume_f_unsafe_split = compute_total_volume(F_unsafe_cannot_split)

    total_volume = volume_v_safe + volume_v_unsafe + volume_f_h + volume_f_safe_violation + volume_f_depth_unsafe + volume_f_depth_safe + volume_f_unsafe_split
    total_uncertain_volume = volume_f_depth_unsafe + volume_f_depth_safe + volume_f_unsafe_split
    uncertain_half = total_uncertain_volume / 2.0

    true_safe_volume = volume_v_safe + volume_f_safe_violation + uncertain_half
    true_unsafe_volume = volume_v_unsafe + volume_f_h + uncertain_half

    R_safe = volume_v_safe / true_safe_volume if true_safe_volume > 0 else 0.0
    R_unsafe = volume_v_unsafe / true_unsafe_volume if true_unsafe_volume > 0 else 0.0

    HarmonicMeanPassRate = 2.0 * R_safe * R_unsafe / (R_safe + R_unsafe) if (R_safe + R_unsafe) > 0 else 0.0

    standard_pass_rate = ((volume_v_safe + volume_v_unsafe) / total_volume * 100) if total_volume > 0 else 0.0
    unsafe_intersect_volume = volume_v_unsafe + volume_f_h
    usr = (volume_v_unsafe / unsafe_intersect_volume * 100) if unsafe_intersect_volume > 0 else 0.0
    f_h_ratio = (volume_f_h / unsafe_intersect_volume * 100) if unsafe_intersect_volume > 0 else 0.0
    uncertainty_ratio = (total_uncertain_volume / total_volume * 100) if total_volume > 0 else 0.0

    return {
        'R_safe': R_safe, 'R_unsafe': R_unsafe, 'HarmonicMeanPassRate': HarmonicMeanPassRate,
        'true_safe_volume': true_safe_volume, 'true_unsafe_volume': true_unsafe_volume,
        'standard_pass_rate': standard_pass_rate, 'usr': usr, 'f_h_ratio': f_h_ratio,
        'uncertainty_ratio': uncertainty_ratio, 'unsafe_intersect_volume': unsafe_intersect_volume,
        'total_volume': total_volume,
        'volumes': {
            'V_safe': volume_v_safe, 'V_unsafe': volume_v_unsafe, 'F_h': volume_f_h,
            'F_safe_violation': volume_f_safe_violation,
            'F_depth_unsafe': volume_f_depth_unsafe, 'F_depth_safe': volume_f_depth_safe,
            'F_unsafe_split': volume_f_unsafe_split, 'total_uncertain': total_uncertain_volume,
            'uncertain_half': uncertain_half,
        }
    }


def extract_feature_points_from_regions(simplices_list, device, dtype=torch.float32):
    if not simplices_list:
        return torch.empty(0, 2, device=device, dtype=dtype)
    all_feature_points, _ = extract_all_feature_points(simplices_list, device=device, dtype=dtype)
    N, num_fp, D = all_feature_points.shape
    return all_feature_points.view(N * num_fp, D)


def select_top_n_v_safe(model, V_safe, dynamics_model, translator, top_n, cbf_margin=0.0):
    """选择最需要保护（h下界最小）的 top_n 个 V_safe 单纯形"""
    if len(V_safe) == 0:
        return []

    n_available = len(V_safe)
    actual_n = min(top_n, n_available)

    BATCH_SIZE = 1024
    all_margins = []

    for batch_start in range(0, n_available, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_available)
        V_safe_batch = V_safe[batch_start:batch_end]

        min_L_batch = compute_simplex_bound_batch(
            model, V_safe_batch, 'safe',
            dynamics_model=dynamics_model, translator=translator
        )

        margins_batch = min_L_batch.detach().cpu().numpy() if isinstance(min_L_batch, torch.Tensor) else np.array(min_L_batch)
        all_margins.append(margins_batch)

    margins = np.concatenate(all_margins, axis=0)

    if actual_n == n_available:
        selected_indices = list(range(n_available))
    else:
        # 选择 h 下界最小的（最接近边界，最需要保护的）
        selected_indices = np.argsort(margins)[:actual_n].tolist()

    top_n_v_safe = [V_safe[i] for i in selected_indices]

    return top_n_v_safe


def select_top_n_v_unsafe(model, V_unsafe, top_n):
    """选择最危险（h上界最大）的 top_n 个 V_unsafe 单纯形"""
    if len(V_unsafe) == 0:
        return []

    n_available = len(V_unsafe)
    actual_n = min(top_n, n_available)

    BATCH_SIZE = 1024
    all_h_ub = []

    for batch_start in range(0, n_available, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_available)
        V_unsafe_batch = V_unsafe[batch_start:batch_end]

        _, h_ub_batch = compute_simplex_bound_batch(
            model, V_unsafe_batch, 'unsafe',
            dynamics_model=None, translator=None
        )

        h_ub_batch = h_ub_batch.detach().cpu().numpy() if isinstance(h_ub_batch, torch.Tensor) else np.array(h_ub_batch)
        all_h_ub.append(h_ub_batch)

    h_ub_all = np.concatenate(all_h_ub, axis=0)

    if actual_n == n_available:
        selected_indices = list(range(n_available))
    else:
        # 选择 h 上界最大的（最危险，最接近违规的）
        selected_indices = np.argsort(h_ub_all)[-actual_n:].tolist()

    top_n_v_unsafe = [V_unsafe[i] for i in selected_indices]

    return top_n_v_unsafe


def select_repair_targets(
    F_h_positive_in_unsafe, F_safe_cbf_violation, F_depth_limit_reached_unsafe,
    F_depth_limit_reached_safe, F_unsafe_cannot_split, current_phase,
):
    if current_phase == 1:
        return list(F_safe_cbf_violation), list(F_h_positive_in_unsafe), "Phase1_Definitive"
    else:
        return (
            list(F_safe_cbf_violation) + list(F_depth_limit_reached_safe),
            list(F_h_positive_in_unsafe) + list(F_unsafe_cannot_split) + list(F_depth_limit_reached_unsafe),
            "Phase2_All"
        )


def check_stop_criteria(F_h_positive_in_unsafe, F_safe_cbf_violation, F_depth_limit_reached_unsafe, F_depth_limit_reached_safe, F_unsafe_cannot_split,
                        current_max_depth, max_depth_limit, phase2_improvement_history,
                        min_improvement_threshold=0.5, max_stagnant_iterations=3,
                        first_max_depth_pass_rate=None, at_max_depth_consecutive_no_improve=0):
    total_fail = len(F_h_positive_in_unsafe) + len(F_safe_cbf_violation) + len(F_depth_limit_reached_unsafe) + len(F_depth_limit_reached_safe) + len(F_unsafe_cannot_split)
    if total_fail == 0:
        return True, "ALL_CERTIFIED"
    if current_max_depth >= max_depth_limit:
        if first_max_depth_pass_rate is not None and at_max_depth_consecutive_no_improve >= 5:
            return True, f"MAX_DEPTH_PLATEAU"
    if len(phase2_improvement_history) >= max_stagnant_iterations:
        if max(phase2_improvement_history[-max_stagnant_iterations:]) < min_improvement_threshold:
            return True, f"PLATEAU_DETECTED"
    return False, ""


def decide_next_max_depth(current_max_depth, current_phase, definitive_fail_count, uncertain_fail_count, depth_schedule, last_verification_pass_rate):
    if current_phase == 1:
        if definitive_fail_count == 0:
            try:
                current_idx = depth_schedule.index(current_max_depth)
                next_max_depth = depth_schedule[current_idx + 1] if current_idx + 1 < len(depth_schedule) else current_max_depth
            except ValueError:
                next_max_depth = min(current_max_depth + 2, max(depth_schedule))
            return next_max_depth, 2, "DEFINITIVE_CLEARED"
        return current_max_depth, 1, "PHASE1_CONTINUE"
    else:
        try:
            current_idx = depth_schedule.index(current_max_depth)
        except ValueError:
            current_idx = -1
        if definitive_fail_count > 0:
            return current_max_depth, 2, "PHASE2_DEFINITIVE_REMAIN"
        if uncertain_fail_count == 0:
            return current_max_depth, 2, "PHASE2_ALL_CLEARED"
        if current_idx + 1 < len(depth_schedule):
            return depth_schedule[current_idx + 1], 2, f"PHASE2_DEPTH_INCREASE"
        return current_max_depth, 2, "PHASE2_MAX_DEPTH"


def main():
    parser = argparse.ArgumentParser(description='Neural CBF 迭代修复 v9 (Top-N V_unsafe 选择)')
    parser.add_argument('--activation', '-a', type=str, required=True, choices=SUPPORTED_ACTIVATIONS)
    parser.add_argument('--system', '-s', type=str, required=True, choices=list(DYNAMICS_SYSTEMS.keys()))
    parser.add_argument('--rs-n', type=int, default=100, help='随机平滑采样次数 N (default: 100)')
    parser.add_argument('--rs-sigma', type=float, default=0.01, help='随机平滑噪声标准差 sigma (default: 0.01)')
    parser.add_argument('--top-n-protect', type=int, default=500, help='Top-N V_safe/V_unsafe 数量 (default: 500)')
    parser.add_argument('--max-depth-start', type=int, default=10)
    parser.add_argument('--max-depth-limit', type=int, default=20)
    parser.add_argument('--depth-schedule', type=str, default="10,12,15")
    parser.add_argument('--num-inner-steps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--target-pass-rate', type=float, default=100.0)
    parser.add_argument('--plateau-threshold', type=float, default=0.5)
    parser.add_argument('--max-stagnant-iterations', type=int, default=5)
    parser.add_argument('--max-total-iterations', type=int, default=30)

    args = parser.parse_args()
    activation = args.activation
    system_name_key = args.system
    rs_n = args.rs_n
    rs_sigma = args.rs_sigma
    top_n_protect = args.top_n_protect
    max_depth_start = args.max_depth_start
    max_depth_limit = args.max_depth_limit
    depth_schedule = [int(d) for d in args.depth_schedule.split(',')]
    num_inner_steps = args.num_inner_steps
    lr = args.lr
    target_pass_rate = args.target_pass_rate
    plateau_threshold = args.plateau_threshold
    max_stagnant_iterations = args.max_stagnant_iterations
    max_total_iterations = args.max_total_iterations

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print(f"Neural CBF 迭代修复 v9 [Top-N V_unsafe 选择]")
    print(f"  激活={activation}, 系统={system_name_key}")
    print(f"  RS 参数: N={rs_n}, sigma={rs_sigma}")
    print(f"  Top-N V_safe/V_unsafe: {top_n_protect}")
    print(f"  深度调度: {depth_schedule}")
    print("=" * 70)

    dynamics_class = DYNAMICS_SYSTEMS[system_name_key]
    dynamics_model = dynamics_class(alpha=1.0)
    dynamics_model.activation_fnc = activation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = f"data/New_models_Hard_{activation}_v1"
    model_path = f"{model_dir}/{dynamics_model.system_name}_cbf.pth"

    model = BarrierNN(input_size=dynamics_model.input_dim, hidden_sizes=dynamics_model.hidden_sizes, device=device, activation_fnc=activation)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"    参数数量: {num_params}")

    regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_v1.pt"
    regions_data = torch.load(regions_path, map_location=device, weights_only=False)
    V_safe_init = regions_data['V_safe']
    V_unsafe_init = regions_data['V_unsafe']
    F_h_positive_in_unsafe_init = regions_data['F_h_positive_in_unsafe']
    F_safe_cbf_violation_init = regions_data['F_safe_cbf_violation']
    F_depth_limit_reached_unsafe_init = regions_data.get('F_depth_limit_reached_unsafe', regions_data.get('F_depth_limit_reached', []))
    F_depth_limit_reached_safe_init = regions_data.get('F_depth_limit_reached_safe', [])
    F_unsafe_cannot_split_init = regions_data['F_unsafe_cannot_split']

    total_fail = len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init) + len(F_depth_limit_reached_unsafe_init) + len(F_depth_limit_reached_safe_init) + len(F_unsafe_cannot_split_init)

    original_safety_metrics = compute_safety_metrics_v8(
        V_safe_init, V_unsafe_init, F_h_positive_in_unsafe_init,
        F_safe_cbf_violation_init, F_depth_limit_reached_unsafe_init, F_depth_limit_reached_safe_init, F_unsafe_cannot_split_init,
    )
    original_max_depth_harmonic = original_safety_metrics['HarmonicMeanPassRate'] * 100
    original_max_depth_standard = original_safety_metrics['standard_pass_rate']
    original_max_depth_R_safe = original_safety_metrics['R_safe'] * 100
    original_max_depth_R_unsafe = original_safety_metrics['R_unsafe'] * 100

    print(f"\n[3.1] 原始区域 v8 指标: HarmonicMeanPassRate={original_max_depth_harmonic:.2f}%, R_safe={original_max_depth_R_safe:.2f}%, R_unsafe={original_max_depth_R_unsafe:.2f}%")
    print(f"    V_safe={len(V_safe_init)}, V_unsafe={len(V_unsafe_init)}, 总需修复={total_fail}")

    # ========== 3.2 检查是否需要修复 ==========
    if original_max_depth_standard >= 99.9 and original_max_depth_harmonic >= 99.9:
        print(f"\n[3.2] 验证通过率已达 99.9%，无需修复！")
        import json
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_v9")
        os.makedirs(results_dir, exist_ok=True)
        run_result = {
            'system': system_name_key, 'activation': activation,
            'method': 'Harmonic Mean CBF Pass Rate (v9, Top-N V_unsafe)',
            'rs_n': rs_n, 'rs_sigma': rs_sigma,
            'top_n_protect': top_n_protect, 'max_depth_start': max_depth_start, 'max_depth_limit': max_depth_limit,
            'depth_schedule': depth_schedule, 'num_inner_steps': num_inner_steps, 'lr': lr,
            'target_pass_rate': target_pass_rate, 'plateau_threshold': plateau_threshold,
            'max_stagnant_iterations': max_stagnant_iterations, 'max_total_iterations': max_total_iterations,
            'original_max_depth_harmonic': original_max_depth_harmonic,
            'original_max_depth_standard': original_max_depth_standard,
            'original_max_depth_R_safe': original_max_depth_R_safe, 'original_max_depth_R_unsafe': original_max_depth_R_unsafe,
            'final_harmonic_pass_rate': original_max_depth_harmonic, 'final_standard_pass_rate': original_max_depth_standard,
            'final_R_safe': original_max_depth_R_safe, 'final_R_unsafe': original_max_depth_R_unsafe,
            'harmonic_improvement': 0.0, 'standard_improvement': 0.0,
            'num_iterations': 0, 'iteration_results': [],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'skip_reason': 'already_100_percent',
        }
        result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v9.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(run_result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {result_file}")
        print("无需修复，程序结束")
        return

    # ========== 3.5 起始深度验证 ==========
    print(f"\n[3.5] 用起始深度验证初始模型 (max_depth={max_depth_start})...")
    pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v9.pth"
    torch.save(model.state_dict(), pytorch_save_path)
    onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v9.onnx"
    pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)
    start_depth_results = verify_model(onnx_path, dynamics_model, 
                                       max_depth=max_depth_start)

    start_depth_safety_metrics = compute_safety_metrics_v8(
        start_depth_results.get('V_safe', []), start_depth_results.get('V_unsafe', []),
        start_depth_results.get('F_h_positive_in_unsafe', []), start_depth_results.get('F_safe_cbf_violation', []),
        start_depth_results.get('F_depth_limit_reached_unsafe', []), start_depth_results.get('F_depth_limit_reached_safe', []),
        start_depth_results.get('F_unsafe_cannot_split', []),
    )

    initial_harmonic_pass_rate = start_depth_safety_metrics['HarmonicMeanPassRate'] * 100
    initial_standard_pass_rate = start_depth_safety_metrics['standard_pass_rate']
    initial_R_safe = start_depth_safety_metrics['R_safe'] * 100
    initial_R_unsafe = start_depth_safety_metrics['R_unsafe'] * 100

    translator = TorchTranslator(device=device)
    iteration_results = []
    phase2_improvement_history = []
    current_max_depth = max_depth_start
    current_phase = 1
    definitive_fail_prev = total_fail
    first_max_depth_pass_rate = None
    at_max_depth_consecutive_no_improve = 0

    print(f"\n开始渐进式深度分层修复 (v9), Phase {current_phase}, max_depth={current_max_depth}")

    for iteration in range(max_total_iterations):
        definitive_fail = len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init)
        uncertain_fail = len(F_depth_limit_reached_unsafe_init) + len(F_depth_limit_reached_safe_init) + len(F_unsafe_cannot_split_init)

        stop, stop_reason = check_stop_criteria(
            F_h_positive_in_unsafe_init, F_safe_cbf_violation_init, F_depth_limit_reached_unsafe_init, F_depth_limit_reached_safe_init, F_unsafe_cannot_split_init,
            current_max_depth, max_depth_limit, phase2_improvement_history,
            plateau_threshold, max_stagnant_iterations, first_max_depth_pass_rate, at_max_depth_consecutive_no_improve,
        )
        if stop:
            print(f"\n  === 停止: {stop_reason} ===")
            break

        next_max_depth, next_phase, depth_reason = decide_next_max_depth(
            current_max_depth, current_phase, definitive_fail, uncertain_fail, depth_schedule, initial_harmonic_pass_rate,
        )
        print(f"\n[迭代 {iteration+1}] {depth_reason}, max_depth {current_max_depth}->{next_max_depth}, phase {current_phase}->{next_phase}")

        failed_safe_simplices, failed_unsafe_simplices, repair_type = select_repair_targets(
            F_h_positive_in_unsafe_init, F_safe_cbf_violation_init, F_depth_limit_reached_unsafe_init, F_depth_limit_reached_safe_init, F_unsafe_cannot_split_init, current_phase,
        )

        if len(failed_safe_simplices) == 0 and len(failed_unsafe_simplices) == 0:
            certified_percentage = initial_harmonic_pass_rate
        else:
            # ========== 5.3 选择 Top-N V_safe 和 Top-N V_unsafe ==========
            print(f"\n[迭代 {iteration+1}.2] 选择 Top-N V_safe/V_unsafe (N={top_n_protect})...")

            top_n_v_safe = select_top_n_v_safe(
                model, list(V_safe_init), dynamics_model, translator, top_n_protect,
            )
            top_n_v_unsafe = select_top_n_v_unsafe(
                model, list(V_unsafe_init), top_n_protect,
            )
            top_n_used = len(top_n_v_safe)
            print(f"    Top-N V_safe: {len(top_n_v_safe)} 个, Top-N V_unsafe: {len(top_n_v_unsafe)} 个")

            # 提取失败区域特征点
            failed_safe_feature_points = extract_feature_points_from_regions(failed_safe_simplices, device)
            failed_unsafe_feature_points = extract_feature_points_from_regions(failed_unsafe_simplices, device)

            # ========== 5.4 计算 RS Jacobian（使用 Top-N） ==========
            print(f"\n[迭代 {iteration+1}.3] 计算混合 Jacobian (V_safe用RS, V_unsafe用精确torch, N={rs_n}, sigma={rs_sigma})...")

            J = compute_jacobian_rs_new(
                model,
                top_n_v_safe,
                top_n_v_unsafe,  # v9: 使用 top_n_v_unsafe 而不是 list(V_unsafe_init)
                dynamics_model=dynamics_model,
                translator=translator,
                N=rs_n,
                sigma=rs_sigma,
            )
            print(f"    J 形状: {J.shape}")

            # ========== 5.5 内循环修复 ==========
            print(f"\n[迭代 {iteration+1}.4] 内循环修复 ({num_inner_steps} 步)...")

            depth_lr_map = {10: 1e-3, 12: 1e-3, 15: 5e-3}
            current_lr = depth_lr_map.get(current_max_depth, lr)

            inner_history = []
            n_safe_fp = failed_safe_feature_points.shape[0]
            n_unsafe_fp = failed_unsafe_feature_points.shape[0]

            for inner_step in range(num_inner_steps):
                t0 = time.perf_counter()
                dtype = next(model.parameters()).dtype
                g_F_total = torch.zeros(num_params, dtype=dtype, device=device)
                total_loss_sum = 0.0
                total_n = 0

                for unsafe_start in range(0, n_unsafe_fp, 1024):
                    unsafe_end = min(unsafe_start + 1024, n_unsafe_fp)
                    unsafe_chunk = failed_unsafe_feature_points[unsafe_start:unsafe_end]
                    chunk_loss, chunk_g = compute_repair_loss_and_grad(
                        model, dynamics_model, torch.empty(0, 2, device=device, dtype=dtype), unsafe_chunk,
                        margin=0.1, cbf_margin=0.0, beta=5.0, grad_clip_norm=10.0, verbose=False, translator=translator,
                    )
                    total_loss_sum += chunk_loss * unsafe_chunk.shape[0]
                    total_n += unsafe_chunk.shape[0]
                    g_F_total.add_(chunk_g)

                for safe_start in range(0, n_safe_fp, 1024):
                    safe_end = min(safe_start + 1024, n_safe_fp)
                    safe_chunk = failed_safe_feature_points[safe_start:safe_end]
                    chunk_loss, chunk_g = compute_repair_loss_and_grad(
                        model, dynamics_model, safe_chunk, torch.empty(0, 2, device=device, dtype=dtype),
                        margin=0.1, cbf_margin=0.0, beta=5.0, grad_clip_norm=10.0, verbose=False, translator=translator,
                    )
                    total_loss_sum += chunk_loss * safe_chunk.shape[0]
                    total_n += safe_chunk.shape[0]
                    g_F_total.add_(chunk_g)

                loss_val = total_loss_sum / total_n if total_n > 0 else 0.0
                grad_norm = g_F_total.norm().item()
                g_F = g_F_total * (10.0 / grad_norm) if grad_norm > 10.0 else g_F_total
                t1 = time.perf_counter()

                g_raw_norm, update_norm, active = qp_project_and_update_gd(
                    model=model, g_raw=g_F, J_verified=J, lr=current_lr, verbose=False,
                )
                t2 = time.perf_counter()

                inner_history.append({
                    'step': inner_step + 1, 'loss': loss_val, 'g_raw_norm': g_raw_norm,
                    'update_norm': update_norm, 'active_constraints': active,
                    't_loss': t1 - t0, 't_qp': t2 - t1,
                })

                if inner_step == num_inner_steps - 1 or (inner_step + 1) % 5 == 0:
                    print(f"    [内步 {inner_step+1}] loss={loss_val:.6f}, |g|={g_raw_norm:.4f}, |d|={update_norm:.6f}, active={active}")

        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v9.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v9.onnx"
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

        results = verify_model(onnx_path, dynamics_model, max_depth=current_max_depth)
        safety_metrics = compute_safety_metrics_v8(
            results.get('V_safe', []), results.get('V_unsafe', []),
            results.get('F_h_positive_in_unsafe', []), results.get('F_safe_cbf_violation', []),
            results.get('F_depth_limit_reached_unsafe', []), results.get('F_depth_limit_reached_safe', []),
            results.get('F_unsafe_cannot_split', []),
        )

        certified_percentage = safety_metrics['HarmonicMeanPassRate'] * 100
        R_safe_pct = safety_metrics['R_safe'] * 100
        R_unsafe_pct = safety_metrics['R_unsafe'] * 100

        print(f"\n[迭代 {iteration+1}.7] 验证结果: HarmonicMeanPassRate={certified_percentage:.2f}%, R_safe={R_safe_pct:.2f}%, R_unsafe={R_unsafe_pct:.2f}%")

        if current_max_depth >= max_depth_limit:
            if first_max_depth_pass_rate is None:
                first_max_depth_pass_rate = certified_percentage
                at_max_depth_consecutive_no_improve = 0
            else:
                improvement = certified_percentage - first_max_depth_pass_rate
                at_max_depth_consecutive_no_improve = at_max_depth_consecutive_no_improve + 1 if improvement < plateau_threshold else 0
        else:
            at_max_depth_consecutive_no_improve = 0

        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired_v9.pt"
        torch.save({
            'V_safe': results.get('V_safe', V_safe_init), 'V_unsafe': results.get('V_unsafe', V_unsafe_init),
            'F_h_positive_in_unsafe': results.get('F_h_positive_in_unsafe', F_h_positive_in_unsafe_init),
            'F_safe_cbf_violation': results.get('F_safe_cbf_violation', F_safe_cbf_violation_init),
            'F_depth_limit_reached_unsafe': results.get('F_depth_limit_reached_unsafe', F_depth_limit_reached_unsafe_init),
            'F_depth_limit_reached_safe': results.get('F_depth_limit_reached_safe', F_depth_limit_reached_safe_init),
            'F_unsafe_cannot_split': results.get('F_unsafe_cannot_split', F_unsafe_cannot_split_init),
            'Certified percentage': certified_percentage,
        }, verified_regions_path)

        updated_data = torch.load(verified_regions_path, map_location=device, weights_only=False)
        V_safe_init = updated_data['V_safe']
        V_unsafe_init = updated_data['V_unsafe']
        F_h_positive_in_unsafe_init = updated_data['F_h_positive_in_unsafe']
        F_safe_cbf_violation_init = updated_data['F_safe_cbf_violation']
        F_depth_limit_reached_unsafe_init = updated_data['F_depth_limit_reached_unsafe']
        F_depth_limit_reached_safe_init = updated_data['F_depth_limit_reached_safe']
        F_unsafe_cannot_split_init = updated_data['F_unsafe_cannot_split']

        definitive_fail_new = len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init)
        improvement = definitive_fail_prev - definitive_fail_new
        if current_phase == 2:
            phase2_improvement_history.append(improvement if improvement > 0 else 0.0)
        definitive_fail_prev = definitive_fail_new

        iteration_results.append({
            'iteration': iteration + 1, 'phase': current_phase, 'max_depth': current_max_depth,
            'loss': inner_history[-1]['loss'] if 'inner_history' in dir() and inner_history else 0.0,
            'HarmonicMeanPassRate': certified_percentage, 'R_safe': R_safe_pct, 'R_unsafe': R_unsafe_pct,
            'standard_pass_rate': safety_metrics['standard_pass_rate'],
            'f_h_positive': len(F_h_positive_in_unsafe_init), 'f_safe_violation': len(F_safe_cbf_violation_init),
            'f_depth_unsafe': len(F_depth_limit_reached_unsafe_init), 'f_depth_safe': len(F_depth_limit_reached_safe_init),
            'f_unsafe_split': len(F_unsafe_cannot_split_init),
            'definitive_fail': definitive_fail_new,
            'top_n_used': top_n_used if 'top_n_used' in dir() else 0,
            'repair_type': repair_type if 'repair_type' in dir() else 'N/A',
        })

        current_max_depth = next_max_depth
        current_phase = next_phase

        if certified_percentage >= target_pass_rate:
            print(f"\n  === 达到目标通过率 {target_pass_rate}%！提前终止 ===")
            break

    final_harmonic = iteration_results[-1]['HarmonicMeanPassRate'] if iteration_results else initial_harmonic_pass_rate
    final_standard = iteration_results[-1]['standard_pass_rate'] if iteration_results else initial_standard_pass_rate
    final_R_safe = iteration_results[-1]['R_safe'] if iteration_results else initial_R_safe
    final_R_unsafe = iteration_results[-1]['R_unsafe'] if iteration_results else initial_R_unsafe

    harmonic_improvement = final_harmonic - original_max_depth_harmonic
    standard_improvement = final_standard - original_max_depth_standard

    print(f"\n{'='*70}")
    print("修复前后对比 (v9: Top-N V_unsafe)")
    print(f"{'='*70}")
    print(f"指标                     原始          最终          变化")
    print(f"───────────────────────────────────────────────────────")
    print(f"HarmonicMeanPassRate:    {original_max_depth_harmonic:>8.2f}%   {final_harmonic:>8.2f}%   ({harmonic_improvement:+.2f}%)")
    print(f"standard_pass_rate:      {original_max_depth_standard:>8.2f}%   {final_standard:>8.2f}%   ({standard_improvement:+.2f}%)")
    print(f"R_safe:                 {original_max_depth_R_safe:>8.2f}%   {final_R_safe:>8.2f}%")
    print(f"R_unsafe:               {original_max_depth_R_unsafe:>8.2f}%   {final_R_unsafe:>8.2f}")

    import json
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_v9")
    os.makedirs(results_dir, exist_ok=True)

    run_result = {
        'system': system_name_key, 'activation': activation,
        'method': 'Harmonic Mean CBF Pass Rate (v9, Top-N V_unsafe)',
        'rs_n': rs_n, 'rs_sigma': rs_sigma,
        'top_n_protect': top_n_protect, 'max_depth_start': max_depth_start, 'max_depth_limit': max_depth_limit,
        'depth_schedule': depth_schedule, 'num_inner_steps': num_inner_steps, 'lr': lr,
        'target_pass_rate': target_pass_rate, 'plateau_threshold': plateau_threshold,
        'max_stagnant_iterations': max_stagnant_iterations, 'max_total_iterations': max_total_iterations,
        'original_max_depth_harmonic': original_max_depth_harmonic,
        'original_max_depth_standard': original_max_depth_standard,
        'original_max_depth_R_safe': original_max_depth_R_safe, 'original_max_depth_R_unsafe': original_max_depth_R_unsafe,
        'final_harmonic_pass_rate': final_harmonic, 'final_standard_pass_rate': final_standard,
        'final_R_safe': final_R_safe, 'final_R_unsafe': final_R_unsafe,
        'harmonic_improvement': harmonic_improvement, 'standard_improvement': standard_improvement,
        'num_iterations': len(iteration_results), 'iteration_results': iteration_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v9.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    main()
