"""
Neural CBF 迭代修复 v7：渐进式深度分层修复算法

核心思想：
    阶段1（低深度）：只修复确定性违规（F_h_positive_in_unsafe, F_safe_cbf_violation）
    阶段2（深度递增）：逐渐修复不确定区域（F_depth_limit_reached, F_unsafe_cannot_split）

修复顺序策略：
    max_depth从低到高，每次只关注当前深度能"看清"的区域
    确定性违规在浅深度就能修复，不确定区域需要更深才能确认
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

# v4: RS Jacobian
from New_repair.geometry_module_new_v4 import compute_jacobian_rs

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


def compute_simplex_area(simplex):
    """
    计算2D单纯形（三边形/三角形）的面积

    Args:
        simplex: 形状为 [3, 2] 的 numpy array，每行是一个顶点坐标

    Returns:
        float: 单纯形面积
    """
    # 2D三角形面积公式: |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| / 2
    x1, y1 = simplex[0]
    x2, y2 = simplex[1]
    x3, y3 = simplex[2]
    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return area


def compute_total_area(simplices_list):
    """
    计算多个单纯形的总面积

    Args:
        simplices_list: 单纯形列表，每个元素是 [3, 2] 的 numpy array

    Returns:
        float: 总面积
    """
    if not simplices_list:
        return 0.0
    return sum(compute_simplex_area(s) for s in simplices_list)


def compute_safety_metrics(
    V_safe,
    V_unsafe,
    F_h_positive_in_unsafe,
    F_safe_cbf_violation,
    F_depth_limit_reached,
    F_unsafe_cannot_split,
):
    """
    计算加权安全率指标（按面积计算）

    核心指标:
    - USR (Unsafe-Set Pass Rate): 所有与"真正不安全集合"相交的面积中，CBF正确识别的比例
    - F_h_ratio: F_h_positive_in_unsafe 占真正unsafe相交面积的比例（越低越好）

    Args:
        V_safe: 验证通过的安全区域列表
        V_unsafe: 验证通过的不安全区域列表
        F_h_positive_in_unsafe: h(x)>0 in unsafe（最危险）
        F_safe_cbf_violation: CBF在安全区违规
        F_depth_limit_reached: 深度达到上限
        F_unsafe_cannot_split: 无法分割unsafe

    Returns:
        dict: 包含各项安全指标的字典
    """
    # 按面积计算
    area_v_safe = compute_total_area(V_safe)
    area_v_unsafe = compute_total_area(V_unsafe)
    area_f_h = compute_total_area(F_h_positive_in_unsafe)
    area_f_safe_violation = compute_total_area(F_safe_cbf_violation)
    area_f_depth = compute_total_area(F_depth_limit_reached)
    area_f_unsafe_split = compute_total_area(F_unsafe_cannot_split)

    total_area = area_v_safe + area_v_unsafe + area_f_h + area_f_safe_violation + area_f_depth + area_f_unsafe_split

    # 标准通过率（按面积）
    standard_pass_rate = ((area_v_safe + area_v_unsafe) / total_area * 100) if total_area > 0 else 0.0

    # 与"真正不安全集合"相交的面积 = area_v_unsafe + area_f_h
    unsafe_intersect_area = area_v_unsafe + area_f_h

    # USR: Unsafe-Set Pass Rate（按面积）
    # 所有与unsafe相交的面积中，CBF正确识别的比例
    usr = (area_v_unsafe / unsafe_intersect_area * 100) if unsafe_intersect_area > 0 else 0.0

    # F_h危险占比: 漏检面积比例
    f_h_ratio = (area_f_h / unsafe_intersect_area * 100) if unsafe_intersect_area > 0 else 0.0

    # 加权安全分数（按面积，权重: V_safe=1, V_unsafe=1, F_safe_violation=5, F_h=10, F_depth=1, F_unsafe_split=1）
    # 理想状态: F_safe_violation=0, F_h=0, 其他正常
    # weighted_safe: 非违规区域（V_safe + V_unsafe + F_depth + F_unsafe_split）的加权面积
    # weighted_total: 理想状态的总加权面积 = weighted_safe + 违规惩罚(F_safe_violation*5 + F_h*10)
    weighted_safe = area_v_safe + area_v_unsafe + area_f_depth + area_f_unsafe_split
    violation_penalty = area_f_safe_violation * 5 + area_f_h * 10
    weighted_total = weighted_safe + violation_penalty
    weighted_safety_score = (weighted_safe / weighted_total * 100) if weighted_total > 0 else 0.0

    metrics = {
        'standard_pass_rate': standard_pass_rate,  # 标准通过率
        'usr': usr,                                  # Unsafe-Set Pass Rate
        'f_h_ratio': f_h_ratio,                      # F_h漏检比例
        'weighted_safety_score': weighted_safety_score,  # 加权安全分数
        'unsafe_intersect_area': unsafe_intersect_area,  # 与unsafe相交的总面积
        'total_area': total_area,                    # 总面积
        'areas': {
            'V_safe': area_v_safe,
            'V_unsafe': area_v_unsafe,
            'F_h': area_f_h,
            'F_safe_violation': area_f_safe_violation,
            'F_depth': area_f_depth,
            'F_unsafe_split': area_f_unsafe_split,
        }
    }

    return metrics


def extract_feature_points_from_regions(
    simplices_list: list,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not simplices_list:
        return torch.empty(0, 2, device=device, dtype=dtype)

    all_feature_points, _ = extract_all_feature_points(
        simplices_list, device=device, dtype=dtype
    )
    N, num_fp, D = all_feature_points.shape
    return all_feature_points.view(N * num_fp, D)


def select_top_n_v_safe(model, V_safe, dynamics_model, translator, top_n, cbf_margin=0.0):
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
        selected_indices = np.argsort(margins)[:actual_n].tolist()

    top_n_v_safe = [V_safe[i] for i in selected_indices]

    return top_n_v_safe


def select_repair_targets(
    F_h_positive_in_unsafe,
    F_safe_cbf_violation,
    F_depth_limit_reached,
    F_unsafe_cannot_split,
    current_phase,
):
    """
    根据当前阶段选择修复目标

    Phase 1: 只修复确定性违规
    Phase 2: 修复确定性违规 + 深度相关违规

    Args:
        F_h_positive_in_unsafe: h(x) > 0 in unsafe
        F_safe_cbf_violation: CBF violation in safe
        F_depth_limit_reached: depth limit reached (uncertain)
        F_unsafe_cannot_split: cannot split unsafe (uncertain)
        current_phase: 1 or 2

    Returns:
        failed_safe_simplices: 要修复的safe区域列表
        failed_unsafe_simplices: 要修复的unsafe区域列表
    """
    if current_phase == 1:
        # 阶段1：只修复确定性违规
        failed_safe_simplices = list(F_safe_cbf_violation)
        failed_unsafe_simplices = list(F_h_positive_in_unsafe)
        repair_type = "Phase1_Definitive"
    else:
        # 阶段2：修复所有类型
        failed_safe_simplices = (
            list(F_safe_cbf_violation) +
            list(F_depth_limit_reached)
        )
        failed_unsafe_simplices = (
            list(F_h_positive_in_unsafe) +
            list(F_unsafe_cannot_split)
        )
        repair_type = "Phase2_All"

    return failed_safe_simplices, failed_unsafe_simplices, repair_type


def check_stop_criteria(
    F_h_positive_in_unsafe,
    F_safe_cbf_violation,
    F_depth_limit_reached,
    F_unsafe_cannot_split,
    current_max_depth,
    max_depth_limit,
    phase2_improvement_history,
    min_improvement_threshold=0.5,
    max_stagnant_iterations=3,
    first_max_depth_pass_rate=None,
    at_max_depth_consecutive_no_improve=0,
):
    """
    检查是否满足停止修复的条件

    停止条件：
    1. 所有区域都已验证通过（无任何失败区域）
    2. 达到最大深度后连续5次改进<0.5%（与首次达到最大深度的基准比）
    3. 连续多次迭代改进小于阈值（ plateau检测，Phase2专用）

    Args:
        F_h_positive_in_unsafe: h(x) > 0 in unsafe
        F_safe_cbf_violation: CBF violation in safe
        F_depth_limit_reached: depth limit reached
        F_unsafe_cannot_split: cannot split unsafe
        current_max_depth: 当前使用的max_depth
        max_depth_limit: 最大允许的max_depth
        phase2_improvement_history: 阶段2的改进历史
        min_improvement_threshold: 最小改进阈值（百分比）
        max_stagnant_iterations: 最大停滞迭代次数
        first_max_depth_pass_rate: 首次达到最大深度时的验证通过率
        at_max_depth_consecutive_no_improve: 在最大深度时连续改进<阈值次数

    Returns:
        stop: bool, 是否停止
        reason: str, 停止原因
    """
    # 条件1：所有区域都验证通过
    total_fail = (
        len(F_h_positive_in_unsafe) +
        len(F_safe_cbf_violation) +
        len(F_depth_limit_reached) +
        len(F_unsafe_cannot_split)
    )

    if total_fail == 0:
        return True, "ALL_CERTIFIED - 所有区域已验证通过"

    # 条件2：达到最大深度后连续5次改进<0.5%（与首次达到最大深度的基准比）
    if current_max_depth >= max_depth_limit:
        if first_max_depth_pass_rate is not None and at_max_depth_consecutive_no_improve >= 5:
            return True, f"MAX_DEPTH_PLATEAU - 在最大深度连续{at_max_depth_consecutive_no_improve}次改进<{min_improvement_threshold}%"

    # 条件3：plateau检测（阶段2专用）
    if len(phase2_improvement_history) >= max_stagnant_iterations:
        recent_improvements = phase2_improvement_history[-max_stagnant_iterations:]
        max_improvement = max(recent_improvements)

        if max_improvement < min_improvement_threshold:
            return True, f"PLATEAU_DETECTED - 连续{max_stagnant_iterations}次改进<{min_improvement_threshold}%"

    return False, ""


def decide_next_max_depth(
    current_max_depth,
    current_phase,
    definitive_fail_count,
    uncertain_fail_count,
    depth_schedule,
    last_verification_pass_rate,
):
    """
    决定下一个max_depth值

    策略：
    - 阶段1：使用低深度，等确定性违规清零后进入阶段2
    - 阶段2：深度逐步增加，直到达到上限或通过率达到目标

    Args:
        current_max_depth: 当前深度
        current_phase: 当前阶段(1或2)
        definitive_fail_count: 确定性违规数量
        uncertain_fail_count: 不确定区域数量
        depth_schedule: 深度调度列表 [10, 12, 15, 18, 20]
        last_verification_pass率: 上次验证通过率

    Returns:
        next_max_depth: 下一个深度
        next_phase: 下一个阶段
        reason: 决定原因
    """
    if current_phase == 1:
        # 阶段1：低深度修复确定性违规
        if definitive_fail_count == 0:
            # 确定性违规已清零，进入阶段2
            # 找到当前深度在schedule中的下一个
            try:
                current_idx = depth_schedule.index(current_max_depth)
                if current_idx + 1 < len(depth_schedule):
                    next_max_depth = depth_schedule[current_idx + 1]
                else:
                    next_max_depth = current_max_depth
            except ValueError:
                # current_max_depth不在schedule中，取下一个
                next_max_depth = min(current_max_depth + 2, max(depth_schedule))

            return next_max_depth, 2, "DEFINITIVE_CLEARED - 确定性违规已清零，进入阶段2"

        else:
            # 继续保持当前深度
            return current_max_depth, 1, "PHASE1_CONTINUE - 确定性违规仍存在"

    else:
        # 阶段2：深度递增策略
        try:
            current_idx = depth_schedule.index(current_max_depth)
        except ValueError:
            current_idx = -1

        # 检查是否还有明显改进
        if definitive_fail_count > 0:
            # 确定性违规还在，继续当前深度
            return current_max_depth, 2, "PHASE2_DEFINITIVE_REMAIN - 确定性违规仍存在"

        # 不确定区域处理
        if uncertain_fail_count == 0:
            # 所有区域已清零
            return current_max_depth, 2, "PHASE2_ALL_CLEARED - 所有违规已清零"

        # 逐步增加深度
        if current_idx + 1 < len(depth_schedule):
            next_max_depth = depth_schedule[current_idx + 1]
            return next_max_depth, 2, f"PHASE2_DEPTH_INCREASE - 深度从{current_max_depth}增至{next_max_depth}"
        else:
            # 已达最深
            return current_max_depth, 2, "PHASE2_MAX_DEPTH - 已达最大深度"


def main():
    # ========== 0. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(
        description='Neural CBF 迭代修复 v7 (渐进式深度分层修复)'
    )
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
    # v7 新增参数
    parser.add_argument('--top-n-protect', type=int, default=500,
                        help='Top-N V_safe 保护数量 (default: 500)')
    parser.add_argument('--max-depth-start', type=int, default=10,
                        help='起始max_depth (default: 10)')
    parser.add_argument('--max-depth-limit', type=int, default=20,
                        help='最大max_depth (default: 20)')
    parser.add_argument('--depth-schedule', type=str, default="10,12,15",
                        help='深度调度列表，逗号分隔 (default: 10,12,15)')
    # 内循环参数
    parser.add_argument('--num-inner-steps', type=int, default=5,
                        help='内循环步数 (default: 5)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='学习率 (default: 5e-3)')
    parser.add_argument('--target-pass-rate', type=float, default=100.0,
                        help='目标通过率 (default: 100.0)')
    parser.add_argument('--plateau-threshold', type=float, default=0.5,
                        help=' plateau检测阈值百分比 (default: 0.5)')
    parser.add_argument('--max-stagnant-iterations', type=int, default=5,
                        help='最大停滞迭代次数 (default: 3)')
    parser.add_argument('--max-total-iterations', type=int, default=30,
                        help='最大总迭代次数 (default: 50)')

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
    print(f"Neural CBF 迭代修复 v7 [渐进式深度分层修复算法]")
    print(f"  激活={activation}, 系统={system_name_key}")
    print(f"  RS 参数: N={rs_n}, sigma={rs_sigma}")
    print(f"  Top-N V_safe 保护: {top_n_protect}")
    print(f"  深度调度: {depth_schedule}")
    print(f"  起始深度: {max_depth_start}, 最大深度: {max_depth_limit}")
    print(f"  目标通过率: {target_pass_rate}%")
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
    print(f"    隐藏层大小: {dynamics_model.hidden_sizes}")

    # ========== 2. 加载初始神经网络 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dir = f"data/New_models_Hard_{activation}"
    # model_path = f"{model_dir}/{dynamics_model.system_name}_cbf.pth"

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
    _initial_pass_rate = initial_pass_rate

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

    # ========== 3.5 用起始深度验证初始模型（获取真正的基准） ==========
    print(f"\n[3.5] 用起始深度验证初始模型 (max_depth={max_depth_start})...")

    pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v7.pth"
    torch.save(model.state_dict(), pytorch_save_path)

    onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v7.onnx"
    pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)
    initial_results = verify_model(onnx_path, dynamics_model, max_depth=max_depth_start)
    baseline_pass_rate, baseline_stats = calculate_pass_rate(initial_results)

    print(f"    基准验证通过率 (深度{max_depth_start}): {baseline_pass_rate:.2f}%")
    print(f"    Certified: {baseline_stats['certified_pct']:.2f}%")
    print(f"    Uncertified: {baseline_stats['uncertified_pct']:.2f}%")

    # 用这个作为基准通过率（修复后比较用），而不是regions里存的（那是深度15的结果）
    initial_pass_rate = baseline_pass_rate

    # ========== 4. 设置 translator ==========
    translator = TorchTranslator(device=device)

    # ========== 5. 渐进式分层修复主循环 ==========
    iteration_results = []
    phase2_improvement_history = []

    # 初始化状态
    current_max_depth = max_depth_start
    current_phase = 1
    definitive_fail_prev = total_fail
    # 最大深度 plateau 检测状态
    first_max_depth_pass_rate = None  # 首次达到最大深度时的基准通过率
    at_max_depth_consecutive_no_improve = 0  # 在最大深度时连续改进<阈值的次数

    print(f"\n{'='*70}")
    print(f"开始渐进式深度分层修复")
    print(f"  初始阶段: Phase {current_phase}, max_depth={current_max_depth}")
    print(f"  确定性违规: {len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init)}")
    print(f"  不确定区域: {len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init)}")
    print(f"{'='*70}")

    for iteration in range(max_total_iterations):
        print(f"\n{'='*70}")
        print(f"迭代 {iteration + 1}/{max_total_iterations}")
        print(f"  阶段: Phase {current_phase}, max_depth={current_max_depth}")
        print(f"{'='*70}")

        # 统计当前失败区域
        definitive_fail = len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init)
        uncertain_fail = len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init)
        total_fail = definitive_fail + uncertain_fail

        print(f"\n  当前状态:")
        print(f"    确定性违规: {definitive_fail} (F_h={len(F_h_positive_in_unsafe_init)}, F_safe_violation={len(F_safe_cbf_violation_init)})")
        print(f"    不确定区域: {uncertain_fail} (F_depth={len(F_depth_limit_reached_init)}, F_unsafe_split={len(F_unsafe_cannot_split_init)})")
        print(f"    总失败区域: {total_fail}")

        # ========== 5.0 停止条件检查 ==========
        stop, stop_reason = check_stop_criteria(
            F_h_positive_in_unsafe_init,
            F_safe_cbf_violation_init,
            F_depth_limit_reached_init,
            F_unsafe_cannot_split_init,
            current_max_depth,
            max_depth_limit,
            phase2_improvement_history,
            min_improvement_threshold=plateau_threshold,
            max_stagnant_iterations=max_stagnant_iterations,
            first_max_depth_pass_rate=first_max_depth_pass_rate,
            at_max_depth_consecutive_no_improve=at_max_depth_consecutive_no_improve,
        )

        if stop:
            print(f"\n  === 满足停止条件: {stop_reason} ===")
            break

        # ========== 5.1 决定下一轮深度和阶段 ==========
        next_max_depth, next_phase, depth_reason = decide_next_max_depth(
            current_max_depth=current_max_depth,
            current_phase=current_phase,
            definitive_fail_count=definitive_fail,
            uncertain_fail_count=uncertain_fail,
            depth_schedule=depth_schedule,
            last_verification_pass_rate=initial_pass_rate,
        )

        print(f"\n[迭代 {iteration+1}.0] 深度/阶段决策: {depth_reason}")
        print(f"    决策: max_depth={current_max_depth} -> {next_max_depth}, phase={current_phase} -> {next_phase}")

        # ========== 5.2 选择修复目标（根据阶段） ==========
        failed_safe_simplices, failed_unsafe_simplices, repair_type = select_repair_targets(
            F_h_positive_in_unsafe_init,
            F_safe_cbf_violation_init,
            F_depth_limit_reached_init,
            F_unsafe_cannot_split_init,
            current_phase,
        )

        print(f"\n[迭代 {iteration+1}.1] 修复目标 (type={repair_type}):")
        print(f"    Safe区域修复: {len(failed_safe_simplices)} 个")
        print(f"    Unsafe区域修复: {len(failed_unsafe_simplices)} 个")

        # 如果没有需要修复的区域，跳过
        if len(failed_safe_simplices) == 0 and len(failed_unsafe_simplices) == 0:
            print(f"\n  [警告] 当前阶段无可修复区域，跳过修复步骤")

            # 保存当前状态
            certified_percentage = initial_pass_rate
            stats = {
                'total': 0,
                'certified_pct': certified_percentage,
                'uncertified_pct': 0.0,
                'certified_percentage': certified_percentage
            }
        else:
            # ========== 5.3 提取特征点 + Top-N V_safe 选择 ==========
            print(f"\n[迭代 {iteration+1}.2] 提取特征点 + 选择 Top-N V_safe (N={top_n_protect})...")

            top_n_v_safe = select_top_n_v_safe(
                model=model,
                V_safe=list(V_safe_init),
                dynamics_model=dynamics_model,
                translator=translator,
                top_n=top_n_protect,
                cbf_margin=0.0,
            )
            top_n_used = len(top_n_v_safe)
            print(f"    Top-N V_safe: {top_n_used} 个")

            # 提取失败区域特征点
            failed_safe_feature_points = extract_feature_points_from_regions(
                failed_safe_simplices, device=device
            )
            failed_unsafe_feature_points = extract_feature_points_from_regions(
                failed_unsafe_simplices, device=device
            )

            print(f"    失败Safe特征点: {failed_safe_feature_points.shape[0]} 个")
            print(f"    失败Unsafe特征点: {failed_unsafe_feature_points.shape[0]} 个")

            # V_safe 特征点（用于J_RS）
            verified_safe_feature_points = extract_feature_points_from_regions(
                V_safe_init, device=device
            )
            print(f"    V_safe 特征点: {verified_safe_feature_points.shape[0]} 个")

            # ========== 5.4 计算 RS Jacobian ==========
            print(f"\n[迭代 {iteration+1}.3] 计算随机平滑 Jacobian (N={rs_n}, sigma={rs_sigma})...")

            J = compute_jacobian_rs(
                model,
                top_n_v_safe,
                list(V_unsafe_init),
                dynamics_model=dynamics_model,
                translator=translator,
                N=rs_n,
                sigma=rs_sigma,
            )
            print(f"    J_RS 形状: {J.shape}")

            # ========== 5.5 内循环修复 ==========
            print(f"\n[迭代 {iteration+1}.4] 内循环修复 ({num_inner_steps} 步)...")

            FEATURE_BATCH_SIZE = 1024

            n_safe_fp = failed_safe_feature_points.shape[0]
            n_unsafe_fp = failed_unsafe_feature_points.shape[0]

            # 深度相关的学习率
            depth_lr_map = {
                10: 5e-3,
                12: 1e-3,
                15: 5e-4,
            }
            current_lr = depth_lr_map.get(current_max_depth, lr)
            print(f"    当前学习率: {current_lr:.0e} (depth={current_max_depth})")

            inner_history = []
            for inner_step in range(num_inner_steps):
                t0 = time.perf_counter()

                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                num_params = sum(p.numel() for p in model.parameters())

                g_F_total = torch.zeros(num_params, dtype=dtype, device=device)
                total_loss_sum = 0.0
                total_n = 0

                # 处理 unsafe 特征点
                for unsafe_start in range(0, n_unsafe_fp, FEATURE_BATCH_SIZE):
                    unsafe_end = min(unsafe_start + FEATURE_BATCH_SIZE, n_unsafe_fp)
                    unsafe_chunk = failed_unsafe_feature_points[unsafe_start:unsafe_end]

                    chunk_loss, chunk_g = compute_repair_loss_and_grad(
                        model=model,
                        dynamics_model=dynamics_model,
                        failed_safe_feature_points=torch.empty(0, 2, device=device, dtype=dtype),
                        failed_unsafe_feature_points=unsafe_chunk,
                        margin=0.1,
                        cbf_margin=0.0,
                        beta=5.0,
                        grad_clip_norm=10.0,
                        verbose=False,
                        translator=translator,
                    )
                    total_loss_sum += chunk_loss * unsafe_chunk.shape[0]
                    total_n += unsafe_chunk.shape[0]
                    g_F_total.add_(chunk_g)

                # 处理 safe 特征点
                for safe_start in range(0, n_safe_fp, FEATURE_BATCH_SIZE):
                    safe_end = min(safe_start + FEATURE_BATCH_SIZE, n_safe_fp)
                    safe_chunk = failed_safe_feature_points[safe_start:safe_end]

                    chunk_loss, chunk_g = compute_repair_loss_and_grad(
                        model=model,
                        dynamics_model=dynamics_model,
                        failed_safe_feature_points=safe_chunk,
                        failed_unsafe_feature_points=torch.empty(0, 2, device=device, dtype=dtype),
                        margin=0.1,
                        cbf_margin=0.0,
                        beta=5.0,
                        grad_clip_norm=10.0,
                        verbose=False,
                        translator=translator,
                    )
                    total_loss_sum += chunk_loss * safe_chunk.shape[0]
                    total_n += safe_chunk.shape[0]
                    g_F_total.add_(chunk_g)

                if total_n > 0:
                    loss_val = total_loss_sum / total_n
                else:
                    loss_val = 0.0

                grad_norm = g_F_total.norm().item()
                if grad_norm > 10.0:
                    g_F = g_F_total * (10.0 / grad_norm)
                else:
                    g_F = g_F_total

                t1 = time.perf_counter()

                g_raw_norm, update_norm, active = qp_project_and_update_gd(
                    model=model,
                    g_raw=g_F,
                    J_verified=J,
                    lr=current_lr,
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

        # ========== 5.6 保存 PyTorch 模型 ==========
        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v7.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        print(f"\n[迭代 {iteration+1}.5] 保存 PyTorch 模型: {pytorch_save_path}")

        # ========== 5.7 转换为 ONNX ==========
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v7.onnx"
        print(f"[迭代 {iteration+1}.6] 转换为 ONNX: {onnx_path}")
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

        # ========== 5.8 运行验证（使用当前max_depth） ==========
        print(f"\n[迭代 {iteration+1}.7] 运行验证 (max_depth={current_max_depth})...")
        results = verify_model(onnx_path, dynamics_model, max_depth=current_max_depth)

        # ========== 5.9 计算通过率 ==========
        certified_percentage, stats = calculate_pass_rate(results)

        print(f"\n[迭代 {iteration+1}.8] 验证结果:")
        print(f"    总样本数: {stats['total']}")
        print(f"    Certified: {stats['certified_pct']:.2f}%")
        print(f"    Uncertified: {stats['uncertified_pct']:.2f}%")
        print(f"    ★ 通过率 (Certified): {certified_percentage:.2f}%")

        # 计算加权安全指标
        safety_metrics = compute_safety_metrics(
            V_safe=results.get('V_safe', []),
            V_unsafe=results.get('V_unsafe', []),
            F_h_positive_in_unsafe=results.get('F_h_positive_in_unsafe', []),
            F_safe_cbf_violation=results.get('F_safe_cbf_violation', []),
            F_depth_limit_reached=results.get('F_depth_limit_reached', []),
            F_unsafe_cannot_split=results.get('F_unsafe_cannot_split', []),
        )
        print(f"    --- 安全指标（按面积）---")
        print(f"    USR (Unsafe-Set Pass Rate): {safety_metrics['usr']:.2f}%")
        print(f"    F_h漏检比例: {safety_metrics['f_h_ratio']:.2f}%")
        print(f"    加权安全分数: {safety_metrics['weighted_safety_score']:.2f}%")
        print(f"    与unsafe相交总面积: {safety_metrics['unsafe_intersect_area']:.4f}")
        print(f"    各区域面积: V_safe={safety_metrics['areas']['V_safe']:.4f}, "
              f"V_unsafe={safety_metrics['areas']['V_unsafe']:.4f}, "
              f"F_h={safety_metrics['areas']['F_h']:.4f}")

        # 最大深度 plateau 检测：记录首次达到最大深度的基准，并追踪连续改进
        if current_max_depth >= max_depth_limit:
            if first_max_depth_pass_rate is None:
                first_max_depth_pass_rate = certified_percentage
                at_max_depth_consecutive_no_improve = 0
                print(f"    [最大深度检测] 首次达到最大深度，基准通过率: {first_max_depth_pass_rate:.2f}%")
            else:
                improvement = certified_percentage - first_max_depth_pass_rate
                if improvement < plateau_threshold:
                    at_max_depth_consecutive_no_improve += 1
                    print(f"    [最大深度检测] 改进 {improvement:+.2f}% < {plateau_threshold}%，连续第{at_max_depth_consecutive_no_improve}次")
                else:
                    at_max_depth_consecutive_no_improve = 0
                    print(f"    [最大深度检测] 改进 {improvement:+.2f}% >= {plateau_threshold}%，重置计数")
        else:
            # 非最大深度时重置计数
            at_max_depth_consecutive_no_improve = 0

        # ========== 5.10 保存验证结果 ==========
        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired_v7.pt"
        print(f"\n[迭代 {iteration+1}.9] 保存验证区域: {verified_regions_path}")

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
            'max_depth': current_max_depth,
            'phase': current_phase,
        }

        torch.save(regions_to_save, verified_regions_path)

        # ========== 5.11 重新读取更新后的验证区域 ==========
        updated_regions_path = verified_regions_path
        print(f"[迭代 {iteration+1}.10] 重新读取验证区域: {updated_regions_path}")

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

        # ========== 5.12 更新状态 ==========
        definitive_fail_new = len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init)
        improvement = definitive_fail_prev - definitive_fail_new

        # 记录阶段2的改进历史
        if current_phase == 2:
            if improvement > 0:
                phase2_improvement_history.append(improvement)
            else:
                phase2_improvement_history.append(0.0)

        definitive_fail_prev = definitive_fail_new

        # 保存迭代结果
        iteration_results.append({
            'iteration': iteration + 1,
            'phase': current_phase,
            'max_depth': current_max_depth,
            'loss': inner_history[-1]['loss'] if 'inner_history' in dir() and inner_history else 0.0,
            'update_norm': inner_history[-1]['update_norm'] if 'inner_history' in dir() and inner_history else 0.0,
            'certified_percentage': stats['certified_percentage'],
            'usr': safety_metrics['usr'],
            'f_h_ratio': safety_metrics['f_h_ratio'],
            'weighted_safety_score': safety_metrics['weighted_safety_score'],
            'f_h_positive': len(F_h_positive_in_unsafe_init),
            'f_safe_violation': len(F_safe_cbf_violation_init),
            'f_depth': len(F_depth_limit_reached_init),
            'f_unsafe_split': len(F_unsafe_cannot_split_init),
            'areas': safety_metrics['areas'],  # 各区域面积
            'definitive_fail': definitive_fail_new,
            'uncertain_fail': len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init),
            'top_n_used': top_n_used if 'top_n_used' in dir() else 0,
            'improvement': improvement,
            'repair_type': repair_type if 'repair_type' in dir() else 'N/A',
        })

        # ========== 5.13 决定下一轮参数 ==========
        current_max_depth = next_max_depth
        current_phase = next_phase

        # 达到目标通过率可提前终止
        if certified_percentage >= target_pass_rate:
            print(f"\n  === 达到目标通过率 {target_pass_rate}%！提前终止 ===")
            break

    # ========== 6. 最终总结 ==========
    print(f"\n{'='*70}")
    print("迭代修复完成 - 总结 (v7: 渐进式深度分层修复)")
    print(f"{'='*70}")

    print(f"\n┌{'─'*6}┬{'─'*6}┬{'─'*8}┬{'─'*10}┬{'─'*8}┬{'─'*8}┬{'─'*8}┬{'─'*8}┐")
    print(f"│ {'迭代':^4} │ {'阶段':^4} │ {'深度':^6} │ {'通过率':^8} │ {'F_h':^6} │ {'F_safe':^6} │ {'F_depth':^6} │ {'F_split':^6} │")
    print(f"├{'─'*6}┼{'─'*6}┼{'─'*8}┼{'─'*10}┼{'─'*8}┼{'─'*8}┼{'─'*8}┼{'─'*8}┤")
    for r in iteration_results:
        print(f"│ {r['iteration']:^4} │ {r['phase']:^4} │ {r['max_depth']:^6} │ {r['certified_percentage']:>8.2f}% │ {r['f_h_positive']:^6} │ {r['f_safe_violation']:^6} │ {r['f_depth']:^6} │ {r['f_unsafe_split']:^6} │")
    print(f"└{'─'*6}┴{'─'*6}┴{'─'*8}┴{'─'*10}┴{'─'*8}┴{'─'*8}┴{'─'*8}┴{'─'*8}┘")

    model_initial_rate = _initial_pass_rate
    final_rate = iteration_results[-1]['certified_percentage'] if iteration_results else _initial_pass_rate
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
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_v7")
    os.makedirs(results_dir, exist_ok=True)

    run_result = {
        'system': system_name_key,
        'activation': activation,
        'method': 'Progressive Depth Phased Repair',
        'rs_n': rs_n,
        'rs_sigma': rs_sigma,
        'top_n_protect': top_n_protect,
        'max_depth_start': max_depth_start,
        'max_depth_limit': max_depth_limit,
        'depth_schedule': depth_schedule,
        'num_inner_steps': num_inner_steps,
        'lr': lr,
        'target_pass_rate': target_pass_rate,
        'plateau_threshold': plateau_threshold,
        'max_stagnant_iterations': max_stagnant_iterations,
        'max_total_iterations': max_total_iterations,
        'initial_pass_rate': model_initial_rate,
        'final_pass_rate': final_rate,
        'improvement': improvement,
        'num_iterations': len(iteration_results),
        'initial_regions': {
            'V_safe': len(V_safe_init) if 'V_safe_init' in dir() else 0,
            'V_unsafe': len(V_unsafe_init) if 'V_unsafe_init' in dir() else 0,
        },
        'iteration_results': iteration_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v7.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    print(f"\n[7] 单次结果已保存: {result_file}")

    print("\n" + "=" * 70)
    print("演示结束")
    print("=" * 70)


if __name__ == "__main__":
    main()