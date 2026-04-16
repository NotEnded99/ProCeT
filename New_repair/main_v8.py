"""
Neural CBF 迭代修复 v8：基于调和平均综合通过率的验证指标

核心改进 (相对于v7):
    用调和平均综合通过率代替原有的面积加权通过率
    解决unsafe区域占比小导致面积加权通过率失真的问题

指标设计:
    R_safe = Area(V_safe) / Area(V_safe + F_safe_cbf_violation + (F_depth + F_unsafe_split) / 2)
    R_unsafe = Area(V_unsafe) / Area(V_unsafe + F_h_positive_in_unsafe + (F_depth + F_unsafe_split) / 2)
    HarmonicMeanPassRate = 2 × R_safe × R_unsafe / (R_safe + R_unsafe)

修复策略（与v7相同）:
    阶段1（低深度）：只修复确定性违规（F_h_positive_in_unsafe, F_safe_cbf_violation）
    阶段2（深度递增）：逐渐修复不确定区域（F_depth_limit_reached, F_unsafe_cannot_split）
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


def compute_simplex_volume(simplex):
    """
    计算n维单纯形的体积（面积）

    公式: Volume = (1/n!) * |det(v1-v0, v2-v0, ..., vn-v0)|

    Args:
        simplex: 形状为 [n+1, n] 的 numpy array，n+1个顶点，每个顶点n维

    Returns:
        float: 单纯形体积（面积）
    """
    # 顶点数量
    num_vertices = simplex.shape[0]
    # 空间维度
    n = simplex.shape[1]

    if n == 0:
        return 0.0

    if num_vertices != n + 1:
        # 如果顶点数量与维度不匹配，尝试用外包框估计
        raise ValueError(f"Invalid simplex shape: expected [n+1, n], got {simplex.shape}")

    # 以第一个顶点为原点，计算其他顶点相对于它的向量
    origin = simplex[0]
    vectors = simplex[1:] - origin  # 形状: [n, n]

    # 体积 = (1/n!) * |det|
    det = np.linalg.det(vectors)
    volume = abs(det) / np.math.factorial(n)

    return volume


def compute_total_volume(simplices_list):
    """
    计算多个单纯形的总体积（面积）

    Args:
        simplices_list: 单纯形列表，每个元素是 [n+1, n] 的 numpy array

    Returns:
        float: 总体积（面积）
    """
    if not simplices_list:
        return 0.0
    return sum(compute_simplex_volume(s) for s in simplices_list)


def compute_safety_metrics_v8(
    V_safe,
    V_unsafe,
    F_h_positive_in_unsafe,
    F_safe_cbf_violation,
    F_depth_limit_reached,
    F_unsafe_cannot_split,
):
    """
    v8版本：计算基于调和平均的综合安全指标

    核心指标 (v8):
    - R_safe: 安全区验证通过率 = V_safe / (V_safe + F_safe_violation + uncertain/2)
    - R_unsafe: 危险区验证通过率 = V_unsafe / (V_unsafe + F_h + uncertain/2)
    - HarmonicMeanPassRate: 调和平均综合通过率 = 2*R_safe*R_unsafe/(R_safe+R_unsafe)

    不确定性均摊:
    - F_depth_limit_reached 和 F_unsafe_cannot_split 各50%分摊到安全区和危险区
    - 这是保守的中性假设

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
    # 按体积计算各区域
    volume_v_safe = compute_total_volume(V_safe)
    volume_v_unsafe = compute_total_volume(V_unsafe)
    volume_f_h = compute_total_volume(F_h_positive_in_unsafe)
    volume_f_safe_violation = compute_total_volume(F_safe_cbf_violation)
    volume_f_depth = compute_total_volume(F_depth_limit_reached)
    volume_f_unsafe_split = compute_total_volume(F_unsafe_cannot_split)

    total_volume = volume_v_safe + volume_v_unsafe + volume_f_h + volume_f_safe_violation + volume_f_depth + volume_f_unsafe_split

    # 不确定区域总体积（均摊）
    total_uncertain_volume = volume_f_depth + volume_f_unsafe_split
    uncertain_half = total_uncertain_volume / 2.0

    # ========== v8 核心指标：调和平均综合通过率 ==========

    # 真实安全区总体积（用于R_safe计算的分母）
    # = V_safe + F_safe_violation + 不确定区的一半
    true_safe_volume = volume_v_safe + volume_f_safe_violation + uncertain_half

    # 真实危险区总体积（用于R_unsafe计算的分母）
    # = V_unsafe + F_h + 不确定区的一半
    true_unsafe_volume = volume_v_unsafe + volume_f_h + uncertain_half

    # R_safe: 安全区验证通过率
    # 反映了CBF在正常行驶时不乱报警的能力
    if true_safe_volume > 0:
        R_safe = volume_v_safe / true_safe_volume
    else:
        R_safe = 0.0

    # R_unsafe: 危险区验证通过率
    # 反映了CBF在即将撞墙时成功拦截危险的能力
    # 致命漏检F_h越多，这个值越低
    if true_unsafe_volume > 0:
        R_unsafe = volume_v_unsafe / true_unsafe_volume
    else:
        R_unsafe = 0.0

    # 调和平均综合通过率（核心指标）
    # 当R_unsafe很低时，综合分数会被大幅拉低，体现"木桶效应"
    if (R_safe + R_unsafe) > 0:
        HarmonicMeanPassRate = 2.0 * R_safe * R_unsafe / (R_safe + R_unsafe)
    else:
        HarmonicMeanPassRate = 0.0

    # ========== 辅助指标 ==========

    # 标准通过率（按体积加权，仅作参考）
    standard_pass_rate = ((volume_v_safe + volume_v_unsafe) / total_volume * 100) if total_volume > 0 else 0.0

    # 与"真正不安全集合"相交的体积（仅作参考）
    unsafe_intersect_volume = volume_v_unsafe + volume_f_h

    # USR: Unsafe-Set Pass Rate（仅作参考）
    usr = (volume_v_unsafe / unsafe_intersect_volume * 100) if unsafe_intersect_volume > 0 else 0.0

    # F_h危险占比（仅作参考）
    f_h_ratio = (volume_f_h / unsafe_intersect_volume * 100) if unsafe_intersect_volume > 0 else 0.0

    # 不确定性覆盖率（仅作参考）
    uncertainty_ratio = (total_uncertain_volume / total_volume * 100) if total_volume > 0 else 0.0

    metrics = {
        # ========== v8 核心指标 ==========
        'R_safe': R_safe,                          # 安全区验证通过率
        'R_unsafe': R_unsafe,                       # 危险区验证通过率
        'HarmonicMeanPassRate': HarmonicMeanPassRate,            # 调和平均综合通过率（核心）
        'true_safe_volume': true_safe_volume,           # 真实安全区总体积
        'true_unsafe_volume': true_unsafe_volume,       # 真实危险区总体积

        # ========== 参考指标 ==========
        'standard_pass_rate': standard_pass_rate,  # 标准通过率（体积加权）
        'usr': usr,                                  # Unsafe-Set Pass Rate
        'f_h_ratio': f_h_ratio,                      # F_h漏检比例
        'uncertainty_ratio': uncertainty_ratio,      # 不确定性占比
        'unsafe_intersect_volume': unsafe_intersect_volume,  # 与unsafe相交的总体积
        'total_volume': total_volume,                    # 总体积

        # ========== 各区域体积 ==========
        'volumes': {
            'V_safe': volume_v_safe,
            'V_unsafe': volume_v_unsafe,
            'F_h': volume_f_h,
            'F_safe_violation': volume_f_safe_violation,
            'F_depth': volume_f_depth,
            'F_unsafe_split': volume_f_unsafe_split,
            'total_uncertain': total_uncertain_volume,
            'uncertain_half': uncertain_half,
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
    use_v8_metric=True,
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
        use_v8_metric: 是否使用v8指标（调和平均）

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
        last_verification_pass_rate: 上次验证通过率

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
        description='Neural CBF 迭代修复 v8 (基于调和平均综合通过率)'
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
    # v8 新增参数
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
    print(f"Neural CBF 迭代修复 v8 [基于调和平均综合通过率]")
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



    total_fail = (len(F_h_positive_in_unsafe_init) + len(F_safe_cbf_violation_init) +
                  len(F_depth_limit_reached_init) + len(F_unsafe_cannot_split_init))

    # ========== 3.1 计算原始区域（最大深度验证）的v8指标 ==========
    original_safety_metrics = compute_safety_metrics_v8(
        V_safe=V_safe_init,
        V_unsafe=V_unsafe_init,
        F_h_positive_in_unsafe=F_h_positive_in_unsafe_init,
        F_safe_cbf_violation=F_safe_cbf_violation_init,
        F_depth_limit_reached=F_depth_limit_reached_init,
        F_unsafe_cannot_split=F_unsafe_cannot_split_init,
    )

    # 原始最大深度验证结果
    original_max_depth_harmonic = original_safety_metrics['HarmonicMeanPassRate'] * 100
    original_max_depth_standard = original_safety_metrics['standard_pass_rate']
    original_max_depth_R_safe = original_safety_metrics['R_safe'] * 100
    original_max_depth_R_unsafe = original_safety_metrics['R_unsafe'] * 100

    print(f"\n[3.1] 原始区域（最大深度验证）v8 指标:")
    print(f"    original_max_depth_HarmonicMeanPassRate: {original_max_depth_harmonic:.2f}%")
    print(f"    original_max_depth_R_safe: {original_max_depth_R_safe:.2f}%")
    print(f"    original_max_depth_R_unsafe: {original_max_depth_R_unsafe:.2f}%")
    print(f"    original_max_depth_standard_pass_rate: {original_max_depth_standard:.2f}%")
    print(f"\n    初始 V_safe: {len(V_safe_init)}")
    print(f"    初始 V_unsafe: {len(V_unsafe_init)}")
    print(f"    初始 F_h_positive_in_unsafe: {len(F_h_positive_in_unsafe_init)}")
    print(f"    初始 F_safe_cbf_violation: {len(F_safe_cbf_violation_init)}")
    print(f"    初始 F_depth_limit_reached: {len(F_depth_limit_reached_init)}")
    print(f"    初始 F_unsafe_cannot_split: {len(F_unsafe_cannot_split_init)}")
    print(f"    总需修复区域数: {total_fail}")

    # ========== 3.2 检查是否需要修复 ==========
    if original_max_depth_standard >= 99.9 and original_max_depth_harmonic >= 99.9:
        print(f"\n[3.2] 验证通过率已达 99.9%，无需修复！")
        print(f"    original_max_depth_standard: {original_max_depth_standard:.2f}%")
        print(f"    original_max_depth_harmonic: {original_max_depth_harmonic:.2f}%")
        # 直接保存初始结果并退出
        import json
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_v8")
        os.makedirs(results_dir, exist_ok=True)
        run_result = {
            'system': system_name_key,
            'activation': activation,
            'method': 'Harmonic Mean CBF Pass Rate (v8)',
            'max_depth_start': max_depth_start,
            'max_depth_limit': max_depth_limit,
            'depth_schedule': depth_schedule,
            'num_inner_steps': num_inner_steps,
            'lr': lr,
            'target_pass_rate': target_pass_rate,
            'plateau_threshold': plateau_threshold,
            'max_stagnant_iterations': max_stagnant_iterations,
            'max_total_iterations': max_total_iterations,
            'original_max_depth_harmonic': original_max_depth_harmonic,
            'original_max_depth_standard': original_max_depth_standard,
            'original_max_depth_R_safe': original_max_depth_R_safe,
            'original_max_depth_R_unsafe': original_max_depth_R_unsafe,
            'final_harmonic_pass_rate': original_max_depth_harmonic,
            'final_standard_pass_rate': original_max_depth_standard,
            'final_R_safe': original_max_depth_R_safe,
            'final_R_unsafe': original_max_depth_R_unsafe,
            'harmonic_improvement': 0.0,
            'standard_improvement': 0.0,
            'num_iterations': 0,
            'iteration_results': [],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'skip_reason': 'already_100_percent',
        }
        result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v8.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(run_result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_file}")
        print("=" * 70)
        print("无需修复，程序结束")
        print("=" * 70)
        return

    # ========== 3.5 用起始深度验证初始模型（作为修复的基准） ==========
    print(f"\n[3.5] 用起始深度验证初始模型 (max_depth={max_depth_start})...")

    pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v8.pth"
    torch.save(model.state_dict(), pytorch_save_path)

    onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v8.onnx"
    pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)
    start_depth_results = verify_model(onnx_path, dynamics_model, max_depth=max_depth_start)

    start_depth_safety_metrics = compute_safety_metrics_v8(
        V_safe=start_depth_results.get('V_safe', []),
        V_unsafe=start_depth_results.get('V_unsafe', []),
        F_h_positive_in_unsafe=start_depth_results.get('F_h_positive_in_unsafe', []),
        F_safe_cbf_violation=start_depth_results.get('F_safe_cbf_violation', []),
        F_depth_limit_reached=start_depth_results.get('F_depth_limit_reached', []),
        F_unsafe_cannot_split=start_depth_results.get('F_unsafe_cannot_split', []),
    )

    # 起始深度验证结果（作为修复的基准）
    baseline_harmonic_pass_rate = start_depth_safety_metrics['HarmonicMeanPassRate'] * 100
    baseline_standard_pass_rate = start_depth_safety_metrics['standard_pass_rate']
    baseline_R_safe = start_depth_safety_metrics['R_safe'] * 100
    baseline_R_unsafe = start_depth_safety_metrics['R_unsafe'] * 100

    # 修复过程中使用的初始基准值
    initial_harmonic_pass_rate = baseline_harmonic_pass_rate
    initial_standard_pass_rate = baseline_standard_pass_rate
    initial_R_safe = baseline_R_safe
    initial_R_unsafe = baseline_R_unsafe

    print(f"    基准验证通过率 (HarmonicMeanPassRate, 深度{max_depth_start}): {baseline_harmonic_pass_rate:.2f}%")
    print(f"    基准 R_safe: {baseline_R_safe:.2f}%")
    print(f"    基准 R_unsafe: {baseline_R_unsafe:.2f}%")
    print(f"    基准 standard_pass_rate (体积加权, 仅参考): {baseline_standard_pass_rate:.2f}%")

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
            last_verification_pass_rate=initial_harmonic_pass_rate,
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
            certified_percentage = initial_harmonic_pass_rate
            stats = {
                'total': 0,
                'R_safe': 0.0,
                'R_unsafe': 0.0,
                'HarmonicMeanPassRate': certified_percentage,
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
        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v8.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        print(f"\n[迭代 {iteration+1}.5] 保存 PyTorch 模型: {pytorch_save_path}")

        # ========== 5.7 转换为 ONNX ==========
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v8.onnx"
        print(f"[迭代 {iteration+1}.6] 转换为 ONNX: {onnx_path}")
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

        # ========== 5.8 运行验证（使用当前max_depth） ==========
        print(f"\n[迭代 {iteration+1}.7] 运行验证 (max_depth={current_max_depth})...")
        results = verify_model(onnx_path, dynamics_model, max_depth=current_max_depth)

        # ========== 5.9 计算 v8 指标 ==========
        safety_metrics = compute_safety_metrics_v8(
            V_safe=results.get('V_safe', []),
            V_unsafe=results.get('V_unsafe', []),
            F_h_positive_in_unsafe=results.get('F_h_positive_in_unsafe', []),
            F_safe_cbf_violation=results.get('F_safe_cbf_violation', []),
            F_depth_limit_reached=results.get('F_depth_limit_reached', []),
            F_unsafe_cannot_split=results.get('F_unsafe_cannot_split', []),
        )

        # 核心指标（百分比）
        certified_percentage = safety_metrics['HarmonicMeanPassRate'] * 100
        R_safe_pct = safety_metrics['R_safe'] * 100
        R_unsafe_pct = safety_metrics['R_unsafe'] * 100

        print(f"\n[迭代 {iteration+1}.8] 验证结果 (v8 调和平均指标):")
        print(f"    ★ 综合通过率 (调和平均): {certified_percentage:.2f}%")
        print(f"    ★ R_safe (安全区通过率): {R_safe_pct:.2f}%")
        print(f"    ★ R_unsafe (危险区通过率): {R_unsafe_pct:.2f}%")
        print(f"    --- 参考指标 ---")
        print(f"    standard_pass_rate (面积加权): {safety_metrics['standard_pass_rate']:.2f}%")
        print(f"    USR (仅参考): {safety_metrics['usr']:.2f}%")
        print(f"    F_h漏检比例 (仅参考): {safety_metrics['f_h_ratio']:.2f}%")
        print(f"    不确定性占比 (仅参考): {safety_metrics['uncertainty_ratio']:.2f}%")
        print(f"    各区域体积:")
        print(f"      V_safe={safety_metrics['volumes']['V_safe']:.4f}, "
              f"V_unsafe={safety_metrics['volumes']['V_unsafe']:.4f}, "
              f"F_h={safety_metrics['volumes']['F_h']:.4f}")
        print(f"      F_safe_violation={safety_metrics['volumes']['F_safe_violation']:.4f}, "
              f"F_depth={safety_metrics['volumes']['F_depth']:.4f}, "
              f"F_unsafe_split={safety_metrics['volumes']['F_unsafe_split']:.4f}")
        print(f"    真实安全区总体积: {safety_metrics['true_safe_volume']:.4f}")
        print(f"    真实危险区总体积: {safety_metrics['true_unsafe_volume']:.4f}")

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
        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired_v8.pt"
        print(f"\n[迭代 {iteration+1}.9] 保存验证区域: {verified_regions_path}")

        regions_to_save = {
            'V_safe': results.get('V_safe', V_safe_init),
            'V_unsafe': results.get('V_unsafe', V_unsafe_init),
            'F_h_positive_in_unsafe': results.get('F_h_positive_in_unsafe', F_h_positive_in_unsafe_init),
            'F_safe_cbf_violation': results.get('F_safe_cbf_violation', F_safe_cbf_violation_init),
            'F_depth_limit_reached': results.get('F_depth_limit_reached', F_depth_limit_reached_init),
            'F_unsafe_cannot_split': results.get('F_unsafe_cannot_split', F_unsafe_cannot_split_init),
            'Certified percentage': certified_percentage,  # v8 调和平均指标
            'R_safe': R_safe_pct,
            'R_unsafe': R_unsafe_pct,
            'stats': {
                'total': sum(len(results.get(k, [])) for k in ['V_safe', 'V_unsafe', 'F_h_positive_in_unsafe',
                                                                'F_safe_cbf_violation', 'F_depth_limit_reached',
                                                                'F_unsafe_cannot_split']),
                'R_safe': R_safe_pct,
                'R_unsafe': R_unsafe_pct,
                'HarmonicMeanPassRate': certified_percentage,
            },
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
            'HarmonicMeanPassRate': certified_percentage,  # v8 调和平均（核心指标）
            'R_safe': R_safe_pct,
            'R_unsafe': R_unsafe_pct,
            'standard_pass_rate': safety_metrics['standard_pass_rate'],  # v7面积加权（参考）
            'f_h_positive': len(F_h_positive_in_unsafe_init),
            'f_safe_violation': len(F_safe_cbf_violation_init),
            'f_depth': len(F_depth_limit_reached_init),
            'f_unsafe_split': len(F_unsafe_cannot_split_init),
            'volumes': safety_metrics['volumes'],
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
    print("迭代修复完成 - 总结 (v8: 基于调和平均综合通过率)")
    print(f"{'='*70}")

    # 迭代详情表格（显示两种通过率）
    print(f"\n┌{'─'*6}┬{'─'*6}┬{'─'*6}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*8}┬{'─'*8}┬{'─'*8}┬{'─'*8}┐")
    print(f"│ {'迭代':^4} │ {'阶段':^4} │ {'深度':^4} │ {'Harmonic%':^8} │ {'Standard%':^8} │ {'R_safe':^8} │ {'R_unsafe':^8} │ {'F_h':^6} │ {'F_safe':^6} │ {'F_depth':^6} │ {'F_split':^6} │")
    print(f"├{'─'*6}┼{'─'*6}┼{'─'*6}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*8}┼{'─'*8}┼{'─'*8}┼{'─'*8}┤")
    for r in iteration_results:
        print(f"│ {r['iteration']:^4} │ {r['phase']:^4} │ {r['max_depth']:^4} │ {r['HarmonicMeanPassRate']:>8.2f}% │ {r['standard_pass_rate']:>8.2f}% │ {r['R_safe']:>8.2f}% │ {r['R_unsafe']:>8.2f}% │ {r['f_h_positive']:^6} │ {r['f_safe_violation']:^6} │ {r['f_depth']:^6} │ {r['f_unsafe_split']:^6} │")
    print(f"└{'─'*6}┴{'─'*6}┴{'─'*6}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*8}┴{'─'*8}┴{'─'*8}┴{'─'*8}┘")

    # 初始vs最终对比
    final_harmonic = iteration_results[-1]['HarmonicMeanPassRate'] if iteration_results else initial_harmonic_pass_rate
    final_standard = iteration_results[-1]['standard_pass_rate'] if iteration_results else initial_standard_pass_rate
    final_R_safe = iteration_results[-1]['R_safe'] if iteration_results else initial_R_safe
    final_R_unsafe = iteration_results[-1]['R_unsafe'] if iteration_results else initial_R_unsafe

    harmonic_improvement = final_harmonic - original_max_depth_harmonic
    standard_improvement = final_standard - original_max_depth_standard

    print(f"\n{'='*70}")
    print("修复前后对比 (原始最大深度验证结果)")
    print(f"{'='*70}")
    print(f"指标                     原始          最终          变化")
    print(f"───────────────────────────────────────────────────────")
    print(f"HarmonicMeanPassRate:    {original_max_depth_harmonic:>8.2f}%   {final_harmonic:>8.2f}%   ({harmonic_improvement:+.2f}%)")
    print(f"standard_pass_rate:      {original_max_depth_standard:>8.2f}%   {final_standard:>8.2f}%   ({standard_improvement:+.2f}%)")
    print(f"R_safe:                 {original_max_depth_R_safe:>8.2f}%   {final_R_safe:>8.2f}%")
    print(f"R_unsafe:               {original_max_depth_R_unsafe:>8.2f}%   {final_R_unsafe:>8.2f}")
    print(f"───────────────────────────────────────────────────────")
    print(f"注: 原始=加载的regions_data的指标（最大深度验证结果）")

    if harmonic_improvement > 0:
        print("\n✓ 修复有效（HarmonicMeanPassRate 提升）!")
    elif harmonic_improvement < 0:
        print("\n✗ 修复效果负向（HarmonicMeanPassRate 下降）")
    else:
        print("\n- 修复效果持平")

    # ========== 7. 保存结果到 JSON ==========
    import json
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_v8")
    os.makedirs(results_dir, exist_ok=True)

    run_result = {
        'system': system_name_key,
        'activation': activation,
        'method': 'Harmonic Mean CBF Pass Rate (v8)',
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
        # 原始最大深度验证结果
        'original_max_depth_harmonic': original_max_depth_harmonic,
        'original_max_depth_standard': original_max_depth_standard,
        'original_max_depth_R_safe': original_max_depth_R_safe,
        'original_max_depth_R_unsafe': original_max_depth_R_unsafe,
        # 最终结果
        'final_harmonic_pass_rate': final_harmonic,
        'final_standard_pass_rate': final_standard,
        'final_R_safe': final_R_safe,
        'final_R_unsafe': final_R_unsafe,
        # 改进量（相对于起始深度基准）
        'harmonic_improvement': harmonic_improvement,
        'standard_improvement': standard_improvement,
        'num_iterations': len(iteration_results),
        'initial_regions': {
            'V_safe': len(V_safe_init) if 'V_safe_init' in dir() else 0,
            'V_unsafe': len(V_unsafe_init) if 'V_unsafe_init' in dir() else 0,
        },
        'iteration_results': iteration_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v8.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    print(f"\n[7] 单次结果已保存: {result_file}")

    print("\n" + "=" * 70)
    print("演示结束")
    print("=" * 70)


if __name__ == "__main__":
    main()
