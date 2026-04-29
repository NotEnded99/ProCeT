"""
Neural CBF 迭代修复 v10_lbp (LBP验证 + LBP损失计算，无McCormick)

与 main_clean_v10_ibp.py 的区别:
    - 验证部分使用 LBP (verify_cbf_lbp_wo_mccormick) 而非 verify_cbf
    - 计算损失部分使用 LBP without McCormick (direct interval multiplication)
      替代 IBP，避免 McCormick 松弛的梯度 NaN 问题

其他内容与 main_clean_v10_ibp.py 一致。
"""

import sys
import os
import random
import argparse
import math
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
from lbp_neural_cbf.translators import TorchTranslator
from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization

# v10_lbp: 使用 LBP 验证 (without McCormick) + LBP 损失计算
from lbp_neural_cbf.cbf.verify_cbf_lbp_wo_McCormick import verify_cbf_lbp_wo_mccormick, compute_simplex_bound_batch_lbp_wo_mccormick, compute_min_L_lbp_wo_mccormick


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
    """
    使用 LBP (verify_cbf_lbp_wo_mccormick) 对模型进行验证

    Args:
        model_path: 已保存的模型路径 (.pth)
        dynamics_model: 动力学系统模型
        max_depth: 最大分裂深度

    Returns:
        验证结果字典
    """
    results = verify_cbf_lbp_wo_mccormick(
        dynamics_model,
        barrier_model_path=model_path,
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
    volume = abs(det) / math.factorial(n)

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

    true_safe_volume = volume_v_safe + volume_f_safe_violation + volume_f_depth_unsafe
    true_unsafe_volume = volume_v_unsafe + volume_f_h + volume_f_depth_safe

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
        }
    }


def select_top_n_v_safe_lbp(model, V_safe, dynamics_model, lbp_linearizer, top_n, cbf_margin=0.0):
    """
    使用 LBP (without McCormick) 计算 min_L 来选择 top_n 个安全区域
    """
    if len(V_safe) == 0:
        return []
    n_available = len(V_safe)
    actual_n = min(top_n, n_available)
    BATCH_SIZE = 1024
    all_margins = []
    for batch_start in range(0, n_available, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_available)
        V_safe_batch = V_safe[batch_start:batch_end]
        min_L_batch = compute_simplex_bound_batch_lbp_wo_mccormick(
            model, V_safe_batch, 'safe', dynamics_model=dynamics_model, network_linearizer=lbp_linearizer
        )
        margins_batch = min_L_batch.detach().cpu().numpy() if isinstance(min_L_batch, torch.Tensor) else np.array(min_L_batch)
        all_margins.append(margins_batch)
    margins = np.concatenate(all_margins, axis=0)
    if actual_n == n_available:
        selected_indices = list(range(n_available))
    else:
        selected_indices = np.argsort(margins)[:actual_n].tolist()
    return [V_safe[i] for i in selected_indices]


def select_repair_targets(F_h_positive_in_unsafe, F_safe_cbf_violation, F_depth_limit_reached_unsafe, F_depth_limit_reached_safe, F_unsafe_cannot_split, current_phase):
    if current_phase == 1:
        return list(F_safe_cbf_violation), list(F_h_positive_in_unsafe), "Phase1_Definitive"
    else:
        return (list(F_safe_cbf_violation) + list(F_depth_limit_reached_safe)), \
               (list(F_h_positive_in_unsafe) + list(F_unsafe_cannot_split) + list(F_depth_limit_reached_unsafe)), "Phase2_All"


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


def simple_gradient_update(model, g_F, lr):
    """简化版梯度更新：直接对模型参数进行梯度下降"""
    params = list(model.parameters())
    g_flat = g_F.clone()
    offset = 0
    for p in params:
        numel = p.numel()
        grad_slice = g_flat[offset:offset + numel].view(p.shape)
        with torch.no_grad():
            p.sub_(lr * grad_slice)
        offset += numel
    return (g_flat * lr).norm().item()


# =============================================================================
# LBP Loss Computation Functions (without McCormick)
# =============================================================================

def vec_min(*tensors: torch.Tensor) -> torch.Tensor:
    """Compute element-wise minimum across multiple tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.minimum(result, t)
    return result


def vec_max(*tensors: torch.Tensor) -> torch.Tensor:
    """Compute element-wise maximum across multiple tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.maximum(result, t)
    return result


def _compute_dynamics_bounds_lbp(batch, dynamics_model, device, dtype):
    """
    Compute dynamics bounds by evaluating f(x) and g(x) at region vertices/centers.
    """
    from lbp_neural_cbf.translators import TorchTranslator
    from lbp_neural_cbf.regions import SimplicialRegion

    n = dynamics_model.input_dim
    m = dynamics_model.control_dim

    translator = TorchTranslator(device=device, dtype=dtype)

    if isinstance(batch[0], SimplicialRegion):
        f_L_list = []
        f_U_list = []
        g_L_list = []
        g_U_list = []

        for sample in batch:
            verts = torch.tensor(sample.vertices, device=device, dtype=dtype)
            f_vals = []
            g_vals = []
            for v in verts:
                x = v.unsqueeze(0)
                with torch.no_grad():
                    f_val = dynamics_model.compute_f(x, translator).squeeze(0)
                    f_vals.append(f_val)
                    if m > 0:
                        g_val = dynamics_model.compute_g(x, translator).squeeze(0)
                        g_vals.append(g_val)

            f_vals = torch.stack(f_vals, dim=0)
            f_L_list.append(f_vals.min(dim=0).values)
            f_U_list.append(f_vals.max(dim=0).values)

            if m > 0:
                g_vals = torch.stack(g_vals, dim=0)
                g_L_list.append(g_vals.min(dim=0).values)
                g_U_list.append(g_vals.max(dim=0).values)

        f_L = torch.stack(f_L_list, dim=0)
        f_U = torch.stack(f_U_list, dim=0)

        if m > 0:
            g_L = torch.stack(g_L_list, dim=0)
            g_U = torch.stack(g_U_list, dim=0)
        else:
            g_L = None
            g_U = None
    else:
        raise TypeError(f"Unsupported region type: {type(batch[0])}")

    f_bounds = (f_L, f_U)
    g_bounds = (g_L, g_U) if m > 0 else None

    return f_bounds, g_bounds


def compute_repair_loss_and_grad_unsafe_lbp_wo_mccormick(
    model: nn.Module,
    dynamics_model,
    unsafe_simplices: list,
    lbp_linearizer,
    margin: float = 0.0,
    beta: float = 5.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
):
    """
    使用 LBP 上界计算 Unsafe 区域违规的修复损失和梯度 (without McCormick).

    对于 F_h_positive_in_unsafe 区域（障碍区中 h >= 0 的违规）:
    - 计算 h 的 LBP 上界 h_ub
    - loss = softplus(h_ub + margin)，推动 h_ub + margin <= 0

    Args:
        model: BarrierNN 网络
        dynamics_model: 动力学系统
        unsafe_simplices: Unsafe 违规区域的单纯形列表
        lbp_linearizer: CrownPartialLinearization 实例
        margin: h 值的容差
        beta: softplus 的 beta 参数
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出

    Returns:
        total_loss: 平均损失值
        g_raw: 修复梯度 [num_params]
    """
    from lbp_neural_cbf.regions import SimplicialRegion

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    if len(unsafe_simplices) == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    regions = []
    for verts in unsafe_simplices:
        if isinstance(verts, np.ndarray):
            verts_np = verts.astype(np.float32)
        else:
            verts_np = verts.detach().cpu().numpy().astype(np.float32)
        regions.append(SimplicialRegion(verts_np, output_dim=None))

    BATCH_SIZE = 1024
    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_loss_sum = 0.0
    total_n = 0

    for batch_start in range(0, len(regions), BATCH_SIZE):
        batch_regions = regions[batch_start:batch_start + BATCH_SIZE]
        B = len(batch_regions)

        lbp_linearizer.compute_network_bounds(batch_regions)
        h_lb_batch, h_ub_batch = lbp_linearizer.get_network_output_bounds()
        h_ub = h_ub_batch.reshape(B, -1)[:, 0]

        loss_batch = torch.nn.functional.softplus(h_ub + margin, beta=beta)

        # if not torch.isfinite(loss_batch).all():
        #     if verbose:
        #         print(f"  [警告] unsafe LBP batch [{batch_start}:{batch_start+B}] 存在 NaN/Inf，跳过")
        #     del h_ub, h_ub_batch, h_lb_batch, loss_batch
        #     torch.cuda.empty_cache()
        #     continue

        total_loss_sum += loss_batch.sum().item()
        total_n += B

        model.zero_grad()
        loss_batch.sum().backward()

        grad_batch = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        # grad_batch = torch.nan_to_num(grad_batch, nan=0.0, posinf=0.0, neginf=0.0)
        if grad_batch.isnan().any() or grad_batch.isinf().any():
            raise ValueError(f"Safe LBP batch [{batch_start}:{batch_start+B}] 计算得到 NaN/Inf 梯度，可能是 min_L 计算异常")
        # grad_batch = torch.nan_to_num(grad_batch, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_batch)

        g_raw.add_(grad_batch)

        del h_ub, h_ub_batch, h_lb_batch, loss_batch, grad_batch
        torch.cuda.empty_cache()

    if total_n == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    mean_loss = total_loss_sum / total_n

    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)

    if verbose:
        print(f"  [Unsafe LBP wo McCormick] loss={mean_loss:.6f}, |g|={grad_norm:.4f}, n={len(regions)}")

    return mean_loss, g_raw


def compute_repair_loss_and_grad_safe_lbp_wo_mccormick(
    model: nn.Module,
    dynamics_model,
    safe_simplices: list,
    lbp_linearizer,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
):
    """
    使用 LBP 下界计算 Safe 区域 CBF 违规的修复损失和梯度 (without McCormick).

    对于 F_safe_cbf_violation / F_depth_limit_reached 区域（CBF 条件违规）:
    - 计算 CBF 条件的 LBP 下界 min_L (使用 direct interval multiplication)
    - loss = softplus(cbf_margin - min_L)，推动 min_L >= cbf_margin

    Args:
        model: BarrierNN 网络
        dynamics_model: 动力学系统
        safe_simplices: CBF 违规区域的单纯形列表
        lbp_linearizer: CrownPartialLinearization 实例
        cbf_margin: CBF 条件的容差
        beta: softplus 的 beta 参数
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出

    Returns:
        total_loss: 平均损失值
        g_raw: 修复梯度 [num_params]
    """
    from lbp_neural_cbf.regions import SimplicialRegion

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    if len(safe_simplices) == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    regions = []
    for verts in safe_simplices:
        if isinstance(verts, np.ndarray):
            verts_np = verts.astype(np.float32)
        else:
            verts_np = verts.detach().cpu().numpy().astype(np.float32)
        regions.append(SimplicialRegion(verts_np, output_dim=None))

    BATCH_SIZE = 1024
    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_loss_sum = 0.0
    total_n = 0

    for batch_start in range(0, len(regions), BATCH_SIZE):
        batch_regions = regions[batch_start:batch_start + BATCH_SIZE]
        B = len(batch_regions)

        lbp_linearizer.compute_network_bounds(batch_regions)
        h_lb_lbp, h_ub_lbp = lbp_linearizer.get_network_output_bounds()
        h_lb_lbp_batch = h_lb_lbp.reshape(B, -1)[:, 0]

        lbp_linearizer.keep_indices(list(range(B)))
        lbp_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=0)

        min_L = compute_min_L_lbp_wo_mccormick(batch_regions, dynamics_model, lbp_linearizer, device, dtype, h_lb_lbp=h_lb_lbp_batch)
        min_L = min_L.reshape(-1)

        loss_batch = torch.nn.functional.softplus(cbf_margin - min_L, beta=beta)

        if not torch.isfinite(loss_batch).all():
            if verbose:
                print(f"  [警告] safe LBP batch [{batch_start}:{batch_start+B}] 存在 NaN/Inf，跳过")
            del h_lb_lbp, h_ub_lbp, h_lb_lbp_batch, min_L, loss_batch
            torch.cuda.empty_cache()
            continue

        total_loss_sum += loss_batch.sum().item()
        total_n += B

        model.zero_grad()
        loss_batch.sum().backward()

        grad_batch = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), dtype=dtype, device=device)
            for p in model.parameters()
        ])
        #
        if grad_batch.isnan().any() or grad_batch.isinf().any():
            raise ValueError(f"Safe LBP batch [{batch_start}:{batch_start+B}] 计算得到 NaN/Inf 梯度，可能是 min_L 计算异常")
        # grad_batch = torch.nan_to_num(grad_batch, nan=0.0, posinf=0.0, neginf=0.0)
        g_raw.add_(grad_batch)

        del h_lb_lbp, h_ub_lbp, h_lb_lbp_batch, min_L, loss_batch, grad_batch
        torch.cuda.empty_cache()

    if total_n == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    mean_loss = total_loss_sum / total_n

    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)

    if verbose:
        print(f"  [Safe LBP wo McCormick] loss={mean_loss:.6f}, |g|={grad_norm:.4f}, n={len(regions)}")

    return mean_loss, g_raw


def compute_repair_loss_and_grad_lbp(
    model: nn.Module,
    dynamics_model,
    safe_simplices: list,
    unsafe_simplices: list,
    lbp_linearizer,
    margin: float = 0.0,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
):
    """
    综合修复损失计算 (LBP without McCormick):

    - Unsafe: LBP 上界 (h_ub)
    - Safe: LBP 下界 (min_L) 使用 direct interval multiplication

    Args:
        model: BarrierNN
        dynamics_model: 动力学系统
        safe_simplices: CBF 违规区域的单纯形列表
        unsafe_simplices: 障碍区违规区域的单纯形列表
        lbp_linearizer: CrownPartialLinearization 实例
        margin: h 值的容差（用于 unsafe）
        cbf_margin: CBF 条件的容差（用于 safe）
        beta: softplus 的 beta 参数
        grad_clip_norm: 梯度裁剪阈值
        verbose: 诊断输出

    Returns:
        total_loss: 平均损失值
        g_raw: 修复梯度 [num_params]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in model.parameters())

    g_raw = torch.zeros(num_params, dtype=dtype, device=device)
    total_loss_sum = 0.0
    n_valid = 0

    # 处理 Unsafe 区域 (使用 LBP 上界)
    if len(unsafe_simplices) > 0:
        loss_unsafe, grad_unsafe = compute_repair_loss_and_grad_unsafe_lbp_wo_mccormick(
            model, dynamics_model, unsafe_simplices, lbp_linearizer,
            margin=margin, beta=beta, grad_clip_norm=grad_clip_norm, verbose=verbose
        )
        if torch.isfinite(grad_unsafe).all():
            g_raw.add_(grad_unsafe)
            total_loss_sum += loss_unsafe
            n_valid += 1

    # 处理 Safe 区域 (使用 LBP 下界，without McCormick)
    if len(safe_simplices) > 0:
        loss_safe, grad_safe = compute_repair_loss_and_grad_safe_lbp_wo_mccormick(
            model, dynamics_model, safe_simplices, lbp_linearizer,
            cbf_margin=cbf_margin, beta=beta, grad_clip_norm=grad_clip_norm, verbose=verbose
        )
        if torch.isfinite(grad_safe).all():
            g_raw.add_(grad_safe)
            total_loss_sum += loss_safe
            n_valid += 1

    if n_valid == 0:
        return 0.0, torch.zeros(num_params, dtype=dtype, device=device)

    total_loss = total_loss_sum / n_valid
    grad_norm = g_raw.norm().item()

    if verbose:
        print(f"  [修复损失 LBP wo McCormick] total_loss={total_loss:.6f}, |g|={grad_norm:.4f}, "
              f"unsafe={len(unsafe_simplices)}, safe={len(safe_simplices)}")

    return total_loss, g_raw


def main():
    parser = argparse.ArgumentParser(description='Neural CBF 迭代修复 v10_lbp (LBP验证 + LBP损失计算，无McCormick)')
    parser.add_argument('--activation', '-a', type=str, required=True, choices=SUPPORTED_ACTIVATIONS)
    parser.add_argument('--system', '-s', type=str, required=True, choices=list(DYNAMICS_SYSTEMS.keys()))
    parser.add_argument('--top-n-protect', type=int, default=500)
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
    print(f"Neural CBF 迭代修复 v10_lbp (LBP验证 + LBP损失，无McCormick)")
    print(f"  激活={activation}, 系统={system_name_key}")
    print(f"  注意: 此版本使用 LBP 验证，LBP损失计算 (without McCormick)")
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

    original_safety_metrics = compute_safety_metrics_v8(V_safe_init, V_unsafe_init, F_h_positive_in_unsafe_init, F_safe_cbf_violation_init, F_depth_limit_reached_unsafe_init, F_depth_limit_reached_safe_init, F_unsafe_cannot_split_init)
    original_max_depth_harmonic = original_safety_metrics['HarmonicMeanPassRate'] * 100
    original_max_depth_standard = original_safety_metrics['standard_pass_rate']
    original_max_depth_R_safe = original_safety_metrics['R_safe'] * 100
    original_max_depth_R_unsafe = original_safety_metrics['R_unsafe'] * 100

    print(f"\n[3.1] 原始区域 v10_lbp 指标: HarmonicMeanPassRate={original_max_depth_harmonic:.2f}%, R_safe={original_max_depth_R_safe:.2f}%, R_unsafe={original_max_depth_R_unsafe:.2f}%")

    # ========== 3.2 检查是否需要修复 ==========
    if original_max_depth_standard >= 99.9 and original_max_depth_harmonic >= 99.9:
        print(f"\n[3.2] 验证通过率已达 99.9%，无需修复！")
        print(f"    original_max_depth_standard: {original_max_depth_standard:.2f}%")
        print(f"    original_max_depth_harmonic: {original_max_depth_harmonic:.2f}%")
        import json
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_v10_lbp")
        os.makedirs(results_dir, exist_ok=True)
        run_result = {
            'system': system_name_key, 'activation': activation,
            'method': 'Harmonic Mean CBF Pass Rate (v10_lbp, LBP验证 + LBP损失，无McCormick)',
            'max_depth_start': max_depth_start, 'max_depth_limit': max_depth_limit,
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
        result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v10_lbp.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(run_result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_file}")
        print("=" * 70)
        print("无需修复，程序结束")
        print("=" * 70)
        return

    pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v10_lbp.pth"
    torch.save(model.state_dict(), pytorch_save_path)
    onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v10_lbp.onnx"
    pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)
    start_depth_results = verify_model(pytorch_save_path, dynamics_model, max_depth=max_depth_start)

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
    top_n_used = 0

    print(f"\n开始渐进式深度分层修复 (v10_lbp, LBP验证 + LBP损失，无McCormick), Phase {current_phase}, max_depth={current_max_depth}")

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
        print(f"\n[迭代 {iteration+1}] depth_reason={depth_reason}, max_depth {current_max_depth}->{next_max_depth}, phase {current_phase}->{next_phase}")

        failed_safe_simplices, failed_unsafe_simplices, repair_type = select_repair_targets(
            F_h_positive_in_unsafe_init, F_safe_cbf_violation_init, F_depth_limit_reached_unsafe_init, F_depth_limit_reached_safe_init, F_unsafe_cannot_split_init, current_phase,
        )

        if len(failed_safe_simplices) == 0 and len(failed_unsafe_simplices) == 0:
            certified_percentage = initial_harmonic_pass_rate
            inner_history = []
        else:
            # 使用 LBP without McCormick 选择修复目标
            lbp_linearizer = CrownPartialLinearization(model, dtype=torch.float32)
            top_n_v_safe = select_top_n_v_safe_lbp(model, list(V_safe_init), dynamics_model, lbp_linearizer, top_n_protect)
            top_n_used = len(top_n_v_safe)

            depth_lr_map = {10: 5e-3, 12: 5e-3, 15: 5e-3}
            current_lr = depth_lr_map.get(current_max_depth, lr)

            inner_history = []
            for inner_step in range(num_inner_steps):
                t0 = time.perf_counter()
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype

                # 使用 LBP without McCormick 计算损失
                loss_val, g_F = compute_repair_loss_and_grad_lbp(
                    model=model,
                    dynamics_model=dynamics_model,
                    safe_simplices=failed_safe_simplices,
                    unsafe_simplices=failed_unsafe_simplices,
                    lbp_linearizer=lbp_linearizer,
                    margin=0.1,
                    cbf_margin=0.0,
                    beta=5.0,
                    grad_clip_norm=10.0,
                    verbose=False,
                )

                grad_norm = g_F.norm().item()
                g_F_clipped = g_F * (10.0 / grad_norm) if grad_norm > 10.0 else g_F
                t1 = time.perf_counter()

                update_norm = simple_gradient_update(model, g_F_clipped, current_lr)
                t2 = time.perf_counter()

                inner_history.append({'step': inner_step + 1, 'loss': loss_val, 'g_raw_norm': grad_norm, 'update_norm': update_norm})
                print(f"    [内步 {inner_step+1}] loss={loss_val:.6f}, |g|={grad_norm:.4f}, |d|={update_norm:.6f}")

        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v10_lbp.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_v10_lbp.onnx"
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

        # 使用 LBP without McCormick 验证
        results = verify_model(pytorch_save_path, dynamics_model, max_depth=current_max_depth)

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
                if improvement < plateau_threshold:
                    at_max_depth_consecutive_no_improve += 1
                else:
                    at_max_depth_consecutive_no_improve = 0
        else:
            at_max_depth_consecutive_no_improve = 0

        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired_v10_lbp.pt"
        regions_to_save = {
            'V_safe': results.get('V_safe', V_safe_init), 'V_unsafe': results.get('V_unsafe', V_unsafe_init),
            'F_h_positive_in_unsafe': results.get('F_h_positive_in_unsafe', F_h_positive_in_unsafe_init),
            'F_safe_cbf_violation': results.get('F_safe_cbf_violation', F_safe_cbf_violation_init),
            'F_depth_limit_reached_unsafe': results.get('F_depth_limit_reached_unsafe', F_depth_limit_reached_unsafe_init),
            'F_depth_limit_reached_safe': results.get('F_depth_limit_reached_safe', F_depth_limit_reached_safe_init),
            'F_unsafe_cannot_split': results.get('F_unsafe_cannot_split', F_unsafe_cannot_split_init),
            'Certified percentage': certified_percentage,
        }
        torch.save(regions_to_save, verified_regions_path)

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
            'loss': inner_history[-1]['loss'] if inner_history else 0.0,
            'HarmonicMeanPassRate': certified_percentage, 'R_safe': R_safe_pct, 'R_unsafe': R_unsafe_pct,
            'standard_pass_rate': safety_metrics['standard_pass_rate'],
            'f_h_positive': len(F_h_positive_in_unsafe_init), 'f_safe_violation': len(F_safe_cbf_violation_init),
            'f_depth_unsafe': len(F_depth_limit_reached_unsafe_init), 'f_depth_safe': len(F_depth_limit_reached_safe_init),
            'f_unsafe_split': len(F_unsafe_cannot_split_init),
            'definitive_fail': definitive_fail_new,
            'top_n_used': top_n_used,
            'repair_type': repair_type,
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
    print("修复前后对比 (v10_lbp, LBP验证 + LBP损失，无McCormick)")
    print(f"{'='*70}")
    print(f"指标                     原始          最终          变化")
    print(f"───────────────────────────────────────────────────────")
    print(f"HarmonicMeanPassRate:    {original_max_depth_harmonic:>8.2f}%   {final_harmonic:>8.2f}%   ({harmonic_improvement:+.2f}%)")
    print(f"standard_pass_rate:      {original_max_depth_standard:>8.2f}%   {final_standard:>8.2f}%   ({standard_improvement:+.2f}%)")
    print(f"R_safe:                 {original_max_depth_R_safe:>8.2f}%   {final_R_safe:>8.2f}%")
    print(f"R_unsafe:               {original_max_depth_R_unsafe:>8.2f}%   {final_R_unsafe:>8.2f}")

    import json
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_v10_lbp")
    os.makedirs(results_dir, exist_ok=True)

    run_result = {
        'system': system_name_key, 'activation': activation,
        'method': 'Harmonic Mean CBF Pass Rate (v10_lbp, LBP验证 + LBP损失，无McCormick)',
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

    result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_v10_lbp.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    main()