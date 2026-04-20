"""
Neural CBF 迭代修复对比实验版: LAST LAYER REPAIR (基于 main_clean_v10_ibp.py)

基于论文 "LAST LAYER REPAIR: A METAMATERIAL-AGNOSTIC METHOD FOR VERIFIABLE..."
的核心思想: 只修复神经网络最后一层的参数，前面的层全部冻结。

与 main_clean_v10_ibp.py 的区别:
    - 冻结网络前面所有层，只训练最后一层
    - 优化目标: min ||W - W_0||^2 subject to CBF 约束
    - 保持与 main_clean_v10_ibp.py 相同的验证流程 (LBP verify_cbf)

修复流程 (与 main_clean_v10_ibp.py 一致):
    Phase 1 -> Phase 2 渐进式深度分层修复
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
from lbp_neural_cbf.translators import TorchTranslator
from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization

# v10_ibp: 使用 LBP 验证
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from verify_cbf_ibp import compute_simplex_bound_batch_ibp, IBPNetworkBoundCalculator

# 使用与 main_compare_v8.py 相同的 Last Layer Repair 模块
from New_repair.geometry_module_new_v3 import extract_all_feature_points
from New_repair.geometry_module_new import compute_simplex_bound_batch


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
    使用 LBP (verify_cbf) 对模型进行验证
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


def extract_feature_points_from_regions(simplices_list, device, dtype=torch.float32):
    if not simplices_list:
        return torch.empty(0, 2, device=device, dtype=dtype)
    all_feature_points, _ = extract_all_feature_points(simplices_list, device=device, dtype=dtype)
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
        min_L_batch = compute_simplex_bound_batch(model, V_safe_batch, 'safe', dynamics_model=dynamics_model, translator=translator)
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
            return current_max_depth, 2, "PHATE2_ALL_CLEARED"
        if current_idx + 1 < len(depth_schedule):
            return depth_schedule[current_idx + 1], 2, f"PHASE2_DEPTH_INCREASE"
        return current_max_depth, 2, "PHASE2_MAX_DEPTH"


# =============================================================================
# LAST LAYER REPAIR: 核心修改部分 (来自论文 20260227-2403.07308v1.pdf)
# =============================================================================

def freeze_except_last_layer(model):
    """
    冻结网络所有层，只保留最后一层可训练。

    BarrierNN 的结构: nn.Sequential(input_layer, activation, hidden1, ..., last_linear)
    最后一层是 nn.Linear(out_features=1)
    """
    for name, param in model.named_parameters():
        if "network" in name:
            layer_idx = name.split('.')[1]
            # 检查是否是最后一层 (索引最大)
            if int(layer_idx) == len(model.network) - 1:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            param.requires_grad_(False)

    # 验证
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"    [Last Layer Repair] 可训练参数: {trainable_params}")
    return model


def get_last_layer_params(model):
    """获取最后一层的参数 (weight 和 bias)"""
    last_layer = model.network[-1]
    return last_layer.weight, last_layer.bias


def last_layer_gradient_update(model, grad_weight, grad_bias, lr):
    """
    只更新最后一层的权重和偏置。

    LAST LAYER REPAIR 的核心:
        W_new = W_old - lr * grad_W
        b_new = b_old - lr * grad_b
    """
    last_weight, last_bias = get_last_layer_params(model)

    with torch.no_grad():
        last_weight.sub_(lr * grad_weight)
        if grad_bias is not None and last_bias is not None:
            last_bias.sub_(lr * grad_bias)

    update_norm = (grad_weight * lr).norm().item()
    if grad_bias is not None:
        update_norm += (grad_bias * lr).norm().item()
    return update_norm


def compute_repair_loss_and_grad_last_layer(
    model: nn.Module,
    dynamics_model,
    failed_safe_feature_points: torch.Tensor,
    failed_unsafe_feature_points: torch.Tensor,
    W_orig: torch.Tensor,
    b_orig: torch.Tensor,
    margin: float = 0.0,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    alpha_reg: float = 1.0,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
    translator=None,
):
    """
    计算 LAST LAYER REPAIR 损失并获取最后一层的梯度。

    论文方法 (20260227-2403.07308v1.pdf):
        L_total = α * ||W - W_0||^2 + β * L_CBF

    其中 L_CBF 包含:
        - unsafe: softplus(h + margin)，h 应该 <= 0
        - safe: softplus(cbf_margin - cbf)，cbf 应该 >= cbf_margin

    CBF条件: h(x) + α·∇h·f(x) >= 0

    Returns:
        total_loss: 总损失 (包括正则项)
        cbf_loss: CBF约束损失
        reg_loss: 最小二乘正则损失
        g_last_weight: 最后一层 weight 的梯度
        g_last_bias: 最后一层 bias 的梯度
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    g_last_weight = torch.zeros_like(model.network[-1].weight)
    g_last_bias = torch.zeros_like(model.network[-1].bias) if model.network[-1].bias is not None else None

    cbf_loss_sum = 0.0
    n_valid = 0

    # ---------- 处理 unsafe（障碍区违规）----------
    # loss_unsafe = softplus(h + margin)，h 应该 <= 0
    if failed_unsafe_feature_points.shape[0] > 0:
        x_nograd = failed_unsafe_feature_points.detach().to(device)
        x_in = x_nograd.requires_grad_(True)

        h = model(x_in).squeeze(-1)
        loss_unsafe = nn.functional.softplus(h + margin, beta=beta)

        if not torch.isfinite(loss_unsafe).all():
            if verbose:
                print(f"  [警告] unsafe 损失存在 NaN/Inf，跳过")
        else:
            model.zero_grad()
            loss_unsafe.mean().backward(retain_graph=False)

            # 只取最后一层的梯度
            w_grad = model.network[-1].weight.grad
            b_grad = model.network[-1].bias.grad if model.network[-1].bias is not None else None

            if w_grad is not None:
                g_last_weight.add_(w_grad)
            if b_grad is not None and g_last_bias is not None:
                g_last_bias.add_(b_grad)

            g_last_weight = torch.nan_to_num(g_last_weight, nan=0.0, posinf=0.0, neginf=0.0)
            if g_last_bias is not None:
                g_last_bias = torch.nan_to_num(g_last_bias, nan=0.0, posinf=0.0, neginf=0.0)

            cbf_loss_sum += loss_unsafe.mean().item()
            n_valid += 1

    # ---------- 处理 safe（CBF 条件违规）----------
    # loss_cbf = softplus(cbf_margin - cbf)，cbf 应该 >= cbf_margin
    # CBF条件: h(x) + α·∇h·f(x) >= 0
    if failed_safe_feature_points.shape[0] > 0:
        x_nograd = failed_safe_feature_points.detach().to(device)
        x_in = x_nograd.requires_grad_(True)

        h = model(x_in).squeeze(-1)

        # 计算 ∇h (h 对输入 x 的梯度)
        grad_h = torch.autograd.grad(
            outputs=h, inputs=x_in,
            grad_outputs=torch.ones_like(h, device=device),
            create_graph=True, retain_graph=True,
        )[0]

        if translator is None:
            x1 = x_in[..., 0]
            x2 = x_in[..., 1]
            D = x_in.shape[1]
            if D == 2 and dynamics_model.control_dim == 0:
                dx1 = x2
                dx2 = -x1 - x2 + (1.0 / 3.0) * torch.pow(x1, 3)
                f_x = torch.stack([dx1, dx2], dim=-1)
            else:
                raise NotImplementedError("需要 translator")
        else:
            f_x = dynamics_model.compute_f(x_in, translator)

        # CBF 条件: grad_h · f + α·h >= 0
        grad_h_dot_f = (grad_h * f_x).sum(dim=-1)
        alpha_h = dynamics_model.alpha_function(h, translator)
        cbf = grad_h_dot_f + alpha_h

        loss_cbf = nn.functional.softplus(cbf_margin - cbf, beta=beta)

        if not torch.isfinite(loss_cbf).all():
            if verbose:
                print(f"  [警告] CBF 损失存在 NaN/Inf，跳过")
        else:
            model.zero_grad()
            loss_cbf.mean().backward(retain_graph=False)

            w_grad = model.network[-1].weight.grad
            b_grad = model.network[-1].bias.grad if model.network[-1].bias is not None else None

            if w_grad is not None:
                g_last_weight.add_(w_grad)
            if b_grad is not None and g_last_bias is not None:
                g_last_bias.add_(b_grad)

            g_last_weight = torch.nan_to_num(g_last_weight, nan=0.0, posinf=0.0, neginf=0.0)
            if g_last_bias is not None:
                g_last_bias = torch.nan_to_num(g_last_bias, nan=0.0, posinf=0.0, neginf=0.0)

            cbf_loss_sum += loss_cbf.mean().item()
            n_valid += 1

    if n_valid == 0:
        return 0.0, 0.0, 0.0, torch.zeros_like(model.network[-1].weight), torch.zeros_like(model.network[-1].bias) if model.network[-1].bias is not None else None

    # ---------- 计算最小二乘正则项的梯度 ----------
    cbf_loss = cbf_loss_sum / n_valid

    W_current = model.network[-1].weight
    b_current = model.network[-1].bias

    # d/dW ||W - W_0||^2 = 2 * (W - W_0)
    g_reg_weight = 2.0 * (W_current - W_orig)
    g_reg_bias = 2.0 * (b_current - b_orig) if b_current is not None and b_orig is not None else None

    # 总梯度 = α * g_reg + β * g_cbf
    g_last_weight_total = alpha_reg * g_reg_weight + beta * g_last_weight
    g_last_bias_total = None
    if g_last_bias is not None and g_reg_bias is not None:
        g_last_bias_total = alpha_reg * g_reg_bias + beta * g_last_bias

    # 梯度裁剪
    grad_norm = g_last_weight_total.norm().item()
    if g_last_bias_total is not None:
        grad_norm += g_last_bias_total.norm().item()

    if grad_norm > grad_clip_norm:
        scale = grad_clip_norm / grad_norm
        g_last_weight_total.mul_(scale)
        if g_last_bias_total is not None:
            g_last_bias_total.mul_(scale)
        grad_norm = grad_clip_norm

    if verbose:
        print(f"  [Last Layer Repair] cbf_loss={cbf_loss:.6f}, |g_cbf|={g_last_weight.norm().item():.4f}, "
              f"|g_total|={grad_norm:.4f}, unsafe_pts={failed_unsafe_feature_points.shape[0]}, "
              f"safe_pts={failed_safe_feature_points.shape[0]}")

    # 返回总损失 (用于记录，但梯度已通过 g_last_weight_total/bias_total 体现)
    total_loss = alpha_reg * ((W_current - W_orig).pow(2).sum().item() + (b_current - b_orig).pow(2).sum().item() if b_current is not None else 0.0) + beta * cbf_loss

    return total_loss, cbf_loss, g_last_weight_total, g_last_bias_total


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Neural CBF 迭代修复对比实验版: LAST LAYER REPAIR (基于 v10_ibp)')
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
    parser.add_argument('--alpha-reg', type=float, default=1.0, help='正则项系数 α')
    parser.add_argument('--beta-cbf', type=float, default=5.0, help='CBF损失系数 β')

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
    alpha_reg = args.alpha_reg
    beta_cbf = args.beta_cbf

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print(f"Neural CBF 迭代修复对比实验版: LAST LAYER REPAIR (基于 v10_ibp)")
    print(f"  激活={activation}, 系统={system_name_key}")
    print(f"  核心思想: 只修复最后一层，前面的层全部冻结")
    print(f"  正则项系数 α={alpha_reg}, CBF损失系数 β={beta_cbf}")
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

    # ---------- 关键修改: 冻结除最后一层外的所有层 ----------
    freeze_except_last_layer(model)

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

    print(f"\n[3.1] 原始区域 v10_ibp 指标: HarmonicMeanPassRate={original_max_depth_harmonic:.2f}%, R_safe={original_max_depth_R_safe:.2f}%, R_unsafe={original_max_depth_R_unsafe:.2f}%")

    # ========== 3.2 检查是否需要修复 ==========
    if original_max_depth_standard >= 99.9 and original_max_depth_harmonic >= 99.9:
        print(f"\n[3.2] 验证通过率已达 99.9%，无需修复！")
        print(f"    original_max_depth_standard: {original_max_depth_standard:.2f}%")
        print(f"    original_max_depth_harmonic: {original_max_depth_harmonic:.2f}%")
        import json
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_compare_v10")
        os.makedirs(results_dir, exist_ok=True)
        run_result = {
            'system': system_name_key, 'activation': activation,
            'method': 'LAST LAYER REPAIR (基于 v10_ibp)',
            'max_depth_start': max_depth_start, 'max_depth_limit': max_depth_limit,
            'depth_schedule': depth_schedule, 'num_inner_steps': num_inner_steps, 'lr': lr,
            'target_pass_rate': target_pass_rate, 'plateau_threshold': plateau_threshold,
            'max_stagnant_iterations': max_stagnant_iterations, 'max_total_iterations': max_total_iterations,
            'alpha_reg': alpha_reg, 'beta_cbf': beta_cbf,
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
        result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_compare_v10.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(run_result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_file}")
        print("=" * 70)
        print("无需修复，程序结束")
        print("=" * 70)
        return

    pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_compare_v10.pth"
    torch.save(model.state_dict(), pytorch_save_path)
    onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_compare_v10.onnx"
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

    # 保存原始最后一层权重（用于正则项）
    W_orig, b_orig = get_last_layer_params(model)

    print(f"\n开始渐进式深度分层修复 (LAST LAYER REPAIR 版), Phase {current_phase}, max_depth={current_max_depth}")

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
            # 选择 top_n 安全区域进行保护
            top_n_v_safe = select_top_n_v_safe(model, list(V_safe_init), dynamics_model, translator, top_n_protect)
            top_n_used = len(top_n_v_safe)

            # 从失败区域提取特征点 (counterexample 点)
            failed_safe_feature_points = extract_feature_points_from_regions(failed_safe_simplices, device)
            failed_unsafe_feature_points = extract_feature_points_from_regions(failed_unsafe_simplices, device)

            depth_lr_map = {10: 5e-3, 12: 1e-3, 15: 1e-3}
            current_lr = depth_lr_map.get(current_max_depth, lr)

            inner_history = []
            for inner_step in range(num_inner_steps):
                t0 = time.perf_counter()
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype

                # ---------- LAST LAYER REPAIR: 计算损失和梯度 ----------
                total_loss_sum = 0.0
                total_n = 0

                # 累加最后一层梯度
                g_last_weight_total = torch.zeros_like(model.network[-1].weight)
                g_last_bias_total = torch.zeros_like(model.network[-1].bias) if model.network[-1].bias is not None else None

                n_unsafe_fp = failed_unsafe_feature_points.shape[0]
                n_safe_fp = failed_safe_feature_points.shape[0]

                # ---------- 处理 unsafe 区域 (障碍区违规) ----------
                for unsafe_start in range(0, n_unsafe_fp, 1024):
                    unsafe_end = min(unsafe_start + 1024, n_unsafe_fp)
                    unsafe_chunk = failed_unsafe_feature_points[unsafe_start:unsafe_end]
                    chunk_loss, chunk_cbf_loss, chunk_g_w, chunk_g_b = compute_repair_loss_and_grad_last_layer(
                        model, dynamics_model, torch.empty(0, 2, device=device, dtype=dtype), unsafe_chunk,
                        W_orig, b_orig,
                        margin=0.1, cbf_margin=0.0, beta=beta_cbf, alpha_reg=alpha_reg,
                        grad_clip_norm=10.0, verbose=False, translator=translator,
                    )
                    total_loss_sum += chunk_loss * unsafe_chunk.shape[0]
                    total_n += unsafe_chunk.shape[0]
                    g_last_weight_total.add_(chunk_g_w)
                    if chunk_g_b is not None and g_last_bias_total is not None:
                        g_last_bias_total.add_(chunk_g_b)

                # ---------- 处理 safe 区域 (CBF 条件违规) ----------
                for safe_start in range(0, n_safe_fp, 1024):
                    safe_end = min(safe_start + 1024, n_safe_fp)
                    safe_chunk = failed_safe_feature_points[safe_start:safe_end]
                    chunk_loss, chunk_cbf_loss, chunk_g_w, chunk_g_b = compute_repair_loss_and_grad_last_layer(
                        model, dynamics_model, safe_chunk, torch.empty(0, 2, device=device, dtype=dtype),
                        W_orig, b_orig,
                        margin=0.1, cbf_margin=0.0, beta=beta_cbf, alpha_reg=alpha_reg,
                        grad_clip_norm=10.0, verbose=False, translator=translator,
                    )
                    total_loss_sum += chunk_loss * safe_chunk.shape[0]
                    total_n += safe_chunk.shape[0]
                    g_last_weight_total.add_(chunk_g_w)
                    if chunk_g_b is not None and g_last_bias_total is not None:
                        g_last_bias_total.add_(chunk_g_b)

                loss_val = total_loss_sum / total_n if total_n > 0 else 0.0
                grad_norm = g_last_weight_total.norm().item()
                if g_last_bias_total is not None:
                    grad_norm += g_last_bias_total.norm().item()

                # 梯度裁剪
                g_clip = 10.0
                if grad_norm > g_clip:
                    scale = g_clip / grad_norm
                    g_last_weight_total.mul_(scale)
                    if g_last_bias_total is not None:
                        g_last_bias_total.mul_(scale)
                    grad_norm = g_clip

                t1 = time.perf_counter()

                # ---------- LAST LAYER REPAIR: 只更新最后一层 ----------
                update_norm = last_layer_gradient_update(model, g_last_weight_total, g_last_bias_total, current_lr)
                t2 = time.perf_counter()

                inner_history.append({'step': inner_step + 1, 'loss': loss_val, 'g_raw_norm': grad_norm, 'update_norm': update_norm})
                print(f"    [内步 {inner_step+1}] loss={loss_val:.6f}, |g|={grad_norm:.4f}, |d|={update_norm:.6f}")

        pytorch_save_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_compare_v10.pth"
        torch.save(model.state_dict(), pytorch_save_path)
        onnx_path = f"New_repair/regions/{dynamics_model.system_name}_{activation}_cbf_repaired_compare_v10.onnx"
        pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

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

        verified_regions_path = f"New_repair/regions/verified_regions_{dynamics_model.system_name}_{activation}_repaired_compare_v10.pt"
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
    print("修复前后对比 (LAST LAYER REPAIR 对比实验版, 基于 v10_ibp)")
    print(f"{'='*70}")
    print(f"指标                     原始          最终          变化")
    print(f"───────────────────────────────────────────────────────")
    print(f"HarmonicMeanPassRate:    {original_max_depth_harmonic:>8.2f}%   {final_harmonic:>8.2f}%   ({harmonic_improvement:+.2f}%)")
    print(f"standard_pass_rate:      {original_max_depth_standard:>8.2f}%   {final_standard:>8.2f}%   ({standard_improvement:+.2f}%)")
    print(f"R_safe:                 {original_max_depth_R_safe:>8.2f}%   {final_R_safe:>8.2f}%")
    print(f"R_unsafe:               {original_max_depth_R_unsafe:>8.2f}%   {final_R_unsafe:>8.2f}")

    import json
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_results_compare_v10")
    os.makedirs(results_dir, exist_ok=True)

    run_result = {
        'system': system_name_key, 'activation': activation,
        'method': 'LAST LAYER REPAIR (基于 v10_ibp)',
        'top_n_protect': top_n_protect, 'max_depth_start': max_depth_start, 'max_depth_limit': max_depth_limit,
        'depth_schedule': depth_schedule, 'num_inner_steps': num_inner_steps, 'lr': lr,
        'target_pass_rate': target_pass_rate, 'plateau_threshold': plateau_threshold,
        'max_stagnant_iterations': max_stagnant_iterations, 'max_total_iterations': max_total_iterations,
        'alpha_reg': alpha_reg, 'beta_cbf': beta_cbf,
        'original_max_depth_harmonic': original_max_depth_harmonic,
        'original_max_depth_standard': original_max_depth_standard,
        'original_max_depth_R_safe': original_max_depth_R_safe, 'original_max_depth_R_unsafe': original_max_depth_R_unsafe,
        'final_harmonic_pass_rate': final_harmonic, 'final_standard_pass_rate': final_standard,
        'final_R_safe': final_R_safe, 'final_R_unsafe': final_R_unsafe,
        'harmonic_improvement': harmonic_improvement, 'standard_improvement': standard_improvement,
        'num_iterations': len(iteration_results), 'iteration_results': iteration_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    result_file = os.path.join(results_dir, f"result_{system_name_key}_{activation}_compare_v10.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    main()