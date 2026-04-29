"""
使用 LBP 方法验证 main_clean_v9_ibp.py 保存的修复模型
对比 IBP 验证和 LBP 验证的通过率差异
"""

import torch
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from lbp_neural_cbf.cbf.fossil_dynamics import (
    Barrier1System, Barrier2System, Barrier3System, Barrier4System
)
from lbp_neural_cbf.cbf.network import BarrierNN

# 支持的动力学系统
DYNAMICS_SYSTEMS = {
    'barr1': Barrier1System,
    'barr2': Barrier2System,
    'barr3': Barrier3System,
    'barr4': Barrier4System,
}

SUPPORTED_ACTIVATIONS = ['Relu', 'Tanh', 'Sigmoid']


def pytorch_to_onnx(model, onnx_path, input_dim=2):
    """将 PyTorch 模型导出为 ONNX"""
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


def verify_model_lbp(onnx_path, dynamics_model, max_depth=13):
    """使用 LBP 方法验证模型（需要 ONNX 路径）"""
    results = verify_cbf(
        dynamics_model,
        onnx_path,
        visualize=False,
        use_gpu=False,
        batch_size=512,
        executor_type="single",
        region_type="simplicial",
        max_depth=max_depth,
    )
    return results


def compute_simplex_volume(simplex):
    """计算单纯形体积"""
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


def compute_safety_metrics(V_safe, V_unsafe, F_h_positive_in_unsafe, F_safe_cbf_violation,
                           F_depth_limit_reached_unsafe, F_depth_limit_reached_safe, F_unsafe_cannot_split):
    """计算安全指标"""
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

    return {
        'HarmonicMeanPassRate': HarmonicMeanPassRate * 100,
        'standard_pass_rate': standard_pass_rate,
        'R_safe': R_safe * 100,
        'R_unsafe': R_unsafe * 100,
        'volumes': {
            'V_safe': volume_v_safe,
            'V_unsafe': volume_v_unsafe,
            'F_h': volume_f_h,
            'F_safe_violation': volume_f_safe_violation,
            'F_depth_unsafe': volume_f_depth_unsafe,
            'F_depth_safe': volume_f_depth_safe,
            'F_unsafe_split': volume_f_unsafe_split,
        }
    }


def verify_and_compare(activation, system_name, max_depth=13):
    """验证单个模型的 IBP vs LBP 通过率"""
    print("=" * 70)
    print(f"验证: {system_name} + {activation}")
    print("=" * 70)

    # 加载动力学模型获取网络结构信息
    dynamics_class = DYNAMICS_SYSTEMS[system_name]
    dynamics_model = dynamics_class(alpha=1.0)
    dynamics_model.activation_fnc = activation

    # 修复后模型路径
    model_path = f"New_repair/regions/{system_name}_{activation}_cbf_repaired_v9_ibp.pth"
    onnx_path = f"New_repair/regions/{system_name}_{activation}_cbf_repaired_v9_ibp.onnx"

    if not os.path.exists(model_path):
        print(f"  模型不存在: {model_path}")
        return None

    print(f"\n加载模型: {model_path}")
    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=torch.device('cpu'),
        activation_fnc=activation
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 导出 ONNX（LBP 验证需要 ONNX 路径）
    print(f"导出 ONNX: {onnx_path}")
    pytorch_to_onnx(model, onnx_path, input_dim=dynamics_model.input_dim)

    # 使用 LBP 验证
    print(f"\n使用 LBP 方法验证 (max_depth={max_depth})...")
    results = verify_model_lbp(onnx_path, dynamics_model, max_depth=max_depth)

    # 计算指标
    metrics = compute_safety_metrics(
        results.get('V_safe', []),
        results.get('V_unsafe', []),
        results.get('F_h_positive_in_unsafe', []),
        results.get('F_safe_cbf_violation', []),
        results.get('F_depth_limit_reached_unsafe', []),
        results.get('F_depth_limit_reached_safe', []),
        results.get('F_unsafe_cannot_split', []),
    )

    print(f"\nLBP 验证结果:")
    print(f"  HarmonicMeanPassRate: {metrics['HarmonicMeanPassRate']:.2f}%")
    print(f"  standard_pass_rate:  {metrics['standard_pass_rate']:.2f}%")
    print(f"  R_safe:              {metrics['R_safe']:.2f}%")
    print(f"  R_unsafe:            {metrics['R_unsafe']:.2f}%")

    # 保存区域信息到 New_repair/nr_results_verify_lbp
    regions_dir = f"New_repair/nr_results_verify_lbp"
    os.makedirs(regions_dir, exist_ok=True)
    regions_file = os.path.join(regions_dir, f"lbp_regions_{system_name}_{activation}_v9_ibp_maxdepth{max_depth}.pt")

    regions_data = {
        'V_safe': results.get('V_safe', []),
        'V_unsafe': results.get('V_unsafe', []),
        'F_h_positive_in_unsafe': results.get('F_h_positive_in_unsafe', []),
        'F_safe_cbf_violation': results.get('F_safe_cbf_violation', []),
        'F_depth_limit_reached_unsafe': results.get('F_depth_limit_reached_unsafe', []),
        'F_depth_limit_reached_safe': results.get('F_depth_limit_reached_safe', []),
        'F_unsafe_cannot_split': results.get('F_unsafe_cannot_split', []),
        'metrics': metrics,
        'system': system_name,
        'activation': activation,
        'max_depth': max_depth,
    }
    torch.save(regions_data, regions_file)
    print(f"  区域信息已保存: {regions_file}")

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='使用 LBP 验证 main_clean_v9_ibp.py 保存的模型')
    parser.add_argument('--activation', '-a', type=str, choices=SUPPORTED_ACTIVATIONS,
                        help='激活函数 (Relu/Tanh/Sigmoid)，不指定则验证所有')
    parser.add_argument('--system', '-s', type=str, choices=list(DYNAMICS_SYSTEMS.keys()),
                        help='动力学系统 (barr1/barr2/barr3/barr4)，不指定则验证所有')
    parser.add_argument('--max-depth', type=int, default=15, help='最大验证深度 (默认13)')

    args = parser.parse_args()

    activations = [args.activation] if args.activation else SUPPORTED_ACTIVATIONS
    systems = [args.system] if args.system else list(DYNAMICS_SYSTEMS.keys())
    max_depth = args.max_depth

    results_all = []

    for activation in activations:
        for system_name in systems:
            result = verify_and_compare(activation, system_name, max_depth)
            if result:
                results_all.append({
                    'activation': activation,
                    'system': system_name,
                    'max_depth': max_depth,
                    **result
                })

    # 打印汇总表格
    print("\n" + "=" * 70)
    print("汇总结果: IBP 修复模型用 LBP 验证")
    print("=" * 70)
    print(f"{'System':<10} {'Activation':<10} {'HarmonicPassRate':<18} {'StandardPassRate':<18}")
    print("-" * 70)
    for r in results_all:
        print(f"{r['system']:<10} {r['activation']:<10} {r['HarmonicMeanPassRate']:>14.2f}%      {r['standard_pass_rate']:>14.2f}%")

    # 保存结果到 JSON
    import json
    results_dir = "New_repair/nr_results_verify_lbp"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(results_dir, f"lbp_verification_results_{timestamp}.json")

    save_data = {
        'timestamp': timestamp,
        'max_depth': max_depth,
        'results': results_all,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    main()