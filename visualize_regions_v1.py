"""
可视化验证区域：绘制 V_safe, V_unsafe, F_h_positive_in_unsafe, F_h_positive_unsafe_contained,
F_safe_cbf_violation, F_depth_limit_reached, F_unsafe_cannot_split 等七类区域

这个脚本从 verified_regions_{system_name}_{activation}.pt 加载验证结果，
并绘制所有七类区域的单纯形分布。

区域类型说明：
- V_safe: 安全区中验证通过的单纯形 (SAT, safe_cbf_verified)
- V_unsafe: 障碍区中验证通过的单纯形 (SAT, unsafe_region)
- F_h_positive_in_unsafe: 与unsafe相交(不在内部)+h_min>=0的违规 (UNSAT)
- F_h_positive_unsafe_contained: 完全在unsafe内部+h_max>=0的违规 (UNSAT)
- F_safe_cbf_violation: 安全区内 CBF 条件违规 (UNSAT)
- F_depth_limit_reached: 达到最大分裂深度 (UNSAT)
- F_unsafe_cannot_split: 障碍区无法继续细分 (UNSAT)
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
from pathlib import Path
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lbp_neural_cbf.cbf.fossil_dynamics import (
    Barrier1System, Barrier2System, Barrier3System, Barrier4System
)

from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem


# 支持的系统映射
SYSTEMS = {
    'barr1': ('Barrier1', 'Barrier1'),
    'barr2': ('Barrier2', 'Barrier2'),
    'barr3': ('Barrier3', 'Barrier3'),
    'simple2d': ('simple_2d', 'simple_2d')
}


# 支持的激活函数
ACTIVATIONS = ['Tanh', 'Relu', 'Sigmoid']


def load_regions(regions_path, device='cpu'):
    """
    加载验证区域数据。

    Args:
        regions_path: .pt 文件路径
        device: 加载设备

    Returns:
        tuple: (dict: 包含所有区域类型的字典, certified_percentage: float 或 None)
    """
    if not os.path.exists(regions_path):
        raise FileNotFoundError(f"Regions file not found: {regions_path}")

    data = torch.load(regions_path, map_location=device, weights_only=False)

    regions = {
        'V_safe': [],
        'V_unsafe': [],
        'F_h_positive_in_unsafe': [],
        'F_h_positive_unsafe_contained': [],
        'F_safe_cbf_violation': [],
        'F_depth_limit_reached': [],
        'F_unsafe_cannot_split': []
    }

    for key in regions.keys():
        if key in data and data[key] is not None:
            regions[key] = data[key]

    # 同时兼容旧的 F_safe 和 F_unsafe 字段名
    if 'F_safe' in data and data['F_safe'] is not None:
        regions['F_safe_cbf_violation'].extend(data['F_safe'])
    if 'F_unsafe' in data and data['F_unsafe'] is not None:
        regions['F_unsafe_cannot_split'].extend(data['F_unsafe'])

    # 提取验证通过率（来自 verify_cbf 直接保存的 certified_percentage）
    certified_percentage = data.get('Certified percentage', None)
    if certified_percentage is None:
        certified_percentage = data.get('certified_percentage', None)

    return regions, certified_percentage


def convert_vertices(vertices):
    """
    将顶点转换为 numpy 数组。

    Args:
        vertices: torch.Tensor 或 np.ndarray

    Returns:
        np.ndarray: 形状为 [num_vertices, 2] 的数组
    """
    if isinstance(vertices, torch.Tensor):
        v = vertices.detach().cpu().numpy()
    elif isinstance(vertices, np.ndarray):
        v = vertices
    else:
        raise TypeError(f"Unsupported vertices type: {type(vertices)}")

    # 确保是 2D 数组
    if v.ndim == 1:
        v = v.reshape(1, -1)

    return v


def draw_simplex(vertices, ax, facecolor, edgecolor, alpha=0.5, linewidth=0.5):
    """
    绘制单个单纯形。

    Args:
        vertices: 顶点数组，形状 [V, 2]
        ax: matplotlib axes
        facecolor: 填充颜色
        edgecolor: 边框颜色
        alpha: 透明度
        linewidth: 边框宽度
    """
    v = convert_vertices(vertices)
    polygon = Polygon(v, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                     alpha=alpha, linewidth=linewidth)
    ax.add_patch(polygon)


def plot_verification_regions(
    regions,
    dynamics_model,
    title=None,
    figsize=(12, 10),
    alpha=0.5,
    output_path=None,
    show_stats=True,
    plot_type='all',
    max_regions=None,
    activation_fnc='Unknown',
    certified_percentage=None,
):
    """
    绘制验证区域。

    Args:
        regions: 包含各类区域的字典
        dynamics_model: 动力学系统（用于获取坐标范围）
        title: 图表标题
        figsize: 图像大小
        alpha: 区域透明度
        output_path: 保存路径（可选）
        show_stats: 是否显示统计信息
        plot_type: 绘图类型 ('all', 'verified', 'failed', 'failed_only')
        max_regions: 每个区域最大绘制数量（用于加速）
        activation_fnc: 激活函数名称
        certified_percentage: 验证通过率（来自 verify_cbf 直接保存的值）
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 获取坐标范围
    input_domain = dynamics_model.input_domain
    if hasattr(input_domain, 'bounds'):
        bounds = input_domain.bounds
    else:
        bounds = input_domain

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # 设置坐标轴
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # 设置标题
    if title is None:
        title = f"Verification Regions: {dynamics_model.system_name} ({activation_fnc})"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 统计信息
    stats = {
        'V_safe': len(regions['V_safe']),
        'V_unsafe': len(regions['V_unsafe']),
        'F_h_positive_in_unsafe': len(regions['F_h_positive_in_unsafe']),
        'F_h_positive_unsafe_contained': len(regions['F_h_positive_unsafe_contained']),
        'F_safe_cbf_violation': len(regions['F_safe_cbf_violation']),
        'F_depth_limit_reached': len(regions['F_depth_limit_reached']),
        'F_unsafe_cannot_split': len(regions['F_unsafe_cannot_split'])
    }

    # 计算总计
    total_verified = stats['V_safe'] + stats['V_unsafe']
    total_failed = sum([
        stats['F_h_positive_in_unsafe'],
        stats['F_h_positive_unsafe_contained'],
        stats['F_safe_cbf_violation'],
        stats['F_depth_limit_reached'],
        stats['F_unsafe_cannot_split']
    ])
    total = total_verified + total_failed
    # 直接使用 verify_cbf 保存的 certified_percentage
    pass_rate = certified_percentage if certified_percentage is not None else 0

    if show_stats:
        stats_text = (
            f"V_safe (Verified Safe): {stats['V_safe']}\n"
            f"V_unsafe (Verified Unsafe): {stats['V_unsafe']}\n"
            f"F_h_pos_in_unsafe: {stats['F_h_positive_in_unsafe']}\n"
            f"F_h_pos_unsafe_contained: {stats['F_h_positive_unsafe_contained']}\n"
            f"F_safe_cbf_violation: {stats['F_safe_cbf_violation']}\n"
            f"F_depth_limit_reached: {stats['F_depth_limit_reached']}\n"
            f"F_unsafe_cannot_split: {stats['F_unsafe_cannot_split']}\n"
            f"{'='*20}\n"
            f"Verified: {total_verified} ({pass_rate:.2f}%)\n"
            f"Failed: {total_failed}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 定义颜色
    colors = {
        'V_safe': ('green', 'darkgreen'),
        'V_unsafe': ('blue', 'darkblue'),
        'F_h_positive_in_unsafe': ('purple', 'indigo'),
        'F_h_positive_unsafe_contained': ('magenta', 'darkmagenta'),
        'F_safe_cbf_violation': ('red', 'darkred'),
        'F_depth_limit_reached': ('orange', 'darkorange'),
        'F_unsafe_cannot_split': ('brown', 'saddlebrown')
    }

    # 绘制区域
    def draw_regions(key):
        fc, ec = colors[key]
        count = 0
        for vertices in regions[key]:
            if max_regions is not None and count >= max_regions:
                break
            try:
                draw_simplex(vertices, ax, fc, ec, alpha=alpha)
                count += 1
            except Exception as e:
                print(f"Warning: Failed to draw {key}: {e}")
        if count > 0:
            print(f"  Drew {count}/{len(regions[key])} {key} regions")
        return count

    if plot_type == 'all':
        for key in colors.keys():
            draw_regions(key)
    elif plot_type == 'verified':
        draw_regions('V_safe')
        draw_regions('V_unsafe')
    elif plot_type == 'failed':
        for key in ['F_h_positive_in_unsafe', 'F_h_positive_unsafe_contained',
                    'F_safe_cbf_violation', 'F_depth_limit_reached', 'F_unsafe_cannot_split']:
            draw_regions(key)
    elif plot_type == 'failed_only':
        # 只绘制 F_safe_cbf_violation 和 F_h_positive_in_unsafe 和 F_h_positive_unsafe_contained
        draw_regions('F_h_positive_in_unsafe')
        draw_regions('F_h_positive_unsafe_contained')
        draw_regions('F_safe_cbf_violation')

    # 添加图例
    legend_elements = [
        Patch(facecolor=colors['V_safe'][0], edgecolor=colors['V_safe'][1], alpha=alpha,
              label=f'V_safe (n={stats["V_safe"]})'),
        Patch(facecolor=colors['V_unsafe'][0], edgecolor=colors['V_unsafe'][1], alpha=alpha,
              label=f'V_unsafe (n={stats["V_unsafe"]})'),
        Patch(facecolor=colors['F_h_positive_in_unsafe'][0], edgecolor=colors['F_h_positive_in_unsafe'][1], alpha=alpha,
              label=f'F_h_pos_in_unsafe (n={stats["F_h_positive_in_unsafe"]})'),
        Patch(facecolor=colors['F_h_positive_unsafe_contained'][0], edgecolor=colors['F_h_positive_unsafe_contained'][1], alpha=alpha,
              label=f'F_h_pos_unsafe_cont (n={stats["F_h_positive_unsafe_contained"]})'),
        Patch(facecolor=colors['F_safe_cbf_violation'][0], edgecolor=colors['F_safe_cbf_violation'][1], alpha=alpha,
              label=f'F_safe_cbf_violation (n={stats["F_safe_cbf_violation"]})'),
        Patch(facecolor=colors['F_depth_limit_reached'][0], edgecolor=colors['F_depth_limit_reached'][1], alpha=alpha,
              label=f'F_depth_limit (n={stats["F_depth_limit_reached"]})'),
        Patch(facecolor=colors['F_unsafe_cannot_split'][0], edgecolor=colors['F_unsafe_cannot_split'][1], alpha=alpha,
              label=f'F_unsafe_cannot_split (n={stats["F_unsafe_cannot_split"]})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()

    # 保存
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig, ax, stats


def plot_iteration_comparison(
    system_name,
    regions_dir='New_repair/regions',
    output_dir='results',
    iterations=None,
    activation=None
):
    """
    绘制多次迭代的验证区域对比图。

    Args:
        system_name: 系统名称（如 'barr1', 'barr3'）
        regions_dir: 验证结果目录
        output_dir: 输出目录
        iterations: 要绘制的迭代次数列表（None 表示所有）
        activation: 激活函数名称（如 'Tanh', 'Relu'）
    """
    # 动力学系统映射
    dynamics_map = {
        'barr1': Barrier1System,
        'barr2': Barrier2System,
        'barr3': Barrier3System,
        'simple2d': Simple2DSystem
    }
    dynamics_class = dynamics_map.get(system_name, Barrier3System)
    dynamics = dynamics_class(alpha=1.0)

    # 查找所有迭代结果文件
    iter_files = []
    base_path = Path(regions_dir)

    for f in sorted(base_path.glob(f'verified_regions_{system_name}_*.pt')):
        fname = f.name
        # 跳过带 iter 的文件（这些是修复后的）
        if '_iter' in fname:
            continue
        # 如果指定了激活函数，匹配
        if activation and f'{activation}.pt' in fname:
            iter_files.append((0, f))
        elif activation is None:
            iter_files.append((0, f))

    # 查找带 iter 的文件
    iter_pattern = f'verified_regions_{system_name}_*_iter*.pt'
    for f in sorted(base_path.glob(iter_pattern)):
        fname = f.name
        # 提取迭代编号
        try:
            if '_iter' in fname:
                iter_num = int(fname.split('_iter')[1].split('.')[0])
                if activation is None or f'_{activation}_iter' in fname or f'_{activation}.' in fname:
                    iter_files.append((iter_num, f))
        except:
            pass

    iter_files.sort(key=lambda x: (x[0] != 0, x[0]))

    if not iter_files:
        print(f"No files found for {system_name}" + (f" ({activation})" if activation else ""))
        return None

    print(f"\nFound {len(iter_files)} files for {system_name}" +
          (f" ({activation})" if activation else ""))

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 为每个文件绘制单独的图
    all_stats = []
    for label, iter_file in iter_files:
        print(f"\nProcessing: {iter_file}")

        regions, certified_percentage = load_regions(str(iter_file))

        # 获取文件名中的激活函数信息
        fname = iter_file.name
        if '_Tanh' in fname:
            act = 'Tanh'
        elif '_Relu' in fname:
            act = 'Relu'
        elif '_Sigmoid' in fname:
            act = 'Sigmoid'
        else:
            act = activation if activation else 'Unknown'

        output_file = output_path / f'regions_{system_name}_{act}_{label}.png'

        fig, ax, stats = plot_verification_regions(
            regions=regions,
            dynamics_model=dynamics,
            title=f'{dynamics.system_name} - {act} - {label}',
            figsize=(10, 8),
            output_path=str(output_file),
            show_stats=True,
            plot_type='all',
            activation_fnc=act,
            certified_percentage=certified_percentage,
        )
        plt.close(fig)

        all_stats.append((label, stats, act))

    return all_stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='可视化 CBF 验证区域 (V_safe, V_unsafe, F_h_positive_in_unsafe, '
                    'F_h_positive_unsafe_contained, F_safe_cbf_violation, '
                    'F_depth_limit_reached, F_unsafe_cannot_split)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        default=None,
        help='指定 .pt 文件的完整路径（优先于 --system 和 --activation）'
    )

    parser.add_argument(
        '--system', '-s',
        type=str,
        default='barr3',
        choices=list(SYSTEMS.keys()),
        help='系统名称 (default: barr3)'
    )

    parser.add_argument(
        '--activation', '-a',
        type=str,
        default=None,
        choices=ACTIVATIONS,
        help='激活函数名称 (Tanh, Relu, Sigmoid)'
    )

    parser.add_argument(
        '--name_out', '-n',
        type=str,
        default=None,
        help='具体保存的名字'
    )

    parser.add_argument(
        '--regions-dir', '-d',
        type=str,
        default='New_repair/regions/figures',
        help='验证结果目录 (default: New_repair/regions/figures)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/New_repair/figures',
    )

    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='绘制所有迭代的对比图'
    )

    parser.add_argument(
        '--iterations', '-i',
        nargs='+',
        type=int,
        help='指定要绘制的迭代次数（如: --iterations 1 3 5）'
    )

    parser.add_argument(
        '--plot-type', '-t',
        type=str,
        default='all',
        choices=['all', 'verified', 'failed', 'failed_only'],
        help='绘图类型: all(全部), verified(只画验证通过), failed(只画失败), failed_only(只画CBF违规)'
    )

    parser.add_argument(
        '--failed-only', '-f',
        action='store_true',
        help='只绘制失败区域（F_safe_cbf_violation, F_h_positive_in_unsafe, F_h_positive_unsafe_contained）'
    )

    parser.add_argument(
        '--max-regions', '-m',
        type=int,
        default=None,
        help='每个区域最大绘制数量（用于加速，默认全部绘制）'
    )

    args = parser.parse_args()

    # 加载动力学系统
    dynamics_map = {
        'barr1': Barrier1System,
        'barr2': Barrier2System,
        'barr3': Barrier3System,
        'simple2d': Simple2DSystem,
    }
    dynamics_class = dynamics_map[args.system]
    dynamics = dynamics_class(alpha=1.0)

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"可视化验证区域: {dynamics.system_name}" +
          (f" ({args.activation})" if args.activation else ""))
    print("=" * 60)

    if args.compare or args.iterations:
        # 绘制迭代对比
        stats = plot_iteration_comparison(
            system_name=args.system,
            regions_dir=args.regions_dir,
            output_dir=args.output_dir,
            iterations=args.iterations,
            activation=args.activation
        )

        if stats:
            print("\n统计汇总:")
            print(f"{'Label':<10} {'V_safe':<10} {'V_unsafe':<10} "
                  f"{'F_h+':<10} {'F_h+_cont':<10} {'F_safe':<10} {'F_depth':<10} "
                  f"{'F_split':<10} {'通过率':<10}")
            print("-" * 100)
            for label, s, act in stats:
                total_verified = s['V_safe'] + s['V_unsafe']
                total = sum(s.values())
                pass_rate = 100 * total_verified / total if total > 0 else 0
                print(f"{label:<10} {s['V_safe']:<10} {s['V_unsafe']:<10} "
                      f"{s['F_h_positive_in_unsafe']:<10} {s['F_h_positive_unsafe_contained']:<10} "
                      f"{s['F_safe_cbf_violation']:<10} "
                      f"{s['F_depth_limit_reached']:<10} {s['F_unsafe_cannot_split']:<10} "
                      f"{pass_rate:<10.2f}%")

    elif args.path:
        # 使用指定的完整路径
        regions_file = Path(args.path)
        print(f"\n加载验证区域文件: {regions_file}")

        if not regions_file.exists():
            print(f"文件不存在: {regions_file}")
            return

        regions, certified_percentage = load_regions(str(regions_file))

        # 从文件名提取激活函数
        fname = regions_file.name
        if '_Tanh' in fname:
            act = 'Tanh'
        elif '_Relu' in fname:
            act = 'Relu'
        elif '_Sigmoid' in fname:
            act = 'Sigmoid'
        else:
            act = args.activation if args.activation else 'Unknown'

        # output_file = output_path / f'regions_{args.system}_{act}.png'
        # output_file = output_path / f'regions_{args.system}_{act}_repaired.png'
        output_file = output_path / f'regions_{args.system}_{act}_{args.name_out}.png'

        print(f"输出路径: {output_file}")

        plot_type = 'failed_only' if args.failed_only else args.plot_type

        fig, ax, stats = plot_verification_regions(
            regions=regions,
            dynamics_model=dynamics,
            title=f'{dynamics.system_name} - {act} - Verification Regions',
            figsize=(10, 8),
            output_path=str(output_file),
            show_stats=True,
            plot_type=plot_type,
            max_regions=args.max_regions,
            activation_fnc=act,
            certified_percentage=certified_percentage,
        )

        total_verified = stats['V_safe'] + stats['V_unsafe']
        total_failed = sum([
            stats['F_h_positive_in_unsafe'],
            stats['F_h_positive_unsafe_contained'],
            stats['F_safe_cbf_violation'],
            stats['F_depth_limit_reached'],
            stats['F_unsafe_cannot_split']
        ])
        total = total_verified + total_failed

        print(f"\n统计信息:")
        print(f"  V_safe (验证通过-安全区): {stats['V_safe']}")
        print(f"  V_unsafe (验证通过-障碍区): {stats['V_unsafe']}")
        print(f"  F_h_positive_in_unsafe (与unsafe相交 h>=0): {stats['F_h_positive_in_unsafe']}")
        print(f"  F_h_positive_unsafe_contained (完全在unsafe内 h>=0): {stats['F_h_positive_unsafe_contained']}")
        print(f"  F_safe_cbf_violation (安全区CBF违规): {stats['F_safe_cbf_violation']}")
        print(f"  F_depth_limit_reached (达到深度限制): {stats['F_depth_limit_reached']}")
        print(f"  F_unsafe_cannot_split (障碍区无法细分): {stats['F_unsafe_cannot_split']}")
        print(f"  总区域数: {total}")
        print(f"  验证通过率: {certified_percentage:.4f}%" if certified_percentage is not None else "  验证通过率: N/A")

        plt.close(fig)
    else:
        print("请提供 --path 或 --compare 参数来指定要绘制的验证区域文件或迭代对比。")
        parser.print_help()

    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
