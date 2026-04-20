"""
Visualize CBF neural network function values for 2D systems.

This script loads trained CBF models from mine_models and mine_models_relu folders
and creates contour plots of the barrier function h(x) and CBF condition.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem, StateDependentControl2DSystem
from lbp_neural_cbf.cbf.fossil_dynamics import (
    Barrier1System, Barrier2System, Barrier3System, Barrier4System,
    HighOrd2System
)
from lbp_neural_cbf.visualization.cbf_plotter import CBFVerificationPlotter


class SimpleCBFNet(nn.Module):
    """Simple CBF network matching the saved model architecture."""
    def __init__(self, input_dim=2, hidden_sizes=[128, 256, 128], activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_model(model_path, dynamics, activation="tanh"):
    """Load a CBF model from .pth file.

    Args:
        model_path: Path to the .pth model file
        dynamics: Dynamics model (used for architecture inference)
        activation: Activation function ('relu' or 'tanh'), default 'tanh'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # First load the state dict to determine architecture
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # Infer architecture from state dict keys like 'network.0.weight', 'network.2.weight', etc.
    # Each Linear layer's weight shape is [output_dim, input_dim]
    # Sequential layers are named network.0, network.2, network.4, ... (with ReLU/Tanh at .1, .3, .5)
    hidden_sizes = []
    input_dim = None
    prev_out_dim = None

    for key in sorted(state_dict.keys()):
        if key.endswith('.weight') and key.startswith('network.'):
            weight_shape = state_dict[key].shape
            out_dim = weight_shape[0]
            in_dim = weight_shape[1]

            if prev_out_dim is None:
                # First layer: input_dim -> out_dim
                input_dim = in_dim
                prev_out_dim = out_dim
            else:
                if out_dim == 1:
                    # Output layer: prev_out_dim is the last hidden layer size
                    hidden_sizes.append(prev_out_dim)
                    break
                else:
                    # Hidden layer: prev_out_dim -> out_dim
                    hidden_sizes.append(prev_out_dim)
                    prev_out_dim = out_dim

    print(f"  Using activation function: {activation}")

    model = SimpleCBFNet(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        activation=activation
    ).to(device)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    return model, device


def visualize_cbf(model_path, dynamics, output_path, alpha=1.0, resolution=300, activation="tanh"):
    """
    Create CBF visualization for a 2D system.

    Args:
        model_path: Path to the .pth model file
        dynamics: Dynamics model instance
        output_path: Path to save the visualization
        alpha: CBF class-K parameter
        resolution: Grid resolution
        activation: Activation function ('relu' or 'tanh')
    """
    try:
        # Load model
        model, device = load_model(model_path, dynamics, activation)

        # Create plotter
        plotter = CBFVerificationPlotter(
            dynamics_model=dynamics,
            barrier_net=model,
            resolution=resolution,
            alpha=alpha,
            figsize=(18, 5)
        )

        # Save plot
        plotter.save_final_plot(output_path)
        print(f"  Saved: {output_path}")

        # Close figure to free memory
        plt.close(plotter.fig)

        return True
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_available_systems():
    """Return available 2D systems and their configurations."""
    return {
        'barr1': {
            'dynamics': lambda: Barrier1System(alpha=1.0),
            'alpha': 1.0,
        },
        'barr2': {
            'dynamics': lambda: Barrier2System(alpha=1.0),
            'alpha': 1.0,
        },
        'barr3': {
            'dynamics': lambda: Barrier3System(alpha=1.0),
            'alpha': 1.0,
        },
        'barr4': {
            'dynamics': lambda: Barrier4System(alpha=1.0),
            'alpha': 1.0,
        },
        'simple_2d': {
            'dynamics': lambda: Simple2DSystem(),
            'alpha': 1.0,
        },
        'hiord2': {
            'dynamics': lambda: HighOrd2System(alpha=1.0),
            'alpha': 1.0,
        },
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize CBF neural network function values for 2D systems.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all models from author_models and mine_models_relu
  python visualize_cbf_2d.py

  # Visualize specific systems only
  python visualize_cbf_2d.py --systems barr1 barr2 simple_2d

  # Visualize from specific directories
  python visualize_cbf_2d.py --dirs data/author_models data/my_models

  # Visualize single model with custom output
  python visualize_cbf_2d.py --dirs data/author_models --systems barr1 --output my_results

  # List all available systems
  python visualize_cbf_2d.py --list
        """
    )

    parser.add_argument(
        '--dirs', '-d',
        nargs='+',
        default=['data/author_models', 'data/mine_models_relu'],
        help='Model directories to process (default: data/author_models data/mine_models_relu)'
    )

    parser.add_argument(
        '--systems', '-s',
        nargs='+',
        default=None,
        help='Specific systems to visualize (default: all available)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for visualizations (default: results)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available systems and exit'
    )

    return parser.parse_args()


def main():
    """Main function to visualize 2D CBF models."""
    # ============================================================
    # 手动配置区域 - 修改这里的值来指定要画的路线和系统
    # ============================================================

    # 指定要可视化的模型目录（路线）
    # 格式: [(目录路径, 显示名称), ...]
    # 例如: [('data/author_models', 'author'), ('data/mine_models_relu', 'relu')]
    manual_dirs = [
        # ('data/author_models', 'author'),      # 作者提供的模型
        # ('data/mine_models_relu', 'relu1'),   # 取消注释以添加更多目录
        ('New_repair/regions', '_cbf_repaired_v9_ibp'),     # ReLU激活函数训练的模型
        # ('data/mine2_models_relu', 'relu2'),   # 取消注释以添加更多目录
    ]

    # 
    

    # 指定要可视化的系统
    # 可选值: 'barr1', 'barr2', 'barr3', 'barr4', 'simple_2d', 'hiord2'
    # 设为 None 表示使用默认（排除 barr4 和 hiord2）
    manual_systems = ['barr1']  # 手动指定系统列表
    # manual_systems = None  # 取消注释此行使用默认设置

    activation = "tanh"  # 激活函数类型（'relu' 或 'tanh'）



    # 输出目录
    output_dir_path = 'results'
    # ============================================================

    # 获取所有可用系统
    all_systems = get_available_systems()

    # 处理系统选择
    if manual_systems is not None:
        # 使用手动指定的系统
        systems_2d = {k: v for k, v in all_systems.items() if k in manual_systems}
        invalid = set(manual_systems) - set(all_systems.keys())
        if invalid:
            print(f"Warning: Invalid system names: {', '.join(invalid)}")
    else:
        # 默认: 排除 barr4 和 hiord2
        systems_2d = {k: v for k, v in all_systems.items()
                      if k not in ['barr4', 'hiord2']}

    if not systems_2d:
        print("Error: No valid systems to visualize")
        return 0, 0

    # 构建模型目录列表
    model_dirs = []
    for d, variant in manual_dirs:
        path = Path(d)
        if path.exists():
            model_dirs.append((d, variant))
        else:
            print(f"Warning: Directory not found: {d}")

    if not model_dirs:
        print("Error: No valid model directories found")
        return 0, 0

    # 输出目录
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CBF Neural Network Visualization")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Systems: {', '.join(systems_2d.keys())}")
    print(f"Model directories: {', '.join(d for d, _ in model_dirs)}")
    print("=" * 70)

    total = 0
    success = 0

    
    for model_dir, variant in model_dirs:
        model_path = Path(model_dir)

        print(f"\n{'='*70}")
        print(f"Processing {variant.upper()} models from {model_dir}")
        print(f"{'='*70}")

        for system_name, config in systems_2d.items():
            model_file = model_path / f"{system_name}_Tanh_cbf_repaired_v10_ibp.pth"

            if not model_file.exists():
                print(f"\n  Model not found: {model_file}")
                continue

            total += 1
            print(f"\n[{total}] Visualizing {model_file})...")

            # Create dynamics instance
            dynamics = config['dynamics']()
            alpha = config['alpha']

            # Create output filename
            output_file = output_dir / f"{system_name}_{variant}_cbf.png"

            # Generate visualization
            if visualize_cbf(model_file, dynamics, output_file, alpha, activation=activation):
                success += 1

    print(f"\n{'='*70}")
    print(f"Visualization Complete: {success}/{total} successful")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*70}")

    return success, total


if __name__ == "__main__":
    success, total = main()
    sys.exit(0 if success == total else 1)
