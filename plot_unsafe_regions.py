"""
Visualize unsafe and safe regions for simple_2d, barr1, barr2, barr3 systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem
from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System


def create_grid(domain_bounds, resolution=200):
    """Create a grid of points over the domain."""
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    return X, Y, points


def classify_points(system, points, translator=None):
    """Classify each point as safe, unsafe interior, or unsafe exterior."""
    from lbp_neural_cbf.translators import NumpyTranslator
    if translator is None:
        translator = NumpyTranslator()

    # Get unsafe set classification
    unsafe_interior_mask = system.unsafe_set_interior.contains(points, translator)

    # Get input domain bounds check
    x_min, x_max = system.input_domain.bounds[0]
    y_min, y_max = system.input_domain.bounds[1]
    in_bounds_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                     (points[:, 1] >= y_min) & (points[:, 1] <= y_max)

    # Safe set: in bounds but not in unsafe interior
    safe_mask = in_bounds_mask & ~unsafe_interior_mask

    # Unsafe exterior: outside input domain
    unsafe_exterior_mask = ~in_bounds_mask

    # Unsafe interior: in bounds and in unsafe set
    unsafe_interior_class = unsafe_interior_mask & in_bounds_mask

    return safe_mask, unsafe_interior_class, unsafe_exterior_mask


def plot_system(system, ax, title, resolution=300):
    """Plot safe/unsafe regions for a given system."""
    X, Y, points = create_grid(system.input_domain.bounds, resolution=resolution)

    safe_mask, unsafe_int_mask, unsafe_ext_mask = classify_points(system, points)

    # Create color map: safe=green, unsafe_int=red, unsafe_ext=gray
    colors = ['#d3d3d3', '#ff6b6b', '#69db7c']  # gray, red, green
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    # Create classification image
    # 0 = unsafe exterior, 1 = unsafe interior, 2 = safe
    classification = np.zeros(len(points), dtype=int)
    classification[safe_mask] = 2
    classification[unsafe_int_mask] = 1

    img_data = classification.reshape(X.shape)

    # Plot
    ax.imshow(img_data, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()],
              cmap=cmap, vmin=0, vmax=2, aspect='auto', alpha=0.7)

    # Add boundary lines
    ax.axhline(y=system.input_domain.bounds[1][0], color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=system.input_domain.bounds[1][1], color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=system.input_domain.bounds[0][0], color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=system.input_domain.bounds[0][1], color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.set_xlim(system.input_domain.bounds[0])
    ax.set_ylim(system.input_domain.bounds[1])
    ax.grid(True, alpha=0.3)


def main():
    systems = [
        (Simple2DSystem(), 'Simple2D', 'Simple 2D System\n(Circle + Parabola)'),
        (Barrier1System(), 'barr1', 'Barrier 1 System\n(Parabola: x1 + x2² ≤ 0)'),
        (Barrier2System(), 'barr2', 'Barrier 2 System\n(Circles + Parabola)'),
        (Barrier3System(), 'barr3', 'Barrier 3 System\n(Circle + L-shape + Parabola)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (sys_obj, sys_name, title) in enumerate(systems):
        plot_system(sys_obj, axes[idx], title)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#69db7c', edgecolor='black', label='Safe Region'),
        Patch(facecolor='#ff6b6b', edgecolor='black', label='Unsafe Region (Interior)'),
        Patch(facecolor='#d3d3d3', edgecolor='black', label='Outside Input Domain'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Safe and Unsafe Regions for CBF Systems', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    output_path = 'unsafe_regions_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {output_path}')
    plt.show()


if __name__ == '__main__':
    main()
