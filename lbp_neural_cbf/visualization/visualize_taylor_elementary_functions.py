"""
Visualization script for certified Taylor expansions of elementary functions.

This script creates comprehensive visualizations showing:
1. True function values
2. First-order Taylor approximations
3. Certified error bounds
4. Expansion points and domains

Covers elementary functions: sin, cos, exp, log, sqrt, cbrt, and power functions.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ..translators import CertifiedFirstOrderTaylorExpansion, TaylorTranslator


class ElementaryFunctionVisualizer:
    """Visualize Taylor expansions for elementary functions."""

    def __init__(self, output_dir="plots/taylor_elementary_functions"):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.translator = TaylorTranslator()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Setup matplotlib for high-quality plots
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 10

    def create_taylor_expansion(self, expansion_point, domain_width, function_name=None):
        """Create a Taylor expansion for the identity function over a domain.

        Args:
            expansion_point: Center point for expansion (scalar)
            domain_width: Half-width of the domain around expansion point
            function_name: Name of function (for domain validation)

        Returns:
            CertifiedFirstOrderTaylorExpansion for f(x) = x
        """
        expansion_point = np.array([expansion_point])
        lower_bound = expansion_point - domain_width
        upper_bound = expansion_point + domain_width

        # Validate domain for certain functions
        if function_name == "log" and np.any(lower_bound <= 0):
            lower_bound = np.maximum(lower_bound, 0.01)
        elif function_name == "sqrt" and np.any(lower_bound < 0):
            lower_bound = np.maximum(lower_bound, 0.0)

        return self.translator.to_format(expansion_point, lower_bound, upper_bound)

    def plot_function_comparison(self, x_vals, true_vals, taylor_expansion, expansion_point, title, save_name):
        """Plot comparison between true function and Taylor expansion.

        Args:
            x_vals: Array of x values for plotting
            true_vals: True function values
            taylor_expansion: CertifiedFirstOrderTaylorExpansion object
            expansion_point: Expansion center point
            title: Plot title
            save_name: Filename for saving
        """
        # Compute Taylor approximation components
        J, f_c = taylor_expansion.linear_approximation
        R_lower, R_upper = taylor_expansion.remainder

        # Linear approximation: f(c) + J(x - c)
        x_expanded = np.atleast_1d(x_vals)
        c_expanded = np.atleast_1d(expansion_point)

        # Handle different dimensionalities properly
        if J.ndim == 1:
            # 1D case: J is already a vector
            linear_approx = f_c + J * (x_expanded - c_expanded)
        elif J.ndim == 2 and J.shape[0] == 1:
            # 1D output but 2D Jacobian matrix
            J_vec = J.flatten()
            linear_approx = f_c + J_vec * (x_expanded - c_expanded)
        else:
            # Multi-dimensional case
            dx = x_expanded.reshape(-1, 1) - c_expanded.reshape(1, -1)
            linear_approx = f_c + (J @ dx.T).flatten()

        # Flatten arrays if needed for consistent shapes
        if hasattr(linear_approx, "shape") and linear_approx.ndim > 1:
            linear_approx = linear_approx.flatten()
        if hasattr(R_lower, "shape") and R_lower.ndim > 1:
            R_lower = R_lower.flatten()
        if hasattr(R_upper, "shape") and R_upper.ndim > 1:
            R_upper = R_upper.flatten()

        # Ensure scalar values are broadcasted correctly
        if np.isscalar(R_lower):
            R_lower = np.full_like(linear_approx, R_lower)
        if np.isscalar(R_upper):
            R_upper = np.full_like(linear_approx, R_upper)

        lower_bound = linear_approx + R_lower
        upper_bound = linear_approx + R_upper

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot components
        ax.plot(x_vals, true_vals, "b-", linewidth=2.5, label="True Function", alpha=0.9)
        ax.plot(x_vals, linear_approx, "r--", linewidth=2, label="Taylor Approximation", alpha=0.8)
        ax.fill_between(x_vals, lower_bound, upper_bound, color="orange", alpha=0.3, label="Certified Bounds")
        ax.plot(x_vals, lower_bound, "g:", linewidth=1.5, label="Lower Bound", alpha=0.7)
        ax.plot(x_vals, upper_bound, "m:", linewidth=1.5, label="Upper Bound", alpha=0.7)

        # Mark expansion point
        if hasattr(f_c, "shape") and f_c.shape == ():
            y_at_expansion = f_c.item()
        elif hasattr(f_c, "shape") and len(f_c.shape) > 0 and f_c.shape[0] > 0:
            y_at_expansion = f_c[0]
        else:
            y_at_expansion = f_c

        ax.plot(
            expansion_point,
            y_at_expansion,
            "ko",
            markersize=10,
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=2,
            label="Expansion Point",
            zorder=5,
        )

        # Formatting
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("f(x)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add domain highlighting
        domain_lower, domain_upper = taylor_expansion.domain
        ax.axvspan(domain_lower[0], domain_upper[0], alpha=0.15, color="gray", label="Taylor Domain", zorder=0)

        # Ensure legend includes all elements
        ax.legend(loc="best", framealpha=0.9)

        # Improve layout and save
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
        plt.show()
        plt.close()

    def visualize_sin(self):
        """Visualize Taylor expansion of sine function."""
        expansion_point = np.pi / 4  # 45 degrees
        domain_width = 1.0

        # Create base Taylor expansion
        x_taylor = self.create_taylor_expansion(expansion_point, domain_width)

        # Apply sine function
        sin_taylor = self.translator.sin(x_taylor)

        # Generate test points
        x_vals = np.linspace(-0.5, 2.5, 300)
        true_vals = np.sin(x_vals)

        self.plot_function_comparison(
            x_vals,
            true_vals,
            sin_taylor,
            expansion_point,
            f"Certified Taylor Expansion: sin(x) around x = π/4 ≈ {expansion_point:.3f}",
            "sin_taylor_expansion",
        )

    def visualize_cos(self):
        """Visualize Taylor expansion of cosine function."""
        expansion_point = np.pi / 6  # 30 degrees
        domain_width = 0.8

        x_taylor = self.create_taylor_expansion(expansion_point, domain_width)
        cos_taylor = self.translator.cos(x_taylor)

        x_vals = np.linspace(-0.5, 2.0, 300)
        true_vals = np.cos(x_vals)

        self.plot_function_comparison(
            x_vals,
            true_vals,
            cos_taylor,
            expansion_point,
            f"Certified Taylor Expansion: cos(x) around x = π/6 ≈ {expansion_point:.3f}",
            "cos_taylor_expansion",
        )

    def visualize_exp(self):
        """Visualize Taylor expansion of exponential function."""
        expansion_point = 1.0
        domain_width = 0.8

        x_taylor = self.create_taylor_expansion(expansion_point, domain_width)
        exp_taylor = self.translator.exp(x_taylor)

        x_vals = np.linspace(-0.5, 2.5, 300)
        true_vals = np.exp(x_vals)

        self.plot_function_comparison(
            x_vals,
            true_vals,
            exp_taylor,
            expansion_point,
            f"Certified Taylor Expansion: exp(x) around x = {expansion_point}",
            "exp_taylor_expansion",
        )

    def visualize_log(self):
        """Visualize Taylor expansion of natural logarithm."""
        expansion_point = 1.0
        domain_width = 0.7

        x_taylor = self.create_taylor_expansion(expansion_point, domain_width, "log")
        log_taylor = self.translator.log(x_taylor)

        x_vals = np.linspace(0.1, 2.5, 300)
        true_vals = np.log(x_vals)

        self.plot_function_comparison(
            x_vals,
            true_vals,
            log_taylor,
            expansion_point,
            f"Certified Taylor Expansion: ln(x) around x = {expansion_point}",
            "log_taylor_expansion",
        )

    def visualize_sqrt(self):
        """Visualize Taylor expansion of square root function."""
        expansion_point = 1.0
        domain_width = 0.8

        x_taylor = self.create_taylor_expansion(expansion_point, domain_width, "sqrt")
        sqrt_taylor = self.translator.sqrt(x_taylor)

        x_vals = np.linspace(0.0, 2.5, 300)
        true_vals = np.sqrt(x_vals)

        self.plot_function_comparison(
            x_vals,
            true_vals,
            sqrt_taylor,
            expansion_point,
            f"Certified Taylor Expansion: √x around x = {expansion_point}",
            "sqrt_taylor_expansion",
        )

    def visualize_cbrt(self):
        """Visualize Taylor expansion of cube root function."""
        expansion_point = 1.0
        domain_width = 1.0

        x_taylor = self.create_taylor_expansion(expansion_point, domain_width)
        cbrt_taylor = self.translator.cbrt(x_taylor)

        x_vals = np.linspace(-1.0, 3.0, 300)
        true_vals = np.cbrt(x_vals)

        self.plot_function_comparison(
            x_vals,
            true_vals,
            cbrt_taylor,
            expansion_point,
            f"Certified Taylor Expansion: ∛x around x = {expansion_point}",
            "cbrt_taylor_expansion",
        )

    def visualize_power_functions(self):
        """Visualize Taylor expansions of various power functions."""
        expansion_point = 1.0
        domain_width = 0.6
        exponents = [2, 3, -1, -2]  # Use integer exponents only

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, exp in enumerate(exponents):
            x_taylor = self.create_taylor_expansion(expansion_point, domain_width)
            power_taylor = self.translator.pow(x_taylor, exp)

            # Generate appropriate x range based on exponent
            if exp < 0:
                x_vals = np.linspace(0.1, 2.5, 300)
            else:
                x_vals = np.linspace(0.0, 2.5, 300)

            true_vals = np.power(x_vals, exp)

            # Compute Taylor approximation
            J, f_c = power_taylor.linear_approximation
            R_lower, R_upper = power_taylor.remainder

            # Ensure arrays are properly shaped
            if J.ndim == 2 and J.shape[0] == 1:
                J = J.flatten()
            if np.isscalar(f_c):
                f_c_val = f_c
            elif hasattr(f_c, "shape") and f_c.shape == (1,):
                f_c_val = f_c[0]
            else:
                f_c_val = f_c

            if np.isscalar(R_lower):
                R_lower_val = R_lower
            elif hasattr(R_lower, "shape") and R_lower.shape == (1,):
                R_lower_val = R_lower[0]
            else:
                R_lower_val = R_lower

            if np.isscalar(R_upper):
                R_upper_val = R_upper
            elif hasattr(R_upper, "shape") and R_upper.shape == (1,):
                R_upper_val = R_upper[0]
            else:
                R_upper_val = R_upper

            linear_approx = f_c_val + J * (x_vals - expansion_point)
            lower_bound = linear_approx + R_lower_val
            upper_bound = linear_approx + R_upper_val

            # Plot on subplot
            ax = axes[i]
            ax.plot(x_vals, true_vals, "b-", linewidth=2.5, label="True Function")
            ax.plot(x_vals, linear_approx, "r--", linewidth=2, label="Taylor Approx")
            ax.fill_between(x_vals, lower_bound, upper_bound, color="orange", alpha=0.3, label="Certified Bounds")

            # Mark expansion point
            y_at_expansion = f_c_val
            ax.plot(
                expansion_point,
                y_at_expansion,
                "ko",
                markersize=8,
                markerfacecolor="yellow",
                markeredgecolor="black",
                markeredgewidth=2,
            )

            # Domain highlighting
            domain_lower, domain_upper = power_taylor.domain
            ax.axvspan(domain_lower[0], domain_upper[0], alpha=0.15, color="gray", label="Taylor Domain" if i == 0 else "")

            ax.set_title(f"x^{exp}", fontsize=12, fontweight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            if i == 0:  # Add legend only to first subplot
                ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)

            # Set reasonable y-limits
            if exp == -1:
                ax.set_ylim(-5, 10)
            elif exp == -2:
                ax.set_ylim(-2, 20)
            elif exp == 3:
                ax.set_ylim(-2, 15)

        plt.suptitle("Certified Taylor Expansions: Power Functions around x = 1", fontsize=16, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "power_functions_taylor_expansions.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
        plt.show()
        plt.close()

    def visualize_composite_function(self):
        """Visualize Taylor expansion of a composite function: sin(exp(x))."""
        expansion_point = 0.5
        domain_width = 0.4

        # Create composite function: sin(exp(x))
        x_taylor = self.create_taylor_expansion(expansion_point, domain_width)
        exp_taylor = self.translator.exp(x_taylor)
        sin_exp_taylor = self.translator.sin(exp_taylor)

        x_vals = np.linspace(0.0, 1.2, 300)
        true_vals = np.sin(np.exp(x_vals))

        self.plot_function_comparison(
            x_vals,
            true_vals,
            sin_exp_taylor,
            expansion_point,
            f"Certified Taylor Expansion: sin(exp(x)) around x = {expansion_point}",
            "sin_exp_composite_taylor_expansion",
        )

    def create_comparison_grid(self):
        """Create a grid comparison of multiple elementary functions."""
        functions = [
            ("sin", np.pi / 4, 1.0, np.sin),
            ("exp", 0.5, 0.6, np.exp),
            ("log", 1.5, 0.8, np.log),
            ("sqrt", 2.0, 1.0, np.sqrt),
            ("x^2", 1.0, 0.8, lambda x: x**2),
            ("sin(exp(x))", 0.5, 0.4, lambda x: np.sin(np.exp(x))),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (name, exp_point, domain_w, true_func) in enumerate(functions):
            ax = axes[i]

            # Create Taylor expansion
            if name == "log":
                x_taylor = self.create_taylor_expansion(exp_point, domain_w, "log")
                log_taylor = self.translator.log(x_taylor)
                taylor_exp = log_taylor
                x_range = np.linspace(0.1, 3.0, 200)
            elif name == "sqrt":
                x_taylor = self.create_taylor_expansion(exp_point, domain_w, "sqrt")
                sqrt_taylor = self.translator.sqrt(x_taylor)
                taylor_exp = sqrt_taylor
                x_range = np.linspace(0.0, 4.0, 200)
            elif name == "x^2":
                x_taylor = self.create_taylor_expansion(exp_point, domain_w)
                pow_taylor = self.translator.pow(x_taylor, 2)  # Use integer 2
                taylor_exp = pow_taylor
                x_range = np.linspace(-1.0, 3.0, 200)
            elif name == "sin(exp(x))":
                x_taylor = self.create_taylor_expansion(exp_point, domain_w)
                exp_taylor = self.translator.exp(x_taylor)
                sin_exp_taylor = self.translator.sin(exp_taylor)
                taylor_exp = sin_exp_taylor
                x_range = np.linspace(0.0, 1.2, 200)
            else:
                x_taylor = self.create_taylor_expansion(exp_point, domain_w)
                if name == "sin":
                    taylor_exp = self.translator.sin(x_taylor)
                    x_range = np.linspace(-1.0, 3.0, 200)
                elif name == "cos":
                    taylor_exp = self.translator.cos(x_taylor)
                    x_range = np.linspace(-1.0, 3.0, 200)
                elif name == "exp":
                    taylor_exp = self.translator.exp(x_taylor)
                    x_range = np.linspace(-0.5, 2.0, 200)

            true_vals = true_func(x_range)

            # Compute Taylor approximation
            J, f_c = taylor_exp.linear_approximation
            R_lower, R_upper = taylor_exp.remainder

            # Handle different array shapes
            if J.ndim == 2 and J.shape[0] == 1:
                J = J.flatten()
            if hasattr(f_c, "shape") and f_c.shape == (1,):
                f_c = f_c[0]
            if hasattr(R_lower, "shape") and R_lower.shape == (1,):
                R_lower = R_lower[0]
            if hasattr(R_upper, "shape") and R_upper.shape == (1,):
                R_upper = R_upper[0]

            linear_approx = f_c + J * (x_range - exp_point)
            lower_bound = linear_approx + R_lower
            upper_bound = linear_approx + R_upper

            # Plot
            ax.plot(x_range, true_vals, "b-", linewidth=2, label="True", alpha=0.9)
            ax.plot(x_range, linear_approx, "r--", linewidth=1.5, label="Taylor", alpha=0.8)
            ax.fill_between(x_range, lower_bound, upper_bound, color="orange", alpha=0.2, label="Bounds")

            # Mark expansion point
            y_at_expansion = f_c
            ax.plot(
                exp_point,
                y_at_expansion,
                "ko",
                markersize=6,
                markerfacecolor="yellow",
                markeredgecolor="black",
                markeredgewidth=1,
                label="Expansion Point" if i == 0 else "",
            )

            # Domain highlighting
            domain_lower, domain_upper = taylor_exp.domain
            ax.axvspan(domain_lower[0], domain_upper[0], alpha=0.15, color="gray", label="Domain" if i == 0 else "")

            # Set title - handle special case for composite functions
            if "(x)" in name:
                title = name  # Already includes (x)
            else:
                title = f"{name}(x)"

            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("f(x)", fontsize=12)
            ax.grid(True, alpha=0.3)

            if i == 0:  # Add legend only to first subplot
                ax.legend(fontsize=15, framealpha=0.9)

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "elementary_functions_comparison_grid.png")
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
        plt.show()
        plt.close()

    def run_all_visualizations(self):
        """Run all visualization functions."""
        print("Creating Taylor expansion visualizations for elementary functions...")
        print(f"Output directory: {self.output_dir}")

        try:
            self.create_comparison_grid()
            self.visualize_sin()
            self.visualize_cos()
            self.visualize_exp()
            self.visualize_log()
            self.visualize_sqrt()
            self.visualize_cbrt()
            self.visualize_power_functions()
            self.visualize_composite_function()

            print("\n✓ All visualizations completed successfully!")
            print(f"  Generated plots saved in: {self.output_dir}")

        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            raise


def main():
    """Main function to run all visualizations."""
    print("=" * 80)
    print("Certified Taylor Expansion Visualizer for Elementary Functions")
    print("=" * 80)

    visualizer = ElementaryFunctionVisualizer()
    visualizer.run_all_visualizations()

    print("\n" + "=" * 80)
    print("Visualization complete! Check the plots directory for results.")
    print("=" * 80)


if __name__ == "__main__":
    main()
