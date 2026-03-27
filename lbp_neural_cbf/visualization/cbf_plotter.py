"""
CBF-specific visualization for barrier functions, gradients, and verification regions.
"""

import matplotlib

# Use non-interactive backend for wandb-only visualization
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from ..regions import SimplicialRegion
from ..translators import NumpyTranslator, TorchTranslator


class CBFVerificationPlotter:
    """
    Specialized plotter for CBF verification with barrier function and CBF condition visualization.

    This plotter creates contour plots of:
    - Barrier function h(x)
    - CBF condition ∇h·f + α(h)
    - Safe and unsafe sets
    - Verification regions (simplicial mesh) that update during verification
    """

    def __init__(self, dynamics_model, barrier_net, resolution=300, figsize=(15, 5), alpha=1.0, update_interval=50):
        """
        Initialize the CBF verification plotter.

        Args:
            dynamics_model: CBF dynamical system
            barrier_net: Trained barrier function network
            resolution: Grid resolution for contour plots
            figsize: Figure size (width, height)
            alpha: Class K function parameter for CBF condition
            update_interval: Number of regions to accumulate before updating display (default: 50)
        """
        if dynamics_model.input_dim != 2:
            raise ValueError("CBF visualization only supports 2D systems")

        self.dynamics_model = dynamics_model
        self.barrier_net = barrier_net
        self.resolution = resolution
        self.figsize = figsize
        self.alpha = alpha
        self.update_interval = update_interval
        self.updates_since_last_draw = 0

        # Get device for PyTorch operations
        self.device = next(barrier_net.parameters()).device if barrier_net else "cpu"
        self.torch_translator = TorchTranslator(device=self.device)
        self.numpy_translator = NumpyTranslator()

        # Create grid for contour plots
        input_domain = dynamics_model.input_domain
        # If input_domain is a domain object, extract bounds
        if hasattr(input_domain, "bounds"):
            bounds = input_domain.bounds
        else:
            bounds = input_domain
        self.x_min, self.x_max = bounds[0]
        self.y_min, self.y_max = bounds[1]

        self.x_grid = np.linspace(self.x_min, self.x_max, resolution)
        self.y_grid = np.linspace(self.y_min, self.y_max, resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

        # Evaluate barrier function and CBF condition on grid
        self._compute_barrier_grid()
        self._compute_safe_unsafe_grid()

        # Initialize figure with subplots
        self.fig, self.axes = plt.subplots(1, 3, figsize=figsize)
        self.ax_barrier, self.ax_gradient, self.ax_verification = self.axes
        # Use equal data aspect to avoid distortion and reflect axis bounds ratio
        for ax in self.axes:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)
        self.fig.set_constrained_layout(True)

        fig_w, fig_h = self.figsize
        approx_ax_w = max(fig_w / 3.0, 1e-6)
        approx_ax_h = max(fig_h, 1e-6)
        data_aspect = 0.75 * (self.y_max - self.y_min) / max(self.x_max - self.x_min, 1e-12)
        eff_h = min(approx_ax_h, approx_ax_w * data_aspect)
        cbar_shrink = float(np.clip(eff_h / approx_ax_h, 0.3, 1.0))

        # Shared colorbar kwargs for consistent sizing
        self._cbar_kwargs = {"shrink": cbar_shrink, "pad": 0.02, "aspect": 25}

        # Storage for verification regions
        self.verified_patches = []  # SAT regions (green)
        self.counterexample_patches = []  # UNSAT regions (red)
        self.maybe_patches = []  # MAYBE regions (yellow)

        # Pending patches to be added (for batching)
        self.pending_patches = []

        # Track colorbars to cleanly refresh them
        self.barrier_colorbar = None
        self.cbf_colorbar = None

        # Handles for optional training sample overlays
        self.sample_handles = []
        self.base_legend_handles = []

        # Initialize plots
        self._init_barrier_plot()
        self._init_cbf_condition_plot()
        self._init_verification_plot()

    def _compute_barrier_grid(self):
        """Compute barrier function values and CBF condition on grid."""
        if self.barrier_net is None:
            print("No barrier network provided - skipping barrier function visualization")
            # Set to None to indicate no barrier function available
            self.H = None
            self.cbf_condition = None
            return

        # Create grid points for evaluation
        grid_points = torch.tensor(np.stack([self.X.ravel(), self.Y.ravel()]), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Evaluate barrier function
            h_values = self.barrier_net(grid_points.T).squeeze()
            self.H = h_values.cpu().numpy().reshape(self.X.shape)

        # Compute CBF condition: ∇h·f + α(h)
        # Build gradient grid as [num_points, state_dim] to match dynamics
        grid_points_grad = torch.tensor(
            np.stack([self.X.ravel(), self.Y.ravel()], axis=1),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        # Evaluate barrier function for gradient computation
        h_values_grad = self.barrier_net(grid_points_grad).squeeze()

        # Compute CBF condition for each point individually
        num_points = h_values_grad.numel()
        cbf_condition_values = []

        # Evaluate barrier function for gradient computation
        h_values_grad = self.barrier_net(grid_points_grad).squeeze()

        # Compute gradients efficiently using backward pass
        grad_outputs = torch.ones_like(h_values_grad)
        grad_h = torch.autograd.grad(outputs=h_values_grad, inputs=grid_points_grad, grad_outputs=grad_outputs, create_graph=False, retain_graph=False)[
            0
        ]  # Shape: [num_points, state_dim]

        # Compute drift dynamics f(x) for all points at once
        with torch.no_grad():
            # Expect shape [num_points, state_dim]; normalize if needed
            f_x = self.dynamics_model.compute_f(grid_points_grad, self.torch_translator)
            if f_x.ndim == 2 and f_x.shape[0] == grad_h.shape[1] and f_x.shape[1] == grad_h.shape[0]:
                # Convert [state_dim, num_points] -> [num_points, state_dim]
                f_x = f_x.T

        # Compute Lie derivative: L_f(h) = ∇h · f (per-point dot product)
        lie_derivative = torch.sum(grad_h * f_x, dim=-1)  # Shape: [num_points]

        # Compute control term: sup_u [∇h·g(x)·u] for affine control systems
        control_term = torch.zeros_like(h_values_grad)
        if self.dynamics_model.control_dim and self.dynamics_model.control_dim > 0:
            with torch.no_grad():
                control_dim = self.dynamics_model.control_dim

                g_x = self.dynamics_model.compute_g(
                    grid_points_grad, self.torch_translator
                )  # Shape: [control_dim, state_dim] or [batch_size, control_dim, state_dim]

                # Normalize g(x) shape to [num_points, state_dim, control_dim]
                if g_x.dim() == 2:
                    # Constant control matrix [state_dim, control_dim] -> broadcast over points
                    g_x = g_x.unsqueeze(0)

                # Compute coefficients c = grad_h · g(x) -> [num_points, control_dim]
                ctrl_coeffs = (grad_h.unsqueeze(-2) * g_x).sum(dim=-1)

                # Bounds as tensors on correct device/dtype
                u_min = torch.as_tensor(self.dynamics_model.u_min, device=ctrl_coeffs.device, dtype=ctrl_coeffs.dtype)
                u_max = torch.as_tensor(self.dynamics_model.u_max, device=ctrl_coeffs.device, dtype=ctrl_coeffs.dtype)

                # Per-dim optimal control contribution
                control_contrib = torch.where(ctrl_coeffs >= 0, ctrl_coeffs * u_max, ctrl_coeffs * u_min)
                control_term = control_contrib.sum(dim=-1)

        # Compute α(h)
        alpha_h = self.alpha * h_values_grad

        # CBF condition: L_f(h) + sup_u[L_g(h)·u] + α(h) ≥ 0
        # This matches exactly what is trained in the loss function
        cbf_condition_tensor = lie_derivative + control_term + alpha_h

        # Convert to numpy array and reshape to grid shape
        cbf_condition_array = cbf_condition_tensor.detach().cpu().numpy()
        self.cbf_condition = cbf_condition_array.reshape(self.X.shape)

        # Validate barrier function values
        h_min, h_max = self.H.min(), self.H.max()
        if np.isnan(h_min) or np.isnan(h_max):
            raise ValueError(f"Barrier function contains NaN values: range=[{h_min}, {h_max}]")
        if np.isinf(h_min) or np.isinf(h_max):
            raise ValueError(f"Barrier function contains infinite values: range=[{h_min}, {h_max}]")
        if abs(h_max - h_min) < 1e-12:
            raise ValueError(f"Barrier function is constant: h(x) = {h_min:.12f}. This indicates a problem with CBF training.")

        # Validate CBF condition values
        cbf_min, cbf_max = self.cbf_condition.min(), self.cbf_condition.max()
        if np.isnan(cbf_min) or np.isnan(cbf_max):
            raise ValueError(f"CBF condition contains NaN values: range=[{cbf_min}, {cbf_max}]")
        if np.isinf(cbf_min) or np.isinf(cbf_max):
            raise ValueError(f"CBF condition contains infinite values: range=[{cbf_min}, {cbf_max}]")

    def _compute_safe_unsafe_grid(self):
        """Compute safe and unsafe set indicators on grid."""
        # Use standard (batch_size, dim) layout to avoid broadcasting issues
        grid_points = np.stack([self.X.ravel(), self.Y.ravel()], axis=1)

        # Evaluate safe set constraint
        safe_values = self.dynamics_model.safe_set_constraint(grid_points, self.numpy_translator)
        self.safe_indicator = safe_values.reshape(self.X.shape)

        # Validate safe set constraint values
        safe_min, safe_max = self.safe_indicator.min(), self.safe_indicator.max()
        if np.isnan(safe_min) or np.isnan(safe_max):
            raise ValueError(f"Safe set constraint contains NaN values: range=[{safe_min}, {safe_max}]")
        if np.isinf(safe_min) or np.isinf(safe_max):
            raise ValueError(f"Safe set constraint contains infinite values: range=[{safe_min}, {safe_max}]")

    def _init_barrier_plot(self):
        """Initialize barrier function contour plot."""
        if self.barrier_colorbar is not None:
            # Remove previous colorbar axis before redrawing
            try:
                self.barrier_colorbar.remove()
            except Exception:
                # Fallback: remove underlying axis if present
                if hasattr(self.barrier_colorbar, "ax"):
                    self.barrier_colorbar.ax.remove()
            finally:
                self.barrier_colorbar = None

        self.ax_barrier.clear()
        self.ax_barrier.set_title("Barrier Function h(x)")
        if self.H is None:
            # No barrier function available
            self.ax_barrier.text(
                0.5,
                0.5,
                "No barrier function available\n(ONNX model or training not completed)",
                transform=self.ax_barrier.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )
        else:
            # Normal contour plot - validation already done in _compute_barrier_grid
            h_min, h_max = float(self.H.min()), float(self.H.max())
            levels = np.linspace(h_min, h_max, 31) if h_max > h_min else 20
            barrier_norm = mcolors.TwoSlopeNorm(vmin=h_min, vcenter=0.0, vmax=h_max) if h_min < 0 < h_max else None
            self.barrier_contour = self.ax_barrier.contourf(self.X, self.Y, self.H, levels=levels, cmap="RdYlBu", alpha=0.8, norm=barrier_norm)
            # Add colorbar
            self.barrier_colorbar = self.fig.colorbar(
                self.barrier_contour,
                ax=self.ax_barrier,
                label="h(x)",
                **self._cbar_kwargs,
            )
            if barrier_norm is not None and self.barrier_colorbar is not None:
                self.barrier_colorbar.ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
            # Zero level set (barrier boundary) in bold if it exists in range
            if h_min <= 0 <= h_max:
                self.ax_barrier.contour(self.X, self.Y, self.H, levels=[0], colors="black", linewidths=3)
        # Safe set boundary (always try to show this)
        try:
            safe_min, safe_max = self.safe_indicator.min(), self.safe_indicator.max()
            if not (np.isnan(safe_min) or np.isnan(safe_max)) and safe_min <= 0 <= safe_max:
                self.ax_barrier.contour(self.X, self.Y, self.safe_indicator, levels=[0], colors="red", linewidths=2, linestyles="--")
        except Exception as e:
            print(f"Warning: Could not plot safe set boundary: {e}")
        self.ax_barrier.set_xlabel("x₁")
        self.ax_barrier.set_ylabel("x₂")
        # Keep consistent limits and aspect after clearing
        self.ax_barrier.set_xlim(self.x_min, self.x_max)
        self.ax_barrier.set_ylim(self.y_min, self.y_max)
        self.ax_barrier.set_aspect("equal", adjustable="box")
        self.ax_barrier.grid(True, alpha=0.3)

    def _init_cbf_condition_plot(self):
        """Initialize CBF condition contour plot."""
        if self.cbf_colorbar is not None:
            try:
                self.cbf_colorbar.remove()
            except Exception:
                if hasattr(self.cbf_colorbar, "ax"):
                    self.cbf_colorbar.ax.remove()
            finally:
                self.cbf_colorbar = None

        self.ax_gradient.clear()
        # Create title based on whether system has control
        if self.dynamics_model.control_dim > 0:
            title = f"Lf(h) + sup_u[Lg(h)·u] + α·h (α={self.alpha})"
            cbar_label = "Lf(h) + sup_u[Lg(h)·u] + α·h"
        else:
            title = f"∇h·f + α·h (α={self.alpha})"
            cbar_label = "∇h·f + α(h)"
        self.ax_gradient.set_title(title)
        if self.cbf_condition is None:
            # No CBF condition information available
            self.ax_gradient.text(
                0.5,
                0.5,
                "No CBF condition available\n(ONNX model or training not completed)",
                transform=self.ax_gradient.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )
        else:
            # Normal contour plot - validation already done in _compute_barrier_grid
            cbf_min, cbf_max = float(self.cbf_condition.min()), float(self.cbf_condition.max())
            # Use robust limits to avoid extreme outliers dominating the colorbar
            flat_cbf = self.cbf_condition.reshape(-1)
            neg_vals = flat_cbf[flat_cbf < 0]
            pos_vals = flat_cbf[flat_cbf > 0]
            # Percentile clipping on each side around 0
            try:
                vmin = float(np.percentile(neg_vals, 2)) if neg_vals.size > 0 else cbf_min
                vmax = float(np.percentile(pos_vals, 98)) if pos_vals.size > 0 else cbf_max
            except Exception:
                vmin, vmax = cbf_min, cbf_max
            # Ensure we straddle zero
            if not (vmin < 0):
                vmin = cbf_min if cbf_min < 0 else -1.0
            if not (vmax > 0):
                vmax = cbf_max if cbf_max > 0 else 1.0
            # Fallback if range collapses
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = -1.0, 1.0
            levels = np.linspace(vmin, vmax, 31)
            cbf_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            # Use RdYlGn colormap: red for negative (violations), green for positive (satisfied)
            self.cbf_contour = self.ax_gradient.contourf(
                self.X,
                self.Y,
                self.cbf_condition,
                levels=levels,
                cmap="RdYlGn",
                alpha=0.8,
                norm=cbf_norm,
                extend="both",  # ensure out-of-range values are colored, not white
            )
            # Add colorbar with appropriate label
            self.cbf_colorbar = self.fig.colorbar(
                self.cbf_contour,
                ax=self.ax_gradient,
                label=cbar_label,
                extend="both",
                **self._cbar_kwargs,
            )
            if cbf_norm is not None and self.cbf_colorbar is not None:
                self.cbf_colorbar.ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
            # Add zero level set (CBF condition boundary) in bold purple if it exists in range
            if vmin <= 0 <= vmax:
                self.ax_gradient.contour(self.X, self.Y, self.cbf_condition, levels=[0], colors="purple", linewidths=3)
        # Safe set boundary and barrier boundary (always try to show these)
        try:
            safe_min, safe_max = self.safe_indicator.min(), self.safe_indicator.max()
            if not (np.isnan(safe_min) or np.isnan(safe_max)) and safe_min <= 0 <= safe_max:
                self.ax_gradient.contour(self.X, self.Y, self.safe_indicator, levels=[0], colors="red", linewidths=2, linestyles="--")
        except Exception as e:
            print(f"Warning: Could not plot safe set boundary on CBF condition plot: {e}")
        if self.H is not None:
            try:
                h_min, h_max = self.H.min(), self.H.max()
                if not (np.isnan(h_min) or np.isnan(h_max)) and h_min <= 0 <= h_max:
                    self.ax_gradient.contour(self.X, self.Y, self.H, levels=[0], colors="black", linewidths=2)
            except Exception as e:
                print(f"Warning: Could not plot barrier boundary on CBF condition plot: {e}")
        self.ax_gradient.set_xlabel("x₁")
        self.ax_gradient.set_ylabel("x₂")
        # Keep consistent limits and aspect after clearing
        self.ax_gradient.set_xlim(self.x_min, self.x_max)
        self.ax_gradient.set_ylim(self.y_min, self.y_max)
        self.ax_gradient.set_aspect("equal", adjustable="box")
        self.ax_gradient.grid(True, alpha=0.3)

    def _init_verification_plot(self):
        """Initialize verification regions plot."""
        self.ax_verification.set_title("Verification Regions")

        # Background: safe set in light green (if valid)
        try:
            safe_min, safe_max = self.safe_indicator.min(), self.safe_indicator.max()
            if not (np.isnan(safe_min) or np.isnan(safe_max)) and safe_min < safe_max:
                self.ax_verification.contourf(self.X, self.Y, self.safe_indicator, levels=[0, np.inf], colors=["lightgreen"], alpha=0.3)
        except Exception as e:
            print(f"Warning: Could not plot safe set background: {e}")

        # Barrier boundary (if available and valid)
        if self.H is not None:
            try:
                h_min, h_max = self.H.min(), self.H.max()
                if not (np.isnan(h_min) or np.isnan(h_max)) and h_min <= 0 <= h_max:
                    self.ax_verification.contour(self.X, self.Y, self.H, levels=[0], colors="black", linewidths=3)
            except Exception as e:
                print(f"Warning: Could not plot barrier boundary on verification plot: {e}")

        # CBF condition zero level set (if available and valid)
        if getattr(self, "cbf_condition", None) is not None:
            try:
                cbf_min, cbf_max = self.cbf_condition.min(), self.cbf_condition.max()
                if not (np.isnan(cbf_min) or np.isnan(cbf_max)) and cbf_min <= 0 <= cbf_max:
                    self.ax_verification.contour(self.X, self.Y, self.cbf_condition, levels=[0], colors="purple", linewidths=2.5)
            except Exception as e:
                print(f"Warning: Could not plot CBF condition boundary on verification plot: {e}")

        # Safe set boundary (if valid)
        try:
            safe_min, safe_max = self.safe_indicator.min(), self.safe_indicator.max()
            if not (np.isnan(safe_min) or np.isnan(safe_max)) and safe_min <= 0 <= safe_max:
                self.ax_verification.contour(self.X, self.Y, self.safe_indicator, levels=[0], colors="red", linewidths=2, linestyles="--")
        except Exception as e:
            print(f"Warning: Could not plot safe set boundary on verification plot: {e}")

        self.ax_verification.set_xlabel("x₁")
        self.ax_verification.set_ylabel("x₂")
        # Keep consistent limits and aspect after clearing
        self.ax_verification.set_xlim(self.x_min, self.x_max)
        self.ax_verification.set_ylim(self.y_min, self.y_max)
        self.ax_verification.set_aspect("equal", adjustable="box")
        self.ax_verification.grid(True, alpha=0.3)

        # Add legend (include barrier line only if barrier function is available)
        from matplotlib.patches import Patch

        legend_elements = [
            # Patch(facecolor='green', alpha=0.7, label='Verified (SAT)'),
            # Patch(facecolor='red', alpha=0.7, label='Counterexample (UNSAT)'),
            # Patch(facecolor='yellow', alpha=0.7, label='Inconclusive (MAYBE)'),
            plt.Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Obstacle boundary")
        ]

        if self.H is not None:
            legend_elements.insert(-1, plt.Line2D([0], [0], color="black", linewidth=3, label="Barrier h=0"))
        if getattr(self, "cbf_condition", None) is not None:
            legend_elements.insert(-1, plt.Line2D([0], [0], color="purple", linewidth=2.5, label="CBF cond = 0"))

        self.base_legend_handles = legend_elements
        self._apply_verification_legend()

    @staticmethod
    def _clone_polygon(patch):
        """Create a shallow copy of a polygon patch preserving style attributes."""
        if patch is None:
            return None
        cloned = Polygon(patch.get_xy(), closed=True, alpha=patch.get_alpha())
        cloned.set_facecolor(patch.get_facecolor())
        cloned.set_edgecolor(patch.get_edgecolor())
        cloned.set_linewidth(patch.get_linewidth())
        cloned.set_linestyle(patch.get_linestyle())
        cloned.set_zorder(patch.get_zorder())
        return cloned

    def _plot_sample_scatter(self, samples):
        """Overlay safe/unsafe sample batches on the verification plot."""
        if samples is None:
            self.sample_handles = []
            return

        if isinstance(samples, torch.Tensor):
            sample_tensor = samples.detach().to(self.device, dtype=torch.float32)
        else:
            sample_tensor = torch.tensor(np.asarray(samples, dtype=np.float32), device=self.device)

        if sample_tensor.ndim != 2 or sample_tensor.shape[1] != 2:
            raise ValueError(f"Sample batch must have shape [N, 2]; got {sample_tensor.shape}")

        with torch.no_grad():
            safe_vals = self.dynamics_model.safe_set_constraint(sample_tensor, self.torch_translator)

        safe_vals = safe_vals.detach().view(-1).to(sample_tensor.device)
        sample_np = sample_tensor.detach().cpu().numpy()

        safe_mask = safe_vals >= 0
        unsafe_mask = ~safe_mask

        boundary_eps = 1e-3
        boundary_mask = safe_vals.abs() <= boundary_eps
        safe_mask = safe_mask & ~boundary_mask
        unsafe_mask = unsafe_mask & ~boundary_mask

        handles = []
        if boundary_mask.any():
            boundary_points = sample_np[boundary_mask.cpu().numpy()]
            handles.append(
                self.ax_verification.scatter(
                    boundary_points[:, 0],
                    boundary_points[:, 1],
                    c="dodgerblue",
                    s=28,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.5,
                    label="Samples (boundary)",
                )
            )
        if safe_mask.any():
            safe_points = sample_np[safe_mask.cpu().numpy()]
            handles.append(
                self.ax_verification.scatter(
                    safe_points[:, 0], safe_points[:, 1], c="#2ca02c", s=20, alpha=0.7, edgecolors="white", linewidths=0.4, label="Samples (safe)"
                )
            )
        if unsafe_mask.any():
            unsafe_points = sample_np[unsafe_mask.cpu().numpy()]
            handles.append(
                self.ax_verification.scatter(
                    unsafe_points[:, 0], unsafe_points[:, 1], c="#d62728", s=20, alpha=0.7, edgecolors="white", linewidths=0.4, label="Samples (unsafe)"
                )
            )

        self.sample_handles = handles

    def _apply_verification_legend(self):
        """Apply legend to verification plot including optional sample handles."""
        handles = list(self.base_legend_handles)
        handles.extend(self.sample_handles)
        if handles:
            # Place the legend inside the axes at the bottom-right with a small inset
            # Use bbox_transform to anchor in axes coordinates so the legend stays
            # within the figure bounds when saving images.
            self.ax_verification.legend(
                handles=handles,
                loc="lower right",
                bbox_to_anchor=(1.00, 0.02),
                bbox_transform=self.ax_verification.transAxes,
                framealpha=0.8,
            )

    def update_figure(self, result):
        """
        Update the verification plot with new verification results.
        Batches both patch additions and canvas draws to minimize overhead.

        Args:
            result: Verification result object
        """
        if not result.isleaf():
            return

        # Extract region from result
        region = result.sample

        # Only handle simplicial regions for this plotter
        if not isinstance(region, SimplicialRegion):
            return

        # Create patch for the simplicial region
        vertices = region.vertices
        patch = Polygon(vertices, closed=True, alpha=0.7)

        # Color based on verification result
        if result.issat():
            # Verified region - green
            patch.set_facecolor("green")
            patch.set_edgecolor("darkgreen")
            self.verified_patches.append(patch)
        elif result.isunsat():
            # Counterexample region - red
            patch.set_facecolor("red")
            patch.set_edgecolor("darkred")
            self.counterexample_patches.append(patch)
        else:
            # Maybe/inconclusive region - yellow
            patch.set_facecolor("yellow")
            patch.set_edgecolor("orange")
            self.maybe_patches.append(patch)

        # Add to pending patches instead of immediately adding to plot
        self.pending_patches.append(patch)
        self.updates_since_last_draw += 1

        # Only update display periodically to avoid expensive operations
        if self.updates_since_last_draw >= self.update_interval:
            self._flush_pending_patches()

    def _flush_pending_patches(self):
        """Add all pending patches to the plot and redraw."""
        if self.pending_patches:
            for patch in self.pending_patches:
                self.ax_verification.add_patch(patch)
            self.pending_patches = []

        if self.updates_since_last_draw > 0:
            # Draw the canvas to update the figure (works with Agg backend)
            self.fig.canvas.draw()
            self.updates_since_last_draw = 0

    def refresh_plots(self, barrier_net=None, recompute_safe=False, samples=None, plot_samples=False):
        """
        Recompute visualization grids using the latest barrier network parameters and redraw plots.

        Args:
            barrier_net: Optional barrier network to adopt before refreshing (defaults to existing one).
            recompute_safe (bool): Recompute safe/unsafe grid as well. Set True if dynamics changed.
            samples (Tensor or ndarray, optional): Batch of points to overlay on verification plot.
            plot_samples (bool): If True and samples provided, scatter plot them.
        """
        if barrier_net is not None:
            self.barrier_net = barrier_net
            self.device = next(barrier_net.parameters()).device
            self.torch_translator = TorchTranslator(device=self.device)

        # Ensure any queued verification patches are added before we redraw
        self._flush_pending_patches()

        # Recompute grids with the up-to-date barrier function
        self._compute_barrier_grid()
        if recompute_safe:
            self._compute_safe_unsafe_grid()

        # Refresh scalar-field plots (these manage their own colorbars)
        self._init_barrier_plot()
        self._init_cbf_condition_plot()

        # Preserve verification patches by cloning them before clearing the axis
        existing_verified = [self._clone_polygon(patch) for patch in self.verified_patches]
        existing_counter = [self._clone_polygon(patch) for patch in self.counterexample_patches]
        existing_maybe = [self._clone_polygon(patch) for patch in self.maybe_patches]

        self.ax_verification.clear()
        self._init_verification_plot()

        self.verified_patches = []
        self.counterexample_patches = []
        self.maybe_patches = []
        self.sample_handles = []

        for patch in existing_verified:
            if patch is not None:
                self.ax_verification.add_patch(patch)
                self.verified_patches.append(patch)
        for patch in existing_counter:
            if patch is not None:
                self.ax_verification.add_patch(patch)
                self.counterexample_patches.append(patch)
        for patch in existing_maybe:
            if patch is not None:
                self.ax_verification.add_patch(patch)
                self.maybe_patches.append(patch)

        if plot_samples and samples is not None:
            try:
                self._plot_sample_scatter(samples)
            except Exception as exc:
                print(f"Warning: Could not plot sample scatter: {exc}")
                self.sample_handles = []
        else:
            self.sample_handles = []

        self._apply_verification_legend()

        # Draw to ensure figure state is current for downstream consumers (e.g., wandb.Image)
        self.fig.canvas.draw()
        self.updates_since_last_draw = 0

    def finalize(self):
        """Force a final draw to ensure all regions are displayed."""
        self._flush_pending_patches()

    def save_final_plot(self, filename="cbf_verification_final.png"):
        """Save the final verification plot."""
        try:
            # Ensure all updates are drawn before saving
            self.finalize()
            self.fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Final CBF verification plot saved to {filename}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

    def get_figure_for_wandb(self):
        """
        Get the current matplotlib figure for wandb logging.
        This creates a temporary image and returns it for wandb.Image.
        Forces a draw if there are pending updates.
        """
        import io

        from PIL import Image

        # Flush any pending patches and redraw
        self._flush_pending_patches()

        # Save figure to buffer with lower DPI and without tight layout for speed
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=100, bbox_inches=None)
        buf.seek(0)

        # Load as PIL Image
        img = Image.open(buf)
        return img

    def get_verification_statistics(self):
        """Get statistics about verification regions."""
        total_regions = len(self.verified_patches) + len(self.counterexample_patches) + len(self.maybe_patches)

        if total_regions == 0:
            return {
                "total_regions": 0,
                "verified_count": 0,
                "counterexample_count": 0,
                "maybe_count": 0,
                "verified_percentage": 0.0,
                "counterexample_percentage": 0.0,
                "maybe_percentage": 0.0,
            }

        return {
            "total_regions": total_regions,
            "verified_count": len(self.verified_patches),
            "counterexample_count": len(self.counterexample_patches),
            "maybe_count": len(self.maybe_patches),
            "verified_percentage": 100.0 * len(self.verified_patches) / total_regions,
            "counterexample_percentage": 100.0 * len(self.counterexample_patches) / total_regions,
            "maybe_percentage": 100.0 * len(self.maybe_patches) / total_regions,
        }


def create_cbf_verification_plotter(dynamics_model, barrier_net=None, resolution=300, alpha=1.0, update_interval=50):
    """
    Factory function to create a CBF verification plotter.

    Args:
        dynamics_model: CBF dynamical system
        barrier_net: Trained barrier function network (optional)
        resolution: Grid resolution for contour plots
        alpha: Class K function parameter for CBF condition
        update_interval: Number of regions to accumulate before updating display (default: 50)

    Returns:
        CBFVerificationPlotter instance or None if not 2D
    """
    if dynamics_model.input_dim != 2:
        print("CBF visualization plotter only supports 2D systems")
        return None

    # try:
    return CBFVerificationPlotter(dynamics_model, barrier_net, resolution, alpha=alpha, update_interval=update_interval)
    # except Exception as e:
    #     print(f"Failed to create CBF verification plotter: {e}")
    #     return None
