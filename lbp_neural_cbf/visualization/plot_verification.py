import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

# Import region types for proper type checking
from ..regions import HyperrectangularRegion, SimplicialRegion


class DynamicsNetworkPlotter:
    """
    Class for visualizing 1D and 2D dynamics and verification results.
    """

    def __init__(self, dynamics_model, network, resolution=100, flat_mesh=True):
        """
        Initialize the plotter with a dynamics model.

        Args:
            dynamics_model: The dynamics model to visualize
            network: The neural network model
            resolution: Number of points to plot for the dynamics function
            flat_mesh: If True, draw mesh regions as flat patches on the floor (z=0) for 2D plots
        """
        self.dynamics_model = dynamics_model
        self.network = network
        self.resolution = resolution
        self.input_dim = dynamics_model.input_dim
        self.output_dim = dynamics_model.output_dim
        self.alpha = 0.4  # Transparency for the patches
        self.flat_mesh = flat_mesh  # New option for flat mesh visualization

        # Initialize figure based on input dimension
        if self.input_dim == 1:
            self._init_1d_plot()
        elif self.input_dim == 2:
            self._init_2d_plot()
        else:
            raise ValueError(f"Visualization only supports 1D and 2D inputs, got {self.input_dim}D")

        # Show plot
        plt.ion()  # Turn on interactive mode
        plt.tight_layout()
        self.fig.show()

    def _init_1d_plot(self):
        """Initialize plot for 1D dynamics"""
        # Create subplots horizontally (side by side) for each output dimension
        self.fig, self.axes = plt.subplots(1, self.output_dim, figsize=(10 * self.output_dim, 6))

        # If there's only one output dimension, axes is not an array
        if self.output_dim == 1:
            self.axes = [self.axes]

        for i, ax in enumerate(self.axes):
            ax.set_xlabel("Input")
            ax.set_ylabel(f"Output {i+1}")
            ax.set_title(f"Dynamics - Component {i+1}")

        # Track certified and uncertified regions
        self.certified_regions = [[] for _ in range(self.output_dim)]
        self.uncertified_regions = [[] for _ in range(self.output_dim)]

        # Initialize plot with the dynamics and network function
        self.plot_dynamics()
        self.plot_network()

    def _init_2d_plot(self):
        """Initialize plot for 2D dynamics"""
        # Create subplots in a grid layout instead of stacked
        # Determine grid dimensions based on output_dim
        if self.output_dim <= 3:
            rows, cols = 1, self.output_dim
        else:
            cols = min(3, self.output_dim)  # Max 3 columns
            rows = (self.output_dim + cols - 1) // cols  # Ceiling division

        self.fig = plt.figure(figsize=(6 * cols, 5 * rows))
        self.axes = []

        for i in range(self.output_dim):
            ax = self.fig.add_subplot(rows, cols, i + 1, projection="3d")
            ax.set_xlabel("Input 1")
            ax.set_ylabel("Input 2")
            ax.set_zlabel(f"Output {i+1}")
            ax.set_title(f"Dynamics - Component {i+1}")
            self.axes.append(ax)

        # Initialize the verification result containers
        self.certified_patches = [[] for _ in range(self.output_dim)]
        self.uncertified_patches = [[] for _ in range(self.output_dim)]

        # Initialize plot with the dynamics function
        self.plot_dynamics()
        self.plot_network()

    def plot_dynamics(self):
        """Plot the dynamics function."""
        if self.input_dim == 1:
            self._plot_1d_dynamics()
        elif self.input_dim == 2:
            self._plot_2d_dynamics()

    def _plot_1d_dynamics(self):
        """Plot 1D dynamics function"""
        domain = self.dynamics_model.input_domain
        x = np.linspace(domain[0][0], domain[0][1], self.resolution)

        # Reshape for the dynamics model input
        x_input = np.array([x_val for x_val in x]).reshape(-1, 1)
        y_outputs = np.array([self.dynamics_model(x_val.reshape(1, -1))[0] for x_val in x_input])

        # Plot each output dimension
        for i, ax in enumerate(self.axes):
            y = y_outputs[:, i] if y_outputs.ndim > 1 else y_outputs
            ax.plot(x, y, "b-", label="Dynamics")
            ax.legend()

    def _plot_2d_dynamics(self):
        """Plot 2D dynamics function as a surface"""
        domain = self.dynamics_model.input_domain

        # Create grid points for each dimension
        grid_points_per_dim = [np.linspace(domain[i][0], domain[i][1], self.resolution) for i in range(self.input_dim)]

        # Create mesh grid
        mesh = np.meshgrid(*grid_points_per_dim)

        # Reshape inputs for vectorized evaluation
        X = np.vstack(list(map(np.ravel, mesh)))
        Y = self.dynamics_model(X)

        # Plot each output dimension
        for i, ax in enumerate(self.axes):
            Z = Y[i].reshape(mesh[0].shape)
            # Use single color (blue) instead of colormap and remove colorbar
            surface = ax.plot_surface(mesh[0], mesh[1], Z, color="blue", alpha=self.alpha, linewidth=0, antialiased=True)

    def plot_network(self):
        """Plot the network function."""
        if self.input_dim == 1:
            self._plot_1d_network()
        elif self.input_dim == 2:
            self._plot_2d_network()

    def _plot_1d_network(self):
        """Plot 1D network function"""
        domain = self.dynamics_model.input_domain
        x = np.linspace(domain[0][0], domain[0][1], self.resolution)

        # Reshape for the network model input
        x_input = np.array([x_val for x_val in x]).reshape(-1, 1)
        y_outputs = np.array([self.network.evaluateWithoutMarabou([x_val])[0].flatten() for x_val in x_input])

        # Plot each output dimension
        for i, ax in enumerate(self.axes):
            y = y_outputs[:, i] if y_outputs.ndim > 1 else y_outputs
            ax.plot(x, y, "r-", label="network")
            ax.legend()

    def _plot_2d_network(self):
        """Plot 2D network function as a surface"""
        domain = self.dynamics_model.input_domain

        # Create grid points for each dimension
        grid_points_per_dim = [np.linspace(domain[i][0], domain[i][1], self.resolution) for i in range(self.input_dim)]

        # Create mesh grid
        mesh = np.meshgrid(*grid_points_per_dim)

        # Reshape inputs for vectorized evaluation
        X = np.vstack(list(map(np.ravel, mesh)))
        Y = [self.network.evaluateWithoutMarabou([X[:, i]])[0].flatten() for i in range(X.shape[-1])]
        Y = np.stack(Y, axis=1)

        # Plot each output dimension
        for i, ax in enumerate(self.axes):
            Z = Y[i].reshape(mesh[0].shape)
            # Use single color (blue) instead of colormap and remove colorbar
            surface = ax.plot_surface(mesh[0], mesh[1], Z, color="red", alpha=self.alpha, linewidth=0, antialiased=True)

        self.z_min, self.z_max = ax.get_zlim()

    def update_figure(self, result):
        """
        Update the figure with verification results.

        Args:
            result: The verification result object
        """
        # Extract the actual region from AugmentedSample if needed
        sample = result.sample
        if hasattr(sample, "region"):
            actual_region = sample.region
        else:
            actual_region = sample

        # Check if this is a simplicial region using proper type checking
        if isinstance(actual_region, SimplicialRegion):
            self._update_figure_simplicial(result)
        elif isinstance(actual_region, HyperrectangularRegion):
            # Original hyperrectangular logic
            center = actual_region.center
            if len(center) != self.input_dim:  # Check dimension match
                return

            # Choose visualization based on input dimension
            if self.input_dim == 1:
                self._update_1d_figure(result)
            elif self.input_dim == 2:
                self._update_2d_figure(result)
        else:
            # Fallback: try to detect region type by attributes
            if hasattr(actual_region, "center") and hasattr(actual_region, "radius"):
                # Likely hyperrectangular region
                center = actual_region.center
                if len(center) == self.input_dim:
                    if self.input_dim == 1:
                        self._update_1d_figure(result)
                    elif self.input_dim == 2:
                        self._update_2d_figure(result)
            elif hasattr(actual_region, "vertices"):
                # Likely simplicial region
                self._update_figure_simplicial(result)
            return

    def _update_1d_figure(self, result):
        """
        Update the 1D figure with verification results by adding colored rectangles.

        For certified regions (result.issat()), green rectangles are added.
        For counterexample regions (result.isunsat()), red rectangles are added.
        The rectangle width corresponds to the input domain of the region,
        and the height covers the range of output values within that region.

        Args:
            result: Verification result object containing sample information
                    (center point, radius, and output dimension)
        """
        # Extract the actual region from AugmentedSample if needed
        sample = result.sample
        if hasattr(sample, "region"):
            actual_region = sample.region
            output_dim = sample.output_dim if hasattr(sample, "output_dim") else actual_region.output_dim
        else:
            actual_region = sample
            output_dim = actual_region.output_dim

        center = actual_region.center
        radius = actual_region.radius
        f = self.dynamics_model(center).flatten()

        # Extract center and radius for 1D case
        x_center = center[0]
        x_radius = radius[0]

        # Calculate rectangle coordinates
        x_min = x_center - x_radius
        width = 2 * x_radius

        # Create rectangle patch for each output dimension
        ax = self.axes[output_dim]
        x_vals = np.linspace(x_min, x_min + width, self.resolution)
        y_vals = [self.dynamics_model(np.array([[x]]))[0][output_dim] for x in x_vals]
        height = max(y_vals) - min(y_vals)
        y_min = f[output_dim] - height / 2

        # Create rectangle patch with alpha transparency
        if result.issat():
            color = "green"
            rect = Rectangle((x_min, y_min), width, height, color=color, alpha=self.alpha, label="Certified")
            ax.add_patch(rect)
        elif result.isunsat():
            color = "red"
            rect = Rectangle((x_min, y_min), width, height, color=color, alpha=self.alpha, label="Counterexample")
            ax.add_patch(rect)
        else:
            return

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_2d_figure(self, result):
        """
        Update the 2D figure with verification results.

        Can draw either 3D rectangular prisms or flat 2D patches on the floor plane.

        Args:
            result: Verification result object containing sample information
                    (center point, radius, and output dimension)
        """
        # Extract the actual region from AugmentedSample if needed
        sample = result.sample
        if hasattr(sample, "region"):
            actual_region = sample.region
            output_dim = sample.output_dim if hasattr(sample, "output_dim") else actual_region.output_dim
        else:
            actual_region = sample
            output_dim = actual_region.output_dim

        center = actual_region.center
        radius = actual_region.radius

        # Extract center and radius for 2D case
        x_center, y_center = center
        x_radius, y_radius = radius

        # Define the grid points for the rectangle corners
        x_min, x_max = x_center - x_radius, x_center + x_radius
        y_min, y_max = y_center - y_radius, y_center + y_radius

        # Create a rectangle in the correct subplot
        ax = self.axes[output_dim]

        # Choose color based on result
        if result.issat():
            color = "green"
            alpha = self.alpha
        elif result.isunsat():
            color = "red"
            alpha = self.alpha
        else:
            return

        if self.flat_mesh:
            # Draw flat rectangular patch on the floor (z=0)
            self._draw_flat_rectangle(ax, x_min, x_max, y_min, y_max, color, alpha)
        else:
            # Draw 3D rectangular prism (original behavior)
            self._draw_3d_rectangle(ax, x_min, x_max, y_min, y_max, output_dim, color, alpha)

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_flat_rectangle(self, ax, x_min, x_max, y_min, y_max, color, alpha):
        """Draw a flat rectangular patch on the z=0 plane."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Create a flat rectangle at z=0
        z_floor = 0.0
        rectangle_vertices = [
            (x_min, y_min, z_floor),
            (x_max, y_min, z_floor),
            (x_max, y_max, z_floor),
            (x_min, y_max, z_floor),
        ]

        pc = Poly3DCollection([rectangle_vertices], alpha=alpha, facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_collection3d(pc)

    def _draw_flat_triangle(self, ax, vertices, color):
        """Draw a flat triangular patch on the z=0 plane."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Create a flat triangle at z=0
        z_floor = 0.0
        triangle_vertices = [
            (vertices[0][0], vertices[0][1], z_floor),
            (vertices[1][0], vertices[1][1], z_floor),
            (vertices[2][0], vertices[2][1], z_floor),
        ]

        pc = Poly3DCollection([triangle_vertices], alpha=self.alpha, facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_collection3d(pc)

    def _draw_3d_rectangle(self, ax, x_min, x_max, y_min, y_max, output_dim, color, alpha):
        """Draw a 3D rectangular prism (original behavior)."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Calculate the height based on the maximum dynamics value over the corners
        corner_values = [
            self.dynamics_model(np.array([x, y])).flatten()[output_dim] for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        ]
        z_min, z_max = min(corner_values), max(corner_values)

        # Define the vertices of the rectangular prism
        corners = np.array(
            [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ]
        )

        # Define the faces of the rectangular prism
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # bottom
            [corners[4], corners[5], corners[6], corners[7]],  # top
            [corners[0], corners[1], corners[5], corners[4]],  # front
            [corners[2], corners[3], corners[7], corners[6]],  # back
            [corners[0], corners[3], corners[7], corners[4]],  # left
            [corners[1], corners[2], corners[6], corners[5]],  # right
        ]

        # Create and add 3D polygon collection with proper z-height
        pc = Poly3DCollection([face for face in faces], alpha=alpha, facecolor=color, edgecolor="black")
        ax.add_collection3d(pc)

    def _update_figure_simplicial(self, result):
        """
        Update the figure with simplicial verification results.

        Args:
            result: Verification result object containing simplicial sample information
        """
        if self.input_dim == 1:
            self._update_1d_figure_simplicial(result)
        elif self.input_dim == 2:
            self._update_2d_figure_simplicial(result)

    def _update_1d_figure_simplicial(self, result):
        """
        Update the 1D figure with simplicial verification results.

        For 1D simplices (line segments), we draw a filled region that shows
        the range of function values over the line segment.
        """
        # Extract the actual region from AugmentedSample if needed
        sample = result.sample
        if hasattr(sample, "region"):
            actual_region = sample.region
            output_dim = sample.output_dim if hasattr(sample, "output_dim") else actual_region.output_dim
        else:
            actual_region = sample
            output_dim = actual_region.output_dim

        vertices = actual_region.vertices  # Shape: (2, 1) for 1D

        # Extract the x-coordinates of the line segment endpoints
        x_coords = vertices[:, 0]
        x_min, x_max = min(x_coords), max(x_coords)

        # Sample multiple points along the line segment to get function range
        n_samples = 20
        x_samples = np.linspace(x_min, x_max, n_samples)
        y_vals = [self.dynamics_model(np.array([x]))[output_dim] for x in x_samples]
        y_min, y_max = min(y_vals), max(y_vals)

        # Create a filled region showing the function range over this simplex
        ax = self.axes[output_dim]

        # Choose color based on result
        if result.issat():
            color = "green"
            label = "Certified"
        elif result.isunsat():
            color = "red"
            label = "Counterexample"
        else:
            color = "orange"
            label = "Maybe"

        # Fill the region between y_min and y_max over the x interval
        ax.fill_between([x_min, x_max], y_min, y_max, color=color, alpha=self.alpha, label=label)

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_2d_figure_simplicial(self, result):
        """
        Update the 2D figure with simplicial verification results.

        For 2D simplices (triangles), we create a proper triangular region
        showing the function values over the triangle.
        """
        # Extract the actual region from AugmentedSample if needed
        sample = result.sample
        if hasattr(sample, "region"):
            actual_region = sample.region
            output_dim = sample.output_dim if hasattr(sample, "output_dim") else actual_region.output_dim
        else:
            actual_region = sample
            output_dim = actual_region.output_dim

        vertices = actual_region.vertices  # Shape: (3, 2) for 2D triangles

        # Choose color based on result
        if result.issat():
            color = "green"
        elif result.isunsat():
            color = "red"
        else:
            return

        ax = self.axes[output_dim]

        if self.flat_mesh:
            self._draw_flat_triangle(ax, vertices, color)
        else:
            # Sample points inside the triangle to get function value range
            n_samples = 50
            sample_points = actual_region.sample_uniform(n_samples)
            z_vals = [self.dynamics_model(point)[output_dim] for point in sample_points]
            z_min, z_max = min(z_vals), max(z_vals)

            # Also evaluate at vertices for more accuracy
            vertex_z_vals = [self.dynamics_model(vertex)[output_dim] for vertex in vertices]
            z_min = min(z_min, min(vertex_z_vals))
            z_max = max(z_max, max(vertex_z_vals))

            # Create triangular faces for the 3D visualization
            # We'll create a "thick" triangle by making it extend from z_min to z_max

            # Bottom triangle at z_min
            bottom_triangle = [(vertices[i][0], vertices[i][1], z_min) for i in range(3)]

            # Top triangle at z_max
            top_triangle = [(vertices[i][0], vertices[i][1], z_max) for i in range(3)]

            # Side faces connecting bottom and top triangles
            side_faces = []
            for i in range(3):
                j = (i + 1) % 3
                # Create a quadrilateral face
                face = [
                    (vertices[i][0], vertices[i][1], z_min),
                    (vertices[j][0], vertices[j][1], z_min),
                    (vertices[j][0], vertices[j][1], z_max),
                    (vertices[i][0], vertices[i][1], z_max),
                ]
                side_faces.append(face)

            # Combine all faces
            faces = [bottom_triangle, top_triangle] + side_faces

            # Use Poly3DCollection for 3D triangular region
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            pc = Poly3DCollection(
                faces,
                alpha=self.alpha,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_collection3d(pc)

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_simplicial_mesh_outline(self, mesh):
        """
        Plot the outline of a simplicial mesh to show the domain decomposition.

        Args:
            mesh: SimplicialMesh object
        """
        if self.input_dim == 1:
            self._plot_1d_mesh_outline(mesh)
        elif self.input_dim == 2:
            self._plot_2d_mesh_outline(mesh)

    def _plot_1d_mesh_outline(self, mesh):
        """Plot 1D simplicial mesh outline (vertical lines at simplex boundaries)."""
        for ax in self.axes:
            for simplex in mesh.simplices:
                vertices = simplex.vertices[:, 0]  # x-coordinates
                for x in vertices:
                    ax.axvline(x, color="gray", alpha=0.3, linestyle="--", linewidth=0.5)

    def _plot_2d_mesh_outline(self, mesh):
        """Plot 2D simplicial mesh outline (triangle edges)."""
        for ax in self.axes:
            for simplex in mesh.simplices:
                vertices = simplex.vertices
                # Draw triangle edges
                for i in range(3):
                    j = (i + 1) % 3
                    ax.plot(
                        [vertices[i][0], vertices[j][0]],
                        [vertices[i][1], vertices[j][1]],
                        "gray",
                        alpha=0.3,
                        linewidth=0.5,
                        linestyle="--",
                    )


class SimplicialDynamicsNetworkPlotter:
    """
    Simplified wrapper for plotting simplicial meshes.

    This provides backward compatibility for code that expects the old
    SimplicialDynamicsNetworkPlotter interface.
    """

    def __init__(self, dynamics_model, network, mesh):
        """
        Initialize the simplicial plotter.

        Args:
            dynamics_model: The dynamics model
            network: The neural network model
            mesh: The simplicial mesh to visualize
        """
        self.dynamics = dynamics_model
        self.network = network
        self.mesh = mesh

    def update_figure(self, result):
        """
        Update the figure with verification results.

        Args:
            result: The verification result object
        """
        # Placeholder for visualization update
        # In a full implementation, this would update the plot with new verification results
        pass

    def save_final_plot(self, filename="simplicial_verification_final.png"):
        """
        Save the final verification plot.

        Args:
            filename: Name of the file to save the plot to
        """
        # Placeholder for saving plot
        # In a full implementation, this would save the current visualization state
        try:
            print(f"Placeholder: Would save simplicial verification plot as {filename}")
        except Exception as e:
            print(f"Could not save plot: {e}")
