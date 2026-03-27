"""
Certified-Subspace Repair (CSR) Module

This module implements the Certified-Subspace Repair algorithm for
neural control barrier functions.

Key components:
1. CertificateDataCollector: Extracts A_L matrices from LBP verification
2. SubspaceAnalyzer: Performs subspace decomposition
3. SubspaceRepair: Implements the constrained repair optimization
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CertificateRegionData:
    """Data for a single certificate region (simplex/hyperrectangle)."""
    region: Any  # HyperrectangularRegion or SimplicialRegion
    A_L: np.ndarray  # Linear coefficient of lower bound: h_L(x) = A_L x + b_L
    b_L: float  # Bias term of lower bound
    is_verified: bool  # Whether this region passed verification


class CertificateDataCollector:
    """
    Collects certificate data (A_L matrices) during LBP verification.

    This class hooks into the verification process to extract the
    linear bound coefficients for each region.
    """

    def __init__(self):
        self.verified_regions: List[CertificateRegionData] = []
        self.failed_regions: List[CertificateRegionData] = []

    def collect_from_verification(self, verification_results: List[Any], network_linearizer: Any):
        """
        Collect A_L matrices from verification results.

        Args:
            verification_results: List of SampleResult objects from verify_cbf
            network_linearizer: CrownPartialLinearization instance
        """
        # NOTE: In practice, you would need to modify the verification
        # to store the A_L matrices during the verification process.
        # This is a placeholder showing the concept.
        pass

    def add_verified_region(self, region: Any, A_L: np.ndarray, b_L: float):
        """Add a verified region to the collection."""
        self.verified_regions.append(
            CertificateRegionData(region=region, A_L=A_L, b_L=b_L, is_verified=True)
        )

    def add_failed_region(self, region: Any, A_L: np.ndarray, b_L: float):
        """Add a failed region to the collection."""
        self.failed_regions.append(
            CertificateRegionData(region=region, A_L=A_L, b_L=b_L, is_verified=False)
        )

    def get_verified_A_L(self) -> List[np.ndarray]:
        """Get all A_L matrices from verified regions."""
        return [data.A_L for data in self.verified_regions]

    def get_failed_A_L(self) -> List[np.ndarray]:
        """Get all A_L matrices from failed regions."""
        return [data.A_L for data in self.failed_regions]


class SubspaceAnalyzer:
    """
    Performs certificate subspace analysis.

    Takes collected A_L matrices and performs:
    1. Covariance matrix computation
    2. Generalized eigenvalue decomposition
    3. Subspace selection
    """

    def __init__(self, d: int):
        """
        Initialize the subspace analyzer.

        Args:
            d: State space dimension
        """
        self.d = d
        self.M_V: Optional[np.ndarray] = None
        self.M_F: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.var_explained: Optional[np.ndarray] = None

    def compute_covariance_matrices(self, A_L_verified: List[np.ndarray], A_L_failed: List[np.ndarray]):
        """
        Compute M_V and M_F covariance matrices.

        Args:
            A_L_verified: List of A_L matrices from verified regions
            A_L_failed: List of A_L matrices from failed regions
        """
        n_verified = len(A_L_verified)
        n_failed = len(A_L_failed)

        if n_verified == 0 or n_failed == 0:
            raise ValueError("Need both verified and failed regions!")

        # Compute M_V: covariance of verified A_L
        self.M_V = np.zeros((self.d, self.d))
        for A_L in A_L_verified:
            A_L_reshaped = A_L.reshape(1, -1)  # Ensure 2D
            self.M_V += A_L_reshaped.T @ A_L_reshaped
        self.M_V /= n_verified

        # Compute M_F: covariance of failed A_L
        self.M_F = np.zeros((self.d, self.d))
        for A_L in A_L_failed:
            A_L_reshaped = A_L.reshape(1, -1)  # Ensure 2D
            self.M_F += A_L_reshaped.T @ A_L_reshaped
        self.M_F /= n_failed

        return self.M_V, self.M_F

    def generalized_eigenvalue_decomposition(self, regularization: float = 1e-6):
        """
        Solve the generalized eigenvalue problem: M_F w = λ M_V w.

        Args:
            regularization: Small value added to M_V for numerical stability

        Returns:
            eigenvalues, eigenvectors (sorted in descending order)
        """
        if self.M_V is None or self.M_F is None:
            raise ValueError("Call compute_covariance_matrices first!")

        # Regularize M_V for numerical stability
        M_V_reg = self.M_V + regularization * np.eye(self.d)

        # For generalized eigenvalue, we can use:
        # (M_V^{-1/2} M_F M_V^{-1/2}) v = λ v
        try:
            # Cholesky decomposition
            L_V = np.linalg.cholesky(M_V_reg)
            L_V_inv = np.linalg.inv(L_V)

            # Transform to standard eigenvalue problem
            M_tilde = L_V_inv @ self.M_F @ L_V_inv.T

            # Standard eigenvalue decomposition
            eigenvalues, eigenvectors_tilde = np.linalg.eigh(M_tilde)

            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors_tilde = eigenvectors_tilde[:, idx]

            # Transform back
            eigenvectors = L_V_inv.T @ eigenvectors_tilde

        except np.linalg.LinAlgError:
            # Fallback: use scipy if available, or just PCA on M_F
            print("Warning: Cholesky failed, using fallback method")
            eigenvalues, eigenvectors = np.linalg.eigh(self.M_F)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        # Compute variance explained
        total_var = np.sum(eigenvalues)
        self.var_explained = eigenvalues / total_var if total_var > 0 else eigenvalues

        return eigenvalues, eigenvectors

    def select_subspace(self, k: Optional[int] = None, var_threshold: float = 0.9):
        """
        Select the failure subspace.

        Args:
            k: Number of dimensions (if None, use var_threshold)
            var_threshold: Select enough components to explain this variance

        Returns:
            W_F: Basis for failure subspace (d x k)
            W_V: Basis for verified subspace (d x (d-k))
        """
        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError("Call generalized_eigenvalue_decomposition first!")

        # Determine k
        if k is None:
            cumulative_var = np.cumsum(self.var_explained)
            k = np.argmax(cumulative_var >= var_threshold) + 1
            k = min(k, self.d)

        W_F = self.eigenvectors[:, :k]
        W_V = self.eigenvectors[:, k:]

        return W_F, W_V

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the subspace analysis."""
        return {
            "d": self.d,
            "M_V": self.M_V,
            "M_F": self.M_F,
            "eigenvalues": self.eigenvalues,
            "var_explained": self.var_explained,
            "condition_number_V": np.linalg.cond(self.M_V) if self.M_V is not None else None,
            "condition_number_F": np.linalg.cond(self.M_F) if self.M_F is not None else None,
        }


class SubspaceRepair:
    """
    Performs subspace-constrained repair optimization.

    Takes the subspace decomposition and performs constrained repair.
    """

    def __init__(self, network: torch.nn.Module, W_F: np.ndarray, W_V: np.ndarray):
        """
        Initialize the subspace repair.

        Args:
            network: The neural CBF network
            W_F: Basis for failure subspace (d x k)
            W_V: Basis for verified subspace (d x (d-k))
        """
        self.network = network
        self.W_F = torch.tensor(W_F, dtype=torch.float32)
        self.W_V = torch.tensor(W_V, dtype=torch.float32)
        self.k = W_F.shape[1]

        # Store original weights for reference
        self.original_weights = {}
        for name, param in network.named_parameters():
            self.original_weights[name] = param.data.clone()

    def project_to_failure_subspace(self, param: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Project a parameter tensor to the failure subspace.

        This is a simplified implementation - the actual projection
        depends on the network architecture.

        Args:
            param: Parameter tensor
            layer_idx: Index of the layer (for architecture-aware projection)

        Returns:
            Projected parameter
        """
        # Simplified: For the last layer, we can project directly
        # For deeper layers, we need a more sophisticated approach

        # This is a placeholder - in practice, you would implement
        # architecture-aware projection
        return param

    def repair(
        self,
        failed_regions: List[Any],
        dynamics_model: Any,
        lr: float = 1e-4,
        num_iterations: int = 100,
        lambda_reg: float = 1e-4,
    ) -> torch.nn.Module:
        """
        Perform subspace-constrained repair.

        Args:
            failed_regions: List of failed regions to repair
            dynamics_model: Dynamical system model
            lr: Learning rate
            num_iterations: Number of optimization iterations
            lambda_reg: Regularization strength

        Returns:
            Repaired network
        """
        # Make a copy of the network
        repaired_network = type(self.network)(
            input_size=self.network.input_size,
            hidden_sizes=self.network.hidden_sizes,
            device=self.network.device
        )
        repaired_network.load_state_dict(self.network.state_dict())
        repaired_network.train()

        # Only optimize parameters in a way that affects the failure subspace
        # For simplicity, we start with the last layer only (like Chen et al.)
        # but we have the theoretical framework to do better

        # Get the last layer
        layers = list(replaced_network.children())
        if hasattr(replaced_network, 'fc_layers'):
            last_layer = replaced_network.fc_layers[-1]
        else:
            last_layer = layers[-1] if isinstance(layers[-1], torch.nn.Linear) else layers[-2]

        # Set up optimizer only for the last layer
        optimizer = torch.optim.Adam(last_layer.parameters(), lr=lr)

        # This is a simplified placeholder
        # In practice, you would:
        # 1. Compute LBP lower bounds on failed regions
        # 2. Maximize those lower bounds
        # 3. Use subspace projection to ensure verified regions stay unchanged

        print("Subspace repair placeholder - in practice, implement the constrained optimization")
        print(f"  - Failure subspace dimension: {self.k}")
        print(f"  - Verified subspace dimension: {self.W_V.shape[1]}")

        return repaired_network


# ============================================================================
# Convenience functions
# ============================================================================

def analyze_certificate_subspace(
    A_L_verified: List[np.ndarray],
    A_L_failed: List[np.ndarray],
    d: int,
    k: Optional[int] = None,
    var_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Convenience function: Analyze certificate subspace in one call.

    Args:
        A_L_verified: List of A_L matrices from verified regions
        A_L_failed: List of A_L matrices from failed regions
        d: State space dimension
        k: Number of failure subspace dimensions (auto-determined if None)
        var_threshold: Variance threshold for auto k

    Returns:
        Dictionary with analysis results
    """
    analyzer = SubspaceAnalyzer(d=d)
    M_V, M_F = analyzer.compute_covariance_matrices(A_L_verified, A_L_failed)
    eigenvalues, eigenvectors = analyzer.generalized_eigenvalue_decomposition()
    W_F, W_V = analyzer.select_subspace(k=k, var_threshold=var_threshold)

    return {
        "analyzer": analyzer,
        "M_V": M_V,
        "M_F": M_F,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "W_F": W_F,
        "W_V": W_V,
        "summary": analyzer.get_analysis_summary(),
    }


def demo_synthetic_data():
    """
    Demo the subspace analysis with synthetic data.

    This shows that the subspace decomposition works as expected.
    """
    d = 2  # 2D state space
    n_verified = 100
    n_failed = 20

    np.random.seed(42)

    # Synthetic verified A_L: centered around [1.0, 0.5]
    center_verified = np.array([1.0, 0.5])
    A_L_verified = [center_verified + 0.1 * np.random.randn(d) for _ in range(n_verified)]

    # Synthetic failed A_L: centered around [-0.5, 1.0]
    center_failed = np.array([-0.5, 1.0])
    A_L_failed = [center_failed + 0.3 * np.random.randn(d) for _ in range(n_failed)]

    # Analyze
    results = analyze_certificate_subspace(A_L_verified, A_L_failed, d=d)

    # Print results
    print("=" * 60)
    print("CERTIFIED SUBSPACE REPAIR - SYNTHETIC DEMO")
    print("=" * 60)

    print(f"\nCollected {n_verified} verified A_L matrices")
    print(f"Collected {n_failed} failed A_L matrices")

    print("\nM_V (verified covariance):")
    print(results["M_V"])

    print("\nM_F (failed covariance):")
    print(results["M_F"])

    print("\nGeneralized eigenvalues (descending):")
    for i, lam in enumerate(results["eigenvalues"]):
        print(f"  lambda_{i+1} = {lam:.4f}")

    print("\nTop eigenvector (failure subspace direction):")
    print(results["W_F"][:, 0])

    print("\nVariance explained by each component:")
    var_explained = results["summary"]["var_explained"]
    for i, var in enumerate(var_explained):
        print(f"  Component {i+1}: {var*100:.1f}%")

    if var_explained[0] > 0.6:
        print("\n✅ SUCCESS: Top component captures most failure variance!")
    else:
        print("\n⚠️  NOTE: Multiple components needed for failure subspace.")

    return results


if __name__ == "__main__":
    demo_synthetic_data()
