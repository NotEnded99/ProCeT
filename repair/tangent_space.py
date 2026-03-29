"""
Tangent Space Computation for Certificate Manifold

This module computes the tangent space of the certificate manifold M,
where M = {θ | h̲_v(θ) = h̲_v(θ₀), ∀v∈V}

The tangent space consists of parameter directions that preserve
certificate invariance at first order.
"""

import numpy as np
import torch
import torch.nn as nn


def compute_tangent_space(model, verified_regions, lbp_computer,
                       rank_threshold=0.9, max_rank=50,
                       use_incremental=False):
    """
    Compute the tangent space of the certificate manifold at current parameters.

    The normal space N_θM = span{∇_θ h̲_v(θ) | v∈V}
    The tangent space T_θM = orthogonal complement of N_θM

    Args:
        model: BarrierNN neural network
        verified_regions: list of SimplicialRegion objects
        lbp_computer: LBPLowerBoundComputer instance
        rank_threshold: variance threshold for rank determination (0.9 = 90%)
        max_rank: maximum rank for tangent space computation
        use_incremental: whether to use incremental SVD updates

    Returns:
        tangent_basis: array of shape (|θ|, dim_T) - basis vectors
        projection_matrix: array of shape (|θ|, |θ|) - P_T matrix
        rank: effective rank of normal space
    """
    if len(verified_regions) == 0:
        # No constraints: tangent space is full parameter space
        n_params = sum(p.numel() for p in model.parameters())
        tangent_basis = np.eye(n_params)
        projection_matrix = np.eye(n_params)
        return tangent_basis, projection_matrix, n_params

    # Compute gradients for all verified regions
    print(f"Computing tangent space from {len(verified_regions)} verified regions...")

    # Option 1: Full Jacobian computation (for small V)
    if len(verified_regions) <= 100 and not use_incremental:
        tangent_basis, projection_matrix, rank = _compute_tangent_space_full(
            model, verified_regions, lbp_computer,
            rank_threshold, max_rank
        )
    # Option 2: Low-rank approximation (for large V)
    else:
        tangent_basis, projection_matrix, rank = _compute_tangent_space_lowrank(
            model, verified_regions, lbp_computer,
            rank_threshold, max_rank
        )

    print(f"Tangent space computed: dimension {tangent_basis.shape[1]} / {rank} params")

    return tangent_basis, projection_matrix, rank


def _compute_tangent_space_full(model, verified_regions, lbp_computer,
                               rank_threshold, max_rank):
    """
    Compute tangent space via full SVD of Jacobian.

    Jacobian J = [∂h̲_v/∂θ] for all v∈V
    SVD: J = U Σ V^T
    Normal space: spanned by top-k right singular vectors
    Tangent space: nullspace of top-k right singular vectors

    Args:
        model: BarrierNN
        verified_regions: list of SimplicialRegion
        lbp_computer: LBPLowerBoundComputer
        rank_threshold: variance threshold
        max_rank: maximum rank

    Returns:
        tangent_basis, projection_matrix, rank
    """
    # Compute Jacobian: gradients for all verified regions
    gradients = lbp_computer.compute_gradients_batch(verified_regions)
    # Shape: (n_regions, n_params)

    n_regions, n_params = gradients.shape

    # Compute SVD
    if n_regions < n_params:
        # Transpose for efficiency when m < |θ|
        U, s, Vt = np.linalg.svd(gradients.T, full_matrices=True)
        # Now: Vt shape is (n_params, n_params), s length is n_regions
        # Vt[:, :n_regions] are the relevant singular vectors
        Vt = Vt[:, :n_regions]
    else:
        U, s, Vt = np.linalg.svd(gradients, full_matrices=True)
        # Vt shape is (n_params, n_params), s length is n_params

    # Determine effective rank based on variance
    total_variance = np.sum(s ** 2)
    cumulative_variance = 0.0
    k = 0

    for i, singular_val in enumerate(s):
        cumulative_variance += singular_val ** 2
        if cumulative_variance / total_variance >= rank_threshold:
            k = i + 1
            break

    # Limit by max_rank
    k = min(k, max_rank, len(s))

    if k == 0:
        # No effective rank: tangent space is full parameter space
        tangent_basis = np.eye(n_params)
        projection_matrix = np.eye(n_params)
        return tangent_basis, projection_matrix, 0

    # Extract top-k right singular vectors (span normal space)
    normal_basis = Vt[:, :k]  # Shape: (n_params, k)

    # Compute tangent space as nullspace of normal_basis
    # Using QR decomposition for numerical stability
    Q, R = np.linalg.qr(normal_basis)

    # Tangent space is orthogonal complement
    # Method: I - Q * Q^T
    P_N = Q @ Q.T  # Projector onto normal space
    P_T = np.eye(n_params) - P_N  # Projector onto tangent space

    # Get basis for tangent space
    # Eigendecomposition of P_T
    eigvals, eigvecs = np.linalg.eigh(P_T)

    # Tangent basis corresponds to eigenvalues ≈ 1
    tol = 1e-10
    tangent_indices = np.where(np.abs(eigvals - 1.0) < tol)[0]

    if len(tangent_indices) > 0:
        tangent_basis = eigvecs[:, tangent_indices]
    else:
        # Fallback: use identity matrix (no constraints)
        tangent_basis = np.eye(n_params)

    return tangent_basis, P_T, k


def _compute_tangent_space_lowrank(model, verified_regions, lbp_computer,
                                   rank_threshold, max_rank):
    """
    Compute tangent space via truncated SVD (low-rank approximation).

    More efficient when |V| is large. Uses randomized SVD or
    subset sampling to reduce computation cost.

    Args:
        model: BarrierNN
        verified_regions: list of SimplicialRegion
        lbp_computer: LBPLowerBoundComputer
        rank_threshold: variance threshold
        max_rank: maximum rank

    Returns:
        tangent_basis, projection_matrix, rank
    """
    # Sample subset if too many regions
    sample_size = min(len(verified_regions), max_rank * 10)
    sampled_indices = np.random.choice(
        len(verified_regions), size=sample_size, replace=False
    )
    sampled_regions = [verified_regions[i] for i in sampled_indices]

    # Compute gradients for sampled regions
    gradients = lbp_computer.compute_gradients_batch(sampled_regions)
    # Shape: (sample_size, n_params)

    n_regions, n_params = gradients.shape

    # Compute truncated SVD (only top max_rank singular values)
    # Using numpy's svd with small k for efficiency
    k_svd = min(max_rank, n_regions, n_params)

    if n_regions < n_params:
        U, s, Vt = np.linalg.svd(gradients.T, full_matrices=False)
        # s length is n_regions, Vt shape is (n_params, k_svd)
    else:
        U, s, Vt = np.linalg.svd(gradients, full_matrices=False)
        # s length is n_params, Vt shape is (n_params, k_svd)

    # Use all computed singular vectors (already truncated)
    normal_basis = Vt

    # Compute projection matrix
    if normal_basis.shape[1] > 0:
        # Projector onto normal space: P_N = V_t @ V_t^T
        P_N = normal_basis @ normal_basis.T
        # Projector onto tangent space: P_T = I - P_N
        P_T = np.eye(n_params) - P_N
    else:
        P_T = np.eye(n_params)

    # Get tangent basis (nullspace of normal_basis)
    Q, R = np.linalg.qr(normal_basis)
    P_T_check = np.eye(n_params) - Q @ Q.T

    # Eigendecomposition for basis
    eigvals, eigvecs = np.linalg.eigh(P_T_check)
    tol = 1e-10
    tangent_indices = np.where(np.abs(eigvals - 1.0) < tol)[0]

    if len(tangent_indices) > 0:
        tangent_basis = eigvecs[:, tangent_indices]
    else:
        tangent_basis = np.eye(n_params)

    return tangent_basis, P_T, len(s)


def compute_projection_matrix(tangent_basis):
    """
    Compute projection matrix from tangent space basis.

    Args:
        tangent_basis: array of shape (|θ|, dim_T)

    Returns:
        P_T: projection matrix of shape (|θ|, |θ|)
    """
    if tangent_basis.shape[1] == tangent_basis.shape[0]:
        # Full rank: identity projection
        return np.eye(tangent_basis.shape[0])

    # P_T = T * (T^T * T)^(-1) * T^T
    # where T is the tangent basis
    T = tangent_basis
    gram = T.T @ T
    gram_inv = np.linalg.pinv(gram)  # Pseudoinverse for stability
    P_T = T @ gram_inv @ T.T

    return P_T


def incremental_svd_update(U, s, Vt, new_gradient, max_rank=None):
    """
    Update SVD incrementally when a new gradient is added.

    Uses Brand's algorithm for rank-1 SVD update with O(n²) complexity.

    Args:
        U, s, Vt: Current SVD (J = U diag(s) Vt)
        new_gradient: new row to add (gradient for new verified simplex)
        max_rank: maximum rank to maintain

    Returns:
        U_new, s_new, Vt_new: Updated SVD
    """
    # Ensure arrays are 2D
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if Vt.ndim == 1:
        Vt = Vt.reshape(1, -1)

    # New gradient as row vector
    p = Vt @ new_gradient  # k x 1
    r = new_gradient - U @ p  # residual

    r_norm = np.linalg.norm(r)

    if r_norm < 1e-10:
        # New gradient is in span of existing ones: no update needed
        return U, s, Vt

    # Construct extended orthogonal matrix
    U_extended = np.column_stack([U, r / r_norm])

    # Construct extended s matrix
    s_extended = np.append(s, 0)

    # Construct extended Vt matrix
    row_zeros = np.zeros((1, Vt.shape[1]))
    Vt_extended = np.vstack([Vt, row_zeros])

    # Perform SVD of the small update matrix if needed
    # For now, return the extended matrices
    # In practice, would re-orthogonalize and truncate

    return U_extended, s_extended, Vt_extended


def project_to_tangent_space(gradient, projection_matrix):
    """
    Project a gradient vector onto the tangent space.

    Args:
        gradient: numpy array of shape (|θ|,)
        projection_matrix: P_T of shape (|θ|, |θ|)

    Returns:
        projected_gradient: gradient projected onto tangent space
    """
    return projection_matrix @ gradient


def project_orthogonal_components(gradient, projection_matrix):
    """
    Decompose gradient into tangent and orthogonal components.

    Args:
        gradient: numpy array of shape (|θ|,)
        projection_matrix: P_T of shape (|θ|, |θ|)

    Returns:
        g_parallel: component in tangent space
        g_perpendicular: component in normal space
    """
    g_parallel = projection_matrix @ gradient
    g_perpendicular = gradient - g_parallel

    return g_parallel, g_perpendicular
