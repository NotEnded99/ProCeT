"""SOCP projection update — Equation (9) of the paper.

Implements the epigraph-form SOCP used by the ProCeT family to project the
repair gradient onto the certified-update cone:

    min_{Δθ, t}  t
    s.t.   ‖Δθ + η·g_f‖_2 ≤ t                  (epigraph tracking)
           ‖Δθ‖_2       ≤ ζ                    (norm trust region)

    Safe   (β^s  < 1):  ⟨∇_θ φ_θ, Δθ⟩ ≥ −ρ_Δ/(1+β^s)        (one-sided)
    Safe   (β^s  > 1):  |⟨∇_θ φ_θ, Δθ⟩| ≤ ρ_Δ/(1+β^s)       (two-sided)
    Unsafe (β^us < 1):  ⟨∇_θ ψ_θ, Δθ⟩ ≤  ρ_Δ/(1+β^us)       (one-sided)
    Unsafe (β^us > 1):  |⟨∇_θ ψ_θ, Δθ⟩| ≤ ρ_Δ/(1+β^us)      (two-sided)

where ``φ_θ`` is the LBP lower bound of ``L(x)`` on safe regions and ``ψ_θ``
is the LBP upper bound of ``h(x)`` on unsafe regions. Gradients are passed
RAW (NOT negated).
"""

import numpy as np
import torch


def socp_project_and_update_formula9(
    model, g_raw, grad_phi, grad_psi,
    rho_safe, rho_unsafe,
    beta_s, beta_us,
    lr=1e-3, delta_theta_norm_bound=0.01,
):
    """Solve the SOCP (Equation 8) and apply ``θ ← θ + Δθ`` in-place.

    Args:
        model: BarrierNN.
        g_raw: Repair gradient (loss-side), shape ``[num_params]``.
        grad_phi: ``∇_θ φ_θ`` for safe regions, shape ``[N_safe, num_params]``.
        grad_psi: ``∇_θ ψ_θ`` for unsafe regions, shape ``[N_unsafe, num_params]``.
        rho_safe: ``ρ_Δ^s`` margins for safe regions, shape ``[N_safe]``.
        rho_unsafe: ``ρ_Δ^us`` margins for unsafe regions, shape ``[N_unsafe]``.
        beta_s, beta_us: Empirical-remainder coefficients.
        lr: Step size ``η``.
        delta_theta_norm_bound: Trust-region radius ``ζ``.

    Returns:
        ``(g_raw_norm, update_norm)``.
    """
    import cvxpy as cp

    device = g_raw.device

    # 1. Flatten parameters
    params = [p for p in model.parameters() if p.requires_grad]
    theta_old = torch.nn.utils.parameters_to_vector(params)

    g_f = g_raw.detach().cpu().numpy().astype(np.float64)
    P = g_f.shape[0]

    A_safe = (
        grad_phi.detach().cpu().numpy().astype(np.float64)
        if grad_phi is not None and grad_phi.shape[0] > 0 else np.empty((0, P))
    )
    A_unsafe = (
        grad_psi.detach().cpu().numpy().astype(np.float64)
        if grad_psi is not None and grad_psi.shape[0] > 0 else np.empty((0, P))
    )

    # 2. Epigraph SOCP: min t  s.t. ‖Δθ + η·g_f‖ ≤ t
    delta_theta = cp.Variable(P)
    t = cp.Variable()
    objective = cp.Minimize(t)
    constraints = [cp.norm(delta_theta + lr * g_f, 2) <= t]
    constraints.append(cp.norm(delta_theta, 2) <= delta_theta_norm_bound)

    rhs_safe   = (rho_safe   / (1.0 + beta_s))  if A_safe.shape[0]   > 0 else None
    rhs_unsafe = (rho_unsafe / (1.0 + beta_us)) if A_unsafe.shape[0] > 0 else None

    # 3. Safe constraints — split by β^s
    if A_safe.shape[0] > 0:
        phi_dt = A_safe @ delta_theta
        if beta_s > 1.0:
            constraints.append(cp.abs(phi_dt) <= rhs_safe)
        else:
            constraints.append(phi_dt >= -rhs_safe)

    # 4. Unsafe constraints — split by β^us
    if A_unsafe.shape[0] > 0:
        psi_dt = A_unsafe @ delta_theta
        if beta_us > 1.0:
            constraints.append(cp.abs(psi_dt) <= rhs_unsafe)
        else:
            constraints.append(psi_dt <= rhs_unsafe)

    # 5. Solve — try ECOS, then SCS, finally fall back to a clipped gradient step
    prob = cp.Problem(objective, constraints)

    delta_theta_star = None
    try:
        prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-6, max_iters=200)
        if delta_theta.value is not None:
            delta_theta_star = delta_theta.value
    except cp.SolverError:
        delta_theta_star = None

    if delta_theta_star is None:
        try:
            prob.solve(solver=cp.SCS, max_iters=5000)
            if delta_theta.value is not None:
                delta_theta_star = delta_theta.value
        except Exception as e:
            print(f"  [SOCP] Warning: both solvers failed ({e}), using scaled gradient.")

    if delta_theta_star is None:
        delta_theta_star = -lr * g_f
        norm_dt = np.linalg.norm(delta_theta_star)
        if norm_dt > delta_theta_norm_bound:
            delta_theta_star = delta_theta_star * (delta_theta_norm_bound / norm_dt)

    # 6. Apply Δθ: θ_new = θ_old + Δθ
    delta_theta_tensor = torch.from_numpy(delta_theta_star).to(device=device, dtype=g_raw.dtype)
    update_norm = delta_theta_tensor.norm().item()
    g_norm = g_raw.norm().item()
    with torch.no_grad():
        theta_new = theta_old + delta_theta_tensor
        torch.nn.utils.vector_to_parameters(theta_new, params)

    return g_norm, update_norm
