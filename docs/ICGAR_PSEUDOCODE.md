# ICGAR Pseudocode: Iterative Certificate-Gradient Aligned Refinement

**Based on ICGAR_FINAL_PROPOSAL.md**
**Date**: 2026-03-29

---

## Overview

This document provides detailed pseudocode for the ICGAR method, which repairs Neural Control Barrier Functions (NCBFs) while preserving verified-region invariance through manifold-constrained optimization.

---

## Algorithm 1: Main ICGAR Repair Loop

```
Algorithm: ICGAR_REPAIR(h(x; θ₀), V, F, options)
    # h(x; θ): Neural CBF network with parameters θ
    # V: Set of verified simplices (where LBP lower bound ≥ 0)
    # F: Set of failed simplices (where LBP lower bound < 0)
    # options: {η, λ, T_max, K_verify, α_schedule, α_params}

    Input:
        - h(x; θ₀): Initial neural CBF (already partially verified)
        - V = {v₁, v₂, ..., v_m}: Verified simplices
        - F = {f₁, f₂, ..., f_n}: Failed simplices to repair
        - η: Learning rate (default: 1e-3)
        - λ: L2 regularization weight (default: 1e-4)
        - T_max: Maximum iterations (default: 1000)
        - K_verify: Verification frequency (default: 10)
        - α_schedule: Type of α(t) schedule (default: 'exponential_decay')
        - α_params: Parameters for α(t) schedule (default: {τ=50, α_0=1.0})

    Output:
        - h(x; θ*): Repaired neural CBF
        - metrics: Dictionary tracking convergence, invariance, etc.

    # =========================================
    # PHASE 1: Precompute Tangent Space
    # =========================================
    # Compute the tangent space T_θ₀M of the certificate manifold
    # This represents directions in parameter space that preserve invariance

    Print("Phase 1: Computing certificate manifold tangent space...")

    if |V| > 0:
        T_θ₀M, P_T ← COMPUTE_TANGENT_SPACE(h, V, θ₀, options)
    else:
        # No verified regions to preserve
        T_θ₀M ← ℝ^{|θ₀|}  # Full parameter space
        P_T ← Identity matrix of size |θ₀| × |θ₀|

    # =========================================
    # PHASE 2: Initialize Optimization
    # =========================================

    θ ← θ₀  # Current parameters
    t ← 0   # Iteration counter
    L_prev ← ∞  # Previous loss
    L_history ← []  # Track loss over iterations
    F_prev ← |F|   # Number of failed simplices
    invariance_violations ← 0  # Count how often we violate invariance

    Print(f"Phase 2: Starting repair with {len(F)} failed simplices...")
    Print(f"Initial tangent space dimension: {dim(T_θ₀M)} / {|θ₀|}")

    # =========================================
    # PHASE 3: Main Optimization Loop
    # =========================================

    repeat until convergence or t > T_max:

        # ----------------------
        # Step 3.1: Compute Repair Loss
        # ----------------------
        # Loss consists of two components:
        # 1. Hinge loss on failed regions (push LBP lower bound to ≥ 0)
        # 2. L2 regularization (keep parameters close to initial)

        L_repair ← 0
        for each f in F:
            h̲_f ← LBP_LOWER_BOUND(h(x; θ), f)  # LBP lower bound over simplex f
            L_repair ← L_repair + max(0, -h̲_f)  # Hinge loss

        L_reg ← λ * ||θ - θ₀||²  # L2 regularization

        L ← L_repair + L_reg

        L_history.append(L)

        # ----------------------
        # Step 3.2: Check Convergence
        # ----------------------

        if |L - L_prev| < ε and t > 100:
            Print(f"Converged at iteration {t}")
            break

        L_prev ← L

        # ----------------------
        # Step 3.3: Compute Loss Gradient
        # ----------------------
        # Compute gradient of loss with respect to parameters

        g ← ∇_θ L  # Full gradient (size: |θ| × 1)

        # ----------------------
        # Step 3.4: Project to Tangent Space
        # ----------------------
        # Decompose gradient into:
        # - g_∥: Component tangent to manifold (preserves invariance)
        # - g_⊥: Component normal to manifold (may violate invariance)

        g_∥ ← P_T * g  # Project onto tangent space
        g_⊥ ← g - g_∥  # Orthogonal complement

        # ----------------------
        # Step 3.5: Adaptive α(t) Schedule
        # ----------------------
        # α_t controls trade-off between:
        # - α=0: Strict invariance (only use g_∥)
        # - α=1: No invariance (use full gradient)
        # - α∈(0,1): Allow controlled violations

        α_t ← COMPUTE_ALPHA(t, |F|, F_prev, L, L_history, α_schedule, α_params)

        F_prev ← |F|  # Store for next iteration

        # ----------------------
        # Step 3.6: Combine Gradients
        # ----------------------
        # Blend tangent and normal components based on α_t

        g_α ← g_∥ + α_t * g_⊥

        # ----------------------
        # Step 3.7: Parameter Update
        # ----------------------
        # Gradient descent step

        θ ← θ - η * g_α

        # ----------------------
        # Step 3.8: Periodic Verification
        # ----------------------
        # Re-verify to track progress and update failed set

        if t % K_verify == 0:

            # Run LBP verification on current network
            V_new, F_new ← VERIFY_WITH_LBP(h(x; θ), state_space)

            # Check if we violated invariance on previously verified simplices
            invariance_violations ← COUNT_VIOLATIONS(V, V_new, h(x; θ))

            Print(f"Iter {t}: Loss={L:.4f}, Failed={|F_new|}, "
                  f"Violations={invariance_violations}, α={α_t:.4f}")

            F ← F_new  # Update failed simplices set

            # Early termination if all simplices verified
            if |F_new| == 0:
                Print(f"All simplices verified at iteration {t}")
                break

        # Increment iteration counter
        t ← t + 1

    # =========================================
    # PHASE 4: Final Verification and Output
    # =========================================

    Print("Phase 4: Running final verification...")
    V_final, F_final ← VERIFY_WITH_LBP(h(x; θ), state_space)

    # Compile metrics
    metrics ← {
        'iterations': t,
        'final_loss': L,
        'initial_failed': |F₀|,
        'final_failed': |F_final|,
        'invariance_violations': invariance_violations,
        'loss_history': L_history,
        'converged': |F_final| == 0
    }

    Print(f"Repair complete: {metrics['final_failed']}/{metrics['initial_failed']} "
          f"simplices repaired ({100*(1-metrics['final_failed']/metrics['initial_failed']):.1f}%)")

    return h(x; θ), metrics
```

---

## Algorithm 2: Tangent Space Computation

```
Algorithm: COMPUTE_TANGENT_SPACE(h, V, θ, options)
    # Computes the tangent space of the certificate manifold M at θ
    # M = {θ' | h̲_v(θ') = h̲_v(θ), ∀v∈V}

    Input:
        - h: Neural CBF network
        - V = {v₁, v₂, ..., v_m}: Verified simplices
        - θ: Current parameter values
        - options: {rank_threshold, use_incremental, max_rank}

    Output:
        - T_θM: Tangent space (represented as basis vectors)
        - P_T: Projection matrix onto tangent space

    # =========================================
    # Strategy: Compute Normal Space First
    # =========================================
    # The normal space N_θM = span{∇_θ h̲_v(θ) | v∈V}
    # The tangent space is the orthogonal complement

    # ------------------------------------------------
    # Method A: Full Computation (when |V| is small)
    # ------------------------------------------------

    if |V| ≤ 100 and use_incremental == false:

        Print("Computing tangent space via full Jacobian...")

        # Step A.1: Compute gradients for all verified simplices
        J ← []  # Jacobian matrix (will be m × |θ|)

        for each v in V:
            # Compute gradient of LBP lower bound with respect to θ
            g_v ← ∇_θ LBP_LOWER_BOUND(h(x; θ), v)
            J.append(g_v)

        J ← stack(J)  # Stack gradients into matrix

        # Step A.2: Compute SVD of Jacobian
        # J = U Σ V^T
        # Columns of V corresponding to non-zero singular values span N_θM

        U, Σ, V^T ← SVD(J, full_matrices=False)

        # Step A.3: Determine effective rank
        # Keep singular values that explain ≥ rank_threshold variance

        total_variance ← Σ₁² + Σ₂² + ... + Σ_r²
        cumulative_variance ← 0
        k ← 0

        for i in 1 to rank(Σ):
            cumulative_variance ← cumulative_variance + Σ_i²
            if cumulative_variance / total_variance ≥ rank_threshold:
                k ← i
                break

        Print(f"Effective rank: {k} / {rank(Σ)} (explains {rank_threshold*100:.0f}% variance)")

        # Step A.4: Extract normal space basis
        # Top-k right singular vectors span N_θM

        N_basis ← V^T[:, :k]  # Columns 1 to k of V^T

        # Step A.5: Compute tangent space basis
        # Tangent space = nullspace of N_basis^T

        T_basis ← NULLSPACE(N_basis^T)

        # Step A.6: Compute projection matrix
        # P_T = I - N_basis * (N_basis^T * N_basis)^(-1) * N_basis^T

        if k > 0:
            P_N ← N_basis * INVERSE(N_basis^T * N_basis) * N_basis^T  # Projector onto N
            P_T ← IdentityMatrix(|θ|) - P_N  # Projector onto T
        else:
            P_T ← IdentityMatrix(|θ|)  # No constraints

        return T_basis, P_T

    # ------------------------------------------------
    # Method B: Low-Rank Approximation (when |V| is large)
    #. This is the primary method for scalability
    # ------------------------------------------------

    else:

        Print("Computing tangent space via low-rank SVD approximation...")

        # Step B.1: Compute gradients subset (random sampling)
        # Only compute gradient for a subset of simplices

        if |V| > max_rank * 10:
            ℓ ← max_rank * 10  # Sample size
            V_sample ← RANDOM_SAMPLE(V, ℓ)
        else:
            V_sample ← V
            ℓ ← |V|

        # Step B.2: Compute gradients for sampled simplices
        J ← []
        for each v in V_sample:
            g_v ← ∇_θ LBP_LOWER_BOUND(h(x; θ), v)
            J.append(g_v)

        J ← stack(J)  # ℓ × |θ| matrix

        # Step B.3: Truncated SVD (compute only top-k singular values)
        # k is max_rank or determined by variance threshold

        U_k, Σ_k, V_k^T ← TRUNCATED_SVD(J, k=max_rank)

        # Step B.4: Tangent space is nullspace of V_k^T
        T_basis ← NULLSPACE(V_k^T)

        # Step B.5: Approximate projection matrix
        if k > 0:
            P_N ← V_k^T.T * INVERSE(V_k^T * V_k^T.T) * V_k^T
            P_T ← IdentityMatrix(|θ|) - P_N
        else:
            P_T ← IdentityMatrix(|θ|)

        Print(f"Low-rank approximation using rank {max_rank}")

        return T_basis, P_T
```

---

## Algorithm 3: Incremental Tangent Space Update

```
Algorithm: INCREMENTAL_SVD_UPDATE(U, Σ, V^T, new_simplex, θ, h)
    # Efficiently update SVD when a new verified simplex is added
    # Avoids recomputing full SVD from scratch

    Input:
        - U, Σ, V^T: Current SVD (J_prev = U Σ V^T)
        - new_simplex: New verified simplex to add
        - θ: Current parameter values
        - h: Neural CBF network

    Output:
        - U_new, Σ_new, V_new^T: Updated SVD

    # =========================================
    # Step 1: Compute gradient for new simplex
    # =========================================

    g_new ← ∇_θ LBP_LOWER_BOUND(h(x; θ), new_simplex)  # |θ| × 1 vector

    # =========================================
    # Step 2: Append new row to Jacobian
    # =========================================
    # J_new = [J_prev; g_new^T]  (stack as new row)

    # Current dimensions:
    #   J_prev: m × |θ|
    #   g_new: |θ| × 1
    #   J_new: (m+1) × |θ|

    # =========================================
    # Step 3: Update SVD using rank-1 update
    # =========================================
    # Use Brand's algorithm or similar for efficient rank-1 SVD update

    # Algorithm sketch (following Brand 2002):
    # 1. Compute p = U^T * g_new  (k × 1)
    # 2. Compute q = g_new - U * p  (|θ| × 1)
    # 3. Compute r = ||q||
    # 4. Compute K = [Σ  p; 0  r]  (k+1 × k+1 matrix)
    # 5. Compute SVD of K: Ũ Σ̃ Ṽ^T
    # 6. Update:
    #    U_new = [U  q/r] * Ũ
    #    Σ_new = Σ̃
    #    V_new^T = Ṽ^T * [V^T; 0]  # (0 is a row vector)

    p ← U.T * g_new
    q ← g_new - U * p
    r ← NORM(q)

    if r > ε:  # If new gradient adds new information
        # Construct extended orthonormal basis
        U_extended ← [U, q/r]  # |θ| × (k+1)

        # Construct K matrix
        K ← [[Σ, p],
             [zeros(1, k), r]]  # (k+1) × (k+1)

        # Compute SVD of small matrix K
        Ũ, Σ̃, Ṽ^T ← SVD(K)

        # Update components
        U_new ← U_extended * Ũ
        Σ_new ← Σ̃
        V_new^T ← Ṽ^T * [V^T; zeros(1, |θ|)]
    else:
        # New gradient is in span of existing gradients
        # No dimension increase needed
        U_new ← U
        Σ_new ← Σ
        V_new^T ← V^T

    return U_new, Σ_new, V_new^T
```

---

## Algorithm 4: α(t) Schedule Computation

```
Algorithm: COMPUTE_ALPHA(t, F_current, F_prev, L_current, L_history, schedule_type, params)
    # Computes adaptive α(t) for controlling invariance trade-off
    # α=0: strict invariance (only tangent component)
    # α=1: no invariance (full gradient)

    Input:
        - t: Current iteration
        - F_current: Current number of failed simplices
        - F_prev: Previous number of failed simplices
        - L_current: Current loss
        - L_history: History of losses
        - schedule_type: Type of schedule
        - params: Schedule-specific parameters

    Output:
        - α_t: Scalar in [0, 1]

    # =========================================
    # Schedule 1: Strict Invariance
    # =========================================
    # Never allow invariance violation

    if schedule_type == 'strict':
        return 0.0

    # =========================================
    # Schedule 2: Constant
    # =========================================
    # Fixed α throughout

    if schedule_type == 'constant':
        return params.get('alpha', 0.5)

    # =========================================
    # Schedule 3: Linear Ramp
    # =========================================
    # α starts at 0, increases linearly to max

    if schedule_type == 'linear_ramp':
        T_ramp ← params.get('T_ramp', 100)  # Duration of ramp
        alpha_max ← params.get('alpha_max', 1.0)

        α_t ← min(alpha_max, t / T_ramp)
        return α_t

    # =========================================
    # Schedule 4: Exponential Decay
    # =========================================
    # α starts high, decays to 0 (opposite of linear ramp)
    # Good when we want to be permissive early, strict later

    if schedule_type == 'exponential_decay':
        τ ← params.get('tau', 50)  # Time constant
        alpha_0 ← params.get('alpha_0', 1.0)

        α_t ← alpha_0 * (1 - exp(-t / τ))
        return α_t

    # =========================================
    # Schedule 5: Inverse Decay
    # =========================================
    # α starts at 1, decays like 1/t

    if schedule_type == 'inverse_decay':
        t_0 ← params.get('t_0', 1)

        α_t ← 1.0 / (1.0 + t / t_0)
        return α_t

    # =========================================
    # Schedule 6: Feedback-Based
    # =========================================
    # α adapts based on repair progress
    # More violations when making slow progress

    if schedule_type == 'feedback':
        alpha_0 ← params.get('alpha_0', 0.5)
        beta ← params.get('beta', 1.0)  # Sensitivity

        # Compute progress ratio (0 = no progress, 1 = all fixed)
        if F_prev > 0:
            progress ← (F_prev - F_current) / F_prev
        else:
            progress ← 0

        # α increases when progress is slow
        α_t ← alpha_0 * (1 - beta * progress)
        α_t ← max(0, min(1, α_t))  # Clamp to [0, 1]

        return α_t

    # =========================================
    # Schedule 7: Loss-Based
    # =========================================
    # α increases when loss plateaus

    if schedule_type == 'loss_based':
        window ← params.get('window', 10)  # Lookback window
        alpha_0 ← params.get('alpha_0', 0.1)
        alpha_max ← params.get('alpha_max', 0.9)

        if len(L_history) >= window:
            recent_losses ← L_history[-window:]
            loss_std ← STANDARD_DEVIATION(recent_losses)

            # If loss has plateaued (low std), increase α
            if loss_std < params.get('plateau_threshold', 0.01):
                α_t ← alpha_max
            else:
                α_t ← alpha_0
        else:
            α_t ← alpha_0

        return α_t

    # =========================================
    # Schedule 8: Cosine Schedule
    # =========================================
    # Smoothly varying α

    if schedule_type == 'cosine':
        T ← params.get('T', 100)  # Period
        alpha_min ← params.get('alpha_min', 0.0)
        alpha_max ← params.get('alpha_max', 1.0)

        α_t ← alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + cos(π * t / T))
        return α_t

    # =========================================
    # Default: Exponential decay
    # =========================================

    α_t ← params.get('alpha_0', 1.0) * (1 - exp(-t / 50))
    return α_t
```

---

## Algorithm 5: LBP Lower Bound Computation

```
Algorithm: LBP_LOWER_BOUND(h(x; θ), simplex)
    # Computes the Lipschitz-based propagation (LBP) lower bound
    # of a neural network over a simplex

    Input:
        - h(x; θ): Neural CBF network
        - simplex: Simplex defined by vertices {v₁, ..., v_d+1}

    Output:
        - h̲: Lower bound of h over the simplex

    # =========================================
    # Step 1: Forward Pass with LBP
    # =========================================
    # LBP propagates input bounds through the network

    # Initialize with simplex bounds
    # For simplex with vertices v₁, ..., v_d+1:
    #   lower_i = min(v₁[i], v₂[i], ..., v_d+1[i])
    #   upper_i = max(v₁[i], v₂[i], ..., v_d+1[i])

    bounds ← Compute_Bounds(simplex)  # List of (lower, upper) pairs

    # =========================================
    # Step 2: Propagate through each layer
    # =========================================

    for each layer in h.layers:

        if layer.type == 'Linear':
            # Linear layer: y = Wx + b
            # Lower bound: min(W * lower, W * upper) + b
            # Upper bound: max(W * lower, W * upper) + b

            new_bounds ← []
            for each output_unit j:
                contributions ← []
                for each input_unit i:
                    w_ji ← layer.weights[j, i]

                    if w_jb > 0:
                        # Positive weight: use lower bound
                        contributions.append(w_jb * bounds[i].lower)
                    else:
                        # Negative weight: use upper bound
                        contributions.append(w_jb * bounds[i].upper)

                y_lower ← sum(contributions) + layer.bias[j]
                y_upper ← sum(contributions) + layer.bias[j]  # Same for linear

                new_bounds.append((y_lower, y_upper))

            bounds ← new_bounds

        elif layer.type == 'ReLU':
            # ReLU: y = max(0, x)
            # Lower bound: max(0, lower)
            # Upper bound: max(0, upper)

            new_bounds ← []
            for (lower, upper) in bounds:
                y_lower ← max(0, lower)
                y_upper ← max(0, upper)
                new_bounds.append((y_lower, y_upper))

            bounds ← new_bounds

        elif layer.type == 'Tanh':
            # Tanh: y = tanh(x)
            # Monotonic increasing, so:
            # Lower bound: tanh(lower)
            # Upper bound: tanh(upper)

            new_bounds ← []
            for (lower, upper) in bounds:
                y_lower ← tanh(lower)
                y_upper ← tanh(upper)
                new_bounds.append((y_lower, y_upper))

            bounds ← new_bounds

        # Add other layer types as needed...

    # =========================================
    # Step 3: Return lower bound of output
    # =========================================

    h̲ ← bounds[0].lower  # Assume scalar output
    return h̲
```

---

## Algorithm 6: Gradient of LBP Lower Bound

```
Algorithm: GRADIENT_LBP_LOWER_BOUND(h, simplex, θ)
    # Computes ∇_θ LBP_LOWER_BOUND(h(x; θ), simplex)
    # The gradient of the LBP lower bound with respect to parameters

    Input:
        - h: Neural CBF network
        - simplex: Input simplex
        - θ: Current parameters

    Output:
        - ∇_θ h̲: Gradient vector (size |θ| × 1)

    # =========================================
    # Strategy: Backpropagation through LBP
    # =========================================
    # The LBP lower bound is a piecewise-linear function of θ
    # We can compute its gradient via LBP-aware backprop

    # =========================================
    # Step 1: Forward pass storing activations
    # =========================================

    activations ← []  # Store (lower, upper) for each layer
    vertices ← simplex.vertices  # {v₁, ..., v_d+1}

    # Initialize with bounds from simplex
    bounds ← Compute_Bounds(simplex)
    activations.append(bounds)

    # Forward propagate bounds
    current_bounds ← bounds
    for each layer in h.layers:
        current_bounds ← LBP_FORWARD_LAYER(layer, current_bounds)
        activations.append(current_bounds)

    # =========================================
    # Step 2: Initialize backward gradient
    # =========================================

    # We're computing gradient of the scalar output lower bound
    d_output ← 1.0  # ∂h̲ / ∂h̲ = 1

    # =========================================
    # Step 3: Backward propagate gradient
    # =========================================

    param_gradients ← []  # Will hold ∂h̟/∂θ for each parameter
    d_bounds ← d_output  # Gradient w.r.t output bounds

    # Iterate backwards through layers
    for l from len(h.layers) down to 1:
        layer ← h.layers[l]
        bounds ← activations[l]  # Bounds at this layer

        if layer.type == 'Linear':
            # For linear layer, gradient flows through weights and bias

            # Weight gradient: ∂h̲/∂W_jb = d_bounds * input_bound
            # where input_bound = lower or upper depending on sign of W_jb

            W_grad ← zeros_like(layer.weights)
            for each output_unit j:
                for each input_unit i:
                    w_jb ← layer.weights[j, i]
                    input_bound ← activations[l-1][i]

                    if w_jb > 0:
                        W_grad[j, i] ← d_bounds * input_bound.lower
                    else:
                        W_grad[j, i] ← d_bounds * input_bound.upper

            # Bias gradient: ∂h̲/∂b_j = d_bounds
            b_grad ← zeros_like(layer.bias)
            for each output_unit j:
                b_grad[j] ← d_bounds

            param_gradients.append({'W': W_grad, 'b': b_grad})

            # Propagate gradient to previous layer
            d_prev ← zeros(len(bounds))
            for each input_unit i:
                d_prev[i] ← d_bounds * sum_j W_grad[j, i]

            d_bounds ← d_prev

        elif layer.type == 'ReLU':
            # ReLU gradient:
            # If lower > 0: gradient passes through (active)
            # If upper < 0: gradient is zero (inactive)
            # Otherwise: gradient depends on which bound determines output

            d_prev ← zeros(len(bounds))
            for each unit i:
                lower, upper ← bounds[i]

                if lower > 0:
                    # Always active: gradient passes through
                    d_prev[i] ← d_bounds
                elif upper < 0:
                    # Always inactive: gradient is zero
                    d_prev[i] ← 0
                else:
                    # Potentially active: gradient passes with probability
                    # This is a simplification - exact gradient is tricky
                    d_prev[i] ← d_bounds * 0.5  # Approximate

            d_bounds ← d_prev

        elif layer.type == 'Tanh':
            # Tanh gradient: sech²(x) * incoming_gradient
            # For bounds, use gradient at the bound that determines output

            d_prev ← zeros(len(bounds))
            for each unit i:
                lower, upper ← bounds[i]

                # Gradient at lower and upper bounds
                grad_lower ← 1 - tanh(lower)²  # sech²(lower)
                grad_upper ← 1 - tanh(upper)²  # sech²(upper)

                # Use gradient at the binding bound
                # (simplification - exact requires more care)
                d_prev[i] ← d_bounds * max(grad_lower, grad_upper)

            d_bounds ← d_prev

    # =========================================
    # Step 4: Flatten gradients into single vector
    # =========================================

    ∇_θ h̲ ← []
    for each param_grad in reversed(param_gradients):
        ∇_θ h̲.extend(param_grad['W'].flatten())
        ∇_θ h̲.extend(param_grad['b'].flatten())

    return array(∇_θ h̲)
```

---

## Algorithm 7: Full ICGAR Pipeline with Verification Loop

```
Algorithm: ICGAR_PIPELINE(network, state_space, verification_config)
    # Complete pipeline: verification → repair → re-verification

    Input:
        - network: Initial neural CBF h(x; θ₀)
        - state_space: Domain over which to verify
        - verification_config: {simplex_size, max_iterations, ...}

    Output:
        - repaired_network: Final repaired network
        - results: Verification results at each stage

    # =========================================
    # Stage 0: Initial Verification
    # =========================================

    Print("Stage 0: Running initial verification...")

    V_0, F_0 ← VERIFY_WITH_LBP(network, state_space, verification_config)

    results ← {
        'initial': {
            'verified': |V_0|,
            'failed': |F_0|,
            'simplices': V_0 ∪ F_0
        }
    }

    Print(f"Initial verification: {results['initial']['verified']} verified, "
          f"{results['initial']['failed']} failed")

    # =========================================
    # Stage 1: Initial Repair
    # =========================================

    if results['initial']['failed'] > 0:

        Print("Stage 1: Running ICGAR repair...")

        # Configure ICGAR options
        icgar_options ← {
            ''η': 1e-3,
            'λ': 1e-4,
            'T_max': 1000,
            'K_verify': 10,
            'α_schedule': 'exponential_decay',
            'α_params': {'τ': 50, 'alpha_0': 1.0}
        }

        # Run ICGAR repair
        network_repaired, metrics ← ICGAR_REPAIR(
            network, V_0, F_0, icgar_options
        )

        results['repair_metrics'] ← metrics

        # =========================================
        # Stage 2: Post-Repair Verification
        # =========================================

        Print("Stage 2: Running post-repair verification...")

        V_1, F_1 ← VERIFY_WITH_LBP(network_repaired, state_space, verification_config)

        results['post_repair'] ← {
            'verified': |V_1|,
            'failed': |F_1|,
            'simplices': V_1 ∪ F_1
        }

        improvement ← results['post_repair']['verified'] - results['initial']['verified']
        total ← results['initial']['verified'] + results['initial']['failed']

        Print(f"Post-repair verification: {results['post_repair']['verified']} verified, "
              f"{results['post_repair']['failed']} failed")
        Print(f"Improvement: {improvement} simplices ({100*improvement/total:.1f}%)")

        # =========================================
        # Stage 3: Iterative Repair (if needed)
        # =========================================

        if results['post_repair']['failed'] > 0:
            Print("Stage 3: Running iterative repair rounds...")

            iteration ← 1
            max_rounds ← 5
            network_current ← network_repaired
            V_current ← V_1
            F_current ← F_1

            results['iterations'] ← []

            while |F_current| > 0 and iteration ≤ max_rounds:

                Print(f"Iteration {iteration}: Repairing remaining {len(F_current)} simplices...")

                # Adjust options for later iterations (more conservative)
                icgar_options_iter ← icgar_options.copy()
                icgar_options_iter['α_params']['tau'] ← 100  # Decay slower
                icgar_options_iter['η'] ← icgar_options['η'] * 0.5  # Smaller learning rate

                # Run repair
                network_new, metrics_iter ← ICGAR_REPAIR(
                    network_current, V_current, F_current, icgar_options_iter
                )

                # Verify
                V_new, F_new ← VERIFY_WITH_LBP(network_new, state_space, verification_config)

                # Record results
                results['iterations'].append({
                    'iteration': iteration,
                    'verified_before': len(V_current),
                    'failed_before': len(F_current),
                    'verified_after': len(V_new),
                    'failed_after': len(F_new),
                    'improvement': len(V_new) - len(V_current),
                    'metrics': metrics_iter
                })

                # Update for next iteration
                network_current ← network_new
                V_current ← V_new
                F_current ← F_new
                iteration ← iteration + 1

        repaired_network ← network_current
        V_final ← V_current
        F_final ← F_current

    else:
        Print("Network already fully verified!")
        repaired_network ← network
        V_final ← V_0
        F_final ← F_0

    # =========================================
    # Final Results
    # =========================================

    results['final'] ← {
        'verified': len(V_final),
        'failed': len(F_final),
        'total': len(V_final) + len(F_final),
        'success_rate': 100 * len(V_final) / (len(V_final) + len(F_final))
    }

    Print("\n" + "="*50)
    Print("ICGAR Pipeline Complete")
    Print("="*50)
    Print(f"Initial verified: {results['initial']['verified']}")
    Print(f"Final verified: {results['final']['verified']}")
    Print(f"Success rate: {results['final']['success_rate']:.1f}%")

    return repaired_network, results
```

---

## Summary of Key Components

| Component | Purpose | Complexity |
|-----------|---------|-----------|
| `IC`GAR_REPAIR` | Main repair loop with manifold projection | O(T·|θ|) |
| `COMPUTE_TANGENT_SPACE` | Compute invariant directions via SVD | O(m·k·|θ|) |
| `INCREMENTAL_SVD_UPDATE` | Update manifold for new verified simplices | O(|θ|²) |
| `COMPUTE_ALPHA` | Adaptive trade-off schedule | O(1) |
| `LBP_LOWER_BOUND` | Compute provable lower bound | O(|L|·d) |
| `GRADIENT_LBP_LOWER_BOUND` | Gradient of LBP for backprop | O(|L|·d) |
| `ICGAR_PIPELINE` | End-to-end verification + repair | O(T·|θ| + V·k·|θ|) |

Where:
- T = number of optimization iterations
- |θ| = number of network parameters
- m = number of verified simplices
- k = effective rank of manifold (k ≪ |θ|)
- |L| = number of layers
- d = input dimension
- V = total number of simplices verified

---

*Pseudocode generated: March 29, 2026*
*Based on ICGAR_FINAL_PROPOSAL.md*
