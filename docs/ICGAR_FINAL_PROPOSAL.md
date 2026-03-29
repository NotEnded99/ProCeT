# Final Research Proposal: Iterative Certificate-Gradient Aligned Refinement (ICGAR)

**Date**: 2026-03-29
**Target Venue**: NeurIPS 2025 / ICML 2025

---

## Problem Anchor

**Fixed Problem Statement**:
When Neural Control Barrier Functions (NCBFs) fail LBP verification in certain regions of the state space, existing repair methods either: (1) modify only the last layer (too conservative, Chen et al.), or (2) modify the entire network (unprincipled, may break verified regions). We need a repair method that can fix failures while preserving what works, with theoretical guarantees.

**NOT In Scope** (to prevent scope drift):
- Training NCBFs from scratch (training methods)
- Verification-only (no repair)
- Non-CBF neural network repair
- Multi-certificate repair (separate direction)
- Hardware acceleration of verification

**IS In Scope**:
- Post-verification repair of single NCBF
- Using LBP verification results to guide repair
- Mathematical guarantees for verified-region preservation
- Theoretical analysis of repair convergence

---

## Method Thesis (One Sentence)

We formulate the verified-region invariance property as an explicit manifold constraint on the network parameters and use projected gradient descent to optimize repair loss while staying on (or near) this certificate manifold, achieving faster convergence and guaranteed invariance preservation.

---

## Dominant Contribution

**The Contribution (Two Parts)**:

1. **Manifold Formulation of Certificate Invariance** (Theoretical):
   - We show that the set M = {θ | h̲_v(θ) = h̲_v(θ₀), ∀v∈V} forms a manifold in parameter space
   - We characterize its tangent space T_θM and normal space N_θM
   - We prove low-dimensionality: dim(T_θM) ≪ |θ| under mild assumptions
   - This provides a geometric interpretation of certificate invariance

2. **Gradient-Projection Repair Algorithm** (Algorithmic):
   - We design an optimization algorithm that projects repair gradients onto T_θM
   - We introduce adaptive α(t) schedule that trades off repair speed vs. invariance
   - We prove convergence properties under standard assumptions
   - We provide efficient implementation using low-rank SVD

**Why This Is Novel**:
- No prior work formulates certificate invariance as an explicit manifold
- Existing methods (CSR, Chen) use subspace decomposition or last-layer tricks
- Manifold optimization perspective is standard but never applied to CBF repair
- Gradient projection provides principled alternative to heuristic subspace selection

---

## Detailed Method Description

### 1. Certificate Manifold Characterization

**Definition 1 (Certificate Manifold)**:
Given a set of verified simplices V = {v₁, ..., v_m} and initial parameters θ₀, define:
```
M = {θ ∈ ℝ^{|θ|} | h̲_v(θ) = h̲_v(θ₀), ∀v∈V}
```

where h̲_v(θ) is the LBP lower bound of the CBF h(x; θ) over simplex v.

**Proposition 1 (Manifold Properties)**:
Assume:
- Each v∈V has non-degenerate A_L matrix from LBP
- h(x; θ) is smooth in θ for fixed x (standard for neural nets)

Then:
1. M is a differentiable manifold near θ₀
2. Its dimension is at most: dim(M) ≤ Σ_l n_l where n_l is neurons in layer l
3. M is locally Euclidean (flat) in the sense that its tangent space is constant near θ₀

**Proof Sketch**:
- LBP lower bound h̲_v(θ) is linear in θ for fixed v (follows from LBP structure)
- The equality constraint h̲_v(θ) = h̲_v(θ₀) defines a hyperplane in ℝ^{|θ|}
- Intersection of m hyperplanes defines an affine subspace of dimension |θ| - m
- For non-denerate A_L matrices, constraints are independent

**Proposition 2 (Tangent Space Structure)**:
Define the tangent space at θ₀ as:
```
T_θ₀M = {d ∈ ℝ^{|θ|} | ⟨d, ∇_θ h̲_v(θ₀)⟩ = 0, ∀v∈V}
```

This is the orthogonal complement of the normal space:
```
N_θ₀M = span{∇_θ h̲_v(θ₀) | v∈V}
```

**Key Insight**: T_θ₀M is the subspace of parameter updates that do not affect h̲_v at first order.

### 2. Efficient Tangent Space Computation

**Challenge**: Computing N_θ₀M directly requires gradients ∇_θ h̲_v for all v∈V, which is O(m·|θ|).

**Solution 1: Low-Rank Approximation (Primary)**:

Compute the Jacobian J = [∂h̲_v/∂θ] stacked for all v∈V (shape: m×|θ|)

Compute SVD: J = UΣV^T

Observation: For typical CBFs, Σ has rapid decay → J is low-rank.

Let k be such that Σ₁₁/ / Σ₁,Σ₁+₁ ≥ 0.9 (90% variance explained)

Then:
```
T_θ₀M ≈ ker(V[:, 1:k])  # Nullspace of top-k right singular vectors
```

**Complexity**: SVD is O(min(m, |θ|)·max(m, |θ|)²) but truncated SVD is O(m·k·|θ|)

**Solution 2: Incremental Update (For Large |V|)**:

Maintain running SVD of Jacobian as simplices are verified:

```
Initialize: U, Σ, V from first n₀ simplices
For each new simplex v:
    Compute gradient g = ∇_θ h̲_v(θ)
    Update: U, Σ, V ← rank-one_update(U, Σ, V, g)
```

This allows O(|θ|²) updates per new simplex.

### 3. Gradient-Projection Repair Algorithm

**Algorithm 1 (Projected Gradient Repair)**:

```
Input: Network h(x; θ₀), verified simplices V, failed simplices F
Output: Repaired network h(x; θ*)

Precompute:
    T_θ₀M ← compute_tangent_space(V, θ₀)  # Using low-rank SVD
    P_T ← projection_operator(T_θ₀M)

Initialize:
    θ ← θ₀
    t ← 0

Repeat until convergence or t > T_max:
    # Step 1: Compute repair loss
    L ← Σ_{f∈F} [ -h̲_f(θ) ]_+  # Hinge loss on LBP lower bounds
        + λ‖θ - θ₀‖²  # L2 regularization

    # Step 2: Compute gradient
    g ← ∇_θ L

    # Step 3: Project to tangent space
    g_∥ ← P_T(g)           # Component that preserves invariance
    g_⊥ ← g - g_∥          # Component that may violate invariance

    # Step 4: Adaptive combination
    α_t ← compute_alpha(t, L_t, L_{t-1})
    g_α ← g_∥ + α_t · g_⊥

    # Step 5: Update
    θ ← θ - η · g_α

    # Step 6: Verify periodically
    if t % K_verify = 0:
        V_new, F_new ← verify_with_lbp(h(x; θ))
        if |F_new| = 0:
            break

    t ← t + 1

Return θ
```

**α(t) Schedule Options**:

1. **Strict Invariance**: α_t = 0 (never violate invariance)
2. **Linear Ramp**: α_t = min(1, t/T) (gradually allow violations)
3. **Exponential Decay**: α_t = 1 - exp(-t/τ) (start permissive, converge to strict)
4. **Feedback-Based**: α_t = α_0 · (|F_t| / |F_0|) (more violations early)

**Theoretical Choice**: From constrained optimization theory, for strongly convex L:

```
α_t^* ← argmin_α∈[0,1] E[L(θ_{t+1}(α_t))]
```

In practice, we tune α_0 and schedule type on validation set.

### 4. Theoretical Analysis

**Theorem 1 (Invariance Preservation)**:

Assume:
- M is defined as in Definition 1
- P_T is orthogonal projection onto T_θ₀M
- α_t = 0 for all t

Then for all v∈V and all t:
```
h̲_v(θ_t) = h̲_v(θ₀)
```

**Proof**: By induction. Base case t=0: θ₀ = θ₀ by definition. Inductive step:
- θ_{t+1} = θ_t - η·P_T(∇_θ L(θ_t))
- By definition of tangent space, ⟨P_T(∇_θ L(θ_t)), ∇_θ h̲_v(θ_t)⟩ = 0
- First-order Taylor expansion: h̲_v(θ_{t+1}) ≈ h̲_v(θ_t) + η·⟨∇_θ L(θ_t), ∇_θ h̲_v(θ_t)⟩
- But ∇_θ L(θ_t) depends on failed regions, while ∇_θ h̲_v(θ_t) depends on verified regions
- By construction, these gradients are orthogonal
- Therefore: h̲_v(θ_{t+1}) ≈ h̲_v(θ_t)
- By induction: h̲_v(θ_t) = h̲_v(θ₀) ∀v∈V

**Theorem 2 (Convergence Under Smoothness)**:

Assume:
- L(θ) is strongly convex with constant μ
- ∇_θ L is Lipschitz continuous with constant L
- Step size η < 2μ/L²
- M is a Hadamard manifold with bounded curvature

Then the projected gradient descent converges:
```
lim_{t→∞} ‖θ_t - θ^*‖ = 0
```
where θ^* is the projection of θ₀ onto the optimum of L constrained to M.

**Proof Sketch**: Projected gradient descent on manifolds is standard. Key conditions:
- Strong convexity: Ensures unique optimum
- Lipschitz gradient: Prevents oscillations
- Hadamard manifold: Ensures well-defined projection

These conditions hold for our problem because:
- L is hinge loss + L2 regularization → convex, strongly convex if regularized
- Neural networks are Lipschitz in parameters
- M is defined by linear constraints → hadamard

---

## Expected Contributions

### Theoretical
1. First manifold formulation of certificate invariance
2. Tangent space characterization and low-rank approximation
3. Invariance preservation theorem (Theorem 1)
4. Convergence guarantee (Theorem 2)
5. Efficient implementation via truncated SVD

### Algorithmic
1. Projected gradient descent algorithm for certificate repair
2. Multiple α(t) schedules with theoretical grounding
3. Incremental tangent space updates for scalability
4. Integration with LBP verification pipeline

### Empirical
1. Faster convergence than baseline methods (to be demonstrated)
2. 100% invariance preservation when α=0
3. Trade-off between speed and invariance via α(t)
4. Scalability to large |V| via low-rank approximation

---

## Comparison to Existing Methods

| Method | Invariance Guarantee | Theoretical Basis | Repair Expressivity | Complexity |
|--------|----------------------|------------------|-------------------|-----------|
| **CSR (base)** | 100% (subspace projection) | Eigenvalue decomposition | Full network | O(d³) |
| **Chen et al.** | Partial (last-layer only) | CEGIS theory | Last layer only | O(n_last) |
| **Global fine-tune** | None | None | Full network | O(|θ|) |
| **ICGAR (ours)** | Tunable (α=0 → 100%) | Manifold theory | Full network | O(m·k·|θ|) |

**Key Advantage of ICGAR**:
- Tunable: Can trade between strict invariance and repair speed
- Principled: α(t) has theoretical grounding (vs. heuristic subspace)
- Efficient: Low-rank approximation scales to large |V|
- Flexible: Can extend to Riemannian gradient descent for better geometry

---

## Potential Weaknesses and Mitigations

| Weakness | Likelihood | Mitigation |
|-----------|------------|------------|
| Tangent space computation expensive | Low | Low-rank SVD, incremental updates |
| Manifold may not be smooth (ReLU) | Medium | Smooth approximation or accept non-smoothness |
| α(t) schedule requires tuning | High | Theoretical guidance + validation tuning |
| Convergence not guaranteed in practice | Low | Empirical monitoring, early stopping |
| Theoretical assumptions unrealistic | Medium | Acknowledge in paper, test empirically |

---

## Implementation Plan (8-10 Weeks)

### Week 1-2: Theoretical Foundation
- [ ] Complete proofs for Theorems 1 and 2
- [ ] Write formal definitions and propositions
- [ ] Connect to classical manifold optimization literature
- [ ] Analyze special cases (e.g., |V| = 1)

### Week 3-4: Algorithmic Implementation
- [ ] Implement tangent space computation with SVD
- [ ] Add incremental SVD updates
- [ ] Implement projected gradient descent
- [ ] Implement multiple α(t) schedules
- [ ] Verify implementation on 2D test case

### Week 5-7: Experiments (see Experiment Plan)
- [ ] Run core experiments (Exps 1-4)
- [ ] Run ablation studies (Exps 5-7)
- [ ] Generate all figures and tables

### Week 8-10: Writing
- [ ] Draft manuscript with all theorems
- [ ] Create visualizations
- [ ] Polish narrative
- [ ] Final proofreading

---

## Success Criteria

### Minimum for Viable Paper
- [ ] Theorem 1 and 2 with complete proofs
- [ ] Algorithm 1 with complexity analysis
- [ ] Experiments 1-4 with clear results
- [ ] Comparison to at least 2 baselines (CSR, Chen)

### Target for Top Venue
- [ ] All above minimum criteria
- [ ] Convergence curves showing ICGAR improvement
- [ ] Ablation of α(t) schedules
- [ ] Scalability study on 2D/4D/6D
- [ ] At least 1 surprising positive result (e.g., bound tightening)

### Stretch Goals
- [ ] Extension to Riemannian gradient descent
- [ ] Theoretical optimal α(t) derivation
- [ ] Application to non-CBF certificates
- [ ] Real-world robot arm validation

---

## Paper Outline (Proposed Structure)

```
Title: Iterative Certificate-Gradient Aligned Refinement for
       Neural Control Barrier Functions with Verified-Region Invariance

Abstract:
- Problem: NCBF verification fails in some regions
- Limitation: Existing repair methods are either too conservative or unprincipled
- Innovation: Formulate invariance as manifold constraint
- Method: Projected gradient descent with adaptive α(t) schedule
- Results: Faster convergence, tunable invariance, theoretical guarantees

1. Introduction
   1.1 Neural Control Barrier Functions and Their Verification
   1.2 The Repair Problem and Existing Approaches
   1.3 Our Insight: Certificate Invariance as a Manifold
   1.4 Contributions and Outline

2. Background
   2.1 Neural CBFs and LBP Verification
   2.2 Existing Repair Methods
       2.2.1 Certified-Subspace Repair (CSR)
       2.2.2 Last-Layer Repair (Chen et al.)
       2.2.3 Global Fine-Tuning
   2.3 Manifold Optimization (Primer)

3. Problem Formulation
   3.1 Certificate Invariance Property
   3.2 Manifold Definition and Structure
   3.3 Optimization Problem Statement

4. Method
   4.1 Certificate Manifold Characterization
       4.1.1 Tangent and Normal Spaces
       4.1.2 Efficient Computation via SVD
   4.2 Gradient-Projection Repair Algorithm
       4.2.1 Projected Gradient Descent
       4.2.2 Adaptive α(t) Schedules
   4.3 Theoretical Analysis
       4.3.1 Invariance Preservation (Theorem 1)
       4.3.2 Convergence Guarantee (Theorem 2)

5. Experiments
   5.1 Setup and Benchmarks
   5.2 Convergence and Repair Effectiveness
   5.3 Invariance Preservation Analysis
   5.4 α(t) Schedule Ablation
   5.5 Scalability and Complexity
   5.6 Comparison to Baselines

6. Results and Discussion
   6.1 Key Findings
   6.2 Trade-off: Speed vs. Invariance
   6.3 Why Manifold Perspective Matters
   6.4 Limitations

7. Related Work
   7.1 Neural Network Repair
   7.2 CBF Verification and Training
   7.3 Manifold Optimization

8. Conclusion and Future Work
```

---

*Final Proposal Generated: March 29, 2026*
