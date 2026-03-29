# Experimental Plan: Iterative Certificate-Gradient Aligned Refinement (ICGAR)

**Method**: ICGAR - Projected gradient descent on certificate manifold
**Date**: 2026-03-29
**Total Estimated GPU Time**: ~16-24 hours (across all experiments)

---

## Experiment Overview

| Exp # | Name | Priority | GPU Hours | Dependencies |
|--------|------|----------|--------------|
| 1 | Convergence Comparison | HIGH | None |
| 2 | Invariance Preservation | HIGH | None |
| 3 | α(t) Schedule Ablation | MEDIUM | None |
| 4 | Tangent Space Scalability | MEDIUM | None |
| 5 | Subspace Dimensionality Sweep | MEDIUM | Exp 1 |
| 6 | Comparison to Full Baselines | HIGH | Exp 1-5 |

---

## Benchmark Systems

| System | State Dim | Network | Trainable % | Difficulty |
|--------|-----------|---------|-------------|------------|
| **Double Integrator** | 2 | [64, 64, 1] | Easy | Simple dynamics, low failure rate |
| **2D Control** | 2 | [64, 64, 2] | Medium | Control-affine, moderate failures |
| **Kinematic Bicycle** | 4 | [64, 64, 1] | Medium-Hard | 4D state space, more complex |
| **6D Quadrotor** | 6 | [64, 64, 1] | Hard | High-dimensional, complex dynamics |

**Experiment Phasing**:
1. Phase 1 (Week 5-6): Double Integrator and 2D Control
2. Phase 2 (Week 7): Kinematic Bicycle
3. Phase 3 (Week 8): 6D Quadrotor (if time permits)

---

## Experiment 1: Convergence Comparison

**Goal**: Demonstrate that ICGAR converges faster than baselines to fix failures.

**Hypothesis**: Gradient projection on certificate manifold achieves faster convergence due to:
1. Better search direction (aligned with feasible set)
2. Reduced parameter space (low-dimensional tangent space)
3. Principled invariance handling (vs. heuristics)

**Setup**:
```
System: Double Integrator (2D)
Initial CBF: Pre-trained, 10-15% failure rate
Failed simplices: F (set from verification)
Verified simplices: V (complement)

Methods to compare:
1. ICGAR-α0 (strict invariance, α=0)
2. ICGAR-αLinear (linear ramp: α = min(1, t/T))
3. ICGAR-αExp (exponential: α = 1 - exp(-t/τ))
4. CSR (original eigenvalue method)
5. Chen et al. (last-layer convex optimization)
6. Global fine-tuning (no invariance)
```

**Procedure**:
```
For each method in [ICGAR-α0, ICGAR-αLinear, ICGAR-αExp, CSR, Chen, Global]:
    θ ← θ₀  # Reset to initial CBF

    For iteration t in 0..100:
        # Compute repair loss
        L ← Σ_{f∈F} [ -h̲_f(θ) ]_+ + λ‖θ - θ₀‖²

        # Compute and apply update
        θ ← method.update(θ, L, t)

        # Verify every 5 iterations
        if t % 5 == 0:
            V_new, F_new ← verify_with_lbp(h(x; θ))
            failures[t/5] ← len(F_new)
            verified[t/5] ← len(V_new)

            if len(F_new) == 0:
                convergence_time ← t
                break

    # Store final results
    results[method] ← {
        'convergence_time': convergence_time,
        'final_failures': len(F_new),
        'final_verified': len(V_new),
        'loss_curve': L_history,
        'failure_curve': failures_history
    }
```

**Metrics to Track**:
- Convergence time (iterations to reach 0 failures)
- Failure reduction curve: |F(t)| vs. t for each method
- Loss curve: L(t) vs. t for each method
- Time per iteration (computation cost)
- Total wall-clock time

**Expected Results**:
- ICGAR variants should converge in ≤50 iterations
- CSR should converge in 60-80 iterations
- Chen et al. may not converge (last-layer limited)
- Global fine-tuning should converge slower (>100 iterations)
- ICGAR-α0 (strict) may be slowest but safest

**Analysis Tasks**:
1. Plot convergence curves (method vs. t on x-axis, |F(t)| on y-axis)
2. Compute convergence speedup: (baseline_time / ICGAR_time) - 1
3. Statistical significance test across 5 random seeds
4. Identify which α(t) schedule works best

**Deliverables**:
1. Figure 1: Convergence curves for all methods
2. Table 1: Convergence time and final failure %
3. Analysis: Convergence speedup breakdown by method

**Estimated Compute**: 1-2 GPU hours

---

## Experiment 2: Invariance Preservation

**Goal**: Quantify how well each method preserves verified-region invariance.

**Hypothesis**:
- ICGAR-α0 achieves near-perfect invariance (δ_v ≈ 0)
- ICGAR-αLinear and ICGAR-αExp achieve controlled violation
- CSR may have small but non-zero violation
- Chen et al. and global fine-tuning likely violate invariance

**Setup**:
```
System: 2D Control (moderate complexity)
Initial CBF: Pre-trained, 12% failure rate
V₀ ← verified simplices from initial verification
```

**Procedure**:
```
For each method in [ICGAR-α0, ICGAR-αLinear, ICGAR-αExp, CSR, Chen, Global]:
    θ ← method.run_to_convergence()

    # Compute invariance violation for each original verified simplex
    violations ← []
    for v in V₀:
        # Compare original and repaired lower bounds
        h̲_original ← compute_LBP_lower_bound(h(x; θ₀), v)
        h̲_repaired ← compute_LBP_lower_bound(h(x; θ), v)

        # Absolute violation
        δ_v ← |h̲_repaired - h̲_original|
        violations.append(δ_v)

    # Compute statistics
    results[method] ← {
        'max_violation': max(violations),
        'mean_violation': mean(violations),
        'std_violation': std(violations),
        'violation_fraction': sum(v > 0.01) / len(violations)
    }
```

**Metrics to Track**:
- Max violation: max_v |h̲_v(θ_final) - h̲_v(θ₀)|
- Mean violation: average across all v∈V₀
- Std deviation of violations
- Fraction violating (threshold 0.01): % of V₀ with δ_v > 0.01
- Violation histogram: distribution of δ_v values

**Expected Results**:
- ICGAR-α0: max_violation < 0.001 (near-perfect)
- ICGAR-αLinear: max_violation < 0.05 (small violations)
- ICGAR-αExp: max_violation < 0.03
- CSR: max_violation < 0.1 (moderate violations)
- Chen et al.: max_violation may be 0.5-1.0 (significant violations)
- Global fine-tuning: max_violation may be 1.0+ (severe violations)

**Analysis Tasks**:
1. Plot violation distribution histograms for each method
2. Compute invariance preservation score: 1 - (mean_violation / range)
3. Compare to theoretical guarantees (ICGAR-α0 should achieve δ ≈ 0)
4. Correlate invariance violation with final repair success

**Deliverables**:
1. Figure 2: Violation distribution histograms
2. Table 2: Invariance preservation statistics
3. Analysis: Theoretical vs. empirical invariance

**Estimated Compute**: 0.5-1 GPU hour (verification dominates)

---

## Experiment 3: α(t) Schedule Ablation

**Goal**: Systematically evaluate different α(t) schedules and their trade-offs.

**Hypothesis**: Different α(t) schedules produce different trade-offs between:
1. Convergence speed (more violation → faster convergence)
2. Invariance preservation (less violation → better preservation)
3. Final repair quality (balanced schedule may be best)

**Setup**:
```
System: Double Integrator
Initial CBF: 10% failure rate

α(t) schedules to test:
1. α = 0 (strict invariance, baseline)
2. α = 0.1 (constant small)
3. α = 0.5 (constant medium)
4. α = 1.0 (constant full)
5. α = min(1, t/50) (linear ramp, T=50)
6. α = min(1, t/100) (linear ramp, T=100)
7. α = 1 - exp(-t/10) (expponential, τ=10)
8. α = 1 - exp(-t/25) (exponential, τ=25)
9. α = adaptive_linear (α = 0.5 + 0.5·t/T, capped at 1.0)
10. α = feedback (α_t = α_{t-1} · (|F_t| / |F_{t-1}|), bounded)
```

**Procedure**:
```
For each α_schedule in [1..10]:
    θ ← θ₀
    results ← []

    For iteration t in 0..100:
        # Compute α value
        α_t ← α_schedule(t, failure_history)

        # Run ICGAR update with this α
        θ ← icgar_update(θ, α_t)

        # Verify periodically
        if t % 5 == 0:
            V, F ← verify_with_lbp(h(x; θ))
            results.append({
                'iteration': t,
                'alpha': α_t,
                'failures': len(F),
                'verified': len(V)
            })

            if len(F) == 0:
                break

    # Analyze results
    analysis[α_schedule] ← analyze_convergence(results)
```

**Metrics to Track**:
- Final convergence time
- Failure reduction curve shape (early/later speed vs. early/late stability)
- Final invariance violation (from Experiment 2)
- α(t) over time curve
- Trade-off score: w1·(1 / convergence_time) - w2·mean_violation

**Expected Results**:
- α=0: Slowest convergence, best invariance
- α=1: Fastest convergence, worst invariance
" Linear/Exponential schedules: Balance convergence and invariance
- Feedback schedule: May adapt well to problem structure
- Optimal schedule likely: Linear ramp with T ≈ convergence_time/2

**Analysis Tasks**:
1. Plot Pareto frontier: (convergence_time, invariance_violation) for all schedules
2. Identify "knee" of Pareto curve (optimal trade-off)
3. Compare α(t) curves with failure reduction curves
4. Statistical analysis across 3 random seeds

**Deliverables**:
1. Figure 3: Pareto frontier of α(t) schedules
2. Table 3: Schedule performance comparison
3. Recommended α(t) for practice

**Estimated Compute**: 1-2 GPU hours

---

## Experiment 4: Tangent Space Scalability

**Goal**: Characterize how tangent space computation scales with |V| and evaluate approximation quality.

**Hypothesis**: Tangent space dimension is low (k ≪ |θ|) and grows slowly with |V|, enabling efficient approximation via truncated SVD.

**Setup**:
```
Systems: [Double Integrator, 2D Control, Kinematic Bicycle]
For each system:
    1. Verify initial CBF → get V₀ (verified simplices)
    2. Sample subsets of V₀ at different sizes
    3. For each subset size s, compute tangent space
```

**Subset sizes to test**: s ∈ {10, 50, 100, 500, 1000, 5000, all}

**Procedure**:
```
For each system in systems:
    V₀ ← get_verified_simplices(system)
    N_total ← len(V₀)

    For each subset_size s in [10, 50, 100, 500, 1000, 5000, N_total]:
        # Sample subset (or take all if s = N_total)
        if s == 'all':
            V_sample ← V₀
        else:
            V_sample ← random_subset(V₀, size=min(s, N_total))

        # Compute exact tangent space (SVD of full Jacobian)
        T_exact ← compute_tangent_space(V_sample, method='exact')
        k_exact ← dim(T_exact)
        t_exact ← time_taken()

        # Compute approximate tangent spaces at different ranks
        for target_rank r in [1, 2, 4, 8, 16, 32, 64, k_exact]:
            T_approx ← compute_tangent_space(V_sample, method='truncated_svd', rank=r)
            error ← subspace_distance(T_exact, T_approx)

            results.append({
                'system': system,
                'subset_size': s,
                'target_rank': r,
                'actual_rank': k_exact,
                'approximation_error': error,
                'compute_time': t_approx
            })
```

**Metrics to Track**:
- Exact tangent space dimension: k_exact (expected to be small, e.g., 2-10)
- Computation time: t vs. subset_size (log-log plot expected)
- Approximation error: ‖T_exact - T_approx‖ for different ranks
- Variance explained: for each rank r, % of variance captured
- Memory usage (for very large |V|)

**Expected Results**:
- k_exact: Should be 2-8 for all systems (much less than |θ| = thousands)
- Computation time: Should scale as O(|V|·k²), not O(|V|·|θ|²)
- Approximation error: Should decrease rapidly with r, reaching <5% at r = k_exact
- Memory: Should be manageable even for |V| = 10,000+ (due to low-rank)

**Analysis Tasks**:
1. Plot k_exact vs. system (bar chart)
2. Plot computation time vs. subset_size (log-log to show polynomial scaling)
3. Plot approximation error vs. rank (exponential decay expected)
4. Determine "optimal rank" where error < 5% threshold

**Deliverables**:
1. Figure 4: Tangent space dimensionality across systems
2. Figure 5: Computation time scaling analysis
3. Figure 6: Approximation quality vs. rank
4. Table 4: Tangent space properties summary

**Estimated Compute**: 2-4 GPU hours (dominated by many SVD computations)

---

## Experiment 5: Subspace Dimensionality Sweep

**Goal**: Determine optimal tangent space rank k for repair quality vs. efficiency.

**Hypothesis**: There exists optimal k* such that:
- k too small: insufficient degrees of freedom → poor repair
- k too large: too much flexibility → breaks invariance
- k ≈ k*: optimal balance of repair and preservation

**Setup**:
```
System: Kinematic Bicycle (4D, moderate complexity)
Initial CBF: Pre-trained, 15% failure rate
Rank values to test: k ∈ [1, 2, 4, 8, 16, 32, 64, 128, all]
```

**Procedure**:
```
For each rank k in [1, 2, 4, 8, 16, 32, 64, 128, all]:
    # Configure ICGAR with fixed k
    θ ← θ₀

    For iteration t in 0..100:
        # Compute tangent space with fixed rank k
        T_k ← compute_tangent_space(V₀, rank=k)

        # Projected gradient update
        L ← Σ_{f∈F} [ -h̲_f(θ) ]_+ + λ‖θ - θ₀‖²
        g ← ∇_θ L
        g_∥ ← P_{T_k}(g)  # Project to rank-k tangent space
        θ ← θ - η · g_∥  # No α(t) for this experiment

        # Verify periodically
        if t % 5 == 0:
            V, F ← verify_with_lbp(h(x; θ))

            if len(F) == 0:
                break

    # Store results
    results[k] ← {
        'convergence_time': t,
        'final_failures': len(F),
        'final_verified': len(V),
        'invariance_violation': compute_invariance_violation(V₀, θ)
    }
```

**Metrics to Track**:
- Convergence time vs. k
- Final failure % vs. k
- Invariance violation vs. k
- Pareto analysis: (convergence_time, invariance_violation) vs. k
- Compute time per iteration vs. k

**Expected Results**:
- k=1-2: Slow convergence, perfect invariance (if tangent space captures failure mode)
- k=4-16: Good balance, moderate convergence, small invariance violation
- k=32-64: Fast convergence, significant invariance violation
- k=128+: May converge fast but break many verified regions
- k=k*: Optimal trade-off point on Pareto frontier

**Analysis Tasks**:
1. Plot convergence_time vs. k (should decrease with k)
2. Plot invariance_violation vs. k (should increase with k)
3. Plot final_failures vs. k (should have optimal k*)
4. Compute Pareto frontier and identify k*
5. Compare k* to exact tangent space dimension from Experiment 4

**Deliverables**:
1. Figure 7: Trade-off curves vs. subspace rank k
2. Table 5: Rank sweep results with optimal k* highlighted
3. Recommendation: Practical k for each system

**Estimated Compute**: 2-3 GPU hours

---

## Experiment 6: Full Baseline Comparison

**Goal**: Comprehensive comparison of ICGAR to all relevant baselines on all systems.

**Hypothesis**: ICGAR provides best combination of:
1. Repair success (low final failure %)
2. Invariance preservation (low δ_v)
3. Convergence speed (fewer iterations)
4. Computational efficiency (reasonable time)

**Setup**:
```
Systems: [Double Integrator, 2D Control, Kinematic Bicycle]
Methods:
1. ICGAR (with optimal α(t) and k*)
2. CSR (original eigenvalue method)
3. Chen et al. (last-layer convex optimization)
4. Global fine-tuning (no constraints)
5. Random subspace repair (for comparison)
6. Gradient subspace repair (for comparison)
7. No repair (baseline, shows initial failures)

For each system and method combination:
    Run 5 random seeds
    Track all metrics from Experiments 1-5
```

**Comprehensive Metrics Table**:

| Metric | ICGAR | CSR | Chen | Global | Random | Gradient |
|--------|--------|-----|------|--------|--------|---------|
| **Repair Success** | % failures fixed | % | % | % | % | % |
| **Invariance Preservation** | max δ_v | max δ_v | max δ_v | max δ_v | max δ_v | max δ_v |
| **Convergence Iterations** | iterations | iter | iter | iter | iter | iter |
| **Wall-Clock Time** | seconds | sec | sec | sec | sec | sec |
| **GPU Memory** | GB | GB | GB | GB | GB | GB |
| **Final Bound Gap** | mean | mean | mean | mean | mean | mean |

**Procedure**:
```
For each system in systems:
    For seed in [1, 2, 3, 4, 5]:
        # Load pre-trained CBF with this seed
        θ₀ ← load_cbf(system, seed)

        # Verify initial state
        V₀, F₀ ← verify_with_lbp(h(x; θ₀))
        initial_fail ← len(F₀) / (len(V₀) + len(F₀))

        # Run each method
        for method in [ICGAR, CSR, Chen, Global, Random, Gradient]:
            θ ← method.run(θ₀, V₀, F₀)
            V_final, F_final ← verify_with_lbp(h(x; θ))

            # Compute all metrics
            results.append({
                'system': system,
                'seed': seed,
                'method': method,
                'initial_fail_rate': initial_fail,
                'final_fail_rate': len(F_final) / (len(V_final) + len(F_final)),
                'failures_fixed': initial_fail - len(F_final) / (len(V_final) + len(F_final)),
                'invariance_violation': max_invariance_violation(V₀, θ₀, θ),
                'convergence_time': method.iterations,
                'wall_time': method.total_time,
                'bound_gap_reduction': compute_gap_reduction(θ₀, θ)
            })
```

**Analysis Tasks**:
1. Statistical significance testing (ANOVA or t-tests)
2. Compute improvement metrics over best baseline
3. Generate comprehensive comparison tables
4. Create radar/spider plots showing method profiles
5. Identify which metric each method excels at

**Expected Results**:
- ICGAR: Best trade-off across all metrics
- CSR: Good invariance, slower convergence
- Chen et al.: Fast but limited repair (may not fix all failures)
- Global: Good repair, poor invariance
- Random/Gradient: Variable performance, worse than ICGAR

**Deliverables**:
1. Table 6: Comprehensive baseline comparison (main results table)
2. Figure 8: Radar/spider plot of method profiles
3. Figure 9: Success rate comparison by system
4. Statistical analysis: Significance tests and confidence intervals

**Estimated Compute**: 8-12 GPU hours (5 systems × 6 methods × 5 seeds)

---

## Ablation Studies

### Ablation 1: Tangent Space Computation Methods
**Variants**:
1. Exact SVD (gold standard)
2. Truncated SVD with rank=k*
3. Incremental SVD (online update)
4. Random subspace (control)

**Metric**: Approximation error vs. compute time

### Ablation 2: Loss Function Design
**Variants**:
1. Hinge loss on LBP bounds (primary)
2. Smooth hinge loss (hinge smoothed at ±ε)
3. Max of lower bounds (alternative)

**Metric**: Convergence speed and final quality

### Ablation 3: Regularization Strength
**Variants**:
1. No regularization (λ=0)
2. Weak regularization (λ=0.0001)
3. Medium regularization (λ=0.001)
4. Strong regularization (λ=0.01)

**Metric**: Overfitting vs. repair quality

---

## Evaluation Protocol

### Reproducibility
- Fix all random seeds (42, 123, 456, 789, 314)
- Report hardware specs (GPU model, CUDA version)
- Provide hyperparameters for all methods
- Release code repository with exact reproduction scripts

### Statistical Rigor
- Minimum 3 random seeds per configuration
- Report mean ± std across seeds
- Perform statistical significance tests
- 95% confidence intervals where applicable

### Failure Cases
Document and report:
- Cases where ICGAR fails to converge
- Systems where tangent space is high-dimensional (k ≈ |θ|)
- α(t) schedules that perform poorly
- Comparison to theoretical predictions

---

## Deliverables Summary

### Figures (10-12 total)
1. Figure 1: Convergence curves (Exp 1)
2. Figure 2: Invariance violation histograms (Exp 2)
3. Figure 3: α(t) schedule Pareto frontier (Exp 3)
4. Figure 4: Tangent space dimensionality (Exp 4)
5. Figure 5: Computation time scaling (Exp 4)
6. Figure 6: Approximation quality vs. rank (Exp 4)
7. Figure 7: Subspace rank trade-offs (Exp 5)
8. Figure 8: Method profile radar plot (Exp 6)
9. Figure 9: Success rate by system (Exp 6)
10. Figure 10: Ablation results
11. (Optional) Figure 11: 2D manifold visualization
12. (Optional) Figure 12: Real-world robot validation

### Tables (5-6 total)
1. Table 1: Convergence time comparison (Exp 1)
2. Table 2: Invariance preservation statistics (Exp 2)
3. Table 3: α(t) schedule performance (Exp 3)
4. Table 4: Tangent space properties (Exp 4)
5. Table 5: Subspace rank sweep results (Exp 5)
6. Table 6: Comprehensive baseline comparison (Exp 6)

### Code and Data
1. ICGAR implementation (with all α(t) schedules)
2. Tangent space computation (exact and approximate)
3. Experiment scripts (reproducible)
4. Results data (.csv files for all experiments)
5. Visualization code (generating all figures)

---

## Timeline (8 Weeks)

**Week 1-2**: Implementation (Exp 1, 2 readiness)
- [ ] Tangent space computation
- [ ] Projected gradient descent
- [ ] α(t) schedule implementations
- [ ] Verification integration

**Week 3-4**: Core Experiments (Exp 1-4)
- [ ] Run Exp 1: Convergence comparison
- [ ] Run Exp 2: Invariance preservation
- [ ] Run Exp 3: α(t) schedule ablation
- [ ] Run Exp 4: Tangent space scalability

**Week 5-6**: Extended Experiments (Exp 5-6)
- [ ] Run Exp 5: Subspace dimensionality sweep
- [ ] Run Exp 6: Full baseline comparison
- [ ] Run ablation studies

**Week 7**: Analysis
- [ ] Generate all figures
- [ ] Compute all tables
- [ ] Statistical analysis
- [ ] Failure case documentation

**Week 8**: Writing and Polish
- [ ] Draft manuscript
- [ ] Create all deliverables
- [ ] Internal review and revision
- [ ] Final proofreading

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Tangent space high-dimensional | Low | Expected low-rank from theory; have Exp 4 to verify |
| ICGAR fails to converge | Medium | Add fallback to CSR; document failures |
| α(t) schedule unclear | Low | Multiple schedules tested; provide recommendation |
| Computational cost too high | Low | Low-rank approximation; incremental updates |
| Baselines stronger than expected | Medium | Comprehensive comparison; honest reporting |

---

## Success Criteria

### Minimum Viable Paper
- [ ] All 6 experiments completed
- [ ] At least 3 systems tested
- [ ] ICGAR shows advantage over at least 1 baseline
- [ ] All figures and tables generated
- [ ] Code reproducible

### Target NeurIPS/ICML
- [ ] All minimum criteria met
- [ ] ICGAR shows clear advantage over CSR
- [ ] Convergence curves demonstrate speedup
- [ ] Invariance preservation quantified and justified
- [ ] Tangent space low-dimensionality confirmed
- [ ] Statistical significance demonstrated
- [ ] Theoretical claims validated empirically
- [ ] Clear narrative and positioning

### Stretch Goals
- [ ] Application to real robot system
- [ ] Extension to Riemannian gradient descent
- [ ] α(t) schedule with theoretical optimality proof
" [ ] Publication-quality code release

---

*Experiment Plan Generated: March 29, 2026*
