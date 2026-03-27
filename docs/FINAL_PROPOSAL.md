# Final Research Proposal: LBP-Guided Sparse Repair of Neural CBFs

## Problem Statement

Given a neural Control Barrier Function (NCBF) $B(x; \theta)$ that has been trained but fails LBP verification on some regions, we want to repair the network such that:
1. Failed regions become verified
2. Previously verified regions remain verified
3. Minimal modification to the network

## Core Method

### 1. LBP-Guided Neuron Selection

**Definition**: For each neuron $j=(l,i)$, compute selection score:

$$
\text{Score}(j) = \frac{\sum_{f \in \mathcal{F}_{td}} w_f \cdot \Gamma^{ben}_{f,j}}{\eta + \sum_{v \in \mathcal{V}} \tilde{w}_v \cdot \Gamma^{harm}_{v,j}}
$$

where:
- $\mathcal{F}_{td}$: true-defect failed simplices
- $\mathcal{V}$: verified simplices
- $w_f = 1 + \lambda_f \Delta_f^-$: weight for failed regions
- $\tilde{w}_v = \frac{1}{\Delta_v^+ + \tau}$: weight for verified regions
- $\Gamma^{ben}_{f,j}$: beneficial sensitivity on failed regions
- $\Gamma^{harm}_{v,j}$: harmful sensitivity on verified regions

**Key Insight**: This score captures "repair benefit / certification harm ratio" specific to LBP certificate structure, not just gradient sensitivity.

### 2. Verified-Region Invariant Repair

**Optimization Problem**:

$$
\min_{\Delta\theta_S} \quad \sum_{f \in \mathcal{F}_{td}} w_f [-\underline{h}_f(\theta + \Delta\theta)]_+ + \beta \sum_{v \in \mathcal{V}} \tilde{w}_v [\underline{h}_v(\theta) - \varepsilon_v - \underline{h}_v(\theta + \Delta\theta)]_+ + \lambda \|\Delta\theta_S\|_1
$$

subject to:
- $\|\Delta\theta_S\|_\infty \leq \delta$ (trust region)
- $\Delta\theta_{\bar{S}} = 0$ (only selected neurons)
- $\min_{v \in \mathcal{V}_{core}} \underline{h}_v(\theta + \Delta\theta) \geq 0$ (acceptance check)

where $S$ is the set of top-k selected neurons.

### 3. Small-Box Failure Diagnosis

**Algorithm**:
```
For each failed simplex s:
    1. Sample M points, compute h_min_sample
    2. If h_min_sample <= -τ_true: return TRUE_DEFECT

    3. Subdivide s into children {s_j}
    4. Compute child bounds {underline_h_{s_j}}

    5. If p_pass >= γ_loose: return BOUND_LOOSE
    6. If chi <= κ and h_min_sample >= -τ_true: return BOUND_LOOSE
    7. If depth == D_max and p_fail >= γ_true: return TRUE_DEFECT

    8. Recursively diagnose failed children
```

**Thresholds** (relative to median verified margin $m_{ref}$):
- $\tau_{fail} = 0.05 \cdot m_{ref}$
- $\tau_{true} = 0.02 \cdot m_{ref}$
- $\gamma_{loose} = 0.8$
- $\gamma_{true} = 0.3$
- $\kappa = 0.5$
- $D_{max} = 2$ or $3$

## Overall Algorithm

```python
def LBP_Sparse_Repair(network, verifier, state_space):
    # Phase 1: Verification
    verified, failed = verifier.verify(network, state_space)

    # Phase 2: Diagnosis
    F_td, F_loose, F_amb = [], [], []
    for s in failed:
        label = small_box_diagnosis(s, network, verifier)
        if label == TRUE_DEFECT: F_td.append(s)
        elif label == BOUND_LOOSE: F_loose.append(s)
        else: F_amb.append(s)

    # Phase 3: Neuron Selection
    scores = {}
    for neuron in network.all_neurons():
        scores[neuron] = compute_LBP_score(neuron, F_td, verified)
    S_repair = top_k_neurons(scores, k)

    # Phase 4: Local Repair
    delta_theta = solve_invariant_repair_optimization(
        network, S_repair, F_td, verified
    )

    # Phase 5: Acceptance Check
    network_prime = network.apply_update(delta_theta)
    verified_new, failed_new = verifier.verify(network_prime, state_space)

    if min_margin(verified_new) >= 0:
        return network_prime, "success"
    else:
        return iterative_repair(network, state_space)  # retry
```

## Theoretical Results

### Proposition 1: Hard-Constraint Invariance
If after repair $\underline{h}_v(\theta') \geq 0$ for all $v \in \mathcal{V}$, then all pre-verified simplices remain verified.

### Proposition 2: Trust-Region Sufficient Condition
If $|\underline{h}_v(\theta + \Delta\theta) - \underline{h}_v(\theta)| \leq L_v \|\Delta\theta\|$ and $\|\Delta\theta\| \leq \min_{v \in \mathcal{V}} \frac{\underline{h}_v(\theta)}{L_v}$, then all pre-verified simplices remain verified.

## Experimental Plan

### Benchmarks

| Benchmark | State Dim | Purpose |
|-----------|-----------|---------|
| Double Integrator | 2D | Method visualization, diagnosis validation |
| Kinematic Bicycle | 4D | Medium-dimensional validation |
| 6D Quadrotor | 6D | Scalability demonstration |

### Baselines

1. **Chen et al.**: Last-layer only repair
2. **Global Finetune**: Full network retraining
3. **Random Selection**: Random neuron selection + local repair
4. **Gradient Selection**: Gradient-based sensitivity + local repair

### Main Metrics

- **Repair Success Rate**: % of failed regions that become verified
- **Preservation Ratio**: % of pre-verified regions still verified
- **Control Performance**: Safety rate, goal-reaching rate
- **Runtime**: Total repair time
- **Modification**: Number of neurons/layers modified

### Ablation Studies

1. **Neuron Selection**: Random vs Gradient vs LBP Score C vs LBP Score A
2. **Preservation**: None vs Trust Region vs Soft Penalty vs Hard Constraint
3. **Diagnosis**: Without vs With diagnosis
4. **Budget**: k = 1, 5, 10, 20 neurons

### Result Tables

**Main Results** (per benchmark):
| Method | Repair Success ↑ | Preservation ↑ | Runtime ↓ | Modified Neurons ↓ |
|--------|------------------|----------------|-----------|-------------------|
| Chen et al. | | | | |
| Global Finetune | | | | |
| Random | | | | |
| Gradient | | | | |
| **Ours** | | | | |

**Diagnosis Evaluation**:
| Benchmark | Failed # | Diagnosed Loose | Diagnosed True Defect | Precision ↑ | Recall ↑ |
|-----------|----------|-----------------|----------------------|-------------|----------|
| Double Integrator | | | | | |
| Bicycle | | | | | |
| Quadrotor | | | | | |

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-2)
- [ ] Integrate LBP verification with repair module
- [ ] Implement neuron scoring functions (Scores A, B, C)
- [ ] Implement invariant repair optimization
- [ ] Test on Double Integrator

### Phase 2: Core Method (Weeks 3-4)
- [ ] Implement small-box diagnosis
- [ ] Implement iterative repair loop
- [ ] Test on Kinematic Bicycle
- [ ] Compare with baselines

### Phase 3: Scaling (Weeks 5-6)
- [ ] Test on 6D Quadrotor
- [ ] Full ablation studies
- [ ] Control performance evaluation
- [ ] Runtime analysis

### Phase 4: Paper Writing (Weeks 7-8)
- [ ] Draft paper
- [ ] Create figures
- [ ] Finalize proofs
- [ ] Prepare submission

## Expected Contributions

1. **First** LBP-guided neuron selection for NCBF repair
2. **Certified-region invariant repair** formulation
3. **Small-box diagnosis** for verification failures
4. Comprehensive experimental validation on standard benchmarks

## Target Venues

- **Primary**: NeurIPS 2025 / ICML 2025
- **Secondary**: AAAI 2026 / ICRA 2026

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| LBP score not better than gradient | Medium | Have Score B and C as backup |
| Preservation constraint too restrictive | Low | Use soft penalty + acceptance check |
| Diagnosis inaccurate | Medium | Use 2D exact validation as ground truth |
| Runtime too slow | Low | Sparse selection reduces problem size |

## Success Criteria

- Repair success rate > 80% on at least 2 benchmarks
- Preservation ratio > 95%
- Significantly outperforms random and gradient selection
- Small-box diagnosis precision > 70%
