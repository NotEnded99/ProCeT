# Experiment Plan: LBP-Guided Sparse Repair

## Experiment Blocks

### Block 1: Neuron Selection Validation (Week 1-2)

**Purpose**: Validate that LBP-guided neuron selection is better than random/gradient selection

**Runs**:
1. Random selection + local repair on Double Integrator
2. Gradient-based selection + local repair on Double Integrator
3. LBP Score A + local repair on Double Integrator
4. LBP Score C + local repair on Double Integrator
5. LBP Score C → A (two-stage) + local repair on Double Integrator

**Metrics**: Repair success rate, Preservation ratio, Runtime, Modified neurons

**Success Criteria**: LBP scores outperform random/gradient by > 10%

---

### Block 2: Preservation Constraint Ablation (Week 2-3)

**Purpose**: Find best preservation constraint formulation

**Runs**:
1. No preservation constraint
2. Trust region only
3. Soft penalty only
4. Hard constraint only
5. Soft + Core hard check (proposed)

**Metrics**: Repair success rate, Preservation ratio, Verified margin drop

**Success Criteria**: Proposed method achieves > 95% preservation with minimal success rate loss

---

### Block 3: Small-Box Diagnosis Validation (Week 3)

**Purpose**: Validate diagnosis accuracy and utility

**Runs**:
1. Without diagnosis (repair all failed regions)
2. Diagnose and skip loose failures
3. Diagnose and prioritize true defects

**Metrics**: Diagnosis precision/recall, Wasted repairs on loose failures, Overall repair success

**Success Criteria**: Diagnosis precision > 70%, saves > 30% wasted repairs

---

### Block 4: Multi-Benchmark Evaluation (Week 4-5)

**Purpose**: Compare full method with baselines across benchmarks

**Runs** (per benchmark):
1. Chen et al. (last-layer only)
2. Global finetune
3. Random selection
4. Gradient selection
5. **Ours**: LBP-guided + invariant repair + diagnosis

**Benchmarks**: Double Integrator, Kinematic Bicycle, 6D Quadrotor

**Metrics**: All main metrics + control performance

**Success Criteria**: Ours achieves best or comparable repair success with highest preservation

---

### Block 5: Scalability Analysis (Week 5-6)

**Purpose**: Analyze method scaling with problem size

**Runs** (varying k):
- k = 1, 5, 10, 20, 50 neurons

**Metrics**: Success rate, Preservation ratio, Runtime, Params modified

**Success Criteria**: Clear trade-off curve showing diminishing returns at higher k

---

### Block 6: Control Performance (Week 6)

**Purpose**: Verify repair doesn't hurt control quality

**Runs**: 1000 rollouts per method per benchmark

**Metrics**: Safety rate, Goal-reaching rate, Episode return, Constraint violation

**Success Criteria**: Control performance within 5% of original network

---

## Compute Budget

| Block | GPU Hours | Notes |
|-------|-----------|-------|
| Block 1 | 8 | Double Integrator only |
| Block 2 | 8 | Double Integrator only |
| Block 3 | 4 | Double Integrator with ground truth |
| Block 4 | 32 | 3 benchmarks × 5 methods |
| Block 5 | 16 | Varying k on all benchmarks |
| Block 6 | 16 | Rollout simulation |
| **Total** | **84 hours** | ~10 GPU days |

## Data Collection

All results saved to:
- `experiments/results/neuron_selection/`
- `experiments/results/preservation_ablation/`
- `experiments/results/diagnosis/`
- `experiments/results/baselines/`
- `experiments/results/scalability/`
- `experiments/results/control_performance/`

## Analysis Scripts

- `analysis/compare_selection.py`: Neuron selection comparison
- `analysis/compare_preservation.py`: Preservation constraint comparison
- `analysis/diagnosis_accuracy.py`: Diagnosis validation
- `analysis/main_results.py`: Main results table
- `analysis/plot_pareto.py`: Success-preservation Pareto curve
- `analysis/plot_budget.py`: Budget scaling curve

## Paper Figures

1. **Figure 1**: 2D failure map before/after repair
2. **Figure 2**: Diagnosis visualization (loose vs true defect)
3. **Figure 3**: Success-preservation Pareto curve
4. **Figure 4**: Neuron selection comparison
5. **Figure 5**: Budget scaling analysis
6. **Figure 6**: Control performance comparison

## Submission Checklist

- [ ] All experiments completed
- [ ] All figures generated
- [ ] All tables formatted
- [ ] Code documented
- [ ] Reproducibility package ready
