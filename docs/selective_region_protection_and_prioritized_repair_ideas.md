# Research Ideas: Selective Region Protection and Prioritized Repair for Neural CBF

## Context

**Code**: `New_repair/main_v4.py` — Neural CBF iterative repair pipeline
**Current Behavior**:
1. **Protection**: All `V_safe` regions are used to compute `J_RS` (Randomized Smoothing Jacobian) each iteration
2. **Repair**: ALL violation regions are fixed simultaneously:
   - `F_h_positive_in_unsafe` (barrier h > 0 in unsafe region)
   - `F_safe_cbf_violation` (CBF condition violated in safe region)
   - `F_depth_limit_reached` (verification depth limit)
   - `F_unsafe_cannot_split` (cannot split)

**Problem**:
- Protecting all `V_safe` is computationally expensive and unnecessary
- Repairing all violations simultaneously ignores severity differences
- No budget control on repair effort per iteration

---

## Ideas

### Idea 1: Margin-Based Top-N Selective Protection

**Summary**: Instead of computing `J_RS` over ALL `V_safe` regions, select only the top N regions with smallest safety margin (most vulnerable to being "pushed out").

**Core Hypothesis**: Most `V_safe` regions have large safety margins and contribute minimally to the Jacobian direction. Selecting vulnerable regions reduces compute while preserving constraint enforcement.

**Method**:
1. Compute `min_L` (CBF lower bound) for all `V_safe` simplices
2. Rank by margin = `min_L - cbf_margin` (smaller = more vulnerable)
3. Select top N = min(500, len(V_safe)) with smallest margins
4. Compute `J_RS` only on selected regions

**Minimum Viable Experiment**:
- Compare: (a) all V_safe vs (b) top-N selected, same repair pipeline
- Metric: final certified percentage, total compute time

**Expected Outcome**: ~same certified rate with 50-80% reduction in Jacobian compute

**Novelty**: Adaptive Jacobian computation guided by verification margins

---

### Idea 2: Severity-Weighted Prioritized Repair

**Summary**: Prioritize repair by violation severity rather than fixing all violations equally.

**Core Hypothesis**: `F_h_positive_in_unsafe` and `F_safe_cbf_violation` are "hard" safety violations; `F_depth_limit_reached` and `F_unsafe_cannot_split` are "soft" verification artifacts. Prioritizing hard violations yields faster convergence.

**Priority Ordering**:
| Priority | Region Type | Reason |
|----------|-------------|--------|
| 1 | `F_h_positive_in_unsafe` | Direct safety violation (h>0 in unsafe) |
| 2 | `F_safe_cbf_violation` | CBF condition violated |
| 3 | `F_depth_limit_reached` | May become safe with more refinement |
| 4 | `F_unsafe_cannot_split` | Structural limitation of verifier |

**Method**:
1. Iteratively repair regions in priority order
2. Within each priority class, rank by violation magnitude
3. Stop when all priority-1 regions are fixed, then move to priority-2, etc.

**Minimum Viable Experiment**:
- Compare: (a) current all-at-once vs (b) priority-ordered repair
- Track: iterations to reach 100% or plateau

**Expected Outcome**: Faster initial convergence, same or better final rate

---

### Idea 3: Budgeted Repair with Per-Iteration Cap M

**Summary**: Limit the number of regions repaired per iteration to M (e.g., M=100), selecting the M most severe violations.

**Core Hypothesis**: Repairing fewer but more severe violations per iteration allows finer control and avoids "repair interference" where fixing one region destabilizes another.

**Method**:
1. Compute severity score for each violation:
   - `F_h_positive_in_unsafe`: severity = h_lb (more positive = worse)
   - `F_safe_cbf_violation`: severity = cbf_margin - min_L (more negative = worse)
2. Select top M = min(100, total_violations) by severity
3. Repair only selected regions this iteration
4. Remaining violations fixed in subsequent iterations

**Variants**:
- **Fixed M**: Always M regions (pad with zeros if insufficient)
- **Adaptive M**: M = f(total_violations, iteration_number)
- **Decay M**: Start large, shrink over iterations

**Minimum Viable Experiment**:
- Compare: (a) current unlimited vs (b) M=50, M=100, M=200
- Metric: iterations to convergence, final rate

**Expected Outcome**: More stable convergence, potentially better final rate

---

### Idea 4: Combined Selective Protection + Budgeted Prioritized Repair

**Summary**: Combine Idea 1 (top-N protection) + Idea 2 (prioritized repair) + Idea 3 (budget M) into a unified framework.

**Core Hypothesis**: The three optimizations are complementary and together achieve the best efficiency-效果 trade-off.

**Full Algorithm**:
```
For each iteration:
    1. Select top N_vulnerable V_safe regions by margin (for J_RS)
    2. Rank all violations by priority (hard > soft) then severity
    3. Select top M violations for repair this iteration
    4. Compute J_RS on selected V_safe only
    5. Compute repair gradient on selected violations only
    6. QP project and update
```

**Minimum Viable Experiment**:
- Baseline: current v4 (all V_safe, all violations)
- Treatment: combined selective strategy (N=500, M=100)
- Compare: certified_rate vs iteration, compute time

**Expected Outcome**: Similar or better certified rate with significantly reduced compute

---

## Priority Recommendation

| Idea | Risk | Effort | Potential Impact | Recommended |
|------|------|--------|------------------|-------------|
| Idea 1: Selective Protection | LOW | 1 day | Moderate compute savings | **Start here** |
| Idea 2: Priority Ordering | LOW | 1 day | Faster convergence | **Start here** |
| Idea 3: Budget M | MEDIUM | 2 days | Better convergence stability | Good follow-up |
| Idea 4: Combined | MEDIUM | 3 days | Best overall | Full treatment |

---

## Implementation Notes

Key functions to modify:
- `compute_jacobian_rs()` in `geometry_module_new_v4.py`: add `top_n` parameter
- `compute_repair_loss_and_grad()` in `optimizer_module_v3.py`: add `max_violations` and `priority_weights` parameters
- `extract_feature_points_from_regions()`: add selection logic for top-N / top-M

Metrics to track:
- Per-iteration: compute time breakdown, number of regions selected
- Final: certified percentage, total iterations, total time
- Per-region-type: repair success rate by priority class

---

## Next Steps

1. Implement Idea 1 + Idea 2 as baseline modifications
2. Run ablation: baseline vs individual ideas vs combined
3. If results positive → integrate into main pipeline
4. If results negative → analyze failure modes, iterate on selection strategy
