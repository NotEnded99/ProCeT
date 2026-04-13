# Neural CBF Repair Method: Problem Analysis and Solutions

**Date**: 2026-04-09
**Code**: `New_repair/main_v1.py` + `optimizer_module.py` + `geometry_module_new.py`

---

## 1. Experimental Evidence of the Problem

From `nr_results_v1/result_barr3_Relu.json`:

| Metric | Initial | Final | Change |
|--------|---------|-------|--------|
| Pass Rate | 53.22% | 52.73% | **-0.49%** |
| Loss | - | 300-500+ | No convergence |
| Update Norm | - | ~1e-4 | Extremely small |

**Key observations**:
- All 968 failure regions are `F_depth_limit_reached` (reach max depth without certification)
- The extremely small `update_norm` (~1e-4) confirms **gradient explosion/vanishing**
- Loss is high but the repair **does not improve** and actually **decreases** pass rate

---

## 2. Root Cause Analysis

### 2.1 Gradient Explosion in CBF Lower Bound

The gradient `∂min_L/∂θ` (Lie derivative lower bound w.r.t. neural network parameters) explodes because:

```
min_L = min_{x∈S} [ ∇h(x)·f(x) + α·h(x) ]  (CBF condition)
```

The computation chain involves:
1. **CROWN linear relaxation** → network Jacobian bounds `J_affine_L, J_affine_U`
2. **McCormick envelopes** → bounds on `J(x)·f(x)` product
3. **Min over simplex** → `_batched_get_affine_function_bounds`
4. **Dynamics Taylor bounds** → `f_affine_L, f_affine_U`

Each stage involves:
- **Interval bounds** that over-approximate the true values
- **Piecewise-linear ReLU boundaries** creating non-smooth gradients
- **Loose lower bounds** that can be orders of magnitude smaller than true values

The gradient backpropagated through this chain is:
```
∂min_L/∂θ = ∂min_L/∂(J·f) · ∂(J·f)/∂J · ∂J/∂θ
```

The term `∂(J·f)/∂J` involves **McCormick envelope derivatives** which amplify small changes in J into large gradient signals. Combined with the `min` operation (which has subgradient `∈ [0,1]` at nondifferentiable points), gradients explode.

### 2.2 Why Repair Doesn't Improve Pass Rate

**Problem 1: Lower bound ≠ actual condition**

The repair optimizes `min_L` (a **lower bound** of the Lie derivative), but the verifier checks the **actual condition** using upper bounds too:

```
Verification uses: h_lb, h_ub (network output bounds)
                   min_L, max_L (Lie derivative bounds)
Certification requires: h_lb ≥ 0 AND min_L ≥ 0
```

Optimizing the lower bound doesn't guarantee the actual Lie derivative becomes positive - it only makes the **over-approximation** more positive.

**Problem 2: First-order linear constraint is insufficient**

The QP constraint `J·d ≤ 0` ensures the first-order Taylor approximation doesn't decrease `min_L`. But:
- The actual change in `min_L` is **nonlinear** in θ
- Higher-order terms dominate when gradients are large
- The Jacobian `J = ∂min_L/∂θ` is only accurate locally

**Problem 3: Fixed Jacobian becomes stale**

`compute_jacobian_matrix` is called once per outer iteration but `inner_loop_repair_with_qp` performs 10 gradient steps **without updating J**. After each step, the QP constraint uses an outdated linearization.

**Problem 4: Normalization destroys gradient information**

In `qp_project_and_update` (line 603-609):
```python
g_hat = g_raw / (g_raw_norm + epsilon)  # Gradient normalized!
J_hat = J / (J_norms + epsilon)  # Each row normalized!
```

This throws away magnitude information. A steep gradient in a safe direction is treated the same as a shallow gradient.

**Problem 5: Tolerance mismatch**

The repair uses `tolerance = -1e-12`, but:
- The verification uses `max_depth=10` as the termination criterion
- `F_depth_limit_reached` means even the lower bound couldn't certify within depth limit
- Repairing to `min_L ≥ -1e-12` doesn't help if the true bound needs `min_L ≥ large_margin`

---

## 3. Structural Issues in the Code

### 3.1 Inconsistent QP Normalization (Two Versions)

There are **two different `qp_project_and_update` functions**:
- Lines 556-668: Normalizes both `g_raw` AND `J_hat` rows → `d = g_hat - J_hat^T·λ`
- Lines 700-817: Only normalizes `J_hat` rows, keeps `g_raw` unnormalized

The inner loop `inner_loop_repair_with_qp` uses the **second version** (lines 700+), which doesn't normalize `g_raw`, causing inconsistent behavior.

### 3.2 Loss Aggregation Instability

In `compute_repair_loss_and_grad`:
```python
total_loss = sum(_valid_terms)  # Line 128
total_loss.backward()           # Line 135
```

Sum of many terms (potentially 1000+ simplexes) creates:
- Large loss magnitude → large gradients
- Uneven term magnitudes (some terms dominate)
- NaN propagation through accumulation

### 3.3 NaN Handling is Ad-hoc

```python
g_raw = torch.nan_to_num(g_raw, nan=0.0, posinf=0.0, neginf=0.0)  # Line 1476
```

Replacing NaN/Inf with 0 silently discards information. Gradient clipping at `grad_clip_norm=10.0` then clips everything.

---

## 4. Solutions and Recommendations

### 4.1 Short-term Fixes

**A. Use natural gradient or Gauss-Newton approximation**

Instead of raw gradients, use:
```python
# Gauss-Newton approximation of Hessian
G = J.T @ J + λ·I  # Fisher information matrix
g_nat = J.T @ r    # gradient in natural gradient space
```

**B. Trust region instead of gradient descent**

Replace QP projection with **trust region** approach:
```python
# Solve: min ||J·d + min_L||²  s.t. ||d|| ≤ δ
# This directly constrains the change in the bound, not the linear approximation
```

**C. Compute J more frequently**

Move Jacobian computation inside inner loop, or use **online Jacobian updates** with moving average.

**D. Use gradient scaling per-layer**

Instead of global normalization, scale gradients by parameter importance:
```python
param_norms = [p.grad.norm() / (p.norm() + ε) for p in model.parameters()]
scale = min(1.0, median(param_norms) / mean(param_norms))  # balance layers
```

### 4.2 Medium-term: Rethink the Objective

**A. Optimize verification margin, not just lower bound**

The goal should be to maximize **certification margin**:
```python
margin = min(h_lb, min_L)  # Both conditions matter
# But don't over-relax - we need BOTH ≥ 0
```

**B. Use episodic repair with model selection**

Instead of gradual gradient updates:
1. Train a new network head on failure regions
2. Evaluate on full verification
3. Keep best model

**C. Separate safe and unsafe region optimization**

- Safe regions (`V_safe`): Maximize margin `min_L - margin_target`
- Unsafe regions (`V_unsafe`): Maximize margin `-|h_ub| - margin_target`
- Failure regions: Directly optimize the violation

### 4.3 Long-term: Reformulation as Constrained Optimization

Current approach:
```
min_θ L(θ)  subject to  J_verified(θ*)·(θ-θ*) ≤ 0  (linearized constraint)
```

Better approach - **verify then repair loop**:
```
for iteration:
    1. Verify current θ → get certified/uncertified sets
    2. If certified == 100%: done
    3. For each uncertified region r:
       - Compute actual violation v_r(θ) = max(0, tolerance - min_L_r(θ))
       - Find parameter direction d_r that most reduces v_r
    4. Take weighted average of directions (weight by violation severity)
    5. Line search in that direction
    6. Go to step 1
```

### 4.4 Alternative: Gradient-Free Optimization for Certified Regions

Given the gradient problems, consider **derivative-free methods**:
- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
- **Bayesian optimization** with Expected Improvement
- **Random search** with safety certification

These avoid gradient computation entirely and can handle the nonsmooth LBP bounds.

---

## 5. Specific Code Changes Recommended

### 5.1 Fix the inconsistent QP normalization

Keep one version of `qp_project_and_update` and use it consistently. The second version (lines 700-817) should be removed.

### 5.2 Add Jacobian staleness diagnostic

```python
def compute_jacobian_with_staleness(model, J_old, ...):
    J_new = compute_jacobian_matrix(model, ...)
    staleness = (J_new - J_old).norm() / (J_old.norm() + ε)
    if staleness > 0.1:  # 10% change
        print(f"WARNING: Jacobian changed by {staleness:.1%}")
    return J_new
```

### 5.3 Replace sum with mean in loss aggregation

```python
# Instead of:
total_loss = sum(_valid_terms)

# Use:
total_loss = torch.stack(_valid_terms).mean()
```

### 5.4 Add gradient magnitude monitoring

```python
grad_stats = {
    'mean': g_raw.mean().item(),
    'std': g_raw.std().item(),
    'max': g_raw.abs().max().item(),
    'min': g_raw.abs().min().item(),
    'nan_count': torch.isnan(g_raw).sum().item(),
    'inf_count': torch.isinf(g_raw).sum().item(),
}
```

---

## 6. Summary

| Issue | Severity | Root Cause |
|-------|----------|------------|
| Gradient explosion in `∂min_L/∂θ` | **Critical** | McCormick + min chain creates nonsmooth, loose bounds |
| Repair doesn't improve pass rate | **Critical** | Optimizing lower bound ≠ improving actual certification |
| Inconsistent QP normalization | **Medium** | Two versions of `qp_project_and_update` |
| Stale Jacobian in inner loop | **Medium** | J computed once, used for 10 gradient steps |
| NaN silently replaced with 0 | **Low** | Information loss in gradient handling |

**Core recommendation**: The fundamental approach of "compute LBP lower bound → take gradient → update" is flawed for this problem because the lower bound is too loose and the gradient is too unstable. Consider switching to a **verify-then-repair** episodic approach with direct violation minimization, or use gradient-free optimization methods.

---

## References

- CROWN linearization: [Anderson et al., 2018]
- McCormick envelopes: [McCormick, 1976]
- Natural gradient: [Amari, 1998]
- Trust region methods: [Conn et al., 2000]



核心问题不是"loose bound"
Loose bound 本身不是问题——只要 bound 是 单调的（tight at optimum），优化 bound 就等价于优化真值。

真正的问题是：梯度链断裂了

真实物理:  θ → h(x;θ) → ∇h·f → min_{x∈S} ∇h·f
你的计算:  θ → LBP bound → min_L_bound
你计算的是 ∂min_L_bound/∂θ，但实际上：

McCormick envelope 的梯度 ≠ 真值梯度
McCormick 是真值的下界，但它的梯度方向和真值梯度方向 没有任何关系。McCormick 在边界处是分段线性的，梯度会在边界处跳变。

min 操作引入的不可微点
min_L = min_{x∈S} (affine_expression) 这个 min 是通过 _batched_get_affine_function_bounds 在单纯形顶点上求的。min 在底部是不可微的，PyTorch 的 subgradient 在这些点会给出任意值。

interval arithmetic 的符号断裂


CROWN bound: J_l ≤ ∇h(x) ≤ J_u
然后用 J_l · f_L 这样的组合去算下界
但 ∂(J_l · f_L)/∂θ ≠ ∂(true_J · true_f)/∂θ。梯度在 interval 的角点处就已经错了。