# IBP-Based CBF Verification Technical Specification

## 1. Overview

### 1.1 Motivation

The original LBP (Linear Bound Propagation) verification code uses CROWN linearization with McCormick relaxations to compute tight bounds on neural network outputs and their Jacobians. While these bounds are tight, the fine-tuning phase suffers from:

- **NaN/gradient explosion**: The McCormick relaxation involves complex slope computations `1/(u-l)` and branching logic
- **Numerical instability**: Recursive Jacobian propagation (`_vectorized_mccormick_product`) amplifies errors
- **Slow compilation**: Complex bound structures slow down autograd during backpropagation

### 1.2 Proposed Solution

Replace LBP with IBP (Interval Bound Propagation) for the fine-tuning phase:

| Aspect | LBP (Original) | IBP (Proposed) |
|--------|---------------|----------------|
| Bound representation | Linear forms `Ax + b` | Simple intervals `[l, u]` |
| Jacobian bounds | McCormick relaxations | Interval matrix multiplication |
| Numerical stability | Complex, prone to NaN | Simple, stable |
| Bound tightness | Tighter | Looser but sufficient for fine-tuning |
| Gradient computation | Recursive, expensive | Direct interval arithmetic |

### 1.3 Design Goals

1. **Stability First**: Eliminate NaN/gradient explosion during fine-tuning
2. **Minimal Changes**: Reuse existing infrastructure (regions, dynamics, etc.)
3. **Generic Activation Support**: Support ReLU, Tanh, Sigmoid
4. **Gradient Bounds**: Compute IBP-based Jacobian bounds for CBF condition

---

## 2. IBP Forward Propagation

### 2.1 Core Data Structures

```python
class IBPBounds:
    """Interval bounds [lower, upper] for a layer's pre/post-activation values."""
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        self.lower = lower      # shape: [..., n_out] or [..., batch, n_out]
        self.upper = upper      # shape: [..., n_out] or [..., batch, n_out]
```

### 2.2 Linear Layer IBP

For a linear layer `y = Wx + b`, with input interval `[x_l, x_u]`:

```
W_pos = relu(W), W_neg = W - W_pos

l_out = W_pos @ l_in + W_neg @ u_in + b
u_out = W_pos @ u_in + W_neg @ l_in + b
```

This is the standard **positive-negative weight decomposition** for interval matrix multiplication.

**Implementation**:
```python
def ibp_linear(input_bounds: IBPBounds, weight: torch.Tensor, bias: torch.Tensor) -> IBPBounds:
    W_pos = torch.relu(weight)
    W_neg = weight - W_pos

    l_out = W_pos @ input_bounds.lower + W_neg @ input_bounds.upper + bias
    u_out = W_pos @ input_bounds.upper + W_neg @ input_bounds.lower + bias

    return IBPBounds(l_out, u_out)
```

### 2.3 Activation Functions

#### 2.3.1 ReLU

**Forward (Value Range)**:
- Monotonically increasing: `l_out = max(0, l_in)`, `u_out = max(0, u_in)`

**Backward (Derivative Range)**:
- Derivative is step function: `σ'(y) ∈ {0, 1}` or `[0, 1]` if crossing zero

```
if u_in <= 0:           # Fully inactive
    sigma_prime_l = sigma_prime_u = 0
elif l_in >= 0:         # Fully active
    sigma_prime_l = sigma_prime_u = 1
else:                   # Crossing zero (unstable)
    sigma_prime_l = 0, sigma_prime_u = 1
```

**Gradient/Jacobian Range**:
- `J_out = diag(sigma_prime) @ J_in` where `sigma_prime ∈ [0, 1]`

#### 2.3.2 Tanh

**Forward (Value Range)**:
- Monotonically increasing: `l_out = tanh(l_in)`, `u_out = tanh(u_in)`

**Backward (Derivative Range)**:
- Derivative: `σ'(y) = 1 - tanh²(y)`, which is a unimodal function symmetric around 0
- Maximum at y=0: `σ'(0) = 1`
- Minimum at boundaries: `min(σ'(l), σ'(u))`

```
# sigma'(y) is unimodal with peak at y=0
sigma_prime_l = min(sigma_prime(l_in), sigma_prime(u_in))
if 0 in [l_in, u_in]:
    sigma_prime_u = 1.0  # peak value
else:
    sigma_prime_u = max(sigma_prime(l_in), sigma_prime(u_in))
```

Where `sigma_prime(y) = 1 - tanh²(y) = 4 * exp(2y) / (exp(2y) + 1)²` for numerical stability.

#### 2.3.3 Sigmoid

**Forward (Value Range)**:
- Monotonically increasing: `l_out = sigmoid(l_in)`, `u_out = sigmoid(u_in)`

**Backward (Derivative Range)**:
- Derivative: `σ'(y) = σ(y)(1 - σ(y))`, unimodal with peak at y=0
- Maximum at y=0: `σ'(0) = 0.25`

```
sigma_prime_l = min(sigma_prime(l_in), sigma_prime(u_in))
if 0 in [l_in, u_in]:
    sigma_prime_u = 0.25  # peak value
else:
    sigma_prime_u = max(sigma_prime(l_in), sigma_prime(u_in))
```

Where `sigma_prime(y) = sigmoid(y) * (1 - sigmoid(y))`.

---

## 3. IBP Gradient Bounds (Jacobian Intervals)

### 3.1 Problem Formulation

We need bounds on `∂h/∂x` (Jacobian of barrier function output w.r.t. input) for the CBF condition:
```
∇h(x)·f(x) + α(h(x)) ≥ 0
```

### 3.2 Interval Matrix Multiplication

Given input Jacobian bounds `J_in ∈ [J_L, J_U]` and activation derivative bounds `σ' ∈ [σ'_l, σ'_u]`:

For **Diagonal Activations** (ReLU, Tanh, Sigmoid):
```
J_out = diag(σ'(y)) @ J_in
```

With interval `σ'(y) ∈ [σ'_l, σ'_u]`:
```
(J_out)_l = min(σ'_l, σ'_u) * J_in if J_in >= 0
(J_out)_u = max(σ'_l, σ'_u) * J_in if J_in <= 0
```

More generally for interval matrix multiplication `[A_l, A_u] @ [B_l, B_u]`:
```
C_l = min_{i,j} (A_l_ij * B_l_jj, A_l_ij * B_u_jj, A_u_ij * B_l_jj, A_u_ij * B_u_jj)
C_u = max_{i,j} (A_l_ij * B_l_jj, A_l_ij * B_u_jj, A_u_ij * B_l_jj, A_u_ij * B_u_jj)
```

### 3.3 Simplified Jacobian Bound Propagation

**Approach**: Propagate Jacobian bounds layer-by-layer using interval arithmetic.

For each layer `i`:
1. Get pre-activation interval `[l_i, u_i]` from forward IBP
2. Compute activation derivative interval `[σ'_l, σ'_u]`
3. Apply interval matrix multiplication: `J_{i+1} = diag([σ'_l, σ'_u]) @ J_i`

**Algorithm**:
```
J_L = identity(n)  # Initial Jacobian w.r.t. input
J_U = identity(n)

for each layer i:
    # Get derivative bounds
    sigma_prime_l, sigma_prime_u = get_activation_derivative_bounds(layer_i, l_i, u_i)

    # Diagonal scaling
    if activation is ReLU:
        # J = diag(sigma_prime) @ J, with sigma_prime ∈ [0, 1]
        # For stability, clamp derivative to [0, 1]
        scale_l = torch.clamp(sigma_prime_l, 0, 1)
        scale_u = torch.clamp(sigma_prime_u, 0, 1)

    elif activation is Tanh:
        scale_l = sigma_prime_l  # Already in [0, 1]
        scale_u = min(sigma_prime_u, 1.0)

    elif activation is Sigmoid:
        scale_l = sigma_prime_l
        scale_u = min(sigma_prime_u, 0.25)

    # Apply scaling to Jacobian bounds
    J_L = J_L * scale_l.unsqueeze(-1)  # Broadcast scaling
    J_U = J_U * scale_u.unsqueeze(-1)

    # If linear layer follows, apply weight decomposition
    if layer is Linear:
        W_pos = relu(W), W_neg = W - W_pos
        J_L_new = W_pos @ J_L + W_neg @ J_U
        J_U_new = W_pos @ J_U + W_neg @ J_L
        J_L, J_U = J_L_new, J_U_new
```

### 3.4 vec_min for Efficient Interval Bounds

For computing products like `J @ f(x)` where `J ∈ [J_L, J_U]` and `f(x) ∈ [f_L, f_U]`:

The lower bound of the dot product can be computed using `vec_min`:
```python
def vec_min(*args):
    """Compute element-wise minimum of multiple tensors."""
    result = args[0]
    for arg in args[1:]:
        result = torch.minimum(result, arg)
    return result

# For J @ f, lower bound:
# L = sum_i min(J_L[i] * f_L[i], J_L[i] * f_U[i], J_U[i] * f_L[i], J_U[i] * f_U[i])
```

---

## 4. CBF Condition Lower Bound (min_L)

### 4.1 CBF Condition Recap

For a control-affine system:
```
ẋ = f(x) + g(x)u
```

The CBF condition is:
```
∇h(x)·f(x) + ∇h(x)·g(x)·u + α(h(x)) ≥ 0
```

For verification, we need a **lower bound** on this condition over a region.

### 4.2 IBP-Based min_L Computation

**Step 1: Network forward pass (IBP)**
- Get `h(x)` interval: `[h_l, h_u]`
- Get `∇h(x)` interval Jacobian: `[J_L, J_U]` (shape: `[batch, n_out, n_in]`)

**Step 2: Dynamics bounds (Taylor/interval)**
- Get `f(x)` interval: `[f_L, f_U]`
- Get `g(x)` interval: `[g_L, g_U]` (shape: `[batch, n_in, m]`)

**Step 3: Compute drift lower bound**

```
L_drift = Σ_j vec_min(
    J_L[:, j] * f_L[:, j],   # J_L * f_L
    J_L[:, j] * f_U[:, j],   # J_L * f_U
    J_U[:, j] * f_L[:, j],   # J_U * f_L
    J_U[:, j] * f_U[:, j]    # J_U * f_U
)
```

**Step 4: Compute control lower bound**

For `∇h·g·u` where `u ∈ [u_min, u_max]`:

First compute `v = ∇h·g` interval: `[v_L, v_U]` (shape: `[batch, m]`)

```
v_L = Σ_j vec_min(J_L[:, :, j] * g_L[:, j, :], ...)
v_U = Σ_j vec_max(J_U[:, :, j] * g_U[:, j, :], ...)
```

Then for each control dimension `k`:
```
if v_L[k] >= 0:
    L_ctrl += v_L[k] * u_min[k]
elif v_U[k] <= 0:
    L_ctrl += v_U[k] * u_max[k]
else:
    # Mixed case: u affects sign
    L_ctrl += min(v_L[k] * u_max[k], v_U[k] * u_min[k])
```

**Step 5: Add class-K term**

```
α_l = alpha(h_l)  # alpha is monotonically increasing
L_total = L_drift + L_ctrl + α_l
```

**Step 6: Minimize over region**

For a simplicial region with vertices `V`, compute:
```
min_L = min_{x ∈ simplex} (L_total(x))
```

Since `L_total` is linear in `x` (interval bounds are affine), the minimum occurs at a vertex:
```
min_L = min_{v ∈ vertices} L_total(v)
```

---

## 5. Architecture

### 5.1 Class Hierarchy

```
IBPNetworkBoundCalculator
├── _ibp_forward()           # Forward interval propagation
├── _ibp_jacobian_bounds()    # Jacobian interval propagation
└── compute_min_L()           # CBF condition lower bound

IBPVerifier (standalone verification)
├── verify_batch()            # Batch verification using IBP
└── classify_region()          # SAT/UNSAT classification
```

### 5.2 Key Functions

```python
class IBPNetworkBoundCalculator:
    """Computes IBP bounds for neural networks."""

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float32):
        self.model = model
        self.dtype = dtype
        self.fc_layers = [...]  # Extracted linear layers
        self.activation_types = [...]  # ['ReLU', 'Tanh', ...]

    def ibp_forward(self, batch) -> List[IBPBounds]:
        """
        Perform IBP forward pass through the network.

        Args:
            batch: List of SimplicialRegion or HyperrectangularRegion

        Returns:
            List of IBPBounds for each layer's post-activation
        """
        # Extract input bounds from batch
        # For each layer:
        #   1. Apply linear IBP
        #   2. Apply activation IBP
        #   3. Store bounds
        pass

    def ibp_jacobian_bounds(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IBP bounds on the Jacobian ∂h/∂x.

        Returns:
            (J_L, J_U): Lower and upper Jacobian bounds, shape [batch, n_out, n_in]
        """
        pass

    def compute_min_L(
        self,
        batch,
        dynamics_model,
    ) -> torch.Tensor:
        """
        Compute lower bound on CBF condition for batch of regions.

        Returns:
            min_L: Shape [batch], lower bound on CBF condition
        """
        # 1. IBP forward pass → h(x) bounds and Jacobian bounds
        # 2. Get dynamics bounds (reuse existing Taylor-based approach)
        # 3. Compute drift lower bound
        # 4. Compute control lower bound
        # 5. Add class-K term
        # 6. Minimize over region
        pass
```

### 5.3 Activation Derivative Bounds

```python
def relu_derivative_bounds(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (sigma_prime_l, sigma_prime_u) for ReLU."""
    sigma_prime_l = torch.zeros_like(l)
    sigma_prime_u = torch.ones_like(u)

    # Active region: both >= 0
    active = (l >= 0) & (u >= 0)
    sigma_prime_l[active] = 1.0
    sigma_prime_u[active] = 1.0

    # Inactive region: both <= 0
    inactive = (l <= 0) & (u <= 0)
    sigma_prime_l[inactive] = 0.0
    sigma_prime_u[inactive] = 0.0

    # Crossing zero: l < 0 < u
    crossing = ~active & ~inactive
    # Already set to [0, 1] by default

    return sigma_prime_l, sigma_prime_u


def tanh_derivative_bounds(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (sigma_prime_l, sigma_prime_u) for Tanh."""
    # sigma_prime(y) = 1 - tanh²(y)
    # = 4 * exp(2y) / (exp(2y) + 1)² for numerical stability

    def sigma_prime(y):
        exp_2y = torch.exp(2 * y.clamp(-50, 50))
        return 4 * exp_2y / (exp_2y + 1) ** 2

    sp_l = sigma_prime(l)
    sp_u = sigma_prime(u)

    sigma_prime_l = torch.minimum(sp_l, sp_u)

    # Check if 0 is in interval
    contains_zero = (l < 0) & (u > 0)
    sigma_prime_u = torch.where(contains_zero, torch.ones_like(l), torch.maximum(sp_l, sp_u))

    return sigma_prime_l, sigma_prime_u


def sigmoid_derivative_bounds(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (sigma_prime_l, sigma_prime_u) for Sigmoid."""
    # sigma_prime(y) = sigmoid(y) * (1 - sigmoid(y))
    # = exp(y) / (exp(y) + 1)² for numerical stability

    def sigma_prime(y):
        exp_y = torch.exp(y.clamp(-50, 50))
        return exp_y / (exp_y + 1) ** 2

    sp_l = sigma_prime(l)
    sp_u = sigma_prime(u)

    sigma_prime_l = torch.minimum(sp_l, sp_u)

    # Check if 0 is in interval
    contains_zero = (l < 0) & (u > 0)
    sigma_prime_u = torch.where(contains_zero, 0.25 * torch.ones_like(l), torch.maximum(sp_l, sp_u))

    return sigma_prime_l, sigma_prime_u
```

---

## 6. Integration with Existing Code

### 6.1 File Structure

```
verify_cbf_ibp.py          # New IBP-based verification
├── IBPNetworkBoundCalculator     # Core IBP class
├── ibp_verify_cbf()              # Standalone verification function
└── compute_simplex_bound_ibp()  # Compatibility wrapper
```

### 6.2 Compatibility Interface

```python
def compute_simplex_bound_ibp(
    model: nn.Module,
    simplex_vertices: Union[torch.Tensor, np.ndarray],
    region_type: str,
    dynamics_model=None,
    translator=None
) -> torch.Tensor:
    """
    IBP-based version of compute_simplex_bound.
    Returns min_L for 'safe' regions, (h_lb, h_ub) for 'unsafe' regions.
    """
    pass


def compute_simplex_bound_batch_ibp(
    model: nn.Module,
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    region_type: str,
    dynamics_model=None,
    translator=None
) -> torch.Tensor:
    """
    Batch version of compute_simplex_bound_ibp.
    """
    pass
```

### 6.3 Relationship to Original Code

- **Reuses**: Region classes, dynamics models, TorchTranslator
- **Replaces**: `CrownPartialLinearization` with `IBPNetworkBoundCalculator`
- **Preserves**: CBF condition structure, min_L formulation, verification flow

---

## 7. Numerical Stability Considerations

### 7.1 Potential Issues and Mitigations

| Issue | Cause | Mitigation |
|-------|-------|------------|
| `exp(2y)` overflow | Large positive y | `clamp(y, -50, 50)` before exp |
| Division by zero | `u - l ≈ 0` in slope calc | IBP doesn't use slopes, no issue |
| Interval overestimation | Wrapping effect | Acceptable for fine-tuning |
| NaN in Jacobian bounds | Gradient explosion | Clamp derivative intervals |

### 7.2 Why IBP is More Stable

1. **No slope computations**: LBP needs `1/(u-l)` which explodes when `u ≈ l`; IBP doesn't need slopes for forward pass
2. **No McCormick branching**: LBP uses complex case analysis for products; IBP uses simple `vec_min`
3. **Direct interval arithmetic**: No recursive bound propagation that amplifies errors

---

## 8. Verification Flow

### 8.1 IBP Verification Algorithm

```python
def ibp_verify_batch(batch, dynamics_model, model):
    """
    Verify CBF conditions using IBP bounds.

    For each region in batch:
        1. Compute IBP forward bounds → h(x) ∈ [h_l, h_u]
        2. Compute IBP Jacobian bounds → ∇h(x) ∈ [J_L, J_U]

        3. Classify region:
           - If h_u < 0: SAT (unsafe_region)
           - If h_l >= 0 and min_L >= 0: SAT (safe_cbf_verified)
           - Else: UNSAT or split

        4. If CBF condition check needed:
           - Get dynamics bounds [f_L, f_U], [g_L, g_U]
           - Compute min_L using IBP bounds
           - If min_L >= 0: SAT
    """
    pass
```

### 8.2 Comparison with LBP

| Step | LBP | IBP |
|------|-----|-----|
| Forward bounds | Linear forms `Ax+b` | Simple intervals `[l,u]` |
| Jacobian computation | Recursive McCormick | Interval matmul |
| Dynamics combination | Complex eta optimization | Simple vec_min |
| Minimize over region | Linear programming | Vertex evaluation |

---

## 9. Implementation Plan

### 9.1 Phase 1: Core IBP Infrastructure
1. Implement `IBPBounds` data structure
2. Implement `ibp_linear`, `ibp_relu`, `ibp_tanh`, `ibp_sigmoid`
3. Implement `IBPNetworkBoundCalculator.ibp_forward()`

### 9.2 Phase 2: Jacobian Bounds
1. Implement activation derivative bounds functions
2. Implement `IBPNetworkBoundCalculator.ibp_jacobian_bounds()`
3. Implement `vec_min` helper

### 9.3 Phase 3: CBF Condition
1. Implement `compute_min_L()` combining network bounds with dynamics
2. Implement `compute_simplex_bound_ibp()` for single region
3. Implement `compute_simplex_bound_batch_ibp()` for batch processing

### 9.4 Phase 4: Verification Integration
1. Create `IBPVerifier` class
2. Implement `ibp_verify_cbf()` function
3. Add compatibility wrappers for existing repair code

---

## 10. References

- Original paper: "Scalable Verification of Neural Control Barrier Functions Using Linear Bound Propagation" (20260106-2511.06341v1)
- LBP implementation: `lbp_neural_cbf/linearization/linear_derivative_bounds.py`
- CBF verification: `lbp_neural_cbf/cbf/verify_cbf.py`
- McCormick relaxations: `_batched_compute_mccormick_product_lower_bound` in `verify_cbf.py`
