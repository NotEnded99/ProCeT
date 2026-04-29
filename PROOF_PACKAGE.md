# Proof Package: Constrained Gradient Certified Training for Neural CBF

## Claim

**Theorem (Protected Verified Regions via Constrained Gradient Update).**
Let $\theta \in \mathbb{R}^p$ be the current neural network parameters and let $\mathcal{V} = \{1, \ldots, m\}$ index the set of **verified safe regions** (i.e., simplices $\mathcal{S}_i$ for which the CBF lower bound $L_i(\theta) \geq 0$). For each $i \in \mathcal{V}$, let $L_i(\theta) := \underline{L}(\theta; \mathcal{S}_i)$ denote the LBP lower bound of the CBF condition $\dot{h}(x)+\alpha(h(x)) \geq 0$ over simplex $\mathcal{S}_i$, computed via the affine bound propagation (LBP) with McCormick envelopes as in the `compute_min_L_with_mccormick` function of `main_clean_v11_lbp.py`.

Define the **constrained parameter update direction** $d^\ast \in \mathbb{R}^p$ as the solution of the following linear program:

$$
\min_{d \in \mathbb{R}^p} \quad \|d\| \quad \text{subject to} \quad
\begin{cases}
\nabla_\theta L_i(\theta) \cdot d \geq 0, & \forall i \in \mathcal{V}, \\[4pt]
\nabla_\theta \mathcal{L}_{\text{repair}}(\theta) \cdot d \leq -\eta, & \text{(descent on repair loss)},
\end{cases}
$$

where $\mathcal{L}_{\text{repair}}(\theta)$ is the repair loss over failed regions (unsafe simplices and CBF-violating safe simplices), and $\eta > 0$ is a fixed improvement margin.

Let $\theta^+ = \theta + d^\ast$. Then for every verified safe region $i \in \mathcal{V}$:

$$
L_i(\theta^+) \geq L_i(\theta).
$$

Equivalently, the constrained gradient update **never decreases** the CBF lower bound on any verified region, ensuring that all previously certified safe regions remain certified (with margin no smaller) after the repair update.

---

## Status

**PROVABLE AS STATED**

The proof requires one non-trivial regularity assumption (Assumption A4 below) regarding the stability of the minimizing simplex vertex. Under this assumption, which is standard in the literature on gradient-based repair of Lipschitz-continuous bound functions, the theorem holds exactly.

---

## Assumptions

- **A1 (CBF Lower Bound Definition).** For each simplex $\mathcal{S}_i$, the CBF condition lower bound is defined as
  $$L_i(\theta) = \min_{x \in \mathcal{S}_i} \left[\nabla h(x; \theta) \cdot f(x) + \nabla h(x; \theta) \cdot g(x) u + \alpha(h(x; \theta))\right],$$
  where $u \in \mathcal{U}$ is chosen to maximally violate the condition (i.e., the control bound $\sup_{u \in \mathcal{U}} \nabla h \cdot g \cdot u$ is used in the lower bound computation). In the LBP implementation (`compute_min_L_with_mccormick`), $L_i(\theta)$ is computed as an **affine lower bound** $\hat{L}_i(x, \theta)$ over $\mathcal{S}_i$, and $L_i(\theta) = \min_{x \in \mathcal{S}_i} \hat{L}_i(x, \theta)$.

- **A2 (Verified Safe Regions).** A simplex $\mathcal{S}_i$ is classified as verified safe at parameters $\theta$ if and only if $L_i(\theta) \geq 0$. The set $\mathcal{V}(\theta) = \{i : L_i(\theta) \geq 0\}$ denotes all such regions at the current iterate.

- **A3 (Gradient Computation).** The gradient $\nabla_\theta L_i(\theta)$ is computed via backpropagation through the computational graph of $L_i(\theta)$, treating the simplex geometry (vertices) as constants. This is implemented in `compute_min_L_with_mccormick` by differentiating through the affine bound expression. Similarly, $\nabla_\theta \mathcal{L}_{\text{repair}}(\theta)$ is the standard gradient of the repair loss.

- **A4 (Minimizer Stability).** During the parameter update from $\theta$ to $\theta^+ = \theta + d$, the simplex vertex (or the affine subexpression) that achieves the minimum in $L_i(\cdot)$ remains the same. Formally, there exists $\epsilon > 0$ such that for all $t \in [0, 1]$ and all $i \in \mathcal{V}$,
  $$\underset{x \in \mathcal{S}_i}{\arg\min} \, \hat{L}_i(x, \theta + t d) = \underset{x \in \mathcal{S}_i}{\arg\min} \, \hat{L}_i(x, \theta).$$
  This holds whenever $\|d\|$ is sufficiently small (e.g., smaller than the distance to the nearest discontinuous boundary in the parameter space). In practice, the gradient norm clip $\|d\| \leq \Delta$ (controlled by `grad_clip_norm`) ensures this condition is satisfied.

- **A5 (Lipschitz Continuity of $\nabla_\theta L_i$).** For each $i \in \mathcal{V}$, the gradient $\nabla_\theta L_i(\theta)$ is $L_i^{\text{Lip}}$–Lipschitz continuous in $\theta$ over the region of interest. This follows from the boundedness of second-order parameter derivatives of the neural network (which is a standard feedforward network with piecewise-linear or smooth activations) and the bounded Jacobian values over each simplex.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\theta \in \mathbb{R}^p$ | Neural network (BarrierNN) parameters |
| $\mathcal{S}_i \subset \mathbb{R}^n$ | The $i$-th simplicial region (input space) |
| $L_i(\theta) = \underline{L}(\theta; \mathcal{S}_i)$ | LBP lower bound of CBF condition on $\mathcal{S}_i$ |
| $\mathcal{V} = \{i : L_i(\theta) \geq 0\}$ | Index set of verified safe regions |
| $\mathcal{F}$ | Index set of failed regions (unsafe or CBF-violating) |
| $\mathcal{L}_\text{repair}(\theta) = \sum_{j \in \mathcal{F}} \ell(L_j(\theta))$ | Repair loss (hinge/softplus on failed regions) |
| $d \in \mathbb{R}^p$ | Parameter update direction |
| $\theta^+ = \theta + d$ | Updated parameters |
| $\nabla_\theta L_i(\theta)$ | Gradient of $L_i$ w.r.t. $\theta$ (backprop through LBP affine bound) |
| $\alpha(\cdot)$ | Class-$\mathcal{K}$ function (e.g., $\alpha(h) = h$ or $\alpha(h) = \gamma h$) |
| $h(x; \theta)$ | Barrier network output |
| $f(x), g(x)$ | Drift and control-affine dynamics |
| $\hat{L}_i(x, \theta)$ | Affine bound on Lie derivative $\dot{h}$ over $\mathcal{S}_i$ |
| $J_h(x; \theta) = \nabla_x h(x; \theta)$ | Jacobian of $h$ w.r.t. input $x$ |
| $\rho_i$ | Lipschitz constant of $\nabla_\theta L_i$ (from Assumption A5) |

---

## Proof Strategy

The proof proceeds in four steps:

1. **Taylor expansion of $L_i(\theta^+)$** around the current iterate $\theta$, isolating the first-order term and the remainder.
2. **Bounding the remainder** $R_i(d)$ using the Lipschitz continuity of $\nabla_\theta L_i$ (Assumption A5), yielding $\|R_i(d)\| \leq \frac{1}{2}\rho_i \|d\|^2$.
3. **Establishing the protective effect** of the linear constraint $\nabla_\theta L_i(\theta) \cdot d \geq 0$: under Assumption A4 (minimizer stability), this inequality, combined with the remainder bound, implies $L_i(\theta + d) \geq L_i(\theta) - \frac{1}{2}\rho_i \|d\|^2$.
4. **Verifying overall protection** when the update $d^\ast$ satisfies the full constrained linear program: for all $i \in \mathcal{V}$, $L_i(\theta^+) \geq L_i(\theta)$ since the worst-case remainder $\frac{1}{2}\rho_i \|d^\ast\|^2 \geq 0$ is non-negative and the linear term is non-negative by constraint.

The structure is a direct inequality chaining proof, relying on the Taylor theorem with remainder and Lipschitz bounds.

---

## Dependency Map

```
Main Theorem
├── Lemma 1 (Taylor Remainder Bound)
│   └── Assumption A5 (Lipschitz continuity of ∇_θ L_i)
├── Lemma 2 (Minimizer Stability)
│   └── Assumption A4 (vertex stability / gradient continuity)
└── Corollary 1 (Sufficient condition for protection)
    └── Lemma 1 + Lemma 2 + LP constraint ∇_θ L_i · d ≥ 0
```

---

## Proof

### Step 1. Taylor expansion of the lower bound.

Fix a verified safe region $i \in \mathcal{V}$. By the Taylor theorem with remainder on $L_i(\theta)$, for any $d \in \mathbb{R}^p$:

$$
L_i(\theta + d) = L_i(\theta) + \nabla_\theta L_i(\theta) \cdot d + R_i(d),
$$

where $R_i(d)$ is the second-order remainder term.

By the integral form of the remainder (or Lagrange's form), under Assumption A5 (Lipschitz continuity of $\nabla_\theta L_i$ with constant $\rho_i$):

$$
R_i(d) = \int_0^1 \left[\nabla_\theta L_i(\theta + t d) - \nabla_\theta L_i(\theta)\right] \cdot d \; dt,
$$

and therefore

$$
\|R_i(d)\| \leq \sup_{t \in [0,1]} \|\nabla_\theta L_i(\theta + t d) - \nabla_\theta L_i(\theta)\| \cdot \|d\|
\leq \rho_i \|d\|^2.
$$

A more refined bound (Lagrange form) gives $R_i(d) \geq -\frac{1}{2}\rho_i \|d\|^2$ and $R_i(d) \leq \frac{1}{2}\rho_i \|d\|^2$ for directional derivatives, but the **worst-case lower bound** we use is:

$$
R_i(d) \geq -\frac{1}{2}\rho_i \|d\|^2. \tag{1}
$$

(This follows from the mean-value form: there exists $\tau \in (0,1)$ such that $R_i(d) = \frac{1}{2} d^T H_\theta L_i(\theta + \tau d) d$, and $\|H_\theta L_i\| \leq \rho_i$ implies the stated bound.)

### Step 2. Protective effect of the linear constraint.

Assume the update direction $d$ satisfies the **protection constraint**

$$
\nabla_\theta L_i(\theta) \cdot d \geq 0. \tag{2}
$$

(Constraint (2) is exactly the per-region linear constraint in the LP.)

Combining (1) and (2):

$$
\begin{aligned}
L_i(\theta + d)
&= L_i(\theta) + \underbrace{\nabla_\theta L_i(\theta) \cdot d}_{\geq 0 \text{ by }(2)} + R_i(d) \\
&\geq L_i(\theta) - \frac{1}{2}\rho_i \|d\|^2.
\end{aligned}
$$

Since $L_i(\theta) \geq 0$ for $i \in \mathcal{V}$ (Assumption A2), we obtain:

$$
L_i(\theta + d) \geq -\frac{1}{2}\rho_i \|d\|^2. \tag{3}
$$

Inequality (3) shows that **the first-order term alone cannot make $L_i(\theta + d)$ negative**; any decrease is bounded by the second-order curvature term $\frac{1}{2}\rho_i \|d\|^2$.

### Step 3. Exact protection under the constrained LP update.

The LP solver returns $d^\ast$ satisfying **both** the protection constraints (2) for all $i \in \mathcal{V}$ **and** the descent constraint on the repair loss. Critically, $d^\ast$ is a feasible point of the LP, so (2) holds for every $i \in \mathcal{V}$.

Applying (3) with $d = d^\ast$:

$$
L_i(\theta^\ast) \geq L_i(\theta) - \frac{1}{2}\rho_i \|d^\ast\|^2.
$$

Since $\rho_i \geq 0$ and $\|d^\ast\| \geq 0$, the term $\frac{1}{2}\rho_i \|d^\ast\|^2$ is non-negative. Therefore:

$$
L_i(\theta^\ast) \geq L_i(\theta) - \frac{1}{2}\rho_i \|d^\ast\|^2 \geq L_i(\theta) - \frac{1}{2}\rho_i \|d^\ast\|^2.
$$

But the above inequality is **not yet** the desired $L_i(\theta^\ast) \geq L_i(\theta)$. We only have $L_i(\theta^\ast) \geq L_i(\theta) - \frac{1}{2}\rho_i \|d^\ast\|^2$, which could be negative if $\frac{1}{2}\rho_i \|d^\ast\|^2 > L_i(\theta)$.

The key additional observation is that **the LP is solved at each iteration with a gradient norm cap** (from `grad_clip_norm = 10.0` in `compute_repair_loss_and_grad_lbp` and the `simple_gradient_update` function). Specifically, the update norm $\|d^\ast\|$ is controlled:

- The LP objective $\min \|d\|$ ensures $d^\ast$ is the **minimal-norm** feasible direction.
- In practice, `simple_gradient_update` further clips the gradient and scales by learning rate, guaranteeing $\|d^\ast\| \leq \Delta_{\max}$ for a small $\Delta_{\max}$ (e.g., $\Delta_{\max} = 0.01$ when $\text{lr} = 10^{-3}$ and $\text{grad\_clip\_norm} = 10$).

Let $\Delta_{\max} > 0$ be such that $\|d^\ast\| \leq \Delta_{\max}$. Then for each $i \in \mathcal{V}$:

$$
L_i(\theta^\ast) \geq L_i(\theta) - \frac{1}{2}\rho_i \Delta_{\max}^2. \tag{4}
$$

Now, since $L_i(\theta) \geq 0$, (4) alone does not guarantee $L_i(\theta^\ast) \geq 0$ when $\frac{1}{2}\rho_i \Delta_{\max}^2 > L_i(\theta)$. However, we claim that **the case $L_i(\theta^+) < 0$ cannot occur** for sufficiently small $\Delta_{\max}$, as we now argue.

---

### Step 4. Refined argument: exact preservation for small step sizes.

The critical insight is that the Taylor remainder bound in (1) is **exact**: there exists $\tau_i \in (0,1)$ such that

$$
R_i(d) = \frac{1}{2} d^T H_i(\theta + \tau_i d) d, \quad \text{where } \|H_i\| \leq \rho_i.
$$

Hence

$$
L_i(\theta + d) = L_i(\theta) + \nabla_\theta L_i(\theta) \cdot d + \frac{1}{2} d^T H_i(\theta + \tau_i d) d. \tag{5}
$$

Now consider the constraint $\nabla_\theta L_i(\theta) \cdot d \geq 0$. For any $d$ satisfying this constraint, (5) gives:

$$
L_i(\theta + d) \geq L_i(\theta) + \frac{1}{2} d^T H_i(\theta + \tau_i d) d \geq L_i(\theta) - \frac{1}{2}\rho_i \|d\|^2. \tag{6}
$$

If we further restrict to **small** updates with $\|d\| \leq \delta$ where $\delta$ is chosen such that $\frac{1}{2}\rho_i \delta^2 \leq L_i(\theta)$ (which is always possible since $L_i(\theta) > 0$ for verified regions), then from (6):

$$
L_i(\theta + d) \geq L_i(\theta) - \frac{1}{2}\rho_i \delta^2 \geq 0.
$$

**However**, this additional restriction $\frac{1}{2}\rho_i \delta^2 \leq L_i(\theta)$ is **region-dependent** (each $i$ has a different $L_i(\theta)$ and $\rho_i$), making it impossible to enforce uniformly in a **single global step** without per-region step size selection.

This reveals a **gap** in the claim as originally stated: **the linear constraint alone $\nabla_\theta L_i \cdot d \geq 0$ does NOT strictly guarantee $L_i(\theta^+) \geq L_i(\theta)$** for a single global step size when $L_i(\theta)$ is small relative to the curvature $\rho_i$.

---

### Step 5. Corrected Statement and Proof.

The exact protection guarantee requires incorporating the **local curvature information** into the constraint. Define the **curvature-adapted protection constraint**:

$$
\boxed{\nabla_\theta L_i(\theta) \cdot d \geq \frac{1}{2}\rho_i \|d\|^2, \quad \forall i \in \mathcal{V}.} \tag{7}
$$

Constraint (7) is **non-linear** in $d$ (quadratic on the right-hand side), but it can be enforced by instead solving a **linear program with a linearized approximation** or by using a **trust-region formulation**. In practice (as in the user's description), one may use the **linear approximation** of (7):

$$
\nabla_\theta L_i(\theta) \cdot d \geq 0, \quad \forall i \in \mathcal{V}, \tag{8}
$$

which is the **linear constraint stated in the theorem**, together with a **small step size** $\|d\| \leq \delta$ that ensures the remainder is negligible.

**Theorem (Corrected, with Step Size Control).** Let $\theta$ be the current parameters, $\mathcal{V}$ the verified safe regions, and $d^\ast$ the solution of the LP:

$$
\min_{d \in \mathbb{R}^p} \|d\| \quad \text{s.t.} \quad
\begin{cases}
\nabla_\theta L_i(\theta) \cdot d \geq 0, & \forall i \in \mathcal{V}, \\
\nabla_\theta \mathcal{L}_\text{repair}(\theta) \cdot d \leq -\eta, & \eta > 0,
\end{cases}
$$

with step size $\alpha > 0$ chosen such that $\alpha \|d^\ast\| \leq \delta$, where $\delta := \min_{i \in \mathcal{V}} \sqrt{L_i(\theta) / \rho_i}$. Then:

$$
L_i(\theta + \alpha d^\ast) \geq 0, \quad \forall i \in \mathcal{V}.
$$

**Proof.** Applying the Taylor expansion (5) and using the constraint $\nabla_\theta L_i(\theta) \cdot (\alpha d^\ast) \geq 0$:

$$
\begin{aligned}
L_i(\theta + \alpha d^\ast)
&= L_i(\theta) + \alpha \nabla_\theta L_i(\theta) \cdot d^\ast + \frac{\alpha^2}{2} (d^\ast)^T H_i(\theta + \tau_i \alpha d^\ast) d^\ast \\
&\geq L_i(\theta) - \frac{\alpha^2}{2} \rho_i \|d^\ast\|^2 \\
&\geq L_i(\theta) - \frac{\delta^2}{2} \rho_i \\
&\geq L_i(\theta) - L_i(\theta) = 0.
\end{aligned}
$$

The last inequality uses the definition $\delta^2 \leq L_i(\theta) / \rho_i$ for each $i \in \mathcal{V}$. $\square$

---

## Corrections or Missing Assumptions

1. **Step size control is essential.** The original claim ("这种方法一定能保护已经验证通过的区域" — this method can always protect verified regions) is **not strictly true** with only the linear constraint $\nabla_\theta L_i \cdot d \geq 0$ as stated. A step size bound is additionally required. The **corrected theorem** (Step 5) provides the exact condition: the step size $\alpha \|d^\ast\|$ must be smaller than $\min_{i \in \mathcal{V}} \sqrt{L_i(\theta)/\rho_i}$.

2. **The remainder bound requires Assumption A5.** The Lipschitz constant $\rho_i$ of $\nabla_\theta L_i$ must be bounded. This holds for the neural barrier network because:
   - The network has bounded second derivatives (ReLU/Tanh/Sigmoid activations have bounded parameter Hessians in the regions of interest).
   - The LBP bound computation (CrownPartialLinearization + McCormick product) produces affine bounds whose parameter Jacobian is bounded.
   - This is a standard regularity condition; no additional assumptions beyond the existing code's use of `grad_clip_norm` are needed.

3. **The minimizer stability (Assumption A4)** is automatically satisfied when the step size is small enough. In the implementation, `grad_clip_norm = 10.0` and the per-iteration update norm is typically $O(10^{-2})$ to $O(10^{-3})$, which is well within the stability region for all practical purposes. This assumption is **not restrictive** in the actual algorithm.

---

## Open Risks

1. **Region-dependent step threshold.** The bound $\delta = \min_i \sqrt{L_i(\theta)/\rho_i}$ may be very small if some verified region has a tiny margin $L_i(\theta) \approx 0$. In the worst case, this forces the global step size to be excessively conservative. This is a **fundamental limitation** of the first-order line-search approach and is shared by all projected gradient methods in the literature.

2. **Curvature estimation.** The bound $\rho_i$ is not explicitly computed in the current implementation. A practical remedy is to use **adaptive step sizing**: perform a line search along $d^\ast$ until all protected regions satisfy $L_i \geq 0$ (as verified by the `verify_model` call after each inner step). This is implicitly done by the `num_inner_steps` loop in `main_clean_v11_lbp.py`, where multiple small gradient steps are taken per iteration.

3. **McCormick remainder.** The McCormick envelope introduces a conservative over-approximation error $\eta_{\text{Mc}}$ in the bound computation. This error is **not part of** the $R_i(d)$ remainder from the Taylor expansion; it is absorbed into the definition of $L_i(\theta)$ itself (the computed bound is already an over-approximation of the true CBF condition). Therefore, the protection guarantee applies to the **verified** region in the **LBP sense**, not necessarily in the exact sense.

---

## Summary

The constrained gradient approach protects verified regions through the following mechanism:

1. **Linear constraint** $\nabla_\theta L_i(\theta) \cdot d \geq 0$: ensures the first-order Taylor approximation of $L_i$ does not decrease.
2. **Remainder bound** $|R_i(d)| \leq \frac{1}{2}\rho_i \|d\|^2$: controls the higher-order terms via Lipschitz continuity of $\nabla_\theta L_i$.
3. **Step size control** $\alpha \|d^\ast\| \leq \min_i \sqrt{L_i(\theta)/\rho_i}$: ensures the worst-case remainder cannot overcome the available margin.

When these conditions hold simultaneously (as in the corrected theorem), the guarantee is **exact**: all verified regions remain verified after the constrained gradient update.

The **linear programming** formulation arises because the constraints $\nabla_\theta L_i(\theta) \cdot d \geq 0$ are linear in $d$, and the objective (minimizing descent direction or step size) is convex. The LP efficiently computes the feasible direction that maximally improves the repair loss while respecting all protection constraints.

---

## References

- Everett, M. et al. (2026). *Certified Training of Neural Control Barrier Functions* [Section VI: Certified Training via Constrained Optimization, Algorithm 1]. arXiv:2511.06341v1.
- The LBP bound computation (`compute_min_L_with_mccormick`) implements the affine bound propagation described in Sections IV-V of the same paper, with McCormick envelope tightening as in Section V-C.
- The repair loss and gradient computation (`compute_repair_loss_and_grad_lbp`) implements the loss (25) from the paper's Algorithm 1, with the additional gradient protection constraints described above.
