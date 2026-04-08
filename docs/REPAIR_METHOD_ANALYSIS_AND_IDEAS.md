# Neural CBF 验证后修复：方法分析、数学描述与改进方案

> 本文档对 Neural CBF 验证后修复的三种方法进行全面分析，描述其数学过程，识别核心问题，并提出系统性改进方案。

---

## 1. 背景：CBF 验证与修复的完整流程

### 1.1 控制屏障函数 (CBF) 基础

对于一个动力学系统 $\dot{x} = f(x)$，控制屏障函数 (CBF) $h: \mathbb{R}^n \to \mathbb{R}$ 需要满足：

$$
\begin{aligned}
\mathcal{S} &= \{x \in \mathbb{R}^n : h(x) \geq 0\} && \text{安全集} \\
\mathcal{C} &= \{x \in \mathbb{R}^n : h(x) < 0\} && \text{障碍区（不安全区域）} \\
\end{aligned}
$$

且在安全集上满足 CBF 条件：

$$
\dot{h}(x) = \nabla h(x) \cdot f(x) + \alpha(h(x)) \geq 0 \quad \forall x \in \mathcal{S}
$$

其中 $\alpha: \mathbb{R} \to \mathbb{R}$ 是一个 class-$\mathcal{K}$ 函数（通常是 $\alpha(h) = h$ 或 $\alpha(h) = \alpha \cdot h$）。

### 1.2 验证过程的区域分裂 (Simplicial Refinement)

验证算法采用**单纯形 (Simplex)** 空间划分 + **CROWN/LBP 线性界传播**的组合策略：

**分裂规则 (Case Analysis)**：对每个单纯形 $\Delta$，计算 $h$ 的界 $[h_{\min}, h_{\max}]$：

1. **Case 1 (h_max < 0)**：单纯形完全在障碍区内 → **SAT (V_unsafe)**，验证障碍区条件
2. **Case 2 (h_min >= 0)**：单纯形完全在安全区内 → **SAT (V_safe)**，验证 CBF 条件
3. **Case 3 (mixed)**：单纯形跨越边界 → **MAYBE**，继续分裂

**depth_limit_reached**：当分裂深度达到 `max_depth` 时，停止分裂，记录该区域为失败（虽然实际可能满足 CBF 条件）

**unsafe_cannot_split**：当单纯形无法继续分裂（体积太小）但仍处于 mixed 状态时，记录为失败

### 1.3 验证结果分类

验证完成后，所有单纯形被分类为：

| 类别 | 符号 | 含义 | 验证状态 |
|------|------|------|----------|
| SAT | `V_safe` | 安全区内 CBF 条件验证通过 | 已验证 |
| SAT | `V_unsafe` | 障碍区内 h(x) < 0 验证通过 | 已验证 |
| UNSAT | `F_h_positive_in_unsafe` | 障碍区内发现 h(x) >= 0 的点 | 违规 |
| UNSAT | `F_safe_cbf_violation` | 安全区内 CBF 条件不满足 | 违规 |
| UNSAT | `F_depth_limit_reached` | 分裂达到深度上限，无法继续 | 不确定 |
| UNSAT | `F_unsafe_cannot_split` | 单纯形太小无法分裂但仍 mixed | 不确定 |

**通过率** = Certified% = |V_safe ∪ V_unsafe| / |V_safe ∪ V_unsafe ∪ F_*|

---

## 2. 三种修复方法的数学描述

### 2.1 方法一：直接梯度下降 (main_clean.py)

**核心思想**：直接对失败区域的 Hinge Loss 进行反向传播，用原始梯度更新网络参数。

**数学过程**：

给定网络参数 $\theta \in \mathbb{R}^P$，对每个失败单纯形 $\Delta_i$，计算 LBP 界传播得到标量边界值：

$$
\hat{h}_i = \text{LBP}(\Delta_i, \theta) \in \mathbb{R}
$$

定义 Hinge Loss：

$$
\mathcal{L}(\theta) = \sum_{i \in \mathcal{F}_h} \max(0, \hat{h}_i) + \sum_{j \in \mathcal{F}_{\text{CBF}}} \max(0, \tau - \hat{L}_j) + \sum_{k \in \mathcal{F}_{\text{depth}}} \max(0, \tau - \hat{L}_k) + \sum_{l \in \mathcal{F}_{\text{split}}} \max(0, \tau - \hat{L}_l)
$$

其中 $\tau = -10^{-12}$ 是容忍度，$\hat{h}_i$ 是障碍区的上界，$\hat{L}_j$ 是安全区 CBF Lie 导数的下界。

**梯度计算**：

$$
g_{\text{raw}} = \frac{\partial \mathcal{L}}{\partial \theta} = \sum_i \frac{\partial \mathcal{L}_i}{\partial \theta}
$$

**参数更新**（SGD）：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot g_{\text{raw}}
$$

其中 $\eta$ 是固定学习率（通常 $\eta = 10^{-3}$）。

**优点**：实现简单，计算效率高
**缺点**：梯度可能破坏已验证区域（V_safe, V_unsafe），且缺乏方向引导

---

### 2.2 方法二：SVD 切向空间投影 (main.py)

**核心思想**：计算雅可比矩阵 $J$ 的 SVD，提取法向空间，将梯度投影到法向空间的正交补（切向空间），保证更新方向不改变已验证区域的边界值。

**数学过程**：

**Step 1：雅可比矩阵计算**

对所有 V_safe 和 V_unsafe 中的单纯形，计算边界值对网络参数的梯度：

$$
J_{ij} = \frac{\partial \hat{b}_i}{\partial \theta_j} \quad i = 1, \ldots, N, \quad j = 1, \ldots, P
$$

其中 $N = |\text{V\_safe}| + |\text{V\_unsafe}|$，$\hat{b}_i$ 是第 $i$ 个单纯形的标量边界值（LBP 传播结果）。

**Step 2：截断 SVD 提取法向空间**

对 $J \in \mathbb{R}^{N \times P}$ 做截断 SVD：

$$
J = U \Sigma V^T \approx U_k \Sigma_k V_k^T
$$

其中 $V_k \in \mathbb{R}^{P \times k}$ 是前 $k$ 个右奇异向量构成的矩阵，$k = \min(k_{\text{rank}}, N)$。

$V_k$ 的列张成 $J$ 的**法向空间 (Normal Space)** $\mathcal{N} = \text{col}(J^T)$，即参数空间中"能够改变已验证区域边界值"的方向。

**Step 3：切向空间投影**

将原始梯度 $g_{\text{raw}}$ 分解为法向分量 $g_{\perp}$ 和切向分量 $g_{\parallel}$：

$$
g_{\perp} = V_k V_k^T g_{\text{raw}} \quad \text{（法向分量）}
$$

$$
g_{\parallel} = g_{\text{raw}} - g_{\perp} \quad \text{（切向分量）}
$$

投影更新方向：

$$
g_{\text{update}} = g_{\parallel} + \alpha \cdot g_{\perp}
$$

- $\alpha = 0$：完全投影到切向空间（梯度方向在已验证区域的边界值上不变）
- $\alpha = 1$：不投影（原始梯度）

**Step 4：参数更新**

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot g_{\text{update}}
$$

**优点**：理论上保证不改变已验证区域的边界值
**缺点**：
1. SVD 截断引入了近似，$k_{\text{rank}}$ 的选择不直观
2. 切向空间中的梯度方向可能与损失下降方向不一致
3. 如果 $k$ 过小，可能找不到有意义的下降方向

---

### 2.3 方法三：QP 对偶约束投影 (main_v1.py)

**核心思想**：将安全约束编码为 QP 问题，求解拉格朗日乘子，在满足约束的前提下最小化与原始梯度的偏离。

**数学过程**：

**Step 1：雅可比归一化**

将 $J$ 的每行归一化为单位向量：

$$
\hat{J}_i = \frac{J_i}{\|J_i\|_2 + \epsilon} \quad \forall i = 1, \ldots, N
$$

同样地，将原始梯度归一化：

$$
\hat{g} = \frac{g_{\text{raw}}}{\|g_{\text{raw}}\|_2 + \epsilon}
$$

**Step 2：构建 QP 问题**

决策变量：$\lambda \in \mathbb{R}^N, \lambda \geq 0$（拉格朗日乘子）

目标函数：

$$
\min_{\lambda \geq 0} \frac{1}{2} \|\hat{g} - \hat{J}^T \lambda\|_2^2
$$

约束：$\lambda_i \geq 0$（隐式保证 $J_i \cdot d \leq 0$，即更新方向不会增加任何已验证区域的边界值）

展开目标函数：

$$
\frac{1}{2} \|\hat{g} - \hat{J}^T \lambda\|_2^2 = \frac{1}{2} \hat{g}^T \hat{g} - \hat{g}^T \hat{J}^T \lambda + \frac{1}{2} \lambda^T \hat{J}\hat{J}^T \lambda
$$

这是一个凸二次规划问题，可以用 OSQP 等求解器高效求解。

**Step 3：安全更新方向**

求解得到 $\lambda^*$ 后，计算更新方向：

$$
d = \hat{g} - \hat{J}^T \lambda^*
$$

该方向 $d$ 满足：对于所有 $i$，有 $J_i \cdot d \leq 0$（即更新不会增大任何已验证区域的边界值）。

**Step 4：参数更新**

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot d
$$

其中 $\eta$ 是步长（建议从 $\eta = 10^{-2}$ 开始调小）。

**优点**：
1. 有理论保证：$J \cdot d \leq 0$ 严格成立（如果 QP 求解成功）
2. 活跃约束识别：$\lambda_i^* > 0$ 的数量表明有多少个区域真正限制了更新方向
3. 不需要截断 SVD

**缺点**：
1. QP 求解有计算开销
2. 步长 $\eta$ 仍需手动调节
3. 归一化操作可能改变梯度方向的语义

---

## 3. 当前方法的核心问题

### 3.1 梯度缺乏正向引导（退化解问题）

**问题描述**：三种方法都只使用失败区域（F_*）计算梯度，没有任何正向引导项。这意味着：

- 即使通过率达到 100%，CBF 可能退化为 $h(x) \equiv 0$ 或 $h(x) < 0$ everywhere（对障碍区满足但无实际控制能力）
- 如果整个状态空间在浅深度下都被验证为 V_unsafe，那么通过率是 100%，但 CBF 完全失去了安全控制功能
- 梯度方向是被动的"哪里破了补哪里"，而不是主动朝向"好 CBF"优化

**根本原因**：损失函数只包含惩罚项 $\mathcal{L}_{\text{penalty}} = \sum_{i \in \mathcal{F}} \max(0, c_i(\theta))$，没有正则化项或稳定性项。

### 3.2 验证深度 (max_depth) 与修复的耦合问题

**问题描述**：`max_depth` 控制分裂深度，直接影响验证精度：

- **depth 太浅**：大量区域成为 `F_depth_limit_reached` 和 `F_unsafe_cannot_split`，这些区域的 CBF 条件**实际可能满足**，但被错误地当作失败区域处理
- **depth 太深**：验证时间指数增长，且 `F_depth_limit_reached` 和 `F_unsafe_cannot_split` 减少但计算成本剧增

**深度与修复的耦合**：

```
max_depth = 8 时：
  → 大量 F_depth_limit_reached (浅层分裂就停止了)
  → 修复梯度被这些"假失败"区域主导
  → 修复后的模型在 depth=13 时通过率仍然很低

max_depth = 13 时：
  → 验证更充分，F_depth_limit_reached 减少
  → 但验证时间显著增加
  → 如果初始通过率本来就不高，修复空间有限
```

**关键洞察**：`F_depth_limit_reached` 和 `F_unsafe_cannot_split` 是**不可靠的失败信号**——它们只表示"无法继续分裂"，而不是"CBF 条件确实被违反"。对这些区域施加惩罚梯度可能导致过拟合到浅层验证。

### 3.3 单步更新策略的局限性

**问题描述**：当前所有方法都是"计算一次梯度 → 更新一次参数 → 重新验证"的串行循环：

1. **梯度方差大**：每次只用一个 batch（所有失败区域）的梯度，没有 mini-batch 采样
2. **步长选择困难**：固定 lr 对不同区域的梯度量级差异不鲁棒
3. **信息浪费**：每次重新计算 J 和梯度，之前的梯度历史完全丢弃
4. **无自适应机制**：不根据验证反馈动态调整策略

---

## 4. 改进方案：系统性框架

### 4.1 改进一：组合损失函数（解决退化解）

在原有惩罚损失的基础上，引入正向引导损失：

$$
\mathcal{L}_{\text{total}}(\theta) = \lambda_1 \mathcal{L}_{\text{penalty}} + \lambda_2 \mathcal{L}_{\text{stability}} + \lambda_3 \mathcal{L}_{\text{barrier}} + \lambda_4 \mathcal{L}_{\text{smooth}}
$$

其中：

**稳定性损失 (V_safe 区域)**：

$$
\mathcal{L}_{\text{stability}} = \frac{1}{|\mathcal{V}_{\text{safe}}|} \sum_{i \in \mathcal{V}_{\text{safe}}} \max(0, \gamma_{\text{safe}} - \hat{L}_i)^2
$$

鼓励安全区域的 Lie 导数下界尽可能大，$\gamma_{\text{safe}} > 0$ 是 margin（如 0.1）。

**屏障强化损失 (V_unsafe 区域)**：

$$
\mathcal{L}_{\text{barrier}} = \frac{1}{|\mathcal{V}_{\text{unsafe}}|} \sum_{i \in \mathcal{V}_{\text{unsafe}}} \max(0, \hat{h}_{i,\max} + \gamma_{\text{unsafe}})^2
$$

鼓励障碍区的 h 上界尽可能负，$\gamma_{\text{unsafe}} > 0$ 是 margin（如 0.1）。

**自适应权重调度**：

$$
\lambda_k = \lambda_k^{(0)} \cdot f(\text{pass\_rate})
$$

当通过率高时增大 $\lambda_2 + \lambda_3$（精细化已有解），通过率低时增大 $\lambda_1$（优先修复失败区域）。

### 4.2 改进二：验证深度感知的分层修复策略

**核心思想**：根据 `F_depth_limit_reached` 和 `F_unsafe_cannot_split` 的比例，动态决定如何处理这些"不可靠失败"：

**分层策略**：

| 深度相关失败比例 | 策略 | 原因 |
|----------------|------|------|
| < 10% | 直接修复所有 F_* | 深度足够，失败信号可靠 |
| 10% ~ 50% | 对 F_depth/F_unsafe_split 加低权重 $\epsilon \ll 1$ | 部分信号不可靠 |
| > 50% | **忽略** F_depth/F_unsafe_split，只修复 F_h/F_safe | 深度太浅，这些失败区域很可能是假阳性 |

**自适应深度调节**：

```
if F_depth_ratio > 0.3 AND F_unsafe_split_ratio > 0.1:
    increase_max_depth_by(2)  # 下次验证用更深的分裂
    降低 F_depth/F_unsafe_split 的损失权重
```

**渐进式深度训练**：

```
for depth in [4, 6, 8, 10, 13]:
    用 depth 运行 verify_cbf 获取 F_*
    对前 4 个深度的 F_* 使用递减权重: weight = 0.1^(depth - 4)
    计算组合损失并更新参数
    保存中间模型
```

这样，深层验证区域有更大的修复权重，浅层区域有更小的权重。

### 4.3 改进三：多步梯度累积 (Mini-batch Gradient Accumulation)

在 `repair_loop` 中，每次内循环迭代只采样失败区域的一个随机子集：

```python
def repair_loop_multistep(model, J, F_*, dynamics_model, translator,
                           num_inner_iters=10, batch_ratio=0.3,
                           lr=1e-3, k_rank=500):
    for inner_iter in range(num_inner_iters):
        # 随机采样失败区域的子集
        F_batch = sample_failure_batch(F_*, ratio=batch_ratio)

        # 计算子集上的损失和梯度
        loss_batch, g_batch = compute_loss_and_grad(model, F_batch, ...)

        # 累积梯度
        if inner_iter == 0:
            g_accumulated = g_batch
        else:
            g_accumulated = g_accumulated + g_batch

    # 用累积梯度做一次投影和更新
    V_k, _ = extract_tangent_space(J, k_rank)
    project_and_update(model, g_accumulated / num_inner_iters, V_k, lr, alpha=0.0)
```

**理论依据**：类似 SGD 的 mini-batch，可以显著减少梯度方差，加速收敛。

### 4.4 改进四：Trust Region 方法 + 自适应步长

用 **Cauchy Point** 方法选择步长：

```
# 1. 计算 trust region radius: delta
delta = initial_delta * decay_rate^(iter)

# 2. 计算 Cauchy point: d_C = min_{d, ||d|| <= delta} g^T d
# 对于二次函数有解析解: d_C = (delta / ||g||) * g

# 3. 可选：计算共轭梯度步 d_CG（如果 J 已知）
# 4. 选择 d = d_C 或 d_CG 中损失更小的

# 5. 线搜索验证
theta_new = theta_old + d
verify_and_compute_loss(theta_new)
```

结合 **Simulated Annealing** 风格的学习率调度：

```
if current_pass_rate > last_pass_rate:
    lr = lr * 1.1  # 增大步长
elif current_pass_rate == last_pass_rate:
    lr = lr * 0.95  # 保持
else:
    lr = lr * 0.5  # 缩小步长
    # 可选：恢复上一步的参数
    theta_old = theta_backup
```

### 4.5 改进五：QP 约束的精细化

在 QP 投影中，对不同类型的区域使用不同的约束强度：

$$
\min_{\lambda \geq 0} \frac{1}{2} \|\hat{g} - \hat{J}^T \lambda\|_2^2 + \frac{\mu}{2} \|\lambda\|_2^2
$$

加入正则化项 $\frac{\mu}{2}\|\lambda\|_2^2$，避免某些 $\lambda_i$ 过大导致数值不稳定。

同时，对不同区域使用不同的归一化权重：

$$
\hat{J}_i^{\text{weighted}} = \frac{w_i J_i}{\|J_i\|_2 + \epsilon}
$$

其中 $w_i$ 根据区域类型设置：

- V_safe: $w_i = 1.0$（保持 CBF 条件）
- V_unsafe: $w_i = 0.5$（允许少量改变）
- F_*: $w_i = 0.0$（不约束失败区域）

### 4.6 改进六：内外双循环重构

将当前的单循环重构为：

```
外循环 (expensive, infrequent):
  - 运行 verify_cbf(max_depth) 获取最新的 F_*
  - 更新 J 矩阵
  - 判断是否需要增加 max_depth
  - 决定是否停止

内循环 (cheap, frequent):
  - 随机采样 F_* 的子集
  - 累积多步梯度
  - 用投影/QP 更新参数
  - 不重新验证（节省计算）
```

这样，验证（expensive）和梯度更新（cheap）分离，内循环可以快速多步更新。

---

## 5. 完整的改进框架：综合方案

将所有改进整合为一个统一的框架：

### 5.1 算法流程

```
输入: 初始模型 θ_0, max_depth_init, 目标通过率 R_*
输出: 修复后的模型 θ

θ = θ_0
max_depth = max_depth_init
outer_iter = 0

while outer_iter < MAX_OUTER_ITERS:
    # === 外循环：验证 ===
    results = verify_cbf(θ, max_depth=max_depth)
    pass_rate, F_* = extract_results(results)
    J = compute_jacobian_matrix(θ, V_safe, V_unsafe, ...)

    # 深度自适应判断
    depth_fail_ratio = (len(F_depth) + len(F_unsafe_split)) / total_regions
    if depth_fail_ratio > 0.3:
        max_depth = min(max_depth + 2, MAX_DEPTH)
        print(f"[Depth Adaptation] max_depth -> {max_depth}")

    # 权重调度
    λ_penalty, λ_stability, λ_barrier = compute_weights(pass_rate)

    # 判断停止
    if pass_rate >= R_*:
        break
    if outer_iter > 0 AND pass_rate <= last_pass_rate AND lr < MIN_LR:
        break  # 收敛

    # === 内循环：多步梯度更新 ===
    g_accumulated = 0
    for inner_iter in range(NUM_INNER_ITERS):
        # 采样失败区域的随机子集
        F_batch = sample_failure_batch(F_*, ratio=BATCH_RATIO)

        # 计算组合损失和梯度
        g_batch = compute_combined_gradient(
            θ, F_batch, V_safe, V_unsafe,
            λ_penalty, λ_stability, λ_barrier,
            translator, dynamics_model
        )
        g_accumulated += g_batch / NUM_INNER_ITERS

    # QP 投影更新
    g_update = qp_project(g_accumulated, J, ...)  # 或 SVD 投影
    θ = θ - lr * g_update

    # 自适应学习率
    lr = adjust_learning_rate(lr, pass_rate, last_pass_rate)

    last_pass_rate = pass_rate
    outer_iter += 1

return θ
```

### 5.2 关键参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_depth_init` | 8 | 初始深度 |
| `MAX_DEPTH` | 13 | 最大深度上限 |
| `NUM_INNER_ITERS` | 5~10 | 内循环迭代次数 |
| `BATCH_RATIO` | 0.3~0.5 | 每次采样的失败区域比例 |
| `lr_init` | 1e-3 (clean) / 1e-2 (v1 QP) | 初始学习率 |
| `γ_safe` | 0.1 | 安全区 margin |
| `γ_unsafe` | 0.1 | 障碍区 margin |
| `depth_fail_ratio_threshold` | 0.3 | 触发深度增加的阈值 |

---

## 6. 三种方法对比与推荐

| 维度 | Clean (直接SGD) | SVD投影 | QP投影 |
|------|----------------|---------|--------|
| **理论基础** | 无约束 SGD | 切向空间投影 | 安全约束优化 |
| **保证不破坏已验证区域** | ❌ | ⚠️ 近似 | ✅ 理论保证 |
| **计算效率** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **实现复杂度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **梯度方差** | 高 | 中 | 低 |
| **步长选择** | 困难 | 中等 | 中等 |
| **可组合改进** | ✅ | ✅ | ✅ |

**推荐方案**：以 **QP 投影** 为基础，结合改进方案 4.1（组合损失）+ 4.2（深度感知）+ 4.3（梯度累积）+ 4.4（自适应步长）。理由：QP 投影有理论保证不破坏已验证区域，在此基础上加入组合损失解决退化解问题是最合理的起点。

---

## 7. 总结

本文档分析了 Neural CBF 验证后修复的三种方法（直接 SGD、SVD 投影、QP 投影），识别了三个核心问题：

1. **退化解问题**：梯度缺乏正向引导，可能优化到无控制能力的退化 CBF
2. **深度耦合问题**：`max_depth` 直接影响失败信号的可靠性，现有方法未考虑这一因素
3. **单步更新问题**：固定步长、无 mini-batch、无自适应机制

提出了六项系统性改进方案，从损失函数设计（组合损失）、验证策略（深度感知分层）到优化算法（梯度累积、Trust Region、自适应步长），形成了一个完整的改进框架。最核心的改进是**组合损失函数**（解决退化解）和**深度感知策略**（区分可靠/不可靠失败信号），这两项改动最小但收益最大。
