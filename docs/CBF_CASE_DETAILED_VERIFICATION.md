# CBF验证Case 1-3详细流程

基于论文：**Scalable Verification of Neural Control Barrier Functions Using Linear Bound Propagation**

命令：`python3 experiments/barrier_certificate.py --system-type barr2 --verify --max-depth 13`

---

## 目录

1. [概述](#概述)
2. [Case 1: 不安全区域](#case-1-不安全区域)
3. [Case 2: 边界区域](#case-2-边界区域)
4. [Case 3: 安全区域](#case-3-安全区域)
5. [验证结果处理](#验证结果处理)
6. [关键数据结构](#关键数据结构)

---

## 概述

CBF验证的核心逻辑在 `verify_cbf.py` 的 `_verify_batch_linbndprop` 函数中。验证过程首先通过CROWN前向传播计算神经网络输出 `h(x)` 的界，然后根据界值将区域分类到不同的Case进行处理。

### 验证流程总览

```
verify_batch(batch)
    │
    ▼
┌─────────────────────────────────────┐
│ 步骤1: 计算网络输出界            │
└─────────────────────────────────────┘
    │
    └─ network_linearizer.compute_network_bounds(batch)
    │
    └─ 每个区域得到 h(x) ∈ [h_min, h_max]
    │
    ▼
┌─────────────────────────────────────┐
│ 步骤2: 遍历每个区域              │
└─────────────────────────────────────┘
    │
    └─ for sample_idx, sample in enumerate(batch)
    │
    ▼

    for 每个区域
        │
        ├─ Case 1: h_max < 0
        │   └─ 不安全区域
        │
        ├─ Case 2: 与真正不安全集相交
        │   ├─ Subcase 2a: h_min >= 0 → UNSAT
        │   └─ Subcase 2b: h_min < 0 → MAYBE (分割)
        │
        └─ Case 3: 安全区域
            └─ 验证CBF条件
```

---

## Case 1: 不安全区域

### 判断条件

```python
if h_max < 0:
    # Case 1: h(x) < 0 everywhere on this region
```

**条件解析**：
- `h_max < 0` 表示障碍函数在整个区域上严格为负
- 根据CBF定义，h(x) < 0 对应不安全区域

### 验证逻辑

```
Case 1: h_max < 0
    │
    ├─┐
    │   含义: 障碍函数在整个区域上为负
    │   │   └───┘
    │   │   │
    │   ├─┐
    │   │   操作: 直接标记为 SAT
    │   │   │   └───┘
    │   │   │
    │   └─ SampleResultSAT(result_type="unsafe_region")
    │
    └───┘
```

### 代码实现

```python
# verify_cbf.py:353-355
if h_max < 0:
    # Region is correctly classified as unsafe
    results[sample_idx] = SampleResultSAT(sample, start_time, result_type="unsafe_region")
```

### 为什么SAT？

CBF条件只在安全区域（h(x) ≥ 0）上需要验证：
- **不安全区域**：h(x) < 0，不需要验证CBF条件
- 正确性条件：h(x) < 0 的区域应该在真正的不安全集内

### 结果

| 情况 | 结果类型 | 说明 |
|------|----------|------|
| h_max < 0 | `unsafe_region` (SAT) | 区域被正确识别为不安全，无需进一步处理 |

---

## Case 2: 边界区域

### 判断条件

```python
elif unsafe_region(sample, dynamics_model, require_complete_containment=False):
    # Case 2: region contains parts of the unsafe set
```

**条件解析**：
- `unsafe_region()` 检查区域是否与真正不安全集相交
- `require_complete_containment=False` 表示部分相交即可

### 子情况分类

```
Case 2: 与真正不安全集相交
    │
    ├─┐
    │   检查: unsafe_region(sample, dynamics_model)
    │   └───┘
    │   │
    ├─ Subcase 2a: h_min >= 0
    │   │   │
    │   │   ├─┐
    │   │   │   含义: h 在真正不安全集上为正（违规）
    │   │   │   │   └───┘
    │   │   │   │   │
    │   │   │   ├─┐
    │   │   │   │   操作: 返回反例 UNSAT
    │   │   │   │   │   └───┘
    │   │   │   │   │
    │   │   │   └─ SampleResultUNSAT(result_type="h_positive_in_unsafe")
    │   │   │   │
    │   │   │   └───┘
    │   │   │
    │   └─ Subcase 2b: h_min < 0
    │       │   │
    │       ├─┐
    │       │   含义: 需要分割
    │       │   │   └───┘
    │       │   │
    │       ├─┐
    │       │   操作: 调用 _handle_split()
    │       │   │   └───┘
    │       │   │
    │       └─ SampleResultMAYBE(result_type="case_1_boundary_unsafe")
    │
    └───┘
```

### Subcase 2a: h_min >= 0 (违规)

**判断逻辑**：
- `h_min >= 0` 表示 h(x) 在整个区域上非负
- 但区域与真正不安全集相交
- **矛盾**：在不安全集上应该有 h(x) < 0

**代码实现**：

```python
# verify_cbf.py:362-365
if h_min >= 0:
    # This is a VIOLATION: h(x) >= 0 but region contains true unsafe set
    counterexample = sample.center
    results[sample_idx] = SampleResultUNSAT(
        sample, start_time, [counterexample],
        result_type="h_positive_in_unsafe"
    )
```

**为什么UNSAT？**
- CBF定义要求：在不安全集上 h(x) < 0
- 这里发现：在不安全集上 h(x) ≥ 0
- **违反CBF定义**，返回反例

### Subcase 2b: h_min < 0 (需要分割)

**判断逻辑**：
- `h_min < 0` 表示 h(x) 在区域上可能为负
- 区域与不安全集相交
- 无法确定 h(x) 在不安全集内是否总是负
- 需要分割细化

**`_handle_split()` 逻辑**：

```python
# verify_cbf.py:367-376
else:
    CBFVerificationStrategy._handle_split(
        sample=sample,
        start_time=start_time,
        results=results,
        sample_idx=sample_idx,
        min_volume=min_volume,
        split_type="case_1_boundary_unsafe",
        unsat_type="unsafe_cannot_split",
        max_depth=max_depth,
    )
```

**分割决策**：

```
_handle_split()
    │
    ├─ 条件1: 深度检查
    │   │
    │   ├─ 如果 depth >= max_depth:
    │   │   └─ UNSAT (depth_limit_reached)
    │   │
    │   └─ 否: 继续
    │
    ├─ 条件2: 体积检查
    │   │
    │   ├─ 如果 volume > min_volume:
    │   │   └─ 分割区域 → MAYBE
    │   │
    │   └─ 否: 继续
    │
    └─ 条件3: 默认
        │
        └─ UNSAT (unsafe_cannot_split)
```

### Case 2 结果总结

| 子情况 | 条件 | 结果类型 | 说明 |
|--------|------|----------|------|
| 2a | h_min >= 0 | `h_positive_in_unsafe` (UNSAT) | 在不安全集上h为正，违反CBF定义 |
| 2b-1 | h_min < 0, 达到max_depth | `depth_limit_reached` (UNSAT) | 达到深度限制，无法进一步分割 |
| 2b-2 | h_min < 0, 体积太小 | `unsafe_cannot_split` (UNSAT) | 区域太小，无法分割 |
| 2b-3 | h_min < 0, 可分割 | `case_1_boundary_unsafe` (MAYBE) | 分割后继续验证 |

---

## Case 3: 安全区域

### 判断条件

```python
else:
    # Case 3: h(x) >= 0 somewhere on this region
    # Region is classified as safe thus we need to verify CBF condition
    to_check_cbf_cond.append(sample_idx)
    reason.append("case_2")
```

**条件解析**：
- 不满足Case 1（h_max < 0）
- 不满足Case 2（与不安全集相交）
- 默认进入Case 3：需要验证CBF条件

### 验证流程

```
Case 3: 安全区域
    │
    ├─┐
    │   含义: h(x) >= 0 且不在不安全集内
    │   │   └───┘
    │   │
    ├─ 添加到 to_check_cbf_cond 列表
    │
    ├─┐
    │   步骤1: 计算Jacobian界
    │   │   └───┘
    │   │
    │   └─ compute_partial_derivative_bounds()
    │       │
    │       └─ ∇h(x) ∈ [J_L, J_U]
    │
    ├─┐
    │   步骤2: eta迭代验证
    │   │   └───┘
    │   │
    │   └─ for eta in [(0.5, 0.5)]
    │       │
    │       └─ _verify_cbf_condition_affine(subbatch, eta)
    │
    └───┘
```

### `_verify_cbf_condition_affine` 详细流程

```
_verify_cbf_condition_affine()
    │
    ├─┐
    │   1. 泰勒展开动力学函数
    │   │   └───┘
    │   │
    │   ├─ f(x) ∈ [A_L^f x + b_L^f, A_U^f x + b_U^f]
    │   └─ g(x) ∈ [A_L^g x) + b_L^g, A_U^g x + b_U^g]
    │
    ├─┐
    │   2. 计算Jacobian界
    │   │   └───┘
    │   │
    │   └─ ∇h(x) ∈ [J_L, J_U]
    │
    ├─┐
    │   3. 计算漂移项下界
    │   │   └───┘
    │   │
    │   ├─ M_D, c_D = McCormick_lower(∇h, f, η=0.5)
    │   └─ drift_lower = sum(M_D · x + c_D)
    │
    ├─┐
    │   4. 计算控制项下界 (如果有控制)
    │   │   └───┘
    │   │
    │   ├─ v(x) = ∇h(x) · g(x)
    │   ├─ v_lower = McCormick_lower(∇h, g, η=0.5)
    │   └─ control_lower = sup_u v_lower · u
    │       │
    │       ├─ v_lower ≥ 0: 用 u_max
    │       ├─ v_lower ≤ 0: 用 u_min
    │       └─ 否则: 在边界上取最大值
    │
    ├─┐
    │   5. 计算class-K项下界
    │   │   └───┘
    │   │
    │   └─ alpha_lower = α(h_min)
    │
    ├─┐
    │   6. 最终CBF条件下界
    │   │   └───┘
    │   │
    │   ├─ M_total = M_D + M_C + α系数
    │   ├─ c_total = c_D + c_C + α常数
    │   └─ cbf_lower(x) = M_total · x + c_total
    │
    ├─┐
    │   7. 检查下界是否非负
    │   │   └───┘
    │   │
    │   ├─ min_L = min_{x∈region} cbf_lower(x)
    │   └─ satisfaction = min_L >= -1e-12
    │
    └─ 返回 satisfaction
```

### 关键数学细节

#### 1. 漂移项：∇h · f

使用McCormick包络计算下界：

```
M_D, c_D = McCormick_lower(J_L, J_U, f_L, f_U, η=0.5)
```

McCormick下界公式：
```
C1 = η · min(J) + (1-η) · max(J)
C2 = η · min(f) + (1-η) · max(f)
M_D = C1_pos · f_L + C1_neg · f_U + C2_pos · J_L + C2_neg · J_U
c_D = -(η · min(J) · min(f) + (1-η) · max(J) · max(f))
```

#### 2. 控制项：sup_u ∇h · g · u

```python
# 先计算 v(x) = ∇h(x) · g(x) 的下界
v_L, v_U = McCormick_lower(J, g, η=0.5)

# 然后求 sup_u v(x) · u
for each control dimension:
    if v_L >= 0:
        # v总是正，用最大控制输入
        term += v_L · u_max
    elif v_U <= 0:
        # v总是负，用最小控制输入
        term += v_L · u_min
    else:
        # v变号，在边界上评估
        term += max(v_L·u_min, v_L·u_max)
```

#### 3. Class-K项：α(h)

```python
# 使用h_min计算保守的下界
alpha_lower = alpha_function(h_min)
```

对于常见的α函数：
- `α(h) = h`: 直接使用h_min
- `α(h) = h²`: 如果h_min ≥ 0，则用h_min²；否则用0

#### 4. 最终条件

```python
# 合并所有项
M_total = M_D + M_C + alpha_A_L
c_total = c_D + c_C + alpha_a_L

# 在区域上求最小值
min_L = min_{x∈region} (M_total · x + c_total)

# 检查非负性
satisfaction = min_L >= -1e-12
```

### 验证结果处理

```python
# verify_cbf.py:429-450
for subsample_idx, sample_idx in enumerate(to_check_cbf_cond):
    sample = batch[sample_idx]

    if cbf_verified[subsample_idx]:
        # 验证通过
        results[sample_idx] = SampleResultSAT(
            sample, start_time,
            result_type="safe_cbf_verified"
        )
    elif find_counterexample and counter_verified[subsample_idx]:
        # 找到反例
        results[sample_idx] = SampleResultUNSAT(
            sample, start_time, [sample.center],
            result_type="safe_cbf_violation"
        )
    else:
        # 需要分割
        _handle_split(...)
```

### Case 3 结果总结

| 情况 | 条件 | 结果类型 | 说明 |
|------|------|----------|------|
| 验证通过 | min_L >= -1e-12 | `safe_cbf_verified` (SAT) | CBF条件在整个区域上成立 |
| 验证失败-反例 | min_L < -1e-12, counter_verified=True | `safe_cbf_violation` (UNSAT) | 确认CBF条件不成立 |
| 需要分割 | min_L < -1e-12, counter_verified=False | `case_2_cbf_failure` (MAYBE) | 无法确定，需要分割 |

### 为什么有时不直接判定失败？

当 `min_L < -1e-12` 时有三种可能性：

1. **真实违规**：CBF条件确实不满足
2. **界太保守**：McCormick包络的界太松
3. **线性近似误差**：Taylor展开的误差

分割策略：
- 在小区域上，界更紧，可能证明CBF成立
- 如果达到深度或体积限制，则判定为UNSAT

---

## 验证结果处理

### 三种结果类型

```
SampleResultSAT
    │
    ├─ result_type: "unsafe_region" 或 "safe_cbf_verified"
    ├─ sample: 验证通过的区域
    └─ 说明: 区域验证完成，无需进一步处理

SampleResultUNSAT
    │
    ├─ result_type: 各种违规类型
    ├─ sample: 失败的区域
    ├─ counterexamples: 反例点列表
    └─ 说明: 发现CBF违规，停止验证

SampleResultMAYBE
    │
    ├─ result_type: 需要进一步处理的类型
    ├─ sample: 原始区域
    ├─ new_samples: 分割后的子区域
    └─ 说明: 将子区域加入队列继续验证
```

### 结果类型完整列表

| 结果类型 | 分类 | 说明 |
|----------|------|------|
| `unsafe_region` | SAT | 区域被正确识别为不安全 |
| `safe_cbf_verified` | SAT | CBF条件在安全区域上验证通过 |
| `h_positive_in_unsafe` | UNSAT | 在不安全集上h为正，违反定义 |
| `safe_cbf_violation` | UNSAT | CBF条件不成立 |
| `depth_limit_reached` | UNSAT | 达到最大深度，无法验证 |
| `unsafe_cannot_split` | UNSAT | 边界区域无法分割 |
| `case_1_boundary_unsafe` | MAYBE | 边界区域，需要分割 |
| `case_2_cbf_failure` | MAYBE | CBF验证不确定，需要分割 |

---

## 关键数据结构

### SimplicialRegion

```python
SimplicialRegion
    │
    ├─ vertices: 单纯形的顶点
    │   │   └─ shape: (n+1, input_dim)
    │
    ├─ depth: 分割深度
    │
   ├─ methods:
    │   ├─ _compute_volume(): 计算体积
    │   ├─ center: 质心
    │   └─ split(): 分割成两个子单纯形
```

### 仿射界

```python
仿射界表示: (A, b)
    │
    ├─ A: 系数矩阵
    │   │   └─ shape: (batch, ..., n, m)
    │
    └─ b: 常数项
        │   └─ shape: (batch, ..., n)

表示的函数: f(x) = A · x + b
```

### CROWN线性化器

```python
CrownPartialLinearization
    │
    ├─ forward_bounds: 各层前向界
    │   │   ├─ [layer, batch, ...]: (A_L, b_L, A_U, b_U)
    │   │
    │
    ├─ partial_deriv_bounds: Jacobian界
    │   │   └─ [output, batch, ...]: (A_L, b_L, A_U, b_U)
    │   │
    │
    └─ methods:
        ├─ compute_network_bounds(batch): 前向传播
        └─ compute_partial_derivative_bounds(...): 反向传播
```

---

## 总结

### 验证策略的本质

| Case | 目标 | 验证内容 |
|------|------|----------|
| 1 | 检查不安全区域 | h(x) < 0 的区域应在不安全集内 |
| 2 | 处理边界 | h(x) 在不安全集边界上的行为 |
| 3 | 验证CBF条件 | 在安全区域上验证 ∇h·f + sup_u∇h·g·u + α(h) ≥ 0 |

### 声音性与完备性

**声音性**（Soundness）：
- 如果返回SAT，CBF条件一定成立
- 通过保守的界计算保证

**完备性**（Completeness）：
- 如果CBF条件成立，最终会返回SAT
- 通过分割策略逐步细化

### 参数调优

| 参数 | 默认值 | 影响 |
|------|--------|------|
| max_depth | 15 | 最大分割深度 |
| min_volume | 1e-8 | 最小可分割体积 |
| eta | (0.5, 0.5) | McCormick包络参数 |
| 容差 | 1e-12 | 数值判断容差 |

---

## 相关文件

- 主验证逻辑: `lbp_neural_cbf/cbf/verify_cbf.py`
- CROWN线性化: `lbp_neural_cbf/linearization/crown.py`
- 区域分割: `lbp_neural_cbf/regions/simplicial_region.py`
- 主入口: `experiments/barrier_certificate.py`
