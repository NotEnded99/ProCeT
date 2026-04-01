# CBF验证中的分割决策逻辑详细解释

问题：区域验证不通过时，如何判断是否需要分割？

---

## 目录

1. [概述](#概述)
2. [分割决策的三个条件](#分割决策的三个条件)
3. [详细代码分析](#详细代码分析)
4. [判断流程图](#判断流程图)
5. [参数含义说明](#参数含义说明)
6. [示例场景分析](#示例场景分析)

---

## 概述

当CBF验证在某个区域上失败时，系统面临三个选择：
1. **分割区域**：将该区域分成更小的子区域，然后继续验证
2. **返回反例（UNSAT）**：认为CBF条件确实被违反
3. **标记为深度限制**：达到最大深度，无法继续分割

决策逻辑在 `_handle_split()` 函数中实现，该函数检查两个主要条件：
- **深度限制**：是否达到允许的最大分割深度
- **体积限制**：区域体积是否足够大以进行有效分割

---

## 分割决策的三个条件

### 条件1：深度限制检查

**优先级**：最高（最先检查）

**代码**：
```python
if max_depth is not None and sample.depth >= max_depth:
    # 达到最大深度，无法继续分割
    counterexample = sample.center
    results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample],
                                          result_type="depth_limit_reached")
    return
```

**判断逻辑**：
- `max_depth`：命令行参数 `--max-depth 13` 设置的最大分割深度
- `sample.depth`：当前区域已经经过的分割次数
- 如果 `sample.depth >= max_depth`：
  - 禁止进一步分割
  - 返回 UNSAT 结果，类型为 `"depth_limit_reached"`
  - 使用区域中心作为反例

**含义**：
- 防止无限细分导致计算不终止
- 提供可预测的资源使用上限
- 如果在最大深度仍无法验证，返回"不确定"结果

---

### 条件2：体积限制检查

**优先级**：中等（深度检查之后）

**代码**：
```python
if sample._compute_volume() > min_volume:
    new_samples = sample.split()
    if new_samples:
        results[sample_idx] = SampleResultMaybe(sample, start_time, new_samples,
                                              split_type=split_type)
        return
```

**判断逻辑**：
- `min_volume`：默认值 `1e-8`（函数参数）
- `sample._compute_volume()`：计算当前区域的体积
- 如果 `volume > min_volume`：
  - 调用 `sample.split()` 分割区域
  - 如果成功生成新区域：
    - 返回 MAYBE 结果，包含子区域列表
    - 执行器会将这些子区域加入队列继续验证
  - 如果分割失败：
    - 继续到条件3

**含义**：
- 只分割"足够大"的区域
- 避免分割极小的区域（数值不稳定、效益低）
- 单纯形的最小半径设为 `1e-3 * max_edge_length`

---

### 条件3：无法分割时的默认处理

**优先级**：最低（前两个条件都不满足时）

**代码**：
```python
counterexample = sample.center
results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample],
                                      result_type=unsat_type)
```

**触发场景**：
1. 达到最大深度（如果条件1通过，此行不执行）
2. 区域体积 ≤ min_volume 且无法进一步分割
3. 分割操作返回空列表（虽然很少见）

**含义**：
- 返回"确定的"失败
- 使用区域中心作为可能的反例
- 结果类型取决于失败的上下文（例如 `"unsafe_cannot_split"`）

---

## 详细代码分析

### _handle_split() 完整函数

**文件位置**：`lbp_neural_cbf/cbf/verify_cbf.py:245-262`

```python
@staticmethod
def _handle_split(sample, start_time, results, sample_idx,
                min_volume, split_type, unsat_type, max_depth=None):
    """
    记录 MAYBE 结果（通过分割）或 UNSAT 反例（如果无法分割或达到深度限制）

    参数：
        sample: 当前验证的区域
        start_time: 验证开始时间
        results: 结果数组
        sample_idx: 区域在批处理中的索引
        min_volume: 允许分割的最小体积
        split_type: 分割类型标识（用于统计）
        unsat_type: UNSAT结果的类型标识
        max_depth: 最大允许深度
    """

    # === 条件1：检查最大深度 ===
    if max_depth is not None and sample.depth >= max_depth:
        # 达到深度限制，不能再分割
        counterexample = sample.center
        results[sample_idx] = SampleResultUNSAT(
            sample, start_time,
            [counterexample],
            result_type="depth_limit_reached"
        )
        return  # 早期返回，不执行后续代码

    # === 条件2：检查体积 ===
    if sample._compute_volume() > min_volume:
        # 区域足够大，可以分割
        new_samples = sample.split()  # 调用区域特定的分割方法

        if new_samples:  # 检查分割是否成功
            # 分割成功，返回 MAYBE 结果
            # 执行器会将 new_samples 加入队列继续处理
            results[sample_idx] = SampleResultMaybe(
                sample, start_time,
                new_samples,
                split_type=split_type
            )
            return  # 早期返回，不执行后续代码

    # === 条件3：默认情况 - 无法分割 ===
    # （达到此处意味着：要么达到深度限制，要么体积太小无法分割）
    counterexample = sample.center
    results[sample_idx] = SampleResultUNSAT(
        sample, start_time,
        [counterexample],
        result_type=unsat_type
    )
    # 注意：此处没有return，函数自然结束
```

### 调用路径分析

从 `_verify_batch_linbndprop()` 函数中，有两个地方调用 `_handle_split()`：

#### 调用点1：边界/混合区域处理

**文件位置**：`lbp_neural_cbf/cbf/verify_cbf.py:367-376`

```python
elif unsafe_region(sample, dynamics_model, require_complete_containment=False):
    if h_min >= 0:
        # 违规：h(x) >= 0 但区域包含真正的不安全集
        counterexample = sample.center
        results[sample_idx] = SampleResultUNSAT(
            sample, start_time,
            [counterexample],
            result_type="h_positive_in_unsafe"
        )
    else:
        # 需要分割
        CBFVerificationStrategy._handle_split(
            sample=sample,
            start_time=start_time,
            results=results,
            sample_idx=sample_idx,
            min_volume=min_volume,
            split_type="case_1_boundary_unsafe",  # 标识分割类型
            unsat_type="unsafe_cannot_split",  # 如果无法分割，返回此类型
            max_depth=max_depth,
        )
```

**场景**：区域跨越安全/不安全边界，但 `h_min < 0`（障碍函数在部分区域为负）
**参数传递**：
- `split_type="case_1_boundary_unsafe"`：用于统计和调试
- `unsat_type="unsafe_cannot_split"`：如果无法分割，返回此UNSAT类型

#### 调用点2：CBF条件验证失败

**文件位置**：`lbp_neural_cbf/cbf/verify_cbf.py:439-448`

```python
else:
    CBFVerificationStrategy._handle_split(
        sample=sample,
        start_time=start_time,
        results=results,
        sample_idx=sample_idx,
        min_volume=min_volume,
        split_type="case_2_cbf_failure" if reason[subsample_idx] == "case_2" else "case_3_fallback",
        unsat_type="safe_cbf_violation",  # 如果无法分割，返回此类型
        max_depth=max_depth,
    )
```

**场景**：区域被归类为安全，但CBF条件验证失败
**参数传递**：
- `split_type`：根据失败原因（`case_2_cbf_failure` 或 `case_3_fallback`）
- `unsat_type="safe_cbf_violation"`：如果无法分割，返回此UNSAT类型

---

## 判断流程图

### 完整决策流程

```
验证失败，调用 _handle_split()
    │
    ▼
┌─────────────────────────┐
│ 条件1：深度检查    │
└─────────────────────────┘
    │
    ├─ max_depth is None？
    │   └─ 是：跳过深度检查，继续到条件2
    │
    └─ sample.depth >= max_depth？
        ├─ 是：达到深度限制
        │   ├─ 返回 UNSAT
        │   └─ result_type = "depth_limit_reached"
        │   （函数返回，不继续）
        │
        └─ 否：继续到条件
    │
    ▼
┌─────────────────────────┐
│ 条件2：体积检查    │
└─────────────────────────┘
    │
    ├─ volume > min_volume？
    │   │
    │   └─ 是：区域足够大
    │       │
    │       ├─ 调用 sample.split()
    │       │   ├─ 分割成功（返回新区域）？
    │       │   │   │
    │       │   │   └─ 是：返回 MAYBE
    │       │   │       （执行器将子区域加入队列）
    │       │   │
    │       │   └─ 否：继续到条件3
    │       │           （虽然很少见，但理论上可能）
    │   │
    │       └─ 否：区域太小，继续到条件3
    │
    └─ ▼
        │
        ┌─────────────────────────┐
        │ 条件3：默认处理      │
        └─────────────────────────┘
            │
            ├─ 返回 UNSAT
            └─ result_type = unsat_type
                （可能是 "unsafe_cannot_split" 或 "safe_cbf_violation"）
```

### 决策真值表

| 深度检查 | 体积检查 | 结果 | 原因 |
|---------|---------|------|------|
| depth ≥ max_depth | （不检查） | UNSAT (depth_limit_reached) | 达到最大深度 |
| depth < max_depth | volume > min_volume | MAYBE | 区域可分割 |
| depth < max_depth | volume ≤ min_volume | UNSAT (unsat_type) | 区域太小 |
| max_depth is None | volume > min_volume | MAYBE | 无深度限制 |
| max_depth is None | volume ≤ min_volume | UNSAT (unsat_type) | 区域太小 |

---

## 参数含义说明

### max_depth 参数

**来源**：命令行参数 `--max-depth 13`

**默认值**：`None`（在 `main()` 函数中设置）

**作用**：
- 限制验证的搜索深度
- 防止无限递归/循环
- 提供可预测的计算时间和资源使用

**在代码中的使用**：
```python
# experiments/barrier_certificate.py:225-226
parser.add_argument("--max-depth", type=int, default=15,
                   help="Maximum depth for region splitting (None for unlimited)")

# verify_cbf.py:225
results = verify_cbf(
    ...
    max_depth=max_depth
)
```

**对验证的影响**：
- 较小的 `max_depth`：更快但可能无法验证通过（区域不够细）
- 较大的 `max_depth`：更精确但计算时间更长
- `max_depth = None`：不限制深度（可能无限循环）

### min_volume 参数

**来源**：`_verify_batch_linbndprop()` 函数的默认参数

**默认值**：`1e-8`

**代码**：
```python
# verify_cbf.py:319
def _verify_batch_linbndprop(
    ...,
    min_volume=1e-8,  # 默认最小体积
    ...
):
```

**作用**：
- 防止分割极小的区域
- 避免数值不稳定（过小的区域会导致梯度/导数界不准确）
- 控制细化效率（分割过小区域得不偿失）

### split_type 参数

**可能的值**：
1. `"case_1_boundary_unsafe"`：区域跨越安全/不安全边界
2. `"case_2_cbf_failure"`：CBF条件验证失败（Case 2）
3. `"case_3_fallback"`：其他验证失败（Case 3 fallback）

**作用**：
- 统计目的：记录不同类型的分割
- 调试目的：理解哪些区域导致分割

**在统计中的使用**：
```python
# stats.py 或类似代码中
split_stats["case_2_cbf_failure"] += 1
split_stats["case_3_fallback"] += 1
```

### unsat_type 参数

**可能的值**：
1. `"unsafe_cannot_split"`：边界区域无法分割
2. `"safe_cbf_violation"`：安全区域CBF条件失败
3. `"depth_limit_reached"`：达到最大深度

**作用**：
- 标记UNSAT结果的失败原因
- 帮助用户理解失败性质

---

## 示例场景分析

### 场景1：深度限制

**初始条件**：
- `max_depth = 5`
- `sample.depth = 5`

**执行过程**：
```
└─ 检查条件1：depth >= max_depth
   └─ 5 >= 5：真！
      ├─ 达到最大深度
      ├─ 返回 UNSAT
      └─ result_type = "depth_limit_reached"
```

**结果**：返回UNSAT，不检查体积

**含义**：即使区域可能还可以分割，也不允许继续

---

### 场景2：正常分割

**初始条件**：
- `max_depth = 13`
- `sample.depth = 3`
- `sample.volume = 1e-5`（例如）
- `min_volume = 1e-8`

**执行过程**：
```
└─ 检查条件1：depth >= max_depth
   └─ 3 >= 13：假！
      （继续到条件2）
      │
      ▼
└─ 检查条件2：volume > min_volume
   └─ 1e-5 > 1e-8：真！
      ├─ 区域足够大
      ├─ 调用 sample.split()
      │   ├─ 生成两个子区域
      │   └─ depth = 4
      ├─ 返回 MAYBE
      └─ 包含新区域
```

**结果**：返回MAYBE，包含两个子区域（depth=4）

**执行器行为**：将两个子区域加入队列，继续深度优先搜索

---

### 场景3：区域太小

**初始条件**：
- `max_depth = 13`
- `sample.depth = 8`
- `sample.volume = 1e-10`（例如）
- `min_volume = 1e-8`

**执行过程**：
```
└─ 检查条件1：depth >= max_depth
   └─ 8 >= 13：假！
      （继续到条件2）
      │
      ▼
└─ 检查条件2：volume > min_volume
   └─ 1e-10 > 1e-8：假！
      （继续到条件3）
      │
      ▼
└─ 条件3：默认处理
   ├─ 返回 UNSAT
   └─ result_type = "safe_cbf_violation"（或 unsat_type 传入的值）
```

**结果**：返回UNSAT，使用区域中心作为反例

**含义**：区域太小，无法有意义地分割，视为验证失败

---

### 场景4：无深度限制

**初始条件**：
- `max_depth = None`
- `sample.depth = 20`（很大的深度）
- `sample.volume = 1e-4`
- `min_volume = 1e-8`

**执行过程**：
```
└─ 检查条件1：max_depth is not None
   └─ None is not None：假！
      （跳过深度检查，直接到条件2）
      │
      ▼
└─ 检查条件2：volume > min_volume
   └─ 1e-4 > 1e-8：真！
      ├─ 区域足够大
      ├─ 调用 sample.split()
      └─ 返回 MAYBE
```

**结果**：可以无限深度地分割

**风险**：可能导致无限循环（需要其他终止条件，如所有区域达到SAT/UNSAT）

---

## 区域分割的实现

### 单纯形分割（SimplicialRegion）

**文件位置**：`lbp_neural_cbf/regions/simplicial.py:320-360`

```python
def split(self, ...):
    """分割单形形，通过平分其最长边"""
    max_length, (v1_idx, v2_idx) = self.get_max_edge_length()

    # 计算最长边的中点
    midpoint = (self.vertices[v1_idx] + self.vertices[v2_idx]) / 2

    # 创建两个新的单形形
    vertices1 = self.vertices.copy()
    vertices1[v2_idx] = midpoint  # 将 v2 替换为中点

    vertices2 = self.vertices.copy()
    vertices2[v1_idx] = midpoint  # 将 v1 替换为中点

    # 创建新区域，深度加1
    region1 = SimplicialRegion(vertices1, ..., depth=self.depth + 1)
    region2 = SimplicialRegion(vertices2, ..., depth=self.depth + 1)

    return region1, region2
```

**分割策略**：
1. 找到单形的最长边
2. 在边的中点处分割
3. 生成两个新单形，共享中点和新边

**为什么选择最长边**：
- 减少狭长、退化区域的产生
- 保持单形形状相对均匀
- 提高数值稳定性

---

### 超矩形分割（HyperrectangularRegion）

**概念**：
```python
def split(self, ...):
    """分割超矩形，在选定维度的中点处"""
    # 选择分割维度（例如交替使用维度）
    split_dim = self._determine_split_dimension(...)

    # 计算分割点
    midpoint = (self.lower_bound[split_dim] + self.upper_bound[split_dim]) / 2

    # 创建两个新超矩形
    lower1 = self.lower_bound.copy()
    upper1 = self.upper_bound.copy()
    upper1[split_dim] = midpoint

    lower2 = self.lower_bound.copy()
    upper2 = self.upper_bound.copy()
    lower2[split_dim] = midpoint

    # 创建新区域，深度加1
    region1 = HyperrectangularRegion(lower1, upper1, ..., depth=self.depth + 1)
    region2 = HyperrectangularRegion(lower2, upper2, ..., depth=self.depth + 1)

    return region1, region2
```

**分割策略**：
- 交替使用不同维度进行分割
- 在选定维度的中点处分割
- 保持其他维度不变

---

## 总结

### 分割决策的三级判断

1. **第一级（最高优先级）**：深度限制
   - 检查：`sample.depth >= max_depth`
   - 如果达到：返回 UNSAT（深度限制）
   - 如果 `max_depth is None`：跳过此检查

2. **第二级（中等优先级）**：体积限制
   - 检查：`sample._compute_volume() > min_volume`
   - 如果足够大：执行 `sample.split()`
   - 如果分割成功：返回 MAYBE（继续验证子区域）

3. **第三级（默认处理）**：无法分割
   - 触发：达到深度限制 或 区域太小
   - 返回：UNSAT（使用区域中心作为反例）

### 关键设计原则

1. **资源限制**：通过 `max_depth` 防止无限细分
2. **数值稳定性**：通过 `min_volume` 避免分割过小区域
3. **深度优先**：使用 LIFO 队列优先完成子树的验证
4. **统计透明**：通过 `split_type` 和 `unsat_type` 记录失败原因

### 验证者视角

对于用户/开发者：
- **100% 通过**：所有区域达到 SAT
- **部分通过**：某些区域 SAT，某些 UNSAT（可能是真正的CBF违规）
- **无法验证**：存在 "depth_limit_reached" 或 "volume_limit" 结果
  - 这不意味着CBF肯定失败
  - 而是验证方法达到极限
  - 可能需要：增加 max_depth、减小 min_volume、或使用更精确的方法

---

## 相关文件索引

| 文件 | 函数 | 行号 |
|------|------|------|
| `lbp_neural_cbf/cbf/verify_cbf.py` | `_handle_split()` | 245-262 |
| `lbp_neural_cbf/cbf/verify_cbf.py` | `_verify_batch_linbndprop()` | 312-452 |
| `lbp_neural_cbf/regions/simplicial.py` | `split()` | 320-360 |
| `lbp_neural_cbf/regions/hyperrectangular.py` | `split()` | （类似） |
