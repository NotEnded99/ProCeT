# V5 实现 Prompt

## 任务

基于现有的 V4 代码，写一版新的 V5 代码文件。

## 参考文件

- 主程序：`New_repair/main_v4.py`
- RS Jacobian：`New_repair/geometry_module_new_v4.py`（`compute_jacobian_rs` 函数，整个文件复制到 `geometry_module_new_v5.py`，不改内容）
- QP 投影：`New_repair/optimizer_module_v3.py`（`qp_project_and_update` 函数，复用，不改内容）
- 单纯形边界计算：`New_repair/geometry_module_new.py`（`compute_simplex_bound` 和 `compute_simplex_bound_batch` 函数）

## 核心改进

**用 RS（随机平滑）方法计算 failure 区域的损失梯度，替代 V4 中的 autograd.backward()**。

### 为什么这样做

V4 中 QP 约束用的是 RS Jacobian（对 Crown 下界求梯度），但损失梯度用的是 autograd（对真实函数求梯度）。两者目标函数不一致。V5 改为都用 Crown 下界的 RS 梯度，完全对齐。

### 数学公式

对每个 failure simplex s，RS 损失梯度为：

```
g_s = (1 / (N · σ²)) · Σ_{i=1}^{N} [ ψ_s(θ + εᵢ) · εᵢ ]

其中 εᵢ ~ N(0, σ² I)

ψ_s(θ) 的定义：
  - safe 类型（CBF 违规）: ψ_s = softplus(cbf_margin - min_L)，min_L 是 CBF 下界
  - unsafe 类型（h 违规）: ψ_s = softplus(h_lb + margin)，h_lb 是 h 上界
```

最终 `g_raw = mean(g_s)` over 所有 loss > 0 的 simplices。

## 需要创建的文件

### 1. `New_repair/optimizer_module_v5.py`

实现 `compute_repair_loss_and_grad_rs` 函数，签名如下：

```python
def compute_repair_loss_and_grad_rs(
    model: nn.Module,
    failed_safe_simplices: List,      # List of simplices: F_safe_cbf_violation + F_depth_limit_reached + F_unsafe_cannot_split
    failed_unsafe_simplices: List,     # List of simplices: F_h_positive_in_unsafe
    dynamics_model,
    translator,
    tolerance: float = -1e-12,
    margin: float = 0.0,
    cbf_margin: float = 0.0,
    beta: float = 5.0,
    rs_n: int = 100,
    rs_sigma: float = 0.01,
    grad_clip_norm: float = 10.0,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    """
    用随机平滑（RS）方法计算 failure 区域的损失梯度。

    流程：
        1. 遍历所有 failure simplices
        2. 对每个 simplex，用 RS 公式估计其对参数的梯度 g_s
        3. 对所有 g_s 取平均得到 g_raw
        4. 梯度裁剪后返回

    不使用任何 autograd.backward()，完全通过前向传播 + RS 公式估计梯度。
    """
    # 参考 geometry_module_new.py 的 compute_simplex_bound / compute_simplex_bound_batch
    # 参考 geometry_module_new_v4.py 的 RS 风格参数扰动逻辑
    ...
```

**实现要点**：
- 保存原始参数 `theta_old`
- 对每个 simplex：用 `torch.nn.utils.vector_to_parameters` 扰动参数为 `theta_old + eps`
- 用 `compute_simplex_bound`（safe）或 `compute_simplex_bound`（unsafe, h_ub）做前向计算
- 损失用 `softplus`，与 v3 的 `compute_repair_loss_and_grad` 保持一致
- 扰动后**必须恢复原始参数**
- 最后梯度裁剪

### 2. `New_repair/geometry_module_new_v5.py`

将 `New_repair/geometry_module_new_v4.py` 的**全部内容**复制过去，文件名改为 `_v5`。

### 3. `New_repair/main_v5.py`

基于 `main_v4.py` 修改：

**导入部分**：
```python
# 复用 v4 的 RS Jacobian（不加不减）
from New_repair.geometry_module_new_v5 import compute_jacobian_rs
# 新：RS 损失梯度
from New_repair.optimizer_module_v5 import compute_repair_loss_and_grad_rs
# 复用 v3 的 QP 投影
from New_repair.optimizer_module_v3 import qp_project_and_update
```

**主循环修改**（内循环部分）：

V4 传入的是特征点：
```python
loss_val, g_F = compute_repair_loss_and_grad(
    model=model,
    dynamics_model=dynamics_model,
    failed_safe_feature_points=failed_safe_feature_points,   # 特征点
    failed_unsafe_feature_points=failed_unsafe_feature_points,  # 特征点
    margin=0.1,
    cbf_margin=0.0,
    beta=5.0,
    ...
)
```

V5 改为传入 **failure simplices 列表**：
```python
# 准备 failure simplices 列表
failed_unsafe_simplices = list(F_h_positive_in_unsafe_init)
failed_safe_simplices = (
    list(F_safe_cbf_violation_init) +
    list(F_depth_limit_reached_init) +
    list(F_unsafe_cannot_split_init)
)

loss_val, g_F = compute_repair_loss_and_grad_rs(
    model=model,
    failed_safe_simplices=failed_safe_simplices,
    failed_unsafe_simplices=failed_unsafe_simplices,
    dynamics_model=dynamics_model,
    translator=translator,
    tolerance=-1e-12,
    margin=0.0,
    cbf_margin=0.0,
    beta=5.0,
    rs_n=rs_n,          # 和 RS Jacobian 共享超参数
    rs_sigma=rs_sigma,
    grad_clip_norm=10.0,
    verbose=False,
)
```

**其他不变**：
- 特征点提取代码（`extract_feature_points_from_regions`）仍然保留，但只用于验证/记录，不需要传给损失函数
- QP 投影 `qp_project_and_update` 调用不变
- 输出文件加 `_v5` 后缀
- 命令行参数不变（rs_n, rs_sigma 等）

## 注意事项

1. `compute_simplex_bound` 和 `compute_simplex_bound_batch` 来自 `geometry_module_new.py`，在 `optimizer_module_v5.py` 中用 `from New_repair.geometry_module_new import compute_simplex_bound, compute_simplex_bound_batch` 导入
2. RS Jacobian 的 `compute_jacobian_rs` 内部已经实现了参数扰动和恢复的逻辑，损失梯度的 RS 实现要复用相同的模式
3. `softplus` 用 `torch.nn.functional.softplus`
4. 所有 `tensor.item()` 用于提取 Python scalar
5. 保持随机种子固定（SEED=42）
