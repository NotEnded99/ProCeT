# V5 代码生成 Prompt

请基于现有的 V4 代码（`New_repair/main_v4.py`）和 V3 几何模块（`New_repair/geometry_module_new_v3.py`），
编写一个 **V5 版本的 Neural CBF 迭代修复代码**。

---

## 核心改进：RS 损失梯度（方案 A — 完全对齐）

### 改进思想

将 failure 区域的损失梯度计算也改为随机平滑（RS）风格，使 **QP 约束的 Jacobian** 和 **损失函数的梯度** 在同一个目标函数空间（Crown 下界 `ψ_s(θ)`）中对齐。

### 数学公式

**RS 损失梯度**（对单个 failure simplex s）:
```
g_s = (1/(N · σ²)) · Σ_{i=1}^{N} [ ψ_s(θ + εᵢ) · εᵢ ]
```
其中：
- `ψ_s(θ)` 是 Crown 下界（对 safe 区域）或 h 上界（对 unsafe 区域）
- `εᵢ ~ N(0, σ² I)` 是高斯噪声
- N 和 σ 是 RS 超参数

**最终梯度**: 对所有有正损失的 failure simplex 的 g_s 取平均

### 为什么这样做

| | V4 方法 | V5 方法（本次改进） |
|---|---|---|
| J_verified（QP 约束） | RS Jacobian：`∂ψ/∂θ` | RS Jacobian：`∂ψ/∂θ`（不变） |
| 损失梯度 g_F | Autograd：`∂L/∂θ`（精确但有次梯度问题） | RS 梯度：`∂ψ/∂θ` 的无偏估计 |
| **一致性** | 中等（Crown 下界 vs 精确函数） | **完全一致**（两者都是 Crown 下界的梯度） |
| ReLU 次梯度问题 | 存在 | **不存在**（RS 不需要 autograd.backward） |

---

## 代码要求

### 1. 新文件结构

```
New_repair/
  ├── main_v5.py                      # 主程序（修改自 v4）
  ├── optimizer_module_v5.py          # 优化器模块（核心改进）
  └── geometry_module_new_v5.py        # 几何模块（RS Jacobian 复用 v4）
```

### 2. `optimizer_module_v5.py` 必须实现

```python
def compute_repair_loss_and_grad_rs(
    model: nn.Module,
    failed_safe_simplices: List,      # F_safe_cbf_violation + F_depth_limit_reached + F_unsafe_cannot_split
    failed_unsafe_simplices: List,    # F_h_positive_in_unsafe
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
    使用随机平滑（RS）方法计算 failure 区域的损失梯度。

    核心公式：
        g_s = (1/(N·σ²)) · Σ[ ψ_s(θ+εᵢ) · εᵢ ]

    其中 ψ_s(θ) 是 failure simplex s 的 Crown 下界:
        - safe 类型: ψ_s = clamp(tolerance - min_L, 0)   [min_L 是 CBF 下界]
        - unsafe 类型: ψ_s = clamp(h_lb, 0)              [h_lb 是 h 上界]

    最终返回: (total_loss, g_raw)，其中 g_raw = mean(g_s) over valid simplices

    注意: 完全不使用 autograd.backward()，避免 ReLU 次梯度震荡问题。
    """
    ...
```

### 3. `main_v5.py` 必须修改的部分

#### 3.1 导入部分（修改）
```python
# v4: RS Jacobian（保持不变）
from New_repair.geometry_module_new_v4 import compute_jacobian_rs

# v5: RS 损失梯度（新方案 A）
from New_repair.optimizer_module_v5 import compute_repair_loss_and_grad_rs

# v5: QP 投影（复用 v3 版本）
from New_repair.optimizer_module_v3 import qp_project_and_update
```

#### 3.2 主循环中的调用替换（关键修改）

**V4 的调用方式**:
```python
loss_val, g_F = compute_repair_loss_and_grad(
    model=model,
    dynamics_model=dynamics_model,
    failed_safe_feature_points=failed_safe_feature_points,  # 特征点
    failed_unsafe_feature_points=failed_unsafe_feature_points,  # 特征点
    ...
)
```

**V5 的调用方式**:
```python
# 将失败单纯形列表直接传入（不再提取特征点）
loss_val, g_F = compute_repair_loss_and_grad_rs(
    model=model,
    failed_safe_simplices=failed_safe_simplices,    # List of simplices
    failed_unsafe_simplices=failed_unsafe_feature_points_simplices,  # List of simplices
    dynamics_model=dynamics_model,
    translator=translator,
    tolerance=-1e-12,
    margin=0.0,
    cbf_margin=0.0,
    beta=5.0,
    rs_n=rs_n,          # 与 RS Jacobian 共享超参数
    rs_sigma=rs_sigma,
    grad_clip_norm=10.0,
    verbose=False,
)
```

#### 3.3 失败单纯形列表准备
```python
# V5: 准备失败单纯形列表（而非特征点）
failed_unsafe_simplices = list(F_h_positive_in_unsafe_init)
failed_safe_simplices = (
    list(F_safe_cbf_violation_init) +
    list(F_depth_limit_reached_init) +
    list(F_unsafe_cannot_split_init)
)
```

### 4. `geometry_module_new_v5.py`

直接复用 `geometry_module_new_v4.py` 的 `compute_jacobian_rs`，不需要修改。

---

## 关键实现细节

### RS 梯度计算的具体步骤

```python
def compute_repair_loss_and_grad_rs(...):
    # 1. 获取原始参数向量
    params = list(model.parameters())
    theta_old = torch.nn.utils.parameters_to_vector(params).detach().clone()

    # 2. 收集所有 failure simplices
    all_simplices = []
    all_types = []  # 'safe' or 'unsafe'
    for simp in failed_safe_simplices:
        all_simplices.append(simp)
        all_types.append('safe')
    for simp in failed_unsafe_simplices:
        all_simplices.append(simp)
        all_types.append('unsafe')

    # 3. 对每个 simplex 计算 RS 梯度
    all_grads = []
    all_losses = []

    for simp, stype in zip(all_simplices, all_types):
        # Step A: 计算当前损失值（判断是否纳入）
        with torch.no_grad():
            if stype == 'unsafe':
                h_lb, h_ub = compute_simplex_bound(model, simp, 'unsafe',
                                                    dynamics_model=None, translator=None)
                loss_val = F.softplus(h_lb + margin, beta=beta)  # h 应该 <= 0
            else:
                min_L = compute_simplex_bound(model, simp, 'safe',
                                              dynamics_model=dynamics_model, translator=translator)
                loss_val = F.softplus(cbf_margin - min_L, beta=beta)  # cbf 应该 >= cbf_margin

        if loss_val.item() <= 0:
            continue

        # Step B: RS 梯度估计
        accumulator = torch.zeros(num_params, dtype=dtype, device=device)
        valid_count = 0

        for _ in range(rs_n):
            eps_i = torch.randn(num_params, dtype=dtype, device=device) * rs_sigma
            theta_i = theta_old + eps_i
            torch.nn.utils.vector_to_parameters(theta_i, params)

            with torch.no_grad():
                if stype == 'unsafe':
                    h_lb_i, _ = compute_simplex_bound(model, simp, 'unsafe', ...)
                    psi_val = F.softplus(h_lb_i + margin, beta=beta)
                else:
                    min_L_i = compute_simplex_bound(model, simp, 'safe', ...)
                    psi_val = F.softplus(cbf_margin - min_L_i, beta=beta)

            if torch.isfinite(psi_val) and not torch.isnan(psi_val):
                accumulator.add_(eps_i * psi_val)
                valid_count += 1

        # 恢复原始参数
        torch.nn.utils.vector_to_parameters(theta_old.clone(), params)

        if valid_count > 0:
            g_s = accumulator / (valid_count * rs_sigma * rs_sigma)
            all_grads.append(g_s)
            all_losses.append(loss_val.item())

    # 4. 聚合梯度
    if len(all_grads) == 0:
        return 0.0, torch.zeros(num_params, device=device, dtype=dtype)

    g_raw = torch.stack(all_grads, dim=0).mean(dim=0)
    total_loss = np.mean(all_losses)

    # 5. 梯度裁剪
    grad_norm = g_raw.norm().item()
    if grad_norm > grad_clip_norm:
        g_raw = g_raw * (grad_clip_norm / grad_norm)

    return total_loss, g_raw
```

---

## 重要约束

1. **必须复用 v4 的 RS Jacobian 计算**（`compute_jacobian_rs`），不要修改
2. **必须复用 v3 的 QP 投影**（`qp_project_and_update`），不要修改
3. **只使用 failure simplices 列表**，不再提取特征点
4. **不使用任何 autograd.backward()**，完全用 RS 估计梯度
5. **RS 超参数**（rs_n, rs_sigma）应该与 RS Jacobian 保持一致（可以在命令行统一定义）
6. **保存版本为 v5**：输出文件加 `_v5` 后缀

---

## 命令行参数（与 v4 保持一致）

```python
parser.add_argument('--rs-n', type=int, default=100, help='随机平滑采样次数 N')
parser.add_argument('--rs-sigma', type=float, default=0.01, help='随机平滑噪声标准差 sigma')
parser.add_argument('--num-inner-steps', type=int, default=5, help='内循环步数')
parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
```

---

## 期望的输出文件

1. `New_repair/main_v5.py`
2. `New_repair/optimizer_module_v5.py`
3. `New_repair/geometry_module_new_v5.py`（仅复制 v4 版本，不改内容）
4. 运行结果保存到 `New_repair/nr_results_v5/result_{system}_{activation}_v5.json`
5. 模型保存到 `New_repair/regions/{system}_{activation}_cbf_repaired_v5.pth`
