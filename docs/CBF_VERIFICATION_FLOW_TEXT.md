# CBF神经网络验证流程文字描述

基于论文：**Scalable Verification of Neural Control Barrier Functions Using Linear Bound Propagation**

命令：`python3 experiments/barrier_certificate.py --system-type barr2 --verify --max-depth 13`

---

## 目录

1. [整体架构调用流程](#整体架构调用流程)
2. [主验证详细流程](#主验证详细流程)
3. [CBF条件验证详细流程](#cbf条件验证详细流程)
4. [CROWN线性界传播流程](#crown线性界传播流程)
5. [泰勒展开详细流程](#泰勒展开详细流程)
6. [区域分割详细流程](#区域分割详细流程)
7. [数据流转过程](#数据流转过程)
8. [类之间的调用关系](#类之间的调用关系)
9. [典型函数调用栈](#典型函数调用栈)

---

## 整体架构调用流程

### 第一阶段：用户入口和初始化

1. **用户执行命令**
   - 文件：`experiments/barrier_certificate.py`
   - 函数：`main()`
   - 解析命令行参数：
     - `--system-type barr2`：选择Barrier2系统
     - `--verify`：启用验证模式
     - `--max-depth 13`：设置最大分割深度

2. **初始化动力学系统**
   - 根据系统类型创建相应对象
   - 对于 `barr2`：创建 `Barrier2System(alpha=alpha)`
   - 该对象包含：
     - `input_dim`：状态维度
     - `control_dim`：控制维度
     - `input_domain`：输入空间（BoxDomain）
     - `safe_set`：安全集（ComplementDomain）
     - 动力学函数 `f(x)` 和 `g(x)`
     - class-K函数参数 `alpha`

3. **加载神经网络模型**
   - 路径：`data/mine_models_relu/barr2_cbf.onnx`
   - 加载对应的 `.pth` 文件到 `BarrierNN` 对象
   - 模型结构：包含多个全连接层和激活函数层

4. **创建验证策略**
   - 类：`CBFVerificationStrategy`
   - 初始化步骤：
     - 加载网络到 `_LOCAL.torch_model`
     - 创建 `CrownPartialLinearization` 对象
     - 保存动力学模型到 `_LOCAL.dynamics_model`
     - 设置设备（CPU/GPU）和数据类型

### 第二阶段：网格生成

5. **创建区域生成器**
   - 函数：`create_region_generator("simplicial")`
   - 返回：`SimplicialRegionGenerator`

6. **生成初始网格**
   - 调用：`region_generator.create_mesh(dynamics_model)`
   - 对于单纯形网格：
     - 在输入域的角点创建初始点
     - 使用 Delaunay 三角剖分生成三角形
     - 返回初始区域列表

### 第三阶段：执行器设置

7. **创建执行器**
   - 类型：`SinglethreadExecutor`（单线程）
   - 替代选项：`MultithreadExecutor` 或 `MultiprocessExecutor`
   - 初始化：创建 LIFO 队列用于深度优先搜索

### 第四阶段：验证执行

8. **调用验证函数**
   - 函数：`verify_cbf(dynamics_model, network_path, ...)`
   - 参数：模型路径、区域类型、最大深度等

9. **执行器运行**
   - 调用：`executor.execute(initializer, process_batch, aggregate, ...)`
   - 初始化worker：调用 `strategy.initialize_worker()`
   - 将初始区域放入队列
   - 进入主循环

10. **主循环处理**
    - 条件：队列不为空
    - 步骤：
      a. 从队列取出一批区域
      b. 调用 `process_batch(batch)` 验证
      c. 处理每个结果：
         - SAT：标记通过
         - UNSAT：记录反例
         - MAYBE：添加子区域到队列
      d. 更新进度和统计信息

11. **结果聚合和输出**
    - 计算通过百分比
    - 计算未验证百分比
    - 记录计算时间
    - 打印最终结果

---

## 主验证详细流程

### 步骤1：参数解析和系统选择

**文件**：`experiments/barrier_certificate.py:17-89`

1. 解析命令行参数
2. 根据 `system_type` 选择系统：
   - `"barr2"` → `Barrier2System`
   - `"barr3"` → `Barrier3System`
   - `"barr4"` → `Barrier4System`
   - 等等...

### 步骤2：验证初始化

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:263-295`

**函数**：`CBF`VerificationStrategy.initialize_worker()`

1. 设置设备（CPU或GPU）
2. 设置数据类型（float32或float64）
3. 将 `.onnx` 路径转换为 `.pth`
4. 创建 `BarrierNN` 对象：
   - 输入大小：`dynamics_model.input_dim`
   - 隐藏层大小：`dynamics_model.hidden_sizes`
5. 加载模型权重：`torch.load(pth_path)`
6. 设置为评估模式：`model.eval()`
7. 创建 `CrownPartialLinearization`：
   - 传入神经网络
   - 设置数据类型
8. 保存到全局变量：
   - `_LOCAL.torch_model`
   - `_LOCAL.network_linearizer`
   - `_LOCAL.dynamics_model`
   - `_LOCAL.device`
   - `_LOCAL.dtype`
   - `_LOCAL.max_depth`

### 步骤3：网格创建

**文件**：`lbp_neural_cbf/regions/simplicial.py:390-493`

1. 创建 `SimplicialMesh` 对象
2. 初始化网格：
   - 在每个维度创建2个点（最小和最大边界）
   - 使用 `np.meshgrid` 生成所有组合
   - 对于2D及更高：使用 Delaunay 三角剖分
   - 对于1D：手动创建线段
3. 获取区域：`mesh.get_regions(0)`
   - 遍历每个三角形（或线段）
   - 创建 `SimplicialRegion` 对象
   - 返回区域列表

### 步骤4：执行器初始化

**文件**：`lbp_neural_cbf/executors/single_thread_executor.py:9-12`

1. 创建 `LifoQueue()` 对象
2. 将所有初始区域放入队列
3. 初始化统计对象

### 步骤5：主执行循环

**文件**：`lbp_neural_cbf/executors/single_thread_executor.py:22-163`

**函数**：`execute(...)`

1. 调用 `initializer()` 初始化worker
2. 创建 LIFO 队列并填入初始区域
3. 进入 while 循环（条件：队列不为空）：
   a. `gather_batch(batch_size)`：获取一批区域
   b. `process_batch(batch)`：验证这批区域
   c. 对每个结果：
      - 更新统计信息
      - 如果有绘图器，更新可视化
      - 聚合结果
      - 如果结果是 MAYBE（有新样本），添加到队列
   d. 更新进度条
4. 计算最终统计：
   - 通过百分比
   - 未验证百分比
   - 计算时间
5. 返回聚合结果和统计

### 步骤6：批量验证处理

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:298-452`

**函数**：`_verify_batch_linbndprop(...)`

1. 初始化结果数组：`results = [None] * len(batch)`
2. 调用 `compute_network_bounds(batch)` 计算所有区域的网络界
3. 遍历每个区域：
   a. 获取障碍函数界：`get_network_output_bounds(sample_idx)`
   b. 判断区域类型（不安全、边界、安全）
4. 对需要验证CBF条件的区域：
   a. 调用 `keep_indices()` 过滤
   b. 调用 `compute_partial_derivative_bounds()` 计算Jacobian界
   c. 准备子批次
   d. 对每个 eta 值进行验证
5. 根据验证结果填充结果数组：
   - 通过：`SampleResultSAT`
   - 失败但有反例：`SampleResultUNSAT`
   - 失败无反例：调用 `_handle_split()` 返回 `SampleResultMAYBE`

---

## CBF条件验证详细流程

### 步骤1：计算网络输出界

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:341-383`

1. 调用 `network_linearizer.compute_network_bounds(batch)`
   - 这个调用会：
     - 提取所有全连接层
     - 对每一层计算输入-输出界
     - 应用激活函数松弛
     - 存储中间结果
2. 对每个区域：
   a. 调用 `get_network_output_bounds(sample_idx)`
   b. 获取：`h_min, h_max`
   c. 这表示：对于所有 x 在区域中，`h_min <= h(x) <= h_max`

### 步骤2：判断区域类型

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:352-382`

对于每个区域：

**Case 1：不安全区域**
- 条件：`h_max < 0`
- 含义：障碍函数在整个区域上为负
- 操作：直接标记为 SAT
- 结果类型：`"unsafe_region"`

**Case 2：边界/混合区域**
- 条件：`unsafe_region(sample, dynamics_model)` 返回 True
  - 检查区域是否与真正的不安全集相交
- 子情况 2a：`h_min >= 0`
  - 含义：障碍函数在真正的不safe区域上为正
  - 操作：这是违规，返回 UNSAT
  - 结果类型：`"h_positive_in_unsafe"`
- 子情况 2b：`h_min < 0`
  - 含义：障碍函数在边界附近
  - 操作：需要分割区域
  - 调用 `_handle_split()` 返回 `SampleResultMAYBE`

**Case 3：安全区域**
- 条件：不属于上述情况
- 含义：障碍函数在某些点为正，可能是安全区域
- 操作：需要验证CBF条件
- 将区域索引添加到 `to_check_cbf_cond` 列表

### 步骤3：计算Jacobian界

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:387-391`

1. 调用 `network_linearizer.keep_indices(to_check_cbf)`

   - 过滤存储的界，只保留需要验证的区域

2. 调用 `network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=0)`

   - 这会：
     a. 从最后一层开始，获取Jacobian界
     b. 反向传播到输入层
     c. 对每个中间层应用McCormick松弛
     d. 最终得到：`∂h/∂x` 的仿射界

3. 准备子批次：`subbatch = [batch[i] for i in to_check_cbf_cond]`

### 步骤4：eta迭代验证

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:395-428`

1. 定义 eta 值列表：`[(0.5, 0.5)]`
   - 第一个 eta 用于漂移项
   - 第二个 eta 用于控制项

2. 对每个 eta 值迭代：
   a. 准备子批次（如果是第一次迭代，使用整个子批次）
   b. 调用 `_verify_cbf_condition_affine(subbatch, eta=eta)`
   c. 获取验证结果：`eta_verified, counter_verified, min_L, max_U`
   d. 更新 `cbf_verified` 数组
   e. 过滤未通过的区域：`keep_indices(eta_verified)`
   f. 更新网络线性化器：`keep_indices(keep_mask)`

3. eta 迭代结束后，`cbf_verified` 数组包含每个子区域是否通过

### 步骤5：处理验证结果

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:429-450`

对于每个需要验证的区域：

1. **通过验证**（`cbf_verified[subsample_idx] == True`）
   - 创建 `SampleResultSAT` 对象
   - 结果类型：`"safe_cbf_verified"`
   - 添加到结果数组

2. **失败但有反例**（`cbf_verified == False` 且 `counter_verified == True`）
   - 创建 `SampleResultUNSAT` 对象
   - 反例：区域中心点
   - 结果类型：`"safe_cbf_violation"`
   - 添加到结果数组

3. **失败无反例**（`cbf_verified == False` 且 `counter_verified == False`）
   - 调用 `_handle_split()`
   - 根据分割结果：
     - 可分割：返回 `SampleResultMAYBE`（包含子区域）
     - 不可分割：`SampleResultUNSAT`（深度或体积限制）

---

## CROWN线性界传播流程

### 步骤1：CrownPartialLinearization 初始化

**文件**：`lbp_neural_cbf/linearization/linear_derivative_bounds.py:13-30`

1. 保存神经网络引用
2. 提取所有全连接层：`_extract_linear_layers()`
   - 遍历 `network.modules()`
   - 过滤 `isinstance(module, nn.Linear)`
3. 检测激活函数类型：`_detect_activation_relaxation()`
   - 遍历网络模块
   - 检查 `nn.ReLU`, `nn.Tanh`, `nn.Sigmoid`, `nn.LeakyReLU`
   - 返回对应的激活函数松弛对象
4. 初始化存储字典：
   - `forward_bounds = {}`：存储前向传播界
   - `derivative_bounds = {}`：存储偏导数界

### 步骤2：计算网络输出界

**文件**：`lbp_neural_cbf/linearization/linear_derivative_bounds.py:87-199`

**函数**：`_compute_network_bounds(batch)`

1. 根据区域类型计算输入界：
   - **超矩形**：
     - 计算中心：所有区域中心的堆叠
     - 计算半径：所有区域半径的堆叠
     - 输入界：`x_lb = center - radius`, `x_ub = center + radius`
   - **单纯形**：
     - 收集所有顶点
     - 输入界：顶点的最小值和最大值

2. 初始化仿射系数：
   - `A_L = A_U = I`（单位矩阵）
   - `a_L = a_U = 0`（零向量）

3. 遍历每个全连接层：
   a. 获取权重和偏置：`W, b = layer.weight, layer.bias`
   b. 分解权重：
     - `W_pos = F.relu(W)`（正部分）
     - `W_neg = W - W_pos`（负部分）
   c. 计算预激活界：
     ```
     A_y_L = W_pos @ A_L + W_neg @ A_U
     a_y_L = W_pos @ a_L + W_neg @ a_U + b
     A_y_U = W_pos @ A_U + W_neg @ A_L
     a_y_U = W_pos @ a_U + W_neg @ a_L + b
     ```
   d. 计算区间界 `y_lb` 和 `y_ub`：
     - **超矩形**：使用中心和半径公式
     - **单纯形**：在所有顶点上求最小/最大值
   e. 检查是否是最后一层：
     - 如果是最后一层：直接存储界
     - 如果不是最后一层：继续激活函数松弛
   f. 激活函数松弛：
     - 调用 `activation_relaxation.relax_activation(y_lb, y_ub)`
     - 返回：`alpha_L, beta_L, alpha_U, beta_U`
   g. 更新仿射系数：
     ```
     A_L = alpha_L_pos @ A_y_L + alpha_L_neg @ A_y_U + alpha_U_pos @ A_L + alpha_U_neg @ A_U
     a_L = alpha_L_pos @ a_y_L + alpha_L_neg @ a_y_U + alpha_U_pos @ a_L + alpha_U_neg @ a_U + beta_L
     ```
   h. 计算激活后界并存储

4. 存储最终界到 `forward_bounds`

### 步骤3：计算偏导数界

**文件**：`lbp_neural_cbf/linearization/linear_derivative_bounds.py:267-317`

**函数**：`compute_partial_derivative_bounds(input_idx, output_idx)`

1. 从最后一层开始：`_get_jacobian_bounds_for_layer(L)`
   - 最后一层是线性的，Jacobian就是权重矩阵
   - 返回：`Lambda_L, lambda_L, Lambda_U, lambda_U`

2. 初始化运行变量为张量形式

3. 反向传播循环（从 L-1 到 1）：
   a. 获取当前层Jacobian界：`_get_jacobian_bounds_for_layer(i)`
      - 获取预激活界：`forward_bounds[f"layer_{i-1}_pre_act_bounds"]`
      - 计算激活函数导数界：`relax_activation_derivative()`
      - 计算Jacobian界（应用链式法则）
      - 返回界张量
   b. 获取公共变量界（预激活）：
      - 从 `forward_bounds` 中提取
   c. 计算矩阵乘积的McCormick界：
      - 调用 `_vectorized_mccormick_product(...)`
      - 公式：`J^(i) = M^(i+1) @ J^(i-1)` 的界
   d. 将界传播到前一层的输入：
      - 调用 `_propagate_bounds_one_layer(...)`
      - 将 `y_i` 的函数界传播到 `y_{i-1}` 的函数界

4. 根据参数选择返回值：
   - 如果 `input_idx=None` 且 `output_idx=None`：返回全部
   - 如果 `output_idx=None`：返回特定输入的所有输出
   - 如果 `input_idx=None`：返回特定输出的所有输入
   - 否则：返回单个偏导数

### 步骤4：获取层Jacobian界

**文件**：`lbp_neural_cbf/linearization/linear_derivative_bounds.py:319-346`

**函数**：`_get_jacobian_bounds_for_layer(i)`

1. 获取权重：`W_i = fc_layers[i-1].weight`

2. 检查是否是最后一层：
   - 如果是最后一层：Jacobian就是权重
     - 返回：`(zeros, W_i, zeros, W_i)`

3. 获取激活函数导数界：
   - 调用：`activation_relaxation.relax_activation_derivative(y_i_lb, y_i_ub)`
   - 返回：`S_L, s_L, S_U, s_U`

4. 分解权重：
   - `W_i_pos = F.relu(W_i)`
   - `W_i_neg = W_i - W_i_pos`

5. 计算对角项：
   ```
   term_L = W_i_pos * S_L + W_i_neg * S_U
   term_U = W_i_pos * S_U + W_i_neg * S_L
   ```

6. 创建对角张量：
   - `Lambda_L`：从 `term_L` 创建的对角张量
   - `Lambda_U`：从 `term_U` 创建的对角张量

7. 计算常数项：
   ```
   lambda_L = W_i_pos * s_L + W_i_neg * s_U
   lambda_U = W_i_pos * s_U + W_i_neg * s_L
   ```

8. 返回：`(Lambda_L, lambda_L, Lambda_U, lambda_U)`

---

## 泰勒展开详细流程

### 步骤1：创建泰勒线性化器

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:856-894`

**函数**：`_compute_dynamics_bounds_taylor(batch, dynamics_model, ...)`

1. 创建数值转换器：`TorchTranslator(device, dtype)`

2. 创建泰勒线性化器：`TaylorLinearization(dynamics_model, numeric_translator)`

3. 根据区域类型创建批次区域对象：
   - **超矩形**：
     - 堆叠所有区域中心和半径
     - 创建 `HyperrectangularRegion(center_points, radius_vecs)`
   - **单纯形**：
     - 堆叠所有区域顶点
     - 创建 `SimplicialRegion(vertices)`

### 步骤2：线性化漂移项 f(x)

**文件**：`lbp_neural_cbf/linearization/taylor.py:75-149`

**函数**：`linearize_sample(sample)`

1. 确定区域类型：
   - 提取实际区域（处理AugmentedSample包装）
   - 区分 `SimplicialRegion` 和 `HyperrectangularRegion`

2. **单纯形区域处理**：
   a. 获取顶点：`vertices = actual_region.vertices`
   b. 计算质心：`center = mean(vertices, dim=-2)`
   c. 调用：`first_order_certified_taylor_expansion_simplex(dynamics, center, vertices, translator)`

3. **超矩形区域处理**：
   a. 获取质心：`center = actual_region.centroid`
   b. 获取半径：`radius = actual_region.radius_vec`
   c. 调用：`first_order_certified_taylor_expansion(dynamics, center, radius, translator)`

4. 从泰勒展开对象提取信息：
   - `jacobian, f_c = taylor_expansion.linear_approximation`
   - `remainder_lower, remainder_upper = taylor_expansion.remainder`
   - `expansion_point = taylor_expansion.expansion_point`

5. 构造仿射界：
   ```
   A_lower = jacobian
   b_lower = f_c - jacobian @ expansion_point + remainder_lower
   A_upper = jacobian
   b_upper = f_c - jacobian @ expansion_point + remainder_upper
   ```

6. 计算最大间隙：`max_gap = remainder_upper - remainder_lower`

7. 返回：`AugmentedSample.from_certification_region(region, ((A_lower, b_lower), (A_upper, b_upper), max_gap))`

### 步骤3：提取仿射界

**函数**：`linearize_sample()` 返回后处理

1. 从 `first_order_model` 提取：
   - `(A_L, b_L), (A_U, b_U), max_gap = taylor_linearization.first_order_model`

2. 存储：
   - `f_affine_bounds = (A_L, b_L), (A_U, b_U)`

### 步骤4：线性化控制项 g(x)（如果有）

**条件**：`dynamics_model.control_dim > 0`

1. 创建包装类 `GDynamics`：
   - `compute_dynamics()` 方法调用原始的 `compute_g()`

2. 创建线性化器：`TaylorLinearization(g_dynamics, numeric_translator)`

3. 调用 `linearize_sample(batch)`

4. 提取并存储：`g_affine_bounds = ((A_L, b_L), (A_U, b_U))`

### 步骤5：返回结果

返回：`(f_affine_bounds, g_affine_bounds)`

---

## 区域分割详细流程

### 步骤1：检查分割条件

**文件**：`lbp_neural_cbf/cbf/verify_cbf.py:246-262`

**函数**：`_handle_split(sample, start_time, results, sample_idx, min_volume, split_type, unsat_type, max_depth)`

1. 检查是否达到最大深度：
   - 条件：`max_depth is not None and sample.depth >= max_depth`
   - 如果达到：
     - 取区域中心作为反例
     - 创建 `SampleResultUNSAT(result_type="depth_limit_reached")`
     - 返回

2. 检查区域体积：
   - 条件：`sample._compute_volume() > min_volume`
   - 如果体积足够大：可以分割
   - 如果体积太小：无法分割

### 步骤2：执行分割

**条件**：`sample._compute_volume() > min_volume`

#### 单纯形分割

**文件**：``lbp_neural_cbf/regions/simplicial.py:320-360`

**函数**：`split()`

1. 找到最长边：
   - 调用 `get_max_edge_length()`
   - 遍历所有边对 `(i, j)`
   - 计算边长：`norm(vertices[i] - vertices[j])`
   - 记录最大长度和端点索引

2. 计算中点：
   ```
   midpoint = (vertices[v1_idx] + vertices[v2_idx]) / 2
   ```

3. 创建两个新单纯形：
   - **单纯形1**：
     - 复制原始顶点
     - 将 `v2_idx` 替换为 `midpoint`
   - **单纯形2**：
     - 复制原始顶点
     - 将 `v1_idx` 替换为 `midpoint`

4. 创建新区域对象：
   - 两个新单纯形的深度都设为 `sample.depth + 1`

5. 返回：`(region1, region2)`

#### 超矩形分割

**文件**：（参考超矩形区域类）

1. 选择分割维度：
   - 通常交替使用维度
   - 或者选择最大范围的维度

2. 计算中点：
   ```
   mid = (lower_bound + upper_bound) / 2
   ```

3. 创建两个新超矩形：
   - **左区域**：`upper_bound = mid`
   - **右区域**：`lower_bound = mid`

4. 返回两个新区域

### 步骤3：处理分割结果

**情况A：成功分割**

1. 调用 `sample.split()` 获取新区域
2. 创建 `SampleResultMAYBE` 对象：
   - 传入：`sample, start_time, new_samples`
   - 设置：`split_type` 和 `unsat_type`
3. 存储：`results[sample_idx] = SampleResultMAYBE`

**情况B：无法分割**

1. 取区域中心作为反例
2. 创建 `SampleResultUNSAT` 对象：
   - 传入：`sample, start_time, [centerpoint]`
   - 设置：`result_type = unsat_type`
3. 存储：`results[sample_idx] = SampleResultUNSAT`

---

## 数据流转过程

### 输入界

1. **神经网络模型**
   - 来源：`.pth` 或 `.onnx` 文件
   - 内容：权重矩阵、偏置向量、激活函数类型

2. **动力学系统**
   - 来源：`Barrier2System` 等系统类
   - 内容：f(x), g(x) 函数实现、控制边界、输入域

3. **验证参数**
   - 来源：命令行或默认值
   - 内容：区域类型（simplicial/hyperrectangular）、最大深度、批大小

4. **初始网格**
   - 来源：Delaunay三角剖分
   - 内容：初始单纯形/超矩形区域列表

### 处理界

1. **区域队列**
   - 类型：LIFO队列（实现深度优先搜索）
   - 流向：从初始区域开始，MAYBE结果添加子区域
   - 终止：队列为空

2. **批量处理器**
   - 输入：一批区域（例如512个）
   - 操作：对每个区域调用验证逻辑
   - 输出：结果数组（SAT/MAYBE/UNSAT）

3. **验证策略**
   - 输入：单个区域
   - 操作：
     - 计算网络界
     - 判断区域类型
     - 验证CBF条件（如果需要）
   - 输出：验证结果

### 计算界

1. **神经网络界（CROWN）**
   - 输入：区域（中心、半径或顶点）
   - 过程：
     - 前向传播：计算每层输出界
     - 反向传播：计算偏导数界
   - 输出：仿射系数和常数项

2. **动力学界（泰勒）**
   - 输入：区域（中心、半径或顶点）
   - 过程：
     - 计算展开点的函数值和Jacobian
     - 计算余项界
     - 构造仿射界
   - 输出：仿射系数和常数项

3. **乘积界（McCormick）**
   - 输入：两个仿射函数的界
   - 过程：
     - 计算两个函数的范围
     - 应用McCormick不等式
     - 构造结果的仿射系数
   - 输出：乘积的仿射界

### 输出界

1. **SAT结果（通过验证）**
   - 类型：`SampleResultSAT`
   - 内容：区域、处理时间、结果类型（unsafe_region, safe_cbf_verified等）
   - 操作：记录通过，不进一步处理

2. **MAYBE结果（需要分割）**
   - 类型：`SampleResultMAYBE`
   - 内容：区域、处理时间、新区域列表、分割类型
   - 操作：将新区域加入队列，继续验证

3. **UNSAT结果（发现反例）**
   - 类型：`SampleResultUNSAT`
   - 内容：区域、处理时间、反例点、结果类型
   - 操作：记录失败，返回反例

4. **统计信息**
   - 通过百分比：`(SAT数量 / 总数量) * 100%`
   - 未验证百分比：`(UNSAT数量 / 总数量) * 100%`
   - 计算时间：从开始到结束的总时间
   - 处理速度：`总数量 / 计算时间`（迭代/秒）

---

## 类之间的调用关系

### CBFVerificationStrategy 类

**职责**：协调整个验证流程

**主要属性**：
- `network_path`：神经网络模型路径
- `dynamics_model`：动力学系统对象
- `max_depth`：最大分割深度
- `use_gpu`：是否使用GPU

**主要方法**：
- `initialize_worker()`：初始化worker环境
  - 加载神经网络
  - 创建 CROWN 线性化器
  - 保存到全局变量

- `verify_batch()`：验证一批区域
  - 调用 CROWN 计算网络界
  - 判断区域类型
  - 对安全区域验证CBF条件
  - 返回结果数组

- `_verify_cbf_condition_affine()`：验证CBF条件
  - 调用泰勒展开计算动力学界
  - 调用 CROWN 计算Jacobian界
  - 应用McCormick松弛
  - 返回验证结果

- `_handle_split()`：处理区域分割
  - 检查深度和体积限制
  - 调用区域的 `split()` 方法
  - 返回MAYBE或UNSAT结果

**依赖关系**：
- 使用 `CrownPartialLinearization` 计算网络界
- 使用 `BarrierNN` 神经网络
- 验证 `SimplicialRegion` 或 `HyperrectangularRegion`
- 从 `Barrier2System` 获取动力学

### CrownPartialLinearization 类

**职责**：计算神经网络的输入-输出关系和偏导数界

**主要属性**：
- `network`：PyTorch神经网络
- `fc_layers`：提取的全连接层列表
- `activation_relaxation`：激活函数松弛对象
- `forward_bounds`：存储前向传播界
- `derivative_bounds`：存储偏导数界

**主要方法**：
- `_compute_network_bounds()`：计算前向传播界
  - 遍历所有层
  - 应用激活函数松弛
  - 存储中间结果

- `compute_network_bounds()`：公共接口调用内部方法

- `get_network_output_bounds()`：获取最终输出界
  - 从 `forward_bounds` 提取
  - 返回 h(x) 的界

- `get_network_linear_bounds()`：获取最终输出的仿射界
  - 返回仿射系数和常数

- `compute_partial_derivative_bounds()`：计算偏导数界
  - 反向传播Jacobian界
  - 应用McCormick松弛
  - 存储结果

- `get_partial_derivative_bounds()`：获取偏导数界
  - 从 `derivative_bounds` 提取
  - 返回 ∂h/∂x 的界

- `_get_jacobian_bounds_for_layer()`：计算单层Jacobian界
  - 应用链式法则
  - 返回界张量

- `_vectorized_mccormick_product()`：矩阵乘积的McCormick松弛
  - 处理向量化的矩阵乘积
  - 返回仿射界

**依赖关系**：
- 使用 PyTorch 张量运算
- 使用激活函数松弛对象
- 从神经网络提取权重和偏置

### TaylorLinearization 类

**职责**：对动力学函数进行泰勒展开

**主要属性**：
- `dynamics`：动力学函数对象
- `translator`：数值转换器

**主要方法**：
- `linearize()`：线性化一批样本
  - 调用 `linearize_sample()` 处理每个样本

- `linearize_sample()`：线性化单个样本
  - 判断区域类型
  - 调用相应的泰勒展开函数
  - 返回 `AugmentedSample` 对象

- `get_taylor_expansion()`：获取完整泰勒展开
  - 返回 `CertifiedFirstOrderTaylorExpansion` 对象

**依赖关系**：
- 使用 `TaylorTranslator` 进行符号计算
- 从动力学对象获取 f(x), g(x) 函数

### SimplicialRegion 类

**职责**：表示和操作单纯形区域

**主要属性**：
- `vertices`：单纯形的顶点（n+1个顶点在n维空间）
- `n_vertices`：顶点数量
- `dim`：空间维度
- `depth`：分割深度

**主要方法**：
- `_compute_centroid()`：计算质心
- `_compute_volume()`：计算体积
- `get_bounds()`：获取边界框
- `contains_point()`：检查点是否在单纯形内
- `get_max_edge_length()`：找到最长边
- `split()`：分割单纯形
  - 在最长边的中点处分割
  - 返回两个新单纯形

### Barrier2System 类

**职责**：定义Barrier2动力学系统

**主要属性**：
- `input_dim`：状态维度
- `control_dim`：控制维度
- `alpha`：class-K函数参数
- `input_domain`：输入空间
- `safe_set`：安全集
- `u_min`, `u_max`：控制边界

**主要方法**：
- `compute_f()`：计算漂移项 f(x)
- `compute_g()`：计算控制项 g(x)
- `alpha_function()`：class-K函数
  - 对于线性class-K：`alpha * h`

### SinglethreadExecutor 类

**职责**：管理验证执行的循环

**主要属性**：
- `queue`：LIFO队列

**主要方法**：
- `execute()`：执行验证
  - 初始化worker
  - 管理区域队列
  - 调用批量处理
  - 聚合结果

- `gather_batch()`：从队列获取一批区域

---

## 典型函数调用栈

### 场景1：处理一个安全区域

```
main()
  └─ verify_cbf()
      └─ executor.execute()
          └─ initialize_worker()              # 初始化worker
          └─ [循环处理队列]
              └─ verify_batch(batch)
                  ├─ compute_network_bounds()         # CROWN前向传播
                  │   └─ [遍历层]
                  │       ├─ 权重分解
                  │       ├─ 计算预激活界
                  │       └─ 激活函数松弛
                  ├─ [遍历区域]
                  │   └─ get_network_output_bounds()
                  ├─ compute_partial_derivative_bounds() # CROWN后向传播
                  │   ├─ _get_jacobian_bounds_for_layer(L)
                  │   └─ [反向循环 L-1 到 1]
                  │       ├─ _get_jacobian_bounds_for_layer(i)
                  │       ├─ _vectorized_mccormick_product()
                  │       └─ _propagate_bounds_one_layer()
                  └─ _verify_cbf_condition_affine()
                      ├─ compute_dynamics_bounds_taylor()     # 泰勒展开
                      │   ├─ TaylorLinearization()
                      │   └─ linearize_sample()
                      │       └─ first_order_certified_taylor_expansion_simplex()
                      ├─ get_partial_derivative_bounds()         # 获取Jacobian界
                      ├─ _batched_compute_mccormick_product_lower_bound() # 漂移项
                      ├─ _batched_compute_mccormick_product_lower_bound() # 控制项
                      ├─ alpha_function()                     # class-K项
                      └─ _batched_get_affine_function_bounds()    # 最终验证
                  └─ [处理结果]
                      └─ SampleResultSAT()             # 通过验证
```

### 场景2：区域需要分割

```
main()
  └─ verify_cbf()
      └─ executor.execute()
          └─ [循环处理队列]
              └─ verify_batch(batch)
                  └─ [遍历区域]
                      ├─ get_network_output_bounds()
                      └─ _handle_split()
                          ├─ 检查深度限制
                          ├─ sample._compute_volume()
                          └─ sample.split()                         # 分割区域
                              ├─ get_max_edge_length()                 # 单纯形
                              │   └─ [遍历所有边]
                              │       └─ 计算边长度
                              └─ 创建两个新SimplicialRegion
                          └─ SampleResultMAYBE()                  # 返回子区域
          └─ 添加子区域到队列                    # 继续处理
```

### 场景3：发现反例

```
main()
  └─ verify_cbf()
      └─ executor.execute()
          └─ [循环处理队列]
              └─ verify_batch(batch)
                  └─ [遍历区域]
                      ├─ compute_partial_derivative_bounds()
                      └─ _verify_cbf_condition_affine()
                          └─ find_counterexample = True       # 启用反例搜索
                              ├─ 计算CBF条件的上界
                              ├─ _batched_get_affine_function_bounds()
                              └─ max_U < 0                  # 确认反例
                  └─ SampleResultUNSAT()                  # 返回反例
```

### 场景4：不安全区域

```
main()
  └─ verify_cbf()
      └─ executor.execute()
          └─ [循环处理队列]
              └─ verify_batch(batch)
                  └─ [遍历区域]
                      ├─ get_network_output_bounds()
                      └─ h_max < 0                       # 不安全区域
                          └─ SampleResultSAT(result_type="unsafe_region")
```

---

## 并行处理流程（多Worker模式）

### 单线程模式（默认）

1. 主进程执行所有工作
2. 使用LIFO队列实现深度优先搜索
3. 顺序处理每个批量
4. 共享内存，无需进程间通信

### 多线程模式

1. 创建多个线程
2. 共享全局状态（`_LOCAL` 变量）
3. Python GIL限制真正的并行性
4. 适合I/O密集型任务

### 多进程模式

1. 创建多个进程
2. 每个进程有独立内存空间
3. 需要序列化/反序列化传递数据
4. 真正的CPU并行，但通信开销较大

---

## 性能关键点

### 1. 批处理优化

- 一次处理多个区域（默认512个）
- 减少函数调用开销
- 向量化计算（PyTorch张量操作）

### 2. GPU加速

- CROWN计算在GPU上执行
- 张量运算使用CUDA
- 显著加速大规模网络

### 3. 搜索策略

- LIFO队列 → 深度优先搜索（DFS）
- 优先完成子树的验证
- 快速找到深层区域的结果

### 4. 区域分割策略

- 单纯形比超矩形更精确
- 在最长边中分减少保守性
- 深度限制防止无限细分

### 5. 缓存机制

- CROWN缓存中间层界
- 避免重复计算
- `keep_indices()` 过滤不必要的界

### 6. 数值稳定性

- 使用适当的数值精度（float32/float64）
- 容错处理（例如 `>= -1e-12`）
- 避免除以零等数值问题

---

## 总结

整个CBF验证流程可以概括为以下阶段：

1. **初始化阶段**
   - 加载神经网络和动力学系统
   - 创建验证策略和执行器
   - 生成初始网格

2. **执行阶段**
   - 使用DFS遍历所有区域
   - 对每个区域进行验证

3. **区域验证**
   - 计算神经网络输出界 h(x)
   - 判断区域类型（不安全/边界/安全）
   - 对安全区域验证CBF条件

4. **CBF条件验证**
   - 使用CROWN计算 ∇h(x) 界
   - 使用泰勒展开计算 f(x), g(x) 界
   - 使用McCormick计算乘积界
   - 检查最终下界是否 ≥ 0

5. **结果处理**
   - SAT：通过，不再处理
   - MAYBE：分割，继续处理子区域
   - UNSAT：失败，返回反例

关键组件：
- **CROWN**：神经网络界传播
- **泰勒展开**：动力学函数线性化
- **McCormick松弛**：乘积的保守界
- **单纯形网格**：状态空间分割
- **DFS执行**：区域遍历策略
