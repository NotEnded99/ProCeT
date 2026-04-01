# CBF神经网络验证流程图

基于论文：**Scalable Verification of Neural Control Barrier Functions Using Linear Bound Propagation**

命令：`python3 experiments/barrier_certificate.py --system-type barr2 --verify --max-depth 13`

---

## 目录

1. [整体架构流程](#整体架构流程)
2. [主验证流程](#主验证流程)
3. [CBF条件验证流程](#cbf条件验证流程)
4. [CROWN线性界传播流程](#crown线性界传播流程)
5. [泰勒展开流程](#泰勒展开流程)
6. [区域分割流程](#区域分割流程)

---

## 整体架构流程

    用户命令行
        │
        ▼
    experiments/barrier_certificate.py main()
        │
        ├───┐
        │   解析命令行参数
        │   │   ├─ --system-type barr2
        │   │   ├─ --verify
        │   │   └─ --max-depth 13
        │   └───┘
        │
        ▼
    创建动力学系统 Barrier2System
        │
        ├───┐
        │   input_dim: 状态维度
        │   │
        │   control_dim: 控制维度
        │   │
        │   alpha: class-K函数参数
        │   │
        │   input_domain: BoxDomain
        │   │
        │   safe_set: ComplementDomain
        │   │
        │   f(x): 漂移项动力学函数
        │   │
        │   └─ g(x): 控制项动力学函数
        │
        └───┘
        │
        ▼
    加载神经网络模型 data/mine_models_relu/barr2_cbf.pth
        │
        ▼
    创建 CBFVerificationStrategy
        │
        ├───┐
        │   CrownPartialLinearization: CROWN线性化器
        │   │
        │   BarrierNN: 神经网络
        │   │
        │   max_depth: 13
        │   │
        │   └─ device: CPU/GPU
        │
        └───┘
        │
        ▼
    创建区域生成器 SimplicialRegionGenerator
        │
        ▼
    创建初始网格 SimplicialMesh
        │
        ├───┐
        │   在输入域角点创建初始点
        │   │
        │   Delaunay三角剖分
        │   │
        │   生成初始单纯形区域列表
        │   │
        │   └───┘
        │
        ▼
    创建执行器 SinglethreadExecutor
        │
        ├───┐
        │   queue: LIFO队列（DFS）
        │   │
        │   batch_size: 512
        │   │
        │   └───┘
        │
        ▼
    调用 verify_cbf()
        │
        ▼
    executor.execute(initializer, process_batch, aggregate)
        │
        ├───┐
        │   初始化worker (initialize_worker)
        │   │
        │   加载模型到本地
        │   │
        │   准备CROWN
        │   │
        │   └───┘
        │
        ├───┐
        │   将初始区域放入LIFO队列
        │   └───┘
        │
        ▼
    ┌─────────────────────────┐
    │ while 队列不为空?     │
    └─────────────────────────┘
        │
        ├─ 是：继续处理
        │   │
        │   ▼
        │   gather_batch(batch_size): 获取一批区域
        │   │
        │   ▼
        │   verify_batch(batch): 验证这批区域
        │   │
        │   │
        │   ├───┐
        │   │   compute_network_bounds(): CROWN前向传播
        │   │   │   │
        │   │   │   └───┘
        │   │   │
        │   │   ├───┐
        │   │   │   遍历每个区域
        │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   获取 h(x) 界
        │   │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   │   ▼
        │   │   │   │   判断区域类型
        │   │   │   │   │
        │   │   │   │   ├─ Case 1: h_max < 0 → 不安全区域 → SAT
        │   │   │   │   │   │
        │   │   │   │   ├─ Case 2: 与不安全集相交 → 边界区域
        │   │   │   │   │   │   ├─ h_min >= 0 → UNSAT (违规）
        │   │   │   │   │   │   └─ h_min < 0 → 调用 _handle_split() → MAYBE
        │   │   │   │   │   │
        │   │   │   │   └─ Case 3: 安全区域 → 验证CBF条件
        │   │   │   │       │
        │   │   │   │       ├───┐
        │   │   │   │       │   compute_partial_derivative_bounds(): CROWN反向传播
        │   │   │   │       │   │   └───┘
        │   │   │   │       │   │
        │   │   │   │       │   ▼
        │   │   │   │       │   _verify_cbf_condition_affine(): 验证CBF条件
        │   │   │   │       │   │
        │   │   │   │       │   ├───┐
        │   │   │   │       │   │   compute_dynamics_bounds_taylor(): 泰勒展开
        │   │   │   │       │   │   │   └───┘
        │   │   │   │       │   │   │
        │   │   │   │       │   │   ├───┐
        │   │   │   │       │   │   │   计算漂移项界: McCormick(∇h, f)
        │   │   │   │       │   │   │   └───┘
        │   │   │   │       │   │   │   │
        │   │   │   │       │   │   │   ├───┐
        │   │   │   │       │   │   │   │   计算控制项界: sup_u McCormick(∇h, g)u
        │   │   │   │       │   │   │   │   └───┘
        │   │   │   │       │   │   │   │   │
        │   │   │   │       │   │   │   │   ├───┐
        │   │   │   │       │   │   │   │   │   计算class-K项界: alpha(h)
        │   │   │   │       │   │   │   │   └───┘
        │   │   │   │       │   │   │   │   │   │
        │   │   │   │       │   │   │   │   │   ├───┐
        │   │   │   │       │   │   │   │   │   │   最终检查: 下界 >= 0?
        │   │   │   │       │   │   │   │   │   │   │
        │   │   │   │       │   │   │   │   │   │   ├─ 是: SAT (通过验证)
        │   │   │   │       │   │   │   │   │   │   └─ 否: 检查是否需要分割
        │   │   │   │       │   │   │   │   │   │       │
        │   │   │   │       │   │   │   │   │   │       ├─── 找到反例: UNSAT
        │   │   │   │       │   │   │   │   │   │       └─ 未找到: 调用 _handle_split() → MAYBE
        │   │   │   │       │   │   │   │   │   │
        │   │   │   │       │   │   │   │   │   └───┘
        │   │   │   │       │   │   │   │
        │   │   │   │       │   │   │   └───┘
        │   │   │   │       │   │   │
        │   │   │   │       │   │   └───┘
        │   │   │   │       │   │
        │   │   │   │       │   └───┘
        │   │   │   │
        │   │   │   └───┘
        │   │   │   │
        │   │   └───┘
        │   │
        │   └───┘
        │   │
        │   ▼
        │   处理结果并更新统计
        │   │
        │   ├───┐
        │   │   SAT: 标记通过，不进一步处理
        │   │   │   └───┘
        │   │   │
        │   │   ├───┐
        │   │   │   MAYBE: 将子区域加入队列，继续处理
        │   │   │   │
        │   │   │   └───┘
        │   │   │
        │   │   └───┐
        │   │   │   UNSAT: 记录反例，不再处理
        │   │   │   └───┘
        │   │
        │   └───┘
        │   │
        │   ▼
        │   更新进度条
        │   │
        └─ 否: 队列已空，退出循环
        │
        └───┘
        │
        ▼
    计算最终统计信息
    │
    ├───┐
    │   通过百分比: (SAT数量 / 总数量) * 100%
    │   │
    │   未验证百分比: (UNSAT数量 / 总数量) * 100%
    │   │
    │   计算时间: end_time - start_time
    │   │
    │   处理速度: 总数量 / 计算时间
    │   │
    └───┘
        │
        ▼
    打印最终结果
        │
        └─── 完成

---


## 主验证流程

    main()
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤1: 参数解析和系统选择 │
    └─────────────────────────┘
        │
        ├─ 解析命令行参数
        │   │
        │   └─ 选择系统类型
        │
        │   ▼
        │   │
        │   ├─── "barr2" → Barrier2System
        │   │   │
        │   │   ├─── "barr3" → Barrier3System
        │   │   │
        │   │   ├─── "barr4" → Barrier4System
        │   │   │
        │   │   └─ ... 其他系统
        │   │   │
        │   └───┘
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤2: 验证初始化       │
    └─────────────────────────┘
        │
        ├─ verify_cbf(dynamics_model, network_path, ...)
        │   │
        │   └─ CBFVerificationStrategy.initialize_worker()
        │
        │   ▼
        │   │
        │   ├───┐
        │   │   加载 .pth 模型到 BarrierNN
        │   │   │   └───┘
        │   │   │   │
        │   │   ├───┐
        │   │   │   创建 CrownPartialLinearization
        │   │   │   │
        │   │   │   └───┘
        │   │   │   │   │
        │   │   └───┘
        │   │
        │   └─── 完成
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤3: 网格生成         │
    └─────────────────────────┘
        │
        ├─ create_region_generator("simplicial")
        │   │
        │   └─ SimplicialRegionGenerator
        │
        │   ▼
        │   │
        │   └─ create_mesh(dynamics_model)
        │
        │   ▼
        │   │
        │   └─ SimplicialMesh
        │
        │   ▼
        │   │
        │   ├───┐
        │   │   在每个维度创建2个点
        │   │   │   └───┘
        │   │   │   │
        │   │   ├───┐
        │   │   │   使用 np.meshgrid 生成所有组合
        │   │   │   └───┘
        │   │   │   │
        │   │   └─ 2D及更高: Delaunay 三角剖分
        │   │   │   └─ 1D: 手动创建线段
        │   │   │   │
        │   │   └─── 完成
        │   │
        │   └─ get_regions(0): 获取初始区域列表
        │   │
        │   └─── 完成
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤4: 执行器初始化      │
    └─────────────────────────┘
        │
        ├─ 创建 SinglethreadExecutor
        │   │
        │   └─ 初始化 LIFO 队列
        │   │
        │   └─ 将初始区域放入队列
        │   │
        └─── 完成
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤5: 主执行循环        │
    └─────────────────────────┘
        │
        └─ executor.execute(initializer, process_batch, ...)
        │
        ▼
    ┌─────────────────────────┐
    │ while 队列不为空?        │
    └─────────────────────────┘
        │
        ├─ 是: 继续处理
        │   │
        │   ▼
        │   gather_batch(batch_size)
        │   │
        │   │
        │   └─ 返回一批区域
        │   │
        │   │
        │   ▼
        │   process_batch(verify_batch): 调用批量验证
        │   │
        │   │
        │   └─ 返回结果数组 [SAT/MAYBE/UNSAT]
        │   │
        │   │
        │   ▼
        │   处理每个结果
        │   │
        │   ├───┐
        │   │   SAT: 更新统计，记录通过
        │   │   │   └───┘
        │   │   │   │
        │   │   ├───┐
        │   │   │   MAYBE: 将子区域加入队列
        │   │   │   │   │
        │   │   │   └───┘
        │   │   │   │   │
        │   │   └───┐
        │   │   │   UNSAT: 记录反例
        │   │   │   └───┘
        │   │   │   │
        │   │   └───┘
        │   │
        │   └─ 更新进度条
        │   │
        │   └─ 否: 队列为空，退出
        │
        │
        ▼
    计算最终统计并输出
        │
        └─── 完成

---
## CBF条件验证流程

    verify_batch(batch)
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤1: 计算网络输出界    │
    └─────────────────────────┘
        │
        └─ compute_network_bounds(batch)
        │
        ▼
    ┌─────────────────────────┐
    │ CROWN 前向传播             │
    └─────────────────────────┘
        │
        ├───┐
        │   初始化输入界: A = I, b = 0
        │   │
        │   └───┘
        │   │   │
        │   ├───┐
        │   │   for 每个全连接层 i
        │   │   │
        │   │   ├───┐
        │   │   │   获取权重和偏置 W, b
        │   │   │   └───┘
        │   │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   分解权重: W⁺ = max(0, W), W⁻ = W - W⁺
        │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   计算预激活界
        │   │   │   │   │
        │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   激活函数松弛
        │   │   │   │   │
        │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   └───┘
        │   │   │   │   │
        │   │   │   └───┘
        │   │   │
        │   └───┘
        │   │
        └─ 存储最终界 h(x) ∈ [h_min, h_max]
        │
        └─── 完成
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤2: 判断区域类型        │
    └─────────────────────────┘
        │
        └─ for 每个区域
        │
        │
        │   ├─── Case 1: h_max < 0
        │   │   │
        │   │   ├───┐
        │   │   │   含义: 障碍函数在整个区域上为负
        │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   操作: 直接标记为 SAT
        │   │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   └─ SampleResultSAT(result_type="unsafe_region")
        │   │   │
        │   │   │
        │   │   └───┘
        │   │
        │   ├─── Case 2: 与真正不安全集相交
        │   │   │
        │   │   ├───┐
        │   │   │   检查: unsafe_region(sample, dynamics_model)
        │   │   │   └───┘
        │   │   │   │   │
        │   │   │   ├─── Subcase 2a: h_min >= 0
        │   │   │   │   │
        │   │   │   │   │   ├───┐
        │   │   │   │   │   │   含义: h 在真正不安全集上为正（违规）
        │   │   │   │   │   │   │   └───┘
        │   │   │   │   │   │   │   │
        │   │   │   │   │   │   │   └─ SampleResultUNSAT(result_type="h_positive_in_unsafe")
        │   │   │   │   │   │   │
        │   │   │   │   │   │   │
        │   │   │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   │   └─ Subcase 2b: h_min < 0
        │   │   │   │   │
        │   │   │   │   │   ├───┐
        │   │   │   │   │   │   含义: 需要分割
        │   │   │   │   │   │   │   └───┘
        │   │   │   │   │   │   │   │
        │   │   │   │   │   │   │   └─ _handle_split() → SampleResultMAYBE
        │   │   │   │   │   │   │
        │   │   │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   └───┘
        │   │   │
        │   │   └───┘
        │   │
        │   └─ Case 3: 安全区域
        │   │   │
        │   │   └─ 不属于上述情况
        │   │   │
        │   │   │
        │   │   └─ 添加到 to_check_cbf_cond 列表
        │   │
        │
        └─── 完成
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤3: 计算Jacobian界      │
    └─────────────────────────┘
        │
        └─ if to_check_cbf_cond 不为空
        │
        │
        │   ├───┐
        │   │   keep_indices(to_check_cbf): 过滤界
        │   │   └───┘
        │   │   │   │
        │   ├───┐
        │   │   compute_partial_derivative_bounds(input_idx=None, output_idx=0)
        │   │   │
        │   │   └─ 计算 ∂h/∂x 的界
        │   │   │   │   │
        │   │   └─── 完成
        │   │
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤4: eta迭代验证        │
    └─────────────────────────┘
        │
        └─ for eta in [(0.5, 0.5)]
        │
        │   ├─── eta[0] = 0.5: 用于漂移项
        │   └─ eta[1] = 0.5: 用于控制项
        │
        │
        │   ▼
        └─ _verify_cbf_condition_affine(subbatch, eta=eta)
        │
        ▼
    ┌─────────────────────────┐
    │ 步骤5: 处理验证结果      │
    └─────────────────────────┘
        │
        └─ for 每个验证的区域
        │
        │
        │   ├─── 情况A: cbf_verified == True
        │   │   │
        │   │   └─ SampleResultSAT(result_type="safe_cbf_verified")
        │   │
        │   │
        │   ├─── 情况B: cbf_verified == False 且 counter_verified == True
        │   │   │
        │   │   └─ SampleResultUNSAT(result_type="safe_cbf_violation")
        │   │
        │   │
        │   └─ 情况C: cbf_verified == False 且 counter_verified == False
        │   │   │
        │   │   └─ _handle_split() → SampleResultMAYBE
        │
        │
        └─── 完成

---

## CROWN线性界传播流程

    compute_network_bounds(batch)
        │
        ▼
    根据区域类型计算输入界
        │
        ├─ SimplicialRegion: 顶点的最小/最大值
        └─ HyperrectangularRegion: center ± radius
        │
        └─── 完成
        │
        ▼
    初始化仿射系数 A = I, b = 0
        │
        └─── 完成
        │
        ▼
    ┌─────────────────────────┐
    │ for 每个全连接层 i       │
    └─────────────────────────┘
        │
        ├───┐
        │   获取权重和偏置 W, b
        │   └───┘
        │   │   │
        │   ├───┐
        │   │   分解权重: W⁺ = max(0, W), W⁻ = W - W⁺
        │   │   └───┘
        │   │   │   │
        │   ├───┐
        │   │   计算预激活界 A_y, a_y
        │   │   │
        │   │   └───┘
        │   │   │   │   │   │
        │   │   │   ├─ A_y_L = W⁺ @ A_L + W⁻ @ A_U
        │   │   │   ├─ a_y_L = W⁺ @ a_L + W⁻ @ a_U + b
        │   │   │   ├─ A_y_U = W⁺ @ A_U + W⁻ @ A_L
        │   │   │   └─ a_y_U = W⁺ @ a_U + W⁻ @ a_L + b
        │   │   │
        │   │   │
        │   │   └───┘
        │   │   │
        │   │   ▼
        │   │   计算区间界 [y_lb, y_ub]
        │   │   │
        │   │   │
        │   │   ├─ SimplicialRegion: 在顶点上求最小/最大
        │   │   └─ HyperrectangularRegion: 使用中心和半径
        │   │   │
        │   │   │   │
        │   │   └───┘
        │   │   │
        │   │   ├───┐
        │   │   │   最后一层?
        │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   ├─ 是: 直接存储界
        │   │   │   └─ 否: 继续激活函数松弛
        │   │   │   │   │   │
        │   │   │   └───┘
        │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   激活函数松弛
        │   │   │   │   │
        │   │   │   │   │
        │   │   │   │   └─ 返回: alpha_L, beta_L, alpha_U, beta_U
        │   │   │   │   │
        │   │   │   │   │   │
        │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   │   更新仿射系数 A, a
        │   │   │   │   │
        │   │   │   │   │
        │   │   │   │   ├─ A_L = alpha_L⁺ @ A_y_L + alpha_L⁻ @ A_y_U
        │   │   │   │   ├─ a_L = alpha_L⁺ @ a_y_L + alpha_L⁻ @ a_y_U + beta_L
        │   │   │   │   ├─ A_U = alpha_U⁺ @ A_y_U + alpha_U⁻ @ A_y_L
        │   │   │   │   └─ a_U = alpha_U⁺ @ a_y_U + alpha_U⁻ @ a_y_L + beta_U
        │   │   │   │   │
        │   │   │   │   │
        │   │   │   │   ▼
        │   │   │   │   计算激活后界 [current_lb, current_ub]
        │   │   │   │   │
        │   │   │   │   │
        │   │   │   │   ├─ SimplicialRegion: 在顶点上求最小/最大
        │   │   │   │   └─ HyperrectangularRegion: 使用中心和半径
        │   │   │   │   │   │
        │   │   │   │   │   │
        │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   │   存储到 forward_bounds
        │   │   │   │
        │   │   │   │   │
        │   │   │   └───┘
        │   │   │
        │   └───┘
        │
        └─── 完成

    compute_partial_derivative_bounds(input_idx=None, output_idx=0)
        │
        ▼
    从最后一层开始
        │
        └─ _get_jacobian_bounds_for_layer(L)
        │
        │
        ┌─────────────────────────┐
    │ 反向传播循环 L-1 到 1 │
    └─────────────────────────┘
        │
        ├───┐
        │   获取当前层Jacobian界: Lambda, lambda
        │   │
        │   └─ _get_jacobian_bounds_for_layer(i)
        │   │   │
        │   ├───┐
        │   │   激活函数导数松弛
        │   │   │   └─ 返回 S_L, s_L, S_U, s_U
        │   │   │   │   │
        │   │   ├───┐
        │   │   │   应用链式法则计算界
        │   │   │   │   └─ J^(i) = M^(i+1) @ J^(i-1)
        │   │   │   │   │   │
        │   │   │   └───┘
        │   │   │   │   │
        │   │   └───┘
        │   │   │
        │   └───┘
        │
        └─── 完成

---

## 泰勒展开流程

    compute_dynamics_bounds_taylor(batch, dynamics_model)
        │
        ▼
    创建数值转换器和泰勒线性化器
        │
        ├─── TorchTranslator
        └─ TaylorLinearization(dynamics_model, translator)
        │
        └─── 完成
        │
        ▼
    根据区域类型创建批次区域对象
        │
        ├─ SimplicialRegion: 堆叠顶点
        └─ HyperrectangularRegion: 堆叠中心和半径
        │
        └─── 完成
        │
        ▼
    ┌─────────────────────────┐
    │ 线性化漂移项 f(x)        │
    └─────────────────────────┘
        │
        └─ linearize_sample(sample)
        │
        │
        ├───┐
        │   确定区域类型
        │   └───┘
        │   │   │
        │   ├─── SimplicialRegion
        │   │   │
        │   │   └─ first_order_certified_taylor_expansion_simplex()
        │   │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   创建 TaylorTranslator
        │   │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   初始化 x = c ± vertices
        │   │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   计算动力学: y = f(x)
        │   │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   └─── 返回 CertifiedFirstOrderTaylorExpansion
        │   │   │   │   │   │
        │   │   │   └───┘
        │   │   │
        │   └─ HyperrectangularRegion
        │   │   │
        │   │   └─ first_order_certified_taylor_expansion()
        │   │   │   │   │
        │   │   │   └─ 类似处理，但用 center ± radius
        │   │   │
        │   │
        │   └───┘
        │
        │   ├───┐
        │   提取泰勒展开信息
        │   └───┘
        │   │   │
        │   ├─── jacobian = ∇f(c)
        │   ├─── f_c = f(c)
        │   ├─── remainder = [r_L, r_U]: 余项界
        │   └─ expansion_point = c
        │   │   │   │
        │   └─── 完成
        │
        │   ├───┐
        │   构造仿射界 f(x) ∈ [A_L x + b_L, A_U x + b_U]
        │   └───┘
        │   │   │
        │   ├─── A_lower = jacobian
        │   ├─── b_lower = f_c - jacobian @ c + r_L
        │   ├─── A_upper = jacobian
        │   └─ b_upper = f_c - jacobian @ c + r_U
        │   │
        │   └─── 完成
        │
        │
        ▼
    ┌─────────────────────────┐
    │ 线性化控制项 g(x) (如果有控制输入) │
    └─────────────────────────┘
        │
        └─ 类似漂移项的处理
        │
        ▼
    返回 (f_affine_bounds, g_affine_bounds)
        │
        └─── 完成

---

## 区域分割流程

    验证失败，调用 _handle_split()
        │
        ▼
    ┌─────────────────────────┐
    │ 条件1: 深度检查            │
    └─────────────────────────┘
        │
        ├─ 检查: max_depth is not None and sample.depth >= max_depth?
        │   │
        │   ├─~ 是: 达到最大深度
        │   │   │
        │   │   ├───┐
        │   │   │   取区域中心作为反例
        │   │   │   └───┘
        │   │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   返回 SampleResultUNSAT(result_type="depth_limit_reached")
        │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   └─── 早期返回，不执行后续代码
        │   │   │
        │   └─ 否: 继续到条件2
        │   │
        │   └───┘
        │
        ▼
    ┌─────────────────────────┐
    │ 条件2: 体积检查            │
    └─────────────────────────┘
        │
        ├─ 检查: sample._compute_volume() > min_volume?
        │   │
        │   ├─ 是: 区域足够大，可以分割
        │   │   │
        │   │   ├───┐
        │   │   │   调用 sample.split()
        │   │   │   └───┘
        │   │   │   │   │
        │   │   │   ▼
        │   │   │   SimplicialRegion.split()
        │   │   │   │
        │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   找到最长边: get_max_edge_length()
        │   │   │   │   │
        │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   │   ├───┐
        │   │   │   │   │   计算中点: midpoint = (v1 + v2) / 2
        │   │   │   │   │   └───┘
        │   │   │   │   │   │   │
        │   │   │   │   ├───┐
        │   │   │   │   │   创建新单纯形1: 替换 v2 为 midpoint
        │   │   │   │   │   └───┘
        │   │   │   │   │   │   │
        │   │   │   │   ├───┐
        │   │   │   │   │   创建新单纯形2: 替换 v1 为 midpoint
        │   │   │   │   │   └───┘
        │   │   │   │   │   │   │   │
        │   │   │   │   │   │   └─ 返回 (region1, region2), depth + 1
        │   │   │   │   │   │
        │   │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   └───┘
        │   │   │   │
        │   │   │   ├───┐
        │   │   │   │   分割成功?
        │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   │   ├─~ 是: 返回了新区域
        │   │   │   │   │   │
        │   │   │   │   │   ├───┐
        │   │   │   │   │   │   │   返回 SampleResultMAYBE(包含新区域)
        │   │   │   │   │   │   │   └───┘
        │   │   │   │   │   │   │   │   │
        │   │   │   │   │   │   │   └─── 早期返回，不执行后续代码
        │   │   │   │   │   │
        │   │   │   │   │   └─ 否: 分割失败，继续到条件3
        │   │   │   │   │   │
        │   │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   └─ 否: 区域太小，继续到条件3
        │   │   │   │
        │   │   │   │   │
        │   │   │   └───┘
        │   │   │
        │   └───┘
        │
        ▼
    ┌─────────────────────────┐
    │ 条件3: 默认处理 (无法分割) │
    └─────────────────────────┘
        │
        ├───┐
        │   取区域中心作为反例
        │   └───┘
        │   │   │
        │   ├───┐
        │   返回 SampleResultUNSAT(result_type=unsat_type)
        │   └───┘
        │   │
        └─── 完成

### 分割决策真值表

| 深度检查 | 体积检查 | 结果 | 原因 |
|---------|---------|------|------|
| depth ≥ max_depth | （不检查）| UNSAT (depth_limit_reached) | 达到最大深度 |
| depth < max_depth | volume > min_volume | MAYBE (继续分割) | 区域可分割 |
| depth < max_depth | volume ≤ min_volume | UNSAT (unsat_type) | 区域太小 |
| max_depth is None | volume > min_volume | MAYBE (继续分割) | 无深度限制 |
| max_depth is None | volume ≤ min_volume | UNSAT (unsat_type) | 区域太小 |

### 典型场景示例

    场景1: 正常分割
        │
        ├─ max_depth = 13
        ├─ sample.depth = 3
        ├─ sample.volume = 1e-5
        ├─ min_volume = 1e-8
        │
        └─ 结果: MAYBE (返回两个子区域，depth=4)

    场景2: 达到深度限制
        │
        ├─ max_depth = 13
        ├─ sample.depth = 13
        ├─ sample.volume = 1e-5
        │
        └─ 结果: UNSAT (depth_limit_reached)

    场景3: 区域太小
        │
        ├─ max_depth = 13
        ├─ sample.depth = 8
        ├─ sample.volume = 1e-10
        ├─ min_volume = 1e-8
        │
        └─ 结果: UNSAT (safe_cbf_violation)

    场景4: 无深度限制
        │
        ├─ max_depth = None
        ├─ sample.depth = 20
        ├─ sample.volume = 1e-4
        ├─ min_volume = 1e-8
        │
        └─ 结果: MAYBE (返回两个子区域，depth=21)

---

## 数据流转总览

    输入数据
        │
        ├─── 神经网络模型 (BarrierNN)
        │   ├─ 权重矩阵 W, 偏置 b
        │   └─ 激活函数 (ReLU/Tanh/Sigmoid)
        │
        ├─── 动力学系统 (Barrier2System)
        │   ├─ f(x): 漂移项
        │   ├─ g(x): 控制项
        │   └─ 控制边界 u_min, u_max
        │
        └─ 初始网格 (SimplicialRegion列表)
        │
        └─── 流向验证流程

    验证处理
        │
        ├─── 区域队列 (LIFO Queue)
        │   ├─ 输入: 初始区域
        │   ├─ 中间: MAYBE结果的子区域
        │   └─ 终止: 队列为ari
        │
        ├─── 批量处理
        │   ├─ 输入: 一批区域
        │   └─ 输出: 结果数组
        │
        └─ 验证策略
        │
        └─── 流向结果输出

    界计算
        │
        ├─── 神经网络界 (CROWN)
        │   ├─ 输入: 区域, 网络参数
        │   ├─ 过程: 前向传播计算 h(x) 和 ∇h(x) 界
        │   └─ 输出: 仿射系数和常数项
        │
        ├─── 动力学界 (Taylor)
        │   ├─ 输入: 区域, f(x)/g(x) 函数
        │   ├─ 过程: 泰勒展开计算 f(x), g(x) 界
        │   └─ 输出: 仿射系数和常数项
        │
        └─ McCormick乘积界
        │
        └─── 流向最终验证

    结果输出
        │
        ├─── SAT (SampleResultSAT)
        │   ├─ 类型: "unsafe_region" 或 "safe_cbf_verified"
        │   └─ 操作: 停止处理该区域
        │
        ├─── MAYBE (SampleResultMAYBE)
        │   ├─ 包含: 子区域列表
        │   └─ 操作: 将子区域加入队列
        │
        └─ UNSAT (SampleResultUNSAT)
        │   ├─ 类型: "depth_limit_reached", "safe_cbf_violation", 等
        │   └─ 操作: 返回反例，停止处理

    最终统计
        │
        ├─── 通过百分比
        ├─── 未验证百分比
        ├─── 总计算时间
        ├─── 处理速度
        └─ 总区域数

---

## 关键组件调用关系

    main()
        │
        ├─── verify_cbf()
        │   │
        │   ├─── CBFVerificationStrategy
        │   │   │
        │   │   ├─── initialize_worker()
        │   │   │   │
        │   │   │   ├─── 加载 BarrierNN 模型
        │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   ├─── 创建 CrownPartialLinearization
        │   │   │   │   │
        │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   └───┘
        │   │   │   │
        │   │   ├─── verify_batch()
        │   │   │   │
        │   │   │   ├─── compute_network_bounds()
        │   │   │   │   │   │
        │   │   │   │   └─ CrownPartialLinearization 前向传播
        │   │   │   │   │
        │   │   │   │   │
        │   │   │   ├─── compute_partial_derivative_bounds()
        │   │   │   │   │   │
        │   │   │   │   └─ CrownPartialLinearization 反向传播
        │   │   │   │   │   │
        │   │   │   │   │
        │   │   │   ├─── _verify_cbf_condition_affine()
        │   │   │   │   │   │
        │   │   │   │   │   ├─── compute_dynamics_bounds_taylor()
        │   │   │   │   │   │   │   │
        │   │   │   │   │   │   └─ TaylorLinearization 泰勒展开
        │   │   │   │   │   │   │   │
        │   │   │   │   │   │   ├─── McCormick 乘积松弛
        │   │   │   │   │   │   │   └─ 计算乘积界
        │   │   │   │   │   │   │   │
        │   │   │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   │   │   ├─── _handle_split()
        │   │   │   │   │   │   │
        │   │   │   │   │   │   ├─── SimplicialRegion.split()
        │   │   │   │   │   │   │   └─ 分割区域
        │   │   │   │   │   │   │   │
        │   │   │   │   │   │   └───┘
        │   │   │   │   │   │
        │   │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   │   └───┘
        │   │   │   │   │
        │   │   │   └───┘
        │   │   │   │   │
        │   │   └───┘
        │   │   │   │
        │   │   └───┘
        │   │   │   │
        │   └───┘
        │   │
        │   └───┘
        │   │
        │   └───┘
        │
        └───┘
        │
        ├─── SinglethreadExecutor.execute()
        │   │
        │   ├─── 管理 LIFO 队列
        │   ├─── 调用批量处理
        │   └─ 聚合结果
        │   │
        └───┘
        │
        └───┘

---

## 总结

整个CBF验证流程的核心层次：

    第一层: 用户入口
        └─ main() → 命令行解析 → 系统选择

    第二层: 验证初始化
        └─ 模型加载 → CROWN初始化 → 策略生成

    第三层: 执行循环
        └─ LIFO队列 → 批量处理 → 结果聚合

    第四层: 区域验证
        └─ 网络界计算 → 区域类型判断 → CBF条件验证

    第五层: 界计算
        ├─ CROWN: 神经网络界传播
        ├─ Taylor: 动力学函数线性化
        └─ McCormick: 乘积松弛

    第六层: 结果处理
        ├─ SAT: 通过验证
        ├─ MAYBE: 分割区域
        └─ UNSAT: 返回反例

关键设计原则:
1. 深度优先搜索: 使用LIFO队列
2. 批量处理: 减少函数调用开销
3. 模块化设计: 独立的组件
4. 三级分割决策: 深度 → 体积 → 默认
5. 保守性控制: 体积和深度限制
