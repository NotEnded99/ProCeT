任务目标

请帮我重构一个用于修复"神经控制障碍函数 (Neural CBF)"的 PyTorch 项目。目前的旧版本 (main_v1.py) 使用了 LBP（线性边界传播）计算的雅可比矩阵，并结合 QP（二次规划）进行参数投影修复。但由于 LBP 梯度的保守性和 McCormick 松弛导致的梯度爆炸，效果并不理想。

现在，我需要你编写 v2 版本 的代码 (main_v2.py, optimizer_module_v2.py, geometry_module_new_v2.py)，核心逻辑改为："对抗采样 (Adversarial Sampling) + 采样估计真实梯度 + QP 约束投影"。

---

核心算法逻辑

定义1. 寻找最坏点 (Adversarial Sampling)：

对于所有已经验证通过的安全区域（Verified Regions - Simplices/单纯形），在每个区域内批量随机采样 K 个点（利用 Dirichlet 分布生成重心坐标），计算这些点的 CBF 李导数条件：
    cbf_condition(x) = ∇h(x;θ)·f(x) + sup_u[∇h(x;θ)·g(x)·u] + α·h(x;θ)
其中：
  - ∇h(x;θ) = torch.autograd.grad(h(x), x)  对输入的梯度
  - f(x) = dynamics_model.compute_f(x)  漂移动力学
  - g(x) = dynamics_model.compute_g(x)  控制动力学
  - sup_u 项：对于仿射控制系统 = sum_j max(∇h·g_j·u_max_j, ∇h·g_j·u_min_j)
  - α 是 class-K 函数参数（代码中用 alpha * h(x)）

选出每个区域内使 cbf_condition 最小的那个点，记为 x^*_v。

对于修复失败的区域（Failed Regions），同样采用这种批量采样方法，提取出违规最严重的具体点：
  - 对于 F_h_positive_in_unsafe（障碍区内 h(x) > 0）：选出 h(x) 最大的点
  - 对于 F_safe_cbf_violation / F_depth_limit_reached（CBF 条件违规）：选出 cbf_condition 最小的点

定义2. 计算真实雅可比矩阵 (J_true)：

把已验证区域的 N 个最坏点 x^*_v 输入网络，计算 cbf_condition 对神经网络参数 θ 的一阶偏导数，构成形状为 [N, P] 的雅可比矩阵 J_true。
使用 torch.func.vmap(jacrev) 加速批量计算。

定义3. 计算破坏性梯度 (g_F)：

在修复失败区域的最坏采样点上计算 Repair Loss 并进行反向传播，得到原始修复梯度 g_F。

Repair Loss 公式（参考 train_cbf.py:compute_cbf_loss）：
  - F_h_positive_in_unsafe（障碍区违规）:
    loss_unsafe = mean( F.softplus(h(x) + margin, beta=5.0) )
    其中 margin = 0（目标 h(x) <= 0）
  - F_safe_cbf_violation / F_depth_limit_reached（CBF 条件违规）:
    loss_cbf = mean( F.softplus(cbf_margin - cbf_condition(x), beta=5.0) )
    其中 cbf_margin = 0（目标 cbf_condition >= 0）
  总损失 = mean(所有有效 loss 项)，然后 .backward()

定义4. QP 约束投影与更新：

将 g_F 和雅可比矩阵 J_true 进行 L2 归一化后，送入 cvxpy 构建 QP 对偶问题：
  min_λ 1/2 || J_true^T · λ - g_F ||^2   s.t.  λ >= 0

求解出 λ^* 后，得到更新方向 d = g_F - J_true^T · λ^*，然后使用定长步长 lr 进行参数更新：
  θ_new = θ_old - lr * d

---

文件划分与功能要求

请提供以下三个完整的、可以直接运行的 Python 文件代码：

1. geometry_module_new_v2.py

核心功能：负责单纯形的采样和最坏点的提取。

实现 sample_simplices_batched(vertices_list, num_samples, device):
  - 输入: vertices_list = [np.ndarray of shape [D+1, D], ...], num_samples = K
  - 利用 Dirichlet 分布：alpha = torch.ones(D+1)，生成 [B, K, D+1] 的重心坐标
  - 将重心坐标映射回欧几里得坐标: x_samples = barycentric @ vertices
  - 返回: x_samples of shape [B, K, D]

实现 find_worst_case_points(model, dynamics, simplices_list, num_samples=1000):
  - 输入: simplices_list = V_safe 或 V_unsafe 列表
  - 对每个单纯形采样 num_samples 个点
  - 对于 V_safe: 计算 cbf_condition(x)，取最小值对应的点
  - 对于 V_unsafe: 计算 h(x)，取最大值对应的点（即障碍函数值最大、最违反 h<0 条件的点）
  - 返回: worst_points [N, D], worst_values [N]

实现 compute_cbf_condition(model, dynamics, x, translator):
  - 计算单个点（或 batch）x 的 CBF 条件值
  - 公式: cbf = ∇h·f + sup_u[∇h·g·u] + α·h
  - 返回: cbf_condition 值 shape [B]

2. optimizer_module_v2.py

核心功能：计算雅可比矩阵、计算修复 Loss 梯度以及执行 QP 参数更新。

实现 compute_jacobian_at_worst_points(model, dynamics, worst_points, translator):
  - 输入: worst_points [N, D]（N 个已验证区域的最坏点）
  - 使用 torch.func.vmap + torch.func.jacrev 对 batch 计算 ∂cbf_condition/∂θ
  - 返回: J of shape [N, num_params]，需要 detach() 但保留 requires_grad=False

实现 compute_sampled_repair_loss_and_grad(model, dynamics, failed_worst_points, margin=0.0, cbf_margin=0.0, beta=5.0):
  - failed_worst_points: dict with keys:
    - 'unsafe': list of (x, h_value) for F_h_positive_in_unsafe
    - 'safe': list of (x, cbf_value) for F_safe_cbf_violation / F_depth_limit_reached
  - 对 unsafe 类: loss = mean( F.softplus(h + margin, beta=beta) )
  - 对 safe 类: loss = mean( F.softplus(cbf_margin - cbf, beta=beta) )
  - total_loss = mean(所有有效项)
  - total_loss.backward()
  - 提取 .grad，展平拼接为 g_raw [num_params]
  - 返回: loss_value, g_raw

实现 qp_project_and_update(model, g_raw, J_verified, lr=1e-3):
  - g_raw: [num_params]，J_verified: [N, num_params]
  - epsilon = 1e-8
  - g_hat = g_raw / (||g_raw|| + epsilon)  # L2 单位化
  - J_hat = J_verified / ||J_verified||_2 along dim=1  # 每行 L2 单位化，shape [N, P]
  - cvxpy: min 0.5*sum((J_hat.T @ lam - g_hat)^2), s.t. lam >= 0
  - d = g_hat - J_hat.T @ lam.value
  - theta = theta - lr * d
  - 返回: loss.item(), active_constraints

3. main_v2.py

核心功能：主调度程序。

- 使用 argparse 解析参数: --activation (Relu/Tanh), --system (barr1/barr2/barr3), --iterations
- 初始化 Model（BarrierNN）、Dynamics（Barrier{N}System）、Translator
- 加载已验证区域: torch.load('New_repair/regions/verified_regions_{system}_{activation}.pt')
- 修复迭代 Loop（默认 10 次）:
  1. 调用 find_worst_case_points(V_safe, num_samples=500) → safe_worst
  2. 调用 find_worst_case_points(F_h_positive_in_unsafe, ...) → unsafe_worst
  3. 调用 find_worst_case_points(F_safe_cbf_violation + F_depth_limit_reached, ...) → cbf_worst
  4. 调用 compute_jacobian_at_worst_points(model, dynamics, safe_worst) → J_true
  5. 构造 failed_worst_points dict
  6. 调用 compute_sampled_repair_loss_and_grad(...) → loss, g_F
  7. 调用 qp_project_and_update(model, g_F, J_true, lr=1e-4) → update
  8. 评估验证通过率
  9. 保存模型

---

注意事项（避坑指南）

1. Dirichlet 分布采样代码参考:
   alpha = torch.ones(D + 1)
   barycentric = torch.distributions.Dirichlet(alpha).sample([batch_size, num_samples])  # [B, K, D+1]
   注意: Dirichlet sample 返回的是归一化的权重，直接与顶点相乘即可

2. vmap + jacrev 使用参考:
   from torch.func import vmap, jacrev
   # jacrev 计算每个样本对参数的手征雅可比，vmap 批量化
   compute_jac = vmap(jacrev(lambda m, x: compute_cbf_condition(m, dynamics, x, translator).squeeze()))
   J = compute_jac(model, worst_points_unbatched)  #worst_points_unbatched: [N, D]

3. 梯度归一化极其重要: 必须把 g_F 和 J 的每一行化为长度为 1 的向量

4. 代码规范: 完整的类型提示，中文注释，张量 shape 标注，可运行 Mock 数据测试
