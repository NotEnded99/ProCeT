"""
修复算法的核心模块：LBP 重计算与雅可比提取

基于 verify_cbf.py 中的精确实现，重新计算：
1. compute_simplex_bound: 返回标量边界 (h_lb, h_ub) 和 min_L
2. compute_jacobian_matrix: 计算标量边界对网络参数的雅可比矩阵

核心思想：完全复用 verify_cbf.py 中的 CrownPartialLinearization 和
_batched_compute_mccormick_product_lower_bound / _batched_get_affine_function_bounds
来保证与 verify_cbf.py 的一致性。
"""

from typing import List, Tuple, Union

import copy
import numpy as np
import torch
import torch.nn as nn
import concurrent.futures
import cvxpy as cp

from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization
from lbp_neural_cbf.regions import SimplicialRegion
from lbp_neural_cbf.cbf.verify_cbf import (
    _compute_dynamics_bounds_taylor,
    _batched_compute_mccormick_product_lower_bound,
    _batched_get_affine_function_bounds,
    _vectorized_get_affine_function_bounds,
)
from torch.func import functional_call, jacrev



def _build_simplicial_batch(
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    device: torch.device,
    dtype: torch.dtype,
) -> List[SimplicialRegion]:
    """
    将单纯形顶点列表转换为 SimplicialRegion 对象列表。

    Args:
        vertices_list: 单纯形顶点列表，每个元素形状 [D+1, D]
        device, dtype: 张量设备和数据类型

    Returns:
        SimplicialRegion 对象列表（长度为 batch_size）
    """
    batch = []
    for verts in vertices_list:
        if isinstance(verts, torch.Tensor):
            verts_np = verts.cpu().numpy()
        else:
            verts_np = verts
        batch.append(SimplicialRegion(verts_np, output_dim=None))
    return batch


def _build_simplicial_batch_from_vertices(
    simplex_vertices: Union[torch.Tensor, np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> List[SimplicialRegion]:

    if isinstance(simplex_vertices, np.ndarray):
        verts = simplex_vertices
    else:
        verts = simplex_vertices.cpu().numpy()

    sample = SimplicialRegion(verts, output_dim=None)
    return [sample]


def _build_simplicial_batch(
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    device: torch.device,
    dtype: torch.dtype,
) -> List[SimplicialRegion]:
    """
    将单纯形顶点列表转换为 SimplicialRegion 对象列表。

    Args:
        vertices_list: 单纯形顶点列表，每个元素形状 [D+1, D]
        device, dtype: 张量设备和数据类型

    Returns:
        SimplicialRegion 对象列表（长度为 batch_size）
    """
    batch = []
    for verts in vertices_list:
        if isinstance(verts, torch.Tensor):
            verts_np = verts.cpu().numpy()
        else:
            verts_np = verts
        batch.append(SimplicialRegion(verts_np, output_dim=None))
    return batch


def _extract_lie_derivative_lower_bound(
    network_linearizer: CrownPartialLinearization,
    dynamics_bounds,          # ((A_L, b_L), (A_U, b_U)) for f(x)
    g_dynamics_bounds,         # ((A_L, b_L), (A_U, b_U)) for g(x) or None
    batch: List,               # SimplicialRegion list
    dynamics_model,
    device: torch.device,
    dtype: torch.dtype,
    eta: tuple = (0.5, 0.5),
) -> torch.Tensor:
    """
    复现 verify_cbf.py 中 _verify_cbf_condition_affine 的逻辑，
    计算 CBF 李导数条件的下界 min_L。

    完全按照 verify_cbf.py:_verify_cbf_condition_affine 的步骤实现：
    1. 从 network_linearizer 获取 Jacobian 界 (J_affine_L, J_affine_U)
    2. 从 dynamics_bounds 获取 f(x) 界 (f_affine_L, f_affine_U)
    3. 计算 J(x)·f(x) 的 McCormick 下界
    4. 添加 class-K 项 alpha(h(x))
    5. 如果有控制维度，计算 g(x) 项
    6. 用 _batched_get_affine_function_bounds 在单纯形上求最小值
    """
    n = dynamics_model.input_dim
    m = dynamics_model.control_dim
    f_affine_bounds = dynamics_bounds
    g_affine_bounds = g_dynamics_bounds

    # 1. 获取 Jacobian 界
    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()

    # print(f"DEBUG Jacobian bounds shapes: A_L={A_L.shape}, b_L={b_L.shape}, A_U={A_U.shape}, b_U={b_U.shape}")
    A_L = A_L.squeeze(1)  # [B, 1, n, n] -> [B, n, n]
    b_L = b_L.squeeze(1)  # [B, 1, n] -> [B, n]
    A_U = A_U.squeeze(1)
    b_U = b_U.squeeze(1)
    J_affine_L, J_affine_U = (A_L, b_L), (A_U, b_U)

    # 2. Dynamics 界
    f_affine_L, f_affine_U = f_affine_bounds
    if torch.isnan(f_affine_L[0]).any() or torch.isnan(f_affine_L[1]).any():
        print("DEBUG NaN: f_affine_L has NaN!")

    # 3. J(x)·f(x) 下界
    eta_drift = eta[0]
    M_D, c_D = _batched_compute_mccormick_product_lower_bound(
        J_affine_L,
        J_affine_U,
        f_affine_L,
        f_affine_U,
        batch,
        eta=eta_drift,
        device=device,
        dtype=dtype,
    )

    if torch.isnan(M_D).any() or torch.isnan(c_D).any():
        print("DEBUG NaN: M_D or c_D has NaN after McCormick!")
    M_D, c_D = M_D.sum(dim=-2), c_D.sum(dim=-1)  # Sum over state dims

    # 5. Class-K 项：alpha(h(x)) = alpha * h(x)
    (A_L_net, a_L_net), (A_U_net, a_U_net) = network_linearizer.get_network_linear_bounds()
    A_L_net = A_L_net.squeeze(1)  # [B, 1, n] -> [B, n]
    a_L_net = a_L_net.squeeze(1)  # [B, 1] -> [B]
    alpha_A_L = dynamics_model.alpha_function(A_L_net)
    alpha_a_L = dynamics_model.alpha_function(a_L_net)

    M_total = M_D + alpha_A_L
    c_total = c_D + alpha_a_L

    # 控制项
    if m > 0 and g_affine_bounds is not None:
        g_affine_L = g_affine_bounds[0][0], g_affine_bounds[0][1]
        g_affine_U = g_affine_bounds[1][0], g_affine_bounds[1][1]

        eta_control_L = eta[1]
        M_v_L, c_v_L = _batched_compute_mccormick_product_lower_bound(
            J_affine_L,
            J_affine_U,
            g_affine_L,
            g_affine_U,
            batch,
            eta=eta_control_L,
            device=device,
            dtype=dtype,
        )
        M_v_L, c_v_L = M_v_L.sum(dim=-2), c_v_L.sum(dim=-1)

        v_affine_L = (M_v_L, c_v_L)
        v_L_min, v_L_max = _batched_get_affine_function_bounds(
            v_affine_L, batch, device=device, dtype=dtype
        )

        u_min = torch.tensor(dynamics_model.u_min, device=device, dtype=dtype)
        u_max = torch.tensor(dynamics_model.u_max, device=device, dtype=dtype)

        M_v_L_u_min = M_v_L * u_min.unsqueeze(-1)
        c_v_L_u_min = c_v_L * u_min
        M_v_L_u_max = M_v_L * u_max.unsqueeze(-1)
        c_v_L_u_max = c_v_L * u_max

        for sample_idx, sample in enumerate(batch):
            M_C = torch.zeros(n, device=device, dtype=dtype)
            c_C = torch.tensor(0.0, device=device, dtype=dtype)

            v_sample_min = v_L_min[sample_idx]
            v_sample_max = v_L_max[sample_idx]

            pos_mask = v_sample_min >= 0
            if pos_mask.any():
                M_C += M_v_L_u_max[sample_idx, pos_mask].sum(dim=0)
                c_C += c_v_L_u_max[sample_idx, pos_mask].sum()

            neg_mask = v_sample_max <= 0
            if neg_mask.any():
                M_C += M_v_L_u_min[sample_idx, neg_mask].sum(dim=0)
                c_C += c_v_L_u_min[sample_idx, neg_mask].sum()

            mixed_mask = ~(pos_mask | neg_mask)
            if mixed_mask.any():
                v_u_min_b, _ = _vectorized_get_affine_function_bounds(
                    (M_v_L_u_min[sample_idx, mixed_mask], c_v_L_u_min[sample_idx, mixed_mask]),
                    sample,
                    device=device,
                    dtype=dtype,
                )
                v_u_max_b, _ = _vectorized_get_affine_function_bounds(
                    (M_v_L_u_max[sample_idx, mixed_mask], c_v_L_u_max[sample_idx, mixed_mask]),
                    sample,
                    device=device,
                    dtype=dtype,
                )
                c_C += torch.maximum(v_u_min_b, v_u_max_b).sum()

            M_total[sample_idx] += M_C
            c_total[sample_idx] += c_C

    # 7. 在单纯形上求最小值
    # print(f"DEBUG M_total.unsqueeze(1) shape: {M_total.unsqueeze(1).shape}")
    # print(f"DEBUG c_total.unsqueeze(1) shape: {c_total.unsqueeze(1).shape}")

    # print( )
    min_L, _ = _batched_get_affine_function_bounds(
        (M_total.unsqueeze(1), c_total.unsqueeze(1)),
        batch,
        device=device,
        dtype=dtype,
    )

    # print(f"DEBUG min_L shape before squeeze: {min_L.shape}")
    min_L = min_L.squeeze(-1)
    # print(f"DEBUG min_L shape after squeeze: {min_L.shape}")

    return min_L



def compute_simplex_bound(
    model: nn.Module,
    simplex_vertices: Union[torch.Tensor, np.ndarray],
    region_type: str,
    dynamics_model=None,
    translator=None
) -> torch.Tensor:
    """
    复用 verify_cbf.py 中的 CrownPartialLinearization + McCormick 方法，
    精确计算单纯形区域上的网络输出界和李导数下界。

    Args:
        model: 神经网络（BarrierNN 或 nn.Sequential）
        simplex_vertices: 单纯形顶点，形状 [V, D]，V = D+1
        region_type: 'unsafe' 或 'safe'
        dynamics_model: 动力学系统（safe 区域需要）
        translator: TorchTranslator（safe 区域需要）

    Returns:
        - unsafe: (h_lb, h_ub) 元组
        - safe: min_L 张量（标量，保留计算图）

    避坑：
        - 必须保证 requires_grad=True（保留计算图用于反向传播）
        - 严禁使用 with torch.no_grad()
        - 严禁任何 in-place 操作
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 转换顶点 ----------
    if isinstance(simplex_vertices, np.ndarray):
        verts_np = simplex_vertices
    else:
        verts_np = simplex_vertices.cpu().numpy()

    V, D = verts_np.shape
    if V != D + 1:
        raise ValueError(f"Simplex must have V=D+1 vertices, got V={V}, D={D}")

    # ---------- 创建 SimplicialRegion batch ----------
    sample = SimplicialRegion(verts_np, output_dim=None)
    batch = [sample]

    # ---------- 创建 CrownPartialLinearization ----------
    network_linearizer = CrownPartialLinearization(model, dtype=dtype)
    network_linearizer.compute_network_bounds(batch)
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    # ---------- unsafe: 网络输出界 ----------
    if region_type == 'unsafe':
        h_lb, h_ub = network_linearizer.get_network_output_bounds_with_grad(sample_idx=0)
        h_lb = h_lb.reshape(-1)
        h_ub = h_ub.reshape(-1)
        return h_lb, h_ub

    # ---------- safe: CBF 李导数下界 ----------
    if region_type == 'safe':
        if dynamics_model is None:
            raise ValueError("dynamics_model is required for 'safe' region type")

        # Dynamics 界（Taylor 线性化）
        try:
            f_affine_bounds, g_affine_bounds = _compute_dynamics_bounds_taylor(
                batch, dynamics_model, device=device, dtype=dtype
            )
            
            # if torch.isnan(f_affine_bounds).any():
            #     print("警告: f_affine_bounds 或者 g_affine_bounds 中存在 NaN ")

        except ValueError:
            raise ValueError("Failed to compute dynamics bounds for this region")

        # 计算 min_L
        min_L = _extract_lie_derivative_lower_bound(
            network_linearizer=network_linearizer,
            dynamics_bounds=f_affine_bounds,
            g_dynamics_bounds=g_affine_bounds,
            batch=batch,
            dynamics_model=dynamics_model,
            device=device,
            dtype=dtype,
        )

        if torch.isnan(min_L).any():
            print("警告: min_L 中存在 NaN 可能是由于 dynamics_bounds 计算不稳定导致的。")
        # min_L: [batch_size=1]，取第一个元素作为标量
        return min_L.reshape(-1)

    raise ValueError(f"Invalid region_type: {region_type}. Must be 'unsafe' or 'safe'")


def compute_simplex_bound_batch(
    model: nn.Module,
    vertices_list: List[Union[torch.Tensor, np.ndarray]],
    region_type: str,
    dynamics_model=None,
    translator=None
):
    """
    批量版 compute_simplex_bound：将多个单纯形区域打包成 batch，一次前向传播完成所有计算。
    大幅减少 CrownPartialLinearization 的创建次数，提升计算效率。

    Args:
        model: 神经网络
        vertices_list: 单纯形顶点列表，每个元素形状 [D+1, D]，长度 B
        region_type: 'unsafe' 或 'safe'
        dynamics_model: 动力学系统（safe 区域需要）
        translator: TorchTranslator（safe 区域需要）

    Returns:
        - unsafe: (h_lb_all, h_ub_all) 元组，每个形状 [B]
        - safe: min_L_all 张量，形状 [B]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- 构建 batch ----------
    batch = []
    for verts in vertices_list:
        if isinstance(verts, torch.Tensor):
            verts_np = verts.cpu().numpy()
        else:
            verts_np = verts
        
        # print(verts_np)
        batch.append(SimplicialRegion(verts_np, output_dim=None))

    B = len(batch)

    # print(len(batch), )


    # ---------- 创建 CrownPartialLinearization（只创建一次）----------
    network_linearizer = CrownPartialLinearization(model, dtype=dtype)
    network_linearizer.compute_network_bounds(batch)
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    # ---------- unsafe: 网络输出界 ----------
    if region_type == 'unsafe':
        # sample_idx=None 返回完整 batch tensor，不需要 clone
        h_lb_all, h_ub_all = network_linearizer.get_network_output_bounds_with_grad(sample_idx=None)
        # 形状 [B, ...]，取第一维
        return h_lb_all.reshape(B, -1)[:, 0], h_ub_all.reshape(B, -1)[:, 0]

    # ---------- safe: CBF 李导数下界 ----------
    if region_type == 'safe':
        if dynamics_model is None:
            raise ValueError("dynamics_model is required for 'safe' region type")

        try:
            f_affine_bounds, g_affine_bounds = _compute_dynamics_bounds_taylor(
                batch, dynamics_model, device=device, dtype=dtype
            )


        except ValueError:
            raise ValueError("Failed to compute dynamics bounds for this batch")

        min_L = _extract_lie_derivative_lower_bound(
            network_linearizer=network_linearizer,
            dynamics_bounds=f_affine_bounds,
            g_dynamics_bounds=g_affine_bounds,
            batch=batch,
            dynamics_model=dynamics_model,
            device=device,
            dtype=dtype,
        )
        # print(min_L.shape, B)
        # min_L 形状 [B]
        return min_L.reshape(B)

    raise ValueError(f"Invalid region_type: {region_type}. Must be 'unsafe' or 'safe'")



def _extract_layer_params(model: nn.Module) -> Tuple[List[nn.Linear], List[str], dict]:
    """从模型中提取层参数和命名信息。"""
    fc_layers = []
    activation_types = []
    param_dict = {}

    for name, param in model.named_parameters():
        param_dict[name] = param

    for module in model.modules():
        if isinstance(module, nn.Linear):
            fc_layers.append(module)
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            activation_types.append(type(module).__name__)

    return fc_layers, activation_types, param_dict



def copy_model_with_grad(model: nn.Module) -> nn.Module:
    """
    克隆模型，并将所有参数的 requires_grad 设为 True。
    用于构建计算图以便 autograd.grad 计算参数梯度。
    """
    cloned = copy.deepcopy(model)
    for p in cloned.parameters():
        p.requires_grad_(True)
    return cloned


def compute_jacobian_matrix(
    model: nn.Module,
    V_safe: List[Union[torch.Tensor, np.ndarray]],
    V_unsafe: List[Union[torch.Tensor, np.ndarray]],
    dynamics_model=None,
    translator=None,
    max_workers: int = 1  # 新增控制并发线程数的参数
) -> torch.Tensor:
    """
    使用多线程逐个单纯形计算雅可比矩阵，以加速原本的 for 循环。
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    all_vertices = []
    all_region_types = []

    for v in V_safe:
        all_vertices.append(v)
        all_region_types.append('safe')
    for v in V_unsafe:
        all_vertices.append(v)
        all_region_types.append('unsafe')

    N = len(all_vertices)
    num_params = sum(p.numel() for p in model.parameters())

    if N == 0:
        return torch.zeros(0, num_params, dtype=dtype, device=device)

    # 克隆模型用于构建计算图 (主线程共享)
    model_grad = copy_model_with_grad(model)
    params_grad = list(model_grad.parameters())

    # ==========================================
    # 1. 定义单线程 Worker 函数
    # ==========================================
    def process_single_simplex(vertices, region_type):
        """处理单个单纯形的前向推导与反向梯度计算"""
        # 前向计算
        if region_type == 'unsafe':
            h_lb, h_ub = compute_simplex_bound(
                model_grad, vertices, 'unsafe',
                dynamics_model=None, translator=None
            )
            output = h_ub.squeeze()
        else:
            min_L = compute_simplex_bound(
                model_grad, vertices, 'safe',
                dynamics_model=dynamics_model, translator=translator
            )
            output = min_L.squeeze()

        # 反向求导 (Autograd 是线程安全的)
        grads = torch.autograd.grad(
            outputs=output,
            inputs=params_grad,
            retain_graph=False, # 计算完毕立即释放局部图
        )
        grad_vec = torch.cat([g.flatten() for g in grads])
        # Debug NaN check
        if torch.isnan(grad_vec).any():
            idx = torch.isnan(grad_vec).nonzero(as_tuple=True)[0][0].item()
            print(f"  DEBUG: simplex {region_type} grad NaN at idx {idx}, first valid={grad_vec[idx+1].item() if idx+1 < len(grad_vec) else 'N/A'}")
        return grad_vec


    print(f"Starting multi-threading computation with {max_workers} workers...")
    
    all_grads = [None] * N  # 预先分配列表以保持顺序
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，记录其原始索引，以保证最终 J 矩阵的行顺序不乱
        future_to_idx = {
            executor.submit(process_single_simplex, all_vertices[i], all_region_types[i]): i 
            for i in range(N)
        }
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                grad_vec = future.result()
                all_grads[idx] = grad_vec
                completed_count += 1
                if completed_count % 100 == 0 or completed_count == N:
                    print(f"  Progress: {completed_count}/{N} simplexes processed.")
            except Exception as exc:
                print(f"Simplex {idx} ({all_region_types[idx]}) generated an exception: {exc}")
                raise exc # 如果发生错误，直接抛出，避免得到不完整的矩阵

    J = torch.stack(all_grads, dim=0)
    return J


def qp_project_and_update(
    model: nn.Module,
    g_raw: torch.Tensor,
    J_verified: torch.Tensor,
    lr: float = 1e-3,
    verbose: bool = False
) -> Tuple[float, float, float]:
    """
    使用二次规划 (QP) 求解拉格朗日对偶问题，计算满足安全约束的参数更新方向，并更新模型。

    Args:
        model: 神经网络
        g_raw: 修复失败区域产生的原始梯度，形状 [P]
        J_verified: 濒危已验证区域的雅可比矩阵，形状 [N, P]
        lr: 学习率 (由于方向被归一化，lr 将直接作为绝对步长)
        verbose: 是否打印调试信息

    Returns:
        (g_raw_norm, g_update_norm, active_constraints): 
        原始梯度范数，最终更新范数，起作用的安全约束数量(lambda > 0 的个数)
    """
    P = g_raw.shape[0]
    N = J_verified.shape[0] if J_verified is not None else 0

    # 获取原始参数的视图
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    if P != num_params:
        raise ValueError(f"梯度维度不匹配: g_raw {P} vs 参数 {num_params}")

    # ==========================================
    # 1. 极端情况防御：如果没有需要防御的已验证区域
    # ==========================================
    if N == 0:
        if verbose: print("  提示: 没有活动的安全约束，执行标准归一化梯度下降。")
        g_norm = g_raw.norm() + 1e-8
        g_update = g_raw / g_norm  # 只保留方向
        theta_old = torch.nn.utils.parameters_to_vector(params)
        theta_new = theta_old - lr * g_update
        torch.nn.utils.vector_to_parameters(theta_new, params)
        return g_norm.item(), g_update.norm().item(), 0

    # ==========================================
    # 2. 核心：强制 L2 归一化 (消除 LBP 梯度爆炸)
    # ==========================================
    epsilon = 1e-8
    
    # 将原始梯度化为单位向量
    g_raw_norm = g_raw.norm()
    g_hat = g_raw / (g_raw_norm + epsilon)

    # 将雅可比矩阵的**每一行**分别化为单位向量
    J_norms = torch.norm(J_verified, dim=1, keepdim=True)
    J_hat = J_verified / (J_norms + epsilon)

    # ==========================================
    # 3. 构建并求解 QP (在 CPU 上使用 cvxpy)
    # ==========================================
    # 将 Tensor 转为 numpy 供 cvxpy 使用
    J_np = J_hat.detach().cpu().numpy()  # [N, P]
    g_np = g_hat.detach().cpu().numpy()  # [P]

    # 定义未知数 lambda，长度为 N
    lam = cp.Variable(N, nonneg=True)

    # 目标函数: min 0.5 * || J^T * lambda - g_hat ||^2
    # 注意矩阵乘法: J_np.T 是 [P, N], lam 是 [N]
    residual = J_np.T @ lam - g_np
    objective = cp.Minimize(0.5 * cp.sum_squares(residual))
    prob = cp.Problem(objective)

    try:
        # 使用 OSQP 求解器，对于 QP 问题速度极快
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"QP 求解器未能找到最优解，状态: {prob.status}")
            
    except Exception as e:
        print(f"  警告: QP 求解失败 ({e})，降级为截断原梯度。")
        # 降级方案：如果不幸失败，强行砍掉原梯度大小，防止爆炸
        lam_value = np.zeros(N)
    else:
        lam_value = lam.value

    # 将求出的 lambda 转换回 GPU Tensor
    lam_star = torch.tensor(lam_value, dtype=g_raw.dtype, device=g_raw.device)

    # ==========================================
    # 4. 合成最终安全的更新方向
    # ==========================================
    # d = g_hat - J_hat^T * lambda_star
    # d_update 将严格保证 J_hat * d_update <= 0
    g_update = g_hat - (J_hat.T @ lam_star)

    # ==========================================
    # 5. 参数更新
    # ==========================================
    theta_old = torch.nn.utils.parameters_to_vector(params)
    theta_new = theta_old - lr * g_update
    torch.nn.utils.vector_to_parameters(theta_new, params)

    # 统计信息
    active_constraints = int(np.sum(lam_value > 1e-4))  # 统计起了实际阻挡作用的墙的数量
    update_norm = g_update.norm().item()

    if verbose:
        print(f"  |g_raw| (原始大小): {g_raw_norm.item():.2e} (已被归一化抛弃)")
        print(f"  |g_update| (最终方向长度): {update_norm:.4f}")
        print(f"  活跃安全约束数量: {active_constraints} / {N}")
        print(f"  |theta_new|: {theta_new.norm().item():.6f}")

    return g_raw_norm.item(), update_norm, active_constraints



