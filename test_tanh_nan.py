"""
测试 LBP 下界对神经网络参数求梯度时，tanh 激活函数是否会导致 NaN。
不修改任何原有代码。
"""

import sys
import os
import copy
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System
from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization
from lbp_neural_cbf.regions import SimplicialRegion
from lbp_neural_cbf.cbf.verify_cbf import (
    _compute_dynamics_bounds_taylor,
    _batched_compute_mccormick_product_lower_bound,
    _batched_get_affine_function_bounds,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# 测试 1: 直接测试 tanh 梯度的 Lipschitz 界（核心问题点）
# ============================================================
print("\n" + "="*60)
print("测试 1: tanh 梯度的 Lipschitz 界")
print("="*60)

def test_tanh_gradient_bound():
    x = torch.randn(5, requires_grad=True, device=device, dtype=torch.float32)
    y = torch.tanh(x)
    sum_y = y.sum()

    # 一阶导数
    grad1 = torch.autograd.grad(sum_y, x, create_graph=True)[0]
    print(f"  一阶导数 grad1: {grad1}")

    # 二阶导数（tanh 的梯度的梯度）
    grad2 = torch.autograd.grad(grad1.sum(), x)[0]
    print(f"  二阶导数 grad2 (tanh''): {grad2}")

    if torch.isnan(grad2).any():
        print(f"  *** 发现 NaN! ***")
    else:
        print(f"  无 NaN")

test_tanh_gradient_bound()

# ============================================================
# 测试 2: 端到端测试 BarrierNN + Tanh + LBP 下界 -> 参数梯度
# ============================================================
print("\n" + "="*60)
print("测试 2: BarrierNN + Tanh + LBP 下界 -> 参数梯度")
print("="*60)

def test_barrier_nn_gradient():
    dynamics_model = Barrier1System(alpha=1.0)
    dynamics_model.activation_fnc = 'Tanh'

    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device,
        activation_fnc='Tanh'
    )
    model.eval()

    # 创建一个随机单纯形
    np.random.seed(42)
    D = dynamics_model.input_dim
    verts_np = np.random.randn(D+1, D) * 2.0
    batch = [SimplicialRegion(verts_np, output_dim=None)]

    # ---------- Unsafe: 网络输出上界 h_ub ----------
    print("\n  [Unsafe] 计算 h_ub 并对参数求梯度...")
    network_linearizer = CrownPartialLinearization(model, dtype=torch.float32)
    network_linearizer.compute_network_bounds(batch)
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    h_lb, h_ub = network_linearizer.get_network_output_bounds_with_grad(sample_idx=0)
    h_ub_scalar = h_ub.reshape(-1)[0]

    print(f"    h_ub = {h_ub_scalar.item():.6f}")

    params = list(model.parameters())
    grad_h_ub = torch.autograd.grad(h_ub_scalar, params, retain_graph=False)
    grad_vec = torch.cat([g.flatten() for g in grad_h_ub])

    print(f"    |grad_h_ub| = {grad_vec.norm().item():.6f}")
    if torch.isnan(grad_vec).any():
        print(f"    *** grad_h_ub 中有 NaN! NaN 数量: {torch.isnan(grad_vec).sum().item()} ***")
    else:
        print(f"    grad_h_ub 无 NaN")

    # ---------- Safe: CBF 李导数下界 min_L ----------
    print("\n  [Safe] 计算 min_L 并对参数求梯度...")
    try:
        f_affine_bounds, _ = _compute_dynamics_bounds_taylor(
            batch, dynamics_model, device=device, dtype=torch.float32
        )
        print(f"    f_affine_bounds 计算成功")

        A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
        # J_affine_L 和 J_affine_U 是 (A, b) 元组
        J_affine_L = (A_L.squeeze(1), b_L.squeeze(1))
        J_affine_U = (A_U.squeeze(1), b_U.squeeze(1))

        f_affine_L, f_affine_U = f_affine_bounds

        eta = (0.5, 0.5)
        M_D, c_D = _batched_compute_mccormick_product_lower_bound(
            J_affine_L,
            J_affine_U,
            f_affine_L,
            f_affine_U,
            batch,
            eta=eta[0],
            device=device,
            dtype=torch.float32,
        )
        M_D = M_D.sum(dim=-2)
        c_D = c_D.sum(dim=-1)

        # Class-K 项
        (A_L_net, a_L_net), _ = network_linearizer.get_network_linear_bounds()
        A_L_net_s = A_L_net.squeeze(1)
        a_L_net_s = a_L_net.squeeze(1)
        alpha_A_L = dynamics_model.alpha_function(A_L_net_s)
        alpha_a_L = dynamics_model.alpha_function(a_L_net_s)

        M_total = M_D + alpha_A_L
        c_total = c_D + alpha_a_L

        min_L, _ = _batched_get_affine_function_bounds(
            (M_total.unsqueeze(1), c_total.unsqueeze(1)),
            batch,
            device=device,
            dtype=torch.float32,
        )
        min_L = min_L.squeeze(-1)
        min_L_scalar = min_L[0]

        print(f"    min_L = {min_L_scalar.item():.6f}")

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        grad_min_L = torch.autograd.grad(min_L_scalar, params, retain_graph=False)
        grad_min_L_vec = torch.cat([g.flatten() for g in grad_min_L])

        print(f"    |grad_min_L| = {grad_min_L_vec.norm().item():.6f}")
        if torch.isnan(grad_min_L_vec).any():
            print(f"    *** grad_min_L 中有 NaN! NaN 数量: {torch.isnan(grad_min_L_vec).sum().item()} ***")
        else:
            print(f"    grad_min_L 无 NaN")

    except Exception as e:
        print(f"    异常: {e}")
        import traceback
        traceback.print_exc()

test_barrier_nn_gradient()

# ============================================================
# 测试 3: 模拟 compute_jacobian_matrix 方式（核心测试）
# ============================================================
print("\n" + "="*60)
print("测试 3: 模拟 compute_jacobian_matrix 的梯度计算方式")
print("="*60)

def test_jacobian_style_gradient():
    dynamics_model = Barrier1System(alpha=1.0)
    dynamics_model.activation_fnc = 'Tanh'

    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device,
        activation_fnc='Tanh'
    )
    model.eval()

    # 克隆模型（与 compute_jacobian_matrix 一样）
    model_grad = copy.deepcopy(model)
    for p in model_grad.parameters():
        p.requires_grad_(True)

    params_grad = list(model_grad.parameters())

    np.random.seed(42)
    D = dynamics_model.input_dim
    verts_np = np.random.randn(D+1, D) * 2.0
    batch = [SimplicialRegion(verts_np, output_dim=None)]

    network_linearizer = CrownPartialLinearization(model_grad, dtype=torch.float32)
    network_linearizer.compute_network_bounds(batch)
    network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

    try:
        f_affine_bounds, _ = _compute_dynamics_bounds_taylor(
            batch, dynamics_model, device=device, dtype=torch.float32
        )

        A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
        J_affine_L = (A_L.squeeze(1), b_L.squeeze(1))
        J_affine_U = (A_U.squeeze(1), b_U.squeeze(1))

        f_affine_L, f_affine_U = f_affine_bounds

        M_D, c_D = _batched_compute_mccormick_product_lower_bound(
            J_affine_L, J_affine_U, f_affine_L, f_affine_U,
            batch, eta=0.5, device=device, dtype=torch.float32,
        )
        M_D = M_D.sum(dim=-2)
        c_D = c_D.sum(dim=-1)

        (A_L_net, a_L_net), _ = network_linearizer.get_network_linear_bounds()
        A_L_net_s = A_L_net.squeeze(1)
        a_L_net_s = a_L_net.squeeze(1)
        alpha_A_L = dynamics_model.alpha_function(A_L_net_s)
        alpha_a_L = dynamics_model.alpha_function(a_L_net_s)

        M_total = M_D + alpha_A_L
        c_total = c_D + alpha_a_L

        min_L, _ = _batched_get_affine_function_bounds(
            (M_total.unsqueeze(1), c_total.unsqueeze(1)),
            batch, device=device, dtype=torch.float32,
        )
        min_L = min_L.squeeze(-1)
        output = min_L[0]

        print(f"  output (min_L) = {output.item():.6f}")

        # 模拟 compute_jacobian_matrix 的梯度计算
        grads = torch.autograd.grad(
            outputs=output,
            inputs=params_grad,
            retain_graph=False,
        )
        grad_vec = torch.cat([g.flatten() for g in grads])

        print(f"  |grad_vec| = {grad_vec.norm().item():.6f}")
        if torch.isnan(grad_vec).any():
            nan_count = torch.isnan(grad_vec).sum().item()
            print(f"  *** grad_vec 中有 NaN! NaN 数量: {nan_count} / {len(grad_vec)} ***")
            idx = 0
            for i, p in enumerate(model_grad.parameters()):
                numel = p.numel()
                grad_slice = grad_vec[idx:idx+numel]
                if torch.isnan(grad_slice).any():
                    print(f"    层 {i} ({p.shape}): {torch.isnan(grad_slice).sum().item()} NaN / {numel}")
                idx += numel
        else:
            print(f"  grad_vec 无 NaN")

    except Exception as e:
        print(f"  异常: {e}")
        import traceback
        traceback.print_exc()

test_jacobian_style_gradient()

# ============================================================
# 测试 4: 多批次测试 NaN 出现概率
# ============================================================
print("\n" + "="*60)
print("测试 4: 多批次测试 NaN 出现概率")
print("="*60)

def test_multiple_simplices():
    dynamics_model = Barrier1System(alpha=1.0)
    dynamics_model.activation_fnc = 'Tanh'

    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device,
        activation_fnc='Tanh'
    )
    model.eval()

    D = dynamics_model.input_dim
    num_tests = 50
    nan_count = 0
    nan_details = []

    for seed in range(num_tests):
        np.random.seed(seed)
        verts_np = np.random.randn(D+1, D) * 2.0
        batch = [SimplicialRegion(verts_np, output_dim=None)]

        model_grad = copy.deepcopy(model)
        for p in model_grad.parameters():
            p.requires_grad_(True)
        params_grad = list(model_grad.parameters())

        try:
            network_linearizer = CrownPartialLinearization(model_grad, dtype=torch.float32)
            network_linearizer.compute_network_bounds(batch)
            network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=None)

            f_affine_bounds, _ = _compute_dynamics_bounds_taylor(
                batch, dynamics_model, device=device, dtype=torch.float32
            )

            A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
            J_affine_L = (A_L.squeeze(1), b_L.squeeze(1))
            J_affine_U = (A_U.squeeze(1), b_U.squeeze(1))

            f_affine_L, f_affine_U = f_affine_bounds

            M_D, c_D = _batched_compute_mccormick_product_lower_bound(
                J_affine_L, J_affine_U, f_affine_L, f_affine_U,
                batch, eta=0.5, device=device, dtype=torch.float32,
            )
            M_D = M_D.sum(dim=-2)
            c_D = c_D.sum(dim=-1)

            (A_L_net, a_L_net), _ = network_linearizer.get_network_linear_bounds()
            A_L_net_s = A_L_net.squeeze(1)
            a_L_net_s = a_L_net.squeeze(1)
            alpha_A_L = dynamics_model.alpha_function(A_L_net_s)
            alpha_a_L = dynamics_model.alpha_function(a_L_net_s)

            M_total = M_D + alpha_A_L
            c_total = c_D + alpha_a_L

            min_L, _ = _batched_get_affine_function_bounds(
                (M_total.unsqueeze(1), c_total.unsqueeze(1)),
                batch, device=device, dtype=torch.float32,
            )
            min_L = min_L.squeeze(-1)
            output = min_L[0]

            grads = torch.autograd.grad(outputs=output, inputs=params_grad, retain_graph=False)
            grad_vec = torch.cat([g.flatten() for g in grads])

            if torch.isnan(grad_vec).any():
                nan_count += 1
                nan_details.append((seed, torch.isnan(grad_vec).sum().item()))

        except Exception as e:
            nan_count += 1
            nan_details.append((seed, f"Exception: {e}"))

    print(f"  NaN 出现比例: {nan_count}/{num_tests}")
    if nan_details:
        print(f"  前5个 NaN 详情: {nan_details[:5]}")

test_multiple_simplices()

print("\n" + "="*60)
print("测试完成")
print("="*60)
