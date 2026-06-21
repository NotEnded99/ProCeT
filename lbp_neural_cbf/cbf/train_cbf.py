import copy
from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.uniform import Uniform
from tqdm import tqdm

from ..cbf.network import BarrierNN, empirical_cbf_validation
from ..translators import TorchTranslator


def save_onnx_model(model, save_path, input_size):
    """Save model in ONNX format."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, input_size, device=device)
    torch.onnx.export(
        model, dummy_input, save_path, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )


def generate_samples(
    batch_size: int,
    dynamics_model,
    barrier_net: torch.nn.Module,
    device: Optional[torch.device] = None,
    translator=None,
    proportions: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Generate a shuffled batch of samples mixing safe, unsafe, boundary,
    and uniform random points with configurable proportions.

    Args:
        batch_size: Total number of samples to generate
        dynamics_model: Model containing safe/unsafe set definitions
        barrier_net: Current barrier network for targeting h(x) = 0
        device: Device to place tensors on (auto-detected if None)
        translator: Torch translator for constraint evaluation
        proportions: Dict with 'safe', 'unsafe', 'boundary' proportions (0.0-1.0).
                    Remainder is uniform. Defaults to {'safe': 0.1, 'unsafe': 0.1,
                    'boundary': 0.5}

    Returns:
        Shuffled tensor of shape [batch_size, input_dim]
    """
    # Constants
    BOUNDARY_STD_FACTOR = 0.02
    MAX_BISECTION_ITERATIONS = 12
    MAX_BARRIER_ATTEMPTS = 5
    CANDIDATE_MULTIPLIER = 6
    MIN_CANDIDATE_BUFFER = 64

    # Determine device from barrier_net
    barrier_device = next(barrier_net.parameters()).device
    if device is None:
        device = barrier_device

    # Initialize translator
    if translator is None:
        translator = TorchTranslator(device=device)

    # Parse and normalize proportions
    if proportions is None:
        proportions = {"safe": 0.1, "unsafe": 0.1, "boundary": 0.5}

    safe_prop = max(0.0, min(1.0, float(proportions.get("safe", 0.01))))
    unsafe_prop = max(0.0, min(1.0, float(proportions.get("unsafe", 0.01))))
    boundary_prop = max(0.0, min(1.0, float(proportions.get("boundary", 0.1))))

    total_prop = safe_prop + unsafe_prop + boundary_prop
    if total_prop > 1.0:
        safe_prop /= total_prop
        unsafe_prop /= total_prop
        boundary_prop /= total_prop

    # Calculate target counts
    target_safe = int(batch_size * safe_prop)
    target_unsafe = int(batch_size * unsafe_prop)
    target_boundary = int(batch_size * boundary_prop)
    target_uniform = batch_size - (target_safe + target_unsafe + target_boundary)

    # Setup domain bounds and sampler
    input_domain = dynamics_model.input_domain
    bounds = input_domain.bounds if hasattr(input_domain, "bounds") else input_domain

    domain_tensor = torch.tensor(np.array(bounds, dtype=np.float32), dtype=torch.float32, device=device)
    lower_bounds = domain_tensor[:, 0].unsqueeze(0)
    upper_bounds = domain_tensor[:, 1].unsqueeze(0)
    domain_width = (upper_bounds - lower_bounds).clamp(min=1e-6)
    sampler = Uniform(domain_tensor[:, 0], domain_tensor[:, 1])

    point_tensors = []

    # Sample safe points
    if target_safe > 0:
        safe_points = dynamics_model.safe_set.sample_points(target_safe, device=device, use_torch=True)
        point_tensors.append(safe_points.reshape(target_safe, -1))

    # Sample unsafe points
    if target_unsafe > 0:
        unsafe_points = dynamics_model.unsafe_set_interior.sample_points(target_unsafe, device=device, use_torch=True)
        point_tensors.append(unsafe_points.reshape(target_unsafe, -1))

    # Sample uniform points
    if target_uniform + target_boundary > 0:
        uniform_points = sampler.sample(sample_shape=torch.Size([target_uniform + target_boundary]))
        # Sample boundary points using barrier network
        if target_boundary > 0:
            signed_distance_fnc = dynamics_model.safe_set.constraint(uniform_points, translator=TorchTranslator(device=uniform_points.device))
            _, boundary_points = torch.topk(signed_distance_fnc.abs(), k=target_boundary, largest=False)
            point_tensors.append(uniform_points[boundary_points])
            uniform_points = sampler.sample(sample_shape=torch.Size([target_uniform]))
        point_tensors.append(uniform_points)

    # Validate and combine
    if not point_tensors:
        raise ValueError("No points were sampled. Check batch_size value.")

    combined_batch = torch.cat(point_tensors, dim=0)

    if combined_batch.shape[0] != batch_size:
        raise ValueError(f"Generated {combined_batch.shape[0]} samples, expected {batch_size}")

    # Shuffle and return
    shuffle_indices = torch.randperm(combined_batch.shape[0], device=device)
    return combined_batch[shuffle_indices]


def compute_cbf_loss(
    barrier_net,
    dynamics_model,
    x_batch,
    translator,
    lambda_safe,
    lambda_unsafe,
    lambda_unsafe_max,
    lambda_cbf,
    lambda_bndry,
    unsafe_margin,
    safe_margin,
    cbf_margin,
    alpha=1.0,
):
    """
    Compute the barrier function training loss with proper prioritization:

    PHASE 1 - BOUNDARY LEARNING (both constraints equally critical):
    1. Safe set constraint: h(x) ≥ safe_margin for x in safe set
    2. Unsafe set constraint: h(x) ≤ -unsafe_margin for x in unsafe set

    PHASE 2 - FORWARD INVARIANCE:
    3. include CBF condition: ∇h·f + sup_u[∇h·g·u] + α(h) ≥ 0 where h(x) ≥ 0

    Uses squared loss for violations to strongly penalize deviations.

    Args:
        barrier_net: Neural barrier function
        dynamics_model: CBF dynamical system
        x_batch: Batch of state samples
        translator: Torch translator
        alpha: Class K function parameter for CBF condition
        lambda_safe: Weight for safe set loss (CRITICAL for boundary, default=1.0)
        lambda_unsafe: Weight for unsafe set loss (CRITICAL for boundary, default=10.0)
        lambda_unsafe_max: Weight for maximum unsafe h(x) loss (aggressive inward push, default=5.0)
        lambda_cbf: Weight for CBF condition loss (adds forward invariance, default=5.0)
        lambda_bndry: Weight for boundary value loss (encourages h(x) ~ 0 on boundary, default=0.1)
        unsafe_margin: Target margin for h(x) in unsafe region (we want h ≤ -unsafe_margin)
        safe_margin: Target margin for h(x) in safe region (we want h ≥ safe_margin)

    Returns:
        Total loss and individual loss components
    """
    # Determine if we need gradients (for CBF condition)
    need_gradients = lambda_cbf > 0

    if need_gradients:
        x_batch.requires_grad_(True)

    # Evaluate barrier function
    h_values = barrier_net(x_batch).squeeze()  # Shape: [batch_size]

    # Compute constraint values once (more efficient)
    safe_mask = dynamics_model.safe_set.contains(x_batch, translator)
    unsafe_mask = ~safe_mask  # Points in unsafe set

    device = h_values.device

    # 1. Safe set loss - we want h(x) ≥ safe_margin in safe regions
    # Using softplus (smooth ReLU) for better gradients
    if safe_mask.any():
        h_safe = h_values[safe_mask]
        safe_violations = F.softplus(safe_margin - h_safe, beta=100)
        safe_loss = torch.mean(safe_violations)
    else:
        safe_loss = torch.tensor(0.0, device=device)

    # 2. Unsafe set loss - we want h(x) ≤ -unsafe_margin in unsafe regions
    # This is CRITICAL: barrier function MUST correctly identify unsafe states
    if unsafe_mask.any():
        h_unsafe = h_values[unsafe_mask]
        unsafe_violations = F.softplus(h_unsafe + unsafe_margin, beta=5.0)
        unsafe_loss = torch.mean(unsafe_violations)

        # Additional penalty on the top 1% of unsafe values to aggressively push boundary inward
        # More robust than using max alone
        n_unsafe = h_unsafe.shape[0]
        top_k = max(1, int(0.01 * n_unsafe))  # Top 1% (at least 1 point)
        h_unsafe_topk, _ = torch.topk(h_unsafe, k=top_k)
        h_unsafe_max = F.softplus(h_unsafe_topk + unsafe_margin, beta=5.0)
        h_unsafe_max_loss = torch.mean(h_unsafe_max)  # Mean of top 1%
    else:
        unsafe_loss = torch.tensor(0.0, device=device)
        h_unsafe_max = torch.tensor(0.0, device=device)
        h_unsafe_max_loss = torch.tensor(0.0, device=device)

    # 3. CBF condition - sup_u[∇h·(f + gu)] + α(h) ≥ 0 where h(x) ≥ 0
    # For affine control: ∇h·f + sup_u[∇h·g·u] + α(h) ≥ 0
    # Ensures forward invariance (safe set remains safe under dynamics with optimal control)
    # Lower weight than boundary constraints since boundary must be learned first
    cbf_loss = torch.tensor(0.0, device=device)
    cbf_activation = torch.zeros_like(h_values)
    if lambda_cbf > 0:
        # Compute gradient of barrier function w.r.t. input
        grad_h = torch.autograd.grad(outputs=h_values, inputs=x_batch, grad_outputs=torch.ones_like(h_values), create_graph=True, retain_graph=True)[
            0
        ]  # Shape: [batch_size, state_dim]

        # Compute drift dynamics f(x)
        f_x = dynamics_model.compute_f(x_batch, translator=translator)

        # Compute drift Lie derivative: L_f(h) = ∇h · f
        lie_derivative_f = torch.sum(grad_h * f_x, dim=-1)  # Shape: [batch_size]

        # Compute control term: sup_u [∇h·g(x)·u] for affine control systems
        control_term = torch.zeros_like(h_values)
        if dynamics_model.control_dim > 0:
            # Compute control matrix g(x)
            g_x = dynamics_model.compute_g(x_batch, translator)  # Shape: [control_dim, state_dim] or [batch_size, control_dim, state_dim]

            # For each control dimension, compute sup_u_j [∇h·g_j·u_j]
            # where g_j is the j-th column of g(x)
            grad_h_g = (grad_h.unsqueeze(-2) * g_x).sum(dim=-1)  # Shape: [batch_size, control_dim]
            control_term = torch.max(
                grad_h_g * torch.tensor(dynamics_model.u_max, device=device, dtype=x_batch.dtype),
                grad_h_g * torch.tensor(dynamics_model.u_min, device=device, dtype=x_batch.dtype),
            ).sum(
                dim=-1
            )  # Shape: [batch_size]

        # Alpha function: α(h) = alpha * h (simple class K function)
        alpha_h = alpha * h_values

        # CBF condition: L_f(h) + sup_u[L_g(h)·u] + α·h >= cbf_margin
        cbf_condition = lie_derivative_f + control_term + alpha_h

        # Enforce where h(x) >= 0 (in the safe region defined by barrier)
        # This is the core CBF requirement: forward invariance of {x | h(x) >= 0}
        h_positive_mask = h_values >= -unsafe_margin

        if h_positive_mask.any():
            # Penalty when CBF condition is violated (L_f + sup_u[L_g·u] + α·h < cbf_margin)
            # This violation means the safe set is NOT forward invariant
            cbf_condition_violation = F.softplus(cbf_margin - cbf_condition[h_positive_mask], beta=5.0)
            # We want cbf_condition >= 0, so penalize when it's negative
            cbf_loss = torch.mean(cbf_condition_violation)

    # 4. Boundary Value Loss - Encourage small h(x) at boundary points
    boundary_loss = torch.tensor(0.0, device=device)
    if lambda_bndry > 0:
        signed_distance_fnc = dynamics_model.safe_set.constraint(x_batch[safe_mask], translator=TorchTranslator(device=x_batch.device))
        top_k = max(1, int(0.01 * signed_distance_fnc.shape[0]))  # Top 1% (at least 1 point)
        _, boundary_mask = torch.topk(signed_distance_fnc.abs(), k=top_k, largest=False)
        boundary_value = F.softplus(h_values[boundary_mask] + unsafe_margin, beta=100.0)
        boundary_loss = torch.mean(boundary_value)

    # Total loss
    total_loss = (
        lambda_safe * safe_loss + lambda_unsafe * unsafe_loss + lambda_unsafe_max * h_unsafe_max_loss + lambda_cbf * cbf_loss + lambda_bndry * boundary_loss
    )

    return total_loss, {
        "safe_loss": safe_loss.item(),
        "unsafe_loss": unsafe_loss.item(),
        "h_unsafe_max_loss": h_unsafe_max_loss.item(),
        "cbf_loss": cbf_loss.item(),
        "boundary_loss": boundary_loss.item(),
        "total_loss": total_loss.item(),
    }


def train_cbf(
    dynamics_model,
    learning_rate=1e-3,
    num_epochs=10000,
    batch_size=32768,
    data_regen_freq=50,
    proportions={"safe": 0.01, "unsafe": 0.01, "boundary": 0.1},
    alpha=1.0,
    lambda_safe=1e-2,
    lambda_unsafe=10.0,
    lambda_unsafe_max=1.0,
    lambda_cbf=100.00,
    lambda_bndry=0.0,
    unsafe_margin=0.01,
    safe_margin=0.01,
    cbf_margin=0.5,
    weight_decay=1e-5,
    min_epochs=1000,
    curriculum_learning=True,
    curriculum_min_epochs=1000,
    save_path_torch=None,
    save_path_onnx=None,
    use_amp=True,
    validate_during_training_freq=0,
    validate_num_samples=2000,
):
    """
    Train a neural barrier function using curriculum learning.

    Phase 1 (Classification): Learn to separate safe vs unsafe regions with margins
    - h(x) > safe_margin for x in safe set
    - h(x) < -unsafe_margin for x in unsafe set

    Phase 2 (CBF Condition): Add the CBF condition constraint
    - ∇h·f + sup_u[∇h·g·u] + α(h) >= 0 in safe regions

    Args:
        dynamics_model: CBF dynamical system
        learning_rate: Initial learning rate for optimizer
        num_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        alpha: Class K function parameter for CBF condition
        lambda_safe: Weight for safe set constraint loss (CRITICAL for boundary learning, default=1.0)
        lambda_unsafe: Weight for unsafe set constraint loss (CRITICAL for boundary learning, default=10.0)
        lambda_cbf: Weight for CBF condition loss (adds forward invariance in phase 2, default=1.0)
        lambda_unsafe_max: Weight for maximum unsafe h(x) loss (aggressive inward push, default=5.0)
        lambda_bndry: Weight for boundary value loss (encourages h(x) ~ 0 on boundary, default=0.1)
        unsafe_margin: Target margin for h(x) in unsafe region (we want h < -unsafe_margin)
        safe_margin: Target margin for h(x) in safe region (we want h > safe_margin)
        weight_decay: L2 regularization weight (default 1e-4)
        save_path_torch: Path to save PyTorch model
        save_path_onnx: Path to save ONNX model
        use_amp: Whether to use automatic mixed precision training (faster on modern GPUs)
        min_epochs: Minimum number of epochs to train before allowing early stopping
        curriculum_learning: Whether to use curriculum learning (phase 1 then phase 2)
        curriculum_min_epochs: Minimum epochs in phase 1 before switching
        data_regen_freq: How often to regenerate training data (in epochs, default=10)

    Returns:
        Trained barrier function network
    """

    # Set manual seed for reproducibility
    # torch.manual_seed(42)
    torch.manual_seed(100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model architecture info
    input_size = dynamics_model.input_dim
    hidden_sizes = dynamics_model.hidden_sizes
    activation_fnc = dynamics_model.activation_fnc

    # Create GradScaler for mixed precision training
    scaler = torch.amp.GradScaler(device=device.type) if use_amp and device.type == "cuda" else None

    # Create barrier function network
    barrier_net = BarrierNN(input_size, hidden_sizes, device=device, activation_fnc=activation_fnc)

    # Optimizer with L2 regularization (weight_decay)
    optimizer = optim.AdamW(barrier_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate schedulers for two-phase curriculum learning
    # We'll manually switch between them during phase transition
    # Phase 1: Classification (safe/unsafe) - use CosineAnnealingLR for smooth decay
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=curriculum_min_epochs,  # Cosine period matches Phase 1 duration
        eta_min=learning_rate * 0.01,  # Min LR is 1% of initial
    )

    # Phase 2: CBF condition - use ReduceLROnPlateau for adaptive learning
    scheduler_phase2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.9,  # Less aggressive reduction for Phase 2
        patience=150,  # More patience since CBF loss is harder to optimize
        threshold=1e-5,  # Stricter threshold for Phase 2
        min_lr=learning_rate * 1e-9,  # Lower minimum LR for fine-tuning
    )

    # Start with Phase 1 scheduler
    scheduler = scheduler_phase1

    # Translator
    translator = TorchTranslator(device=device)

    # Training loop
    best_loss = float("inf")
    best_model_state = None

    # Curriculum learning state
    current_lambda_cbf = 0.0  # Start with phase 1 (no CBF loss)
    phase = 1

    # Gradual CBF weight transition settings
    cbf_transition_epochs = 1000  # Number of epochs to ramp up CBF weight
    cbf_transition_start = curriculum_min_epochs  # When to start ramping up
    cbf_transition_end = cbf_transition_start + cbf_transition_epochs

    if not curriculum_learning and lambda_cbf > 0:
        current_lambda_cbf = lambda_cbf

    # Initialize loss weights (will update 'cbf' dynamically)
    loss_weights = {
        "safe": lambda_safe,
        "unsafe": lambda_unsafe,
        "unsafe_max": lambda_unsafe_max,
        "cbf": current_lambda_cbf,
        "bndry": lambda_bndry,
    }

    # Use tqdm for progress tracking
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    # Cache for training data - regenerate every data_regen_freq epochs
    x_batch_cache = None
    cex = None  # Counterexamples from validation to include in training

    ######################################
    #           Start Training           #
    ######################################

    for epoch in pbar:
        # Generate or reuse training batch
        # Regenerate data every data_regen_freq epochs for better coverage
        if x_batch_cache is None or epoch % data_regen_freq == 0:
            x_batch_cache = generate_samples(
                batch_size=batch_size,
                dynamics_model=dynamics_model,
                device=device,
                translator=translator,
                proportions=proportions,
                barrier_net=barrier_net,
            )
            if cex is not None:
                if cex.shape[1] == x_batch.shape[1]:
                    x_batch_cache = torch.cat([x_batch, cex], dim=0)

        x_batch = x_batch_cache

        # Forward pass and loss computation with optional mixed precision
        optimizer.zero_grad()

        autocast_ctx = torch.amp.autocast(device_type=device.type) if scaler is not None else nullcontext()
        with autocast_ctx:
            total_loss, loss_components = compute_cbf_loss(
                barrier_net,
                dynamics_model,
                x_batch,
                translator,
                alpha=alpha,
                lambda_safe=loss_weights["safe"],
                lambda_unsafe=loss_weights["unsafe"],
                lambda_unsafe_max=loss_weights["unsafe_max"],
                lambda_cbf=loss_weights["cbf"],
                lambda_bndry=loss_weights["bndry"],
                unsafe_margin=unsafe_margin,
                safe_margin=safe_margin,
                cbf_margin=cbf_margin,
            )

        if scaler is not None:
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()

            # Gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(barrier_net.parameters(), max_norm=1.0)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Backward pass
            total_loss.backward()

            # Gradient clipping and norm tracking
            grad_norm = torch.nn.utils.clip_grad_norm_(barrier_net.parameters(), max_norm=1.0)

            optimizer.step()

        # Save current loss for scheduler (before it gets overwritten)
        current_loss = total_loss.item()

        ##################################
        #           Scheduling           #
        ##################################

        # Gradually increase CBF loss weight after curriculum_min_epochs
        if curriculum_learning:
            if epoch < cbf_transition_start:
                loss_weights["cbf"] = 0.0
            elif cbf_transition_start <= epoch < cbf_transition_end:
                # Linear ramp-up
                progress = (epoch - cbf_transition_start) / (cbf_transition_end - cbf_transition_start)
                loss_weights["cbf"] = lambda_cbf * progress
            else:
                loss_weights["cbf"] = lambda_cbf

        # Step the learning rate scheduler
        # Phase 1 uses CosineAnnealingLR (no args), Phase 2 uses ReduceLROnPlateau (needs loss)
        if phase == 1:
            scheduler.step()
        else:
            scheduler.step(current_loss)

        # Switch to Phase 2 scheduler and learning rate at start of ramp-up
        if curriculum_learning and phase == 1 and epoch == cbf_transition_start:
            phase = 2
            # Reset best_loss for Phase 2 since the loss will be higher now (includes CBF term)
            best_loss = float("inf")
            best_model_state = None
            # Boost learning rate for Phase 2
            phase2_lr = learning_rate * 0.1  # Start Phase 2 with 10% of initial LR
            for param_group in optimizer.param_groups:
                param_group["lr"] = phase2_lr
            # Switch to Phase 2 scheduler
            scheduler = scheduler_phase2
            # Reset optimizer momentum for fresh start in phase 2
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p)
                    if state is None:
                        continue
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()

        if epoch > (num_epochs * 0.9) and current_loss < best_loss:
            best_loss = current_loss
            best_model_state = copy.deepcopy(barrier_net.state_dict())

        ###############################
        #           Logging           #
        ###############################

        if validate_during_training_freq and epoch > 0 and epoch % validate_during_training_freq == 0:
            _, cex = empirical_cbf_validation(
                barrier_net,
                dynamics_model,
                num_samples=validate_num_samples,
                alpha=alpha,
            )

            if cex is not None:
                if cex.shape[1] == x_batch.shape[1]:
                    x_batch_cache = torch.cat([x_batch_cache, cex], dim=0)

        # Update progress bar with current metrics
        postfix_dict = {
            "phase": f"{phase}",
            "loss": f"{current_loss:.6f}",
            "best": f"{best_loss:.6f}",
            "safe": f'{loss_components["safe_loss"]:.6f}',
            "unsafe": f'{loss_components["unsafe_loss"]:.6f}',
            "unsafe_max": f'{loss_components["h_unsafe_max_loss"]:.6f}',
            "boundary": f'{loss_components["boundary_loss"]:.6f}',
            "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
        }
        if phase == 2:
            postfix_dict["cbf"] = f'{loss_components["cbf_loss"]:.6f}'
        pbar.set_postfix(postfix_dict)

    #########################################
    #           Finished Training           #
    #########################################

    # Close progress bar
    pbar.close()

    # Load best model
    if best_model_state is not None:
        barrier_net.load_state_dict(best_model_state)

    # Save models
    if save_path_torch is None:
        save_path_torch = f"data/models/relu_models/{dynamics_model.system_name}_cbf.pth"
    if save_path_onnx is None:
        save_path_onnx = f"data/models/relu_models/{dynamics_model.system_name}_cbf.onnx"

    torch.save(barrier_net.state_dict(), save_path_torch)
    save_onnx_model(barrier_net, save_path_onnx, input_size)

    return barrier_net
