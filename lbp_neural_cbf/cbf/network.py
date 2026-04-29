import torch
import torch.nn as nn
import torch.nn.init as init

from ..translators import TorchTranslator


def empirical_cbf_validation(barrier_net, dynamics_model, num_samples=50000, alpha=1.0):
    """
    Fast empirical validation of CBF conditions by sampling many points.

    This is much faster than formal verification and suitable for hyperparameter tuning.
    We check:
    1. Classification violations: h(x) >= 0 in UNSAFE set (barrier incorrectly classifies unsafe as safe)
    2. Set invariance violations: h(x) >= 0 but CBF condition dh/dt + alpha*h < 0 fails
    3. Control invariant set size: # of points in SAFE set where h(x) >= 0 (GOOD, maximize this!)

    Args:
        barrier_net: Trained barrier function network
        dynamics_model: CBF dynamical system
        num_samples: Number of points to sample for validation
        alpha: Class K function parameter

    Returns:
        Dictionary with validation metrics
    """
    device = next(barrier_net.parameters()).device
    translator = TorchTranslator(device=device)

    # Sample points uniformly from domain via the domain API (supports BoxDomain)
    samples = dynamics_model.input_domain.sample_points(num_samples, device=device, use_torch=True)  # [num_samples, input_dim]
    cex = None
    with torch.no_grad():
        # Evaluate barrier function
        h_values = barrier_net(samples).squeeze()  # [num_samples]

        # Determine which points are actually safe vs unsafe
        safe_mask = dynamics_model.safe_set.contains(samples, translator)
        unsafe_mask = ~safe_mask

        # unsafe_set_violations: h(x) >= 0 in UNSAFE set
        # (barrier says it's safe, but it's actually unsafe - this is BAD)
        unsafe_set_violations = 0
        if unsafe_mask.any():
            h_unsafe = h_values[unsafe_mask]
            unsafe_set_violations_mask = h_unsafe >= 0
            unsafe_set_violations = unsafe_set_violations_mask.sum().item()

        # Control invariant set size: h(x) >= 0 in SAFE set
        # (this is GOOD - we want to maximize this!)
        invariant_set_size = 0
        if safe_mask.any():
            h_safe = h_values[safe_mask]
            invariant_set_size = (h_safe >= 0).sum().item()

    # Check CBF condition on a subset to keep this fast:
    # sup_u [ ∇h · (f + g u) ] + α(h) >= 0  for points with h(x) >= 0
    set_invariance_violations = 0
    cbf_condition = None

    # choose subset of samples where barrier >= 0
    samples_for_grad = samples.detach()
    h_values = barrier_net(samples_for_grad).squeeze()
    positive_mask = h_values >= 0

    if positive_mask.any():
        # select indices (cap to keep cost bounded)
        idx = torch.nonzero(positive_mask, as_tuple=False).squeeze(-1)

        # xs will require grad for autograd
        xs = samples_for_grad[idx].detach().clone().requires_grad_(True)  # [k, state_dim]
        h_subset = barrier_net(xs).squeeze()  # [k]

        # gradient of h wrt x for subset (vectorized)
        grad_h = torch.autograd.grad(
            outputs=h_subset, inputs=xs, grad_outputs=torch.ones_like(h_subset, device=device), create_graph=True, retain_graph=True, allow_unused=False
        )[
            0
        ]  # [k, state_dim]

        # Drift term L_f h = ∇h · f(x)
        f_x = dynamics_model.compute_f(xs, translator=translator)  # [k, state_dim]
        lie_derivative_f = torch.sum(grad_h * f_x, dim=-1)  # [k]

        # Control term sup_u ∇h·g·u
        control_term = torch.zeros_like(lie_derivative_f)
        if getattr(dynamics_model, "control_dim", 0) and dynamics_model.control_dim > 0:
            # g_x shape expected: [k, control_dim, state_dim]
            g_x = dynamics_model.compute_g(xs, translator=translator)
            grad_h_g = (grad_h.unsqueeze(-2) * g_x).sum(dim=-1)

            # bounds for each control dim (make tensors same device & dtype)
            u_max = torch.as_tensor(dynamics_model.u_max, device=device, dtype=xs.dtype)
            u_min = torch.as_tensor(dynamics_model.u_min, device=device, dtype=xs.dtype)
            # elementwise maximize over allowed u for each control coordinate:
            # sup_u_j grad_h_g_j * u_j = max( grad_h_g_j * u_min_j, grad_h_g_j * u_max_j )
            # then sum across control dimensions
            prod_max = grad_h_g * u_max  # [k, control_dim]
            prod_min = grad_h_g * u_min  # [k, control_dim]
            control_term = torch.max(prod_min, prod_max).sum(dim=-1)  # [k]

        # alpha(h) term (use h_subset here since we evaluated barrier on xs)
        alpha_h = alpha * h_subset  # [k]

        # full cbf condition on the subset
        cbf_condition_subset = lie_derivative_f + control_term + alpha_h  # [k]

        # count violations (cbf_condition < 0 means violation)
        set_invariance_violations_mask = cbf_condition_subset < 0
        set_invariance_violations = set_invariance_violations_mask.sum().item()
    else:
        set_invariance_violations = 0

    # Calculate metrics - safe set violation + invariance violation
    total_violations = unsafe_set_violations + set_invariance_violations
    violation_rate = (total_violations / num_samples) * 100.0

    # Percentage of unsafe points correctly classified (h < 0 in unsafe)
    unsafe_classification_rate = 100.0 - (unsafe_set_violations / max(unsafe_mask.sum().item(), 1)) * 100.0

    # Percentage of invariant set that satisfies CBF condition
    if safe_mask.any():
        set_invariance_satisfaction_rate = ((safe_mask.sum().item() - set_invariance_violations) / safe_mask.sum().item()) * 100.0
        invariant_set_coverage = (invariant_set_size / safe_mask.sum().item()) * 100.0
    else:
        set_invariance_satisfaction_rate = 100.0
        invariant_set_coverage = 0.0

    # Composite validity score that balances:
    # 1. Zero violations (most important)
    # 2. Large control invariant set (maximize coverage of safe set)
    validity_score = float(100 * torch.exp(-torch.tensor(2 * violation_rate)) + 0.05 * invariant_set_coverage)

    cex = None
    if total_violations > 0:
        unsafe_cex = samples[unsafe_mask][unsafe_set_violations_mask] if unsafe_mask.any() and "unsafe_set_violations_mask" in locals() else None
        set_inv_cex = samples[positive_mask][set_invariance_violations_mask] if positive_mask.any() and "set_invariance_violations_mask" in locals() else None
        cex = torch.cat([unsafe_cex, set_inv_cex], dim=0) if unsafe_cex is not None and set_inv_cex is not None else None

    result = {
        "total_violations": total_violations,
        "unsafe_set_violations": unsafe_set_violations,
        "set_invariance_violations": set_invariance_violations,
        "violation_rate": violation_rate,
        "unsafe_classification_rate": unsafe_classification_rate,
        "set_invariance_satisfaction_rate": set_invariance_satisfaction_rate,
        "invariant_set_size": invariant_set_size,
        "invariant_set_coverage": invariant_set_coverage,
        "num_samples_checked": num_samples,
        "num_safe_samples": safe_mask.sum().item(),
        "num_unsafe_samples": unsafe_mask.sum().item(),
        "validity_score": validity_score,
    }

    return result, cex


class BarrierNN(nn.Module):
    """Neural network for control barrier functions with proper weight initialization."""

    def __init__(self, input_size, hidden_sizes, device=None, activation_fnc="Tanh", seed=None):
        """
        Args:
            input_size (int)
            hidden_sizes (list[int])
            device (torch.device or None)
            activation_fnc (str): "Tanh", "Relu", "Sigmoid", or "LeakyRelu"
            seed (int or None): if given, sets torch.manual_seed for reproducibility
        """
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        if activation_fnc == "Tanh":
            activation_layer = nn.Tanh()
            init_type = "xavier_tanh"
        elif activation_fnc == "Relu":
            activation_layer = nn.ReLU()
            init_type = "kaiming_relu"
        elif activation_fnc == "Sigmoid":
            activation_layer = nn.Sigmoid()
            init_type = "xavier_tanh"
        elif activation_fnc == "LeakyRelu":
            activation_layer = nn.LeakyReLU(negative_slope=0.01)
            init_type = "kaiming_relu"

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_layer)
            prev_size = hidden_size
        # final output (scalar)
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

        # Move parameters to the device in one go
        self.to(self.device)

        # Initialize weights
        self._init_weights(init_type=init_type)

    def forward(self, x):
        # ensure x is on the network device
        if x.device != self.device:
            x = x.to(self.device)
        return self.network(x)

    def _init_weights(self, init_type="xavier_tanh"):
        """
        init_type:
          - "xavier_tanh": Xavier uniform with gain for tanh (recommended for Tanh activations)
          - "orthogonal": orthogonal init (works well in many cases)
          - "kaiming_relu": kaiming normal recommended if you switch to ReLU
        """
        # compute gain for tanh if using xavier_tanh
        gain_tanh = init.calculate_gain("tanh")

        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                # Detect final linear layer by checking out_features == 1
                is_final = module.out_features == 1

                if init_type == "xavier_tanh":
                    # Use Xavier suitable for Tanh
                    init.xavier_uniform_(module.weight, gain=gain_tanh)
                elif init_type == "orthogonal":
                    init.orthogonal_(module.weight, gain=1.0)
                elif init_type == "kaiming_relu":
                    # only use if using ReLU activations
                    init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    # fallback
                    init.xavier_uniform_(module.weight, gain=gain_tanh)
