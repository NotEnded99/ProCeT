"""Strategy interface for repair methods.

The three published methods — CeT, α-ProCeT, β-ProCeT — share 95 % of their
control flow (load model → iterate → verify → log). They differ only in:

    * which failed regions they push the repair loss on (with or without
      LBP-aware pruning),
    * whether they select a Top-N vulnerable set to *protect* via SOCP,
    * how they compute ``Δθ`` (plain GD vs SOCP projection),
    * whether they run the protection audit after each inner step,
    * whether they adaptively switch strategy mid-run (β-ProCeT only).

``RepairMethod`` is the template-method hook surface that captures those
differences. ``IterationContext`` is the per-iteration mutable bag the runner
threads through every hook.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config & per-iteration context
# ---------------------------------------------------------------------------

@dataclass
class MethodConfig:
    """All knobs the runner needs, set by argparse in ``scripts/run_*.py``."""

    system: str
    activation: str
    num_inner_steps: int = 5
    lr: float = 5e-3
    target_pass_rate: float = 100.0
    patience: int = 3
    max_total_iterations: int = 10
    lambda_: float = 1.0
    top_n_protect: int = 100
    delta_theta_norm_bound: float = 0.01
    seed: int = 2026
    # Filled in by the runner before the loop starts.
    max_depth_limit: int = 12


@dataclass
class IterationContext:
    """Per-iteration mutable state threaded through every method hook.

    The runner fills in fields incrementally:
      1. ``failed_safe`` / ``failed_unsafe`` ← ``method.select_targets``
      2. ``safe_weights`` / ``unsafe_weights`` ← runner (category weighting)
      3. ``top_n_v_*`` ← runner if ``method.needs_top_n()``
      4. ``J_safe`` / ``J_unsafe`` ← ``method.prepare_inner_loop``
    Methods read whatever they need from ``ctx`` in ``compute_update``.
    """

    cfg: MethodConfig
    model: Any
    dynamics_model: Any
    lbp_linearizer: Any
    device: torch.device
    dtype: torch.dtype
    iteration: int

    failed_safe: list = field(default_factory=list)
    failed_unsafe: list = field(default_factory=list)
    safe_weights: list = field(default_factory=list)
    unsafe_weights: list = field(default_factory=list)
    repair_type: str = "N/A"

    top_n_v_safe: list = field(default_factory=list)
    top_n_v_unsafe: list = field(default_factory=list)
    top_n_v_safe_margins: Optional[np.ndarray] = None
    top_n_v_unsafe_h_ub: Optional[np.ndarray] = None
    protected_safe_snapshot: Optional[list] = None
    protected_unsafe_snapshot: Optional[list] = None

    # Jacobian matrices populated by ``prepare_inner_loop`` (ProCeT family)
    J_safe: Optional[torch.Tensor] = None
    J_unsafe: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

class RepairMethod(ABC):
    """Template-method strategy for one repair variant.

    Subclasses override the hooks that differ; everything else (loading,
    verification, patience, result serialisation) lives in ``procet.runner``.
    """

    # ----- Class-level metadata (override in subclass) -----
    name: str = ""                  # CLI identifier, e.g. 'cet'
    display_name: str = ""          # for prints & result JSON, e.g. 'CeT'
    supported_activations: tuple = ()   # e.g. ('Tanh', 'Sigmoid')
    output_dir_suffix: str = ""     # results subdir + file stem, e.g. 'CeT'

    def __init__(self, cfg: MethodConfig):
        self.cfg = cfg

    # ----- Iteration hooks -----

    @abstractmethod
    def select_targets(self, regions: dict, ctx: IterationContext) -> tuple:
        """Choose failed simplices to push the repair loss on this iteration.

        Returns ``(failed_safe, failed_unsafe, repair_type)``.
        """

    @abstractmethod
    def needs_top_n(self) -> bool:
        """Whether the runner should select Top-N vulnerable regions.

        Returns ``True`` for methods that either protect via SOCP or audit.
        """

    @abstractmethod
    def audits(self) -> bool:
        """Whether to run ``check_protection_status`` after each inner step."""

    @abstractmethod
    def get_lr(self, max_depth_limit: int) -> float:
        """Per-method learning rate (depth-aware)."""

    @abstractmethod
    def prepare_inner_loop(self, ctx: IterationContext) -> None:
        """Optional pre-computation before the inner loop (e.g. Jacobians).

        Mutates ``ctx`` (typically ``ctx.J_safe`` / ``ctx.J_unsafe``).
        """

    @abstractmethod
    def compute_update(self, ctx: IterationContext, g_F_clipped: torch.Tensor,
                       current_lr: float) -> tuple:
        """Compute and apply ``Δθ`` to ``ctx.model`` in-place.

        Returns ``(g_raw_norm, update_norm)``.
        """

    # ----- Optional hooks with defaults -----

    def get_beta(self) -> tuple:
        """Empirical-remainder coefficients ``(beta_s, beta_us)``.

        Default ``0.999`` for both — only meaningful for protection methods.
        """
        return 0.999, 0.999

    def on_iteration_end(self, ctx: IterationContext, certified_pct: float,
                         prev_certified_pct: Optional[float],
                         patience_counter: int) -> dict:
        """Hook fired after verification + patience update.

        Returns a dict of optional actions:
            ``{'backtrack_and_redo': bool, 'reset_patience': bool}``
        Default: no-op. β-ProCeT overrides this to trigger its phase switch.
        """
        return {}

    # ----- Naming -----

    def result_file_stem(self) -> str:
        """File-stem for the result JSON (without extension).

        Default encodes system/activation/depth/lambda. Methods
        that need extra parameters in the filename (e.g. ``_k{K}_zeta{Z}``)
        should override.
        """
        c = self.cfg
        return (f"result_{c.system}_{c.activation}_{self.output_dir_suffix}"
                f"_depth{c.max_depth_limit}_w{c.lambda_}")
