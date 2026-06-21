"""Shared outer repair loop (template method).

All three methods (CeT, α-ProCeT, β-ProCeT) are driven by ``run_repair``.
Method-specific behaviour lives in the ``RepairMethod`` instance; this
function handles everything else: reproducibility, loading, the iteration
loop, verification, patience, the optional CeT→ProCeT backtrack-redo, and
result serialisation.

Typical use (see ``scripts/run_*.py``)::

    from procet import build_method, run_repair
    from procet.methods.base import MethodConfig

    cfg = MethodConfig(system='barr1', activation='Tanh', num_inner_steps=5, ...)
    method = build_method('alpha-procet', cfg)
    run_repair(method, cfg)
"""

import json
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch

from lbp_neural_cbf.cbf.network import BarrierNN
from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization

from .core.systems import DYNAMICS_SYSTEMS, SYSTEM_DEPTH
from .core.io import pytorch_to_onnx, verify_model
from .core.metrics import NumpyJSONEncoder, compute_safety_metrics_v8
from .core.lbp_loss import compute_repair_loss_and_grad_lbp_weighted
from .core.selection import select_top_n_v_safe_lbp, select_top_n_v_unsafe_lbp
from .core.audit import check_protection_status
from .methods.base import IterationContext, MethodConfig, RepairMethod


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    """Filesystem locations used by the runner.

    Defaults match the original layout so existing artefacts (trained barrier
    networks, pre-verified regions) are reused without copying. Output
    directories are created on demand.
    """

    # Input: trained barrier networks
    model_dir_template: str = "data/models/{activation_lower}_models"
    # Input: pre-verified simplicial regions
    regions_dir: str = "data/regions"
    regions_template: str = (
        "data/regions/verified_regions_{system}_{activation}_v1_depth{depth}.pt"
    )
    # Output: repaired models + result JSONs
    output_model_dir: str = "results/models"
    results_dir_template: str = "results/{method_suffix}"


def _default_paths() -> Paths:
    return Paths()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_dynamics(system_key: str, activation: str):
    dynamics_class = DYNAMICS_SYSTEMS[system_key]
    dynamics_model = dynamics_class(alpha=1.0)
    dynamics_model.activation_fnc = activation
    return dynamics_model


def _load_model(dynamics_model, activation: str, device: torch.device, paths: Paths):
    model_dir = paths.model_dir_template.format(activation_lower=activation.lower())
    model_path = f"{model_dir}/{dynamics_model.system_name}_cbf.pth"
    model = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device,
        activation_fnc=activation,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model


def _load_regions(dynamics_model, activation: str, max_depth_limit: int, paths: Paths) -> dict:
    regions_path = paths.regions_template.format(
        system=dynamics_model.system_name,
        activation=activation,
        depth=max_depth_limit,
    )
    data = torch.load(regions_path, map_location="cpu", weights_only=False)
    return {
        "V_safe":                     data["V_safe"],
        "V_unsafe":                   data["V_unsafe"],
        "F_h_positive_in_unsafe":     data["F_h_positive_in_unsafe"],
        "F_safe_cbf_violation":       data["F_safe_cbf_violation"],
        # Legacy files stored both under 'F_depth_limit_reached'.
        "F_depth_limit_reached_unsafe": data.get(
            "F_depth_limit_reached_unsafe", data.get("F_depth_limit_reached", [])),
        "F_depth_limit_reached_safe":   data.get("F_depth_limit_reached_safe", []),
        "F_unsafe_cannot_split":        data["F_unsafe_cannot_split"],
    }


def _extract_regions(verify_results: dict, fallback: dict) -> dict:
    """Pull the seven region buckets out of a verify_cbf result dict,
    falling back to the previous iteration's buckets when verify_cbf omits one.
    """
    keys = [
        "V_safe", "V_unsafe",
        "F_h_positive_in_unsafe", "F_safe_cbf_violation",
        "F_depth_limit_reached_unsafe", "F_depth_limit_reached_safe",
        "F_unsafe_cannot_split",
    ]
    return {k: verify_results.get(k, fallback[k]) for k in keys}


def _build_category_weights(failed_safe, failed_unsafe, regions, lambda_):
    """Per-sample weights: definitive violations get ``lambda_``,
    uncertain regions get 1.

    Definitive-ness is decided by membership in the pre-verification buckets:
      - safe side:   F_safe_cbf_violation
      - unsafe side: F_h_positive_in_unsafe
    """

    def simplex_to_bytes(s):
        arr = s.cpu().numpy() if hasattr(s, "cpu") else s
        return arr.tobytes()

    safe_definitive_keys = {simplex_to_bytes(s) for s in regions["F_safe_cbf_violation"]}
    safe_weights = [
        lambda_ if simplex_to_bytes(s) in safe_definitive_keys else 1.0
        for s in failed_safe
    ]

    unsafe_definitive_keys = {simplex_to_bytes(s) for s in regions["F_h_positive_in_unsafe"]}
    unsafe_weights = [
        lambda_ if simplex_to_bytes(s) in unsafe_definitive_keys else 1.0
        for s in failed_unsafe
    ]
    return safe_weights, unsafe_weights


def _adaptive_min_delta(certified_pct: float) -> float:
    """Tighter improvement threshold once we are close to 100 %."""
    if certified_pct < 90:
        return 0.5
    if certified_pct < 98:
        return 0.1
    return 0.05


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------

def _run_inner_loop(method: RepairMethod, ctx: IterationContext, current_lr: float):
    """Run ``cfg.num_inner_steps`` updates; return ``(inner_history, audit_after_steps)``.

    The audit list is empty when ``method.audits()`` is False (CeT).
    """
    cfg = ctx.cfg
    inner_history = []
    audit_after_steps = []

    for inner_step in range(cfg.num_inner_steps):
        t0 = time.perf_counter()

        loss_val, g_F = compute_repair_loss_and_grad_lbp_weighted(
            model=ctx.model,
            dynamics_model=ctx.dynamics_model,
            safe_simplices=ctx.failed_safe,
            unsafe_simplices=ctx.failed_unsafe,
            lbp_linearizer=ctx.lbp_linearizer,
            margin=0.1,
            cbf_margin=0.0,
            beta=5.0,
            grad_clip_norm=10.0,
            verbose=False,
            safe_weights=ctx.safe_weights,
            unsafe_weights=ctx.unsafe_weights,
        )

        grad_norm = g_F.norm().item()
        g_F_clipped = g_F * (10.0 / grad_norm) if grad_norm > 10.0 else g_F.clone()
        t1 = time.perf_counter()

        g_raw_norm, update_norm = method.compute_update(ctx, g_F_clipped, current_lr)
        t2 = time.perf_counter()

        # Per-method mode label (CeT vs ProCeT for β-ProCeT; otherwise just method name).
        mode_label = method.current_mode_label() if hasattr(method, "current_mode_label") else method.display_name

        print(f"    Inner step {inner_step + 1}/{cfg.num_inner_steps} [{mode_label}]: "
              f"loss={loss_val:.6f}, |g|={g_raw_norm:.4f}, |d|={update_norm:.6f}, "
              f"t_loss={t1 - t0:.2f}s, t_qp={t2 - t1:.2f}s")
        inner_history.append({
            "step": inner_step + 1, "mode": mode_label,
            "loss": loss_val, "g_raw_norm": g_raw_norm,
            "update_norm": update_norm,
            "t_loss": t1 - t0, "t_qp": t2 - t1,
        })

        if method.audits():
            audit_after = check_protection_status(
                model=ctx.model,
                top_n_v_safe=ctx.protected_safe_snapshot,
                top_n_v_unsafe=ctx.protected_unsafe_snapshot,
                dynamics_model=ctx.dynamics_model,
                lbp_linearizer=ctx.lbp_linearizer,
                device=ctx.device, dtype=ctx.dtype,
            )
            print(f"    [Audit AFTER step {inner_step + 1}] "
                  f"safe: {audit_after['safe_success']}/{audit_after['safe_total']}, "
                  f"unsafe: {audit_after['unsafe_success']}/{audit_after['unsafe_total']}")
            audit_after_steps.append(_audit_to_log_dict(audit_after, inner_step + 1))

    return inner_history, audit_after_steps


def _audit_to_log_dict(audit: dict, inner_step: int) -> dict:
    return {
        "inner_step": inner_step,
        "safe": {
            "total":      audit["safe_total"],
            "success":    audit["safe_success"],
            "failure":    audit["safe_failure"],
            "min_L_min":  audit["safe_min_L_min"],
            "min_L_mean": audit["safe_min_L_mean"],
        },
        "unsafe": {
            "total":       audit["unsafe_total"],
            "success":     audit["unsafe_success"],
            "failure":     audit["unsafe_failure"],
            "h_max_max":   audit["unsafe_h_max_max"],
            "h_max_mean":  audit["unsafe_h_max_mean"],
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_repair(method: RepairMethod, cfg: MethodConfig, paths: Paths = None):
    """Drive one repair experiment end-to-end.

    Args:
        method: A ``RepairMethod`` instance (CeT / α-ProCeT / β-ProCeT).
        cfg:    All hyperparameters. ``cfg.max_depth_limit`` is auto-filled
                from ``SYSTEM_DEPTH`` if not set explicitly.
        paths:  Optional filesystem layout. Defaults preserve compatibility
                with the original ``data/`` and ``data/regions/`` inputs.
    """
    paths = paths or _default_paths()
    if not cfg.max_depth_limit:
        cfg.max_depth_limit = SYSTEM_DEPTH[cfg.system]

    # ---- Setup ----
    _setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"Neural CBF Iterative Repair — {method.display_name}")
    print(f"  activation={cfg.activation}, system={cfg.system}")
    if method.needs_top_n():
        print(f"  Top-N most vulnerable regions: {cfg.top_n_protect}")
    print(f"  num_inner_steps={cfg.num_inner_steps}, lr={cfg.lr}, "
          f"patience={cfg.patience}, max_total_iterations={cfg.max_total_iterations}")
    if hasattr(method, "use_protection"):
        print(f"  Strategy: start in CeT mode, switch to α-ProCeT on first plateau")
    print(f"  max_depth_limit={cfg.max_depth_limit}")
    print("=" * 70)

    dynamics_model = _load_dynamics(cfg.system, cfg.activation)
    model = _load_model(dynamics_model, cfg.activation, device, paths)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Number of parameters: {num_params}")

    regions = _load_regions(dynamics_model, cfg.activation, cfg.max_depth_limit, paths)
    initial_metrics = compute_safety_metrics_v8(**regions)
    original_harmonic  = initial_metrics["HarmonicMeanPassRate"] * 100
    original_standard  = initial_metrics["standard_pass_rate"]
    original_R_safe    = initial_metrics["R_safe"] * 100
    original_R_unsafe  = initial_metrics["R_unsafe"] * 100

    total_fail = (len(regions["F_h_positive_in_unsafe"])
                  + len(regions["F_safe_cbf_violation"])
                  + len(regions["F_depth_limit_reached_unsafe"])
                  + len(regions["F_depth_limit_reached_safe"])
                  + len(regions["F_unsafe_cannot_split"]))
    print(f"\n[Init] HarmonicMeanPassRate={original_harmonic:.2f}%, "
          f"R_safe={original_R_safe:.2f}%, R_unsafe={original_R_unsafe:.2f}%")
    print(f"    V_safe={len(regions['V_safe'])}, "
          f"V_unsafe={len(regions['V_unsafe'])}, total_fail={total_fail}")

    # ---- Output paths ----
    method_results_dir = paths.results_dir_template.format(method_suffix=method.output_dir_suffix)
    os.makedirs(method_results_dir, exist_ok=True)
    os.makedirs(paths.output_model_dir, exist_ok=True)

    initial_pth = os.path.join(
        paths.output_model_dir,
        f"{dynamics_model.system_name}_{cfg.activation}_cbf_repaired_{method.output_dir_suffix}.pth",
    )
    initial_onnx = initial_pth.replace(".pth", ".onnx")

    # ---- Init protection log (audit-only methods) ----
    beta_s, beta_us = method.get_beta()
    protection_log = {
        "system": cfg.system, "activation": cfg.activation,
        "method": method.display_name,
        "top_n_protect": cfg.top_n_protect,
        "beta_s": beta_s, "beta_us": beta_us,
        "delta_theta_norm_bound": cfg.delta_theta_norm_bound,
        "total_safe_checks": 0, "total_safe_success": 0, "total_safe_failure": 0,
        "total_unsafe_checks": 0, "total_unsafe_success": 0, "total_unsafe_failure": 0,
        "iterations": [],
    }

    # ---- Skip if already certified ----
    if original_standard == 100.00 and original_harmonic == 100.00:
        print(f"\n[Skip] Pass rate already 100%, no repair needed.")
        _save_result_json(
            method, cfg, method_results_dir,
            original_harmonic=original_harmonic, original_standard=original_standard,
            original_R_safe=original_R_safe, original_R_unsafe=original_R_unsafe,
            final_harmonic=original_harmonic, final_standard=original_standard,
            final_R_safe=original_R_safe, final_R_unsafe=original_R_unsafe,
            iteration_results=[], repair_time_seconds=0.0,
            protection_log=protection_log if method.audits() else None,
            skip_reason="already_100_percent",
        )
        return

    # Save initial model
    torch.save(model.state_dict(), initial_pth)
    pytorch_to_onnx(model, initial_onnx, input_dim=dynamics_model.input_dim)

    iteration_results = []
    prev_certified_pct = None
    patience_counter = 0
    last_min_delta = 0.5

    print(f"\nStarting repair loop with {method.display_name}, "
          f"max_depth={cfg.max_depth_limit}")

    repair_start = time.perf_counter()

    for iteration in range(cfg.max_total_iterations):
        # ---- Stop checks ----
        total_fail = (len(regions["F_h_positive_in_unsafe"])
                      + len(regions["F_safe_cbf_violation"])
                      + len(regions["F_depth_limit_reached_unsafe"])
                      + len(regions["F_depth_limit_reached_safe"])
                      + len(regions["F_unsafe_cannot_split"]))
        if total_fail == 0:
            print(f"\n  === Stop: ALL_CERTIFIED ===")
            break
        if patience_counter >= cfg.patience:
            print(f"\n  === Stop: PLATEAU_DETECTED (no improvement ≥ {last_min_delta:.1f}pp "
                  f"for {cfg.patience} consecutive iterations) ===")
            break

        mode_label = method.current_mode_label() if hasattr(method, "current_mode_label") else method.display_name
        print(f"\n[Iteration {iteration + 1}] max_depth={cfg.max_depth_limit}, mode={mode_label}")

        # ---- Per-iteration linearizer + context ----
        lbp_linearizer = CrownPartialLinearization(model, dtype=torch.float32)
        dtype = next(model.parameters()).dtype
        ctx = IterationContext(
            cfg=cfg, model=model, dynamics_model=dynamics_model,
            lbp_linearizer=lbp_linearizer, device=device, dtype=dtype,
            iteration=iteration,
        )

        ctx.failed_safe, ctx.failed_unsafe, ctx.repair_type = method.select_targets(regions, ctx)
        top_n_used = 0
        audit_before = None
        audit_after_steps = []
        inner_history = []

        if not ctx.failed_safe and not ctx.failed_unsafe:
            certified_pct = prev_certified_pct if prev_certified_pct is not None else original_harmonic
            print(f"    No failed simplices to repair this iteration.")
        else:
            current_lr = method.get_lr(cfg.max_depth_limit)
            ctx.safe_weights, ctx.unsafe_weights = _build_category_weights(
                ctx.failed_safe, ctx.failed_unsafe, regions, cfg.lambda_,
            )

            # ---- Top-N + audit BEFORE ----
            if method.needs_top_n():
                print(f"\n[Iteration {iteration + 1}.1] Selecting Top-N vulnerable regions "
                      f"(N={cfg.top_n_protect})...")
                ctx.top_n_v_safe, ctx.top_n_v_safe_margins = select_top_n_v_safe_lbp(
                    ctx.model, list(regions["V_safe"]),
                    ctx.dynamics_model, ctx.lbp_linearizer, cfg.top_n_protect,
                )
                ctx.top_n_v_unsafe, ctx.top_n_v_unsafe_h_ub = select_top_n_v_unsafe_lbp(
                    ctx.model, list(regions["V_unsafe"]),
                    ctx.dynamics_model, ctx.lbp_linearizer, cfg.top_n_protect,
                )
                ctx.protected_safe_snapshot = list(ctx.top_n_v_safe)
                ctx.protected_unsafe_snapshot = list(ctx.top_n_v_unsafe)
                top_n_used = len(ctx.top_n_v_safe)
                print(f"    Top-N V_safe: {len(ctx.top_n_v_safe)}, "
                      f"Top-N V_unsafe: {len(ctx.top_n_v_unsafe)}")

                if method.audits():
                    audit_before = check_protection_status(
                        model=ctx.model,
                        top_n_v_safe=ctx.protected_safe_snapshot,
                        top_n_v_unsafe=ctx.protected_unsafe_snapshot,
                        dynamics_model=ctx.dynamics_model,
                        lbp_linearizer=ctx.lbp_linearizer,
                        device=ctx.device, dtype=ctx.dtype,
                    )
                    print(f"    [Audit BEFORE inner loop] "
                          f"safe: {audit_before['safe_success']}/{audit_before['safe_total']}, "
                          f"unsafe: {audit_before['unsafe_success']}/{audit_before['unsafe_total']}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ---- Method-specific preparation (Jacobians for ProCeT family) ----
            method.prepare_inner_loop(ctx)

            # ---- Snapshot params (for potential backtrack-redo) ----
            pre_params = {n: p.clone() for n, p in model.named_parameters()}

            # ---- Inner loop ----
            print(f"\n[Iteration {iteration + 1}.3] Inner loop ({cfg.num_inner_steps} steps)...")
            inner_history, audit_after_steps = _run_inner_loop(method, ctx, current_lr)

        # ---- Save + verify + metrics ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.save(model.state_dict(), initial_pth)
        pytorch_to_onnx(model, initial_onnx, input_dim=dynamics_model.input_dim)

        verify_results = verify_model(initial_pth, dynamics_model, max_depth=cfg.max_depth_limit)
        new_regions = _extract_regions(verify_results, regions)
        new_metrics = compute_safety_metrics_v8(**new_regions)
        certified_pct = new_metrics["HarmonicMeanPassRate"] * 100
        R_safe_pct = new_metrics["R_safe"] * 100
        R_unsafe_pct = new_metrics["R_unsafe"] * 100
        print(f"\n[Iteration {iteration + 1}.4] Verification: "
              f"HarmonicMeanPassRate={certified_pct:.2f}%, "
              f"R_safe={R_safe_pct:.2f}%, R_unsafe={R_unsafe_pct:.2f}%")

        # ---- Patience update ----
        min_delta = _adaptive_min_delta(certified_pct)
        last_min_delta = min_delta
        if prev_certified_pct is None:
            patience_counter = 0
        elif certified_pct - prev_certified_pct >= min_delta:
            patience_counter = 0
        else:
            patience_counter += 1

        # ---- Method-specific post-iteration hook (β-ProCeT backtrack-redo) ----
        action = method.on_iteration_end(ctx, certified_pct, prev_certified_pct, patience_counter)
        if action.get("backtrack_and_redo") and (ctx.failed_safe or ctx.failed_unsafe):
            print(f"\n    *** {method.display_name}: BACKTRACKING + REDO with new phase ***")
            # Restore params
            for n, p in model.named_parameters():
                p.data = pre_params[n].clone()
            if action.get("reset_patience"):
                patience_counter = 0

            # Re-create linearizer (model state changed; cached bounds are stale)
            ctx.lbp_linearizer = CrownPartialLinearization(model, dtype=torch.float32)
            method.prepare_inner_loop(ctx)  # now computes Jacobians

            print(f"\n[Iteration {iteration + 1}.3-REDO] Inner loop ({cfg.num_inner_steps} steps)...")
            current_lr = method.get_lr(cfg.max_depth_limit)
            inner_history, audit_after_steps = _run_inner_loop(method, ctx, current_lr)

            # Re-verify
            torch.save(model.state_dict(), initial_pth)
            pytorch_to_onnx(model, initial_onnx, input_dim=dynamics_model.input_dim)
            verify_results = verify_model(initial_pth, dynamics_model, max_depth=cfg.max_depth_limit)
            new_regions = _extract_regions(verify_results, regions)
            new_metrics = compute_safety_metrics_v8(**new_regions)
            certified_pct = new_metrics["HarmonicMeanPassRate"] * 100
            R_safe_pct = new_metrics["R_safe"] * 100
            R_unsafe_pct = new_metrics["R_unsafe"] * 100
            prev_certified_pct = certified_pct
            print(f"\n[Iteration {iteration + 1}.4-REDO] Verification: "
                  f"HarmonicMeanPassRate={certified_pct:.2f}%, "
                  f"R_safe={R_safe_pct:.2f}%, R_unsafe={R_unsafe_pct:.2f}%")
        else:
            prev_certified_pct = certified_pct

        # ---- Accumulate audit log for this iteration ----
        if method.audits() and audit_before is not None:
            for a in audit_after_steps:
                protection_log["total_safe_checks"]   += a["safe"]["total"]
                protection_log["total_safe_success"]  += a["safe"]["success"]
                protection_log["total_safe_failure"]  += a["safe"]["failure"]
                protection_log["total_unsafe_checks"] += a["unsafe"]["total"]
                protection_log["total_unsafe_success"]+= a["unsafe"]["success"]
                protection_log["total_unsafe_failure"]+= a["unsafe"]["failure"]
            protection_log["iterations"].append({
                "iteration": iteration + 1,
                "max_depth": cfg.max_depth_limit,
                "mode": method.current_mode_label() if hasattr(method, "current_mode_label") else method.display_name,
                "top_n_safe":   len(ctx.protected_safe_snapshot or []),
                "top_n_unsafe": len(ctx.protected_unsafe_snapshot or []),
                "audit_before_inner_loop": _audit_to_log_dict(audit_before, 0),
                "audit_after_steps": audit_after_steps,
            })

        # ---- Update regions for next iteration ----
        regions = new_regions
        definitive_fail_new = (len(regions["F_h_positive_in_unsafe"])
                               + len(regions["F_safe_cbf_violation"]))
        print(f"    [Patience] current={certified_pct:.2f}%, "
              f"patience={patience_counter}/{cfg.patience} (min_delta={min_delta:.2f}pp), "
              f"mode={method.current_mode_label() if hasattr(method, 'current_mode_label') else method.display_name}")

        iter_entry = {
            "iteration": iteration + 1, "max_depth": cfg.max_depth_limit,
            "mode": method.current_mode_label() if hasattr(method, "current_mode_label") else method.display_name,
            "loss": inner_history[-1]["loss"] if inner_history else 0.0,
            "HarmonicMeanPassRate": certified_pct,
            "R_safe": R_safe_pct, "R_unsafe": R_unsafe_pct,
            "standard_pass_rate": new_metrics["standard_pass_rate"],
            "f_h_positive":     len(regions["F_h_positive_in_unsafe"]),
            "f_safe_violation": len(regions["F_safe_cbf_violation"]),
            "f_depth_unsafe":   len(regions["F_depth_limit_reached_unsafe"]),
            "f_depth_safe":     len(regions["F_depth_limit_reached_safe"]),
            "f_unsafe_split":   len(regions["F_unsafe_cannot_split"]),
            "definitive_fail":  definitive_fail_new,
            "repair_type":      ctx.repair_type,
        }
        if method.needs_top_n():
            iter_entry["top_n_used"] = top_n_used
            iter_entry["beta_s"] = beta_s
            iter_entry["beta_us"] = beta_us
        iteration_results.append(iter_entry)

        if certified_pct >= cfg.target_pass_rate:
            print(f"\n  === Target pass rate {cfg.target_pass_rate}% reached! Early termination ===")
            break

    repair_total = time.perf_counter() - repair_start

    # ---- Final summary ----
    final_harmonic = iteration_results[-1]["HarmonicMeanPassRate"] if iteration_results else original_harmonic
    final_standard = iteration_results[-1]["standard_pass_rate"]   if iteration_results else original_standard
    final_R_safe   = iteration_results[-1]["R_safe"]               if iteration_results else original_R_safe
    final_R_unsafe = iteration_results[-1]["R_unsafe"]             if iteration_results else original_R_unsafe

    print(f"\n{'=' * 70}")
    print(f"Before/After Comparison ({method.display_name})")
    print(f"{'=' * 70}")
    print(f"Metric                   Original      Final         Change")
    print(f"---------------------------------------------------------------")
    print(f"HarmonicMeanPassRate:    {original_harmonic:>8.2f}%   {final_harmonic:>8.2f}%   "
          f"({final_harmonic - original_harmonic:+.2f}%)")
    print(f"standard_pass_rate:      {original_standard:>8.2f}%   {final_standard:>8.2f}%   "
          f"({final_standard - original_standard:+.2f}%)")
    print(f"R_safe:                  {original_R_safe:>8.2f}%   {final_R_safe:>8.2f}%")
    print(f"R_unsafe:                {original_R_unsafe:>8.2f}%   {final_R_unsafe:>8.2f}%")
    print(f"Repair time:             {repair_total:.2f}s")

    if method.audits():
        print(f"\n{'-' * 70}")
        print("Protection Audit Summary")
        print(f"{'-' * 70}")
        print(f"Safe regions (min_L > 0):   checks={protection_log['total_safe_checks']}, "
              f"success={protection_log['total_safe_success']}, "
              f"failure={protection_log['total_safe_failure']}")
        print(f"Unsafe regions (h_max < 0): checks={protection_log['total_unsafe_checks']}, "
              f"success={protection_log['total_unsafe_success']}, "
              f"failure={protection_log['total_unsafe_failure']}")

    # ---- Save results ----
    _save_result_json(
        method, cfg, method_results_dir,
        original_harmonic=original_harmonic, original_standard=original_standard,
        original_R_safe=original_R_safe, original_R_unsafe=original_R_unsafe,
        final_harmonic=final_harmonic, final_standard=final_standard,
        final_R_safe=final_R_safe, final_R_unsafe=final_R_unsafe,
        iteration_results=iteration_results, repair_time_seconds=repair_total,
        protection_log=protection_log if method.audits() else None,
    )


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def _save_result_json(method: RepairMethod, cfg: MethodConfig, results_dir: str, **kwargs):
    """Write the main result JSON (and the protection-log sidecar if present)."""
    stem = method.result_file_stem()
    result_file = os.path.join(results_dir, f"{stem}.json")

    run_result = {
        "system": cfg.system, "activation": cfg.activation,
        "method": method.display_name,
        "max_depth_limit": cfg.max_depth_limit,
        "num_inner_steps": cfg.num_inner_steps, "lr": cfg.lr,
        "target_pass_rate": cfg.target_pass_rate, "patience": cfg.patience,
        "max_total_iterations": cfg.max_total_iterations,
        "lambda_": cfg.lambda_,
        "original_max_depth_harmonic": kwargs["original_harmonic"],
        "original_max_depth_standard": kwargs["original_standard"],
        "original_max_depth_R_safe": kwargs["original_R_safe"],
        "original_max_depth_R_unsafe": kwargs["original_R_unsafe"],
        "final_harmonic_pass_rate": kwargs["final_harmonic"],
        "final_standard_pass_rate": kwargs["final_standard"],
        "final_R_safe": kwargs["final_R_safe"],
        "final_R_unsafe": kwargs["final_R_unsafe"],
        "harmonic_improvement": kwargs["final_harmonic"] - kwargs["original_harmonic"],
        "standard_improvement": kwargs["final_standard"] - kwargs["original_standard"],
        "num_iterations": len(kwargs["iteration_results"]),
        "iteration_results": kwargs["iteration_results"],
        "repair_time_seconds": kwargs["repair_time_seconds"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if method.needs_top_n():
        beta_s, beta_us = method.get_beta()
        run_result["top_n_protect"] = cfg.top_n_protect
        run_result["delta_theta_norm_bound"] = cfg.delta_theta_norm_bound
        run_result["beta_s"] = beta_s
        run_result["beta_us"] = beta_us
    if "skip_reason" in kwargs:
        run_result["skip_reason"] = kwargs["skip_reason"]

    protection_log = kwargs.get("protection_log")
    if protection_log is not None:
        run_result["protection_audit_summary"] = {
            "total_safe_checks":    protection_log["total_safe_checks"],
            "total_safe_success":   protection_log["total_safe_success"],
            "total_safe_failure":   protection_log["total_safe_failure"],
            "total_unsafe_checks":  protection_log["total_unsafe_checks"],
            "total_unsafe_success": protection_log["total_unsafe_success"],
            "total_unsafe_failure": protection_log["total_unsafe_failure"],
        }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
    print(f"\nResults saved: {result_file}")

    if protection_log is not None:
        protection_file = os.path.join(results_dir, f"{stem}_protection.json")
        protection_log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(protection_file, "w", encoding="utf-8") as f:
            json.dump(protection_log, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        print(f"Protection log saved: {protection_file}")
