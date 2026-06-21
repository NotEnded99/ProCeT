# ProCeT: Certified Training of Neural Control Barrier Functions

**ProCeT** is a framework for **certified Training** of neural Control Barrier Functions (CBFs). Given a trained barrier network that fails verification on some simplicial regions of the state space, ProCeT produces a *certifiably trained* network via Linear Bound Propagation (LBP) with McCormick envelopes and Second-Order Cone Programming (SOCP) projection.

This repository implements three Training methods that form a strict hierarchy:

| Method | Paper name | One-line summary |
|--------|------------|------------------|
| **CeT** | `CeT` | Certified Training — plain gradient descent on the LBP training loss (baseline). |
| **α-ProCeT** | `α-ProCeT` | SOCP projection (Eq. 8) with Jacobian-based Top-N protection and per-step audit. |
| **β-ProCeT** | `β-ProCeT` | Adaptive — starts in CeT mode, escalates to α-ProCeT (with backtracking) on the first plateau. |

---

## Table of Contents

- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Method Comparison](#method-comparison)
- [CLI Reference](#cli-reference)
- [Inputs and Outputs](#inputs-and-outputs)
- [Results Format](#results-format)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA 12.1 (recommended; CPU-only works but is ~10× slower)

### Steps

```bash
git clone <repo-url> ProCeT
cd ProCeT

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies
pip install -r requirements.txt
```

`requirements.txt` pins all runtime dependencies (NumPy, SciPy, cvxpy + solvers, `bound_propagation`, ONNX, tqdm, matplotlib).

---

## Repository Layout

```
ProCeT/
├── procet/                       # Training framework
│   ├── core/
│   │   ├── systems.py            # DYNAMICS_SYSTEMS, SYSTEM_DEPTH, activations
│   │   ├── io.py                 # pytorch_to_onnx, verify_model
│   │   ├── metrics.py            # compute_safety_metrics_v8, NumpyJSONEncoder
│   │   ├── selection.py          # Top-N vulnerable region selection
│   │   ├── jacobian.py           # ∇_θ φ_θ / ∇_θ ψ_θ Jacobians
│   │   ├── lbp_loss.py           # LBP training loss with McCormick envelopes
│   │   ├── socp.py               # SOCP projection update (Eq. 8)
│   │   └── audit.py              # Protection audit after each inner step
│   ├── methods/
│   │   ├── base.py               # RepairMethod ABC + MethodConfig + IterationContext
│   │   ├── cet.py                # CeT
│   │   ├── alpha_procet.py       # α-ProCeT
│   │   └── beta_procet.py        # β-ProCeT
│   └── runner.py                 # Outer training loop (template method)
│
├── scripts/                      # CLI entry points
│   ├── _common.py                # Shared argparse + main_for() factory
│   ├── run_cet.py
│   ├── run_alpha_procet.py
│   └── run_beta_procet.py
│
├── initial_training/             # Step 1: train + verify barrier networks
│   └── barrier_certificate.py
│
├── lbp_neural_cbf/               # LBP/CROWN verification framework
│
├── data/                         # Inputs (gitignored except small files)
│   ├── models/                   # Trained barrier networks (.pth / .onnx)
│   └── regions/                  # Pre-verified simplicial regions (.pt)
│
├── results/                      # Outputs (gitignored)
│   ├── models/                   # Trained barrier networks
│   ├── CeT/
│   ├── alphaProCeT_prab/
│   └── betaProCeT_prab/
│
├── read_all_results.py           # Aggregate results → LaTeX comparison table
├── requirements.txt
└── README.md
```

---

## Quickstart

Run a single training experiment (assumes `data/models/` and `data/regions/` are populated — see [End-to-End Pipeline](#end-to-end-pipeline)):

```bash
# CeT — baseline plain GD
python scripts/run_cet.py -a Tanh -s barr1 --lambda 2.0

# α-ProCeT — SOCP + Top-N protection + audit
python scripts/run_alpha_procet.py -a Tanh -s barr1 \
    --lambda 2.0 --delta-theta-norm-bound 0.05 --top-n-protect 50 --num-inner-steps 5

# β-ProCeT — adaptive (CeT → α-ProCeT on first plateau)
python scripts/run_beta_procet.py -a Tanh -s barr1 \
    --lambda 2.0 --delta-theta-norm-bound 0.05 --top-n-protect 50 --num-inner-steps 5
```

Batch over all systems × activations (single GPU):

```bash
for sys in simple2d barr1 barr2 barr3 barr4 cartpole; do
  for act in Tanh Sigmoid; do
    CUDA_VISIBLE_DEVICES=0 python scripts/run_alpha_procet.py -a $act -s $sys \
        --lambda 2.0 --delta-theta-norm-bound 0.05 --top-n-protect 50 --num-inner-steps 5
  done
done
```
---

## End-to-End Pipeline

ProCeT assumes a three-stage pipeline. Steps 1–2 produce the inputs that step 3 (the actual training) consumes.

### Stage 1 — Train the initial barrier network

```bash
python initial_training/barrier_certificate.py \
    --system-type barr1 --train \
    --activation Tanh --hidden-sizes '32,64,32' \
    --save-path data/models/tanh_models
```

This writes `data/models/tanh_models/{system}_cbf.pth` and `.onnx`.

### Stage 2 — Verify and dump simplicial regions

```bash
python initial_training/barrier_certificate.py \
    --system-type barr1 --verify \
    --activation Tanh --max-depth 12 --hidden-sizes '32,64,32'
```

This runs full-LBP verification (with McCormick) on the trained network and saves the resulting region buckets to:

```
data/regions/verified_regions_{system}_{activation}_v1_depth{N}.pt
```

The depth `N` depends on the system (see [Systems and Depths](#systems-and-depths)).

### Stage 3 — Training

See [Quickstart](#quickstart).

---

## Method Comparison

All three methods share the same outer loop (load → verify → train → re-verify → log) implemented in [`procet/runner.py`](procet/runner.py). They differ only in the inner update step.

| Aspect | CeT | α-ProCeT | β-ProCeT |
|--------|-----|----------|----------|
| Update rule | Plain GD on training loss | SOCP projection (Eq. 8) | CeT → SOCP (one-way switch) |
| Top-N vulnerable protection | No | Yes (Jacobian-based) | Yes (after escalation) |
| Per-step audit | No | Yes | Yes (after escalation) |
| Per-step compute | Cheap | Expensive (Jacobian + SOCP) | Cheap, then expensive |
| Supported activations | Tanh, Sigmoid | Tanh, Sigmoid (smooth only) | Tanh, Sigmoid |

### β-ProCeT escalation

β-ProCeT starts in CeT mode. The first time the patience counter hits 1 *while still in Phase 1*, the method:

1. Restores the pre-iteration parameters (backtracking).
2. Flips `use_protection = True`.
3. Re-runs the inner loop with the full α-ProCeT machinery (Top-N snapshot, Jacobians, SOCP).
4. Stays in Phase 2 for all subsequent iterations.

This is the algorithmic difference vs. α-ProCeT (which is in Phase 2 from iteration 1).

---

## CLI Reference

All three scripts share the same argument parser ([`scripts/_common.py`](scripts/_common.py)).

### Common arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-a, --activation` | *(required)* | Activation function. `Tanh` / `Sigmoid` for α/β-ProCeT; also `Relu` / `LeakyRelu` for CeT. |
| `-s, --system` | *(required)* | Dynamical system (see [Systems and Depths](#systems-and-depths)). |
| `--lambda` | `2.0` | Weight for definitive-fail regions (was `--definitive-weight`). |
| `--num-inner-steps` | `5` | Number of inner updates per outer iteration (K). |
| `--lr` | `5e-3` | Base learning rate. Per-method depth-aware overrides apply (see below). |
| `--target-pass-rate` | `100.0` | Stop early when this pass rate is reached. |
| `--patience` | `3` | Stop after N consecutive iterations without improvement. |
| `--max-total-iterations` | `10` | Hard cap on outer iterations. |
| `--seed` | `2026` | RNG seed (Python, NumPy, PyTorch CPU + CUDA + cuDNN deterministic). |

### α/β-ProCeT only

| Flag | Default  | Description |
|------|-----------------|-------------|
| `--top-n-protect` | `50` | Number of most-vulnerable regions to protect / audit. |
| `--delta-theta-norm-bound` | `0.05` | L2 trust-region radius ζ for the SOCP update. |

### Systems and Depths

| System key | Depth | Description |
|------------|-------|-------------|
| `simple2d` | 12 | 2D toy system |
| `barr1`, `barr2`, `barr3` | 12 | FOSSIL barrier systems |
| `barr4` | 14 | Higher-dimensional barrier system |
| `cartpole` | 14 | Cartpole |

---

## Inputs and Outputs

### Inputs

| Path | Contents |
|------|----------|
| `data/models/{activation}_models/{system}_cbf.{pth,onnx}` | Initial trained barrier network. |
| `data/regions/verified_regions_{system}_{activation}_v1_depth{N}.pt` | Pre-verified simplicial region buckets (`V_safe`, `V_unsafe`, `F_h_positive_in_unsafe`, `F_safe_cbf_violation`, `F_depth_limit_reached_unsafe`, `F_depth_limit_reached_safe`, `F_unsafe_cannot_split`). |

### Outputs

| Path | Contents |
|------|----------|
| `results/models/{system}_{activation}_cbf_repaired_{suffix}.{pth,onnx}` | Latest trained barrier network (overwritten each outer iteration). |
| `results/{suffix}/result_{system}_{activation}_{suffix}_depth{N}_w{λ}[_k{K}_zeta{ζ}].json` | Per-run summary (metrics, hyperparameters, per-iteration history). |
| `results/{suffix}/result_..._protection.json` | Per-step protection audit log (α/β-ProCeT only). |

`suffix` is `CeT` / `alphaProCeT_prab` / `betaProCeT_prab`.

---

## Results Format

The main result JSON has the following schema (abbreviated):

```jsonc
{
  "system": "barr1",
  "activation": "Tanh",
  "method": "α-ProCeT",
  "max_depth_limit": 12,
  "num_inner_steps": 5,
  "lambda_": 2.0,                          // CeT name: "lambda_"
  // α/β-ProCeT also include:
  "top_n_protect": 50,
  "delta_theta_norm_bound": 0.05,
  "beta_s": 0.999, "beta_us": 0.999,

  "original_max_depth_harmonic": 93.5,     // Pre-training harmonic pass rate
  "original_max_depth_standard": 93.2,     // Pre-training standard pass rate
  "final_harmonic_pass_rate": 96.9,
  "final_standard_pass_rate": 96.8,

  "num_iterations": 7,
  "iteration_results": [ /* per-iteration entries */ ],
  "repair_time_seconds": 142.3
}
```

Per-iteration entries record loss, gradient/update norms, region counts, and (for α/β-ProCeT) Top-N audit snapshots.

---

## Acknowledgements

The LBP/CROWN verification framework in [`lbp_neural_cbf/`](lbp_neural_cbf/) is based on [verification-of-neural-cbf-via-lbp](https://github.com/Zinoex/verification-of-neural-cbf-via-lbp) by Vertovec, Mathiesen, Badings, Laurenti, and Abate, from the paper *"Scalable Verification of Neural Control Barrier Functions Using Linear Bound Propagation."*

---

## License

[MIT](LICENSE) © TODO
