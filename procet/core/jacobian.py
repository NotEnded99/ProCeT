"""Jacobian of LBP bounds w.r.t. flattened model parameters.

Two implementations are provided:

``compute_jacobian_for_lbp_bounds``
    Single-threaded reference. One row per region: ``J[i] = ∇_θ b_i``.

``compute_jacobian_for_lbp_bounds_v1``
    Multi-threaded variant using one model + linearizer + CUDA stream per
    worker. Output is identical (modulo floating-point non-determinism on
    concurrent CUDA streams). Used by the ProCeT family for speed.
"""

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch

from lbp_neural_cbf.linearization.linear_derivative_bounds import CrownPartialLinearization
from lbp_neural_cbf.regions import SimplicialRegion

from .lbp_loss import compute_min_L_with_mccormick, compute_h_max_via_network_bounds


def _to_simplicial_regions(simplices):
    regions = []
    for verts in simplices:
        if isinstance(verts, np.ndarray):
            verts_np = verts.astype(np.float32)
        else:
            verts_np = verts.detach().cpu().numpy().astype(np.float32)
        regions.append(SimplicialRegion(verts_np, output_dim=None))
    return regions


def _bound_fn(bound_type):
    if bound_type == "safe":
        return compute_min_L_with_mccormick
    return compute_h_max_via_network_bounds


# ---------------------------------------------------------------------------
# Single-threaded reference
# ---------------------------------------------------------------------------

def compute_jacobian_for_lbp_bounds(model, simplices, bound_type, dynamics_model,
                                    lbp_linearizer, device, dtype):
    """Single-threaded Jacobian ``J[i] = ∇_θ b_i`` of shape ``[N, num_params]``.

    Args:
        bound_type: ``'safe'`` for ``min_L`` (used by V_safe protection) or
            ``'unsafe'`` for ``h_max`` (used by V_unsafe protection).
    """
    num_params = sum(p.numel() for p in model.parameters())
    if len(simplices) == 0:
        return torch.empty(0, num_params, device=device, dtype=dtype)

    regions = _to_simplicial_regions(simplices)
    bound_function = _bound_fn(bound_type)

    BATCH_SIZE = 32
    J_rows = []
    for bs in range(0, len(regions), BATCH_SIZE):
        be = min(bs + BATCH_SIZE, len(regions))
        print(f"  Computing Jacobian for {bound_type} LBP batch [{bs}:{be}]...")
        batch_regions = regions[bs:be]
        bounds = bound_function(batch_regions, dynamics_model, lbp_linearizer, device, dtype)

        for i in range(len(batch_regions)):
            b_i = bounds[i:i + 1]
            model.zero_grad()
            b_i.backward(retain_graph=True)
            grad_vec = torch.cat([
                p.grad.flatten() if p.grad is not None
                else torch.zeros(p.numel(), dtype=dtype, device=device)
                for p in model.parameters()
            ])
            J_rows.append(grad_vec)

        del bounds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.stack(J_rows, dim=0)


# ---------------------------------------------------------------------------
# Multi-threaded variant (one model copy + stream per worker)
# ---------------------------------------------------------------------------

def compute_jacobian_for_lbp_bounds_v1(model, simplices, bound_type, dynamics_model,
                                       lbp_linearizer, device, dtype, num_threads=16):
    """Multi-threaded Jacobian. Bitwise-identical to the reference modulo
    floating-point non-determinism across CUDA streams.

    Trade-offs vs single-threaded:
        + Higher throughput on launch-bound workloads (small per-sample backward).
        - N × model memory (one deep-copy per thread).
        - Slight non-determinism in CUDA matmul ordering across streams.
    """
    num_params = sum(p.numel() for p in model.parameters())
    n = len(simplices)
    if n == 0:
        return torch.empty(0, num_params, device=device, dtype=dtype)

    regions = _to_simplicial_regions(simplices)
    bound_function = _bound_fn(bound_type)

    num_threads = min(num_threads, n)
    chunk_size = (n + num_threads - 1) // num_threads
    chunks = []
    for t in range(num_threads):
        start = t * chunk_size
        end = min(start + chunk_size, n)
        if start < end:
            chunks.append(regions[start:end])
    actual_num_threads = len(chunks)

    # Pre-create per-thread model + linearizer copies in the MAIN thread
    # (avoids concurrent CUDA allocator contention from multi-thread deepcopy).
    print(f"  [v1 multi-thread] threads={actual_num_threads}, "
          f"chunk_sizes={[len(c) for c in chunks]}, bound_type={bound_type}")
    thread_models = [copy.deepcopy(model) for _ in range(actual_num_threads)]
    thread_linearizers = [CrownPartialLinearization(m, dtype=torch.float32) for m in thread_models]

    BATCH_SIZE = 32

    def worker(tid, chunk_regions):
        thread_model = thread_models[tid]
        thread_linearizer = thread_linearizers[tid]
        params_list = list(thread_model.parameters())
        chunk_len = len(chunk_regions)

        J_chunk = torch.empty(chunk_len, num_params, device=device, dtype=dtype)

        stream_ctx = (
            torch.cuda.stream(torch.cuda.Stream(device=device))
            if torch.cuda.is_available() else nullcontext()
        )

        with stream_ctx:
            row_offset = 0
            for bs in range(0, chunk_len, BATCH_SIZE):
                be = min(bs + BATCH_SIZE, chunk_len)
                batch_regions = chunk_regions[bs:be]
                bounds = bound_function(batch_regions, dynamics_model, thread_linearizer, device, dtype)

                for i in range(len(batch_regions)):
                    b_i = bounds[i:i + 1]
                    grads = torch.autograd.grad(
                        b_i, params_list,
                        retain_graph=True, create_graph=False, allow_unused=True,
                    )
                    grad_vec = torch.cat([
                        g.flatten() if g is not None
                        else torch.zeros(p.numel(), dtype=dtype, device=device)
                        for g, p in zip(grads, params_list)
                    ])
                    J_chunk[row_offset] = grad_vec
                    row_offset += 1

                del bounds
                # NOTE: intentionally no empty_cache() — caller said memory is not a concern.

        return J_chunk

    with ThreadPoolExecutor(max_workers=actual_num_threads) as executor:
        futures = [executor.submit(worker, tid, chunk) for tid, chunk in enumerate(chunks)]
        chunk_results = [future.result() for future in futures]

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    return torch.cat(chunk_results, dim=0)
