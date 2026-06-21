import itertools
import os
import time
import traceback
import types
import logging
from datetime import datetime

import numpy as np
import torch

from ..cbf.network import BarrierNN, empirical_cbf_validation
from ..certification_results import SampleResultMaybe, SampleResultSAT, SampleResultUNSAT
from ..executors import SinglethreadExecutor, MultithreadExecutor, MultiprocessExecutor
from ..linearization import CrownPartialLinearization, TaylorLinearization
from ..translators import TorchTranslator
from ..regions import HyperrectangularRegion, SimplicialRegion, create_region_generator
from ..visualization.cbf_plotter import create_cbf_verification_plotter
from .domain import unsafe_region

# Global namespace for worker-specific objects
_LOCAL = types.SimpleNamespace()


def aggregate(agg, result):
    """
    Aggregate verification results for sound verification.

    For sound CBF verification, we collect all results to analyze:
    - SAT results: Regions where CBF condition is verified
    - UNSAT results: Regions where CBF condition is violated (counterexamples found)
    - MAYBE results: Regions where verification is inconclusive
    """
    if agg is None:
        agg = []
    agg.append(result)
    return agg


def verify_cbf(
    dynamics_model,
    barrier_model_path=None,
    executor_type="single",
    region_type="simplicial",
    visualize=False,
    use_gpu=True,
    batch_size=512,
    max_depth=None,
    save_verification_regions=False,
):
    """
    Main function to verify a neural control barrier function using CROWN linearization.

    Args:
        dynamics_model: CBF dynamical system
        barrier_model_path: Path to trained barrier function model
        executor_type: Type of executor to use ("single", "multi-thread", or "multi-process")
                      Each executor automatically determines optimal number of workers
        region_type: Type of regions to use ("hyperrectangular" or "simplicial")
        visualize: Whether to create live visualization during verification (2D only)
        use_gpu: Whether to use GPU for verification (auto-detects if None: True for single/multi-thread, False for multi-process)
        max_depth: Maximum depth for region splitting (None for unlimited)

    Returns:
        Verification results with guaranteed soundness for SAT results
    """
    if barrier_model_path is None:
        raise ValueError("barrier_model_path must be provided for verification")

    print(f"Verifying CBF: {barrier_model_path}")

    print(f"Using {'GPU' if use_gpu and torch.cuda.is_available() else 'CPU'} for verification")

    # Create verification strategy
    strategy = CBFVerificationStrategy(
        barrier_model_path,
        dynamics_model,
        use_gpu=use_gpu,
        max_depth=max_depth,
    )

    # Generate initial samples
    region_generator = create_region_generator(region_type)
    samples = region_generator.create_mesh(dynamics_model).get_regions(0)

    # Create plotter if visualization is requested
    plotter = None
    if visualize and dynamics_model.input_dim == 2:
        try:
            # Load barrier network for visualization (must use BarrierNN, not SimpleNN!)
            barrier_net = None
            if barrier_model_path:
                device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
                # Try PyTorch file first
                if barrier_model_path.endswith(".pth") and os.path.exists(barrier_model_path):
                    barrier_net = BarrierNN(input_size=dynamics_model.input_dim, hidden_sizes=getattr(dynamics_model, "hidden_sizes"), activation_fnc=getattr(dynamics_model, "activation_fnc", "Tanh"), device=device)
                    barrier_net.load_state_dict(torch.load(barrier_model_path, map_location=device, weights_only=False))
                    barrier_net.eval()
                    print(f"Loaded BarrierNN model from {barrier_model_path}")
                # For ONNX files, try corresponding PyTorch file
                elif barrier_model_path.endswith(".onnx"):
                    pth_path = barrier_model_path.replace(".onnx", ".pth")
                    if os.path.exists(pth_path):
                        barrier_net = BarrierNN(input_size=dynamics_model.input_dim, hidden_sizes=getattr(dynamics_model, "hidden_sizes"), activation_fnc=getattr(dynamics_model, "activation_fnc", "Tanh"), device=device)
                        barrier_net.load_state_dict(torch.load(pth_path, map_location=device, weights_only=False))
                        barrier_net.eval()
                        print(f"Loaded BarrierNN model from {pth_path} for visualization (ONNX verification)")
                    else:
                        print(f"ONNX model at {barrier_model_path} found, but no corresponding PyTorch file at {pth_path} - showing verification regions only")
                else:
                    print(f"Barrier model not found at {barrier_model_path} - showing verification regions only")
            else:
                print("No barrier model path provided - showing verification regions only")

            plotter = create_cbf_verification_plotter(dynamics_model, barrier_net, resolution=100)
            if plotter:
                print("Created CBF verification plotter - live visualization enabled")
            else:
                print("Failed to create CBF verification plotter")
        except Exception as e:
            print(f"Failed to create visualization plotter: {e}")
            import traceback

            traceback.print_exc()
            plotter = None

    # Create executor based on type
    if executor_type == "single":
        executor = SinglethreadExecutor()
    elif executor_type == "multi-thread":
        executor = MultithreadExecutor()
    elif executor_type == "multi-process":
        executor = MultiprocessExecutor()
    else:
        raise ValueError(f"Invalid executor_type: {executor_type}. Must be 'single', 'multi-thread', or 'multi-process'")

    print(f"Using {executor_type} executor")

    agg, certified_percentage, uncertified_percentage, computation_time = executor.execute(
        initializer=strategy.initialize_worker,
        process_batch=strategy.verify_batch,
        aggregate=aggregate,
        samples=samples,
        plotter=plotter,
        batch_size=batch_size,
    )

    print("\n" + "=" * 60)
    print("CBF VERIFICATION RESULTS")
    print("=" * 60)
    print(f"System: {dynamics_model.system_name}")
    print(f"Certified percentage: {certified_percentage:.4f}%")
    print(f"Uncertified percentage: {uncertified_percentage:.4f}%")
    print(f"Computation time: {computation_time:.2f} seconds")

    # Calculate total samples processed and iterations per second
    total_samples = len(agg) if agg else 0
    iterations_per_second = total_samples / computation_time if computation_time > 0 else 0
    print(f"Total samples processed: {total_samples}")
    print(f"Iterations per second: {iterations_per_second:.2f} it/s")


    V_safe = []              # SAT, safe_cbf_verified
    V_unsafe = []             # SAT, unsafe_region 
    F_h_positive_in_unsafe = []    # UNSAT, h_positive_in_unsafe
    F_safe_cbf_violation = []      # UNSAT, safe_cbf_violation 
    F_depth_limit_reached_unsafe = []     # UNSAT, depth_limit_reached_unsafe 
    F_depth_limit_reached_safe = []       # UNSAT, depth_limit_reached_safe 
    F_unsafe_cannot_split = []      # UNSAT, unsafe_cannot_split

    for result in agg:
        sample = result.sample

        if hasattr(sample, 'vertices'):
            vertices = np.array(sample.vertices, dtype=np.float32)
        elif hasattr(sample, 'center_point') and hasattr(sample, 'radius_vec'):
            center = np.array(sample.center_point, dtype=np.float32)
            radius = np.array(sample.radius_vec, dtype=np.float32)
            vertices = np.stack([
                center - radius,
                center + radius
            ], axis=0)
        else:
            print(f"Warning: Cannot extract vertices from sample {type(sample)}")
            continue

        if isinstance(result, SampleResultSAT):
            result_type = result.result_type
            if result_type == "unsafe_region":
                V_unsafe.append(vertices)
            elif result_type == "safe_cbf_verified":
                V_safe.append(vertices)

        elif isinstance(result, SampleResultUNSAT):
            result_type = result.result_type
            if result_type == "h_positive_in_unsafe":
                F_h_positive_in_unsafe.append(vertices)
            elif result_type == "safe_cbf_violation":
                F_safe_cbf_violation.append(vertices)
            elif result_type == "depth_limit_reached_unsafe":
                F_depth_limit_reached_unsafe.append(vertices)
            elif result_type == "depth_limit_reached_safe":
                F_depth_limit_reached_safe.append(vertices)
            elif result_type == "unsafe_cannot_split":
                F_unsafe_cannot_split.append(vertices)

    results = {
        "regions": agg,
        "certified_percentage": certified_percentage,
        "uncertified_percentage": uncertified_percentage,
        "computation_time": computation_time,
        "total_samples": total_samples,
        "iterations_per_second": iterations_per_second,
        "V_safe": V_safe,
        "V_unsafe": V_unsafe,
        "F_h_positive_in_unsafe": F_h_positive_in_unsafe,
        "F_safe_cbf_violation": F_safe_cbf_violation,
        "F_depth_limit_reached_unsafe": F_depth_limit_reached_unsafe,
        "F_depth_limit_reached_safe": F_depth_limit_reached_safe,
        "F_unsafe_cannot_split": F_unsafe_cannot_split
    }

    # Print visualization statistics if plotter was used
    if plotter:
        stats = plotter.get_verification_statistics()
        print(f"Visualization regions: {stats['total_regions']}")
        print(f"  Verified: {stats['verified_count']} ({stats['verified_percentage']:.2f}%)")
        print(f"  Counterexamples: {stats['counterexample_count']} ({stats['counterexample_percentage']:.2f}%)")

        results.update(stats)

        # Save final plot
        plot_filename = f"plots/{dynamics_model.system_name}_cbf_verification.png"
        os.makedirs("plots", exist_ok=True)
        plotter.save_final_plot(plot_filename)

    if save_verification_regions:
        print("\n" + "=" * 60)
        print("SAVING VERIFICATION REGIONS FOR REPAIR")
        print("=" * 60)

        V_safe = []              # SAT, safe_cbf_verified 
        V_unsafe = []             # SAT, unsafe_region
        F_h_positive_in_unsafe = []    # UNSAT, h_positive_in_unsafe
        F_safe_cbf_violation = []      # UNSAT, safe_cbf_violation
        F_depth_limit_reached_unsafe = []     # UNSAT, depth_limit_reached_unsafe
        F_depth_limit_reached_safe = []       # UNSAT, depth_limit_reached_safe
        F_unsafe_cannot_split = []      # UNSAT, unsafe_cannot_split

        for result in agg:
            sample = result.sample

            if hasattr(sample, 'vertices'):
                vertices = np.array(sample.vertices, dtype=np.float32)
            elif hasattr(sample, 'center_point') and hasattr(sample, 'radius_vec'):
                center = np.array(sample.center_point, dtype=np.float32)
                radius = np.array(sample.radius_vec, dtype=np.float32)
                vertices = np.stack([
                    center - radius,
                    center + radius
                ], axis=0)
            else:
                print(f"Warning: Cannot extract vertices from sample {type(sample)}")
                continue

            if isinstance(result, SampleResultSAT):
                result_type = result.result_type
                if result_type == "unsafe_region":
                    V_unsafe.append(vertices)
                elif result_type == "safe_cbf_verified":
                    V_safe.append(vertices)

            elif isinstance(result, SampleResultUNSAT):
                result_type = result.result_type
                if result_type == "h_positive_in_unsafe":
                    F_h_positive_in_unsafe.append(vertices)
                elif result_type == "safe_cbf_violation":
                    F_safe_cbf_violation.append(vertices)
                elif result_type == "depth_limit_reached_unsafe":
                    F_depth_limit_reached_unsafe.append(vertices)
                elif result_type == "depth_limit_reached_safe":
                    F_depth_limit_reached_safe.append(vertices)
                elif result_type == "unsafe_cannot_split":
                    F_unsafe_cannot_split.append(vertices)
                # else:
                #     # Unknown UNSAT type, classify as safe_cbf_violation
                #     F_safe_cbf_violation.append(vertices)

        # Print statistics
        print(f"V_safe (safe_cbf_verified): {len(V_safe)}")
        print(f"V_unsafe (unsafe_region): {len(V_unsafe)}")
        print(f"F_h_positive_in_unsafe: {len(F_h_positive_in_unsafe)}")
        print(f"F_safe_cbf_violation: {len(F_safe_cbf_violation)}")
        print(f"F_depth_limit_reached_unsafe: {len(F_depth_limit_reached_unsafe)}")
        print(f"F_depth_limit_reached_safe: {len(F_depth_limit_reached_safe)}")
        print(f"F_unsafe_cannot_split: {len(F_unsafe_cannot_split)}")

        # Get activation function name
        activation_fnc = getattr(dynamics_model, 'activation_fnc', 'Unknown')

        # Save to file (filename includes activation function)
        regions_dir = "data/regions"
        os.makedirs(regions_dir, exist_ok=True)
        save_path = f"{regions_dir}/verified_regions_{dynamics_model.system_name}_{activation_fnc}_v1_depth{max_depth}.pt"

        regions_data = {
            'V_safe': V_safe,
            'V_unsafe': V_unsafe,
            'F_h_positive_in_unsafe': F_h_positive_in_unsafe,
            'F_safe_cbf_violation': F_safe_cbf_violation,
            'F_depth_limit_reached_unsafe': F_depth_limit_reached_unsafe,
            'F_depth_limit_reached_safe': F_depth_limit_reached_safe,
            'F_unsafe_cannot_split': F_unsafe_cannot_split,
            'system_name': dynamics_model.system_name,
            'activation_fnc': activation_fnc,
            'input_dim': dynamics_model.input_dim,
            'max_depth': max_depth,
            "Certified percentage": certified_percentage,
            "Uncertified percentage": uncertified_percentage,
        }

        torch.save(regions_data, save_path)
        print(f"Verification regions saved to: {save_path}")

        # ========== Save verification log to file ==========
        logs_dir = "data/logs"
        os.makedirs(logs_dir, exist_ok=True)

        # Build log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"verify_{dynamics_model.system_name}_{activation_fnc}_depth{max_depth}_{timestamp}.log"
        log_path = os.path.join(logs_dir, log_filename)

        # Configure logger
        logger = logging.getLogger('cbf_verification')
        logger.setLevel(logging.INFO)
        logger.handlers = []  # Clear existing handlers

        # File handler
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Write verification results to log
        logger.info("=" * 60)
        logger.info("CBF VERIFICATION LOG")
        logger.info("=" * 60)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"System: {dynamics_model.system_name}")
        logger.info(f"Activation Function: {activation_fnc}")
        logger.info(f"Network Path: {barrier_model_path}")
        logger.info(f"Region Type: {region_type}")
        logger.info(f"Executor Type: {executor_type}")
        logger.info(f"Max Depth: {max_depth}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Input Dimension: {dynamics_model.input_dim}")
        logger.info(f"Control Dimension: {dynamics_model.control_dim}")
        logger.info("-" * 60)
        logger.info("VERIFICATION RESULTS")
        logger.info("-" * 60)
        logger.info(f"Certified percentage: {certified_percentage:.4f}%")
        logger.info(f"Uncertified percentage: {uncertified_percentage:.4f}%")
        logger.info(f"Computation time: {computation_time:.2f} seconds")
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Iterations per second: {iterations_per_second:.2f} it/s")
        logger.info("-" * 60)
        logger.info("REGION STATISTICS")
        logger.info("-" * 60)
        logger.info(f"V_safe (safe_cbf_verified): {len(V_safe)}")
        logger.info(f"V_unsafe (unsafe_region): {len(V_unsafe)}")
        logger.info(f"F_h_positive_in_unsafe: {len(F_h_positive_in_unsafe)}")
        logger.info(f"F_safe_cbf_violation: {len(F_safe_cbf_violation)}")
        logger.info(f"F_depth_limit_reached_unsafe: {len(F_depth_limit_reached_unsafe)}")
        logger.info(f"F_depth_limit_reached_safe: {len(F_depth_limit_reached_safe)}")
        logger.info(f"F_unsafe_cannot_split: {len(F_unsafe_cannot_split)}")
        logger.info("=" * 60)

        # Close handlers to flush log to file
        file_handler.close()
        console_handler.close()
        logger.removeHandler(file_handler)
        logger.removeHandler(console_handler)

        print(f"Verification log saved to: {log_path}")
        print("=" * 60)
    else:
        pass

    return results


class CBFVerificationStrategy:
    """
    Verification strategy for CBF conditions using CROWN linearization.
    """

    def __init__(self, network_path, dynamics_model, use_gpu=True, max_depth=None):
        self.network_path = network_path
        self.dynamics_model = dynamics_model
        self.use_gpu = use_gpu
        self.max_depth = max_depth

    @staticmethod
    def _handle_split(sample, start_time, results, sample_idx, min_volume, split_type, unsat_type, max_depth=None, depth_limit_type=None):
        """Record a MAYBE result via splitting or an UNSAT counterexample if splitting is not possible or depth limited."""
        # Check if maximum depth is reached
        if max_depth is not None and sample.depth >= max_depth:
            counterexample = sample.center
            result_type = depth_limit_type if depth_limit_type is not None else "depth_limit_reached"
            results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type=result_type)
            return

        if sample._compute_volume() > min_volume:
            new_samples = sample.split()
            if new_samples:
                results[sample_idx] = SampleResultMaybe(sample, start_time, new_samples, split_type=split_type)
                return

        counterexample = sample.center  # TODO: For now we are just returning the center as counterexample - we could return vertices instead or in addition
        results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type=unsat_type)

    def initialize_worker(self):
        """Initialize the PyTorch model and CROWN for each worker process."""
        global _LOCAL
        # Use GPU if explicitly enabled and available, otherwise CPU
        # GPU is recommended for single-threaded execution for better performance
        # For multiprocessing, CPU is safer to avoid CUDA context issues
        device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
        dtype = torch.float32

        pth_path = self.network_path.replace(".onnx", ".pth")
        activation_fnc = getattr(self.dynamics_model, 'activation_fnc', 'Tanh')
        _LOCAL.torch_model = BarrierNN(
            input_size=self.dynamics_model.input_dim,
            hidden_sizes=self.dynamics_model.hidden_sizes,
            activation_fnc=activation_fnc,
            device=device,
        )
        _LOCAL.torch_model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=False))
        _LOCAL.torch_model = _LOCAL.torch_model.to(dtype=dtype)
        _LOCAL.torch_model.eval()

        if False:  # For debugging: empirical validation of CBF
            ver, cex = empirical_cbf_validation(
                _LOCAL.torch_model,
                self.dynamics_model,
                num_samples=50000,
                alpha=self.dynamics_model.alpha,
            )

        _LOCAL.network_linearizer = CrownPartialLinearization(_LOCAL.torch_model, dtype=dtype)

        _LOCAL.dynamics_model = self.dynamics_model
        _LOCAL.device = device
        _LOCAL.dtype = dtype
        _LOCAL.max_depth = self.max_depth

    @staticmethod
    def verify_batch(batch):
        """Verify a batch of samples using linear bound propagation. Called by the executor."""
        return CBFVerificationStrategy._verify_batch_linbndprop(
            batch,
            _LOCAL.dynamics_model,
            _LOCAL.network_linearizer,
            _LOCAL.torch_model,
            _LOCAL.device,
            _LOCAL.dtype,
            max_depth=_LOCAL.max_depth,
        )

    @staticmethod
    @torch.no_grad()
    def _verify_batch_linbndprop(
        batch,
        dynamics_model,
        network_linearizer,
        torch_model,
        device,
        dtype,
        min_volume=1e-8,
        find_counterexample=False,
        max_depth=None,
    ):
        """
        Helper method to verify a batch of samples using the paper's method.

        This implements the CBF verification theory with consistency checks:
        - Safe regions (h >= 0): Must be in true safe set AND verify CBF condition ∇h·f + α(h) >= 0
        - Unsafe regions (h < 0): Must be in true unsafe set AND h < 0 (automatically satisfied)
        - Boundary regions: Must split and verify separately

        Critical: We verify that h(x) ≥ 0 only when actually in the true safe set,
        and h(x) < 0 only when actually in the true unsafe set.
        """
        start_time = time.time()

        results = [None for _ in range(len(batch))]

        to_check_cbf_cond = []
        reason = []

        # Compute network bounds once for this batch
        # This uses the public API which caches bounds and avoids redundant computation
        network_linearizer.compute_network_bounds(batch)

        # sample_idx is the index of the sample within the batch.
        for sample_idx, sample in enumerate(batch):
            # Extract barrier function bounds h(x) using the public getter method
            h_min, h_max = network_linearizer.get_network_output_bounds(sample_idx)

            # Check consistency between barrier function and true safe/unsafe sets

            # Case 1: h(x) < 0 everywhere on this region (barrier indicates everywhere unsafe)
            if h_max < 0:
                # Region is correctly classified as unsafe
                results[sample_idx] = SampleResultSAT(sample, start_time, result_type="unsafe_region")

            # Case 2: the region is contains parts of the unsafe set
            elif unsafe_region(sample, dynamics_model, require_complete_containment=False):
                # TODO: if sample is _contained_ in the unsafe set, then we should pick
                # the largest lower bound (i.e. there exists x in region s.t. h(x) >= 0, which is a violation)
                
                if h_min >= 0: 
                    # This is a VIOLATION: h(x) >= 0 but region contains the true unsafe set
                    counterexample = sample.center
                    results[sample_idx] = SampleResultUNSAT(sample, start_time, [counterexample], result_type="h_positive_in_unsafe")
                else:
                    CBFVerificationStrategy._handle_split(
                        sample=sample,
                        start_time=start_time,
                        results=results,
                        sample_idx=sample_idx,
                        min_volume=min_volume,
                        split_type="case_1_boundary_unsafe",
                        unsat_type="unsafe_cannot_split",
                        max_depth=max_depth,
                        depth_limit_type="depth_limit_reached_unsafe",
                    )
            # Case 3: h(x) >= 0 somewhere on this region (barrier indicates somewhere safe)
            else:
                # Region is classified as safe thus we need the to verify CBF condition
                to_check_cbf_cond.append(sample_idx)
                reason.append("case_2")

        if len(to_check_cbf_cond) == 0:
            return results

        # Pre-compute Jacobian bounds for CBF condition verification
        network_linearizer.keep_indices(to_check_cbf_cond)
        network_linearizer.compute_partial_derivative_bounds(input_idx=None, output_idx=0)
        subbatch = [batch[i] for i in to_check_cbf_cond]

        cbf_verified = torch.ones(len(subbatch), dtype=torch.bool, device=device)
        current_indices = torch.arange(len(subbatch), device=device)  # Track mapping to original batch

        eta_values_list = list(itertools.product([0.5], repeat=2))
        for iteration_idx, eta in enumerate(eta_values_list):
            if len(current_indices) == 0:
                break

            if iteration_idx > 0:
                # Prepare subbatch for current indices
                subbatch_to_check = [subbatch[i.item()] for i in current_indices]
            else:
                subbatch_to_check = subbatch

            eta_verified, counter_verified, _, _ = _verify_cbf_condition_affine(
                subbatch_to_check, dynamics_model, network_linearizer, device, dtype, eta=eta, find_counterexample=find_counterexample
            )

            # Mark failures in original cbf_verified array
            failed_in_current = ~eta_verified
            original_failed_indices = current_indices[failed_in_current]
            cbf_verified[original_failed_indices] = False

            # Mark successes in counter_verified array
            succeeded_in_current = counter_verified
            original_succeeded_indices = current_indices[succeeded_in_current]
            cbf_verified[original_succeeded_indices] = True

            # Update current_indices to only include verified ones
            current_indices = current_indices[eta_verified]

            # Filter network_linearizer for next iteration (keep only verified)
            if len(current_indices) > 0 and iteration_idx < len(eta_values_list) - 1:
                # Find indices relative to current network_linearizer state
                keep_mask = eta_verified
                network_linearizer.keep_indices(keep_mask.nonzero(as_tuple=True)[0], include_partial_deriv_bounds=True)

        for subsample_idx, sample_idx in enumerate(to_check_cbf_cond):
            sample = batch[sample_idx]

            if cbf_verified[subsample_idx]:
                results[sample_idx] = SampleResultSAT(sample, start_time, result_type="safe_cbf_verified")
                continue
            elif find_counterexample and counter_verified[subsample_idx]:
                results[sample_idx] = SampleResultUNSAT(sample, start_time, [sample.center], result_type="safe_cbf_violation")
                continue
            else:
                CBFVerificationStrategy._handle_split(
                    sample=sample,
                    start_time=start_time,
                    results=results,
                    sample_idx=sample_idx,
                    min_volume=min_volume,
                    split_type="case_2_cbf_failure" if reason[subsample_idx] == "case_2" else "case_3_fallback",
                    unsat_type="safe_cbf_violation",
                    max_depth=max_depth,
                    depth_limit_type="depth_limit_reached_safe",
                )

                continue

        return results


def _verify_cbf_condition_affine(batch, dynamics_model, network_linearizer, device, dtype, eta=(0.5, 0.5), find_counterexample=False):
    """Verify the CBF condition using PyTorch for NN bounds and NumPy for dynamics."""
    n = dynamics_model.input_dim
    m = dynamics_model.control_dim

    try:
        f_affine_bounds, g_affine_bounds = _compute_dynamics_bounds_taylor(batch, dynamics_model, device, dtype)
    except ValueError:
        """Failed to compute dynamics bounds - return all False."""
        return torch.zeros(len(batch), dtype=torch.bool, device=device), None

    # 1. Get Jacobian J(x) affine bounds from linear bound propagation (as PyTorch tensors)
    # Get all partial derivatives at once for output 0 (barrier function)
    A_L, b_L, A_U, b_U = network_linearizer.get_partial_derivative_bounds()
    # print(f"DEBUG A_L shape: {A_L.shape}, b_L shape: {b_L.shape}, A_U shape: {A_U.shape}, b_U shape: {b_U.shape}")
    # DEBUG A_L shape: torch.Size([106, 2, 2]), b_L shape: torch.Size([106, 2]), A_U shape: torch.Size([106, 2, 2]), b_U shape: torch.Size([106, 2])
    J_affine_L, J_affine_U = (A_L, b_L), (A_U, b_U)

    

    # 2. Get Dynamics f(x) and g(x) affine bounds using Taylor linearization (as NumPy arrays)
    f_affine_L, f_affine_U = f_affine_bounds

    # 3. Compute Lower Bound for Drift Term: J(x)f(x)
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

    # print(f"DEBUG M_D shape: {M_D.shape}, c_D shape: {c_D.shape}")
    # torch.Size([11, 2, 2]), c_D shape: torch.Size([11, 2])

    M_D, c_D = M_D.sum(dim=-2), c_D.sum(dim=-1)  # Sum over all state dimensions

    # 5. Compute Lower Bound for Class-K Term using already-computed barrier bounds
    # Extract h_min from the network bounds (already computed by compute_partial_derivative_bounds)
    (A_L, a_L), _ = network_linearizer.get_network_linear_bounds()
    # print(f"DEBUG A_L shape: {A_L.shape}, a_L shape: {a_L.shape}")
    # DEBUG A_L shape: torch.Size([23, 1, 2]), a_L shape: torch.Size([23, 1])
    alpha_A_L = dynamics_model.alpha_function(A_L[..., 0, :])
    alpha_a_L = dynamics_model.alpha_function(a_L[..., 0])

    M_total, c_total = M_D + alpha_A_L, c_D + alpha_a_L

    if m > 0:
        # g_affine [0].shape i [batch, control, nout, nin]
        # g_affine [0].shape i [batch, control, nout]
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
        v_L_min, v_L_max = _batched_get_affine_function_bounds(v_affine_L, batch, device=device, dtype=dtype)  # Get interval bounds on v(x)

        u_min, u_max = torch.tensor(dynamics_model.u_min, device=device, dtype=dtype), torch.tensor(dynamics_model.u_max, device=device, dtype=dtype)

        M_v_L_u_min, c_v_L_u_min = M_v_L * u_min.unsqueeze(-1), c_v_L * u_min
        M_v_L_u_max, c_v_L_u_max = M_v_L * u_max.unsqueeze(-1), c_v_L * u_max

        for sample_idx, sample in enumerate(batch):
            # 4. Compute Lower Bound for Control Term: sup_u J(x)g(x)u
            M_C = torch.zeros(n, device=device, dtype=dtype)
            c_C = torch.tensor(0.0, device=device, dtype=dtype)
            if m > 0:
                v_Lsample_min = v_L_min[sample_idx]
                v_Lsample_max = v_L_max[sample_idx]

                pos_mask = v_Lsample_min >= 0
                if pos_mask.any():
                    M_C += (M_v_L_u_max[sample_idx, pos_mask]).sum(dim=0)
                    c_C += (c_v_L_u_max[sample_idx, pos_mask]).sum()

                neg_mask = v_Lsample_max <= 0
                if neg_mask.any():
                    M_C += (M_v_L_u_min[sample_idx, neg_mask]).sum(dim=0)
                    c_C += (c_v_L_u_min[sample_idx, neg_mask]).sum()

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

            # 6. Final affine lower bound
            M_total[sample_idx] += M_C
            c_total[sample_idx] += c_C

    #     print(f"DEBUG M_total.unsqueeze(1) shape: {M_total.unsqueeze(1).shape}")
    #     print(f"DEBUG c_total.unsqueeze(1) shape: {c_total.unsqueeze(1).shape}")
    # DEBUG M_total.unsqueeze(1) shape: torch.Size([23, 1, 2])
    # DEBUG c_total.unsqueeze(1) shape: torch.Size([23, 1])

    # 7. Find minimum over the hyper-rectangle
    min_L, _ = _batched_get_affine_function_bounds((M_total.unsqueeze(1), c_total.unsqueeze(1)), batch, device=device, dtype=dtype)
    min_L = min_L.squeeze(-1)
    satisfaction = min_L >= -1e-12

    if find_counterexample:
        # Compute upper bound of CBF condition
        M_D_U, c_D_U = _batched_compute_mccormick_product_upper_bound(
            J_affine_L, J_affine_U, f_affine_L, f_affine_U, batch, nu=eta_drift, device=device, dtype=dtype
        )
        M_D_U, c_D_U = M_D_U.sum(dim=-2), c_D_U.sum(dim=-1)

        (_, a_U), (A_U, _) = network_linearizer.get_network_linear_bounds()
        alpha_A_U = dynamics_model.alpha_function(A_U[..., 0, :])
        alpha_a_U = dynamics_model.alpha_function(a_U[..., 0])

        M_total_U, c_total_U = M_D_U + alpha_A_U, c_D_U + alpha_a_U

        if m > 0:
            M_v_U, c_v_U = _batched_compute_mccormick_product_upper_bound(
                J_affine_L, J_affine_U, g_affine_L, g_affine_U, batch, nu=eta_control_L, device=device, dtype=dtype
            )
            M_v_U, c_v_U = M_v_U.sum(dim=-2), c_v_U.sum(dim=-1)
            v_affine_U = (M_v_U, c_v_U)
            _, v_U_max = _batched_get_affine_function_bounds(v_affine_L, batch, v_affine_U, device=device, dtype=dtype)

            M_v_U_u_min, c_v_U_u_min = M_v_U * u_min.unsqueeze(-1), c_v_U * u_min
            M_v_U_u_max, c_v_U_u_max = M_v_U * u_max.unsqueeze(-1), c_v_U * u_max

            for sample_idx, sample in enumerate(batch):
                M_C_U = torch.zeros(n, device=device, dtype=dtype)
                c_C_U = torch.tensor(0.0, device=device, dtype=dtype)
                if m > 0:
                    v_Usample_max = v_U_max[sample_idx]
                    pos_mask = v_L_min[sample_idx] >= 0
                    if pos_mask.any():
                        M_C_U += (M_v_U_u_max[sample_idx, pos_mask]).sum(dim=0)
                        c_C_U += (c_v_U_u_max[sample_idx, pos_mask]).sum()
                    neg_mask = v_Usample_max <= 0
                    if neg_mask.any():
                        M_C_U += (M_v_U_u_min[sample_idx, neg_mask]).sum(dim=0)
                        c_C_U += (c_v_U_u_min[sample_idx, neg_mask]).sum()
                    mixed_mask = ~(pos_mask | neg_mask)
                    if mixed_mask.any():
                        _, v_u_min_b_U = _vectorized_get_affine_function_bounds(
                            (M_v_L_u_min[sample_idx, mixed_mask], c_v_L_u_min[sample_idx, mixed_mask]),
                            sample,
                            (M_v_U_u_min[sample_idx, mixed_mask], c_v_U_u_min[sample_idx, mixed_mask]),
                            device=device,
                            dtype=dtype,
                        )
                        _, v_u_max_b_U = _vectorized_get_affine_function_bounds(
                            (M_v_L_u_max[sample_idx, mixed_mask], c_v_L_u_max[sample_idx, mixed_mask]),
                            sample,
                            (M_v_U_u_max[sample_idx, mixed_mask], c_v_U_u_max[sample_idx, mixed_mask]),
                            device=device,
                            dtype=dtype,
                        )
                        c_C_U += torch.maximum(v_u_min_b_U, v_u_max_b_U).sum()
                M_total_U[sample_idx] += M_C_U
                c_total_U[sample_idx] += c_C_U

        _, max_U = _batched_get_affine_function_bounds(
            (M_total.unsqueeze(1), c_total.unsqueeze(1)), batch, (M_total_U.unsqueeze(1), c_total_U.unsqueeze(1)), device=device, dtype=dtype
        )
        max_U = max_U.squeeze(-1)
        counterexample = max_U < 0
        return satisfaction, counterexample, min_L, max_U

    return satisfaction, torch.zeros_like(satisfaction), min_L, None


def _get_affine_function_bounds(affine_L, region, affine_U=None, device="cpu", dtype=torch.float64):
    """Computes min/max of a Torch affine function over a region (hyperrectangular or simplicial)."""
    (A, b) = affine_L

    if isinstance(region, HyperrectangularRegion):
        # For hyperrectangular regions, use center and radius
        center = torch.tensor(region.center_point, device=device, dtype=dtype)
        radius = torch.tensor(region.radius_vec, device=device, dtype=dtype)

        A_abs = torch.abs(A)
        lower_b = b + torch.dot(A, center) - torch.dot(A_abs, radius)

        if affine_U is not None:
            (A_U, b_U) = affine_U
            A_U_abs = torch.abs(A_U)
            upper_b = b_U + torch.dot(A_U, center) + torch.dot(A_U_abs, radius)
        else:
            upper_b = b + torch.dot(A, center) + torch.dot(A_abs, radius)

    elif isinstance(region, SimplicialRegion):
        # For simplicial regions, evaluate affine function at all vertices
        vertices = torch.tensor(region.vertices, device=device, dtype=dtype)

        # Evaluate lower bound function at all vertices
        values_L = torch.matmul(vertices, A) + b
        lower_b = torch.min(values_L)

        if affine_U is not None:
            (A_U, b_U) = affine_U
            values_U = torch.matmul(vertices, A_U) + b_U
            upper_b = torch.max(values_U)
        else:
            upper_b = torch.max(values_L)
    else:
        raise TypeError(f"Unsupported region type: {type(region)}. Expected HyperrectangularRegion or SimplicialRegion.")

    return lower_b.item(), upper_b.item()


def _vectorized_compute_mccormick_product_lower_bound(affine1_L, affine1_U, affine2_L, affine2_U, region, eta, device, dtype):
    """Computes McCormick lower bound for (Torch affine) * (Torch affine)."""
    y1_min, y1_max = _vectorized_get_affine_function_bounds(affine1_L, region, affine1_U, device, dtype)
    y2_min, y2_max = _vectorized_get_affine_function_bounds(affine2_L, region, affine2_U, device, dtype)

    (A1_L, b1_L), (A1_U, b1_U) = affine1_L, affine1_U
    (A2_L, b2_L), (A2_U, b2_U) = affine2_L, affine2_U

    C1 = eta * y1_min + (1 - eta) * y1_max
    C2 = eta * y2_min + (1 - eta) * y2_max
    const_part = -(eta * y1_min * y2_min + (1 - eta) * y1_max * y2_max)

    C1_pos, C1_neg = C1.clamp(min=0), C1.clamp(max=0)
    C2_pos, C2_neg = C2.clamp(min=0), C2.clamp(max=0)

    M = C1_pos.unsqueeze(-1) * A2_L + C1_neg.unsqueeze(-1) * A2_U + C2_pos.unsqueeze(-1) * A1_L + C2_neg.unsqueeze(-1) * A1_U
    c = C1_pos * b2_L + C1_neg * b2_U + C2_pos * b1_L + C2_neg * b1_U + const_part
    return M, c


def _vectorized_compute_mccormick_product_upper_bound(affine1_L, affine1_U, affine2_L, affine2_U, region, nu, device, dtype):
    """Computes McCormick upper bound for (Torch affine) * (Torch affine)."""
    y1_min, y1_max = _vectorized_get_affine_function_bounds(affine1_L, region, affine1_U, device, dtype)
    y2_min, y2_max = _vectorized_get_affine_function_bounds(affine2_L, region, affine2_U, device, dtype)

    (A1_L, b1_L), (A1_U, b1_U) = affine1_L, affine1_U
    (A2_L, b2_L), (A2_U, b2_U) = affine2_L, affine2_U

    C1_v = nu * y1_max + (1 - nu) * y1_min
    C2_v = nu * y2_min + (1 - nu) * y2_max
    const_part_v = -(nu * y1_max * y2_min + (1 - nu) * y1_min * y2_max)

    C1_v_pos, C1_v_neg = C1_v.clamp(min=0), C1_v.clamp(max=0)
    C2_v_pos, C2_v_neg = C2_v.clamp(min=0), C2_v.clamp(max=0)

    M = C1_v_pos.unsqueeze(-1) * A2_U + C1_v_neg.unsqueeze(-1) * A2_L + C2_v_pos.unsqueeze(-1) * A1_U + C2_v_neg.unsqueeze(-1) * A1_L
    c = C1_v_pos * b2_U + C1_v_neg * b2_L + C2_v_pos * b1_U + C2_v_neg * b1_L + const_part_v
    return M, c


def _vectorized_get_affine_function_bounds(affine_L, region, affine_U=None, device="cpu", dtype=torch.float64):
    """Computes min/max of a Torch affine function over a region (hyperrectangular or simplicial)."""
    (A, b) = affine_L

    if isinstance(region, HyperrectangularRegion):
        # For hyperrectangular regions, use center and radius
        center = torch.tensor(region.center_point, device=device, dtype=dtype)
        radius = torch.tensor(region.radius_vec, device=device, dtype=dtype)

        A_abs = torch.abs(A)
        lower_b = b + A @ center - A_abs @ radius

        if affine_U is not None:
            (A_U, b_U) = affine_U
            A_U_abs = torch.abs(A_U)
            upper_b = b_U + A_U @ center + A_U_abs @ radius
        else:
            upper_b = b + A @ center + A_abs @ radius

    elif isinstance(region, SimplicialRegion):
        # For simplicial regions, evaluate affine function at all vertices
        vertices = torch.tensor(region.vertices, device=device, dtype=dtype)

        # Evaluate lower bound function at all vertices
        values_L = A @ vertices.T
        lower_b = torch.min(values_L, dim=-1).values + b

        if affine_U is not None:
            (A_U, b_U) = affine_U
            values_U = A_U @ vertices.T
            upper_b = torch.max(values_U, dim=-1).values + b_U
        else:
            upper_b = torch.max(values_L, dim=-1).values + b
    else:
        raise TypeError(f"Unsupported region type: {type(region)}. Expected HyperrectangularRegion or SimplicialRegion.")

    return lower_b, upper_b


def _batched_compute_mccormick_product_lower_bound(affine1_L, affine1_U, affine2_L, affine2_U, batch, eta, device, dtype):
    """Computes McCormick lower bound for (Torch affine) * (Torch affine)."""
    y1_min, y1_max = _batched_get_affine_function_bounds(affine1_L, batch, affine1_U, device=device, dtype=dtype)
    y2_min, y2_max = _batched_get_affine_function_bounds(affine2_L, batch, affine2_U, device=device, dtype=dtype)

    (A1_L, b1_L), (A1_U, b1_U) = affine1_L, affine1_U
    (A2_L, b2_L), (A2_U, b2_U) = affine2_L, affine2_U

    # DEBUG
    # print(f"  DEBUG McCormick: A1_L={A1_L.shape}, b1_L={b1_L.shape}")
    # print(f"  DEBUG McCormick: A2_L={A2_L.shape}, b2_L={b2_L.shape}")
    # print(f"  DEBUG McCormick: y1_min={y1_min.shape}, y2_min={y2_min.shape}, y2_min.ndim={y2_min.ndim}")

    if y2_min.ndim == 3:
        y1_min = y1_min.unsqueeze(-2)
        y1_max = y1_max.unsqueeze(-2)
        A1_L = A1_L.unsqueeze(-3)
        A1_U = A1_U.unsqueeze(-3)
        b1_L = b1_L.unsqueeze(-2)
        b1_U = b1_U.unsqueeze(-2)

    C1 = eta * y1_min + (1 - eta) * y1_max
    C2 = eta * y2_min + (1 - eta) * y2_max
    const_part = -(eta * y1_min * y2_min + (1 - eta) * y1_max * y2_max)

    C1_pos, C1_neg = C1.clamp(min=0), C1.clamp(max=0)
    C2_pos, C2_neg = C2.clamp(min=0), C2.clamp(max=0)

    M = C1_pos.unsqueeze(-1) * A2_L + C1_neg.unsqueeze(-1) * A2_U + C2_pos.unsqueeze(-1) * A1_L + C2_neg.unsqueeze(-1) * A1_U
    c = C1_pos * b2_L + C1_neg * b2_U + C2_pos * b1_L + C2_neg * b1_U + const_part
    # # DEBUG
    # print(f"  DEBUG McCormick: M={M.shape}, c={c.shape}")
    return M, c


def _batched_compute_mccormick_product_upper_bound(affine1_L, affine1_U, affine2_L, affine2_U, batch, nu, device, dtype):
    """Computes McCormick upper bound for (Torch affine) * (Torch affine)."""
    y1_min, y1_max = _batched_get_affine_function_bounds(affine1_L, batch, affine1_U, device=device, dtype=dtype)
    y2_min, y2_max = _batched_get_affine_function_bounds(affine2_L, batch, affine2_U, device=device, dtype=dtype)

    (A1_L, b1_L), (A1_U, b1_U) = affine1_L, affine1_U
    (A2_L, b2_L), (A2_U, b2_U) = affine2_L, affine2_U

    if y2_min.ndim == 3:
        y1_min = y1_min.unsqueeze(-2)
        y1_max = y1_max.unsqueeze(-2)
        A1_L = A1_L.unsqueeze(-3)
        A1_U = A1_U.unsqueeze(-3)
        b1_L = b1_L.unsqueeze(-2)
        b1_U = b1_U.unsqueeze(-2)

    C1_v = nu * y1_max + (1 - nu) * y1_min
    C2_v = nu * y2_min + (1 - nu) * y2_max
    const_part_v = -(nu * y1_max * y2_min + (1 - nu) * y1_min * y2_max)

    C1_v_pos, C1_v_neg = C1_v.clamp(min=0), C1_v.clamp(max=0)
    C2_v_pos, C2_v_neg = C2_v.clamp(min=0), C2_v.clamp(max=0)

    M = C1_v_pos.unsqueeze(-1) * A2_U + C1_v_neg.unsqueeze(-1) * A2_L + C2_v_pos.unsqueeze(-1) * A1_U + C2_v_neg.unsqueeze(-1) * A1_L
    c = C1_v_pos * b2_U + C1_v_neg * b2_L + C2_v_pos * b1_U + C2_v_neg * b1_L + const_part_v
    return M, c


def _batched_get_affine_function_bounds(affine_L, batch, affine_U=None, device="cpu", dtype=torch.float64):
    """Computes min/max of a Torch affine function over a region (hyperrectangular or simplicial)."""
    (A, b) = affine_L
    # print(f"  DEBUG get_affine: A={A.shape}, b={b.shape}")

    if isinstance(batch[0], HyperrectangularRegion):
        # For hyperrectangular regions, use center and radius
        centers = [torch.tensor(region.center_point, device=device, dtype=dtype) for region in batch]
        centers = torch.stack(centers, dim=0)
        radii = [torch.tensor(region.radius_vec, device=device, dtype=dtype) for region in batch]
        radii = torch.stack(radii, dim=0)

        if A.ndim == 4:
            centers = centers.unsqueeze(-2)  # Probably correct, add control dim (no vertex dim compared to simplicial region)
            radii = radii.unsqueeze(-2)

        A_abs = torch.abs(A)
        lower_b = b + (A @ centers.unsqueeze(-1)).squeeze(-1) - (A_abs @ radii.unsqueeze(-1)).squeeze(-1)

        if affine_U is not None:
            (A_U, b_U) = affine_U
            A_U_abs = torch.abs(A_U)
            upper_b = b_U + (A_U @ centers.unsqueeze(-1)).squeeze(-1) - (A_U_abs @ radii.unsqueeze(-1)).squeeze(-1)
        else:
            upper_b = b + (A @ centers.unsqueeze(-1)).squeeze(-1) + (A_abs @ radii.unsqueeze(-1)).squeeze(-1)

    elif isinstance(batch[0], SimplicialRegion):
        # For simplicial regions, evaluate affine function at all vertices
        vertices = [torch.tensor(region.vertices, device=device, dtype=dtype) for region in batch]
        vertices = torch.stack(vertices, dim=0)

        if A.ndim == 4:
            vertices = vertices.unsqueeze(-3)  # Add control dim

        # Evaluate lower bound function at all vertices
        values_L = A @ vertices.transpose(-2, -1)
        lower_b = torch.min(values_L, dim=-1).values + b

        if affine_U is not None:
            (A_U, b_U) = affine_U
            values_U = A_U @ vertices.transpose(-2, -1)
            upper_b = torch.max(values_U, dim=-1).values + b_U
        else:
            upper_b = torch.max(values_L, dim=-1).values + b
    else:
        raise TypeError(f"Unsupported region type: {type(batch[0])}. Expected HyperrectangularRegion or SimplicialRegion.")

    return lower_b, upper_b


def _compute_dynamics_bounds_taylor(batch, dynamics_model, device="cpu", dtype=torch.float64):
    """Compute dynamics affine bounds using Taylor linearization, returning NumPy arrays."""
    numeric_translator = TorchTranslator(device=device, dtype=dtype)
    taylor_linearizer = TaylorLinearization(dynamics_model, numeric_translator)
    # Create region object of batch for f(x) with output_dim=None - handle both region types
    if isinstance(batch[0], HyperrectangularRegion):
        center_points = torch.stack([torch.tensor(sample.center_point, device=device, dtype=dtype) for sample in batch])
        radius_vecs = torch.stack([torch.tensor(sample.radius_vec, device=device, dtype=dtype) for sample in batch])
        batch = HyperrectangularRegion(center_points, radius_vecs, output_dim=None, numeric_translator=numeric_translator)
    elif isinstance(batch[0], SimplicialRegion):
        vertices = torch.stack([torch.tensor(sample.vertices, device=device, dtype=dtype) for sample in batch])
        batch = SimplicialRegion(vertices, output_dim=None, numeric_translator=numeric_translator)
    else:
        raise TypeError(f"Unsupported region type: {type(batch[0])}")

    f_linearization = taylor_linearizer.linearize_sample(batch)
    (A_L, b_L), (A_U, b_U), _ = f_linearization.first_order_model
    # print(f"  DEBUG Taylor f: A_L={A_L.shape}, b_L={b_L.shape}, A_U={A_U.shape}, b_U={b_U.shape}")
    f_affine_bounds = (A_L, b_L), (A_U, b_U)

    g_affine_bounds = None
    if dynamics_model.control_dim > 0:

        class GDynamics:
            def __init__(self, original_dynamics):
                self.original_dynamics = original_dynamics
                self.input_dim = original_dynamics.input_dim

            def compute_dynamics(self, x, translator):
                return self.original_dynamics.compute_g(x, translator)

        g_dynamics = GDynamics(dynamics_model)
        g_linearizer = TaylorLinearization(g_dynamics, numeric_translator)

        g_linearization = g_linearizer.linearize_sample(batch)
        (A_L, b_L), (A_U, b_U), _ = g_linearization.first_order_model
        g_affine_bounds = ((A_L, b_L), (A_U, b_U))

    return f_affine_bounds, g_affine_bounds
