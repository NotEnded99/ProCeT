import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import LifoQueue

from tqdm import tqdm

from ..cbf import verify_cbf
from ..linearization.linear_derivative_bounds import CrownPartialLinearization
from .stats import Statistics

# The following helper classes are used for dynamically adding tasks to the
# `as_completed` iterator, identical to the multiprocess version.

# Possible future states (for internal use by the futures package).
PENDING = "PENDING"
RUNNING = "RUNNING"
CANCELLED = "CANCELLED"
CANCELLED_AND_NOTIFIED = "CANCELLED_AND_NOTIFIED"
FINISHED = "FINISHED"


class _Waiter:
    """Provides the event that wait() and as_completed() block on."""

    def __init__(self):
        self.event = threading.Event()
        self.finished_futures = []

    def add_result(self, future):
        self.finished_futures.append(future)

    def add_exception(self, future):
        self.finished_futures.append(future)

    def add_cancelled(self, future):
        self.finished_futures.append(future)


class _AsCompletedWaiter(_Waiter):
    """Used by as_completed()."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

    def add_result(self, future):
        with self.lock:
            super().add_result(future)
            self.event.set()

    def add_exception(self, future):
        with self.lock:
            super().add_exception(future)
            self.event.set()

    def add_cancelled(self, future):
        with self.lock:
            super().add_cancelled(future)
            self.event.set()


class _AcquireFutures:
    """A context manager that does an ordered acquire of Future conditions."""

    def __init__(self, futures):
        self.futures = sorted(futures, key=id)

    def __enter__(self):
        for future in self.futures:
            future._condition.acquire()

    def __exit__(self, *args):
        for future in self.futures:
            future._condition.release()


class ExpandableAsCompleted:
    """A wrapper for as_completed that allows for adding new futures dynamically."""

    def __init__(self, fs, timeout=None):
        self.fs = set(fs)
        self.timeout = timeout
        self.waiter = None

    def __len__(self):
        return len(self.fs)

    def add(self, f):
        if self.waiter is None:
            raise RuntimeError("Cannot add futures before calling __iter__()")
        f._waiters.append(self.waiter)
        self.fs.add(f)

    def create_and_install_as_completed_waiter(self):
        waiter = _AsCompletedWaiter()
        for f in self.fs:
            f._waiters.append(waiter)
        self.waiter = waiter

    def yield_finished_futures(self, finished_futures):
        while finished_futures:
            f = finished_futures.pop()
            self.fs.remove(f)
            with f._condition:
                f._waiters.remove(self.waiter)
            yield f

    def as_completed(self):
        if self.timeout is not None:
            end_time = self.timeout + time.monotonic()

        with _AcquireFutures(self.fs):
            finished = {f for f in self.fs if f._state in [CANCELLED_AND_NOTIFIED, FINISHED]}
            self.create_and_install_as_completed_waiter()
        finished = list(finished)

        try:
            yield from self.yield_finished_futures(finished)
            while self.fs:
                if self.timeout is None:
                    wait_timeout = None
                else:
                    wait_timeout = end_time - time.monotonic()
                    if wait_timeout < 0:
                        raise TimeoutError(f"{len(self.fs)} futures unfinished")
                self.waiter.event.wait(wait_timeout)
                with self.waiter.lock:
                    finished = self.waiter.finished_futures
                    self.waiter.finished_futures = []
                    self.waiter.event.clear()
                finished.reverse()
                yield from self.yield_finished_futures(finished)
        finally:
            for f in self.fs:
                with f._condition:
                    if self.waiter in f._waiters:
                        f._waiters.remove(self.waiter)


class MultithreadExecutor:
    """
    An executor that uses a ThreadPoolExecutor to run tasks in parallel,
    optimized for a shared GPU resource.
    """

    def __init__(self, num_workers=None):
        # If num_workers is not provided, use the default of ThreadPoolExecutor
        # which is min(32, (os.process_cpu_count() or 1) + 4)
        self.num_workers = num_workers
        self._thread_local = threading.local()
        self._shared_state = {}

    def _initialize_master(self, initializer_func):
        """
        Runs the user-provided initializer once to load the shared model onto the GPU.
        It hijacks the initializer to capture the model and other configuration.
        """
        temp_ns = verify_cbf.types.SimpleNamespace()
        original_local = getattr(verify_cbf, "_LOCAL", None)
        verify_cbf._LOCAL = temp_ns
        try:
            initializer_func()
            self._shared_state = temp_ns.__dict__
        finally:
            verify_cbf._LOCAL = original_local

    def _worker_task_wrapper(self, process_sample_func, sample):
        """
        A wrapper that runs in each thread. It sets up a thread-local environment
        before calling the user's sample processing function.
        """
        # Initialize the thread on its first task run
        if not hasattr(self._thread_local, "initialized"):
            self._thread_local.state = verify_cbf.types.SimpleNamespace()

            # Copy all shared states (e.g., model, device, config)
            for key, value in self._shared_state.items():
                setattr(self._thread_local.state, key, value)

            # Create a new, thread-local instance of the stateful linearizer
            shared_model = self._shared_state["torch_model"]
            self._thread_local.state.network_linearizer = CrownPartialLinearization(shared_model)
            self._thread_local.initialized = True

        # Set the global _LOCAL to this thread's state before running the task
        original_local = getattr(verify_cbf, "_LOCAL", None)
        verify_cbf._LOCAL = self._thread_local.state
        try:
            return process_sample_func(sample)
        finally:
            # Restore global state to prevent leakage
            verify_cbf._LOCAL = original_local

    def execute(self, initializer, process_sample, aggregate, samples, plotter=None, use_wandb=False):
        overall_start = time.time()
        self._initialize_master(initializer)

        # Determine actual number of workers that will be used
        if self.num_workers is None:
            import os

            # Default ThreadPoolExecutor formula: min(32, (cpu_count or 1) + 4)
            # Using os.cpu_count() which is available in Python 3.4+
            actual_workers = min(32, (os.cpu_count() or 1) + 4)
        else:
            actual_workers = self.num_workers

        print(f"Executor starting with {len(samples)} samples using {actual_workers} threads")

        agg = None
        statistics = Statistics(samples)

        if use_wandb:
            import wandb

            log_interval = 10  # Log every 10 samples processed
            viz_log_interval = 500  # Log visualization every 500 samples

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            executor._work_ids = LifoQueue()
            with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
                futures = [executor.submit(self._worker_task_wrapper, process_sample, s) for s in samples]
                for future in futures:
                    future.add_done_callback(lambda p: pbar.update())

                waiter = ExpandableAsCompleted(futures)
                start_time = None
                samples_processed = 0

                for future in waiter.as_completed():
                    result = future.result()
                    if start_time is None:
                        start_time = result.start_time
                    else:
                        start_time = min(start_time, result.start_time)

                    statistics.add_sample(result)
                    if result.isleaf() and plotter is not None:
                        plotter.update_figure(result)

                    agg = aggregate(agg, result)

                    if result.hasnewsamples():
                        for new_sample in result.newsamples():
                            new_future = executor.submit(self._worker_task_wrapper, process_sample, new_sample)
                            new_future.add_done_callback(lambda p: pbar.update())
                            waiter.add(new_future)

                    pbar.set_description_str(
                        f"Overall Progress (remaining samples: {len(waiter)}, "
                        f"certified: {statistics.get_certified_percentage():.4f}%, "
                        f"uncertified: {statistics.get_uncertified_percentage():.4f}%)"
                    )

                    samples_processed += 1
                    if use_wandb and samples_processed % log_interval == 0:
                        # Calculate iterations per second
                        elapsed_time = time.time() - start_time
                        iterations_per_second = samples_processed / elapsed_time if elapsed_time > 0 else 0

                        log_dict = {
                            "verification/certified_percentage": statistics.get_certified_percentage(),
                            "verification/uncertified_percentage": statistics.get_uncertified_percentage(),
                            "verification/remaining_samples": len(waiter),
                            "verification/iterations_per_second": iterations_per_second,
                        }

                        # Add split statistics if available (for CBF verification)
                        split_stats = statistics.split_stats
                        total_splits = split_stats["total_splits"]
                        if total_splits > 0:
                            split_percentages = statistics.get_split_percentages()
                            log_dict.update(
                                {
                                    "splits/total_splits": total_splits,
                                    "splits/case_2_cbf_failure": split_stats["case_2_cbf_failure"],
                                    "splits/case_3_mixed_unsafe": split_stats["case_3_mixed_unsafe"],
                                    "splits/case_3_fallback": split_stats["case_3_fallback"],
                                    **split_percentages,
                                }
                            )

                        # Add SAT result type statistics
                        sat_stats = statistics.sat_stats
                        total_sat = sat_stats["total_sat"]
                        if total_sat > 0:
                            sat_percentages = statistics.get_sat_percentages()
                            log_dict.update(
                                {
                                    "sat/total_sat": total_sat,
                                    "sat/unsafe_region": sat_stats["unsafe_region"],
                                    "sat/safe_cbf_verified": sat_stats["safe_cbf_verified"],
                                    "sat/mixed_unsafe_only": sat_stats["mixed_unsafe_only"],
                                    **sat_percentages,
                                }
                            )

                        # Add UNSAT result type statistics
                        unsat_stats = statistics.unsat_stats
                        total_unsat = unsat_stats["total_unsat"]
                        if total_unsat > 0:
                            unsat_percentages = statistics.get_unsat_percentages()
                            log_dict.update(
                                {
                                    "unsat/total_unsat": total_unsat,
                                    "unsat/h_positive_in_unsafe": unsat_stats["h_positive_in_unsafe"],
                                    "unsat/safe_cbf_violation": unsat_stats["safe_cbf_violation"],
                                    "unsat/indeterminate_volume_limit": unsat_stats["indeterminate_volume_limit"],
                                    **unsat_percentages,
                                }
                            )

                        if plotter and samples_processed % viz_log_interval == 0:
                            try:
                                log_dict["verification/progress_plot"] = wandb.Image(plotter.get_figure_for_wandb())
                            except Exception as e:
                                print(f"Warning: Failed to log visualization to wandb: {e}")
                        wandb.log(log_dict, step=samples_processed)

        certified_pct = statistics.get_certified_percentage()
        uncertified_pct = statistics.get_uncertified_percentage()
        end_time = time.time()
        computation_time = end_time - start_time if start_time else 0
        total_time = end_time - overall_start

        print(f"Total executor time: {total_time:.3f}s")
        print(f"Actual computation time: {computation_time:.3f}s")
        print(f"Overhead time: {total_time - computation_time:.3f}s")

        return agg, certified_pct, uncertified_pct, computation_time
