import time
from queue import LifoQueue

from tqdm import tqdm  # Added tqdm for progress tracking

from .stats import Statistics


class SinglethreadExecutor:
    def __init__(self):
        self.queue = LifoQueue()

    def gather_batch(self, batch_size):
        batch = []
        for _ in range(batch_size):
            if not self.queue.empty():
                batch.append(self.queue.get())
            else:
                break
        return batch

    def execute(self, initializer, process_batch, aggregate, samples, batch_size=1000, plotter=None, use_wandb=False):
        initializer()

        agg = None
        statistics = Statistics(samples)

        start_time = None

        # Import wandb only if needed
        if use_wandb:
            import wandb

            log_interval = 100  # Log every N samples processed
            next_log_time = log_interval
            viz_log_interval = 5000  # Log visualization every N samples (less frequent)
            next_viz_time = viz_log_interval

        # Use a LifoQueue to achieve DFS (Depth-First Search)-like behavior.
        # For a single-threaded executor, this is true DFS, but for a multi-threaded
        # executor, it depends on the order results are available.
        self.queue = LifoQueue()
        for sample in samples:
            self.queue.put(sample)

        samples_processed = 0
        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not self.queue.empty():
                batch = self.gather_batch(batch_size)

                # Execute the batches
                results = process_batch(batch)

                for result in results:
                    # Take earliest start time from all futures.
                    # This is to subtract the process spawn time
                    # from the computation time.
                    if start_time is None:
                        start_time = result.start_time
                    else:
                        start_time = min(start_time, result.start_time)

                    # Update statistics
                    statistics.add_sample(result)

                    # Update visualization if plotter is provided
                    if result.isleaf() and plotter is not None:
                        plotter.update_figure(result)

                    # Store results however caller wants
                    agg = aggregate(agg, result)

                    # Add new results to the queue
                    if result.hasnewsamples():
                        # Get the new samples
                        new_samples = result.newsamples()

                        # Put the new samples back into the queue
                        for new_sample in new_samples:
                            self.queue.put(new_sample)

                # Update the progress bar
                pbar.update(len(results))
                pbar.set_description_str(
                    f"Overall Progress (remaining samples: {self.queue.qsize()}, "
                    f"certified: {statistics.get_certified_percentage():.4f}%, "
                    f"uncertified: {statistics.get_uncertified_percentage():.4f}%)"
                )

                # Log to W&B periodically
                samples_processed += len(results)
                if use_wandb and samples_processed > next_log_time:
                    next_log_time = samples_processed + log_interval
                    # Calculate iterations per second
                    elapsed_time = time.time() - start_time
                    iterations_per_second = samples_processed / elapsed_time if elapsed_time > 0 else 0

                    log_dict = {
                        "verification/certified_percentage": statistics.get_certified_percentage(),
                        "verification/uncertified_percentage": statistics.get_uncertified_percentage(),
                        "verification/remaining_samples": self.queue.qsize(),
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

                    # Add visualization if plotter is available and it's time to log it
                    if plotter is not None and samples_processed > next_viz_time:
                        next_viz_time = samples_processed + viz_log_interval
                        try:
                            log_dict["verification/progress_plot"] = wandb.Image(plotter.get_figure_for_wandb())
                        except Exception as e:
                            print(f"Warning: Failed to log visualization to wandb: {e}")

                    wandb.log(log_dict, step=samples_processed)

        end_time = time.time()
        computation_time = end_time - start_time

        return agg, statistics.get_certified_percentage(), statistics.get_uncertified_percentage(), computation_time
