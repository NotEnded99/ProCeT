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

    def execute(self, initializer, process_batch, aggregate, samples, batch_size=1000, plotter=None):
        initializer()

        agg = None
        statistics = Statistics(samples)

        start_time = None

        # Use a LifoQueue to achieve DFS (Depth-First Search)-like behavior.
        # For a single-threaded executor, this is true DFS, but for a multi-threaded
        # executor, it depends on the order results are available.
        self.queue = LifoQueue()
        for sample in samples:
            self.queue.put(sample)

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

        end_time = time.time()
        computation_time = end_time - start_time

        return agg, statistics.get_certified_percentage(), statistics.get_uncertified_percentage(), computation_time
