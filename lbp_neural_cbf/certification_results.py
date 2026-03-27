import abc

import numpy as np


class AugmentedSample:
    """
    Container that wraps any certification region with linearization information.

    This class provides a unified interface for linearized regions, supporting both
    hyperrectangular and simplicial regions without inheritance constraints.
    """

    def __init__(self, region, first_order_model):
        """
        Initialize an augmented sample.

        Args:
            region: The original certification region (HyperrectangularRegion or SimplicialRegion)
            first_order_model: Linearization information as ((A_lower, b_lower), (A_upper, b_upper), max_gap)
        """
        self.region = region
        self.first_order_model = first_order_model

    @staticmethod
    def from_certification_region(region, first_order_model):
        """
        Create an AugmentedSample from any certification region.

        Args:
            region: Any certification region (hyperrectangular or simplicial)
            first_order_model: Linearization information

        Returns:
            AugmentedSample: Unified container for linearized regions
        """
        return AugmentedSample(region, first_order_model)

    def isfinite(self):
        """Check if the linearization bounds are finite."""
        return (
            np.isfinite(self.first_order_model[0][0]).all()
            and np.isfinite(self.first_order_model[0][1]).all()
            and np.isfinite(self.first_order_model[1][0]).all()
            and np.isfinite(self.first_order_model[1][1]).all()
        )

    # Delegate all region-specific methods to the wrapped region
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped region."""
        # For pickle support, let Python handle its own special methods
        # Only delegate to region for non-dunder attributes and specific methods we know exist
        if name.startswith("__") and name.endswith("__"):
            # Let Python's default behavior handle pickle-related dunder methods
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Safely check if region has the attribute to avoid recursion
        try:
            region = object.__getattribute__(self, "region")
            if hasattr(region, name):
                return getattr(region, name)
        except AttributeError:
            pass

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getstate__(self):
        """Custom pickle support to avoid recursion issues."""
        return {"region": self.region, "first_order_model": self.first_order_model}

    def __setstate__(self, state):
        """Custom unpickle support to avoid recursion issues."""
        self.region = state["region"]
        self.first_order_model = state["first_order_model"]

    def __repr__(self):
        return f"AugmentedSample(region={self.region}, first_order_model={type(self.first_order_model)})"


class SampleResult(abc.ABC):
    def __init__(self, sample, start_time):
        self.sample = sample
        self.start_time = start_time

    @abc.abstractmethod
    def issat(self) -> bool:
        pass

    @abc.abstractmethod
    def isunsat(self) -> bool:
        pass

    def isleaf(self) -> bool:
        return self.issat() or self.isunsat()

    @abc.abstractmethod
    def hasnewsamples(self) -> bool:
        pass

    def newsamples(self):
        raise ValueError("New samples not available for this sample result.")

    @abc.abstractmethod
    def hascounterexamples(self) -> bool:
        pass

    def counterexamples(self):
        raise ValueError("Counterexamples not available for this sample result.")

    def lebesguemeasure(self):
        return self.sample.lebesguemeasure()


class SampleResultSAT(SampleResult):
    def __init__(self, sample, start_time, result_type=None):
        """
        Initialize a SAT result (verification successful).

        Args:
            sample: The sample being verified
            start_time: Timestamp when verification started
            result_type: Optional string indicating why the verification succeeded (for analytics)
                        Common CBF values: 'unsafe_region', 'safe_cbf_verified', 'mixed_unsafe_only'
        """
        super().__init__(sample, start_time)
        self.result_type = result_type

    def issat(self) -> bool:
        return True

    def isunsat(self) -> bool:
        return False

    def hasnewsamples(self) -> bool:
        return False

    def hascounterexamples(self) -> bool:
        return False

    def __repr__(self):
        type_info = f", result_type={self.result_type}" if self.result_type else ""
        return f"SAT: {self.sample}{type_info}"


class SampleResultUNSAT(SampleResult):
    def __init__(self, sample, start_time, counterexamples, result_type=None):
        """
        Initialize an UNSAT result (verification failed - counterexample found).

        Args:
            sample: The sample being verified
            start_time: Timestamp when verification started
            counterexamples: List of counterexamples found
            result_type: Optional string indicating why the verification failed (for analytics)
                        Common CBF values: 'h_positive_in_unsafe', 'safe_cbf_violation', 'indeterminate_volume_limit'
        """
        super().__init__(sample, start_time)
        self._counterexamples = counterexamples
        self.result_type = result_type

    def issat(self) -> bool:
        return False

    def isunsat(self) -> bool:
        return True

    def hasnewsamples(self) -> bool:
        return False

    def hascounterexamples(self) -> bool:
        return True

    def counterexamples(self):
        return self._counterexamples

    def __repr__(self):
        type_info = f", result_type={self.result_type}" if self.result_type else ""
        return f"UNSAT: {self.sample}{type_info}"


class SampleResultMaybe(SampleResult):
    def __init__(self, sample, start_time, new_samples, split_type=None):
        """
        Initialize a MAYBE result (inconclusive verification).

        Args:
            sample: The sample being verified
            start_time: Timestamp when verification started
            new_samples: List of sub-samples to verify after splitting
            split_type: Optional string indicating why the split occurred (for analytics)
                       Common values: 'case_2_cbf_failure', 'case_3_mixed_unsafe', 'case_3_fallback'
        """
        super().__init__(sample, start_time)
        self._new_samples = new_samples
        self.split_type = split_type

    def issat(self) -> bool:
        return False

    def isunsat(self) -> bool:
        return False

    def hasnewsamples(self) -> bool:
        return True

    def newsamples(self):
        return self._new_samples

    def hascounterexamples(self) -> bool:
        return False

    def __repr__(self):
        split_info = f", split_type={self.split_type}" if self.split_type else ""
        return f"Maybe: {self.sample}{split_info}"
