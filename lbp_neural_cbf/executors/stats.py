class Statistics:
    def __init__(self, samples):
        # Calculate the total domain size from initial samples
        # This represents the complete domain we're trying to verify
        self._initial_total_size = sum(sample.lebesguemeasure() for sample in samples)

        # Track the sizes of different leaf region types
        self.certified_domain_size = 0.0
        self.uncertified_domain_size = 0.0
        self.inconclusive_domain_size = 0.0  # MAYBE regions that are leaf nodes

        # Track split statistics for CBF verification analysis
        self.split_stats = {
            "case_1_boundary_unsafe": 0,  # Case 1: h mixed, region intersects unsafe set
            "case_2_cbf_failure": 0,  # Case 2: h >= 0, safe region, but CBF condition fails
            "case_3_mixed_unsafe": 0,  # Case 3: h mixed, region intersects unsafe set
            "case_3_fallback": 0,  # Case 3: h mixed, safe region fallback, CBF fails
            "total_splits": 0,  # Total number of splits
            "total_samples": 0,  # Total samples processed
        }

        # Track SAT (verified) result types for CBF verification
        self.sat_stats = {
            "unsafe_region": 0,  # Region entirely in unsafe set (h < 0 verified)
            "safe_cbf_verified": 0,  # Region in safe set with CBF condition verified
            "mixed_unsafe_only": 0,  # Mixed region but only unsafe portion (h <= 0)
            "total_sat": 0,
        }

        # Track UNSAT (counterexample) result types for CBF verification
        self.unsat_stats = {
            "h_positive_in_unsafe": 0,  # h >= 0 but region intersects unsafe set
            "safe_cbf_violation": 0,  # h >= 0, safe region, but CBF condition violated
            "indeterminate_volume_limit": 0,  # Couldn't split further, treated as violation
            "total_unsat": 0,
        }

    def _get_total_domain_size(self):
        """
        Get total domain size as the sum of all leaf regions.
        This dynamically tracks the actual coverage as regions are processed.
        """
        return self.certified_domain_size + self.uncertified_domain_size + self.inconclusive_domain_size

    def add_sample(self, sample):
        """Add a sample result, but only count leaf nodes in the totals."""
        # Track every sample processed
        self.split_stats["total_samples"] += 1

        # Track split information if available
        if hasattr(sample, "split_type") and sample.split_type:
            self.split_stats["total_splits"] += 1
            if sample.split_type in self.split_stats:
                self.split_stats[sample.split_type] += 1

        # Track leaf results
        if sample.isleaf():
            if sample.issat():
                self.certified_domain_size += sample.lebesguemeasure()
                # Track SAT result type
                self.sat_stats["total_sat"] += 1
                if hasattr(sample, "result_type") and sample.result_type:
                    if sample.result_type in self.sat_stats:
                        self.sat_stats[sample.result_type] += 1

            elif sample.isunsat():
                self.uncertified_domain_size += sample.lebesguemeasure()
                # Track UNSAT result type
                self.unsat_stats["total_unsat"] += 1
                if hasattr(sample, "result_type") and sample.result_type:
                    if sample.result_type in self.unsat_stats:
                        self.unsat_stats[sample.result_type] += 1

            else:
                # This is a MAYBE result that is a leaf (couldn't be split further or hit max depth)
                self.inconclusive_domain_size += sample.lebesguemeasure()

    def get_certified_percentage(self):
        """Get percentage of the original domain that has been certified."""
        if self._initial_total_size == 0:
            return 0.0
        return (self.certified_domain_size / self._initial_total_size) * 100

    def get_uncertified_percentage(self):
        """Get percentage of the original domain that has been proven unsafe."""
        if self._initial_total_size == 0:
            return 0.0
        return (self.uncertified_domain_size / self._initial_total_size) * 100

    def get_inconclusive_percentage(self):
        """Get percentage of the original domain that is inconclusive (leaf MAYBE regions)."""
        if self._initial_total_size == 0:
            return 0.0
        return (self.inconclusive_domain_size / self._initial_total_size) * 100

    def get_stats_summary(self):
        """Get a summary of verification statistics."""
        return (
            f"Domain coverage: "
            f"Certified: {self.get_certified_percentage():.2f}%, "
            f"Uncertified: {self.get_uncertified_percentage():.2f}%, "
            f"Inconclusive: {self.get_inconclusive_percentage():.2f}%"
        )

    def get_split_percentages(self):
        """Get split statistics with percentages."""
        total_splits = self.split_stats["total_splits"]
        if total_splits == 0:
            return {}

        return {
            "case_2_cbf_failure_pct": 100.0 * self.split_stats["case_2_cbf_failure"] / total_splits,
            "case_3_mixed_unsafe_pct": 100.0 * self.split_stats["case_3_mixed_unsafe"] / total_splits,
            "case_3_fallback_pct": 100.0 * self.split_stats["case_3_fallback"] / total_splits,
        }

    def get_sat_percentages(self):
        """Get SAT result type statistics with percentages."""
        total_sat = self.sat_stats["total_sat"]
        if total_sat == 0:
            return {}

        return {
            "unsafe_region_pct": 100.0 * self.sat_stats["unsafe_region"] / total_sat,
            "safe_cbf_verified_pct": 100.0 * self.sat_stats["safe_cbf_verified"] / total_sat,
            "mixed_unsafe_only_pct": 100.0 * self.sat_stats["mixed_unsafe_only"] / total_sat,
        }

    def get_unsat_percentages(self):
        """Get UNSAT result type statistics with percentages."""
        total_unsat = self.unsat_stats["total_unsat"]
        if total_unsat == 0:
            return {}

        return {
            "h_positive_in_unsafe_pct": 100.0 * self.unsat_stats["h_positive_in_unsafe"] / total_unsat,
            "safe_cbf_violation_pct": 100.0 * self.unsat_stats["safe_cbf_violation"] / total_unsat,
            "indeterminate_volume_limit_pct": 100.0 * self.unsat_stats["indeterminate_volume_limit"] / total_unsat,
        }
