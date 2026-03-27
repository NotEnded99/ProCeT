import torch
from bound_propagation import LinearBounds

from ..certification_results import AugmentedSample
from ..regions import HyperrectangularRegion, SimplicialRegion
from ..translators import BoundPropagationTranslator


class CrownLinearization:
    def __init__(self, dynamics):
        """
        Initialize the Crown linearization strategy.

        :param dynamics: The dynamics of the system.
        """
        self.dynamics = dynamics
        self.translator = BoundPropagationTranslator()
        self.traced_model = None

    def linearize(self, samples):
        """
        Linearizes a batch of samples using Taylor expansion.
        """
        if self.traced_model is None:
            x = self.translator.to_format(samples[0].center)
            self.traced_model = self.dynamics.compute_dynamics(x, self.translator)
            if torch.cuda.is_available():
                self.traced_model = self.traced_model.to(torch.device("cuda"))

        # Extract centers and deltas based on region type
        centers = []
        deltas = []

        for sample in samples:
            # Get center (use centroid for all region types)
            centers.append(torch.as_tensor(sample.centroid, dtype=torch.float64))

            # Get delta (size measure) based on region type
            if isinstance(sample, HyperrectangularRegion):
                # HyperrectangularRegion
                deltas.append(torch.as_tensor(sample.radius_vec, dtype=torch.float64))
            elif isinstance(sample, SimplicialRegion):
                # SimplicialRegion - use bounding box extents
                bounds = sample.get_bounds()
                extents = (bounds[:, 1] - bounds[:, 0]) / 2.0
                deltas.append(torch.as_tensor(extents, dtype=torch.float64))
            else:
                raise TypeError(f"Unsupported region type: {type(sample)}. " f"Expected HyperrectangularRegion or SimplicialRegion.")

        centers = torch.stack(centers)
        deltas = torch.stack(deltas)

        if torch.cuda.is_available():
            centers = centers.to(torch.device("cuda"))
            deltas = deltas.to(torch.device("cuda"))

        linear_bounds = self.translator.bound(self.traced_model, centers, deltas)

        A_lower = linear_bounds.lower[0]
        b_lower = linear_bounds.lower[1]

        A_upper = linear_bounds.upper[0]
        b_upper = linear_bounds.upper[1]

        A_gap = A_upper - A_lower
        b_gap = b_upper - b_lower
        lbp_gap = LinearBounds(linear_bounds.region, None, (A_gap, b_gap))
        interval_gap = lbp_gap.concretize()  # Turn linear bounds into interval bounds

        A_lower = A_lower.cpu().numpy()
        b_lower = b_lower.cpu().numpy()
        A_upper = A_upper.cpu().numpy()
        b_upper = b_upper.cpu().numpy()
        max_gap = interval_gap.upper.cpu().numpy()

        def to_augmented_sample(i):
            sample = samples[i]
            j = sample.output_dim
            first_order_model = ((A_lower[i, j], b_lower[i, j]), (A_upper[i, j], b_upper[i, j]), max_gap[i, j].item())
            return AugmentedSample.from_certification_region(sample, first_order_model)

        augmented_samples = [to_augmented_sample(i) for i in range(len(samples))]

        return augmented_samples

    def linearize_sample(self, sample):
        """
        Linearizes a single sample using Taylor expansion.
        """
        return self.linearize([sample])[0]
