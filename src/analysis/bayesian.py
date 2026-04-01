"""
Bayesian ideal observer reference model.

Provides the analytically-derived optimal gaze policy under the
assumption that the agent seeks to maximise information gain.
Used as a normative benchmark for evaluating the learned agent.

Based on Radulescu et al. (2022) and Rashidi, Xu et al. (2023).

Member D is responsible for this module.
"""

import numpy as np
from src.envs.foveation import FoveationTransform


class BayesianIdealObserver:
    """Bayesian ideal observer for optimal gaze control.

    Given perfect knowledge of the environment layout and
    foveation properties, computes the information-gain-maximising
    gaze sequence. This serves as an upper bound for the learned
    agent's gaze strategy.

    Args:
        foveation_transform: FoveationTransform instance.
        image_size: Observation image size.
    """

    def __init__(self, foveation_transform: FoveationTransform, image_size: int = 64):
        self.foveation = foveation_transform
        self.image_size = image_size
        self.gaze_history = []

    def reset(self):
        """Reset for a new episode."""
        self.gaze_history = []

    def get_optimal_gaze(self, n_candidates: int = 25) -> tuple[float, float]:
        """Compute the next optimal gaze position.

        Uses a grid search over candidate positions to find the
        fixation that maximally reduces uncertainty.

        Returns:
            (gx, gy): Optimal gaze position.
        """
        from src.analysis.information import optimal_gaze_position

        uncertainty = self.foveation.get_uncertainty_map(self.gaze_history)
        gaze = optimal_gaze_position(
            uncertainty, self.foveation, self.gaze_history, n_candidates
        )
        self.gaze_history.append(gaze)
        return gaze

    def get_uncertainty_map(self) -> np.ndarray:
        """Get current uncertainty map."""
        return self.foveation.get_uncertainty_map(self.gaze_history)

    def compute_optimal_sequence(self, n_fixations: int, n_candidates: int = 25) -> list[tuple]:
        """Compute a full sequence of optimal fixations.

        Args:
            n_fixations: Number of fixations to plan.
            n_candidates: Grid resolution for search.

        Returns:
            sequence: List of (gx, gy) positions.
        """
        self.reset()
        return [self.get_optimal_gaze(n_candidates) for _ in range(n_fixations)]
