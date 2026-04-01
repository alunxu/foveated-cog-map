"""
Information-theoretic analysis tools.

Computes information gain, detectability maps, and related quantities
for evaluating whether the agent's behaviour is rational from an
information-theoretic perspective.

Based on the framework from Rashidi, Xu et al. (2023).

Member D is responsible for this module.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy


def information_gain(
    prior_uncertainty: np.ndarray,
    posterior_uncertainty: np.ndarray,
) -> np.ndarray:
    """Compute per-location information gain from a fixation.

    IG(x) = H_prior(x) - H_posterior(x)

    Where H is the entropy (uncertainty) at location x.

    Args:
        prior_uncertainty: (H, W) uncertainty before fixation.
        posterior_uncertainty: (H, W) uncertainty after fixation.

    Returns:
        ig_map: (H, W) information gain at each location.
    """
    # Ensure non-negative (uncertainty should be >= 0)
    prior = np.clip(prior_uncertainty, 0, None)
    posterior = np.clip(posterior_uncertainty, 0, None)

    # Information gain = reduction in uncertainty
    ig = prior - posterior
    return ig


def expected_information_gain(
    uncertainty_map: np.ndarray,
    candidate_gaze_positions: list[tuple[float, float]],
    foveation_transform,
    current_gaze_history: list[tuple[float, float]],
) -> list[float]:
    """Compute expected info gain for each candidate gaze position.

    This gives the Bayesian ideal observer's preferred gaze direction.
    The agent's actual gaze can be compared to this reference.

    Args:
        uncertainty_map: (H, W) current uncertainty map.
        candidate_gaze_positions: List of (gx, gy) to evaluate.
        foveation_transform: FoveationTransform instance.
        current_gaze_history: Past gaze positions.

    Returns:
        eig_values: List of expected IG for each candidate.
    """
    eig_values = []

    for gx, gy in candidate_gaze_positions:
        # Hypothetical gaze history with this new fixation
        new_history = current_gaze_history + [(gx, gy)]
        new_uncertainty = foveation_transform.get_uncertainty_map(new_history)

        # Total information gain across all locations
        ig = information_gain(uncertainty_map, new_uncertainty)
        total_ig = ig.sum()
        eig_values.append(total_ig)

    return eig_values


def optimal_gaze_position(
    uncertainty_map: np.ndarray,
    foveation_transform,
    current_gaze_history: list[tuple[float, float]],
    n_candidates: int = 25,
) -> tuple[float, float]:
    """Find the information-gain-maximising gaze position.

    Uses a grid of candidate positions. This is the Bayesian ideal
    observer reference for H3 (gaze-memory coupling).

    Args:
        uncertainty_map: Current uncertainty map.
        foveation_transform: FoveationTransform.
        current_gaze_history: Past fixations.
        n_candidates: Number of candidate positions per axis.

    Returns:
        (gx, gy): Optimal gaze position in pixel coordinates.
    """
    img_size = uncertainty_map.shape[0]
    positions = np.linspace(0, img_size, n_candidates, endpoint=False) + img_size / (2 * n_candidates)

    candidates = [(x, y) for x in positions for y in positions]
    eig = expected_information_gain(
        uncertainty_map, candidates, foveation_transform, current_gaze_history
    )

    best_idx = np.argmax(eig)
    return candidates[best_idx]


def spatial_entropy(occupancy_probs: np.ndarray) -> float:
    """Compute spatial entropy of a probability map.

    Args:
        occupancy_probs: (H, W) probability map (values in [0, 1]).

    Returns:
        H: Total entropy in bits.
    """
    # Flatten and treat each cell as a Bernoulli RV
    p = occupancy_probs.flatten()
    p = np.clip(p, 1e-10, 1 - 1e-10)

    cell_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return cell_entropy.sum()
