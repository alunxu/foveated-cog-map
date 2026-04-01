"""
Gaze-memory coupling analysis.

Tests H3: Is the foveated agent's gaze strategy coupled with its memory content?
- Does the agent look towards regions its memory is uncertain about?
- How does the agent's gaze compare to the Bayesian ideal observer?

Member D is responsible for this module.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from loguru import logger

from src.analysis.information import optimal_gaze_position, expected_information_gain


def gaze_uncertainty_correlation(
    gaze_positions: np.ndarray,
    uncertainty_maps: np.ndarray,
    image_size: int = 64,
) -> dict:
    """Compute correlation between gaze direction and memory uncertainty.

    For H3: does the agent look towards uncertain regions?

    At each timestep, computes the uncertainty at the gaze location
    and correlates it with the mean uncertainty across all locations.

    Args:
        gaze_positions: (T, 2) gaze (x, y) in pixel coords.
        uncertainty_maps: (T, H, W) uncertainty at each timestep.
        image_size: Image size.

    Returns:
        results: dict with correlation metrics.
    """
    T = len(gaze_positions)
    gaze_uncertainties = np.zeros(T)
    mean_uncertainties = np.zeros(T)

    for t in range(T):
        gx, gy = gaze_positions[t]
        gx_int = int(np.clip(gx, 0, image_size - 1))
        gy_int = int(np.clip(gy, 0, image_size - 1))

        gaze_uncertainties[t] = uncertainty_maps[t, gy_int, gx_int]
        mean_uncertainties[t] = uncertainty_maps[t].mean()

    # Does gaze target high-uncertainty regions?
    rho, p_value = spearmanr(gaze_uncertainties, mean_uncertainties)

    # Alternative: is gaze uncertainty higher than average?
    ratio = gaze_uncertainties.mean() / (mean_uncertainties.mean() + 1e-8)

    return {
        "spearman_rho": rho,
        "p_value": p_value,
        "gaze_uncertainty_mean": gaze_uncertainties.mean(),
        "overall_uncertainty_mean": mean_uncertainties.mean(),
        "gaze_to_mean_ratio": ratio,
        # ratio > 1 → agent preferentially looks at uncertain regions
    }


def gaze_optimality_score(
    actual_gaze: np.ndarray,
    uncertainty_maps: np.ndarray,
    foveation_transform,
    gaze_histories: list[list[tuple]],
    n_samples: int = 100,
) -> dict:
    """Compare agent's gaze to Bayesian ideal observer.

    Computes what fraction of the ideal observer's information gain
    the agent actually achieves.

    Args:
        actual_gaze: (T, 2) actual gaze positions.
        uncertainty_maps: (T, H, W) uncertainty maps.
        foveation_transform: FoveationTransform instance.
        gaze_histories: Cumulative gaze histories.
        n_samples: Number of timesteps to sample (expensive computation).

    Returns:
        results: dict with optimality metrics.
    """
    T = len(actual_gaze)
    sample_indices = np.random.choice(T, size=min(n_samples, T), replace=False)

    actual_igs = []
    optimal_igs = []
    gaze_distances = []

    for t in sample_indices:
        unc_map = uncertainty_maps[t]
        history = gaze_histories[t] if t < len(gaze_histories) else []

        # Agent's actual information gain
        actual_ig = expected_information_gain(
            unc_map, [tuple(actual_gaze[t])], foveation_transform, history
        )[0]
        actual_igs.append(actual_ig)

        # Optimal gaze position
        opt_gaze = optimal_gaze_position(unc_map, foveation_transform, history)
        opt_ig = expected_information_gain(
            unc_map, [opt_gaze], foveation_transform, history
        )[0]
        optimal_igs.append(opt_ig)

        # Distance between actual and optimal gaze
        dist = np.sqrt(
            (actual_gaze[t, 0] - opt_gaze[0]) ** 2 +
            (actual_gaze[t, 1] - opt_gaze[1]) ** 2
        )
        gaze_distances.append(dist)

    actual_igs = np.array(actual_igs)
    optimal_igs = np.array(optimal_igs)
    gaze_distances = np.array(gaze_distances)

    # Optimality ratio: how much of the ideal IG does the agent achieve?
    optimality_ratio = actual_igs.sum() / (optimal_igs.sum() + 1e-8)

    return {
        "optimality_ratio": optimality_ratio,
        "mean_actual_ig": actual_igs.mean(),
        "mean_optimal_ig": optimal_igs.mean(),
        "mean_gaze_distance": gaze_distances.mean(),
        "median_gaze_distance": np.median(gaze_distances),
    }
