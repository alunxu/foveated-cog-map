"""
Probe training and evaluation pipeline.

Orchestrates: load probing data → compute targets → train probes → report.

Member C is responsible for this module.
"""

import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split

from src.probing.probes import (
    OccupancyProbe,
    TargetLocationProbe,
    FoveationHistoryProbe,
    UncertaintyProbe,
)
from src.probing.targets import (
    compute_occupancy_targets,
    compute_target_location_targets,
    compute_foveation_history_targets,
    compute_uncertainty_targets,
)


def run_probing_analysis(
    hidden_states: np.ndarray,
    ground_truth_data: list[dict],
    gaze_histories: list[list[tuple]],
    foveation_transform=None,
    grid_size: tuple = (9, 9),
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Run full probing analysis on collected hidden states.

    Args:
        hidden_states: (N, hidden_size) GRU hidden states.
        ground_truth_data: List of GT dicts from rollout.
        gaze_histories: Gaze histories (empty list for uniform agent).
        foveation_transform: FoveationTransform instance (None for uniform).
        grid_size: Environment grid size.
        test_size: Fraction of data for test set.
        seed: Random seed.

    Returns:
        results: dict mapping probe_name → {train_score, test_score}.
    """
    results = {}

    # Split data
    indices = np.arange(len(hidden_states))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

    h_train, h_test = hidden_states[train_idx], hidden_states[test_idx]

    # --- Probe 1: Occupancy grid ---
    logger.info("Training occupancy probe...")
    occ_targets = compute_occupancy_targets(ground_truth_data)
    probe = OccupancyProbe().fit(h_train, occ_targets[train_idx])
    results["occupancy"] = {
        "train_score": probe.score(h_train, occ_targets[train_idx]),
        "test_score": probe.score(h_test, occ_targets[test_idx]),
    }
    logger.info(f"  Occupancy probe R²: train={results['occupancy']['train_score']:.4f}, "
                f"test={results['occupancy']['test_score']:.4f}")

    # --- Probe 2: Target location ---
    logger.info("Training target location probe...")
    target_targets = compute_target_location_targets(ground_truth_data)
    probe = TargetLocationProbe().fit(h_train, target_targets[train_idx])
    results["target_location"] = {
        "train_score": probe.score(h_train, target_targets[train_idx]),
        "test_score": probe.score(h_test, target_targets[test_idx]),
    }
    logger.info(f"  Target location R²: train={results['target_location']['train_score']:.4f}, "
                f"test={results['target_location']['test_score']:.4f}")

    # --- Probe 3: Foveation history (foveated agent only) ---
    if gaze_histories and len(gaze_histories[0]) > 0:
        logger.info("Training foveation history probe...")
        fov_targets = compute_foveation_history_targets(gaze_histories, grid_size)
        probe = FoveationHistoryProbe().fit(h_train, fov_targets[train_idx])
        results["foveation_history"] = {
            "train_score": probe.score(h_train, fov_targets[train_idx]),
            "test_score": probe.score(h_test, fov_targets[test_idx]),
        }
        logger.info(f"  Foveation history acc: train={results['foveation_history']['train_score']:.4f}, "
                    f"test={results['foveation_history']['test_score']:.4f}")

    # --- Probe 4: Uncertainty map (foveated agent only) ---
    if foveation_transform is not None and gaze_histories:
        logger.info("Training uncertainty probe...")
        unc_targets = compute_uncertainty_targets(gaze_histories, foveation_transform)
        probe = UncertaintyProbe().fit(h_train, unc_targets[train_idx])
        results["uncertainty"] = {
            "train_score": probe.score(h_train, unc_targets[train_idx]),
            "test_score": probe.score(h_test, unc_targets[test_idx]),
        }
        logger.info(f"  Uncertainty probe R²: train={results['uncertainty']['train_score']:.4f}, "
                    f"test={results['uncertainty']['test_score']:.4f}")

    return results
