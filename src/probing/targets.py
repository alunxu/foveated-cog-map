"""
Probe target computation.

Computes ground-truth targets for each probe from rollout data:
- Occupancy grid (flattened)
- Target position
- Foveation history (which regions were foveated vs. peripheral)
- Perceptual uncertainty map

Member C is responsible for this module.
"""

import numpy as np
from src.envs.foveation import FoveationTransform


def compute_occupancy_targets(ground_truth_data: list[dict]) -> np.ndarray:
    """Extract flattened occupancy grids as probe targets.

    Args:
        ground_truth_data: List of ground truth dicts from rollout buffer.

    Returns:
        targets: (N, grid_h * grid_w) binary occupancy.
    """
    targets = []
    for gt in ground_truth_data:
        occ = gt["occupancy_grid"]
        targets.append(occ.flatten())
    return np.stack(targets)


def compute_target_location_targets(ground_truth_data: list[dict]) -> np.ndarray:
    """Extract goal positions as probe targets.

    Args:
        ground_truth_data: List of ground truth dicts.

    Returns:
        targets: (N, 2) goal (x, y) positions.
    """
    targets = []
    for gt in ground_truth_data:
        pos = gt.get("target_pos", (0, 0))
        targets.append(list(pos) if pos is not None else [0, 0])
    return np.array(targets, dtype=np.float32)


def compute_foveation_history_targets(
    gaze_histories: list[list[tuple[float, float]]],
    grid_size: tuple[int, int],
    fovea_radius: int = 8,
    image_size: int = 64,
) -> np.ndarray:
    """Compute per-cell binary label: was this cell ever foveated?

    Maps pixel-space gaze positions to grid cells and labels each
    cell as foveated (1) or peripheral-only (0).

    Args:
        gaze_histories: List of gaze position histories per timestep.
        grid_size: (h, w) of the environment grid.
        fovea_radius: Foveal region radius in pixels.
        image_size: Image size in pixels.

    Returns:
        targets: (N, grid_h * grid_w) binary labels.
    """
    gh, gw = grid_size
    cell_h = image_size / gh
    cell_w = image_size / gw

    targets = []
    for gaze_history in gaze_histories:
        foveated = np.zeros((gh, gw), dtype=np.float32)
        for gx, gy in gaze_history:
            # Mark grid cells within foveal radius as foveated
            for j in range(gh):
                for i in range(gw):
                    cx = (i + 0.5) * cell_w
                    cy = (j + 0.5) * cell_h
                    dist = np.sqrt((cx - gx) ** 2 + (cy - gy) ** 2)
                    if dist <= fovea_radius:
                        foveated[j, i] = 1.0
        targets.append(foveated.flatten())

    return np.stack(targets)


def compute_uncertainty_targets(
    gaze_histories: list[list[tuple[float, float]]],
    foveation_transform: FoveationTransform,
) -> np.ndarray:
    """Compute per-pixel uncertainty maps as probe targets.

    Uses the foveation transform's uncertainty computation.
    Tests H2: does the hidden state encode *confidence*?

    Args:
        gaze_histories: Gaze histories per timestep.
        foveation_transform: FoveationTransform instance.

    Returns:
        targets: (N, image_size * image_size) uncertainty values.
    """
    targets = []
    for gaze_history in gaze_histories:
        uncertainty = foveation_transform.get_uncertainty_map(gaze_history)
        targets.append(uncertainty.flatten())
    return np.stack(targets)
