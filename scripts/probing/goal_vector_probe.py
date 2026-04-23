"""
A1: Goal-vector probe across 5 conditions.

Tests whether the hidden state encodes the *ego-to-goal vector* (not just
absolute position). Goal-vector = R(-heading) @ (goal_position - position),
i.e. the direction and distance to goal in egocentric coordinates.

Interpretation for paper §4.5 rescue:
- If uniform/blind decode goal-vector near-perfectly and foveated does not,
  our "content differs across conditions" story holds: foveated's memory
  is less goal-specific (explains why it is less shortcut-brittle).
- If foveated also decodes goal-vector near-perfectly, the rescue fails;
  the shortcut ordering must be explained by some other mechanism.

Usage:
    python goal_vector_probe.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out /scratch/izar/wxu/probing_results/goal_vector.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


CONDITIONS = [
    "blind_gibson",
    "uniform_gibson",
    "foveated_gibson",
    "foveated_learned_gibson",
    "matched_gibson",
]


def ego_goal_vector(positions: np.ndarray, goals: np.ndarray, headings: np.ndarray) -> np.ndarray:
    """Rotate the world-frame displacement (goal - pos) by -heading.

    Habitat uses (x, y, z) with y as up. We use (x, z) as the floor plane,
    and heading is the rotation around y-axis (yaw), with heading=0 facing -z.
    Returns (n, 2): forward (along facing direction) and lateral (right).
    """
    dxyz = goals - positions                           # (n, 3)
    dx = dxyz[:, 0]                                    # world-x
    dz = dxyz[:, 2]                                    # world-z
    # Rotate by -heading so the axis aligned with facing direction becomes +y-axis.
    # In Habitat, heading=0 is -z facing; heading increases counter-clockwise around +y.
    cos_h = np.cos(-headings)
    sin_h = np.sin(-headings)
    forward = cos_h * (-dz) - sin_h * dx
    lateral = sin_h * (-dz) + cos_h * dx
    return np.stack([forward, lateral], axis=1)        # (n, 2)


def split_by_episode(ep_ids: np.ndarray, seed: int = 0, test_frac: float = 0.2):
    unique = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(test_frac * len(unique)))
    test_eps = set(unique[:n_test].tolist())
    test_mask = np.isin(ep_ids, list(test_eps))
    return ~test_mask, test_mask


def probe_goal_vector(cond_path: Path) -> dict | None:
    try:
        d = np.load(cond_path, allow_pickle=True)
    except FileNotFoundError:
        return None

    if "positions" not in d or "goal_positions" not in d or "headings" not in d:
        return {"error": "missing positions / goal_positions / headings"}

    X = d["hidden_states"].astype(np.float32)
    positions = d["positions"].astype(np.float32)
    goals = d["goal_positions"].astype(np.float32)
    headings = d["headings"].astype(np.float32)
    ep_ids = d["episode_ids"]

    y = ego_goal_vector(positions, goals, headings)     # (n, 2) forward/lateral
    dist = np.linalg.norm(y, axis=1)                    # scalar distance
    direction = np.arctan2(y[:, 1], y[:, 0])            # radians, 0 = straight ahead

    train_mask, test_mask = split_by_episode(ep_ids)

    # --- Probe 1: 2-D forward/lateral goal vector
    clf = Ridge(alpha=10.0).fit(X[train_mask], y[train_mask])
    y_pred = clf.predict(X[test_mask])
    r2_vec = r2_score(y[test_mask], y_pred, multioutput="uniform_average")
    mae_vec = float(np.abs(y[test_mask] - y_pred).mean())

    # Per-dim R²
    r2_fwd = r2_score(y[test_mask, 0], y_pred[:, 0])
    r2_lat = r2_score(y[test_mask, 1], y_pred[:, 1])

    # --- Probe 2: scalar goal distance
    clf_d = Ridge(alpha=10.0).fit(X[train_mask], dist[train_mask])
    d_pred = clf_d.predict(X[test_mask])
    r2_dist = r2_score(dist[test_mask], d_pred)
    mae_dist = float(np.abs(dist[test_mask] - d_pred).mean())

    # --- Probe 3: goal direction (radians) — use sin/cos encoding
    dir_sc = np.stack([np.sin(direction), np.cos(direction)], axis=1)
    clf_dir = Ridge(alpha=10.0).fit(X[train_mask], dir_sc[train_mask])
    dir_pred = clf_dir.predict(X[test_mask])
    r2_dir = r2_score(dir_sc[test_mask], dir_pred, multioutput="uniform_average")
    mae_dir_deg = float(np.rad2deg(
        np.abs(np.arctan2(dir_pred[:, 0], dir_pred[:, 1]) - direction[test_mask]).mean()
    ))

    # --- Hewitt-Liang control: shuffle labels within episode, refit probe
    rng = np.random.default_rng(0)
    y_ctrl = y.copy()
    for ep in np.unique(ep_ids[train_mask]):
        idx = np.where((ep_ids == ep) & train_mask)[0]
        rng.shuffle(idx)
        y_ctrl[idx] = y[idx][np.argsort(idx)]            # permute within episode
    clf_ctrl = Ridge(alpha=10.0).fit(X[train_mask], y_ctrl[train_mask])
    y_ctrl_pred = clf_ctrl.predict(X[test_mask])
    r2_ctrl = r2_score(y[test_mask], y_ctrl_pred, multioutput="uniform_average")
    selectivity = r2_vec - r2_ctrl

    # --- Also: GPS (absolute world position) for comparison
    clf_gps = Ridge(alpha=10.0).fit(X[train_mask], d["gps"][train_mask])
    gps_pred = clf_gps.predict(X[test_mask])
    r2_gps = r2_score(d["gps"][test_mask], gps_pred, multioutput="uniform_average")

    return {
        "n_steps": int(X.shape[0]),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_episodes": int(len(np.unique(ep_ids))),
        "goal_vector_r2": float(r2_vec),
        "goal_vector_mae_m": mae_vec,
        "r2_forward": float(r2_fwd),
        "r2_lateral": float(r2_lat),
        "goal_dist_r2": float(r2_dist),
        "goal_dist_mae_m": mae_dist,
        "goal_direction_r2": float(r2_dir),
        "goal_direction_mae_deg": mae_dir_deg,
        "goal_vector_r2_control": float(r2_ctrl),
        "goal_vector_selectivity": float(selectivity),
        "gps_r2_reference": float(r2_gps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = {}
    for cond in CONDITIONS:
        path = args.in_dir / f"{cond}.npz"
        print(f"\n=== {cond} ({path}) ===", flush=True)
        r = probe_goal_vector(path)
        if r is None:
            print(f"  MISSING: {path}")
            continue
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            results[cond] = r
            continue
        print(f"  goal-vector R²  = {r['goal_vector_r2']:+.3f} "
              f"(fwd {r['r2_forward']:+.3f}, lat {r['r2_lateral']:+.3f}) "
              f"sel {r['goal_vector_selectivity']:+.3f}")
        print(f"  goal-dist   R²  = {r['goal_dist_r2']:+.3f}  "
              f"mae {r['goal_dist_mae_m']:.3f} m")
        print(f"  goal-dir    R²  = {r['goal_direction_r2']:+.3f}  "
              f"mae {r['goal_direction_mae_deg']:.1f}°")
        print(f"  [ref] GPS   R²  = {r['gps_r2_reference']:+.3f}")
        results[cond] = r

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
