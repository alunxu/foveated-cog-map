"""
Comprehensive single-condition probing analysis for Habitat navigation agents.

Loads data from collect_probes.py and runs a full suite of linear probing
experiments to characterize spatial representations in LSTM hidden states.

Experiments:
  Phase 1 — Baseline probes
    1a. Absolute position (per-scene, temporal split)
    1b. GPS / compass (global, episode split)
    1c. Distance-to-goal
    1d. Multi-layer comparison (h1/h2/h3, c1/c2/c3)
    1e. Control task (shuffled labels)
    1f. Selectivity index

  Phase 2 — H1: Compensatory memory
    2a. Probe accuracy vs. timestep-in-episode
    2b. Cross-heading position-probe generalization
    2c. Path-history probe (lag-k decoding) — inspired by SPACE route retracing
    2d. Visited-region probe (spatial working memory) — inspired by SPACE CSWM
    2e. Occupancy map reconstruction — inspired by SPACE map sketching

  Phase 5 — Per-unit analysis
    5a. Per-unit spatial information (rate maps)
    5b. Place-cell count and statistics

Usage:
    python scripts/probing/analyze.py \
        --data /scratch/izar/$USER/probing_data/blind_gibson.npz \
        --out  /scratch/izar/$USER/probing_results/blind_gibson_full.json
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.probing import fit_probe, fit_probe_cv, prepare_features, angular_mae, episode_split


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive probing analysis")
    p.add_argument("--data", required=True, help="Path to .npz from collect_probes.py")
    p.add_argument("--out", default=None, help="Output .json path")
    p.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization")
    p.add_argument("--pca-dim", type=int, default=0, help="PCA dims (0=skip, use full hidden state)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--min-steps-scene", type=int, default=20, help="Min steps per scene for per-scene probing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-rate-maps", action="store_true", help="Skip rate map computation")
    p.add_argument("--rate-map-bins", type=int, default=20, help="Number of bins per axis for rate maps")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: Baseline Probes
# ═══════════════════════════════════════════════════════════════════════

def probe_1b_global_gps_compass(H, gps, compass, ep_ids, alpha, pca_dim, train_frac, seed):
    """1b. Global GPS + compass probe (episode-level split + k-fold CV)."""
    train_mask, test_mask = episode_split(ep_ids, train_frac, seed)
    H_tr, H_te = prepare_features(H[train_mask], H[test_mask], pca_dim)
    gps_tr, gps_te = gps[train_mask], gps[test_mask]
    comp_tr, comp_te = compass[train_mask], compass[test_mask]

    # GPS
    gps_r2, gps_mae, _ = fit_probe(H_tr, H_te, gps_tr, gps_te, alpha)

    # Compass (sin/cos encoding)
    comp_sc_tr = np.hstack([np.sin(comp_tr), np.cos(comp_tr)])
    comp_sc_te = np.hstack([np.sin(comp_te), np.cos(comp_te)])
    comp_r2, _, comp_pred = fit_probe(H_tr, H_te, comp_sc_tr, comp_sc_te, alpha)
    comp_mae_deg = angular_mae(comp_pred, comp_te.ravel())

    # Combined
    Y_tr = np.hstack([gps_tr, comp_sc_tr])
    Y_te = np.hstack([gps_te, comp_sc_te])
    comb_r2, _, _ = fit_probe(H_tr, H_te, Y_tr, Y_te, alpha)

    # Cross-validated GPS and compass probes (mean ± std across folds)
    gps_cv = fit_probe_cv(H, gps, ep_ids, alpha=alpha, pca_dim=pca_dim, seed=seed)
    comp_sincos = np.hstack([np.sin(compass), np.cos(compass)])
    comp_cv = fit_probe_cv(H, comp_sincos, ep_ids, alpha=alpha, pca_dim=pca_dim, seed=seed)

    return {
        "gps_r2": gps_r2, "gps_mae_m": gps_mae,
        "gps_cv_r2_mean": gps_cv["r2_mean"], "gps_cv_r2_std": gps_cv["r2_std"],
        "gps_cv_mae_mean": gps_cv["mae_mean"], "gps_cv_mae_std": gps_cv["mae_std"],
        "compass_r2": comp_r2, "compass_mae_deg": comp_mae_deg,
        "compass_cv_r2_mean": comp_cv["r2_mean"], "compass_cv_r2_std": comp_cv["r2_std"],
        "combined_r2": comb_r2,
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "n_cv_folds": gps_cv["n_folds"],
    }


def probe_1a_per_scene_position(H, P, theta, scene_ids, ep_ids, alpha, train_frac, min_steps, pca_dim, seed=42):
    """1a. Absolute position probe per scene (episode-level split within scene).

    Prior version did a step-level temporal split (`h_s[:split]` /
    `h_s[split:]`), which leaks because consecutive steps within the
    same episode are highly correlated. If a single episode straddles
    the 80% boundary, its late steps are in test while its early steps
    are in train --- trivial to interpolate. Numbers were inflated.

    Fix: split episodes inside the scene, not steps. Each scene's
    episodes are partitioned 80/20; all steps of a given episode go to
    the same side.
    """
    unique_scenes = np.unique(scene_ids)
    results = []

    for sid in unique_scenes:
        mask = scene_ids == sid
        n = mask.sum()
        if n < min_steps:
            continue

        h_s = H[mask]
        p_s = P[mask]
        t_s = theta[mask]
        ep_s = ep_ids[mask]

        # Episode-level split within this scene's data.
        unique_eps_s = np.unique(ep_s)
        if len(unique_eps_s) < 2:
            continue  # need at least 2 episodes to hold one out

        rng = np.random.RandomState(seed + int(sid))
        perm = rng.permutation(unique_eps_s)
        n_tr_eps = max(1, int(len(unique_eps_s) * train_frac))
        # Guarantee at least one test episode
        if n_tr_eps >= len(unique_eps_s):
            n_tr_eps = len(unique_eps_s) - 1
        train_eps = set(perm[:n_tr_eps].tolist())
        tr_mask = np.isin(ep_s, list(train_eps))
        te_mask = ~tr_mask

        if tr_mask.sum() < 10 or te_mask.sum() < 5:
            continue

        H_tr, H_te = prepare_features(h_s[tr_mask], h_s[te_mask], pca_dim)

        # Position (x, z)
        pos_r2, pos_mae, _ = fit_probe(
            H_tr, H_te, p_s[tr_mask][:, [0, 2]], p_s[te_mask][:, [0, 2]], alpha
        )

        # Heading (sin/cos)
        Y_tr_h = np.stack([np.sin(t_s[tr_mask]), np.cos(t_s[tr_mask])], axis=1)
        Y_te_h = np.stack([np.sin(t_s[te_mask]), np.cos(t_s[te_mask])], axis=1)
        head_r2, _, head_pred = fit_probe(H_tr, H_te, Y_tr_h, Y_te_h, alpha)
        head_mae = angular_mae(head_pred, t_s[te_mask])

        results.append({
            "scene_id": int(sid),
            "n_steps": int(n),
            "n_episodes": int(len(unique_eps_s)),
            "n_train_eps": int(n_tr_eps),
            "pos_r2": pos_r2, "pos_mae": pos_mae,
            "head_r2": head_r2, "head_mae": head_mae,
        })

    if results:
        return {
            "n_scenes_probed": len(results),
            "median_pos_r2": float(np.median([r["pos_r2"] for r in results])),
            "median_pos_mae": float(np.median([r["pos_mae"] for r in results])),
            "median_head_r2": float(np.median([r["head_r2"] for r in results])),
            "median_head_mae": float(np.median([r["head_mae"] for r in results])),
            "scenes": results,
        }
    return {"n_scenes_probed": 0, "scenes": []}


def probe_1c_distance_to_goal(H, dtg, ep_ids, alpha, pca_dim, train_frac, seed):
    """1c. Distance-to-goal probe (with k-fold CV)."""
    train_mask, test_mask = episode_split(ep_ids, train_frac, seed)
    H_tr, H_te = prepare_features(H[train_mask], H[test_mask], pca_dim)
    dtg_tr, dtg_te = dtg[train_mask, None], dtg[test_mask, None]
    r2, mae, _ = fit_probe(H_tr, H_te, dtg_tr, dtg_te, alpha)

    dtg_cv = fit_probe_cv(H, dtg[:, None], ep_ids, alpha=alpha, pca_dim=pca_dim, seed=seed)

    return {
        "r2": r2, "mae_m": mae,
        "cv_r2_mean": dtg_cv["r2_mean"], "cv_r2_std": dtg_cv["r2_std"],
        "cv_mae_mean": dtg_cv["mae_mean"], "cv_mae_std": dtg_cv["mae_std"],
        "n_cv_folds": dtg_cv["n_folds"],
    }


def probe_1d_multilayer(h_layers, c_layers, gps, compass, ep_ids, alpha, pca_dim, train_frac, seed):
    """1d. Probe each LSTM layer separately (h and c states)."""
    n_layers = h_layers.shape[1]
    train_mask, test_mask = episode_split(ep_ids, train_frac, seed)

    gps_tr, gps_te = gps[train_mask], gps[test_mask]
    comp_tr, comp_te = compass[train_mask], compass[test_mask]
    comp_sc_tr = np.hstack([np.sin(comp_tr), np.cos(comp_tr)])
    comp_sc_te = np.hstack([np.sin(comp_te), np.cos(comp_te)])

    layer_results = []
    for li in range(n_layers):
        for state_type, states in [("h", h_layers), ("c", c_layers)]:
            H = states[:, li, :]
            H_tr, H_te = prepare_features(H[train_mask], H[test_mask], pca_dim)

            gps_r2, gps_mae, _ = fit_probe(H_tr, H_te, gps_tr, gps_te, alpha)
            comp_r2, _, comp_pred = fit_probe(H_tr, H_te, comp_sc_tr, comp_sc_te, alpha)
            comp_mae = angular_mae(comp_pred, comp_te.ravel())

            layer_results.append({
                "layer": li, "state": state_type,
                "gps_r2": gps_r2, "gps_mae_m": gps_mae,
                "compass_r2": comp_r2, "compass_mae_deg": comp_mae,
            })

    return layer_results


def probe_1ef_control_and_selectivity(H, gps, compass, ep_ids, alpha, pca_dim, train_frac, seed):
    """1e-f. Control task (shuffled labels) + selectivity index.

    Hewitt & Liang (2019): if a probe achieves high R² even on shuffled
    labels, the probe is too expressive and the original R² is unreliable.
    Selectivity = (real_R² - control_R²) / (1 - control_R²).
    """
    train_mask, test_mask = episode_split(ep_ids, train_frac, seed)
    H_tr, H_te = prepare_features(H[train_mask], H[test_mask], pca_dim)

    gps_tr, gps_te = gps[train_mask], gps[test_mask]
    comp_tr, comp_te = compass[train_mask], compass[test_mask]
    comp_sc_tr = np.hstack([np.sin(comp_tr), np.cos(comp_tr)])
    comp_sc_te = np.hstack([np.sin(comp_te), np.cos(comp_te)])

    # Real probes
    gps_r2, _, _ = fit_probe(H_tr, H_te, gps_tr, gps_te, alpha)
    comp_r2, _, _ = fit_probe(H_tr, H_te, comp_sc_tr, comp_sc_te, alpha)

    # Control: shuffle labels within training set, keeping test labels real.
    # The probe should not generalize to real test labels after training on
    # shuffled labels → control R² should be near 0 or negative.
    rng = np.random.RandomState(seed + 1000)
    n_shuffles = 5
    ctrl_gps_r2s, ctrl_comp_r2s = [], []
    for _ in range(n_shuffles):
        gps_shuf = gps_tr.copy()
        rng.shuffle(gps_shuf)
        comp_shuf = comp_sc_tr.copy()
        rng.shuffle(comp_shuf)

        r2_g, _, _ = fit_probe(H_tr, H_te, gps_shuf, gps_te, alpha)
        r2_c, _, _ = fit_probe(H_tr, H_te, comp_shuf, comp_sc_te, alpha)
        ctrl_gps_r2s.append(r2_g)
        ctrl_comp_r2s.append(r2_c)

    ctrl_gps = float(np.mean(ctrl_gps_r2s))
    ctrl_comp = float(np.mean(ctrl_comp_r2s))

    # Selectivity
    sel_gps = (gps_r2 - ctrl_gps) / max(1 - ctrl_gps, 1e-6) if ctrl_gps < 1 else 0
    sel_comp = (comp_r2 - ctrl_comp) / max(1 - ctrl_comp, 1e-6) if ctrl_comp < 1 else 0

    return {
        "gps_r2_real": gps_r2, "gps_r2_control": ctrl_gps, "gps_selectivity": sel_gps,
        "compass_r2_real": comp_r2, "compass_r2_control": ctrl_comp, "compass_selectivity": sel_comp,
        "n_shuffles": n_shuffles,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: H1 — Compensatory Memory
# ═══════════════════════════════════════════════════════════════════════

def probe_2a_accuracy_vs_timestep(H, gps, compass, ep_ids, step_in_ep, alpha, pca_dim, train_frac, seed):
    """2a. Probe accuracy stratified by timestep within episode.

    Trains one global probe, then evaluates R² separately for early,
    mid, and late timesteps. For the foveated agent, later timesteps
    have had more time to accumulate spatial information.
    """
    train_mask, test_mask = episode_split(ep_ids, train_frac, seed)
    H_tr, H_te = prepare_features(H[train_mask], H[test_mask], pca_dim)
    gps_tr, gps_te = gps[train_mask], gps[test_mask]

    # Train global probe
    reg = Ridge(alpha=alpha)
    reg.fit(H_tr, gps_tr)

    # Evaluate on test set, stratified by timestep
    test_steps = step_in_ep[test_mask]
    pred = reg.predict(H_te)

    # Bin by timestep: 0, 1, 2, 3+
    bins = [(0, 0, "t=0"), (1, 1, "t=1"), (2, 2, "t=2"), (3, 999, "t≥3")]
    stratified = []
    for lo, hi, label in bins:
        mask = (test_steps >= lo) & (test_steps <= hi)
        n = mask.sum()
        if n < 5:
            continue
        r2 = float(r2_score(gps_te[mask], pred[mask], multioutput="uniform_average"))
        mae = float(mean_absolute_error(gps_te[mask], pred[mask]))
        stratified.append({"timestep_bin": label, "n_steps": int(n), "gps_r2": r2, "gps_mae_m": mae})

    return stratified


def probe_2b_cross_heading_generalization(H, positions, headings, scene_ids, alpha, pca_dim, min_steps):
    """2b. Train position probe on one heading range, test on another.

    If an agent uses appearance-invariant (allocentric) codes, position
    probes should generalize across headings. If it uses egocentric codes
    tied to visual appearance, cross-heading generalization will be poor.
    """
    # Bin headings into 4 quadrants
    heading_bins = np.digitize(headings, bins=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi]) - 1
    heading_bins = np.clip(heading_bins, 0, 3)

    # Use scenes with enough data
    results = []
    unique_scenes = np.unique(scene_ids)

    for sid in unique_scenes:
        smask = scene_ids == sid
        if smask.sum() < min_steps:
            continue

        H_s = H[smask]
        P_s = positions[smask][:, [0, 2]]
        hb_s = heading_bins[smask]

        # For each pair of heading quadrants
        for train_q in range(4):
            test_q = (train_q + 2) % 4  # opposite quadrant
            tr_mask = hb_s == train_q
            te_mask = hb_s == test_q

            if tr_mask.sum() < 5 or te_mask.sum() < 5:
                continue

            scaler = StandardScaler()
            H_tr = scaler.fit_transform(H_s[tr_mask])
            H_te = scaler.transform(H_s[te_mask])

            r2, mae, _ = fit_probe(H_tr, H_te, P_s[tr_mask], P_s[te_mask], alpha)
            results.append({
                "scene_id": int(sid), "train_heading_q": train_q, "test_heading_q": test_q,
                "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
                "pos_r2": r2, "pos_mae": mae,
            })

    if results:
        return {
            "n_pairs": len(results),
            "median_pos_r2": float(np.median([r["pos_r2"] for r in results])),
            "pairs": results,
        }
    return {"n_pairs": 0, "pairs": []}


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2c-d: SPACE-inspired probes (Ramakrishnan et al., ICLR 2025)
# ═══════════════════════════════════════════════════════════════════════

def probe_2c_path_history(H, gps, ep_ids, step_in_ep, alpha, pca_dim, train_frac, seed, max_lag=5):
    """2c. Path-history probe (lag-k position decoding).

    Inspired by SPACE's route-retracing task: does the hidden state
    retain memory of *where the agent was* k steps ago?

    For each lag k, we train a probe to predict GPS(t-k) from hidden(t).
    The decay of R² with lag measures how much trajectory history the
    memory retains. Compensatory memory (H1) predicts slower decay for
    agents with degraded input (they need to remember more).

    Reliability guards:
      - max_lag is capped at the median episode length minus 1
      - Minimum 100 test samples per lag (was 10 — too few for stable R²)
      - Minimum 200 train samples (Ridge with 512-d features needs this)
      - Reports MAE alongside R² since R² is unreliable when test variance
        is near zero (common at high lags with short episodes)
    """
    train_mask, test_mask = episode_split(ep_ids, train_frac, seed)

    # Cap max_lag at median episode length - 1 to avoid degenerate samples
    ep_lengths = np.array([
        int(step_in_ep[ep_ids == ep].max()) + 1
        for ep in np.unique(ep_ids)
    ])
    median_ep_len = int(np.median(ep_lengths))
    effective_max_lag = min(max_lag, max(median_ep_len - 1, 0))

    N = len(H)
    lag_results = []

    # Minimum sample thresholds for reliable probing
    MIN_TRAIN = 200  # Ridge with 512-d features: need n >> d
    MIN_TEST = 100   # For stable R² estimation

    for k in range(effective_max_lag + 1):
        # For lag k, we need timesteps where step_in_ep >= k
        valid = step_in_ep >= k
        if valid.sum() < MIN_TRAIN + MIN_TEST:
            continue

        # Build lagged GPS target: for each valid timestep t, find the
        # timestep in the same episode that is k steps earlier
        gps_lagged = np.zeros_like(gps)
        valid_mask = np.zeros(N, dtype=bool)

        # Group by episode for efficiency
        for ep in np.unique(ep_ids):
            ep_mask = ep_ids == ep
            ep_indices = np.where(ep_mask)[0]
            ep_steps = step_in_ep[ep_mask]

            for local_i, global_i in enumerate(ep_indices):
                if ep_steps[local_i] >= k:
                    # Find the index that is k steps earlier
                    target_step = ep_steps[local_i] - k
                    local_target = np.where(ep_steps == target_step)[0]
                    if len(local_target) > 0:
                        gps_lagged[global_i] = gps[ep_indices[local_target[0]]]
                        valid_mask[global_i] = True

        if valid_mask.sum() < MIN_TRAIN + MIN_TEST:
            continue

        # Split using the same train/test episodes
        tr = train_mask & valid_mask
        te = test_mask & valid_mask
        if tr.sum() < MIN_TRAIN or te.sum() < MIN_TEST:
            lag_results.append({
                "lag_k": k, "r2": None, "mae_m": None,
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
                "skipped": f"insufficient samples (train={int(tr.sum())}, test={int(te.sum())})",
            })
            continue

        # Check test target variance — near-zero variance makes R² meaningless
        test_target_var = float(np.var(gps_lagged[te]))

        H_tr, H_te = prepare_features(H[tr], H[te], pca_dim)
        r2, mae, _ = fit_probe(H_tr, H_te, gps_lagged[tr], gps_lagged[te], alpha)
        lag_results.append({
            "lag_k": k, "r2": r2, "mae_m": mae,
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "test_target_var": test_target_var,
            "reliable": bool(test_target_var > 0.01 and te.sum() >= MIN_TEST),
        })

    return {
        "lags": lag_results,
        "median_episode_length": median_ep_len,
        "effective_max_lag": effective_max_lag,
        "requested_max_lag": max_lag,
        "min_train_threshold": MIN_TRAIN,
        "min_test_threshold": MIN_TEST,
    }


def probe_2d_visited_region(H, positions, ep_ids, step_in_ep, scene_ids, alpha, pca_dim, train_frac, seed, grid_res=1.0):
    """2d. Visited-region probe (spatial working memory).

    Inspired by SPACE's Cambridge Spatial Working Memory task: does the
    hidden state encode which parts of the environment have been visited?

    For each timestep, we construct a binary vector of visited grid cells
    (discretized at grid_res meters) for the current episode up to that
    point. A linear probe predicts this visited-set from the hidden state.
    """
    train_mask, test_mask = episode_split(ep_ids, train_frac, seed)

    # Determine global grid bounds
    xz = positions[:, [0, 2]]
    x_min, z_min = xz.min(axis=0) - 0.5
    x_max, z_max = xz.max(axis=0) + 0.5
    nx = max(1, int(np.ceil((x_max - x_min) / grid_res)))
    nz = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    n_cells = nx * nz

    if n_cells > 500:
        # Grid too large — use per-scene analysis or coarser resolution
        grid_res = max(grid_res, np.sqrt((x_max - x_min) * (z_max - z_min) / 500))
        nx = max(1, int(np.ceil((x_max - x_min) / grid_res)))
        nz = max(1, int(np.ceil((z_max - z_min) / grid_res)))
        n_cells = nx * nz

    # Build visited-cell targets per timestep
    visited_targets = np.zeros((len(H), n_cells), dtype=np.float32)

    for ep in np.unique(ep_ids):
        ep_mask = ep_ids == ep
        ep_indices = np.where(ep_mask)[0]
        ep_steps = step_in_ep[ep_mask]
        order = np.argsort(ep_steps)
        ep_indices = ep_indices[order]

        visited_set = np.zeros(n_cells, dtype=np.float32)
        for idx in ep_indices:
            x_bin = min(int((positions[idx, 0] - x_min) / grid_res), nx - 1)
            z_bin = min(int((positions[idx, 2] - z_min) / grid_res), nz - 1)
            cell = x_bin * nz + z_bin
            visited_set[cell] = 1.0
            visited_targets[idx] = visited_set.copy()

    # Probe: predict visited grid from hidden state
    tr = train_mask
    te = test_mask
    if tr.sum() < 20 or te.sum() < 10:
        return {"error": "not enough data"}

    H_tr, H_te = prepare_features(H[tr], H[te], pca_dim)
    Y_tr, Y_te = visited_targets[tr], visited_targets[te]

    # Use Ridge regression (treating each cell as a regression target)
    reg = Ridge(alpha=alpha)
    reg.fit(H_tr, Y_tr)
    pred = reg.predict(H_te)

    # Metrics: overall R², and per-cell binary accuracy (threshold at 0.5)
    r2 = float(r2_score(Y_te, pred, multioutput="uniform_average"))

    pred_binary = (pred > 0.5).astype(float)
    accuracy = float(np.mean(pred_binary == Y_te))
    # Only count cells that are ever visited (non-trivial)
    active_cells = Y_te.sum(axis=0) > 0
    if active_cells.sum() > 0:
        active_acc = float(np.mean(pred_binary[:, active_cells] == Y_te[:, active_cells]))
    else:
        active_acc = None

    return {
        "r2": r2,
        "binary_accuracy": accuracy,
        "active_cell_accuracy": active_acc,
        "n_cells": int(n_cells),
        "n_active_cells": int(active_cells.sum()) if active_cells is not None else 0,
        "grid_res_m": grid_res,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2e: Occupancy Map Reconstruction (SPACE map-sketching analogue)
# ═══════════════════════════════════════════════════════════════════════

def probe_2e_occupancy_decoding(H, local_occupancy, ep_ids, alpha, pca_dim, train_frac, seed):
    """2e. Occupancy map reconstruction from hidden state.

    Inspired by SPACE's map-sketching task: can a linear probe decode the
    local navigability layout (walls vs. free space) from the LSTM hidden
    state? This is the most direct test of whether the hidden state encodes
    a cognitive map of the environment.

    The occupancy grid is a G×G binary array (1=navigable, 0=obstacle)
    centered on the agent's position in world coordinates.
    """
    train_mask, test_mask = episode_split(ep_ids, train_frac, seed)

    # Flatten occupancy grids: (N, G, G) → (N, G*G)
    N, G1, G2 = local_occupancy.shape
    occ_flat = local_occupancy.reshape(N, -1)
    n_cells = G1 * G2

    H_tr, H_te = prepare_features(H[train_mask], H[test_mask], pca_dim)
    Y_tr, Y_te = occ_flat[train_mask], occ_flat[test_mask]

    # Ridge regression: hidden → occupancy
    reg = Ridge(alpha=alpha)
    reg.fit(H_tr, Y_tr)
    pred = reg.predict(H_te)

    # R² (how well does the probe reconstruct the layout?)
    r2 = float(r2_score(Y_te, pred, multioutput="uniform_average"))

    # Binary accuracy (threshold at 0.5)
    pred_binary = (pred > 0.5).astype(float)
    accuracy = float(np.mean(pred_binary == Y_te))

    # Per-cell metrics: separate navigable vs. obstacle accuracy
    nav_mask = Y_te == 1.0
    obs_mask = Y_te == 0.0
    nav_acc = float(np.mean(pred_binary[nav_mask] == 1.0)) if nav_mask.sum() > 0 else None
    obs_acc = float(np.mean(pred_binary[obs_mask] == 0.0)) if obs_mask.sum() > 0 else None

    # F1 score for navigable cells
    tp = float(np.sum((pred_binary == 1.0) & (Y_te == 1.0)))
    fp = float(np.sum((pred_binary == 1.0) & (Y_te == 0.0)))
    fn = float(np.sum((pred_binary == 0.0) & (Y_te == 1.0)))
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "r2": r2,
        "binary_accuracy": accuracy,
        "navigable_accuracy": nav_acc,
        "obstacle_accuracy": obs_acc,
        "f1_navigable": float(f1),
        "n_cells": n_cells,
        "grid_shape": [int(G1), int(G2)],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Phase 5: Per-Unit Analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_rate_maps(H, positions, scene_ids, n_bins=20, min_steps=20):
    """5a. Per-unit spatial information via rate maps.

    For each scene, bin the (x,z) space and compute each neuron's mean
    activation per bin. Then measure spatial information content:
    SI = sum_i (lambda_i / lambda) * log2(lambda_i / lambda) * p_i
    where lambda_i is the mean firing rate in bin i, lambda is the
    overall mean, and p_i is the occupancy of bin i.
    """
    unique_scenes = np.unique(scene_ids)
    all_spatial_info = []  # (n_scenes_used, hidden_dim)

    for sid in unique_scenes:
        mask = scene_ids == sid
        if mask.sum() < min_steps:
            continue

        h_s = H[mask]           # (n, hidden_dim)
        p_s = positions[mask]   # (n, 3)
        xz = p_s[:, [0, 2]]    # (n, 2)

        # Create 2D histogram bins
        x_edges = np.linspace(xz[:, 0].min() - 1e-6, xz[:, 0].max() + 1e-6, n_bins + 1)
        z_edges = np.linspace(xz[:, 1].min() - 1e-6, xz[:, 1].max() + 1e-6, n_bins + 1)

        # Bin each step
        x_bin = np.digitize(xz[:, 0], x_edges) - 1
        z_bin = np.digitize(xz[:, 1], z_edges) - 1
        x_bin = np.clip(x_bin, 0, n_bins - 1)
        z_bin = np.clip(z_bin, 0, n_bins - 1)
        bin_idx = x_bin * n_bins + z_bin

        # Occupancy and mean activation per bin
        hidden_dim = h_s.shape[1]
        occupancy = np.zeros(n_bins * n_bins)
        sum_act = np.zeros((n_bins * n_bins, hidden_dim))

        for i in range(len(h_s)):
            b = bin_idx[i]
            occupancy[b] += 1
            sum_act[b] += h_s[i]

        # Only use occupied bins
        occ_mask = occupancy > 0
        if occ_mask.sum() < 4:
            continue

        p_occ = occupancy[occ_mask] / occupancy[occ_mask].sum()  # occupancy probability
        mean_act = sum_act[occ_mask] / occupancy[occ_mask, None]  # (n_occ_bins, hidden_dim)

        # Global mean activation per neuron
        global_mean = h_s.mean(axis=0)  # (hidden_dim,)

        # Spatial information per neuron (bits)
        # SI_j = sum_i p_i * (lambda_ij / lambda_j) * log2(lambda_ij / lambda_j)
        # Handle zero/negative activations by shifting to positive range
        si = np.zeros(hidden_dim)
        for j in range(hidden_dim):
            lam = global_mean[j]
            if abs(lam) < 1e-8:
                continue
            ratio = mean_act[:, j] / lam
            ratio = np.clip(ratio, 1e-8, None)
            si[j] = np.sum(p_occ * ratio * np.log2(ratio))

        all_spatial_info.append(si)

    if not all_spatial_info:
        return {"n_scenes_used": 0}

    # Average spatial info across scenes
    si_matrix = np.array(all_spatial_info)  # (n_scenes, hidden_dim)
    mean_si = si_matrix.mean(axis=0)        # (hidden_dim,)

    # Identify place-cell-like units: top spatial information
    top_k = min(20, len(mean_si))
    top_indices = np.argsort(mean_si)[::-1][:top_k]

    return {
        "n_scenes_used": len(all_spatial_info),
        "hidden_dim": int(len(mean_si)),
        "mean_spatial_info_bits": float(mean_si.mean()),
        "max_spatial_info_bits": float(mean_si.max()),
        "std_spatial_info_bits": float(mean_si.std()),
        "top_units": [{"unit": int(i), "spatial_info": float(mean_si[i])} for i in top_indices],
        "n_place_cells_1bit": int((mean_si > 1.0).sum()),
        "n_place_cells_05bit": int((mean_si > 0.5).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if args.out is None:
        args.out = args.data.replace(".npz", "_analysis.json")

    print(f"Loading {args.data} ...")
    data = np.load(args.data, allow_pickle=True)

    H = data["hidden_states"]
    P = data["positions"]
    theta = data["headings"]
    ep_ids = data["episode_ids"]
    scene_ids = data["scene_ids"]

    # New fields (with backward compat fallbacks)
    h_layers = data["h_layers"] if "h_layers" in data else None
    c_layers = data["c_layers"] if "c_layers" in data else None
    gps = data["gps"] if "gps" in data else None
    compass = data["compass"] if "compass" in data else None
    dtg = data["distance_to_goal"] if "distance_to_goal" in data else None
    step_in_ep = data["step_in_episode"] if "step_in_episode" in data else None

    N = len(H)
    n_eps = len(np.unique(ep_ids))
    n_scenes = len(np.unique(scene_ids))
    print(f"  Steps: {N}, Hidden dim: {H.shape[1]}")
    print(f"  Episodes: {n_eps}, Scenes: {n_scenes}")
    if h_layers is not None:
        print(f"  LSTM layers: {h_layers.shape[1]} (h) + {c_layers.shape[1]} (c)")

    results = {
        "source": args.data,
        "n_steps": N,
        "hidden_dim": int(H.shape[1]),
        "n_episodes": n_eps,
        "n_scenes": n_scenes,
        "alpha": args.alpha,
        "pca_dim": args.pca_dim,
    }

    # ── Phase 1: Baseline Probes ──────────────────────────────────────

    # 1b. Global GPS/Compass
    if gps is not None and compass is not None:
        print(f"\n{'─'*60}")
        print("  1b. Global GPS + Compass probe")
        print(f"{'─'*60}")
        res_1b = probe_1b_global_gps_compass(
            H, gps, compass, ep_ids,
            args.alpha, args.pca_dim, args.train_frac, args.seed,
        )
        results["1b_global_gps_compass"] = res_1b
        print(f"  GPS R²={res_1b['gps_r2']:+.4f}  MAE={res_1b['gps_mae_m']:.3f}m")
        print(f"    CV: R²={res_1b['gps_cv_r2_mean']:+.4f} ± {res_1b['gps_cv_r2_std']:.4f}")
        print(f"  Compass R²={res_1b['compass_r2']:+.4f}  MAE={res_1b['compass_mae_deg']:.1f}°")
        print(f"    CV: R²={res_1b['compass_cv_r2_mean']:+.4f} ± {res_1b['compass_cv_r2_std']:.4f}")
        print(f"  Combined R²={res_1b['combined_r2']:+.4f}")

    # 1a. Per-scene absolute position
    print(f"\n{'─'*60}")
    print(f"  1a. Per-scene absolute position (min_steps={args.min_steps_scene})")
    print(f"{'─'*60}")
    res_1a = probe_1a_per_scene_position(
        H, P, theta, scene_ids, ep_ids,
        args.alpha, args.train_frac, args.min_steps_scene, args.pca_dim,
        seed=args.seed,
    )
    results["1a_per_scene_position"] = res_1a
    if res_1a["n_scenes_probed"] > 0:
        print(f"  {res_1a['n_scenes_probed']} scenes probed")
        print(f"  Median Pos R²={res_1a['median_pos_r2']:+.4f}  MAE={res_1a['median_pos_mae']:.3f}m")
        print(f"  Median Head R²={res_1a['median_head_r2']:+.4f}  MAE={res_1a['median_head_mae']:.1f}°")
    else:
        print("  No scenes with enough data.")

    # 1c. Distance-to-goal
    if dtg is not None:
        print(f"\n{'─'*60}")
        print("  1c. Distance-to-goal probe")
        print(f"{'─'*60}")
        res_1c = probe_1c_distance_to_goal(
            H, dtg, ep_ids, args.alpha, args.pca_dim, args.train_frac, args.seed,
        )
        results["1c_distance_to_goal"] = res_1c
        print(f"  R²={res_1c['r2']:+.4f}  MAE={res_1c['mae_m']:.3f}m")
        print(f"    CV: R²={res_1c['cv_r2_mean']:+.4f} ± {res_1c['cv_r2_std']:.4f}")

    # 1d. Multi-layer comparison
    if h_layers is not None and gps is not None and compass is not None:
        print(f"\n{'─'*60}")
        print("  1d. Multi-layer comparison")
        print(f"{'─'*60}")
        res_1d = probe_1d_multilayer(
            h_layers, c_layers, gps, compass, ep_ids,
            args.alpha, args.pca_dim, args.train_frac, args.seed,
        )
        results["1d_multilayer"] = res_1d
        for r in res_1d:
            print(f"  Layer {r['layer']} ({r['state']}):  GPS R²={r['gps_r2']:+.4f}  "
                  f"Compass R²={r['compass_r2']:+.4f}")

    # 1e-f. Control task + selectivity
    if gps is not None and compass is not None:
        print(f"\n{'─'*60}")
        print("  1e-f. Control task + selectivity (Hewitt & Liang)")
        print(f"{'─'*60}")
        res_1ef = probe_1ef_control_and_selectivity(
            H, gps, compass, ep_ids,
            args.alpha, args.pca_dim, args.train_frac, args.seed,
        )
        results["1ef_control_selectivity"] = res_1ef
        print(f"  GPS:     real R²={res_1ef['gps_r2_real']:+.4f}  "
              f"control R²={res_1ef['gps_r2_control']:+.4f}  "
              f"selectivity={res_1ef['gps_selectivity']:.3f}")
        print(f"  Compass: real R²={res_1ef['compass_r2_real']:+.4f}  "
              f"control R²={res_1ef['compass_r2_control']:+.4f}  "
              f"selectivity={res_1ef['compass_selectivity']:.3f}")

    # ── Phase 2: H1 — Compensatory Memory ─────────────────────────────

    # 2a. Accuracy vs timestep
    if step_in_ep is not None and gps is not None and compass is not None:
        print(f"\n{'─'*60}")
        print("  2a. Probe accuracy vs. timestep-in-episode")
        print(f"{'─'*60}")
        res_2a = probe_2a_accuracy_vs_timestep(
            H, gps, compass, ep_ids, step_in_ep,
            args.alpha, args.pca_dim, args.train_frac, args.seed,
        )
        results["2a_accuracy_vs_timestep"] = res_2a
        for r in res_2a:
            print(f"  {r['timestep_bin']:>5s}: GPS R²={r['gps_r2']:+.4f}  "
                  f"MAE={r['gps_mae_m']:.3f}m  (n={r['n_steps']})")

    # 2b. Cross-heading generalization
    print(f"\n{'─'*60}")
    print("  2b. Cross-heading position-probe generalization")
    print(f"{'─'*60}")
    res_2b = probe_2b_cross_heading_generalization(
        H, P, theta, scene_ids,
        args.alpha, args.pca_dim, args.min_steps_scene,
    )
    results["2b_cross_heading"] = {
        "n_pairs": res_2b["n_pairs"],
        "median_pos_r2": res_2b.get("median_pos_r2"),
    }
    if res_2b["n_pairs"] > 0:
        print(f"  {res_2b['n_pairs']} heading pairs across scenes")
        print(f"  Median cross-heading Pos R²={res_2b['median_pos_r2']:+.4f}")
    else:
        print("  Not enough data for cross-heading analysis.")

    # ── Phase 2c-d: SPACE-inspired probes ─────────────────────────────

    # 2c. Path-history probe (lag-k decoding)
    if step_in_ep is not None and gps is not None:
        print(f"\n{'─'*60}")
        print("  2c. Path-history probe (lag-k position decoding)")
        print(f"{'─'*60}")
        res_2c = probe_2c_path_history(
            H, gps, ep_ids, step_in_ep,
            args.alpha, args.pca_dim, args.train_frac, args.seed,
            max_lag=5,
        )
        results["2c_path_history"] = res_2c
        print(f"  Median episode length: {res_2c['median_episode_length']} steps")
        print(f"  Effective max lag: {res_2c['effective_max_lag']} "
              f"(requested: {res_2c['requested_max_lag']})")
        for r in res_2c["lags"]:
            if r.get("skipped"):
                print(f"  lag={r['lag_k']}: SKIPPED — {r['skipped']}")
            else:
                reliable = "✓" if r.get("reliable", False) else "⚠"
                print(f"  lag={r['lag_k']}: GPS R²={r['r2']:+.4f}  "
                      f"MAE={r['mae_m']:.3f}m  (n={r['n_test']}) "
                      f"[var={r['test_target_var']:.4f}] {reliable}")

    # 2d. Visited-region probe (spatial working memory)
    if step_in_ep is not None:
        print(f"\n{'─'*60}")
        print("  2d. Visited-region probe (spatial working memory)")
        print(f"{'─'*60}")
        res_2d = probe_2d_visited_region(
            H, P, ep_ids, step_in_ep, scene_ids,
            args.alpha, args.pca_dim, args.train_frac, args.seed,
            grid_res=1.0,
        )
        results["2d_visited_region"] = res_2d
        if "error" not in res_2d:
            print(f"  Grid: {res_2d['n_cells']} cells ({res_2d['n_active_cells']} active), "
                  f"res={res_2d['grid_res_m']:.1f}m")
            print(f"  R²={res_2d['r2']:+.4f}  Binary acc={res_2d['binary_accuracy']:.3f}")
            if res_2d['active_cell_accuracy'] is not None:
                print(f"  Active-cell acc={res_2d['active_cell_accuracy']:.3f}")
        else:
            print(f"  {res_2d['error']}")

    # 2e. Occupancy map reconstruction
    local_occ = data["local_occupancy"] if "local_occupancy" in data else None
    if local_occ is not None:
        print(f"\n{'─'*60}")
        print(f"  2e. Occupancy map reconstruction (map-sketching analogue)")
        print(f"{'─'*60}")
        res_2e = probe_2e_occupancy_decoding(
            H, local_occ, ep_ids,
            args.alpha, args.pca_dim, args.train_frac, args.seed,
        )
        results["2e_occupancy_decoding"] = res_2e
        print(f"  Grid: {res_2e['grid_shape'][0]}×{res_2e['grid_shape'][1]} = "
              f"{res_2e['n_cells']} cells")
        print(f"  R²={res_2e['r2']:+.4f}  Accuracy={res_2e['binary_accuracy']:.3f}")
        print(f"  Navigable acc={res_2e['navigable_accuracy']:.3f}  "
              f"Obstacle acc={res_2e['obstacle_accuracy']:.3f}")
        print(f"  F1 (navigable)={res_2e['f1_navigable']:.3f}")

    # ── Phase 5: Per-Unit Analysis ─────────────────────────────────────

    if not args.skip_rate_maps:
        print(f"\n{'─'*60}")
        print(f"  5a. Per-unit rate maps (bins={args.rate_map_bins})")
        print(f"{'─'*60}")
        res_5 = compute_rate_maps(
            H, P, scene_ids, n_bins=args.rate_map_bins,
            min_steps=args.min_steps_scene,
        )
        results["5a_rate_maps"] = res_5
        if res_5["n_scenes_used"] > 0:
            print(f"  Scenes used: {res_5['n_scenes_used']}")
            print(f"  Mean spatial info: {res_5['mean_spatial_info_bits']:.3f} bits")
            print(f"  Max spatial info:  {res_5['max_spatial_info_bits']:.3f} bits")
            print(f"  Place cells (>1 bit): {res_5['n_place_cells_1bit']}")
            print(f"  Place cells (>0.5 bit): {res_5['n_place_cells_05bit']}")
            print(f"  Top-5 units: {[u['unit'] for u in res_5['top_units'][:5]]}")
        else:
            print("  Not enough per-scene data for rate maps.")

    # ── Save ───────────────────────────────────────────────────────────

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
