"""Cross-Condition Generalisation Performance (CCGP) — Bernardi et al. 2020 (Cell).

A representation is "abstract" (factorised) iff a linear decoder of one variable
trained on a subset of conditions generalises to held-out conditions of another
variable. This file maps the original cogneuro analysis onto our 5-condition
LSTM PointNav agents and computes an abstraction-index ratio per condition per
target variable.

Pre-registered protocol (matches docs/manuscript/sample/cogneuro_frameworks/
ccgp_bernardi.md):

  Dichotomies:
    D1 = scene_id parity (split scenes into A/B, balanced count).
    D2 = goal-distance bin (near = bottom 1/3 of dist_to_goal, far = top 1/3,
         middle dropped to keep classes separable).

  Target variables (linear classifiers):
    pos_x_bin   — agent_pos[0] discretised into 4 bins
    heading_oct — heading discretised into 8 octants
    dist_bin    — dist_to_goal into 3 bins (near/mid/far)
    goal_quadrant — goal-relative angle into 4 quadrants (FL, FR, BL, BR)

  For each target V and each dichotomy D in {D1, D2}:
    CCGP(V, D) = avg over hold ∈ {0, 1} of (decoder fit on D≠hold, eval on D=hold)
    WCV(V, D) = avg over d ∈ {0, 1} of 5-fold CV of (decoder on D=d)
    AI(V, D)  = CCGP(V, D) / max(WCV(V, D), 1e-6)

  Pre-registered prediction (as filed in ccgp_bernardi.md):
    For goal-relative variables (dist_bin, goal_quadrant):
      AI ordering = foveated > uniform > coarse > blind
    For allocentric pose (pos_x_bin, heading_oct):
      AI ordering = inverted or flat
    Magnitude (raw WCV): foveated > uniform > coarse > blind for all V.

  Decision rules (frozen):
    - "Strong" if predicted ordering holds for ≥ 1 goal-relative variable AND
      pose ordering is non-monotone (i.e. inversion or flat).
    - "Moderate" if WCV ordering is monotone but AI ordering is flat.
    - "Null" if WCV ordering does not match magnitude prediction.
    - In all cases we report all four targets and both dichotomies; we do NOT
      pick the panel that best matches the prediction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

DATA_ROOT = Path("/tmp/rcp_analysis_v3")
FILES = {
    "blind": "blind_izar_det_RCP.npz",
    "coarse": "coarse_det_RCP.npz",
    "foveated": "foveated_det_RCP.npz",
    "foveated_logpolar": "foveated_logpolar_det_RCP.npz",
    "uniform": "uniform_det_RCP.npz",
}


def discretise_pos_x(pos: np.ndarray, n_bins: int = 4) -> np.ndarray:
    qs = np.quantile(pos[:, 0], np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(pos[:, 0], qs)


def discretise_heading(heading: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """heading in radians [-pi, pi], output 0..n_bins-1 octants."""
    h = np.where(heading < 0, heading + 2 * np.pi, heading)  # [0, 2pi)
    return (h / (2 * np.pi) * n_bins).astype(int) % n_bins


def discretise_dist(d: np.ndarray, n_bins: int = 3) -> np.ndarray:
    qs = np.quantile(d, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(d, qs)


def goal_quadrant(positions: np.ndarray, headings: np.ndarray,
                   goal_pos: np.ndarray) -> np.ndarray:
    """4-class label: front-left (0), front-right (1), back-left (2), back-right (3)
    in agent frame."""
    dx = goal_pos[:, 0] - positions[:, 0]
    dy = goal_pos[:, 1] - positions[:, 1]
    # rotate into agent frame
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    fwd = cos_h * dx + sin_h * dy
    side = -sin_h * dx + cos_h * dy
    front = fwd > 0
    right = side > 0
    return (front.astype(int) * 0) + (right.astype(int) * 1) + ((~front).astype(int) * 2)


def goal_quadrant_safe(positions, headings, goal_pos):
    """Return 4-class quadrant labels with consistent encoding."""
    dx = goal_pos[:, 0] - positions[:, 0]
    dy = goal_pos[:, 1] - positions[:, 1]
    cos_h, sin_h = np.cos(headings), np.sin(headings)
    fwd = cos_h * dx + sin_h * dy
    side = -sin_h * dx + cos_h * dy
    return (fwd > 0).astype(int) * 2 + (side > 0).astype(int)  # 0=BL, 1=BR, 2=FL, 3=FR


def ccgp_one(h, V, D):
    """Average decoder accuracy when held-out on one half of dichotomy D.

    Returns mean across the two holdouts (D=0, D=1).
    """
    scores = []
    for hold in (0, 1):
        train_mask = D != hold
        test_mask = D == hold
        if train_mask.sum() < 50 or test_mask.sum() < 50:
            return np.nan
        # Need >=2 classes in each split
        if len(np.unique(V[train_mask])) < 2 or len(np.unique(V[test_mask])) < 2:
            return np.nan
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=2, solver="lbfgs")
        try:
            clf.fit(h[train_mask], V[train_mask])
            scores.append(clf.score(h[test_mask], V[test_mask]))
        except Exception:
            return np.nan
    return float(np.mean(scores))


def wcv_one(h, V, D, n_folds=5):
    """5-fold CV decoding accuracy averaged over D=0 and D=1 separately."""
    scores = []
    for d in (0, 1):
        mask = D == d
        if mask.sum() < 50:
            return np.nan
        if len(np.unique(V[mask])) < 2:
            return np.nan
        kf = KFold(n_folds, shuffle=True, random_state=0)
        fold_scores = []
        for tr, te in kf.split(np.arange(mask.sum())):
            try:
                clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=2, solver="lbfgs")
                X = h[mask][tr]
                y = V[mask][tr]
                if len(np.unique(y)) < 2:
                    continue
                clf.fit(X, y)
                fold_scores.append(clf.score(h[mask][te], V[mask][te]))
            except Exception:
                continue
        if fold_scores:
            scores.append(np.mean(fold_scores))
    return float(np.mean(scores)) if scores else np.nan


def subsample(arrays: list[np.ndarray], n: int, seed: int = 0):
    """Subsample to n rows (matching across all arrays). For speed only."""
    if len(arrays[0]) <= n:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arrays[0]), n, replace=False)
    return [a[idx] for a in arrays]


def analyse_one_condition(npz_path: Path, n_max: int = 30000) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    # Standardise per-feature so LogReg converges faster.
    mu = h.mean(0, keepdims=True)
    sd = h.std(0, keepdims=True) + 1e-6
    h = (h - mu) / sd
    pos = d["positions"].astype(np.float32)
    head = d["headings"].astype(np.float32)
    dist = d["distance_to_goal"].astype(np.float32)
    goal = d["goal_positions"].astype(np.float32)
    scenes = d["scene_ids"]
    eps = d["episode_ids"]

    # Subsample for speed
    h, pos, head, dist, goal, scenes, eps = subsample(
        [h, pos, head, dist, goal, scenes, eps], n_max, seed=0
    )

    # Define dichotomies (per-row)
    # D1 = scene parity: hash scene_id to get {0, 1}
    scene_str = np.asarray([str(s) for s in scenes])
    d1 = np.array([hash(s) % 2 for s in scene_str], dtype=int)
    # D2 = dist near/far (drop middle)
    q_lo, q_hi = np.quantile(dist, [1/3, 2/3])
    d2 = np.full_like(dist, fill_value=-1, dtype=int)
    d2[dist <= q_lo] = 0
    d2[dist >= q_hi] = 1

    # Targets (per-row)
    targets = {
        "pos_x_bin": discretise_pos_x(pos, n_bins=4),
        "heading_oct": discretise_heading(head, n_bins=8),
        "dist_bin": discretise_dist(dist, n_bins=3),
        "goal_quadrant": goal_quadrant_safe(pos, head, goal),
    }

    out = {}
    for name, V in targets.items():
        cell = {}
        # CCGP across D1 (scene-generalisation): always defined
        cell["ccgp_d1"] = ccgp_one(h, V, d1)
        cell["wcv_d1"] = wcv_one(h, V, d1)
        cell["ai_d1"] = (cell["ccgp_d1"] / max(cell["wcv_d1"], 1e-6)
                         if cell["wcv_d1"] is not None and not np.isnan(cell["wcv_d1"])
                         else np.nan)
        # CCGP across D2 (dist-generalisation): only on labelled (drop d2==-1)
        m2 = d2 != -1
        if m2.sum() > 1000:
            cell["ccgp_d2"] = ccgp_one(h[m2], V[m2], d2[m2])
            cell["wcv_d2"] = wcv_one(h[m2], V[m2], d2[m2])
            cell["ai_d2"] = (cell["ccgp_d2"] / max(cell["wcv_d2"], 1e-6)
                             if cell["wcv_d2"] is not None and not np.isnan(cell["wcv_d2"])
                             else np.nan)
        out[name] = cell
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/ccgp_results.json")
    ap.add_argument("--n_max", type=int, default=30000)
    args = ap.parse_args()

    results = {}
    for cond, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f"  MISSING {path}; skip {cond}")
            continue
        print(f"\n=== condition: {cond} ===")
        results[cond] = analyse_one_condition(path, n_max=args.n_max)
        for v, c in results[cond].items():
            ai_d1 = c.get("ai_d1", float("nan"))
            ai_d2 = c.get("ai_d2", float("nan"))
            wcv_d1 = c.get("wcv_d1", float("nan"))
            print(f"  {v:>15s}  WCV_d1={wcv_d1:.3f}  CCGP_d1={c['ccgp_d1']:.3f}  "
                   f"AI_d1={ai_d1:.3f}  AI_d2={ai_d2:.3f}")

    json.dump(results, open(args.out_path, "w"), indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
