"""
H3 analysis: gaze-memory coupling.

H3 hypothesis (from project proposal §2): *the learned-gaze agent fixates
where its memory is weak, so navigation loss alone is sufficient to
discover information-gain-maximizing fixation*.

Inputs
------
A ``.npz`` produced by ``scripts/probing/collect.py`` for the
*foveated-learned-gaze* condition. Must contain the additional key
``gaze_positions`` (shape ``(N, 2)`` in [0,1]²), which is recorded by
the forward hook in ``collect.py`` when a ``gaze_decoder`` submodule is
present on the policy.

Outputs
-------
A JSON file with:
  - ``position_probe_error_per_step``: per-step position-probe residual
    (2-norm of predicted minus true xy), fitted with episode-level splits.
  - ``gaze_distribution``: overall gaze histogram on a 10×10 grid (sanity
    check — is the agent centre-biased or spread?).
  - ``gaze_vs_memory_uncertainty``: Pearson + Spearman correlation
    between per-step position-probe error and (i) gaze displacement from
    image centre, (ii) frame-to-frame gaze change (saccade magnitude),
    (iii) running gaze entropy. A positive gaze-error correlation is the
    H3 signature: memory is weak → agent looks away from the default
    centre / saccades more / distributes gaze more widely.
  - ``permutation_pvalues``: p-values from shuffling the episode↔gaze
    alignment, to guard against spurious correlation from episode-level
    confounds (easy episodes at the start of a scene, hard ones later).
  - ``per_condition_comparison`` (if multiple npz files are passed): same
    statistics for foveated-fixed, for the contrast required by H3.

Usage
-----
    python scripts/probing/analyze_h3.py \\
        --learned-gaze-npz foveated_learned_gibson.npz \\
        --fixed-gaze-npz   foveated_gibson.npz \\
        --out              h3_analysis.json

If ``--fixed-gaze-npz`` is omitted, only the intra-condition analysis
on the learned-gaze agent runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import entropy, pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.probing import fit_probe_cv  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


REQUIRED_KEYS = (
    "h_layers", "positions", "episode_ids", "scene_ids", "step_in_episode",
)


def _load(npz_path: str) -> dict:
    d = dict(np.load(npz_path, allow_pickle=False))
    for k in REQUIRED_KEYS:
        if k not in d:
            raise KeyError(f"{npz_path}: missing required key {k!r}")
    return d


# ---------------------------------------------------------------------------
# Per-step memory-uncertainty proxy
# ---------------------------------------------------------------------------


def _position_probe_residuals(data: dict, layer_idx: int = -1) -> np.ndarray:
    """Fit a GPS probe with episode-level splits; return per-step 2-norm
    residual of (xy_pred - xy_true) on the held-out fold each step
    belonged to.

    We use cross-validation so every step receives an out-of-fold
    prediction, avoiding the probe seeing its own step's label.
    """
    X = data["h_layers"][:, layer_idx]        # (N, hidden)
    xy = data["positions"][:, [0, 2]]         # (N, 2) — Habitat xy on ground plane
    ep = data["episode_ids"]

    # fit_probe_cv returns per-fold R² + MAE; to get per-step residuals we
    # re-implement a small episode-level k-fold here.
    residuals = np.empty(X.shape[0], dtype=np.float32)
    ep_ids = np.unique(ep)
    k = min(5, len(ep_ids))
    rng = np.random.default_rng(0)
    rng.shuffle(ep_ids)
    folds = np.array_split(ep_ids, k)

    from sklearn.linear_model import Ridge

    for fold in folds:
        test_mask = np.isin(ep, fold)
        train_mask = ~test_mask
        model = Ridge(alpha=10.0)
        model.fit(X[train_mask], xy[train_mask])
        preds = model.predict(X[test_mask])
        residuals[test_mask] = np.linalg.norm(preds - xy[test_mask], axis=1)
    return residuals


# ---------------------------------------------------------------------------
# Gaze statistics
# ---------------------------------------------------------------------------


def _gaze_histogram(gaze: np.ndarray, n_bins: int = 10) -> np.ndarray:
    H, _, _ = np.histogram2d(
        gaze[:, 0], gaze[:, 1], bins=n_bins, range=[[0, 1], [0, 1]]
    )
    return (H / H.sum()).astype(np.float32)


def _gaze_center_distance(gaze: np.ndarray) -> np.ndarray:
    return np.linalg.norm(gaze - 0.5, axis=1)


def _saccade_magnitude(gaze: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    """Frame-to-frame gaze displacement, zeroed at episode boundaries."""
    diff = np.linalg.norm(np.diff(gaze, axis=0, prepend=gaze[:1]), axis=1)
    boundary = episode_ids != np.concatenate([[episode_ids[0]], episode_ids[:-1]])
    diff[boundary] = 0.0
    return diff


def _running_gaze_entropy(
    gaze: np.ndarray, episode_ids: np.ndarray, window: int = 20
) -> np.ndarray:
    """Rolling-window Shannon entropy of gaze positions (coarser 4×4 grid)
    within each episode."""
    out = np.zeros(gaze.shape[0], dtype=np.float32)
    for ep in np.unique(episode_ids):
        idx = np.where(episode_ids == ep)[0]
        if len(idx) < 2:
            continue
        g = gaze[idx]
        for i in range(len(idx)):
            lo = max(0, i - window + 1)
            H, _, _ = np.histogram2d(
                g[lo : i + 1, 0], g[lo : i + 1, 1],
                bins=4, range=[[0, 1], [0, 1]],
            )
            p = H.ravel()
            p = p / (p.sum() + 1e-12)
            out[idx[i]] = entropy(p + 1e-12)
    return out


# ---------------------------------------------------------------------------
# Correlations + permutation test
# ---------------------------------------------------------------------------


def _correlate(a: np.ndarray, b: np.ndarray) -> dict:
    if np.std(a) == 0 or np.std(b) == 0:
        return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_r": 0.0, "spearman_p": 1.0}
    pr, pp = pearsonr(a, b)
    sr, sp = spearmanr(a, b)
    return {
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
    }


def _permutation_pvalue(
    residuals: np.ndarray,
    metric: np.ndarray,
    episode_ids: np.ndarray,
    n_perms: int = 1000,
    seed: int = 0,
) -> float:
    """Shuffle residuals within episode identities and compute fraction of
    permuted correlations ≥ the observed one. Controls for episode-level
    confounds (episode length, scene, task difficulty)."""
    rng = np.random.default_rng(seed)
    observed_r = abs(pearsonr(residuals, metric)[0])
    ep_ids = np.unique(episode_ids)
    n_ge = 0
    for _ in range(n_perms):
        # shuffle the residual-to-episode assignment
        permuted_ep = ep_ids.copy()
        rng.shuffle(permuted_ep)
        ep_map = dict(zip(ep_ids, permuted_ep))
        new_ep = np.array([ep_map[e] for e in episode_ids])
        # pull residual values by swapped episode
        new_res = np.empty_like(residuals)
        for orig, new in zip(ep_ids, permuted_ep):
            new_res[episode_ids == orig] = residuals[episode_ids == new]
        r = abs(pearsonr(new_res, metric)[0])
        if r >= observed_r:
            n_ge += 1
    return (n_ge + 1) / (n_perms + 1)


# ---------------------------------------------------------------------------
# Main per-condition analysis
# ---------------------------------------------------------------------------


def _analyse_condition(data: dict, name: str, run_perm: bool = True) -> dict:
    if "gaze_positions" not in data:
        return {
            "name": name,
            "note": "no gaze_positions in .npz — this condition has no learned gaze",
        }

    gaze = data["gaze_positions"]
    ep = data["episode_ids"]

    residuals = _position_probe_residuals(data)
    center_dist = _gaze_center_distance(gaze)
    saccade = _saccade_magnitude(gaze, ep)
    gaze_ent = _running_gaze_entropy(gaze, ep)

    out = {
        "name": name,
        "n_steps": int(len(residuals)),
        "position_probe_residual": {
            "mean": float(residuals.mean()),
            "median": float(np.median(residuals)),
            "std": float(residuals.std()),
        },
        "gaze_distribution": _gaze_histogram(gaze).tolist(),
        "gaze_center_distance": {
            "mean": float(center_dist.mean()),
            "std": float(center_dist.std()),
        },
        "correlations_with_probe_error": {
            "gaze_center_distance": _correlate(residuals, center_dist),
            "saccade_magnitude": _correlate(residuals, saccade),
            "running_gaze_entropy": _correlate(residuals, gaze_ent),
        },
    }

    if run_perm:
        out["permutation_pvalues_abs_pearson"] = {
            "gaze_center_distance": _permutation_pvalue(residuals, center_dist, ep),
            "saccade_magnitude": _permutation_pvalue(residuals, saccade, ep),
            "running_gaze_entropy": _permutation_pvalue(residuals, gaze_ent, ep),
        }

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--learned-gaze-npz", required=True, help=".npz for foveated-learned")
    p.add_argument("--fixed-gaze-npz", default=None, help="(optional) .npz for foveated-fixed")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--no-permutation", action="store_true",
                   help="skip permutation test (fast path for quick checks)")
    args = p.parse_args()

    data_l = _load(args.learned_gaze_npz)
    if "gaze_positions" not in data_l:
        raise RuntimeError(
            f"{args.learned_gaze_npz}: missing 'gaze_positions'. Re-run collect.py "
            f"on a learned-gaze policy (FoveatedLearnedGazePolicy)."
        )

    report: dict = {"learned_gaze": _analyse_condition(
        data_l, name="foveated_learned", run_perm=not args.no_permutation,
    )}

    if args.fixed_gaze_npz:
        # For foveated-fixed we don't have gaze_positions, so the cross-
        # condition contrast is limited to comparing probe residuals: does
        # learned-gaze reduce memory uncertainty relative to fixed-gaze?
        data_f = _load(args.fixed_gaze_npz)
        residuals_f = _position_probe_residuals(data_f)
        report["fixed_gaze_residual"] = {
            "mean": float(residuals_f.mean()),
            "median": float(np.median(residuals_f)),
            "std": float(residuals_f.std()),
        }
        # quick t-test-style comparison
        l_res = _position_probe_residuals(data_l)
        diff = float(l_res.mean() - residuals_f.mean())
        report["learned_minus_fixed_mean_residual"] = diff

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote H3 report -> {args.out}")


if __name__ == "__main__":
    main()
