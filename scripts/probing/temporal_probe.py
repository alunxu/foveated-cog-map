"""
Temporal probe: GPS / compass R^2 as a function of step-in-episode.

Question (raised by length-matched probe, scripts/probing/length_matched_probe.py):
the H1 ordering is preserved, but the magnitude difference between
bottleneck and rich-encoder conditions grows substantially as the
per-episode step cap rises.  At cap ≤ 200, rich-encoder conditions
decode GPS at R² > 0.6; only when long-episode steps are added does
their R² drop to chance / sub-chance.

This script measures how the GPS/compass code stability evolves *along*
an episode — directly, not just under a length cap.

Method (out-of-fold predictions):
  1. For each condition, run 5-fold episode-level CV with a Ridge probe
     on the full data. Record per-test-sample (step_in_episode, true,
     predicted) tuples.  Each sample's prediction is from a probe that
     did NOT see that episode (clean OOF).
  2. Bin samples by step-in-episode into ranges
     [0, 25), [25, 50), [50, 100), [100, 200), [200, 400), [400, 800), [800, ∞).
  3. Compute R² per bin separately.

Output: JSON with per-condition × per-bin R² and sample counts.

Usage:
    python scripts/probing/temporal_probe.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --suffix _det \\
        --out /scratch/izar/wxu/probing_results/temporal_probe_det.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


CONDS = [
    "blind_gibson",
    "uniform_gibson",
    "foveated_gibson",
    "foveated_learned_gibson",
    "matched_gibson",
]

BINS = [
    (0, 25),
    (25, 50),
    (50, 100),
    (100, 200),
    (200, 400),
    (400, 800),
    (800, 100_000),  # 800+ catch-all
]


def oof_predictions(
    H: np.ndarray,
    Y: np.ndarray,
    ep_ids: np.ndarray,
    n_folds: int = 5,
    alpha: float = 10.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """5-fold CV; return (true, predicted) for every sample in OOF order.

    Each sample's predicted value is the prediction made by the fold-probe
    in which its episode was held out.
    """
    unique_eps = np.unique(ep_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_eps)
    kf = KFold(n_splits=n_folds, shuffle=False)

    n_samples = H.shape[0]
    if Y.ndim == 1:
        Y_target = Y[:, None]
    else:
        Y_target = Y
    pred = np.full_like(Y_target, np.nan, dtype=np.float32)

    for tri, tei in kf.split(unique_eps):
        train_mask = np.isin(ep_ids, unique_eps[tri])
        test_mask = np.isin(ep_ids, unique_eps[tei])
        sc = StandardScaler()
        Xtr = sc.fit_transform(H[train_mask])
        Xte = sc.transform(H[test_mask])
        Ytr = Y_target[train_mask]
        ridge = Ridge(alpha=alpha).fit(Xtr, Ytr)
        pred[test_mask] = ridge.predict(Xte).astype(np.float32)

    return Y_target, pred


def per_bin_r2(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    step: np.ndarray,
    bins: list[tuple[int, int]],
) -> list[dict]:
    """Compute per-bin R² across multi-output target."""
    out = []
    for lo, hi in bins:
        mask = (step >= lo) & (step < hi)
        n = int(mask.sum())
        if n < 50:
            out.append({
                "lo": lo, "hi": hi, "n": n,
                "r2": None,  # too few samples
            })
            continue
        r2 = r2_score(
            Y_true[mask], Y_pred[mask], multioutput="uniform_average",
        )
        out.append({
            "lo": lo, "hi": hi, "n": n,
            "r2": float(r2),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--suffix", type=str, default="_det")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"{'cond':<26} {'bin':<14} {'n':>8} {'GPS R^2':>10} {'Cmp R^2':>10}")
    print("-" * 75)
    results: dict = {"per_condition": {}}
    for cond in CONDS:
        p = args.in_dir / f"{cond}{args.suffix}.npz"
        if not p.exists():
            print(f"[skip] {p}")
            continue
        d = np.load(p)
        H = d["hidden_states"].astype(np.float32)
        gps = d["gps"]
        comp = d["compass"]
        comp_sc = np.column_stack([np.sin(comp), np.cos(comp)])
        step = d["step_in_episode"]
        ep_ids = d["episode_ids"]

        Y_gps, P_gps = oof_predictions(H, gps, ep_ids)
        Y_cmp, P_cmp = oof_predictions(H, comp_sc, ep_ids)

        gps_bins = per_bin_r2(Y_gps, P_gps, step, BINS)
        cmp_bins = per_bin_r2(Y_cmp, P_cmp, step, BINS)

        results["per_condition"][cond] = {
            "n_total": int(H.shape[0]),
            "gps_r2_per_bin": gps_bins,
            "compass_r2_per_bin": cmp_bins,
        }

        for g, c in zip(gps_bins, cmp_bins):
            bin_str = f"[{g['lo']}, {g['hi']})" if g['hi'] < 100_000 else f"[{g['lo']}+]"
            gv = "n/a" if g["r2"] is None else f"{g['r2']:+8.3f}"
            cv = "n/a" if c["r2"] is None else f"{c['r2']:+8.3f}"
            print(f"{cond:<26} {bin_str:<14} {g['n']:>8} {gv:>10} {cv:>10}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
