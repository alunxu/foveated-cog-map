"""
I1 verification: length-matched GPS probe.

Question: are the H1 GPS R² differences between conditions partly
explained by per-condition rollout-length differences (blind has
longer episodes than uniform because of lower SPL → more probe
training data)?

Method: subsample each condition's deterministic-rollout NPZ to a
common per-episode max-length, then re-run the 5-fold CV global GPS
probe.  If the H1 ordering is preserved, length is not the driver.

Usage:
    python scripts/probing/length_matched_probe.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out /scratch/izar/wxu/probing_results/length_matched_det.json
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


def truncate_per_episode(
    H: np.ndarray,
    Y: np.ndarray,
    ep_ids: np.ndarray,
    step_in_ep: np.ndarray,
    max_len: int,
) -> tuple:
    """Keep only steps where step_in_ep < max_len within each episode."""
    keep = step_in_ep < max_len
    return H[keep], Y[keep], ep_ids[keep]


def kfold_cv_r2(
    H: np.ndarray,
    Y: np.ndarray,
    ep_ids: np.ndarray,
    n_folds: int = 5,
    alpha: float = 10.0,
    seed: int = 42,
) -> tuple[float, float]:
    """Episode-level 5-fold CV R²."""
    unique_eps = np.unique(ep_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_eps)
    kf = KFold(n_splits=n_folds, shuffle=False)
    r2s = []
    for tri, tei in kf.split(unique_eps):
        train_mask = np.isin(ep_ids, unique_eps[tri])
        test_mask = np.isin(ep_ids, unique_eps[tei])
        sc = StandardScaler()
        Xtr = sc.fit_transform(H[train_mask])
        Xte = sc.transform(H[test_mask])
        Ytr, Yte = Y[train_mask], Y[test_mask]
        if Ytr.ndim == 1:
            Ytr = Ytr[:, None]
            Yte = Yte[:, None]
        ridge = Ridge(alpha=alpha).fit(Xtr, Ytr)
        pred = ridge.predict(Xte)
        r2s.append(r2_score(Yte, pred, multioutput="uniform_average"))
    return float(np.mean(r2s)), float(np.std(r2s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--suffix", type=str, default="_det")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--max-lens",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400],
        help="Per-episode step caps to evaluate (Y axis of the table).",
    )
    args = ap.parse_args()

    # Load all conditions
    data = {}
    for cond in CONDS:
        p = args.in_dir / f"{cond}{args.suffix}.npz"
        if not p.exists():
            print(f"[skip] {p}")
            continue
        d = np.load(p)
        data[cond] = {
            "H": d["hidden_states"].astype(np.float32),
            "gps": d["gps"],
            "compass": d["compass"],
            "ep_ids": d["episode_ids"],
            "step_in_ep": d["step_in_episode"],
        }
        print(f"  {cond}: total {len(d['episode_ids'])} steps")

    print(f"\n{'cond':<26} {'cap':>6} {'n_kept':>10} {'GPS_r2':>14} {'compass_r2':>16}")
    print("-" * 78)
    results: dict = {"per_cap": {}}
    for max_len in args.max_lens:
        results["per_cap"][max_len] = {}
        for cond, d in data.items():
            H, gps_kept, ep_kept = truncate_per_episode(
                d["H"], d["gps"], d["ep_ids"], d["step_in_ep"], max_len,
            )
            comp_full = d["compass"]
            keep = d["step_in_ep"] < max_len
            comp_kept = np.column_stack(
                [np.sin(comp_full[keep]), np.cos(comp_full[keep])]
            )
            gps_r2, gps_std = kfold_cv_r2(H, gps_kept, ep_kept)
            comp_r2, comp_std = kfold_cv_r2(H, comp_kept, ep_kept)
            results["per_cap"][max_len][cond] = {
                "n_kept": int(H.shape[0]),
                "gps_r2_mean": gps_r2,
                "gps_r2_std": gps_std,
                "compass_r2_mean": comp_r2,
                "compass_r2_std": comp_std,
            }
            print(
                f"{cond:<26} {max_len:>6} {H.shape[0]:>10} "
                f"{gps_r2:+8.3f}\u00b1{gps_std:.3f}  "
                f"{comp_r2:+8.3f}\u00b1{comp_std:.3f}"
            )
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
