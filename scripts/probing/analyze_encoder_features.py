"""
Probe encoder feature output (post ResNet, pre-LSTM) for GPS / compass.

Companion to collect_encoder_features.py.  Reads
<cond>_encfeat_det.npz files containing flattened encoder feature
vectors + ground-truth labels, and runs 5-fold episode-level CV ridge
probes for GPS, compass, distance-to-goal.

Compares to the LSTM hidden-state probe (analyze.py 1b_global_gps_compass)
to diagnose whether the LSTM-state divergence (H1) is explained by the
encoder output already discarding the relevant information, or by the
LSTM downstream of the encoder doing something different.

Usage:
    python scripts/probing/analyze_encoder_features.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out /scratch/izar/wxu/probing_results/encoder_features_det.json \\
        --conditions matched uniform foveated
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


def cv_probe_r2(
    X: np.ndarray, Y: np.ndarray, ep_ids: np.ndarray,
    n_folds: int = 5, alpha: float = 10.0, seed: int = 42,
) -> dict:
    unique_eps = np.unique(ep_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_eps)
    kf = KFold(n_splits=n_folds, shuffle=False)
    r2s = []
    for tri, tei in kf.split(unique_eps):
        train_mask = np.isin(ep_ids, unique_eps[tri])
        test_mask = np.isin(ep_ids, unique_eps[tei])
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[train_mask])
        Xte = sc.transform(X[test_mask])
        if Y.ndim == 1:
            Ytr, Yte = Y[train_mask, None], Y[test_mask, None]
        else:
            Ytr, Yte = Y[train_mask], Y[test_mask]
        ridge = Ridge(alpha=alpha).fit(Xtr, Ytr)
        pred = ridge.predict(Xte)
        r2s.append(r2_score(Yte, pred, multioutput="uniform_average"))
    return {"r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["matched", "uniform", "foveated"],
    )
    ap.add_argument("--suffix", type=str, default="_encfeat_det")
    args = ap.parse_args()

    print(f"{'condition':<26} {'enc_dim':>8} {'GPS_r2':>14} {'Comp_r2':>14}")
    print("-" * 70)
    results = {}
    for cond in args.conditions:
        p = args.in_dir / f"{cond}_gibson{args.suffix}.npz"
        if not p.exists():
            print(f"[skip] {p}")
            continue
        d = np.load(p)
        feats = d["encoder_features"].astype(np.float32)
        gps = d["gps"]
        comp = d["compass"]
        comp_sc = np.column_stack([np.sin(comp), np.cos(comp)])
        ep_ids = d["episode_ids"]

        gps_r = cv_probe_r2(feats, gps, ep_ids)
        cmp_r = cv_probe_r2(feats, comp_sc, ep_ids)
        results[cond] = {
            "n_steps": int(feats.shape[0]),
            "encoder_dim": int(feats.shape[1]),
            "encoder_features_gps_r2_mean": gps_r["r2_mean"],
            "encoder_features_gps_r2_std": gps_r["r2_std"],
            "encoder_features_compass_r2_mean": cmp_r["r2_mean"],
            "encoder_features_compass_r2_std": cmp_r["r2_std"],
        }
        print(
            f"{cond:<26} {feats.shape[1]:>8} "
            f"{gps_r['r2_mean']:+8.3f}\u00b1{gps_r['r2_std']:.3f}  "
            f"{cmp_r['r2_mean']:+8.3f}\u00b1{cmp_r['r2_std']:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
