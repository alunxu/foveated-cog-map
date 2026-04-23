"""
Extended-lag retrospective-memory probe (k=0..20) for H1.

The current paper reports path-history R² at lag k∈{0,...,5}. This extends
the sweep to k=20 to characterise how long-lived the compensatory trace is
in each condition. Shows whether the foveated>uniform gap persists
indefinitely or plateaus.

Usage:
    python scripts/probing/extended_lag_probe.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out   /scratch/izar/wxu/probing_results/extended_lag.json \\
        --max-lag 20
"""
from __future__ import annotations

import argparse
import json
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


def split_by_episode(ep_ids, seed=0, test_frac=0.2):
    unique = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(test_frac * len(unique)))
    test_eps = set(unique[:n_test].tolist())
    test_mask = np.isin(ep_ids, list(test_eps))
    return ~test_mask, test_mask


def build_lag_pairs(h, positions, ep_ids, lag):
    """Pairs (h_t, position_{t-lag}) where both are in the same episode."""
    X, y, ep = [], [], []
    for e in np.unique(ep_ids):
        idx = np.where(ep_ids == e)[0]
        if len(idx) <= lag:
            continue
        for i in range(lag, len(idx)):
            X.append(h[idx[i]])
            y.append(positions[idx[i - lag]])
            ep.append(e)
    return np.stack(X) if X else None, np.stack(y) if y else None, np.array(ep) if ep else None


def probe_lag(X, y, ep_rep, seed=0):
    tr, te = split_by_episode(ep_rep, seed=seed)
    clf = Ridge(alpha=10.0).fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return float(r2_score(y[te], pred, multioutput="uniform_average")), \
           float(np.abs(y[te] - pred).mean()), \
           int(tr.sum()), int(te.sum())


def run_condition(path: Path, max_lag: int) -> dict:
    d = np.load(path, allow_pickle=True)
    X = d["hidden_states"].astype(np.float32)
    # Use ego-frame GPS (start-of-episode relative) rather than world-frame
    # `positions`; world-frame coords are scene-dependent and never linearly
    # decodable. GPS is the same target used by the main-text lag probe.
    positions = d["gps"].astype(np.float32)
    ep_ids = d["episode_ids"]

    out = {"n_steps": int(X.shape[0]), "n_episodes": int(len(np.unique(ep_ids))),
           "lag_results": {}}
    for k in range(max_lag + 1):
        pairs = build_lag_pairs(X, positions, ep_ids, k)
        if pairs[0] is None:
            out["lag_results"][str(k)] = {"r2": None, "reason": "insufficient samples"}
            continue
        X_k, y_k, ep_k = pairs
        r2, mae, n_tr, n_te = probe_lag(X_k, y_k, ep_k)
        out["lag_results"][str(k)] = {
            "r2": r2, "mae_m": mae, "n_train": n_tr, "n_test": n_te,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-lag", type=int, default=20)
    args = ap.parse_args()

    results = {"max_lag": args.max_lag, "per_condition": {}}
    for c in CONDITIONS:
        path = args.in_dir / f"{c}.npz"
        if not path.exists():
            continue
        print(f"\n=== {c} ===", flush=True)
        r = run_condition(path, args.max_lag)
        results["per_condition"][c] = r
        print(f"  lag | R²      MAE    n_test")
        for k in range(args.max_lag + 1):
            entry = r["lag_results"][str(k)]
            if entry.get("r2") is None:
                print(f"  {k:>3} | (insufficient samples)")
            else:
                print(f"  {k:>3} | {entry['r2']:+.3f}  {entry['mae_m']:.2f}  {entry['n_test']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
