"""
Per-step GPS decodability progression.

Asks: at step t within an episode, how well does the hidden state
decode GPS? If the foveated agent uses its recurrent memory to
accumulate compensatory spatial information, GPS R² should CLIMB with
episode step. Uniform, with reliable current frames, should be flat.

Implementation: we fit a SINGLE probe on the full dataset (all steps), then
report out-of-sample R² separately within step-bins. This is a "holistic
probe, stratified evaluation" — the probe is not specialised per bin, so
differences across bins reflect the quality of the underlying hidden
state as a function of how much the agent has accumulated.

Usage:
    python scripts/probing/per_step_progression.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out   /scratch/izar/wxu/probing_results/per_step_progression.json
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

STEP_BINS = [(0, 4), (5, 9), (10, 19), (20, 39), (40, 79), (80, 999)]


def split_by_episode(ep_ids, seed=0, test_frac=0.2):
    unique = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(test_frac * len(unique)))
    test_eps = set(unique[:n_test].tolist())
    test_mask = np.isin(ep_ids, list(test_eps))
    return ~test_mask, test_mask


def run_condition(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    X = d["hidden_states"].astype(np.float32)
    y = d["gps"].astype(np.float32)
    ep_ids = d["episode_ids"]
    step = d["step_in_episode"]

    tr_mask, te_mask = split_by_episode(ep_ids)

    clf = Ridge(alpha=10.0).fit(X[tr_mask], y[tr_mask])
    y_pred_all = clf.predict(X)

    out = {
        "n_test_total": int(te_mask.sum()),
        "r2_all": float(r2_score(y[te_mask], y_pred_all[te_mask],
                                  multioutput="uniform_average")),
        "by_step_bin": {},
    }
    for (lo, hi) in STEP_BINS:
        mask = te_mask & (step >= lo) & (step <= hi)
        n = int(mask.sum())
        if n < 5:
            out["by_step_bin"][f"{lo}-{hi}"] = {"n": n, "r2": None}
            continue
        r2 = float(r2_score(y[mask], y_pred_all[mask],
                            multioutput="uniform_average"))
        out["by_step_bin"][f"{lo}-{hi}"] = {"n": n, "r2": r2}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = {"per_condition": {}}
    for c in CONDITIONS:
        path = args.in_dir / f"{c}.npz"
        if not path.exists():
            continue
        print(f"\n=== {c} ===", flush=True)
        r = run_condition(path)
        print(f"  R²_all = {r['r2_all']:+.3f}  (n_test = {r['n_test_total']})")
        print(f"  step-bin | n      R²")
        for key in r["by_step_bin"]:
            entry = r["by_step_bin"][key]
            if entry["r2"] is None:
                print(f"  {key:>9} | {entry['n']:<6} (too few)")
            else:
                print(f"  {key:>9} | {entry['n']:<6} {entry['r2']:+.3f}")
        results["per_condition"][c] = r

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
