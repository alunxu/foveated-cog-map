"""
Analysis-only follow-up: probe the LSTM cell state c_t and lower-layer
hidden states h_0 / h_1 (in addition to the top-layer h_2 reported in
the main paper). Resolves §3.2 fn4 / MASTER_TRACK §3.0b O2.

Reads existing collected probing data:
  /scratch/izar/wxu/probing_data/{cond}_gibson_det.npz

The npz files contain `h_layers` (N, 3, 512) and `c_layers` (N, 3, 512)
already, so this is true analysis-only — no rollouts needed.

For each condition we probe GPS + compass at all 6 (h, c) × (L0, L1, L2)
combinations using the same Ridge / 5-fold episode-level CV protocol as
the main paper.

Outputs:  <out-dir>/<cond>_extra_states.json  (per-condition Ridge R²)
          <out-dir>/extra_states_summary.json  (combined)

Run on Izar (where data is):
    conda activate cs503_project
    python scripts/probing/analyze_extra_states.py \\
        --data-dir /scratch/izar/wxu/probing_data \\
        --out-dir /scratch/izar/wxu/probing_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


CONDS_DEFAULT = [
    "blind", "matched", "uniform", "foveated", "foveated_learned",
]


def fit_per_state(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  alpha: float = 10.0, n_splits: int = 5) -> dict:
    """Episode-level 5-fold Ridge with mean ± std R²."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []
    for tr, te in gkf.split(X, y, groups):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        fold_r2.append(r2_score(y[te], m.predict(X[te])))
    return {
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2)),
        "fold_r2": [float(r) for r in fold_r2],
        "n_samples": int(X.shape[0]),
    }


def analyze_condition(npz_path: Path, alpha: float = 10.0) -> dict:
    """Probe GPS + compass at (h, c) × (L0, L1, L2) = 6 representations."""
    print(f"  loading {npz_path.name}")
    d = np.load(npz_path, allow_pickle=True)

    gps = d["gps"]              # (N, 2)
    compass = d["compass"]      # (N, 1)
    episode_ids = d["episode_ids"]
    h_layers = d["h_layers"]    # (N, 3, 512)
    c_layers = d["c_layers"]    # (N, 3, 512)
    n = h_layers.shape[0]

    print(f"    n={n} samples, "
          f"h_layers={h_layers.shape}, c_layers={c_layers.shape}")

    out = {}
    for state_name, layers in [("h", h_layers), ("c", c_layers)]:
        for layer_idx in range(layers.shape[1]):
            X = layers[:, layer_idx, :]
            for target_name, target in [("gps", gps),
                                         ("compass", compass)]:
                key = f"{state_name}_layer{layer_idx}_{target_name}"
                print(f"    probing {key} ...")
                out[key] = fit_per_state(X, target, episode_ids,
                                          alpha=alpha)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--conds", nargs="+", default=CONDS_DEFAULT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cond in args.conds:
        npz = args.data_dir / f"{cond}_gibson_det.npz"
        if not npz.exists():
            print(f"[skip] {cond}: {npz} missing")
            continue
        print(f"=== {cond} ===")
        out = analyze_condition(npz, alpha=args.alpha)
        summary[cond] = out
        out_path = args.out_dir / f"{cond}_extra_states.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path}")

    sum_path = args.out_dir / "extra_states_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== summary written to {sum_path} ===")
    print("\nGPS R² by (state × layer × condition):")
    print(f"{'state':<3} {'layer':<5} {'cond':<20} {'r2_mean':>10}")
    for cond, data in summary.items():
        for key, val in data.items():
            if "gps" in key:
                state, layer, _ = key.split("_")
                print(f"{state:<3} {layer:<5} {cond:<20} {val['r2_mean']:+10.3f}")


if __name__ == "__main__":
    main()
