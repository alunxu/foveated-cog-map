"""WJ-F analyzer: per-segment GPS R² + recovery probe.

Processes excursion_forgetting.py NPZ output (h_2 + position + segment) and
fits a linear probe h_2 → (x, z) per segment (warmup / detour / recovery).
The decision rule from project_wijmans_replication_plan.md:

  - Bottleneck conditions (blind, matched128 coarse): if path-integration
    is the only mechanism, the LSTM's h_2 should encode position throughout
    BUT the recovery-segment R² should drop relative to warmup, because
    the forced random detour injects noise into the path-integral.
  - Rich-encoder conditions (uniform, foveated, matched): if the LSTM
    leans on direct visual cues, recovery R² should be ~= warmup R².
    (Vision pipes through regardless of detour.)

Reads:  --in-dir <dir>/{cond}_det.npz   from excursion_forgetting.py
Writes: <out-path>.json with per-condition {warmup, detour, recovery}
        R² (5-fold CV) for both x and z components.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


CONDS = ["blind", "matched128", "matched", "uniform", "foveated"]
SEGMENTS = {0: "warmup", 1: "detour", 2: "recovery"}


def cv_r2(X: np.ndarray, y: np.ndarray, alpha: float, n_folds: int) -> tuple[float, float]:
    """5-fold CV R² for ridge regression. Returns (mean, std)."""
    if len(X) < 50:
        return float("nan"), float("nan")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    r2s = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha)
        m.fit(X[tr], y[tr])
        ss_res = ((y[te] - m.predict(X[te])) ** 2).sum()
        ss_tot = ((y[te] - y[tr].mean(axis=0)) ** 2).sum()
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-9))
    return float(np.mean(r2s)), float(np.std(r2s))


def analyze_one(npz_path: Path, alpha: float, n_folds: int) -> dict:
    d = np.load(npz_path)
    h = d["hidden_states"]            # (N, 512)
    pos = d["positions"][:, [0, 2]]   # (N, 2) — XZ plane
    seg = d["segments"]               # (N,) int8 with 0/1/2
    print(f"  {npz_path.stem}: N={len(h)}, segments=" +
          ", ".join(f"{name}={int((seg==i).sum())}" for i, name in SEGMENTS.items()))

    out = {
        "n_total": int(len(h)),
        "warmup_steps": int(d.get("wjf_warmup_steps", 50)),
        "detour_steps": int(d.get("wjf_detour_steps", 25)),
        "recovery_steps": int(d.get("wjf_recovery_steps", 100)),
        "alpha": alpha,
        "n_cv_folds": n_folds,
        "segments": {},
    }
    for sid, name in SEGMENTS.items():
        mask = (seg == sid)
        n = int(mask.sum())
        if n < 50:
            out["segments"][name] = {"n_steps": n, "gps_cv_r2_mean": float("nan"),
                                     "gps_cv_r2_std": float("nan")}
            continue
        Xs = h[mask]
        ys = pos[mask]
        r2_x, std_x = cv_r2(Xs, ys[:, 0], alpha, n_folds)
        r2_z, std_z = cv_r2(Xs, ys[:, 1], alpha, n_folds)
        # Combined R²: average of x and z (proxy for joint position recovery).
        out["segments"][name] = {
            "n_steps": n,
            "gps_x_cv_r2_mean": r2_x, "gps_x_cv_r2_std": std_x,
            "gps_z_cv_r2_mean": r2_z, "gps_z_cv_r2_std": std_z,
            "gps_cv_r2_mean": (r2_x + r2_z) / 2.0,
            "gps_cv_r2_std": (std_x + std_z) / 2.0,
        }

    # Recovery delta — primary readout
    w = out["segments"].get("warmup", {}).get("gps_cv_r2_mean", float("nan"))
    r = out["segments"].get("recovery", {}).get("gps_cv_r2_mean", float("nan"))
    out["delta_recovery_minus_warmup"] = float(r - w) if (
        np.isfinite(w) and np.isfinite(r)) else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    results: dict = {}
    print(f"Analyzing excursion data from {args.in_dir}")
    for c in CONDS:
        npz = args.in_dir / f"{c}_det.npz"
        if not npz.exists():
            print(f"  {c}: missing — skip")
            continue
        results[c] = analyze_one(npz, args.alpha, args.n_folds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")

    # Print summary
    print("\n=== Per-segment GPS R² (5-fold CV) ===")
    print(f"{'cond':<12} {'warmup':<10} {'detour':<10} {'recovery':<10} {'Δ recov-warm':<12}")
    for c, r in results.items():
        w = r["segments"].get("warmup", {}).get("gps_cv_r2_mean", float("nan"))
        d = r["segments"].get("detour", {}).get("gps_cv_r2_mean", float("nan"))
        rec = r["segments"].get("recovery", {}).get("gps_cv_r2_mean", float("nan"))
        delta = r["delta_recovery_minus_warmup"]
        print(f"{c:<12} {w:>+.3f}    {d:>+.3f}    {rec:>+.3f}    {delta:>+.3f}")


if __name__ == "__main__":
    main()
