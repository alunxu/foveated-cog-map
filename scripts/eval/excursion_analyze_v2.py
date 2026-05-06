"""WJ-F analyzer v2: variance-matched per-segment metric.

The v1 analyzer (per-segment 5-fold CV R²) showed all conditions recover>warmup
with $\\Delta$ +0.14 to +0.23 — likely a position-variance artefact, not a
forgetting signal. Warmup (50 steps walking toward goal) covers a small
spatial footprint; detour (25 random) + recovery (100 steps) cover much more.
R² is variance-normalized so it inflates when target spread is larger.

This v2 fixes the confound by:

  1. **Train probe ONCE** on the full-episode (all-segment) pool, episode-
     level 5-fold CV. The same probe is tested on every segment of each
     held-out fold.
  2. **MAE-based metric** per segment, plus **MAE / position-spread**
     (scale-invariant). If the GPS encoding is consistent across segments,
     MAE/spread is similar across warmup/detour/recovery. If detour breaks
     the integrated code, recovery's MAE/spread should increase even
     though raw MAE might not.

Reads:  --in-dir <dir>/{cond}_det.npz   from excursion_forgetting.py
Writes: <out-path>.json with per-condition per-segment {mae, spread, mae_over_spread}.
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


def episode_cv_mae(X, y, episode_ids, segments, alpha, n_folds):
    """Episode-level CV. Train ridge on (n_folds-1) groups of episodes' full
    data (all segments mixed); test on held-out group. For each held-out
    episode-segment, compute MAE and position-spread."""
    unique_eps = np.unique(episode_ids)
    if len(unique_eps) < n_folds * 2:
        n_folds = max(2, len(unique_eps) // 2)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_results = {sid: {"maes": [], "spreads": []} for sid in SEGMENTS}

    for tr_eps_idx, te_eps_idx in kf.split(unique_eps):
        tr_eps = unique_eps[tr_eps_idx]
        te_eps = unique_eps[te_eps_idx]
        tr_mask = np.isin(episode_ids, tr_eps)
        te_mask = np.isin(episode_ids, te_eps)
        if tr_mask.sum() < 100 or te_mask.sum() < 30:
            continue
        m = Ridge(alpha=alpha)
        m.fit(X[tr_mask], y[tr_mask])
        pred = m.predict(X[te_mask])
        true = y[te_mask]
        seg_test = segments[te_mask]

        for sid in SEGMENTS:
            sm = (seg_test == sid)
            if sm.sum() < 10:
                continue
            mae = float(np.mean(np.linalg.norm(pred[sm] - true[sm], axis=1)))
            # Position spread = RMS distance from segment-mean (per fold-segment)
            spread = float(np.sqrt(np.mean(
                np.sum((true[sm] - true[sm].mean(axis=0)) ** 2, axis=1)
            )))
            fold_results[sid]["maes"].append(mae)
            fold_results[sid]["spreads"].append(spread)

    out = {}
    for sid, name in SEGMENTS.items():
        maes = np.array(fold_results[sid]["maes"], dtype=float)
        spreads = np.array(fold_results[sid]["spreads"], dtype=float)
        if len(maes) == 0:
            out[name] = {"n_folds": 0, "mae_mean": float("nan"),
                         "spread_mean": float("nan"),
                         "mae_over_spread": float("nan")}
            continue
        out[name] = {
            "n_folds": int(len(maes)),
            "mae_mean": float(maes.mean()),
            "mae_std": float(maes.std()),
            "spread_mean": float(spreads.mean()),
            "spread_std": float(spreads.std()),
            "mae_over_spread": float((maes / np.maximum(spreads, 1e-9)).mean()),
        }
    return out


def analyze_one(npz_path: Path, alpha: float, n_folds: int) -> dict:
    d = np.load(npz_path)
    h = d["hidden_states"]
    pos = d["positions"][:, [0, 2]]
    seg = d["segments"]
    ep_ids = d["episode_ids"]
    print(f"  {npz_path.stem}: N={len(h)}, "
          f"episodes={len(np.unique(ep_ids))}, segments=" +
          ", ".join(f"{name}={int((seg==i).sum())}" for i, name in SEGMENTS.items()))

    out = {
        "n_total": int(len(h)),
        "n_episodes": int(len(np.unique(ep_ids))),
        "warmup_steps": int(d.get("wjf_warmup_steps", 50)),
        "detour_steps": int(d.get("wjf_detour_steps", 25)),
        "recovery_steps": int(d.get("wjf_recovery_steps", 100)),
        "alpha": alpha,
        "n_cv_folds": n_folds,
        "segments": episode_cv_mae(h, pos, ep_ids, seg, alpha, n_folds),
    }

    # Forgetting signal: recovery vs warmup MAE/spread.
    # If GPS encoding consistent → mae_over_spread same → no forgetting.
    # If detour breaks integrated code → recovery mae_over_spread up → forgetting.
    w = out["segments"].get("warmup", {}).get("mae_over_spread", float("nan"))
    r = out["segments"].get("recovery", {}).get("mae_over_spread", float("nan"))
    out["forgetting_index"] = float(r - w) if (
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
    print(f"Analyzing excursion data from {args.in_dir} with v2 metric")
    for c in CONDS:
        npz = args.in_dir / f"{c}_det.npz"
        if not npz.exists():
            print(f"  {c}: missing — skip")
            continue
        results[c] = analyze_one(npz, args.alpha, args.n_folds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")

    # Summary table
    print("\n=== Per-segment MAE / position-spread (variance-matched, ep-CV) ===")
    print(f"{'cond':<12} {'warmup':<24} {'detour':<24} {'recovery':<24} {'forget'}")
    print(f"{'':<12} {'mae | spread | m/s':<24} {'mae | spread | m/s':<24} {'mae | spread | m/s':<24} {'(rec-warm m/s)'}")
    for c, r in results.items():
        s = r["segments"]
        line = [f"{c:<12}"]
        for sid_name in ["warmup", "detour", "recovery"]:
            seg = s.get(sid_name, {})
            mae = seg.get("mae_mean", float("nan"))
            spread = seg.get("spread_mean", float("nan"))
            mos = seg.get("mae_over_spread", float("nan"))
            line.append(f"{mae:>5.2f} | {spread:>5.2f} | {mos:>5.3f}")
        line.append(f"{r['forgetting_index']:>+.3f}")
        print(" ".join(line))


if __name__ == "__main__":
    main()
