"""Analyze visual-ablation rollout NPZs.

Tests §4.2 substitution mechanism causally: does rich-encoder LSTM h_2
re-engage integrated GPS code when visual route is severed mid-rollout?

Reads:  --in-dir <dir>/{cond}_visual_ablation_at50.npz   (collect.py output)
Writes: <out-path>.json with per-condition pre-vs-post-ablation GPS R².

Probe protocol:
  - Pre-ablation segment: steps 0–49 (normal vision)
  - Post-ablation segment: steps 50–149 (visual obs zeroed)
  - Train one Ridge probe on full-episode pooled data (episode-level
    5-fold CV); test on each segment of held-out fold; report MAE / spread.

Substitution prediction:
  - Rich-encoder (uniform, foveated, foveated_learned): post-ablation
    MAE/spread should DROP relative to pre-ablation (LSTM falls back to
    integrated GPS, which is more linearly decodable).
  - Bottleneck (blind, matched): roughly unchanged, both segments use
    integration.

If rich-encoder shows post < pre AND bottleneck shows post ≈ pre, the
substitution mechanism is causally supported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# Display name (paper §4) -> NPZ stem (legacy training cond name).
# "coarse" was historically named "matched" (compute-matched to other sighted
# conditions); both refer to the same agent (48x48 RGB -> 1x1 ResNet feature).
COND_NPZ_MAP = {
    "blind": "blind",
    "coarse": "matched",
    "uniform": "uniform",
    "foveated": "foveated",
    "foveated_learned": "foveated_learned",
}
CONDS = list(COND_NPZ_MAP.keys())
ABLATION_STEP = 50
SEGMENTS = {"pre": (0, ABLATION_STEP), "post": (ABLATION_STEP, 1000)}


def per_segment_metrics(X, y, step, ep_ids, alpha=10.0, n_folds=5):
    """Episode-level 5-fold CV; one probe per fold trained on full-episode
    data, tested on each segment of held-out episodes."""
    unique_eps = np.unique(ep_ids)
    if len(unique_eps) < n_folds * 2:
        n_folds = max(2, len(unique_eps) // 2)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    seg_results = {name: {"maes": [], "spreads": []} for name in SEGMENTS}
    for tr_idx, te_idx in kf.split(unique_eps):
        tr_eps, te_eps = unique_eps[tr_idx], unique_eps[te_idx]
        tr_mask = np.isin(ep_ids, tr_eps)
        te_mask = np.isin(ep_ids, te_eps)
        if tr_mask.sum() < 100 or te_mask.sum() < 30:
            continue
        m = Ridge(alpha=alpha)
        m.fit(X[tr_mask], y[tr_mask])
        pred = m.predict(X[te_mask])
        true = y[te_mask]
        steps_te = step[te_mask]

        for seg_name, (lo, hi) in SEGMENTS.items():
            sm = (steps_te >= lo) & (steps_te < hi)
            if sm.sum() < 10:
                continue
            mae = float(np.mean(np.linalg.norm(pred[sm] - true[sm], axis=1)))
            spread = float(np.sqrt(np.mean(
                np.sum((true[sm] - true[sm].mean(axis=0)) ** 2, axis=1)
            )))
            seg_results[seg_name]["maes"].append(mae)
            seg_results[seg_name]["spreads"].append(spread)

    out = {}
    for seg_name in SEGMENTS:
        maes = np.array(seg_results[seg_name]["maes"])
        spreads = np.array(seg_results[seg_name]["spreads"])
        if len(maes) == 0:
            out[seg_name] = {"mae": float("nan"), "spread": float("nan"),
                             "mae_over_spread": float("nan")}
            continue
        out[seg_name] = {
            "n_folds": int(len(maes)),
            "mae": float(maes.mean()),
            "spread": float(spreads.mean()),
            "mae_over_spread": float((maes / np.maximum(spreads, 1e-9)).mean()),
        }
    return out


def analyze_one(npz_path: Path) -> dict:
    d = np.load(npz_path)
    h = d["hidden_states"]
    pos = d["positions"][:, [0, 2]]
    step = d["step_in_episode"]
    ep_ids = d["episode_ids"]
    print(f"  {npz_path.stem}: N={len(h)}, eps={len(np.unique(ep_ids))}, "
          f"pre={int((step < ABLATION_STEP).sum())}, "
          f"post={int((step >= ABLATION_STEP).sum())}")

    return {
        "n_total": int(len(h)),
        "n_episodes": int(len(np.unique(ep_ids))),
        "ablation_step": ABLATION_STEP,
        "segments": per_segment_metrics(h, pos, step, ep_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = {}
    for c in CONDS:
        npz = args.in_dir / f"{COND_NPZ_MAP[c]}_visual_ablation_at50.npz"
        if not npz.exists():
            print(f"  {c} (npz '{COND_NPZ_MAP[c]}_*'): missing — skip")
            continue
        results[c] = analyze_one(npz)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}\n")

    # Summary table
    print(f"{'cond':<18} {'pre MAE/spr':<15} {'post MAE/spr':<15} {'Δ (post-pre)':<10}")
    for c, r in results.items():
        s = r["segments"]
        pre = s.get("pre", {}).get("mae_over_spread", float("nan"))
        post = s.get("post", {}).get("mae_over_spread", float("nan"))
        delta = post - pre
        print(f"{c:<18} {pre:>+.3f}          {post:>+.3f}          {delta:>+.3f}")
    print("\nSubstitution prediction: rich-encoder post < pre (Δ < 0); bottleneck Δ ≈ 0")


if __name__ == "__main__":
    main()
