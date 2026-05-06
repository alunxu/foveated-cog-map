"""Analyze GPS+compass ablation rollout NPZs.

Tests §4.5 dissociation claim: bottleneck (blind, matched) policy reads
integrated GPS; rich-encoder (uniform, foveated) substitutes visual route.
Predict: bottleneck SPL drops sharply when GPS sensor masked from step 0;
rich-encoder less affected.

Reads:  --in-dir <dir>/{cond}_gps_masked.npz   (collect.py + --mask-gps --mask-compass)
        --baseline-dir <dir>/{cond}_gibson_det.npz   (existing no-mask data)

Computes per condition:
  - mean final distance_to_goal (lower = better, proxy for navigation)
  - success rate proxy = fraction of episodes with final dist < 0.2m
  - delta vs baseline (positive = SPL hurt)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
SUCCESS_DIST = 0.2  # habitat default success threshold


def episode_stats(npz_path: Path) -> tuple[float, float, int]:
    """Per-episode final distance_to_goal mean, success rate, n_eps."""
    d = np.load(npz_path)
    dist = d["distance_to_goal"]
    ep_ids = d["episode_ids"]
    step = d["step_in_episode"]

    # Last-step distance per episode
    final_dists = []
    for ep in np.unique(ep_ids):
        ep_mask = (ep_ids == ep)
        ep_steps = step[ep_mask]
        last_idx = np.argmax(ep_steps)
        final_dists.append(dist[ep_mask][last_idx])
    final_dists = np.array(final_dists)
    return (float(final_dists.mean()),
            float((final_dists < SUCCESS_DIST).mean()),
            len(final_dists))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="GPS-ablation NPZ dir (mask-gps + mask-compass)")
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="Baseline (no-mask) NPZ dir for comparison")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ablation-suffix", default="_gps_masked")
    ap.add_argument("--baseline-suffix", default="_gibson_det")
    args = ap.parse_args()

    results = {}
    for c in CONDS:
        npz_stem = COND_NPZ_MAP[c]
        ab_path = args.in_dir / f"{npz_stem}{args.ablation_suffix}.npz"
        bl_path = args.baseline_dir / f"{npz_stem}{args.baseline_suffix}.npz"

        ab_dist, ab_success, ab_n = (float("nan"), float("nan"), 0) \
            if not ab_path.exists() else episode_stats(ab_path)
        bl_dist, bl_success, bl_n = (float("nan"), float("nan"), 0) \
            if not bl_path.exists() else episode_stats(bl_path)

        results[c] = {
            "ablation": {"final_dist": ab_dist, "success_rate": ab_success, "n_episodes": ab_n},
            "baseline": {"final_dist": bl_dist, "success_rate": bl_success, "n_episodes": bl_n},
            "delta_dist": ab_dist - bl_dist,            # positive = ablation hurt
            "delta_success": ab_success - bl_success,   # negative = ablation hurt
        }
        print(f"  {c}: ablation final_dist={ab_dist:.2f} success={ab_success:.2f}  vs  "
              f"baseline final_dist={bl_dist:.2f} success={bl_success:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}\n")

    print(f"{'cond':<18} {'baseline succ':<14} {'ablate succ':<14} {'Δ success':<10}")
    for c, r in results.items():
        bl_s = r["baseline"]["success_rate"]
        ab_s = r["ablation"]["success_rate"]
        print(f"{c:<18} {bl_s:>+.2f}          {ab_s:>+.2f}          {(ab_s-bl_s):>+.2f}")
    print("\nPrediction (§4.5 'policy reads integrated GPS'): bottleneck (blind/matched) "
          "Δ-success more negative than rich-encoder (uniform/foveated)")


if __name__ == "__main__":
    main()
