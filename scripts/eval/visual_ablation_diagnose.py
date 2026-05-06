"""Diagnostic for the surprising visual-ablation result.

Paper §4.2 substitution prediction was: rich-encoder LSTM should re-engage
the integrated GPS code post-visual-ablation -- post-segment MAE/spread
should DROP relative to pre. Actual result: uniform post = 2.41 vs pre = 1.22
(post WORSE than pre by ~2x), foveated +0.44, coarse only +0.19, blind +0.37.

Before re-framing the paper, rule out the following artifacts:

  (A) BEHAVIORAL: agents simply fail / freeze post-ablation -> position spread
      collapses -> MAE/spread inflates artificially.

  (B) PROBE EXTRAPOLATION: post-segment dominates data (coarse 30:1 ratio);
      probe fit on full-episode data is essentially a post-segment probe.
      Pre-segment falls outside training-distribution.

  (C) H-STATE DISTRIBUTION SHIFT: zeroed-out visual input is never seen during
      training -> post-segment h-state goes off-manifold. Probe can't decode
      because state distribution itself shifted, not because GPS info is gone.

Diagnostic per condition + segment:
  - n_episodes, n_samples
  - mean episode length (steps)
  - SPL proxy: success rate at end of episode (final dist < 0.2)
  - position-spread (absolute, in meters): sqrt(mean ||pos - mean||^2)
  - MAE (absolute, in meters)
  - h-state mean Euclidean distance to PRE-segment centroid (distribution shift)

Reads:  --in-dir <dir>/{cond}_visual_ablation_at50.npz
Writes: <out>.json with diagnostic per (cond, segment).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
SUCCESS_DIST = 0.2


def diagnose_one(npz_path: Path) -> dict:
    d = np.load(npz_path)
    H = d["hidden_states"].astype(np.float64)
    pos = d["positions"][:, [0, 2]].astype(np.float64)
    step = d["step_in_episode"]
    ep_ids = d["episode_ids"]
    dist_goal = d["distance_to_goal"].astype(np.float64)

    # Per-episode terminal stats
    unique_eps = np.unique(ep_ids)
    final_dists = []
    ep_lengths = []
    for ep in unique_eps:
        em = ep_ids == ep
        ep_steps = step[em]
        last = np.argmax(ep_steps)
        final_dists.append(dist_goal[em][last])
        ep_lengths.append(ep_steps.max() + 1)
    final_dists = np.array(final_dists)
    ep_lengths = np.array(ep_lengths)

    # Pre-segment centroid (for distribution-shift check)
    pre_mask = step < ABLATION_STEP
    pre_centroid = H[pre_mask].mean(axis=0)

    out = {
        "n_episodes": int(len(unique_eps)),
        "n_samples": int(len(H)),
        "mean_episode_length_steps": float(np.mean(ep_lengths)),
        "median_episode_length_steps": float(np.median(ep_lengths)),
        "final_dist_mean_m": float(np.mean(final_dists)),
        "success_rate_proxy": float((final_dists < SUCCESS_DIST).mean()),
        "segments": {},
    }

    for seg_name, (lo, hi) in SEGMENTS.items():
        sm = (step >= lo) & (step < hi)
        if sm.sum() < 5:
            continue
        seg_pos = pos[sm]
        seg_h = H[sm]
        # Position spread (absolute, in meters)
        spread = float(np.sqrt(np.mean(np.sum((seg_pos - seg_pos.mean(axis=0)) ** 2, axis=1))))
        # H-state distance from pre-segment centroid
        h_centroid_dist = float(np.mean(np.linalg.norm(seg_h - pre_centroid, axis=1)))
        out["segments"][seg_name] = {
            "n_samples": int(sm.sum()),
            "n_samples_frac": float(sm.sum() / len(H)),
            "position_spread_m": spread,
            "h_dist_from_pre_centroid": h_centroid_dist,
            "step_range": (lo, hi),
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = {}
    for c in CONDS:
        npz = args.in_dir / f"{COND_NPZ_MAP[c]}_visual_ablation_at50.npz"
        if not npz.exists():
            print(f"  {c}: missing -- skip")
            continue
        print(f"\n  {c} <- {npz.name}")
        r = diagnose_one(npz)
        results[c] = r

        print(f"    n_eps={r['n_episodes']}  N={r['n_samples']}  "
              f"avg_ep_len={r['mean_episode_length_steps']:.0f} steps  "
              f"final_dist={r['final_dist_mean_m']:.2f}m  "
              f"success={r['success_rate_proxy']:.2f}")
        for sn, s in r["segments"].items():
            print(f"    [{sn}] n={s['n_samples']}  "
                  f"({s['n_samples_frac']*100:.1f}%)  "
                  f"pos_spread={s['position_spread_m']:.2f}m  "
                  f"h_dist={s['h_dist_from_pre_centroid']:.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}\n")

    # Diagnostic interpretation
    print("=" * 70)
    print("Diagnostic table:")
    print(f"{'cond':<10} {'success':<10} {'pre/post n ratio':<18} "
          f"{'pre/post pos_spread ratio':<26} {'h-state shift (post)':<22}")
    for c, r in results.items():
        pre = r["segments"].get("pre", {})
        post = r["segments"].get("post", {})
        if not pre or not post:
            continue
        n_ratio = pre["n_samples"] / max(post["n_samples"], 1)
        ps_ratio = (pre["position_spread_m"]
                    / max(post["position_spread_m"], 1e-9))
        h_shift = post["h_dist_from_pre_centroid"]
        print(f"{c:<10} {r['success_rate_proxy']:<10.2f} "
              f"{n_ratio:<18.3f} {ps_ratio:<26.3f} {h_shift:<22.1f}")

    print(
        "\nInterpretation rules of thumb:\n"
        "  - success ~ 0 means agent failed entirely post-ablation\n"
        "  - n_ratio << 1 means post-segment dominates; if probe trained on full-ep,\n"
        "    pre-segment is treated as out-of-distribution by the probe\n"
        "  - pos_spread ratio: if pre/post pos_spread ratio is small, post has\n"
        "    larger position range; if large, post got stuck (artifact for MAE/spread)\n"
        "  - h-state shift LARGE means post-ablation h-state is off-manifold relative to\n"
        "    pre, supporting (C) distribution-shift hypothesis.\n"
    )


if __name__ == "__main__":
    main()
