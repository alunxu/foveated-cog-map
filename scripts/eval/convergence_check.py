"""Per-condition convergence trajectory analyzer.

Reads existing intermediate-ckpt NPZs (rolled out by collect.py at
deterministic-eval ckpts) and reports per-checkpoint:
  - n_episodes
  - success rate proxy (fraction of episodes ending with final dist < 0.2m)
  - mean final dist
  - mean episode length

so we can plot success vs frames per condition and pick the convergence
point (where success rate plateaus).

Frames-per-ckpt (from earlier num_steps verification of ckpt files):
  blind:            10M / ckpt   (ckpt.34 = 340M)
  coarse(matched):   5M / ckpt   (ckpt.49 = 245M-250M)
  uniform:           5M / ckpt   (ckpt.49 = 250M)
  foveated(corrupt): 5M / ckpt   (ckpt.36 = 180M, NaN bug past)
  foveated_learned: 5M / ckpt    (ckpt.49 = 245M)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


COND_PATTERNS = [
    ("blind",            r"^blind_gibson_ckpt(\d+)_det\.npz$",            10),
    ("coarse",           r"^matched_gibson_ckpt(\d+)_det\.npz$",           5),
    ("uniform",          r"^uniform_gibson_ckpt(\d+)_det\.npz$",           5),
    ("foveated",         r"^foveated_gibson_ckpt(\d+)_det\.npz$",          5),
    ("foveated_learned", r"^foveated_learned_gibson_ckpt(\d+)_det\.npz$",  5),
]
SUCCESS_DIST = 0.2


def stats_one(npz_path: Path) -> dict:
    d = np.load(npz_path)
    dist = d["distance_to_goal"].astype(np.float64)
    ep_ids = d["episode_ids"]
    step = d["step_in_episode"]
    eps = np.unique(ep_ids)
    final_dists = []
    ep_lengths = []
    for ep in eps:
        em = ep_ids == ep
        last = np.argmax(step[em])
        final_dists.append(dist[em][last])
        ep_lengths.append(step[em].max() + 1)
    final_dists = np.array(final_dists)
    return {
        "n_episodes": int(len(eps)),
        "success_rate": float((final_dists < SUCCESS_DIST).mean()),
        "mean_final_dist_m": float(final_dists.mean()),
        "mean_ep_len_steps": float(np.mean(ep_lengths)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results: dict[str, list] = {}
    for cond_name, pat, frames_per_ckpt in COND_PATTERNS:
        regex = re.compile(pat)
        rows = []
        for npz in sorted(args.in_dir.glob("*.npz")):
            m = regex.match(npz.name)
            if not m:
                continue
            ckpt = int(m.group(1))
            frames = ckpt * frames_per_ckpt * 1_000_000
            print(f"  {cond_name} ckpt.{ckpt} ({frames/1e6:.0f}M frames) "
                  f"<- {npz.name}")
            s = stats_one(npz)
            row = {"ckpt": ckpt, "frames": frames, **s}
            rows.append(row)
            print(f"    success={s['success_rate']:.2f}  "
                  f"final_dist={s['mean_final_dist_m']:.2f}m  "
                  f"ep_len={s['mean_ep_len_steps']:.0f}")
        rows.sort(key=lambda r: r["ckpt"])
        results[cond_name] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}\n")

    # Pretty per-cond trajectory
    print("=" * 70)
    print("Convergence trajectory:")
    print(f"{'cond':<18} {'ckpt':<6} {'frames':<10} {'success':<10} {'mean_dist':<10} {'ep_len':<8}")
    for cond, rows in results.items():
        for r in rows:
            print(f"{cond:<18} {r['ckpt']:<6} {r['frames']/1e6:>4.0f}M    "
                  f"{r['success_rate']:<10.3f} "
                  f"{r['mean_final_dist_m']:<10.2f} "
                  f"{r['mean_ep_len_steps']:<8.0f}")

    print("\nConvergence rule of thumb: success_rate within ~5pp of final, "
          "and mean_final_dist within ~10% of final value. "
          "Pick smallest 'frames' meeting that bar across all 5 conds.")


if __name__ == "__main__":
    main()
