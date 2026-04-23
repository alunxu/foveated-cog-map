"""
Correct causal test for H3: with compass sensor masked during rollout,
can hidden state still decode TRUE heading (from simulator groundtruth)?

The analyze.py pipeline regresses hidden state → `compass` field, which
is the observed sensor value. When compass is masked the sensor value is
always zero, which makes R² trivially 1.0 — a degenerate-target artefact
of the masking protocol, not a real finding.

This script regresses hidden state → sin/cos of `headings` (the true
world-frame heading from simulator), which is unaffected by masking.

Usage:
    python scripts/probing/masked_heading_probe.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out   /scratch/izar/wxu/probing_results/masked_heading.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def split_by_episode(ep_ids, seed=0, test_frac=0.2):
    unique = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(test_frac * len(unique)))
    test_eps = set(unique[:n_test].tolist())
    test_mask = np.isin(ep_ids, list(test_eps))
    return ~test_mask, test_mask


def probe(npz_path: Path, max_step_per_ep: int | None = None) -> dict | None:
    try:
        d = np.load(npz_path, allow_pickle=True)
    except FileNotFoundError:
        return None
    # Full-episode data first, then optionally truncate to first
    # `max_step_per_ep` steps per episode (matches paper's truncated-to-
    # matched protocol for fov-learned).
    X_full = d["hidden_states"].astype(np.float32)
    headings_full = d["headings"].astype(np.float32)
    positions_full = d["positions"].astype(np.float32)
    ep_full = d["episode_ids"]
    step_full = d["step_in_episode"]
    if max_step_per_ep is not None:
        keep = step_full < max_step_per_ep
        X = X_full[keep]
        headings = headings_full[keep]
        positions = positions_full[keep]
        ep_ids = ep_full[keep]
    else:
        X = X_full
        headings = headings_full
        positions = positions_full
        ep_ids = ep_full

    # The PointGoal compass sensor is EGOCENTRIC: heading relative to the
    # agent's pose at episode start. Reconstruct by subtracting each
    # episode's step-0 heading from every within-episode heading.
    ep_compass = np.zeros_like(headings)
    ep_gps = np.zeros((len(headings), 2), dtype=np.float32)
    for e in np.unique(ep_ids):
        mask = ep_ids == e
        steps = np.arange(int(mask.sum()))
        h_ep = headings[mask]
        p_ep = positions[mask][:, [0, 2]]  # x, z
        # Episode-relative heading: (current - start), wrapped to [-pi, pi]
        rel_h = np.arctan2(np.sin(h_ep - h_ep[0]), np.cos(h_ep - h_ep[0]))
        ep_compass[mask] = rel_h
        # Episode-relative GPS: rotate displacement into start frame
        dx = p_ep - p_ep[0]
        cos0, sin0 = np.cos(-h_ep[0]), np.sin(-h_ep[0])
        rot_dx = np.stack([cos0 * dx[:, 0] - sin0 * dx[:, 1],
                           sin0 * dx[:, 0] + cos0 * dx[:, 1]], axis=1)
        ep_gps[mask] = rot_dx

    # Probe target: sin/cos of episodic compass (circular).
    y_c = np.stack([np.sin(ep_compass), np.cos(ep_compass)], axis=1)
    tr, te = split_by_episode(ep_ids)
    clf_c = Ridge(alpha=10.0).fit(X[tr], y_c[tr])
    pred_c = clf_c.predict(X[te])
    r2_c = float(r2_score(y_c[te], pred_c, multioutput="uniform_average"))
    mae_rad = float(np.abs(np.arctan2(pred_c[:, 0], pred_c[:, 1]) - ep_compass[te]).mean())

    # Probe target: episodic GPS (ego-to-start displacement in start frame).
    clf_g = Ridge(alpha=10.0).fit(X[tr], ep_gps[tr])
    pred_g = clf_g.predict(X[te])
    r2_g = float(r2_score(ep_gps[te], pred_g, multioutput="uniform_average"))

    return {
        "n_steps": int(X.shape[0]),
        "n_episodes": int(len(np.unique(ep_ids))),
        "avg_steps_per_ep": float(X.shape[0] / len(np.unique(ep_ids))),
        "episodic_compass_r2": r2_c,
        "episodic_compass_mae_deg": float(np.rad2deg(mae_rad)),
        "episodic_gps_r2": r2_g,
    }


CONDITIONS = [
    # (display_name, baseline_npz, masked_npz)
    ("foveated_fix",     "foveated_gibson.npz",         "foveated_gibson_mask_compass.npz"),
    ("foveated_learned", "foveated_learned_gibson.npz", "foveated_learned_gibson_mask_compass.npz"),
    ("uniform",          "uniform_gibson.npz",          "uniform_gibson_mask_compass.npz"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--max-step", type=int, default=None,
                    help="Truncate each episode to its first N steps before "
                         "probing. Matches paper's truncated-to-matched "
                         "protocol for fov-learned cross-condition comparison.")
    args = ap.parse_args()

    results = {}
    print(f"{'Condition':<20} {'run':<8} "
          f"{'n_ep':>6} {'steps/ep':>8} "
          f"{'ep-compass R²':>14} {'MAE°':>6} {'ep-GPS R²':>11}")
    for cond, base, mask in CONDITIONS:
        for tag, fname in [("base", base), ("masked", mask)]:
            r = probe(args.in_dir / fname, max_step_per_ep=args.max_step)
            if r is None:
                print(f"{cond:<20} {tag:<8} (missing)")
                continue
            results[f"{cond}_{tag}"] = r
            print(f"{cond:<20} {tag:<8} "
                  f"{r['n_episodes']:>6} {r['avg_steps_per_ep']:>8.1f} "
                  f"{r['episodic_compass_r2']:>+14.3f} "
                  f"{r['episodic_compass_mae_deg']:>6.1f} "
                  f"{r['episodic_gps_r2']:>+11.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
