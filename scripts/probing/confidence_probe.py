"""
Confidence-proxy probe: can the hidden state decode how much ``cumulative
information'' the agent has accumulated within an episode?

The proposal's confidence probe was ``cumulative blur $\\sigma$'' --- the
integrated uncertainty budget over the agent's trajectory. Computing that
directly requires per-step eccentricity maps, which we leave to future
work.  As a first-pass proxy we decode ``step_in_episode'' from the top-
layer hidden state. A compensatory memory should track elapsed time /
accumulated experience; an outsourcing memory (uniform) should not, since
each frame re-derives spatial state.

Outputs a JSON with per-condition R^2 (time-elapsed probe), and whether
the probe is more successful for foveated/blind conditions than for
uniform (H1 supporting evidence).

Usage:
    python scripts/probing/confidence_probe.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out /scratch/izar/wxu/probing_results/confidence_proxy.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


def episode_split(ep_ids: np.ndarray, train_frac: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    unique = np.unique(ep_ids)
    rng.shuffle(unique)
    n_train = int(len(unique) * train_frac)
    train_ep = set(unique[:n_train].tolist())
    mask_train = np.isin(ep_ids, list(train_ep))
    return mask_train, ~mask_train


def probe_one_condition(npz_path: Path, alpha: float = 10.0) -> dict:
    d = np.load(npz_path)
    H = d["h_layers"][:, -1].astype(np.float32)   # (N, 512) top layer
    step = d["step_in_episode"].astype(np.float32)
    ep = d["episode_ids"]

    # Also track per-episode max step as a simple normaliser.
    ep_ids_unique = np.unique(ep)
    max_step_per_ep = {e: step[ep == e].max() for e in ep_ids_unique}

    # Target 1: raw step-in-episode.
    # Target 2: relative progress = step / max_step_per_ep (in [0, 1]).
    rel_step = np.array([
        step[i] / max(max_step_per_ep[ep[i]], 1.0) for i in range(len(step))
    ], dtype=np.float32)

    results = {}
    for target_name, target in (("step_in_episode", step),
                                ("relative_progress", rel_step)):
        # Episode-level split (probe training never sees test episodes).
        train_mask, test_mask = episode_split(ep, train_frac=0.8, seed=0)
        H_tr, H_te = H[train_mask], H[test_mask]
        y_tr, y_te = target[train_mask], target[test_mask]

        if H_tr.shape[0] < 50 or H_te.shape[0] < 50:
            results[target_name] = {"skipped": "insufficient samples"}
            continue

        model = Ridge(alpha=alpha)
        model.fit(H_tr, y_tr)
        yp = model.predict(H_te)
        ss_res = float(np.sum((y_te - yp) ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
        mae = float(np.mean(np.abs(y_te - yp)))

        # Hewitt-Liang control: permute labels within each episode.
        rng = np.random.default_rng(0)
        y_ctrl = target.copy()
        for e in ep_ids_unique:
            idx = np.where(ep == e)[0]
            rng.shuffle(idx)
            # note: this shuffles IDs but leaves y unchanged; instead do
            # assignment swap -- simpler: shuffle the targets across the
            # whole dataset but within each episode
            y_ctrl[np.where(ep == e)[0]] = target[np.random.default_rng(seed=int(e)).permutation(np.where(ep == e)[0])]

        y_ctrl_tr, y_ctrl_te = y_ctrl[train_mask], y_ctrl[test_mask]
        ctrl_model = Ridge(alpha=alpha)
        ctrl_model.fit(H_tr, y_ctrl_tr)
        ctrl_pred = ctrl_model.predict(H_te)
        ctrl_ss_res = float(np.sum((y_ctrl_te - ctrl_pred) ** 2))
        ctrl_ss_tot = float(np.sum((y_ctrl_te - y_ctrl_te.mean()) ** 2))
        ctrl_r2 = 1.0 - ctrl_ss_res / max(ctrl_ss_tot, 1e-9)

        selectivity = (r2 - ctrl_r2) / max(1.0 - ctrl_r2, 1e-9)

        results[target_name] = {
            "r2": float(r2),
            "mae": float(mae),
            "r2_control": float(ctrl_r2),
            "selectivity": float(selectivity),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }

    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    conditions = {
        "blind":             "blind_gibson.npz",
        "uniform":           "uniform_gibson.npz",
        "foveated":          "foveated_gibson.npz",
        "foveated_learned":  "foveated_learned_gibson_truncated.npz",
        "matched":           "matched_gibson.npz",
    }

    all_results: dict = {}
    for cond, fname in conditions.items():
        path = args.in_dir / fname
        if not path.exists():
            sys.stderr.write(f"[skip] {cond}: {path} missing\n")
            continue
        print(f"probing {cond} from {path}")
        all_results[cond] = probe_one_condition(path)
        for k, v in all_results[cond].items():
            if "r2" in v:
                print(f"  {k}: R^2={v['r2']:+.3f}  sel={v['selectivity']:+.3f}")
            else:
                print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
