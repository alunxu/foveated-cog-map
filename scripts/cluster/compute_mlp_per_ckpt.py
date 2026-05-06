"""Compute MLP probe R^2 per (condition, training checkpoint) for the
training-trajectory format-shift figure.

For each (cond, ckpt) NPZ on RCP scratch, train a 2-layer MLP probe
(5-fold CV, episode-level) on h_2 → GPS, and record the mean R^2.

Output: consolidated JSON keyed by (cond, ckpt) with linear + MLP R^2.
Linear values pulled from existing analysis JSONs; MLP computed fresh.

Usage (inside RCP pod):
    python compute_mlp_per_ckpt.py \\
        --probing-dir /scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp \\
        --out /scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/mlp_per_ckpt.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


CONDS = ["coarse", "foveated", "foveated_logpolar", "uniform"]
CKPTS = [10, 20, 30, 40]
RNG_SEED = 0


def episode_kfold_cv(H, Y, ep_ids, n_splits=5, mlp_kw=None):
    """Episode-level 5-fold CV for an MLP probe."""
    if mlp_kw is None:
        mlp_kw = dict(hidden_layer_sizes=(256, 128), max_iter=200,
                       early_stopping=True, random_state=RNG_SEED)
    unique_eps = np.unique(ep_ids)
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(unique_eps)
    folds = np.array_split(perm, n_splits)

    r2s = []
    for f in range(n_splits):
        test_eps = set(folds[f].tolist())
        test_mask = np.array([e in test_eps for e in ep_ids])
        train_mask = ~test_mask
        Xtr, Xte = H[train_mask], H[test_mask]
        Ytr, Yte = Y[train_mask], Y[test_mask]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        mlp = MLPRegressor(**mlp_kw).fit(Xtr_s, Ytr)
        # R^2 over the GPS targets (multi-output)
        pred = mlp.predict(Xte_s)
        ss_res = ((Yte - pred) ** 2).sum()
        ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum()
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(r2s)), float(np.std(r2s))


def run(probing_dir: Path, out: Path):
    out_data = {}
    for cond in CONDS:
        out_data[cond] = {}
        for ck in CKPTS:
            npz_p = probing_dir / f"{cond}_det_ckpt{ck}.npz"
            if not npz_p.exists():
                print(f"missing: {npz_p}", flush=True)
                continue
            t0 = time.time()
            d = np.load(npz_p, allow_pickle=True)
            H = d["hidden_states"].astype(np.float32)
            # GPS positions: assume positions[:, [0, 2]] (x, z), or fall back
            pos = d["positions"]
            if pos.ndim == 2 and pos.shape[1] >= 3:
                Y = pos[:, [0, 2]].astype(np.float32)
            else:
                Y = pos.astype(np.float32)
            # Episode IDs for episode-level CV
            ep_ids = d.get("episode_ids", None)
            if ep_ids is None:
                # Fallback: per-step granularity (worse CV but available)
                ep_ids = np.arange(len(H))
            else:
                ep_ids = ep_ids.astype(np.int64)
            print(f"{cond} ckpt{ck}: H={H.shape} Y={Y.shape} eps={len(np.unique(ep_ids))}", flush=True)
            try:
                r2_mean, r2_std = episode_kfold_cv(H, Y, ep_ids)
                out_data[cond][f"ckpt{ck}"] = {
                    "mlp_r2_mean": r2_mean,
                    "mlp_r2_std": r2_std,
                    "n_steps": int(len(H)),
                    "n_episodes": int(len(np.unique(ep_ids))),
                }
                print(f"  -> mlp_r2 = {r2_mean:.3f} +/- {r2_std:.3f}  ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"  ERR: {e}", flush=True)
                out_data[cond][f"ckpt{ck}"] = {"error": str(e)}

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(out, "w"), indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probing-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.probing_dir, args.out)
