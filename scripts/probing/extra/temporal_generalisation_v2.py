"""TGM variant: probe target = allocentric position (positions[:, :2]).

This complements the goal-vec TGM (the original) and asks whether the linear
read-out for *world-frame* position is stable across time, vs the egocentric
goal-vec.

Predictions (post-hoc, given goal-vec result):
  - For position: same overall ordering (coarse > uniform > fov-LP > fov > blind)
    if the "code stability" is a consumption-axis property of the agent's
    representation, not specific to goal-vec.
  - If the ordering inverts, the goal-vec result is target-specific (the agent's
    goal-vec is computed from position differences, so a stable position code
    can support a varying goal-vec code).
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_generalisation import (
    DATA_ROOT, FILES, build_step_tensor, build_target_tensor, tgm,
)

import numpy as np
from sklearn.decomposition import PCA


def analyse_position(npz_path, T=50, max_eps=300, n_pcs=30, ridge_alpha=10.0):
    d = np.load(npz_path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    pos = d["positions"].astype(np.float32)[:, :2]  # x, y in world frame
    eps_id = d["episode_ids"]
    sip = d["step_in_episode"]

    eps_unique = np.unique(eps_id)
    keep_eps = []
    for e in eps_unique:
        m = eps_id == e
        if m.sum() >= T:
            keep_eps.append(e)
    keep_eps = np.array(keep_eps)
    if len(keep_eps) < 10:
        keep_eps = eps_unique[: max_eps]
    keep_mask = np.isin(eps_id, keep_eps)
    h, pos, eps_id, sip = h[keep_mask], pos[keep_mask], eps_id[keep_mask], sip[keep_mask]

    # PCA pre-reduce + standardise
    pca = PCA(n_components=n_pcs, random_state=0).fit(h)
    h_pcs = pca.transform(h).astype(np.float32)
    print(f"    PCA top-{n_pcs} cum var = {pca.explained_variance_ratio_.sum():.3f}")
    H, M = build_step_tensor(h_pcs, eps_id, sip, T)
    Y = build_target_tensor(pos, eps_id, sip, T)
    print(f"  H.shape={H.shape}  Y.shape={Y.shape}  cells_filled_avg={M.mean():.2f}")
    return tgm(H, Y, M, T=T, max_eps=max_eps, alpha=ridge_alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/tgm_pos_results.npz")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--max_eps", type=int, default=300)
    args = ap.parse_args()

    out = {}
    for cond, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            continue
        print(f"\n=== {cond} (allocentric position TGM) ===")
        out[cond] = analyse_position(path, T=args.T, max_eps=args.max_eps)
        m = out[cond]
        diag = float(np.nanmean(np.diagonal(m)))
        off = float(np.nanmean(m[np.tril_indices(args.T, k=-5)]))
        lag30 = float(np.nanmean(np.diagonal(m, offset=-30))) if args.T > 30 else float("nan")
        print(f"  diag mean = {diag:.3f}  off-diag mean = {off:.3f}  lag-30 = {lag30:.3f}")

    np.savez_compressed(args.out_path, **out)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
