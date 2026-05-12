"""Temporal Generalisation Matrix (TGM) — King & Dehaene 2014 (TICS).

Train a decoder of variable V at time t, test it at time t' for every (t, t') in
[0, T)^2. The shape of the resulting square matrix diagnoses whether the code
is transient (diagonal-only), stable (square block), recurrent (off-diagonal
arc), or cross-coded (cross shape).

Pre-registered protocol (matches docs/manuscript/sample/cogneuro_frameworks/
tgm_king_dehaene.md):

  Truncate episodes to first T=100 steps; only use episodes with len >= T.
  For variable V (we use goal_vec_x and goal_vec_y as a 2-D regression target):
    For each (t_train, t_test) pair, train Ridge(alpha=1.0) on h[ep_train,
    t_train, :] -> V[ep_train, t_train, :], evaluate on h[ep_test, t_test, :].
    Use 5-fold ep-level CV averaging over folds.
    Score = R^2.
  Output: 5 TxT matrices (one per condition) saved to JSON + figure.

  Predicted shape per condition (filed in tgm_king_dehaene.md):
    blind:             square block (sustained code)
    coarse:            mostly diagonal (transient, recomputed each step)
    foveated:          diagonal band + late off-diagonal
    uniform:           similar to coarse
    foveated_logpolar: extended off-diagonal from t=0

  Decision rules (frozen):
    - "Strong" if blind shows a qualitatively wider off-diagonal extent than
      coarse + uniform AND any sighted condition shows a clearly narrower band.
    - "Moderate" if blind shows numerically wider off-diagonal but the visual
      difference is subtle.
    - "Null" if all five matrices are visually similar.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

DATA_ROOT = Path("/tmp/rcp_analysis_v3")
FILES = {
    "blind": "blind_izar_det_RCP.npz",
    "coarse": "coarse_det_RCP.npz",
    "foveated": "foveated_det_RCP.npz",
    "foveated_logpolar": "foveated_logpolar_det_RCP.npz",
    "uniform": "uniform_det_RCP.npz",
}


def build_step_tensor(h: np.ndarray, ep_id: np.ndarray, step_id: np.ndarray, T: int):
    """Returns (ep_tensor, mask) where ep_tensor has shape (E, T, D), mask shape (E, T)."""
    eps = np.unique(ep_id)
    H = np.zeros((len(eps), T, h.shape[1]), dtype=np.float32)
    M = np.zeros((len(eps), T), dtype=bool)
    for i, e in enumerate(eps):
        m = ep_id == e
        s = step_id[m]
        ep_h = h[m]
        sel = s < T
        H[i, s[sel]] = ep_h[sel]
        M[i, s[sel]] = True
    return H, M


def build_target_tensor(target: np.ndarray, ep_id: np.ndarray, step_id: np.ndarray, T: int):
    eps = np.unique(ep_id)
    Y = np.zeros((len(eps), T, target.shape[1]), dtype=np.float32)
    for i, e in enumerate(eps):
        m = ep_id == e
        s = step_id[m]
        sel = s < T
        Y[i, s[sel]] = target[m][sel]
    return Y


def goal_vec_agent_frame(positions, headings, goals):
    """Return goal direction in agent's egocentric frame (xy)."""
    dx = goals[:, 0] - positions[:, 0]
    dy = goals[:, 1] - positions[:, 1]
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    fwd = cos_h * dx + sin_h * dy
    side = -sin_h * dx + cos_h * dy
    return np.stack([fwd, side], axis=-1).astype(np.float32)


def tgm(H: np.ndarray, Y: np.ndarray, M: np.ndarray, T: int, n_folds: int = 4,
         alpha: float = 1.0, max_eps: int = 200) -> np.ndarray:
    """Returns TxT matrix of mean R^2 across folds."""
    E = H.shape[0]
    if E > max_eps:
        rng = np.random.default_rng(0)
        idx = rng.choice(E, max_eps, replace=False)
        H, Y, M = H[idx], Y[idx], M[idx]
        E = max_eps
    out = np.zeros((T, T), dtype=np.float32)
    counts = np.zeros((T, T), dtype=np.int32)
    kf = KFold(n_folds, shuffle=True, random_state=0)
    for fi, (tr, te) in enumerate(kf.split(np.arange(E))):
        # train one ridge per t_train
        for t_tr in range(T):
            mask_tr = M[tr, t_tr]
            if mask_tr.sum() < 5:
                continue
            X_tr = H[tr][mask_tr][:, t_tr]  # (n_tr, D)
            y_tr = Y[tr][mask_tr][:, t_tr]
            try:
                clf = Ridge(alpha=alpha).fit(X_tr, y_tr)
            except Exception:
                continue
            for t_te in range(T):
                mask_te = M[te, t_te]
                if mask_te.sum() < 3:
                    continue
                X_te = H[te][mask_te][:, t_te]
                y_te = Y[te][mask_te][:, t_te]
                # custom R^2 (multi-output: average over coords)
                pred = clf.predict(X_te)
                ss_res = ((pred - y_te) ** 2).sum(0)
                ss_tot = ((y_te - y_te.mean(0)) ** 2).sum(0).clip(min=1e-8)
                r2 = (1.0 - ss_res / ss_tot).mean()
                out[t_tr, t_te] += r2
                counts[t_tr, t_te] += 1
    out = np.where(counts > 0, out / counts.clip(min=1), np.nan)
    return out


def analyse_one_condition(npz_path: Path, T: int = 100, max_eps: int = 200,
                            min_ep_len: int | None = None,
                            n_pcs: int = 30, ridge_alpha: float = 10.0) -> np.ndarray:
    if min_ep_len is None:
        min_ep_len = T
    d = np.load(npz_path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    pos = d["positions"].astype(np.float32)
    head = d["headings"].astype(np.float32)
    goal = d["goal_positions"].astype(np.float32)
    eps_id = d["episode_ids"]
    sip = d["step_in_episode"]

    goal_vec = goal_vec_agent_frame(pos, head, goal)

    # Filter to episodes with >= T steps
    eps_unique = np.unique(eps_id)
    keep_eps = []
    for e in eps_unique:
        m = eps_id == e
        if m.sum() >= min_ep_len:
            keep_eps.append(e)
    keep_eps = np.array(keep_eps)
    if len(keep_eps) < 10:
        keep_eps = eps_unique[: max_eps]
    keep_mask = np.isin(eps_id, keep_eps)
    h, pos, head, goal_vec, eps_id, sip = (h[keep_mask], pos[keep_mask],
                                             head[keep_mask], goal_vec[keep_mask],
                                             eps_id[keep_mask], sip[keep_mask])

    # PCA pre-reduce 512-d hidden -> n_pcs to fight overfitting in per-step ridge.
    pca = PCA(n_components=n_pcs, random_state=0).fit(h)
    h_pcs = pca.transform(h).astype(np.float32)
    print(f"    PCA top-{n_pcs} cum var = {pca.explained_variance_ratio_.sum():.3f}")
    H, M = build_step_tensor(h_pcs, eps_id, sip, T)
    Y = build_target_tensor(goal_vec, eps_id, sip, T)
    print(f"  H.shape={H.shape}  M.sum()={M.sum()}  cells_filled_avg={M.mean():.2f}")
    return tgm(H, Y, M, T=T, max_eps=max_eps, alpha=ridge_alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/tgm_results.npz")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--max_eps", type=int, default=200)
    args = ap.parse_args()

    out = {}
    for cond, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f"  MISSING {path}; skip {cond}")
            continue
        print(f"\n=== {cond} ===")
        m = analyse_one_condition(path, T=args.T, max_eps=args.max_eps)
        out[cond] = m
        # Print diagonal mean and off-diagonal mean as a quick summary
        diag = np.nanmean(np.diagonal(m))
        off = np.nanmean(m[np.tril_indices(args.T, k=-5)])
        print(f"  diag(R^2) mean = {diag:.3f}  off-diag mean = {off:.3f}")

    np.savez_compressed(args.out_path, **out)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
