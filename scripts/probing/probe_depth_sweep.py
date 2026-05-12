"""Probe-depth sweep: linear (Ridge) → MLP-1 → MLP-2 → MLP-4 per condition.

For each converged-ckpt NPZ, fit four probes of increasing depth and record
mean/std R^2 under 5-fold episode-level CV. Bottleneck conditions should
plateau at depth 0 (linear suffices); rich-encoder conditions should ramp up
with probe depth (position is non-linearly encoded). Output anchors a single
multi-line panel that shows format-shift severity per condition.

Output: /tmp/rcp_analysis/probe_depth_sweep.json
Runtime: ~5-10 min on local CPU (5 conds × 4 depths × 5 folds = 100 fits,
30 000-step subsample per condition).

Usage:
    python scripts/probing/probe_depth_sweep.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold  # noqa: F401  (kept for reference)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Source NPZs (cached from RCP scratch)
CONDS = [
    ("blind",             "/tmp/rcp_analysis_v3/blind_izar_det_RCP.npz"),
    ("coarse",            "/tmp/rcp_analysis_v3/coarse_det_RCP.npz"),
    ("foveated",          "/tmp/rcp_analysis_v3/foveated_det_RCP.npz"),
    ("foveated_logpolar", "/tmp/rcp_analysis_v3/foveated_logpolar_det_RCP.npz"),
    ("uniform",           "/tmp/rcp_analysis_v3/uniform_det_RCP.npz"),
]

# Probe stack: linear -> increasingly deep MLPs.
PROBES = [
    ("linear", None),
    ("MLP-1",  (256,)),
    ("MLP-2",  (256, 128)),
    ("MLP-4",  (256, 128, 64, 32)),
]

SUBSAMPLE = 30_000   # steps per condition (matches eigenspectrum methodology)
N_SPLITS = 5
RNG_SEED = 0
RIDGE_ALPHA = 10.0


def episode_kfold_cv(H, Y, ep_ids, hidden_sizes, n_splits=N_SPLITS):
    """Episode-level k-fold CV. hidden_sizes=None -> Ridge linear probe."""
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
        if hidden_sizes is None:
            model = Ridge(alpha=RIDGE_ALPHA).fit(Xtr_s, Ytr)
        else:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_sizes,
                max_iter=200,
                early_stopping=True,
                random_state=RNG_SEED,
                alpha=1e-4,
            ).fit(Xtr_s, Ytr)
        pred = model.predict(Xte_s)
        ss_res = ((Yte - pred) ** 2).sum()
        ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum()
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(r2s)), float(np.std(r2s))


def run():
    out = {}
    for cond_name, npz_path in CONDS:
        out[cond_name] = {}
        npz_p = Path(npz_path)
        if not npz_p.exists():
            print(f"missing: {npz_path}", flush=True)
            continue
        d = np.load(npz_p, allow_pickle=True)
        H = d["hidden_states"].astype(np.float32)
        # GPS sensor reading (relative to episode start) — matches main-text
        # §3.1 protocol. The 'positions' field is world-frame absolute and
        # gives uninformative R^2 in the cross-scene setting.
        Y = d["gps"].astype(np.float32)
        if "episode_ids" in d.files:
            ep_ids = d["episode_ids"].astype(np.int64)
        else:
            ep_ids = np.arange(len(H))

        # Subsample for tractable runtime.
        if len(H) > SUBSAMPLE:
            idx = np.random.default_rng(RNG_SEED).choice(
                len(H), SUBSAMPLE, replace=False
            )
            H, Y, ep_ids = H[idx], Y[idx], ep_ids[idx]

        print(
            f"\n=== {cond_name} ===  H={H.shape}  Y={Y.shape}  "
            f"eps={len(np.unique(ep_ids))}",
            flush=True,
        )

        for probe_name, hidden_sizes in PROBES:
            t0 = time.time()
            try:
                r2_m, r2_s = episode_kfold_cv(H, Y, ep_ids, hidden_sizes)
                out[cond_name][probe_name] = {
                    "r2_mean": r2_m,
                    "r2_std": r2_s,
                }
                print(
                    f"  {probe_name:<7s}  r2 = {r2_m:+.3f} +/- {r2_s:.3f}   "
                    f"({time.time()-t0:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {probe_name}: ERR {e}", flush=True)
                out[cond_name][probe_name] = {"error": str(e)}

    out_p = Path("/tmp/rcp_analysis/probe_depth_sweep.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_p, "w"), indent=2)
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    run()
