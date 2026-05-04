#!/bin/bash
# Run lag-k probe on 4 retrain NPZs via inline base64-encoded python (avoids
# the heredoc/quote issues in runai bash -c). Updates paper L286.
set -e

JOB_NAME="lagk-recompute"
B64=$(cat <<'PYBODY' | base64 | tr -d '\n'
"""Lag-k probe on 4-condition NPZs (paper L286 update)."""
import argparse, os, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def kfold_r2(H, Y, ep, n_folds=5, alpha=10.0, seed=42):
    ue = np.unique(ep)
    rng = np.random.RandomState(seed); rng.shuffle(ue)
    kf = KFold(n_splits=n_folds, shuffle=False)
    r2s = []
    for tri, tei in kf.split(ue):
        tr = np.isin(ep, ue[tri]); te = np.isin(ep, ue[tei])
        sc = StandardScaler()
        Xtr = sc.fit_transform(H[tr]); Xte = sc.transform(H[te])
        r = Ridge(alpha=alpha).fit(Xtr, Y[tr])
        r2s.append(r2_score(Y[te], r.predict(Xte)))
    r2s = np.array(r2s); return r2s.mean(), r2s.std()


def lag_pairs(v, ep, k):
    Xi, Yv, E = [], [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        if len(idx) <= k: continue
        Xi.extend(idx[k:]); Yv.extend(v[idx[:len(idx) - k]]); E.extend([e] * (len(idx) - k))
    return np.array(Xi), np.array(Yv), np.array(E) if E else None


CONDS = ["coarse", "foveated", "uniform", "foveated_logpolar", "blind_izar"]
LAGS = [0, 2, 5, 10, 20]
in_dir = "/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
out_path = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/lagk_summary.json"

results = {}
print(f"{'cond':<22} {'target':<10} " + " ".join(f"k={k:<3}".rjust(14) for k in LAGS))
print("-" * 100)
for c in CONDS:
    p = f"{in_dir}/{c}_det.npz"
    if not os.path.exists(p):
        print(f"{c:<22} (no NPZ)"); continue
    d = np.load(p, allow_pickle=True)
    H = d["hidden_states"].astype(np.float32)
    ep = d["episode_ids"]
    gps = d["gps"]
    comp = d["compass"][:, 0] if d["compass"].ndim > 1 else d["compass"]
    comp_sc = np.column_stack([np.sin(comp), np.cos(comp)])
    dtg = d["distance_to_goal"]
    results[c] = {}
    for tgt_name, tgt_vals in [("GPS", gps), ("compass", comp_sc), ("DtG", dtg)]:
        results[c][tgt_name] = {}
        rs = []
        for k in LAGS:
            Xi, Yv, E = lag_pairs(tgt_vals, ep, k)
            if Xi is None or len(Xi) < 100:
                rs.append("     -     "); results[c][tgt_name][f"k{k}"] = None; continue
            m, s = kfold_r2(H[Xi], Yv, E)
            rs.append(f"{m:+.2f}±{s:.2f}".rjust(14))
            results[c][tgt_name][f"k{k}"] = {"mean": float(m), "std": float(s)}
        print(f"{c:<22} {tgt_name:<10} " + " ".join(rs))
    print()
with open(out_path, "w") as f: json.dump(results, f, indent=2)
print(f"wrote {out_path}")
print("\nPaper L286: blind R²>=0.92 throughout k=0-20; coarse R²>=0.50; rich-encoder unstable")
PYBODY
)

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; export PATH=/home/wxu/.local/bin:\$PATH; pip install --quiet --user scikit-learn 2>&1 | tail -2; mkdir -p /scratch/wxu/habitat_checkpoints_rcp/analysis_results; echo $B64 | base64 -d > /tmp/lagk.py; python -u /tmp/lagk.py 2>&1 | tee /scratch/wxu/habitat_checkpoints_rcp/analysis_results/lagk_summary.log; echo LAGK_DONE"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu --image="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2" \
    --cpu=4 --memory=24G --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu \
    --command -- bash -c "$INNER_CMD"
echo "Submitted $JOB_NAME (CPU-only)"
