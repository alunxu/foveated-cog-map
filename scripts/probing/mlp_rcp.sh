#!/bin/bash
# 2-layer MLP probe on 4 retrain NPZs (paper L245 MLP gap claim).
# CPU-only. Skips blind (no own retrain ckpt yet).
set -e

JOB_NAME="mlp-recompute"
B64=$(cat <<'PYBODY' | base64 | tr -d '\n'
"""2-layer MLP probe (hidden=256, L2=1e-4, 5-fold episode-level CV).
Compares against linear Ridge probe to compute MLP-linear gap.
Paper L245: gap larger for rich-encoder (info present non-linearly).
"""
import os, json
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


CONDS = ["coarse", "foveated", "uniform", "foveated_logpolar"]
NPZ_DIR = "/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
OUT = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/mlp_probe.json"


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
    def forward(self, x): return self.net(x)


def fit_mlp(X_tr, y_tr, X_te, hidden=256, l2=1e-4, epochs=30, batch=512, lr=1e-3):
    m = MLP(X_tr.shape[1], y_tr.shape[1], hidden=hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=l2)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_te, dtype=torch.float32)
    n = len(Xt)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i+batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(Xt[b]), yt[b])
            loss.backward(); opt.step()
    with torch.no_grad():
        return m(Xv).numpy()


def cv_probe(H, Y, ep, n_folds=5, max_samples=20000):
    """Returns (linear_r2_mean, linear_r2_std, mlp_r2_mean, mlp_r2_std)."""
    if len(H) > max_samples:
        idx = np.random.RandomState(42).choice(len(H), max_samples, replace=False)
        H, Y, ep = H[idx], Y[idx], ep[idx]
    ue = np.unique(ep)
    rng = np.random.RandomState(42); rng.shuffle(ue)
    kf = KFold(n_splits=n_folds, shuffle=False)
    lin_r2, mlp_r2 = [], []
    for tri, tei in kf.split(ue):
        tr = np.isin(ep, ue[tri]); te = np.isin(ep, ue[tei])
        sc = StandardScaler()
        Xt = sc.fit_transform(H[tr]); Xv = sc.transform(H[te])
        # linear
        lin = Ridge(alpha=10).fit(Xt, Y[tr])
        lin_r2.append(r2_score(Y[te], lin.predict(Xv)))
        # MLP
        try:
            yhat = fit_mlp(Xt, Y[tr], Xv)
            mlp_r2.append(r2_score(Y[te], yhat))
        except Exception as e:
            print(f"  MLP fold failed: {e}"); mlp_r2.append(np.nan)
    return (np.mean(lin_r2), np.std(lin_r2), np.nanmean(mlp_r2), np.nanstd(mlp_r2))


print(f"{'cond':<22} {'linear_R²':>14} {'MLP_R²':>14} {'MLP-linear gap':>18}")
print("-" * 75)
out = {}
for c in CONDS:
    p = f"{NPZ_DIR}/{c}_det.npz"
    if not os.path.exists(p):
        print(f"{c:<22} (no NPZ)"); continue
    d = np.load(p, allow_pickle=True)
    H = d["hidden_states"].astype(np.float32)
    gps = d["gps"].astype(np.float32)
    ep = d["episode_ids"]
    lr_m, lr_s, mr_m, mr_s = cv_probe(H, gps, ep)
    gap = mr_m - lr_m
    print(f"{c:<22} {lr_m:+.3f}±{lr_s:.3f} {mr_m:+.3f}±{mr_s:.3f}    {gap:+.3f}")
    out[c] = {"linear_r2_mean": float(lr_m), "linear_r2_std": float(lr_s),
              "mlp_r2_mean": float(mr_m), "mlp_r2_std": float(mr_s),
              "mlp_minus_linear_gap": float(gap)}
with open(OUT, "w") as f: json.dump(out, f, indent=2)
print(f"\nwrote {OUT}")
print("Paper L245: MLP gap ~0 blind, +0.10 coarse, +0.37 fov, +1.56 uniform")
PYBODY
)

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; export PATH=/home/wxu/.local/bin:\$PATH; pip install --quiet --user scikit-learn 2>&1 | tail -2; mkdir -p /scratch/wxu/habitat_checkpoints_rcp/analysis_results; echo $B64 | base64 -d > /tmp/mlp.py; python -u /tmp/mlp.py 2>&1 | tee /scratch/wxu/habitat_checkpoints_rcp/analysis_results/mlp_probe.log; echo MLP_DONE"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu --image="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2" \
    --cpu=8 --memory=32G --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu \
    --command -- bash -c "$INNER_CMD"
echo "Submitted $JOB_NAME (CPU-only, ~10 min for 4 conds × 5 folds × 30 epochs)"
