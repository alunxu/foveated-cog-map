#!/bin/bash
# Interim sanity-check: 4 lenses on the partial blind_izar_det.npz
# (currently ~50 eps after ep 50 checkpoint).
# Runs in parallel with the still-going probe-collect.
# CPU only, ~5-10 min.
set -e

JOB_NAME="blind-izar-partial"
RESULTS_DIR="/scratch/wxu/habitat_checkpoints_rcp/analysis_results"

B64=$(cat <<'PYBODY' | base64 | tr -d '\n'
"""Interim sanity check: 4 lenses on partial blind_izar NPZ.
Pointed at the live-being-overwritten blind_izar_det.npz.
Output: blind_izar_partial_preview.json
"""
import os, json, numpy as np, torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out))
    def forward(self, x): return self.net(x)


def fit_mlp(X_tr, y_tr, X_te, hidden=256, l2=1e-4, epochs=30, batch=512, lr=1e-3):
    m = MLP(X_tr.shape[1], y_tr.shape[1], hidden=hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=l2)
    Xt = torch.tensor(X_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_te, dtype=torch.float32); n = len(Xt)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i+batch]; opt.zero_grad()
            loss = nn.functional.mse_loss(m(Xt[b]), yt[b]); loss.backward(); opt.step()
    with torch.no_grad():
        return m(Xv).numpy()


def kfold_r2(H, Y, ep, n_folds=5, alpha=10.0):
    ue = np.unique(ep); rng = np.random.RandomState(42); rng.shuffle(ue)
    kf = KFold(n_splits=n_folds, shuffle=False)
    r2s = []
    for tri, tei in kf.split(ue):
        tr = np.isin(ep, ue[tri]); te = np.isin(ep, ue[tei])
        sc = StandardScaler(); Xtr = sc.fit_transform(H[tr]); Xte = sc.transform(H[te])
        r = Ridge(alpha=alpha).fit(Xtr, Y[tr])
        r2s.append(r2_score(Y[te], r.predict(Xte)))
    return float(np.mean(r2s)), float(np.std(r2s))


def cv_probe(H, Y, ep, n_folds=5, max_samples=20000):
    if len(H) > max_samples:
        idx = np.random.RandomState(42).choice(len(H), max_samples, replace=False)
        H, Y, ep = H[idx], Y[idx], ep[idx]
    ue = np.unique(ep); rng = np.random.RandomState(42); rng.shuffle(ue)
    kf = KFold(n_splits=n_folds, shuffle=False)
    lin_r2, mlp_r2 = [], []
    for tri, tei in kf.split(ue):
        tr = np.isin(ep, ue[tri]); te = np.isin(ep, ue[tei])
        sc = StandardScaler(); Xt = sc.fit_transform(H[tr]); Xv = sc.transform(H[te])
        lin = Ridge(alpha=10).fit(Xt, Y[tr])
        lin_r2.append(r2_score(Y[te], lin.predict(Xv)))
        try:
            yhat = fit_mlp(Xt, Y[tr], Xv); mlp_r2.append(r2_score(Y[te], yhat))
        except Exception as e:
            print(f"  MLP fold failed: {e}"); mlp_r2.append(np.nan)
    return float(np.mean(lin_r2)), float(np.std(lin_r2)), float(np.nanmean(mlp_r2)), float(np.nanstd(mlp_r2))


def lag_pairs(v, ep, k):
    Xi, Yv, E = [], [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        if len(idx) <= k: continue
        Xi.extend(idx[k:]); Yv.extend(v[idx[:len(idx) - k]]); E.extend([e] * (len(idx) - k))
    return (np.array(Xi), np.array(Yv), np.array(E)) if E else (None, None, None)


def skaggs_rectified(H, positions, scene_ids, n_bins=20, min_steps=20):
    H_rect = np.maximum(H, 0)
    unique_scenes = np.unique(scene_ids); all_si = []
    for sid in unique_scenes:
        mask = scene_ids == sid
        if mask.sum() < min_steps: continue
        h_s = H_rect[mask]; xz = positions[mask][:, [0, 2]]
        x_edges = np.linspace(xz[:,0].min()-1e-6, xz[:,0].max()+1e-6, n_bins+1)
        z_edges = np.linspace(xz[:,1].min()-1e-6, xz[:,1].max()+1e-6, n_bins+1)
        x_bin = np.clip(np.digitize(xz[:,0], x_edges)-1, 0, n_bins-1)
        z_bin = np.clip(np.digitize(xz[:,1], z_edges)-1, 0, n_bins-1)
        bin_idx = x_bin * n_bins + z_bin
        occupancy = np.bincount(bin_idx, minlength=n_bins*n_bins).astype(float)
        sum_act = np.zeros((n_bins*n_bins, h_s.shape[1])); np.add.at(sum_act, bin_idx, h_s)
        occ_mask = occupancy > 0
        if occ_mask.sum() < 4: continue
        p_occ = occupancy[occ_mask] / occupancy[occ_mask].sum()
        mean_act = sum_act[occ_mask] / occupancy[occ_mask, None]
        global_mean = h_s.mean(axis=0); si = np.zeros(h_s.shape[1])
        for j in range(h_s.shape[1]):
            lam = global_mean[j]
            if lam < 1e-8: continue
            ratio = mean_act[:, j] / lam; ratio = np.clip(ratio, 1e-8, None)
            si[j] = np.sum(p_occ * ratio * np.log2(ratio))
        all_si.append(si)
    return np.array(all_si) if all_si else None


NPZ = "/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/blind_izar_det.npz"
OUT = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/blind_izar_partial_preview.json"

print(f"Loading {NPZ}...")
d = np.load(NPZ, allow_pickle=True)
H = d["hidden_states"].astype(np.float32)
gps = d["gps"].astype(np.float32)
ep = d["episode_ids"]
positions = d["positions"]; scene_ids = d["scene_ids"]
comp = d["compass"][:, 0] if d["compass"].ndim > 1 else d["compass"]
comp_sc = np.column_stack([np.sin(comp), np.cos(comp)])
dtg = d["distance_to_goal"]
print(f"  shape: H={H.shape}, gps={gps.shape}, episodes={len(np.unique(ep))}, scenes={len(np.unique(scene_ids))}")

out = {"source": "blind_izar (partial, 50ep checkpoint)", "ckpt": "ckpt.34", "n_episodes": int(len(np.unique(ep)))}

print("\n1. Linear + MLP probe (GPS target)...")
lr_m, lr_s, mr_m, mr_s = cv_probe(H, gps, ep)
out["linear_mlp"] = {"linear_r2_mean": lr_m, "linear_r2_std": lr_s,
                     "mlp_r2_mean": mr_m, "mlp_r2_std": mr_s,
                     "mlp_minus_linear_gap": mr_m - lr_m}
print(f"  linear: {lr_m:+.3f}±{lr_s:.3f}  MLP: {mr_m:+.3f}±{mr_s:.3f}  gap: {mr_m - lr_m:+.3f}")

print("\n2. Lag-k profile...")
LAGS = [0, 1, 2, 5, 10, 20, 50]
out["lagk"] = {}
for tgt_name, tgt_vals in [("GPS", gps), ("compass", comp_sc), ("DtG", dtg)]:
    out["lagk"][tgt_name] = {}
    for k in LAGS:
        Xi, Yv, E = lag_pairs(tgt_vals, ep, k)
        if Xi is None or len(Xi) < 100:
            out["lagk"][tgt_name][f"k{k}"] = None; continue
        m, s = kfold_r2(H[Xi], Yv, E)
        out["lagk"][tgt_name][f"k{k}"] = {"mean": m, "std": s}
        print(f"  {tgt_name:7} k={k:3}: {m:+.3f}±{s:.3f}")

print("\n3. Skaggs (rectified)...")
si_mat = skaggs_rectified(H, positions, scene_ids)
if si_mat is not None:
    mean_si = si_mat.mean(axis=0)
    out["skaggs"] = {
        "mean_per_unit_per_scene": float(mean_si.mean()),
        "std_per_unit_per_scene": float(mean_si.std()),
        "max_per_unit_per_scene": float(mean_si.max()),
        "n_scenes_used": int(len(si_mat)),
        "n_place_units_1bit": int((mean_si > 1.0).sum()),
        "n_place_units_05bit": int((mean_si > 0.5).sum()),
    }
    print(f"  mean per (unit,scene): {out['skaggs']['mean_per_unit_per_scene']:.4f}")
    print(f"  place-units >1bit: {out['skaggs']['n_place_units_1bit']}, >0.5bit: {out['skaggs']['n_place_units_05bit']}")

with open(OUT, "w") as f: json.dump(out, f, indent=2)
print(f"\nWrote {OUT}")
print("\nCompare vs friend's seed=2 50-ep preview (linear +0.36, MLP +0.77, lag-k unstable):")
print("  If our linear ~0.95: friend's was anomaly. If similar: blind agents have ~0.4 linear post-retrain.")
PYBODY
)

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; export PATH=/home/wxu/.local/bin:\$PATH; pip install --quiet --user scikit-learn 2>&1 | tail -2; mkdir -p ${RESULTS_DIR}; echo $B64 | base64 -d > /tmp/blind_izar_partial.py; python -u /tmp/blind_izar_partial.py 2>&1 | tee ${RESULTS_DIR}/blind_izar_partial_preview.log; echo PARTIAL_DONE"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu --image="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2" \
    --cpu=8 --memory=32G --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu \
    --command -- bash -c "$INNER_CMD"
echo "Submitted $JOB_NAME (CPU, ~5-10 min)"
