#!/bin/bash
# Phase 1a: linear/MLP probe + MINE for F2 (and F-LP2 if NPZ ready).
# Writes results into mlp_probe.json and mine_multiseed_5cond.json (in-place merge).
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat
pip install scikit-learn 2>&1 | tail -2

cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:$PYTHONPATH

ANALYSIS_DIR=/scratch/wxu/habitat_checkpoints_rcp/analysis_results
NPZ_DIR=/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp

# ---- (a) linear + MLP probe for F2 / F-LP2 ----
python3 << 'PYEOF'
import json, os, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def fit_mlp(X_tr, y_tr, X_te, y_te, hidden=256, epochs=50, lr=1e-3, l2=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, y_tr.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_te, dtype=torch.float32, device=device)
    bs = 512
    for ep in range(epochs):
        idx = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), bs):
            b = idx[i:i+bs]
            loss = loss_fn(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return r2_score(y_te, model(Xv).cpu().numpy(), multioutput="uniform_average")

def cond_probe(npz_path, max_samples=20000, n_folds=5):
    d = np.load(npz_path)
    h = d["h_layers"][:, 2, :].astype(np.float32)
    gps = d["gps"].astype(np.float32)
    ep_ids = d["episode_ids"]
    if len(h) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(h), max_samples, replace=False)
        h, gps, ep_ids = h[idx], gps[idx], ep_ids[idx]
    h = h - h.mean(0, keepdims=True)
    gps = gps - gps.mean(0, keepdims=True)
    ueps = np.unique(ep_ids)
    rng = np.random.default_rng(42); rng.shuffle(ueps)
    folds = np.array_split(ueps, n_folds)
    lr2, mr2 = [], []
    for te in folds:
        tr_m = ~np.isin(ep_ids, te); te_m = np.isin(ep_ids, te)
        rg = Ridge(alpha=10.0).fit(h[tr_m], gps[tr_m])
        lr2.append(r2_score(gps[te_m], rg.predict(h[te_m]), multioutput="uniform_average"))
        mr2.append(fit_mlp(h[tr_m], gps[tr_m], h[te_m], gps[te_m]))
    return {
        "linear_r2_mean": float(np.mean(lr2)), "linear_r2_std": float(np.std(lr2)),
        "mlp_r2_mean": float(np.mean(mr2)), "mlp_r2_std": float(np.std(mr2)),
    }

OUT = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/mlp_probe.json"
data = json.load(open(OUT))
NPZ_DIR = "/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
for k, n in [("fnorm", "fnorm_det_ckpt49.npz"), ("flp2", "flp2_det_ckpt49.npz")]:
    p = os.path.join(NPZ_DIR, n)
    if not os.path.exists(p):
        print(f"SKIP {p} (not ready)"); continue
    print(f"=== {k}: probe on {n} ===")
    r = cond_probe(p)
    print(f"  linear R^2 = {r['linear_r2_mean']:+.4f} +- {r['linear_r2_std']:.4f}")
    print(f"  MLP-2 R^2 = {r['mlp_r2_mean']:+.4f} +- {r['mlp_r2_std']:.4f}")
    data[k] = r
with open(OUT, "w") as f:
    json.dump(data, f, indent=2)
print(f"PATCHED {OUT}")
PYEOF

# ---- (b) MINE on F2 / F-LP2 ----
python3 << 'PYEOF'
import json, os, numpy as np, torch, torch.nn as nn

def mine_estimate(h, y, hidden=256, batch=512, lr=1e-4, n_steps=4000, seed=0, device=None):
    """MINE I(h; y) via Donsker-Varadhan bound on a 3-layer MLP critic."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    critic = nn.Sequential(
        nn.Linear(h.shape[1] + y.shape[1], hidden), nn.ELU(),
        nn.Linear(hidden, hidden), nn.ELU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    H = torch.tensor(h, dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.float32, device=device)
    n = len(H)
    ema_et = None
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch,), device=device)
        idxs = torch.randint(0, n, (batch,), device=device)
        hb, yb = H[idx], Y[idx]
        yb_s = Y[idxs]
        joint = critic(torch.cat([hb, yb], dim=1))
        marg  = critic(torch.cat([hb, yb_s], dim=1))
        et = torch.exp(marg).mean()
        ema_et = (0.99 * ema_et + 0.01 * et.detach()) if ema_et is not None else et.detach()
        # bias-corrected gradient (Belghazi 2018 eq.12)
        loss = -(joint.mean() - (et / (ema_et + 1e-8)).detach() * torch.log(et + 1e-8) - torch.log(et + 1e-8) + torch.log(ema_et + 1e-8))
        opt.zero_grad(); loss.backward(); opt.step()
    # final MI estimate (DV form)
    critic.eval()
    with torch.no_grad():
        # use a full-data estimate
        ix = torch.randperm(n, device=device)[:min(8192, n)]
        ix_s = torch.randperm(n, device=device)[:min(8192, n)]
        j = critic(torch.cat([H[ix], Y[ix]], dim=1)).mean()
        m = torch.exp(critic(torch.cat([H[ix], Y[ix_s]], dim=1))).mean()
        I_nats = (j - torch.log(m + 1e-8)).item()
    return I_nats, I_nats / np.log(2)  # nats, bits

def run_mine_3seeds(npz_path, max_samples=20000):
    d = np.load(npz_path)
    h = d["h_layers"][:, 2, :].astype(np.float32)
    pos = d["positions"][:, [0, 2]].astype(np.float32)  # (x, z)
    if len(h) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(h), max_samples, replace=False)
        h, pos = h[idx], pos[idx]
    h = h - h.mean(0, keepdims=True)
    pos = pos - pos.mean(0, keepdims=True)
    results = []
    for s in range(3):
        I_nats, I_bits = mine_estimate(h, pos, seed=s)
        results.append({"seed": s, "I_nats": float(I_nats), "I_bits": float(I_bits)})
    nats = [r["I_nats"] for r in results]
    bits = [r["I_bits"] for r in results]
    return {
        "label": npz_path.split("/")[-1].split("_")[0].upper(),
        "seeds": results,
        "I_nats_mean": float(np.mean(nats)),
        "I_nats_std": float(np.std(nats)),
        "I_bits_mean": float(np.mean(bits)),
        "I_bits_std": float(np.std(bits)),
    }

OUT = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/mine_multiseed_5cond.json"
data = json.load(open(OUT))
NPZ_DIR = "/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
for k, n in [("fnorm", "fnorm_det_ckpt49.npz"), ("flp2", "flp2_det_ckpt49.npz")]:
    p = os.path.join(NPZ_DIR, n)
    if not os.path.exists(p):
        print(f"SKIP {p}"); continue
    print(f"=== {k}: MINE on {n} ===")
    r = run_mine_3seeds(p)
    print(f"  I(h; pos) = {r['I_bits_mean']:.2f} +- {r['I_bits_std']:.2f} bits")
    data[k] = r
with open(OUT, "w") as f:
    json.dump(data, f, indent=2)
print(f"PATCHED {OUT}")
PYEOF

echo "PHASE 1a DONE"
