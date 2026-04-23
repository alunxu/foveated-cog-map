"""
B3: Non-linear (MLP) probe sanity check.

Re-runs the core §4 probes with a 2-layer MLP in place of Ridge regression.
Purpose: confirm the condition ordering is not a linear-probe artefact.
Expected behaviour:
  - Absolute R² values may rise (MLP is more expressive).
  - Ordering across conditions should be preserved for every target.
  - If ordering flips, the linear-probe story is incomplete.

Targets: GPS (2-D), compass (sin/cos), path-history lag-5 position, goal vector.

Usage:
    python scripts/probing/mlp_probe_sanity.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out   /scratch/izar/wxu/probing_results/mlp_sanity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score


CONDITIONS = [
    "blind_gibson",
    "uniform_gibson",
    "foveated_gibson",
    "foveated_learned_gibson",
    "matched_gibson",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def split_by_episode(ep_ids: np.ndarray, seed: int = 0, test_frac: float = 0.2):
    unique = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(test_frac * len(unique)))
    test_eps = set(unique[:n_test].tolist())
    test_mask = np.isin(ep_ids, list(test_eps))
    return ~test_mask, test_mask


def train_mlp(X_tr, y_tr, X_te, y_te, epochs: int = 80, lr: float = 1e-3,
              batch_size: int = 256):
    d_in = X_tr.shape[1]
    d_out = y_tr.shape[1]
    net = MLP(d_in, d_out).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).float().to(DEVICE)
    X_te_t = torch.from_numpy(X_te).float().to(DEVICE)

    n = X_tr_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = net(X_tr_t[idx])
            loss = ((pred - y_tr_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        y_pred = net(X_te_t).cpu().numpy()
    return r2_score(y_te, y_pred, multioutput="uniform_average")


def path_history_lag5(h_layers, positions, ep_ids):
    """Build (X=h_t, y=position at lag 5 before t) pairs where both exist
    inside the same episode. Same construction as analyze_legacy path_history."""
    X, y = [], []
    for ep in np.unique(ep_ids):
        idx = np.where(ep_ids == ep)[0]
        if len(idx) < 6:
            continue
        for i in range(5, len(idx)):
            X.append(h_layers[idx[i]])
            y.append(positions[idx[i - 5]])
    if not X:
        return None, None, None
    X = np.stack(X)
    y = np.stack(y)
    # Episode IDs aligned to X (the "current-step" episode of each row)
    ep_rep = []
    for ep in np.unique(ep_ids):
        idx = np.where(ep_ids == ep)[0]
        if len(idx) < 6:
            continue
        for _ in range(5, len(idx)):
            ep_rep.append(ep)
    return X, y, np.array(ep_rep)


def ego_goal_vector(positions, goals, headings):
    dxyz = goals - positions
    dx, dz = dxyz[:, 0], dxyz[:, 2]
    cos_h, sin_h = np.cos(-headings), np.sin(-headings)
    fwd = cos_h * (-dz) - sin_h * dx
    lat = sin_h * (-dz) + cos_h * dx
    return np.stack([fwd, lat], axis=1)


def run_one_condition(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    X_top = d["hidden_states"].astype(np.float32)
    positions = d["positions"].astype(np.float32)
    goals = d["goal_positions"].astype(np.float32)
    headings = d["headings"].astype(np.float32)
    ep_ids = d["episode_ids"]
    gps = d["gps"].astype(np.float32)
    compass = d["compass"].astype(np.float32)           # (n, 1) radians

    out = {"n_steps": int(X_top.shape[0]), "n_episodes": int(len(np.unique(ep_ids)))}

    # GPS
    tr, te = split_by_episode(ep_ids, seed=0)
    out["gps_r2_mlp"] = float(train_mlp(X_top[tr], gps[tr], X_top[te], gps[te]))

    # Compass as sin/cos
    c_sc = np.concatenate([np.sin(compass), np.cos(compass)], axis=1)
    out["compass_r2_mlp"] = float(train_mlp(X_top[tr], c_sc[tr], X_top[te], c_sc[te]))

    # Goal vector
    gv = ego_goal_vector(positions, goals, headings)
    out["goal_vector_r2_mlp"] = float(train_mlp(X_top[tr], gv[tr], X_top[te], gv[te]))

    # Path history lag 5
    Xp, yp, ep_p = path_history_lag5(X_top, positions, ep_ids)
    if Xp is not None:
        tr_p, te_p = split_by_episode(ep_p, seed=0)
        out["path_history_lag5_r2_mlp"] = float(
            train_mlp(Xp[tr_p], yp[tr_p], Xp[te_p], yp[te_p])
        )
    else:
        out["path_history_lag5_r2_mlp"] = None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = {}
    for c in CONDITIONS:
        path = args.in_dir / f"{c}.npz"
        if not path.exists():
            print(f"[skip] {path}")
            continue
        print(f"\n=== {c} ===", flush=True)
        r = run_one_condition(path)
        print(f"  GPS  MLP R² = {r['gps_r2_mlp']:+.3f}")
        print(f"  Comp MLP R² = {r['compass_r2_mlp']:+.3f}")
        print(f"  PH5  MLP R² = {r.get('path_history_lag5_r2_mlp', 'n/a')}")
        print(f"  GVec MLP R² = {r['goal_vector_r2_mlp']:+.3f}")
        results[c] = r

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
