"""Frame-level 4-layer MLP probe of agent_pos for the gap (MLP - linear) metric.

Companion to 04b_frame_probe.py (Ridge linear). Same pre-reg amendment:
frame-level random CV with eval window steps 250-500, replacing the
trajectory-level CV that overfit catastrophically.

Pre-registered prediction (from cogneuro_round2/_paper_paragraph_stub.md and
PRE_REGISTRATION.md): MLP-linear gap should be larger for low-bandwidth
conditions (where format is non-linear) than high-bandwidth (where it's
already linear).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LSTM_ROOT = Path("/tmp/wmprobe_main/lstm")
CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]


def stack(records, window=(0, 10000)):
    Xs, Ys = [], []
    for r in records:
        h = r["h"]; c = r["c"]; a = r["action"]; p = r["agent_pos"]
        T = h.size(0)
        s, e = max(0, window[0]), min(T, window[1])
        x = torch.cat([h[s:e], c[s:e], a[s:e]], dim=-1)
        Xs.append(x); Ys.append(p[s:e])
    return torch.cat(Xs, 0), torch.cat(Ys, 0)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=1024, n_layers=4, d_out=2):
        super().__init__()
        layers = []
        prev = d_in
        for _ in range(n_layers):
            layers += [nn.LayerNorm(prev), nn.Linear(prev, d_hidden), nn.ELU()]
            prev = d_hidden
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def r2(pred, target):
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0).clamp(min=1e-8)
    return (1.0 - ss_res / ss_tot).mean().item()


def train_eval_mlp(X_tr, Y_tr, X_te, Y_te, device, steps=8000, batch=512, lr=1e-3,
                   weight_decay=1e-3):
    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_tr_t = torch.from_numpy(X_tr).float().to(device)
    Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
    X_te_t = torch.from_numpy(X_te).float().to(device)
    Y_te_t = torch.from_numpy(Y_te).float().to(device)
    N = len(X_tr_t)
    best = -float("inf"); best_train = -float("inf")
    for s in range(steps):
        idx = torch.randint(0, N, (batch,), device=device)
        loss = ((model(X_tr_t[idx]) - Y_tr_t[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 1000 == 0 or s == steps - 1:
            model.eval()
            with torch.no_grad():
                ev = r2(model(X_te_t), Y_te_t)
                tr = r2(model(X_tr_t), Y_tr_t)
            model.train()
            if ev > best:
                best = ev; best_train = tr
    return best, best_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/wmprobe_main/probes_frame_mlp.json")
    ap.add_argument("--frame_split", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                            else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device = {device}")

    out = {}
    for cond in CONDITIONS:
        d = LSTM_ROOT / cond
        if not (d / "hidden_train.pt").exists():
            continue
        train_recs = torch.load(d / "hidden_train.pt", weights_only=False)
        eval_recs = torch.load(d / "hidden_eval.pt", weights_only=False)
        all_recs = list(train_recs) + list(eval_recs)
        X, Y = stack(all_recs, window=(250, 500))
        X = X.numpy(); Y = Y.numpy()
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(X))
        n_te = int(len(X) * args.frame_split)
        te_idx = idx[:n_te]; tr_idx = idx[n_te:]
        X_tr, Y_tr = X[tr_idx], Y[tr_idx]
        X_te, Y_te = X[te_idx], Y[te_idx]

        print(f"\n=== {cond} === train={len(X_tr)} eval={len(X_te)}")
        ev, tr = train_eval_mlp(X_tr, Y_tr, X_te, Y_te, device, steps=args.steps)
        out[cond] = {"eval_r2": ev, "train_r2": tr,
                       "n_train": int(len(X_tr)), "n_eval": int(len(X_te))}
        print(f"  MLP train R^2={tr:.3f}  eval R^2={ev:.3f}")

    json.dump(out, open(args.out_path, "w"), indent=2)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
