"""Linear + 4-layer MLP probe of agent_pos from cached LSTM hidden states.

Pre-registered protocol (PRE_REGISTRATION.md):
  - Probe input: concat(h_t, c_t, last_action) - dim 512+512+6 = 1030
  - Linear probe: nn.Linear(1030, 2)
  - MLP probe: 4 hidden layers x 1024 with ELU + LayerNorm
  - Train objective: MSE on agent_pos (units = maze cells)
  - Eval window: steps 250-500 (second half of T=500 trajectories)
  - Optimisation: Adam lr=1e-3, 30k gradient steps, batch 256 frames

Output: per-condition JSON with eval R^2 (linear, MLP), gap, train R^2,
sanity checks (static-encoder baseline, label-shuffled control).

Plus: a *static-encoder* baseline that probes agent_pos directly from the
DINOv2 CLS feature without any LSTM. This isolates the encoder-side
contribution from the recurrent integration.
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R^2 = 1 - SS_res / SS_tot. Computed across both pred coords averaged."""
    ss_res = ((pred - target) ** 2).sum(0)  # (2,)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)  # (2,)
    r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
    return r2.mean().item()


class LinearProbe(nn.Module):
    def __init__(self, d_in: int, d_out: int = 2):
        super().__init__()
        self.net = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 1024, n_layers: int = 4, d_out: int = 2):
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


def stack_records(records, window=(250, 500), include_last_action=True):
    """Concat per-trajectory records into flat (N, D), (N, 2) tensors restricted
    to the eval window.

    For training data the window is (0, T) since we want signal from full
    trajectory; for eval we use (250, 500) per pre-reg.
    """
    Xs, Ys = [], []
    for r in records:
        h = r["h"]  # (T, H)
        c = r["c"]  # (T, H)
        a = r["action"]  # (T, 6)
        p = r["agent_pos"]  # (T, 2)
        T = h.size(0)
        s, e = window
        s = max(0, s)
        e = min(T, e)
        x = torch.cat([h[s:e], c[s:e], a[s:e]] if include_last_action else [h[s:e], c[s:e]],
                       dim=-1)
        y = p[s:e]
        Xs.append(x)
        Ys.append(y)
    return torch.cat(Xs, 0), torch.cat(Ys, 0)


def train_probe(probe, X_train, Y_train, X_eval, Y_eval, device,
                  lr=1e-3, steps=10000, batch=256):
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_eval, Y_eval = X_eval.to(device), Y_eval.to(device)
    N = X_train.size(0)
    best_eval_r2 = -float("inf")
    best_train_r2 = -float("inf")
    losses = []
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), device=device)
        x = X_train[idx]
        y = Y_train[idx]
        pred = probe(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0 or step == steps - 1:
            probe.eval()
            with torch.no_grad():
                eval_r2 = r2_score(probe(X_eval), Y_eval)
                train_r2 = r2_score(probe(X_train), Y_train)
            probe.train()
            if eval_r2 > best_eval_r2:
                best_eval_r2 = eval_r2
                best_train_r2 = train_r2
    return best_eval_r2, best_train_r2, losses


def static_encoder_probe(condition: str, feat_root: str, raw_dir: str,
                          eval_feat_root: str, raw_eval: str, device,
                          window=(250, 500)):
    """Probe agent_pos directly from DINOv2 CLS feature with no LSTM.

    Tests encoder-side contribution alone. Should be near zero for blind,
    increasing in bandwidth.
    """
    def load(feat_dir, raw_dir, max_traj):
        Xs, Ys = [], []
        feat_files = sorted(glob(os.path.join(feat_dir, "traj_*.pt")))[:max_traj]
        for fp in feat_files:
            f = torch.load(fp, weights_only=True)
            raw = np.load(os.path.join(raw_dir, os.path.basename(fp).replace(".pt", ".npz")))
            p = torch.from_numpy(raw["agent_pos"]).float()
            T = f.size(0)
            s, e = window
            s, e = max(0, s), min(T, e)
            Xs.append(f[s:e])
            Ys.append(p[s:e])
        return torch.cat(Xs, 0), torch.cat(Ys, 0)

    X_tr, Y_tr = load(os.path.join(feat_root, condition), raw_dir, max_traj=200)
    X_ev, Y_ev = load(os.path.join(eval_feat_root, condition), raw_eval, max_traj=None)
    if X_tr.numel() == 0 or X_ev.numel() == 0:
        return {"linear_eval_r2": None, "mlp_eval_r2": None}

    out = {}
    for name, probe in [
        ("linear", LinearProbe(X_tr.size(1))),
        ("mlp", MLPProbe(X_tr.size(1))),
    ]:
        r2_ev, r2_tr, _ = train_probe(probe, X_tr, Y_tr, X_ev, Y_ev, device,
                                         steps=5000, batch=256)
        out[f"{name}_eval_r2"] = r2_ev
        out[f"{name}_train_r2"] = r2_tr
    return out


def shuffle_control(X_eval, Y_eval, device):
    """Shuffle agent_pos labels across trajectories within the eval set;
    both linear and MLP R^2 should be ~ 0 (chance level)."""
    Y_shuffled = Y_eval.clone()
    Y_shuffled = Y_shuffled[torch.randperm(Y_shuffled.size(0))]
    out = {}
    for name, probe in [
        ("linear", LinearProbe(X_eval.size(1))),
        ("mlp", MLPProbe(X_eval.size(1))),
    ]:
        # Train on the shuffled training labels (re-use eval as train for control)
        r2_ev, _, _ = train_probe(probe, X_eval, Y_shuffled, X_eval, Y_shuffled,
                                    device, steps=2000, batch=256)
        out[f"shuffle_{name}_eval_r2"] = r2_ev
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm_dir", required=True,
                     help="Dir from 03_train_lstm.py containing hidden_train.pt + hidden_eval.pt")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--feat_root", required=True,
                     help="For static-encoder baseline.")
    ap.add_argument("--eval_feat_root", required=True)
    ap.add_argument("--raw_train", required=True)
    ap.add_argument("--raw_eval", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--probe_steps", type=int, default=10000)
    ap.add_argument("--no_static", action="store_true")
    ap.add_argument("--no_shuffle", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device = {device}")

    train_records = torch.load(os.path.join(args.lstm_dir, "hidden_train.pt"),
                                 weights_only=False)
    eval_records = torch.load(os.path.join(args.lstm_dir, "hidden_eval.pt"),
                                 weights_only=False)
    X_tr, Y_tr = stack_records(train_records, window=(0, 10000))  # all steps for train
    X_ev, Y_ev = stack_records(eval_records, window=(250, 500))  # eval window
    print(f"  train: X {tuple(X_tr.shape)}  Y {tuple(Y_tr.shape)}")
    print(f"  eval:  X {tuple(X_ev.shape)}  Y {tuple(Y_ev.shape)}")

    out = {"condition": args.condition,
            "n_train_frames": int(X_tr.size(0)),
            "n_eval_frames": int(X_ev.size(0)),
            "input_dim": int(X_tr.size(1))}

    print("  linear probe ...")
    r2_ev, r2_tr, _ = train_probe(LinearProbe(X_tr.size(1)), X_tr, Y_tr, X_ev, Y_ev,
                                    device, steps=args.probe_steps)
    out["linear_eval_r2"] = r2_ev
    out["linear_train_r2"] = r2_tr
    print(f"    linear: eval R^2 = {r2_ev:.4f}  train R^2 = {r2_tr:.4f}")

    print("  MLP probe ...")
    r2_ev, r2_tr, _ = train_probe(MLPProbe(X_tr.size(1)), X_tr, Y_tr, X_ev, Y_ev,
                                    device, steps=args.probe_steps)
    out["mlp_eval_r2"] = r2_ev
    out["mlp_train_r2"] = r2_tr
    print(f"    MLP: eval R^2 = {r2_ev:.4f}  train R^2 = {r2_tr:.4f}")
    out["mlp_minus_linear_gap"] = out["mlp_eval_r2"] - out["linear_eval_r2"]

    if not args.no_static:
        print("  static-encoder baseline ...")
        out["static"] = static_encoder_probe(
            args.condition, args.feat_root, args.raw_train, args.eval_feat_root,
            args.raw_eval, device,
        )
        print(f"    static linear: {out['static']['linear_eval_r2']}")
        print(f"    static MLP   : {out['static']['mlp_eval_r2']}")

    if not args.no_shuffle:
        print("  shuffle control ...")
        out["shuffle"] = shuffle_control(X_ev, Y_ev, device)
        print(f"    shuffle linear: {out['shuffle']['shuffle_linear_eval_r2']}")
        print(f"    shuffle MLP   : {out['shuffle']['shuffle_mlp_eval_r2']}")

    json.dump(out, open(args.out_path, "w"), indent=2)
    print(f"  saved {args.out_path}")


if __name__ == "__main__":
    main()
