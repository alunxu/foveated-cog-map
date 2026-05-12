"""Transplant analogue: feed encoder features from condition Y into the LSTM
trained under condition X, then probe agent_pos.

This is the world-model analogue of the 5x5 memory-transplant in our paper.
The prediction (from §4 Transplant): the asymmetry — LSTM trained on
high-bandwidth features should NOT generalise to low-bandwidth features
(no information to integrate), but LSTM trained on low-bandwidth features
should HANDLE high-bandwidth features at no worse than at-source (extra
information, just unused).

Output: a (5, 5) matrix of agent_pos R^2 for {LSTM_X, encoder_Y} pairs,
with linear and MLP probes.
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn

# Reuse the LSTM and probe definitions from the sibling scripts. To keep this
# file standalone we redefine them rather than relying on relative imports.


class IntegratorLSTM(nn.Module):
    def __init__(self, d_in: int, d_action: int = 6, d_hidden: int = 512, n_layers: int = 2):
        super().__init__()
        self.d_in = d_in
        self.d_action = d_action
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=d_in + d_action, hidden_size=d_hidden,
                              num_layers=n_layers, batch_first=True)
        self.proj = nn.Linear(d_hidden, d_in)

    def encode_full(self, feats, actions):
        device = feats.device
        T, D = feats.shape
        L = self.n_layers
        H = self.d_hidden
        x = torch.cat([feats, actions], dim=-1).unsqueeze(0)  # (1, T, D+6)
        cells = nn.ModuleList()
        for layer in range(L):
            in_size = (D + 6) if layer == 0 else H
            cell = nn.LSTMCell(input_size=in_size, hidden_size=H)
            cell.weight_ih = nn.Parameter(getattr(self.lstm, f"weight_ih_l{layer}"))
            cell.weight_hh = nn.Parameter(getattr(self.lstm, f"weight_hh_l{layer}"))
            cell.bias_ih = nn.Parameter(getattr(self.lstm, f"bias_ih_l{layer}"))
            cell.bias_hh = nn.Parameter(getattr(self.lstm, f"bias_hh_l{layer}"))
            cells.append(cell)
        cells = cells.to(device).eval()
        h = [torch.zeros(1, H, device=device) for _ in range(L)]
        c = [torch.zeros(1, H, device=device) for _ in range(L)]
        h_outs = torch.empty(T, L, H, device=device)
        c_outs = torch.empty(T, L, H, device=device)
        with torch.no_grad():
            for t in range(T):
                inp = x[:, t]
                for layer in range(L):
                    h[layer], c[layer] = cells[layer](inp, (h[layer], c[layer]))
                    inp = h[layer]
                    h_outs[t, layer] = h[layer].squeeze(0)
                    c_outs[t, layer] = c[layer].squeeze(0)
        return h_outs[:, -1], c_outs[:, -1]  # last layer (T, H), (T, H)


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


def r2_score(pred, target):
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    return (1.0 - ss_res / ss_tot.clamp(min=1e-8)).mean().item()


def cache_hidden_for_pair(lstm_X, feat_dir_Y, raw_dir, device, max_traj=200):
    """Run LSTM_X over features from condition Y. Return train+eval (X, Y) tensors."""
    feat_files = sorted(glob(os.path.join(feat_dir_Y, "traj_*.pt")))[:max_traj]
    Xs, Ys = [], []
    for fp in feat_files:
        feats = torch.load(fp, weights_only=True).to(device)
        npz = np.load(os.path.join(raw_dir, os.path.basename(fp).replace(".pt", ".npz")))
        actions = torch.from_numpy(npz["action"]).float().to(device)
        agent_pos = torch.from_numpy(npz["agent_pos"]).float()
        h_seq, c_seq = lstm_X.encode_full(feats, actions)
        x = torch.cat([h_seq, c_seq, actions], dim=-1).cpu()
        Xs.append(x)
        Ys.append(agent_pos)
    return torch.cat(Xs), torch.cat(Ys)


def train_probe(probe, X_tr, Y_tr, X_ev, Y_ev, device, lr=1e-3, steps=5000, batch=256):
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)
    X_ev, Y_ev = X_ev.to(device), Y_ev.to(device)
    N = X_tr.size(0)
    best = -float("inf")
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), device=device)
        loss = ((probe(X_tr[idx]) - Y_tr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            probe.eval()
            with torch.no_grad():
                r2 = r2_score(probe(X_ev), Y_ev)
            probe.train()
            if r2 > best:
                best = r2
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm_root", required=True)
    ap.add_argument("--feat_train", required=True)
    ap.add_argument("--feat_eval", required=True)
    ap.add_argument("--raw_train", required=True)
    ap.add_argument("--raw_eval", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--max_train_traj", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
    conditions = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]

    out = {"conditions": conditions, "linear": [[None]*5 for _ in range(5)],
            "mlp": [[None]*5 for _ in range(5)]}

    for i, X in enumerate(conditions):
        ckpt = os.path.join(args.lstm_root, X, "lstm.pt")
        if not os.path.exists(ckpt):
            print(f"  no LSTM ckpt for {X}, skipping row")
            continue
        # Determine D_in from saved weight
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        d_in = sd["lstm.weight_ih_l0"].shape[1] - 6  # subtract action one-hot
        lstm_X = IntegratorLSTM(d_in=d_in).to(device)
        lstm_X.load_state_dict(sd)
        lstm_X.eval()
        for j, Y in enumerate(conditions):
            print(f"  LSTM_{X}  fed encoder_{Y} ...", flush=True)
            try:
                X_tr, Y_tr = cache_hidden_for_pair(
                    lstm_X, os.path.join(args.feat_train, Y), args.raw_train,
                    device, max_traj=args.max_train_traj,
                )
                X_ev, Y_ev = cache_hidden_for_pair(
                    lstm_X, os.path.join(args.feat_eval, Y), args.raw_eval,
                    device, max_traj=10000,
                )
                # Use second-half eval window (250-500)
                # Re-slice: index 250-500 within each trajectory of length 500
                # Done in stack already? No, here we have flattened. Re-slice ok.
                # Quick mask:
                T = 500
                mask_eval = np.zeros(X_ev.size(0), dtype=bool)
                for t in range(0, X_ev.size(0), T):
                    s = t + 250
                    e = min(t + T, X_ev.size(0))
                    mask_eval[s:e] = True
                X_ev = X_ev[mask_eval]
                Y_ev = Y_ev[mask_eval]
                lin = train_probe(LinearProbe(X_tr.size(1)), X_tr, Y_tr, X_ev, Y_ev,
                                    device, steps=args.steps)
                mlp = train_probe(MLPProbe(X_tr.size(1)), X_tr, Y_tr, X_ev, Y_ev,
                                    device, steps=args.steps)
                out["linear"][i][j] = lin
                out["mlp"][i][j] = mlp
                print(f"    linear={lin:.4f} mlp={mlp:.4f}")
            except Exception as e:
                print(f"    ERROR {e}")
        json.dump(out, open(args.out_path, "w"), indent=2)
        print(f"  ... wrote partial {args.out_path}")

    # Final summary
    print("\n=== Linear R^2 transplant matrix ===")
    print("(rows = LSTM trained on; cols = encoder fed in)")
    print("        ", "  ".join(f"{c[:6]:>7s}" for c in conditions))
    for i, X in enumerate(conditions):
        row = "  ".join(f"{(out['linear'][i][j] if out['linear'][i][j] is not None else float('nan')):>7.3f}" for j in range(5))
        print(f"  {X[:6]:>6s}  {row}")
    print(f"\nSaved {args.out_path}")


if __name__ == "__main__":
    main()
