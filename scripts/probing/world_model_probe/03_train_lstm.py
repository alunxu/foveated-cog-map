"""Train a small LSTM integrator on cached DINOv2 features.

The LSTM is held fixed across all 5 sensor conditions (architecture, optimiser,
schedule), so any difference in probe R^2 across conditions reflects the
encoder-bandwidth axis we are testing, not the recurrent integration capacity.

Architecture (mirrors our paper's PointNav recurrent core in spirit):
  nn.LSTM(input_size = D_enc + 6 (one-hot last action),
          hidden_size = 512, num_layers = 2, batch_first = True)

Training objective: MSE on next-frame DINOv2 feature given current feature
+ last action. This is the canonical representation-prediction loss used in
e.g. SPR / DreamerV3 (without reconstruction).

Output: a checkpoint .pt with state_dict + per-step (h_t, c_t) saved for
the train and eval splits at every frame (used by 04_probe.py).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from glob import glob

import numpy as np
import torch
import torch.nn as nn


class FeatureSeqDataset(torch.utils.data.Dataset):
    """Returns full (T, D) feature sequences + action one-hots from cached .pt files."""

    def __init__(self, feat_dir: str, raw_dir: str):
        self.feat_files = sorted(glob(os.path.join(feat_dir, "traj_*.pt")))
        self.raw_dir = raw_dir
        if not self.feat_files:
            raise RuntimeError(f"No feature files in {feat_dir}")

    def __len__(self):
        return len(self.feat_files)

    def __getitem__(self, idx):
        feat_path = self.feat_files[idx]
        feats = torch.load(feat_path, weights_only=True)  # (T, D)
        # Match raw npz to get actions + agent_pos
        raw_path = os.path.join(self.raw_dir, os.path.basename(feat_path).replace(".pt", ".npz"))
        d = np.load(raw_path)
        actions = torch.from_numpy(d["action"]).float()  # (T, 6)
        agent_pos = torch.from_numpy(d["agent_pos"]).float()  # (T, 2)
        return feats, actions, agent_pos


def collate_seq(batch, seq_len: int):
    """Random-crop each sequence to seq_len. Returns (B, S, D), (B, S, 6), (B, S, 2)."""
    feats, actions, pos = [], [], []
    for f, a, p in batch:
        T = f.size(0)
        if T <= seq_len:
            # pad
            pad_T = seq_len - T
            f = torch.cat([f, torch.zeros(pad_T, f.size(1))], 0)
            a = torch.cat([a, torch.zeros(pad_T, a.size(1))], 0)
            p = torch.cat([p, torch.zeros(pad_T, p.size(1))], 0)
        else:
            start = torch.randint(0, T - seq_len, (1,)).item()
            f = f[start:start + seq_len]
            a = a[start:start + seq_len]
            p = p[start:start + seq_len]
        feats.append(f)
        actions.append(a)
        pos.append(p)
    return torch.stack(feats), torch.stack(actions), torch.stack(pos)


class IntegratorLSTM(nn.Module):
    def __init__(self, d_in: int, d_action: int = 6, d_hidden: int = 512, n_layers: int = 2):
        super().__init__()
        self.d_in = d_in
        self.d_action = d_action
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=d_in + d_action, hidden_size=d_hidden,
                              num_layers=n_layers, batch_first=True)
        # Predict next feature: hidden -> D_in
        self.proj = nn.Linear(d_hidden, d_in)

    def forward(self, feats, actions):
        """feats (B, S, D), actions (B, S, 6).

        At each step t the LSTM sees feats[t] || actions[t], we predict feats[t+1].
        Standard "predict next" loss.
        """
        x = torch.cat([feats, actions], dim=-1)  # (B, S, D+6)
        h_seq, (h_n, c_n) = self.lstm(x)
        pred_next = self.proj(h_seq)  # (B, S, D)
        return pred_next, h_seq

    def encode_full(self, feats, actions):
        """Run LSTM over (1, T, D), return (h_t, c_t) at every step.

        We need h_seq AND c_seq at every step for the probe input. PyTorch's
        cuDNN LSTM doesn't expose c_seq, so we step manually.
        """
        device = feats.device
        T, D = feats.shape
        x = torch.cat([feats, actions], dim=-1).unsqueeze(0)  # (1, T, D+6)

        # Run layer-by-layer manually to capture (h, c) at each step.
        # Simpler: use nn.LSTMCell stack mirror. To keep weight compatibility,
        # we extract weights from self.lstm and run a hand-rolled loop.
        L = self.n_layers
        H = self.d_hidden

        h_outs = torch.empty(T, L, H, device=device)
        c_outs = torch.empty(T, L, H, device=device)

        # Reuse the trained weights via individual cells
        cells = nn.ModuleList()
        for layer in range(L):
            in_size = (D + 6) if layer == 0 else H
            cell = nn.LSTMCell(input_size=in_size, hidden_size=H)
            # Copy weights from self.lstm
            cell.weight_ih = nn.Parameter(getattr(self.lstm, f"weight_ih_l{layer}"))
            cell.weight_hh = nn.Parameter(getattr(self.lstm, f"weight_hh_l{layer}"))
            cell.bias_ih = nn.Parameter(getattr(self.lstm, f"bias_ih_l{layer}"))
            cell.bias_hh = nn.Parameter(getattr(self.lstm, f"bias_hh_l{layer}"))
            cells.append(cell)
        cells = cells.to(device)
        cells.eval()

        h = [torch.zeros(1, H, device=device) for _ in range(L)]
        c = [torch.zeros(1, H, device=device) for _ in range(L)]
        with torch.no_grad():
            for t in range(T):
                inp = x[:, t]
                for layer in range(L):
                    h[layer], c[layer] = cells[layer](inp, (h[layer], c[layer]))
                    inp = h[layer]
                    h_outs[t, layer] = h[layer].squeeze(0)
                    c_outs[t, layer] = c[layer].squeeze(0)
        return h_outs, c_outs  # (T, L, H), (T, L, H)


def train_lstm(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    feat_dir = os.path.join(args.feat_root, args.condition)
    train_ds = FeatureSeqDataset(feat_dir, args.raw_train)
    eval_ds = FeatureSeqDataset(os.path.join(args.eval_feat_root, args.condition), args.raw_eval)
    sample_f, _, _ = train_ds[0]
    D = sample_f.size(1)
    print(f"  train trajs: {len(train_ds)}  eval trajs: {len(eval_ds)}  D={D}")

    model = IntegratorLSTM(d_in=D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def loader(ds, batch, seq_len, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch, shuffle=shuffle, num_workers=0,
            collate_fn=lambda b: collate_seq(b, seq_len),
        )

    train_loader = loader(train_ds, args.batch_size, args.seq_len, shuffle=True)
    train_iter = iter(train_loader)

    losses = []
    t0 = time.time()
    for step in range(args.steps):
        try:
            feats, actions, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            feats, actions, _ = next(train_iter)
        feats = feats.to(device)
        actions = actions.to(device)
        # Target: feats shifted by one step
        pred, _ = model(feats[:, :-1], actions[:, :-1])
        target = feats[:, 1:].detach()
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 200 == 0 or step == args.steps - 1:
            avg = np.mean(losses[-50:])
            elapsed = time.time() - t0
            print(f"    step {step:5d}  loss={avg:.4f}  elapsed={elapsed/60:.1f}m", flush=True)

    # Cache hidden states for full eval split + a subset of train
    print("  caching hidden states for probe...")
    out = {
        "config": {
            "condition": args.condition, "lr": args.lr, "steps": args.steps,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
        },
        "loss_curve": losses,
    }

    @torch.no_grad()
    def cache_hiddens(ds, name, max_traj=None):
        records = []
        for i in range(min(len(ds), max_traj or len(ds))):
            f, a, p = ds[i]
            f, a = f.to(device), a.to(device)
            h, c = model.encode_full(f, a)  # (T, L, H), (T, L, H)
            # Use last layer's h, c
            h_last = h[:, -1]
            c_last = c[:, -1]
            records.append({
                "h": h_last.cpu(),
                "c": c_last.cpu(),
                "action": a.cpu(),
                "agent_pos": p,
            })
        torch.save(records, os.path.join(args.out_dir, f"hidden_{name}.pt"))
        print(f"    wrote hidden_{name}.pt ({len(records)} trajectories)")
        return records

    os.makedirs(args.out_dir, exist_ok=True)
    cache_hiddens(eval_ds, "eval")
    cache_hiddens(train_ds, "train", max_traj=args.train_cache_n)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "lstm.pt"))
    json.dump(out, open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)
    print(f"  saved {args.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_root", required=True)
    ap.add_argument("--eval_feat_root", required=True)
    ap.add_argument("--raw_train", required=True, help="Dir with raw .npz for train")
    ap.add_argument("--raw_eval", required=True, help="Dir with raw .npz for eval")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_cache_n", type=int, default=200,
                     help="Cache hidden states for first N train trajectories.")
    args = ap.parse_args()
    train_lstm(args)


if __name__ == "__main__":
    main()
