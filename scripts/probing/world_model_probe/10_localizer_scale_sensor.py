"""Supervised localization pilot for encoder scale x sensor route robustness.

Why this exists:
    The feature-prediction world-model pilot is useful but weak for the blind
    condition: when visual input is blank, the next-feature target is nearly
    constant, so the recurrent state is not forced to integrate motion. This
    script trains a small recurrent localizer instead. Its target is agent_pos,
    so the blind condition becomes a path-integration control, while sighted
    conditions can use visual route information.

Inputs per step:
    frozen visual feature, one-hot action, heading vector, start position

Target:
    current absolute agent_pos

Evaluation:
    held-out trajectories, optionally restricted to a late window.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ENCODER_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0).clamp(min=1e-8)
    return (1.0 - ss_res / ss_tot).mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(((pred - target) ** 2).mean()).item()


def position_stats(raw_dir: str):
    files = sorted(glob(os.path.join(raw_dir, "traj_*.npz")))
    pos = []
    for path in files:
        pos.append(np.load(path)["agent_pos"].astype(np.float32))
    arr = np.concatenate(pos, axis=0)
    mean = torch.from_numpy(arr.mean(axis=0)).float()
    std = torch.from_numpy(arr.std(axis=0).clip(min=1e-4)).float()
    return mean, std


class LocalizerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_dir: str,
        feat_root: str,
        encoder: str,
        condition: str,
        split: str,
        pos_mean: torch.Tensor,
        pos_std: torch.Tensor,
    ):
        self.raw_files = sorted(glob(os.path.join(raw_dir, "traj_*.npz")))
        if not self.raw_files:
            raise RuntimeError(f"No raw trajectories in {raw_dir}")
        self.feat_root = Path(feat_root)
        self.encoder = encoder
        self.condition = condition
        self.split = split
        self.pos_mean = pos_mean
        self.pos_std = pos_std
        if encoder not in ENCODER_DIMS:
            raise ValueError(f"Unknown encoder {encoder}; known={sorted(ENCODER_DIMS)}")
        self.d_feat = ENCODER_DIMS[encoder]

    def __len__(self):
        return len(self.raw_files)

    def _load_features(self, raw_path: str, length: int):
        if self.condition == "blind":
            return torch.zeros(length, self.d_feat, dtype=torch.float32)
        filename = Path(raw_path).name.replace(".npz", ".pt")
        candidates = [
            self.feat_root / self.encoder / self.split / self.condition / filename,
            self.feat_root / self.encoder / self.condition / filename,
            self.feat_root / self.condition / filename,
        ]
        feat_path = next((path for path in candidates if path.exists()), candidates[0])
        if not feat_path.exists():
            raise FileNotFoundError(feat_path)
        return torch.load(feat_path, weights_only=True).float()

    def __getitem__(self, idx):
        raw_path = self.raw_files[idx]
        d = np.load(raw_path)
        pos = torch.from_numpy(d["agent_pos"].astype(np.float32))
        feats = self._load_features(raw_path, pos.size(0))
        action = torch.from_numpy(d["action"].astype(np.float32))
        heading = torch.from_numpy(d["agent_dir"].astype(np.float32))
        start = pos[0:1].expand_as(pos)
        x = torch.cat([feats, action, heading, start], dim=-1)
        y = (pos - self.pos_mean) / self.pos_std
        return x, y, pos


def collate_crop(batch, seq_len: int):
    xs, ys = [], []
    for x, y, _ in batch:
        t = x.size(0)
        if t <= seq_len:
            start = 0
        else:
            start = torch.randint(0, t - seq_len, (1,)).item()
        end = min(t, start + seq_len)
        xs.append(x[start:end])
        ys.append(y[start:end])
    max_len = max(v.size(0) for v in xs)
    padded_x, padded_y, mask = [], [], []
    for x, y in zip(xs, ys):
        pad = max_len - x.size(0)
        if pad:
            x = torch.cat([x, torch.zeros(pad, x.size(1))], 0)
            y = torch.cat([y, torch.zeros(pad, y.size(1))], 0)
        padded_x.append(x)
        padded_y.append(y)
        mask.append(torch.cat([torch.ones(max_len - pad), torch.zeros(pad)]))
    return torch.stack(padded_x), torch.stack(padded_y), torch.stack(mask)


class Localizer(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))

    def forward(self, x):
        x = self.norm(x)
        h, _ = self.lstm(x)
        return self.head(h)


def train_one(
    train_ds,
    eval_ds,
    seed: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    hidden: int,
    lr: float,
    device: torch.device,
    window: tuple[int, int],
):
    seed_all(seed)
    sample_x, _, _ = train_ds[0]
    model = Localizer(sample_x.size(1), hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_crop(b, seq_len),
    )
    iterator = iter(loader)
    t0 = time.time()
    losses = []
    for step in range(steps):
        try:
            x, y, mask = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, mask = next(iterator)
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device).unsqueeze(-1)
        pred = model(x)
        loss = (((pred - y) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 500 == 0 or step == steps - 1:
            print(
                f"    step {step:5d} loss={np.mean(losses[-50:]):.4f} "
                f"elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    model.eval()
    preds, targets = [], []
    start, end = window
    with torch.no_grad():
        for x, y_norm, pos in eval_ds:
            x = x.unsqueeze(0).to(device)
            pred_norm = model(x).squeeze(0).cpu()
            pred = pred_norm * eval_ds.pos_std + eval_ds.pos_mean
            s = max(0, start)
            e = min(pos.size(0), end)
            preds.append(pred[s:e])
            targets.append(pos[s:e])
    pred_all = torch.cat(preds, 0)
    target_all = torch.cat(targets, 0)
    return {
        "eval_r2": r2_score(pred_all, target_all),
        "eval_rmse": rmse(pred_all, target_all),
        "final_loss": float(np.mean(losses[-50:])),
    }


def fmt(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_train", required=True)
    ap.add_argument("--data_eval", required=True)
    ap.add_argument("--feat_train_root", required=True)
    ap.add_argument("--feat_eval_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--encoders", nargs="+", default=["dinov2_vits14", "dinov2_vitb14"])
    ap.add_argument("--conditions", nargs="+", default=["blind", "foveated", "uniform"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window_start", type=int, default=250)
    ap.add_argument("--window_end", type=int, default=500)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    pos_mean, pos_std = position_stats(args.data_train)
    print(f"position mean={pos_mean.tolist()} std={pos_std.tolist()}")

    cells = {}
    for encoder in args.encoders:
        cells[encoder] = {}
        for condition in args.conditions:
            print(f"\n=== {encoder}/{condition} ===", flush=True)
            train_ds = LocalizerDataset(
                args.data_train, args.feat_train_root, encoder, condition, "train", pos_mean, pos_std
            )
            eval_ds = LocalizerDataset(
                args.data_eval, args.feat_eval_root, encoder, condition, "eval", pos_mean, pos_std
            )
            seed_rows = []
            for seed in args.seeds:
                print(f"  seed {seed}", flush=True)
                row = train_one(
                    train_ds,
                    eval_ds,
                    seed=seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    hidden=args.hidden,
                    lr=args.lr,
                    device=device,
                    window=(args.window_start, args.window_end),
                )
                row["seed"] = seed
                seed_rows.append(row)
                print(
                    f"    eval R2={row['eval_r2']:.4f} "
                    f"RMSE={row['eval_rmse']:.4f}",
                    flush=True,
                )
            r2s = [r["eval_r2"] for r in seed_rows]
            rmses = [r["eval_rmse"] for r in seed_rows]
            cells[encoder][condition] = {
                "seeds": seed_rows,
                "mean_eval_r2": float(np.mean(r2s)),
                "std_eval_r2": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                "mean_eval_rmse": float(np.mean(rmses)),
                "std_eval_rmse": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            }

    small = cells["dinov2_vits14"]["foveated"]["mean_eval_r2"]
    large = cells["dinov2_vitb14"]["uniform"]["mean_eval_r2"]
    out = {
        "task": "supervised recurrent localization",
        "data_train": args.data_train,
        "data_eval": args.data_eval,
        "window": [args.window_start, args.window_end],
        "cells": cells,
        "key_comparison": {
            "small_constrained": "dinov2_vits14/foveated",
            "large_unconstrained": "dinov2_vitb14/uniform",
            "delta_eval_r2": small - large,
            "match_or_beat": bool(small >= large),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    lines = [
        "# Supervised localizer route robustness pilot",
        "",
        "| encoder | route/sensor | mean R2 | sd R2 | mean RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for encoder in args.encoders:
        for condition in args.conditions:
            row = cells[encoder][condition]
            route = "integration" if condition == "blind" else "visual"
            lines.append(
                f"| {encoder} | {condition} ({route}) | "
                f"{fmt(row['mean_eval_r2'])} | {fmt(row['std_eval_r2'])} | "
                f"{fmt(row['mean_eval_rmse'])} |"
            )
    lines += [
        "",
        "## Key comparison",
        "",
        "`dinov2_vits14/foveated` vs `dinov2_vitb14/uniform`",
        "",
        f"Delta R2: {fmt(small - large)}",
        f"Match/beat: {small >= large}",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
