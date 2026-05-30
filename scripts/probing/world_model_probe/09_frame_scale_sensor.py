"""Frame-level spatial readout for the encoder-scale x sensor pilot.

This is a follow-up diagnostic for 05_run_scale_sensor.sh. The trajectory-held
out probe in 04_probe.py is a hard extrapolation test and failed for all cells
in pilot8. Here we ask the simpler question first: when held-out frames are
sampled from the same trajectory pool, is position linearly readable at all,
and does the small constrained encoder match a larger unconstrained encoder?

The result is a pilot sanity check, not the final Habitat evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def stack(records, window=(250, 500)):
    xs, ys = [], []
    for r in records:
        h = r["h"]
        c = r["c"]
        a = r["action"]
        p = r["agent_pos"]
        start, end = window
        start = max(0, start)
        end = min(h.size(0), end)
        xs.append(torch.cat([h[start:end], c[start:end], a[start:end]], dim=-1))
        ys.append(p[start:end])
    return torch.cat(xs, 0).numpy(), torch.cat(ys, 0).numpy()


def r2_per_dim(pred, target):
    ss_res = ((pred - target) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0).clip(min=1e-8)
    return 1.0 - ss_res / ss_tot


def load_records(cell_dir: Path, split: str):
    path = cell_dir / f"hidden_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=False)


def split_frames(x, y, seed: int, holdout: float):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_eval = max(1, int(len(idx) * holdout))
    eval_idx = idx[:n_eval]
    train_idx = idx[n_eval:]
    return x[train_idx], y[train_idx], x[eval_idx], y[eval_idx]


def fit_best_ridge(x_train, y_train, x_eval, y_eval, alphas):
    best = None
    rows = {}
    for alpha in alphas:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha),
        )
        model.fit(x_train, y_train)
        pred_eval = model.predict(x_eval)
        pred_train = model.predict(x_train)
        eval_r2 = r2_per_dim(pred_eval, y_eval)
        train_r2 = r2_per_dim(pred_train, y_train)
        row = {
            "alpha": float(alpha),
            "eval_r2_x": float(eval_r2[0]),
            "eval_r2_y": float(eval_r2[1]),
            "eval_r2_mean": float(eval_r2.mean()),
            "train_r2_mean": float(train_r2.mean()),
        }
        rows[str(alpha)] = row
        if best is None or row["eval_r2_mean"] > best["eval_r2_mean"]:
            best = row
    return best, rows


def fmt(x):
    return "NA" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--encoders", nargs="+", default=["dinov2_vits14", "dinov2_vitb14"])
    ap.add_argument("--conditions", nargs="+", default=["foveated", "uniform"])
    ap.add_argument("--small_encoder", default="dinov2_vits14")
    ap.add_argument("--large_encoder", default="dinov2_vitb14")
    ap.add_argument("--constrained_condition", default="foveated")
    ap.add_argument("--unconstrained_condition", default="uniform")
    ap.add_argument("--window_start", type=int, default=250)
    ap.add_argument("--window_end", type=int, default=500)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    args = ap.parse_args()

    lstm_root = Path(args.lstm_root)
    window = (args.window_start, args.window_end)

    cells = {}
    for encoder in args.encoders:
        cells[encoder] = {}
        for condition in args.conditions:
            cell_dir = lstm_root / encoder / condition
            train_records = load_records(cell_dir, "train")
            eval_records = load_records(cell_dir, "eval")
            x, y = stack(list(train_records) + list(eval_records), window=window)
            x_train, y_train, x_eval, y_eval = split_frames(x, y, args.seed, args.holdout)
            best, rows = fit_best_ridge(x_train, y_train, x_eval, y_eval, args.alphas)
            cells[encoder][condition] = {
                "n_frames": int(len(x)),
                "n_train": int(len(x_train)),
                "n_eval": int(len(x_eval)),
                "input_dim": int(x.shape[1]),
                "best": best,
                "alphas": rows,
            }
            print(
                f"{encoder}/{condition}: best alpha={best['alpha']} "
                f"eval R2={best['eval_r2_mean']:.4f} train R2={best['train_r2_mean']:.4f}",
                flush=True,
            )

    small = cells[args.small_encoder][args.constrained_condition]["best"]["eval_r2_mean"]
    large = cells[args.large_encoder][args.unconstrained_condition]["best"]["eval_r2_mean"]
    delta = small - large

    out = {
        "lstm_root": str(lstm_root),
        "window": list(window),
        "holdout": args.holdout,
        "seed": args.seed,
        "cells": cells,
        "key_comparison": {
            "small_constrained": f"{args.small_encoder}/{args.constrained_condition}",
            "large_unconstrained": f"{args.large_encoder}/{args.unconstrained_condition}",
            "small_constrained_eval_r2": small,
            "large_unconstrained_eval_r2": large,
            "delta": delta,
            "match_or_beat": bool(small >= large),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    lines = [
        "# Frame-level scale x sensor pilot",
        "",
        "| encoder | sensor | best alpha | linear R2 | train R2 | n eval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for encoder in args.encoders:
        for condition in args.conditions:
            row = cells[encoder][condition]["best"]
            lines.append(
                f"| {encoder} | {condition} | {row['alpha']:.1g} | "
                f"{fmt(row['eval_r2_mean'])} | {fmt(row['train_r2_mean'])} | "
                f"{cells[encoder][condition]['n_eval']} |"
            )
    lines += [
        "",
        "## Key comparison",
        "",
        f"`{out['key_comparison']['small_constrained']}` vs "
        f"`{out['key_comparison']['large_unconstrained']}`",
        "",
        f"Delta: {fmt(delta)}",
        f"Match/beat: {out['key_comparison']['match_or_beat']}",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
