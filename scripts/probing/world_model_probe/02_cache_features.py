"""Cache DINOv2-Base CLS features for each Memory-Maze trajectory under each
of the 5 sensor conditions.

The 5 conditions mirror our paper's chassis:
  blind             - RGB replaced by zeros (encoder still runs; constant token)
  coarse            - bilinear downsample 64 -> 14, upsample 14 -> 56 (1 patch worth of detail)
  foveated          - native 56x56 with Gaussian blur sigma=4 px
  uniform           - native 56x56 (downsampled from 64 via bilinear)
  foveated_logpolar - 56x56 log-polar warp with central magnification

Output: one .pt per trajectory per condition, shape (T, 768) float32.
Path layout: /tmp/wmprobe_features/<condition>/traj_<idx>.pt

Throughput on Mac MPS (DINOv2-Base):
  56x56  -> ~920 fps
  28x28  -> ~2000 fps
  14x14  -> ~3400 fps
~5h budget if we cache 5 conds x ~600 traj x 500 steps = 1.5M frames per cond.
At avg 1500 fps this is 1000s per cond x 5 conds = ~80 min total.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]


def make_transform(condition: str, target_res: int = 56, blur_sigma: float = 4.0):
    """Return a function that maps (B, 3, 64, 64) float [0,1] -> (B, 3, R, R) float."""
    R = target_res

    def to_R(x):
        return F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False, antialias=True)

    if condition == "blind":
        def f(x):
            return torch.zeros(x.size(0), 3, R, R, device=x.device, dtype=x.dtype)
        return f

    if condition == "coarse":
        # Downsample to 14, upsample back to R
        def f(x):
            small = F.interpolate(x, size=(14, 14), mode="bilinear",
                                  align_corners=False, antialias=True)
            return F.interpolate(small, size=(R, R), mode="bilinear",
                                  align_corners=False, antialias=True)
        return f

    if condition == "uniform":
        def f(x):
            return to_R(x)
        return f

    if condition == "foveated":
        # Gaussian blur sigma=blur_sigma at native R
        # kernel size: 2*ceil(3*sigma) + 1
        ksize = 2 * int(math.ceil(3 * blur_sigma)) + 1

        def f(x):
            x_R = to_R(x)
            return TF.gaussian_blur(x_R, kernel_size=[ksize, ksize], sigma=[blur_sigma, blur_sigma])
        return f

    if condition == "foveated_logpolar":
        # Compute log-polar sampling grid once
        ys = torch.linspace(-1, 1, R)
        xs = torch.linspace(-1, 1, R)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        # rho in [0, 1], theta in [-pi, pi]
        rho = torch.sqrt(xx ** 2 + yy ** 2).clamp(min=1e-6)
        theta = torch.atan2(yy, xx)
        # Map to log-polar: r' = log(1 + alpha*rho)
        alpha = 4.0
        r_lp = torch.log1p(alpha * rho) / math.log1p(alpha)  # [0, 1]
        # Reverse: source pixel at (r_lp * cos(theta), r_lp * sin(theta))
        sx = r_lp * torch.cos(theta)
        sy = r_lp * torch.sin(theta)
        grid = torch.stack([sx, sy], dim=-1).unsqueeze(0)  # (1, R, R, 2)

        def f(x):
            x_R = to_R(x)
            g = grid.to(x.device, x.dtype).expand(x.size(0), -1, -1, -1)
            return F.grid_sample(x_R, g, mode="bilinear", padding_mode="border",
                                  align_corners=False)
        return f

    raise ValueError(condition)


@torch.no_grad()
def cache_one_condition(model, transform, npz_files, out_dir, device, batch_size=64,
                         normalize=True):
    """Cache DINOv2-B CLS features for all trajectories under one transform.

    Output is one .pt per trajectory: shape (T, 768) float32.
    """
    os.makedirs(out_dir, exist_ok=True)
    # ImageNet normalisation (DINOv2 was trained on this stat)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    t0 = time.time()
    n_frames_total = 0
    for fi, npz_path in enumerate(npz_files):
        out_path = os.path.join(out_dir, os.path.basename(npz_path).replace(".npz", ".pt"))
        if os.path.exists(out_path):
            continue
        d = np.load(npz_path)
        imgs = d["image"]  # (T, 64, 64, 3) uint8
        T = imgs.shape[0]
        # to (T, 3, 64, 64) float [0, 1]
        x_all = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
        feats = torch.empty(T, 768, dtype=torch.float32)
        for i in range(0, T, batch_size):
            x = x_all[i:i + batch_size].to(device, non_blocking=True)
            x = transform(x)
            if normalize:
                x = (x - mean) / std
            y = model(x)  # (B, 768)
            feats[i:i + batch_size] = y.float().cpu()
        torch.save(feats, out_path)
        n_frames_total += T
        if (fi + 1) % 20 == 0:
            elapsed = time.time() - t0
            fps = n_frames_total / elapsed
            eta = (len(npz_files) - fi - 1) * (T / fps) if fps > 0 else 0
            print(f"    [{fi+1}/{len(npz_files)}] {fps:.0f} fps  eta {eta/60:.1f}m", flush=True)
    return time.time() - t0, n_frames_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                     help="Dir containing traj_*.npz")
    ap.add_argument("--out_root", type=str, required=True,
                     help="Output root; per-condition subdirs will be created.")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS,
                     choices=CONDITIONS)
    ap.add_argument("--target_res", type=int, default=56)
    ap.add_argument("--blur_sigma", type=float, default=4.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--encoder", type=str, default="dinov2_vitb14",
                     choices=["dinov2_vits14", "dinov2_vitb14"])
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process first N trajectories (for debug).")
    args = ap.parse_args()

    os.environ["TORCH_HOME"] = "/tmp/wmprobe_venv/torch_hub"

    device = torch.device("mps" if torch.backends.mps.is_available()
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device = {device}")
    print(f"loading {args.encoder} ...")
    model = torch.hub.load("facebookresearch/dinov2", args.encoder, verbose=False)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded ({n_params/1e6:.1f}M params)")

    npz_files = sorted(glob(os.path.join(args.data_dir, "traj_*.npz")))
    if args.limit is not None:
        npz_files = npz_files[: args.limit]
    print(f"will cache features for {len(npz_files)} trajectories from {args.data_dir}")
    if len(npz_files) == 0:
        print("No trajectories found; aborting.")
        return

    for cond in args.conditions:
        print(f"\n=== condition: {cond} ===")
        out_dir = os.path.join(args.out_root, cond)
        transform = make_transform(cond, args.target_res, args.blur_sigma)
        elapsed, frames = cache_one_condition(
            model, transform, npz_files, out_dir, device, batch_size=args.batch_size,
        )
        fps = frames / elapsed if elapsed else 0
        print(f"  done {cond}: {frames} frames in {elapsed/60:.1f}m ({fps:.0f} fps)")


if __name__ == "__main__":
    main()
