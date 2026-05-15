r"""
WJ-C stage 2-3: train an occupancy decoder per condition (Wijmans Fig 4
replication).

Stage 2 (per-episode dataset construction): for each episode in the
condition's existing probing rollout, extract:
  - the final-step (h_T, c_T) at the top LSTM layer (1024-d)
  - the full agent trajectory (positions in world frame)
  - a 32×32 allocentric occupancy target grid (16m × 16m at 0.5m/cell)
    centered on the episode start, in the scene's world frame
  - a "within 2.5m of trajectory" mask over the same grid

Stage 3 (decoder training + 5-fold episode-level CV): a small MLP maps
1024-d (h_T, c_T) input to 32×32 occupancy prediction. Trained with BCE
loss restricted to the trajectory mask. Reports per-fold IoU.

Reads:
    --hidden-npz  /scratch/.../<cond>_gibson_det.npz   (existing)
    --scenes-txt  /scratch/.../<cond>_gibson_det_scenes.txt  (existing)
    --scene-occ-dir /scratch/.../scene_occupancy/<scene>.{npz,json}
                                                (compute_scene_occupancy.py)

Writes:
    --out <out>/<cond>_occupancy.json + sample renders + per-cond IoU
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold


# Decoder architecture: small MLP, output 32×32 = 1024 logits.
class OccupancyDecoder(nn.Module):
    def __init__(self, in_dim: int = 1024, grid_size: int = 32, hidden: int = 512):
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, grid_size * grid_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.grid_size, self.grid_size)


def project_to_grid(positions_xz, start_xz, grid_size: int, grid_res: float):
    """Map world-frame (x, z) → grid cell (row, col).

    Grid is centered on start_xz; row 0 = max z, row G-1 = min z.
    Cell width = grid_res. Total span = grid_size × grid_res.
    """
    half = (grid_size * grid_res) / 2.0
    rel = positions_xz - start_xz
    cols = ((rel[:, 0] + half) / grid_res).astype(np.int64)
    rows = ((half - rel[:, 1]) / grid_res).astype(np.int64)  # flip z axis
    valid = (cols >= 0) & (cols < grid_size) & (rows >= 0) & (rows < grid_size)
    return rows, cols, valid


def make_target_grid(scene_occ_data: dict, start_xz, grid_size: int,
                     grid_res: float) -> np.ndarray:
    """Sample the scene's allocentric occupancy grid into a grid_size ×
    grid_size patch centered on start_xz."""
    occ = scene_occ_data["occupancy"]                  # (H, W) uint8
    lo = np.asarray(scene_occ_data["world_lower"])
    scene_res = float(scene_occ_data["grid_res"])

    H, W = occ.shape
    half = (grid_size * grid_res) / 2.0

    target = np.zeros((grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            world_x = start_xz[0] - half + (j + 0.5) * grid_res
            world_z = start_xz[1] + half - (i + 0.5) * grid_res
            col_s = int((world_x - lo[0]) / scene_res)
            row_s = int((world_z - lo[2]) / scene_res)
            if 0 <= col_s < W and 0 <= row_s < H:
                target[i, j] = occ[row_s, col_s]
    return target


def make_trajectory_mask(positions_xz, start_xz, grid_size: int,
                         grid_res: float, dilate_m: float = 2.5) -> np.ndarray:
    """Boolean mask: cells whose center is within `dilate_m` of any
    point in the agent's trajectory."""
    half = (grid_size * grid_res) / 2.0
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    # Per-cell distance to nearest trajectory point.
    for i in range(grid_size):
        for j in range(grid_size):
            cx = start_xz[0] - half + (j + 0.5) * grid_res
            cz = start_xz[1] + half - (i + 0.5) * grid_res
            d2 = (positions_xz[:, 0] - cx) ** 2 + (positions_xz[:, 1] - cz) ** 2
            if d2.min() <= dilate_m ** 2:
                mask[i, j] = True
    return mask


def build_dataset(args):
    print(f"Loading hidden states: {args.hidden_npz}")
    d = np.load(args.hidden_npz)
    h_layers = d["h_layers"]   # (N, n_layers, hidden) for LSTM
    c_layers = d["c_layers"]
    positions = d["positions"]  # (N, 3) world frame
    episode_ids = d["episode_ids"]
    scene_ids = d["scene_ids"]

    scenes_txt = Path(str(args.hidden_npz).replace(".npz", "_scenes.txt"))
    if not scenes_txt.exists():
        raise FileNotFoundError(f"Missing scenes lookup: {scenes_txt}")
    with open(scenes_txt) as f:
        scene_lookup = []
        for l in f:
            l = l.strip()
            if not l:
                continue
            parts = l.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                scene_lookup.append(parts[1])
            else:
                scene_lookup.append(l)

    unique_eps = np.unique(episode_ids)
    print(f"Episodes: {len(unique_eps)} (total steps: {len(h_layers)})")

    Xs, Ys, Ms = [], [], []
    skip = 0
    for ep in unique_eps:
        ep_mask = (episode_ids == ep)
        idxs = np.where(ep_mask)[0]
        if len(idxs) < 2: skip += 1; continue
        last_idx = idxs[-1]

        # Top-layer (h, c) → 1024-d input.
        h_top = h_layers[last_idx, -1].astype(np.float32)
        c_top = c_layers[last_idx, -1].astype(np.float32)
        feat = np.concatenate([h_top, c_top])          # (1024,)

        # Get scene basename.
        sid = int(scene_ids[last_idx])
        scene_path = scene_lookup[sid] if sid < len(scene_lookup) else None
        if scene_path is None: skip += 1; continue
        scene_name = Path(scene_path).stem

        scene_npz = args.scene_occ_dir / f"{scene_name}.npz"
        scene_json = args.scene_occ_dir / f"{scene_name}.json"
        if not (scene_npz.exists() and scene_json.exists()):
            skip += 1; continue
        scene_occ_data = {
            "occupancy": np.load(scene_npz)["occupancy"],
            **json.loads(scene_json.read_text()),
        }

        # Trajectory in (x, z) world coords.
        traj = positions[ep_mask][:, [0, 2]].astype(np.float64)
        start_xz = traj[0]

        target = make_target_grid(scene_occ_data, start_xz, args.grid_size, args.grid_res)
        traj_mask = make_trajectory_mask(traj, start_xz, args.grid_size,
                                         args.grid_res, dilate_m=args.dilate_m)
        if traj_mask.sum() < 5:
            skip += 1; continue

        Xs.append(feat); Ys.append(target); Ms.append(traj_mask.astype(np.float32))

    print(f"Built dataset: {len(Xs)} episodes ({skip} skipped)")
    X = np.stack(Xs); Y = np.stack(Ys); M = np.stack(Ms)
    return X, Y, M


def train_and_eval(X, Y, M, args, device):
    n = X.shape[0]
    folds = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_ious = []; fold_preds = []

    for fi, (tr, te) in enumerate(folds.split(np.arange(n))):
        Xtr = torch.from_numpy(X[tr]).to(device)
        Ytr = torch.from_numpy(Y[tr]).to(device)
        Mtr = torch.from_numpy(M[tr]).to(device)
        Xte = torch.from_numpy(X[te]).to(device)
        Yte = torch.from_numpy(Y[te]).to(device)
        Mte = torch.from_numpy(M[te]).to(device)

        model = OccupancyDecoder(in_dim=X.shape[1], grid_size=args.grid_size).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        for ep in range(args.epochs):
            model.train()
            idx = torch.randperm(len(tr), device=device)
            for s in range(0, len(idx), args.batch):
                b = idx[s:s + args.batch]
                logits = model(Xtr[b])
                loss = bce(logits, Ytr[b]) * Mtr[b]
                loss = loss.sum() / Mtr[b].sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(Xte)) > 0.5
            gt = Yte > 0.5
            mask = Mte > 0.5
            inter = (preds & gt & mask).float().sum(dim=(1, 2))
            union = ((preds | gt) & mask).float().sum(dim=(1, 2))
            iou = (inter / union.clamp(min=1)).cpu().numpy()
            fold_ious.append(float(iou.mean()))
            # Save a few sample predictions for visualization.
            if fi == 0:
                fold_preds = (preds.float().cpu().numpy(),
                              Yte.cpu().numpy(),
                              Mte.cpu().numpy())
        print(f"  fold {fi}: IoU={iou.mean():.3f} (n={len(te)})")

    return fold_ious, fold_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-npz", type=Path, required=True)
    ap.add_argument("--scene-occ-dir", type=Path, required=True)
    ap.add_argument("--cond-name", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--grid-res", type=float, default=0.5,
                    help="Decoder grid resolution (m/cell). 32×0.5=16m view")
    ap.add_argument("--dilate-m", type=float, default=2.5,
                    help="Loss mask: only cells within this distance of "
                         "trajectory contribute to BCE / IoU")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print(f"  WJ-C occupancy decoder — {args.cond_name}")
    print("=" * 64)

    X, Y, M = build_dataset(args)
    if len(X) < args.n_folds:
        print(f"WARN too few episodes ({len(X)}); aborting"); return

    fold_ious, sample_preds = train_and_eval(X, Y, M, args, device)

    out = {
        "condition": args.cond_name,
        "n_episodes": int(len(X)),
        "grid_size": int(args.grid_size),
        "grid_res": float(args.grid_res),
        "dilate_m": float(args.dilate_m),
        "fold_ious": [float(v) for v in fold_ious],
        "mean_iou": float(np.mean(fold_ious)),
        "std_iou": float(np.std(fold_ious)),
    }
    out_json = args.out_dir / f"{args.cond_name}_occupancy.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nMean IoU: {out['mean_iou']:.3f} ± {out['std_iou']:.3f}")
    print(f"Wrote {out_json}")

    if sample_preds:
        out_npz = args.out_dir / f"{args.cond_name}_samples.npz"
        np.savez_compressed(out_npz,
                            preds=sample_preds[0],
                            targets=sample_preds[1],
                            masks=sample_preds[2])
        print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
