#!/usr/bin/env python3
"""Capture a real RCP rollout video with synchronized LSTM-memory traces.

This script is intended to run inside the RCP Habitat container. It renders a
single deterministic PointGoal episode for the five canonical agents and writes
a compact composite MP4:

  row 1: display-friendly visualization of the policy's visual bandwidth
  row 2: ground-truth simulator trajectory on a Habitat bird-view map
  row 3: h_2 manifold trajectory, using a condition-specific PCA basis fit
         on the same excursion rollouts; dots are colored by true relative x
         position, so the visual asks whether memory geometry carries a
         spatial axis without forcing a decoder to draw a map

The output is a website-ready asset; no raw NPZ or frame dump needs to be copied
back to the laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.habitat  # noqa: F401  register custom sensors/policies
import habitat
from src.utils.habitat_env import heading_from_quaternion, load_habitat_config, load_policy


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    color: tuple[int, int, int]
    config: str
    ckpt: str
    excursion: str


CONDITIONS = [
    Condition(
        "blind", "Blind", (111, 116, 123),
        "pointnav/ddppo_pointnav_blind_gibson",
        "/scratch/wxu/habitat_checkpoints_rcp/dh-blind/ckpt.49.pth",
        "blind_excursion.npz",
    ),
    Condition(
        "coarse", "Coarse", (43, 109, 166),
        "pointnav/ddppo_pointnav_coarse_gibson",
        "/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth",
        "coarse_excursion.npz",
    ),
    Condition(
        "foveated", "Foveated", (198, 77, 62),
        "pointnav/ddppo_pointnav_foveated_gibson",
        "/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2/ckpt.49.pth",
        "foveated_excursion.npz",
    ),
    Condition(
        "logpolar", "Log-polar", (213, 138, 23),
        "pointnav/ddppo_pointnav_foveated_logpolar_gibson",
        "/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4/ckpt.49.pth",
        "foveated_logpolar_excursion.npz",
    ),
    Condition(
        "uniform", "Uniform", (63, 155, 79),
        "pointnav/ddppo_pointnav_uniform_gibson",
        "/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3/ckpt.49.pth",
        "uniform_excursion.npz",
    ),
]


def top_h2(rnn: torch.Tensor) -> np.ndarray:
    """Return policy-readable top-layer h_2 from Habitat recurrent state."""
    # LSTM state layout used in this project is [h0,c0,h1,c1,h2,c2].
    idx = 4 if rnn.shape[1] >= 5 else rnn.shape[1] - 1
    return rnn[0, idx].detach().cpu().numpy().astype(np.float32)


def top_layer_from_npz(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "h_layers" in data.files:
        h = data["h_layers"].astype(np.float32)
        return h[:, 4, :] if h.shape[1] >= 5 else h[:, -1, :]
    return data["hidden_states"].astype(np.float32)


def episode_relative_xz(positions: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    """World-frame displacement from each episode start, using x/z only."""
    pos = positions[:, [0, 2]].astype(np.float32)
    rel = np.empty_like(pos)
    for eid in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == eid)
        rel[idx] = pos[idx] - pos[idx[0]]
    return rel


def fit_linear_readout(h: np.ndarray, y: np.ndarray, alpha: float):
    """Centered ridge readout h -> episode-relative x/z."""
    x_mean = h.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x = h - x_mean
    yc = y - y_mean
    xtx = x.T @ x
    xtx.flat[:: xtx.shape[0] + 1] += alpha
    w = np.linalg.solve(xtx, x.T @ yc).astype(np.float32)
    return {
        "x_mean": x_mean[0].astype(np.float32),
        "y_mean": y_mean[0].astype(np.float32),
        "w": w,
    }


def decode_readout(h: np.ndarray, decoder: dict) -> np.ndarray:
    return (h.astype(np.float32) - decoder["x_mean"]) @ decoder["w"] + decoder["y_mean"]


def fit_condition_readouts(excursion_dir: Path, max_per_condition: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    readouts, counts, r2 = {}, {}, {}
    for cond in CONDITIONS:
        data = np.load(excursion_dir / cond.excursion, allow_pickle=True)
        h = top_layer_from_npz(data)
        y = episode_relative_xz(data["positions"], data["episode_ids"])
        n = len(h)
        take = min(max_per_condition, n)
        idx = rng.choice(n, size=take, replace=False)
        dec = fit_linear_readout(h[idx], y[idx], alpha)
        pred = decode_readout(h[idx], dec)
        ss_res = float(np.sum((y[idx] - pred) ** 2))
        ss_tot = float(np.sum((y[idx] - y[idx].mean(axis=0, keepdims=True)) ** 2)) or 1.0
        readouts[cond.key] = dec
        counts[cond.key] = int(take)
        r2[cond.key] = float(1.0 - ss_res / ss_tot)
    return readouts, counts, r2


def fit_condition_manifolds(excursion_dir: Path, max_per_condition: int, cloud_points: int, seed: int):
    """Fit a condition-specific 2D PCA basis for top-layer h2 trajectories."""
    rng = np.random.default_rng(seed + 1009)
    manifolds = {}
    for cond in CONDITIONS:
        data = np.load(excursion_dir / cond.excursion, allow_pickle=True)
        h = top_layer_from_npz(data)
        rel_xz = episode_relative_xz(data["positions"], data["episode_ids"])
        n = len(h)
        take = min(max_per_condition, n)
        idx = rng.choice(n, size=take, replace=False)
        x = h[idx].astype(np.float32)
        pos_x = rel_xz[idx, 0].astype(np.float32)
        mean = x.mean(axis=0, keepdims=True)
        xc = x - mean
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        basis = vt[:2].T.astype(np.float32)
        var = np.square(s)
        explained = float(var[:2].sum() / max(var.sum(), 1e-12))
        cloud_take = min(cloud_points, take)
        cloud_idx = rng.choice(take, size=cloud_take, replace=False)
        cloud = ((x[cloud_idx] - mean) @ basis).astype(np.float32)
        manifolds[cond.key] = {
            "mean": mean[0].astype(np.float32),
            "basis": basis,
            "cloud": cloud,
            "cloud_rel_x": pos_x[cloud_idx].astype(np.float32),
            "explained": explained,
            "sample_count": int(take),
        }
    return manifolds


def project_manifold(h: np.ndarray, manifold: dict) -> np.ndarray:
    return (h.astype(np.float32) - manifold["mean"]) @ manifold["basis"]


def init_state(num_layers: int, hidden_size: int, device: torch.device):
    rnn = torch.zeros(1, num_layers, hidden_size, device=device)
    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
    not_done = torch.zeros(1, 1, dtype=torch.bool, device=device)
    return rnn, prev_action, not_done


def pin_episode(env: habitat.Env, episode_index: int):
    ep = env.episodes[episode_index]
    env._episode_iterator = iter([ep])
    env._episode_from_iter_on_reset = True
    env._episode_over = False
    return env.reset(), ep


def rgb_from_obs(obs: dict, size=(300, 190)) -> Image.Image:
    if "rgb" not in obs:
        img = Image.new("RGB", size, (10, 12, 14))
        d = ImageDraw.Draw(img)
        d.text((size[0] // 2 - 34, size[1] // 2 - 7), "no camera", fill=(190, 190, 190))
        return img
    arr = np.asarray(obs["rgb"])
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize(size, Image.Resampling.BILINEAR)


def coarse_display(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    small = img.resize((36, 24), Image.Resampling.BILINEAR)
    return small.resize(size, Image.Resampling.NEAREST)


def foveated_display(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.resize(size, Image.Resampling.BILINEAR)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=7))
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.5, h * 0.5
    radius = min(w, h) * 0.24
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((radius * 1.7 - dist) / max(radius * 0.7, 1.0), 0.0, 1.0)
    alpha = (alpha[..., None] * 255).astype(np.uint8)
    mask = Image.fromarray(alpha.squeeze(), mode="L").filter(ImageFilter.GaussianBlur(radius=8))
    out = Image.composite(img, blurred, mask)
    draw = ImageDraw.Draw(out, "RGBA")
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 255, 255, 150), width=2)
    return out


def logpolar_display(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(img)
    try:
        import cv2  # type: ignore

        center = (arr.shape[1] / 2.0, arr.shape[0] / 2.0)
        max_radius = min(center) * 0.98
        warped = cv2.warpPolar(
            arr,
            (size[0], size[1]),
            center,
            max_radius,
            cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS,
        )
        out = Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    except Exception:
        out = img.filter(ImageFilter.GaussianBlur(radius=2))
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = size
    for frac in (0.18, 0.32, 0.50, 0.72, 0.94):
        x = int(frac * w)
        draw.line((x, 0, x, h), fill=(255, 255, 255, 62), width=1)
    for frac in np.linspace(0.125, 0.875, 7):
        y = int(frac * h)
        draw.line((0, y, w, y), fill=(255, 255, 255, 42), width=1)
    draw.text((8, h - 18), "log radius x angle", fill=(255, 255, 255, 190))
    return out


def sensor_view_from_obs(cond_key: str, obs: dict, size=(300, 170)) -> Image.Image:
    """Make visual bandwidth visible in the video without changing rollout data."""
    if cond_key == "blind":
        img = Image.new("RGB", size, (4, 5, 6))
        d = ImageDraw.Draw(img)
        d.text((size[0] // 2 - 26, size[1] // 2 - 7), "blind", fill=(220, 220, 220))
        return img
    base = rgb_from_obs(obs, size=size)
    if cond_key == "coarse":
        return coarse_display(base, size)
    if cond_key == "foveated":
        return foveated_display(base, size)
    if cond_key == "logpolar":
        return logpolar_display(base, size)
    return base


def topdown_from_metrics(env: habitat.Env):
    try:
        from habitat.utils.visualizations import maps

        metrics = env.get_metrics()
        metrics_map = metrics.get("top_down_map", {}).get("map")
        if metrics_map is not None:
            top_down = metrics_map
        else:
            top_down = maps.get_topdown_map_from_sim(env.sim, map_resolution=640)
        colorized = maps.colorize_topdown_map(top_down)
        return Image.fromarray(colorized).convert("RGB"), tuple(top_down.shape[:2])
    except Exception as exc:
        print("warning: could not render Habitat topdown map:", exc)
        return None, None


def map_coord_from_metrics(metrics: dict) -> np.ndarray | None:
    coords = metrics.get("top_down_map", {}).get("agent_map_coord")
    if not coords:
        return None
    coord = coords[-1] if isinstance(coords, list) else coords
    return np.asarray(coord[:2], dtype=np.float32)


def world_to_topdown_coord(env: habitat.Env, position: np.ndarray, map_shape) -> np.ndarray | None:
    if map_shape is None:
        return None
    try:
        from habitat.utils.visualizations import maps

        row, col = maps.to_grid(
            float(position[2]),
            float(position[0]),
            map_shape,
            pathfinder=env.sim.pathfinder,
        )
        return np.asarray([row, col], dtype=np.float32)
    except Exception:
        return None


def run_condition(cond: Condition, episode_index: int, max_steps: int,
                  dataset_split: str, data_path: str | None, device: torch.device,
                  record_frames: bool = True, render_topdown: bool = True):
    ckpt = Path(cond.ckpt)
    if cond.key == "blind" and not ckpt.exists():
        ckpt = Path("/scratch/wxu/habitat_checkpoints_rcp/blind_seed_2_friend/ckpt.49.pth")
    overrides = [
        f"habitat.dataset.split={dataset_split}",
        "habitat.environment.iterator_options.shuffle=False",
        "habitat.environment.iterator_options.group_by_scene=False",
        "habitat.environment.max_episode_steps=2000",
    ]
    if data_path:
        overrides.insert(0, f"habitat.dataset.data_path={data_path}")
    config = load_habitat_config(cond.config, str(ckpt), overrides=overrides)
    env = habitat.Env(config=config.habitat)
    obs, ep = pin_episode(env, episode_index)
    topdown_img, topdown_shape = topdown_from_metrics(env) if render_topdown else (None, None)
    goal = np.asarray(ep.goals[0].position, dtype=np.float32)
    goal_map_coord = world_to_topdown_coord(env, goal, topdown_shape)
    policy, hidden_size, num_layers, _ = load_policy(config, env, str(ckpt), device)
    policy.eval()

    rnn, prev_action, not_done = init_state(num_layers, hidden_size, device)
    frames, positions, map_coords, headings, h2, actions = [], [], [], [], [], []
    ended = False
    metrics = {}

    for _ in range(max_steps):
        step_metrics = env.get_metrics()
        state = env.sim.get_agent_state()
        positions.append(np.asarray(state.position, dtype=np.float32))
        coord = map_coord_from_metrics(step_metrics)
        if coord is None:
            coord = world_to_topdown_coord(env, positions[-1], topdown_shape)
        if coord is not None:
            map_coords.append(coord)
        headings.append(float(heading_from_quaternion(state.rotation)))
        if record_frames:
            frames.append(sensor_view_from_obs(cond.key, obs))

        batch = {k: torch.from_numpy(np.expand_dims(v, 0)).to(device) for k, v in obs.items()}
        with torch.no_grad():
            action_data = policy.act(batch, rnn, prev_action, not_done, deterministic=True)
        rnn = action_data.rnn_hidden_states
        h2.append(top_h2(rnn))
        prev_action = action_data.actions
        not_done = torch.ones(1, 1, dtype=torch.bool, device=device)
        action_int = int(action_data.env_actions[0].item())
        actions.append(action_int)

        obs = env.step(action_int)
        metrics = env.get_metrics()
        if env.episode_over or action_int == 0:
            ended = True
            break

    if not metrics:
        metrics = env.get_metrics()
    env.close()
    success = float(metrics.get("success", 0.0)) >= 0.5
    return {
        "key": cond.key,
        "label": cond.label,
        "color": cond.color,
        "scene": os.path.basename(str(ep.scene_id)).replace(".glb", ""),
        "episode_index": int(episode_index),
        "ended": bool(ended),
        "success": bool(success),
        "spl": float(metrics.get("spl", 0.0)),
        "distance_to_goal": float(metrics.get("distance_to_goal", np.nan)),
        "frames": frames,
        "positions": np.stack(positions, axis=0),
        "map_coords": np.stack(map_coords, axis=0) if map_coords else np.zeros((0, 2), dtype=np.float32),
        "headings": np.asarray(headings, dtype=np.float32),
        "h2": np.stack(h2, axis=0),
        "actions": np.asarray(actions, dtype=np.int32),
        "start": np.asarray(ep.start_position, dtype=np.float32),
        "goal": goal,
        "topdown": topdown_img,
        "topdown_shape": topdown_shape,
        "goal_map_coord": goal_map_coord,
    }


def find_shared_success_episode(args, device: torch.device) -> int:
    """Search deterministic MP3D-test episodes until all five agents succeed."""
    last_status = None
    for ep_idx in range(args.search_start, args.search_start + args.search_episodes):
        status = []
        all_ok = True
        print(f"searching episode {ep_idx}")
        for cond in CONDITIONS:
            run = run_condition(
                cond,
                ep_idx,
                args.success_max_steps,
                args.dataset_split,
                args.data_path,
                device,
                record_frames=False,
                render_topdown=False,
            )
            status.append(
                f"{cond.key}:success={int(run['success'])},steps={len(run['actions'])},"
                f"spl={run['spl']:.2f},dist={run['distance_to_goal']:.2f}"
            )
            if not run["success"]:
                all_ok = False
                break
        last_status = "; ".join(status)
        print("  " + last_status)
        if all_ok:
            print(f"found shared-success episode: {ep_idx}")
            return ep_idx
    raise RuntimeError(
        "Could not find an episode where all five agents succeeded. "
        f"Last status: {last_status}"
    )


def bounds_from(arrs: list[np.ndarray], dims=(0, 1), pad: float = 0.08):
    pts = np.concatenate([a[:, dims] for a in arrs], axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    return lo - span * pad, hi + span * pad


def xy_to_box(x: float, y: float, lo: np.ndarray, hi: np.ndarray, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    u = (x - lo[0]) / max(float(hi[0] - lo[0]), 1e-6)
    v = (y - lo[1]) / max(float(hi[1] - lo[1]), 1e-6)
    return int(x0 + u * (x1 - x0)), int(y1 - v * (y1 - y0))


def draw_panel(draw: ImageDraw.ImageDraw, box, title, fill, outline=(218, 209, 195)):
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=1)
    draw.text((box[0] + 10, box[1] + 7), title, fill=(16, 35, 58))


def coord_to_xy(coord: np.ndarray | list | tuple) -> tuple[float, float]:
    # Habitat top_down_map coordinates are row/col; PIL drawing is x/y.
    row, col = float(coord[0]), float(coord[1])
    return col, row


def topdown_crop_by_coords(img: Image.Image | None, coords: np.ndarray, pad: int = 54):
    if img is None or coords.size == 0:
        return None
    w, h = img.size
    xy = np.asarray([coord_to_xy(c) for c in coords], dtype=np.float32)
    left, top = xy.min(axis=0)
    right, bottom = xy.max(axis=0)
    left = int(max(0, np.floor(left - pad)))
    right = int(min(w, np.ceil(right + pad)))
    top = int(max(0, np.floor(top - pad)))
    bottom = int(min(h, np.ceil(bottom + pad)))
    if right - left < 16 or bottom - top < 16:
        return None
    crop_box = (left, top, right, bottom)
    return img.crop(crop_box), crop_box


def map_coord_to_box(coord: np.ndarray, crop_box: tuple[int, int, int, int], box: tuple[int, int, int, int]):
    x, y = coord_to_xy(coord)
    left, top, right, bottom = crop_box
    x0, y0, x1, y1 = box
    u = (x - left) / max(float(right - left), 1e-6)
    v = (y - top) / max(float(bottom - top), 1e-6)
    return int(x0 + u * (x1 - x0)), int(y0 + v * (y1 - y0))


def draw_trace(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], fill, width: int):
    if len(pts) > 1:
        draw.line(pts, fill=fill, width=width, joint="curve")


def viridis_like(value: float, alpha: int = 210) -> tuple[int, int, int, int]:
    anchors = np.asarray(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    v = float(np.clip(value, 0.0, 1.0)) * (len(anchors) - 1)
    i = int(np.floor(v))
    j = min(i + 1, len(anchors) - 1)
    t = v - i
    rgb = (anchors[i] * (1.0 - t) + anchors[j] * t).astype(int)
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha


def value_to_color(value: float, lo: float, hi: float, alpha: int = 210):
    return viridis_like((value - lo) / max(hi - lo, 1e-6), alpha=alpha)


def render_video(
    runs: list[dict],
    manifold_paths: dict[str, np.ndarray],
    manifolds: dict[str, dict],
    out_mp4: Path,
    fps: int,
    meta: dict,
):
    import imageio.v2 as imageio

    width, height = 1800, 1000
    col_w = width // 5
    paper = (244, 239, 229)
    ink = (16, 35, 58)
    muted = (105, 114, 125)
    font = ImageFont.load_default()

    manifold_bounds = {}
    for run in runs:
        key = run["key"]
        manifold_bounds[key] = bounds_from(
            [manifolds[key]["cloud"], manifold_paths[key]], dims=(0, 1), pad=0.14,
        )
    map_img, map_bounds = None, None
    for run in runs:
        if run.get("topdown") is not None:
            map_img = run["topdown"]
            break
    coord_arrays = [r["map_coords"] for r in runs if len(r["map_coords"]) > 0]
    goal_coords = [r["goal_map_coord"][None, :] for r in runs if r.get("goal_map_coord") is not None]
    route_crop = topdown_crop_by_coords(map_img, np.concatenate(coord_arrays + goal_coords, axis=0)) if coord_arrays else None
    rel_x_values = [m["cloud_rel_x"] for m in manifolds.values()]
    for run in runs:
        rel_x_values.append((run["positions"][:, 0] - run["start"][0]).astype(np.float32))
    rel_x_all = np.concatenate(rel_x_values, axis=0)
    color_lo, color_hi = np.percentile(rel_x_all, [2, 98]).astype(float)
    if color_hi - color_lo < 1e-3:
        color_lo -= 1.0
        color_hi += 1.0
    total_frames = max(len(r["frames"]) for r in runs)

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    try:
        for t in range(total_frames):
            canvas = Image.new("RGB", (width, height), paper)
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.text((26, 16), "Real Habitat rollout + LSTM h2 memory manifolds", fill=ink, font=font)
            draw.text((26, 36), "One held-out MP3D episode where all five policies succeed; h2 points are colored by true relative x position",
                      fill=muted, font=font)
            draw.text((width - 124, 22), f"t = {t:03d}", fill=ink, font=font)

            for ci, run in enumerate(runs):
                x0 = ci * col_w + 10
                x1 = (ci + 1) * col_w - 10
                color = run["color"]
                draw.text((x0 + 2, 66), run["label"], fill=color, font=font)
                status = "success" if run["success"] else "not reached"
                draw.text((x1 - 88, 66), status, fill=(63, 91, 74) if run["success"] else (150, 62, 50), font=font)
                draw.line((x0, 84, x1, 84), fill=color + (255,), width=4)

                idx = min(t, len(run["frames"]) - 1)
                rgb_box = (x0, 98, x1, 268)
                draw_panel(draw, rgb_box, "policy input view", (251, 247, 238))
                rgb = run["frames"][idx].resize((x1 - x0 - 16, 134), Image.Resampling.BILINEAR)
                canvas.paste(rgb, (x0 + 8, 126))

                traj_box = (x0, 284, x1, 526)
                draw_panel(draw, traj_box, "bird-view trajectory", (251, 247, 238))
                plot_box = (x0 + 18, 316, x1 - 18, 508)
                if route_crop is not None and len(run["map_coords"]) > 0:
                    route_map, crop_box = route_crop
                    bg = route_map.resize((plot_box[2] - plot_box[0], plot_box[3] - plot_box[1]), Image.Resampling.BILINEAR)
                    bg = bg.convert("RGBA")
                    bg.putalpha(148)
                    canvas.paste(bg, (plot_box[0], plot_box[1]), bg)
                    draw = ImageDraw.Draw(canvas, "RGBA")
                    draw.rectangle(plot_box, outline=(16, 35, 58, 42), width=1)
                else:
                    for gx in np.linspace(plot_box[0], plot_box[2], 5):
                        draw.line((gx, plot_box[1], gx, plot_box[3]), fill=(16, 35, 58, 32), width=1)
                    for gy in np.linspace(plot_box[1], plot_box[3], 4):
                        draw.line((plot_box[0], gy, plot_box[2], gy), fill=(16, 35, 58, 32), width=1)
                    crop_box = None
                if crop_box is not None and len(run["map_coords"]) > 0:
                    pts = [map_coord_to_box(c, crop_box, plot_box) for c in run["map_coords"]]
                else:
                    pos_lo, pos_hi = bounds_from([r["positions"][:, [0, 2]] for r in runs], dims=(0, 1), pad=0.18)
                    pts = [xy_to_box(float(p[0]), float(p[2]), pos_lo, pos_hi, plot_box) for p in run["positions"]]
                if len(pts) > 1:
                    draw_trace(draw, pts, (16, 35, 58, 72), 2)
                    draw_trace(draw, pts[:idx + 1], color + (238,), 4)
                if crop_box is not None and len(run["map_coords"]) > 0:
                    sx, sy = map_coord_to_box(run["map_coords"][0], crop_box, plot_box)
                    goal_coord = run.get("goal_map_coord")
                    gx, gy = map_coord_to_box(goal_coord, crop_box, plot_box) if goal_coord is not None else pts[-1]
                else:
                    sx, sy = pts[0]
                    gx, gy = pts[-1]
                draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=(16, 35, 58, 170))
                draw.line((gx - 7, gy, gx + 7, gy), fill=(198, 77, 62, 230), width=2)
                draw.line((gx, gy - 7, gx, gy + 7), fill=(198, 77, 62, 230), width=2)
                cx, cy = pts[idx]
                draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color + (255,), outline=(255, 255, 255, 255))

                man_box = (x0, 542, x1, 970)
                ev = manifolds[run["key"]]["explained"]
                draw_panel(draw, man_box, f"h2 manifold, color = relative x ({ev:.0%})", (251, 247, 238))
                qbox = (x0 + 24, 586, x1 - 38, 940)
                lo, hi = manifold_bounds[run["key"]]
                draw.line((qbox[0], qbox[3], qbox[2], qbox[3]), fill=(16, 35, 58, 95), width=1)
                draw.line((qbox[0], qbox[1], qbox[0], qbox[3]), fill=(16, 35, 58, 95), width=1)
                for pt, rel_x in zip(manifolds[run["key"]]["cloud"], manifolds[run["key"]]["cloud_rel_x"]):
                    px, py = xy_to_box(float(pt[0]), float(pt[1]), lo, hi, qbox)
                    draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=value_to_color(float(rel_x), color_lo, color_hi, alpha=72))
                qpts = [xy_to_box(float(p[0]), float(p[1]), lo, hi, qbox) for p in manifold_paths[run["key"]]]
                rel_path_x = (run["positions"][:, 0] - run["start"][0]).astype(np.float32)
                trail_start = max(0, idx - 120)
                for pi in range(trail_start, idx + 1):
                    qx0, qy0 = qpts[pi]
                    fill = value_to_color(float(rel_path_x[pi]), color_lo, color_hi, alpha=235)
                    draw.ellipse((qx0 - 2, qy0 - 2, qx0 + 2, qy0 + 2), fill=fill)
                qx, qy = qpts[idx]
                draw.ellipse((qx - 7, qy - 7, qx + 7, qy + 7), fill=color + (245,), outline=(255, 255, 255, 255), width=2)
                draw.text((qbox[0] + 2, qbox[3] + 8), "PC1", fill=ink, font=font)
                draw.text((qbox[0] - 16, qbox[1] + 2), "PC2", fill=ink, font=font)
                cbar_x = x1 - 28
                cbar_y0, cbar_y1 = qbox[1], qbox[3]
                for yy in range(cbar_y0, cbar_y1):
                    frac = 1.0 - (yy - cbar_y0) / max(cbar_y1 - cbar_y0, 1)
                    draw.line((cbar_x, yy, cbar_x + 5, yy), fill=viridis_like(frac, alpha=220), width=1)
                draw.text((cbar_x - 4, cbar_y0 - 13), "x", fill=ink, font=font)

            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()

    meta.update({
        "video": str(out_mp4),
        "fps": fps,
        "frames": int(total_frames),
        "conditions": [
            {
                "key": r["key"],
                "label": r["label"],
                "scene": r["scene"],
                "episode_index": r["episode_index"],
                "steps": int(len(r["actions"])),
                "ended": bool(r["ended"]),
                "success": bool(r["success"]),
                "spl": float(r["spl"]),
                "distance_to_goal": float(r["distance_to_goal"]),
            }
            for r in runs
        ],
    })
    out_mp4.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--auto-find-success", action="store_true")
    ap.add_argument("--search-start", type=int, default=0)
    ap.add_argument("--search-episodes", type=int, default=60)
    ap.add_argument("--success-max-steps", type=int, default=240)
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--data-path", default="data/datasets/pointnav/mp3d/v1/test/test.json.gz")
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--readout-samples-per-condition", type=int, default=8000)
    ap.add_argument("--manifold-samples-per-condition", type=int, default=12000)
    ap.add_argument("--manifold-cloud-points", type=int, default=1000)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--excursion-dir", type=Path, default=Path("/scratch/wxu/habitat_checkpoints_rcp/excursion_results"))
    ap.add_argument("--out", type=Path, default=Path("/scratch/wxu/habitat_checkpoints_rcp/website_media/real_navigation_memory.mp4"))
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if args.auto_find_success:
        args.episode_index = find_shared_success_episode(args, device)
        args.max_steps = max(args.max_steps, args.success_max_steps)

    print("fitting h2 manifold PCA bases from RCP excursion rollouts")
    manifolds = fit_condition_manifolds(
        args.excursion_dir, args.manifold_samples_per_condition, args.manifold_cloud_points, args.seed,
    )
    runs = []
    for cond in CONDITIONS:
        print(f"capturing {cond.key}")
        runs.append(run_condition(
            cond, args.episode_index, args.max_steps, args.dataset_split, args.data_path, device,
        ))

    manifold_paths = {}
    for r in runs:
        manifold_paths[r["key"]] = project_manifold(r["h2"], manifolds[r["key"]])
    meta = {
        "source": "RCP /scratch/wxu/habitat_checkpoints_rcp excursion rollouts and checkpoints",
        "manifold_sample_counts": {k: int(v["sample_count"]) for k, v in manifolds.items()},
        "manifold_pca_explained_variance_2d": {k: float(v["explained"]) for k, v in manifolds.items()},
        "dataset_split": args.dataset_split,
        "data_path": args.data_path,
        "episode_index": int(args.episode_index),
        "h2_definition": "top-layer LSTM hidden state index 4, layout [h0,c0,h1,c1,h2,c2]",
        "memory_visualization": "condition-specific 2D PCA basis fit on h2 states from RCP excursion rollouts; dots are colored by true episode-relative x position and the colored marker shows the displayed episode trajectory",
        "topdown_visualization": "Habitat top_down_map with trajectory drawn in Habitat's own map-coordinate frame from top_down_map.agent_map_coord, avoiding world-to-pixel overlay mismatch",
        "policy_input_visualization": "display-friendly rendering of each sensor's visual bandwidth; rollout actions are generated from the actual Habitat observations, not from these rendered panels",
    }
    render_video(runs, manifold_paths, manifolds, args.out, args.fps, meta)
    print("wrote", args.out)
    print("wrote", args.out.with_suffix(".json"))


if __name__ == "__main__":
    main()
