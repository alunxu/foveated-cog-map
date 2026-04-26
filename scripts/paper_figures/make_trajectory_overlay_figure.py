"""
Trajectory overlay figure for Fig 1 setup.

For one Gibson val episode (chosen as a representative case where all 5
conditions reach the goal with SPL near each condition's per-condition
median), overlay the (x, z) world-frame trajectories of all 5 agents.
Optionally lays the trajectories on top of the Habitat top-down
occupancy map for the scene (requires render_scene_topdown.py to have
been run first), giving a floor-plan-aware visualisation comparable
to the previous fig_topdown.png.

Selection criterion: among 500 Gibson val episodes (deterministic
rollouts, same val split per condition), find episodes where all five
conditions succeeded and each condition's SPL is within ±0.1 of its
per-condition median SPL.  Among those, the longest-geodesic episode
gives the most interesting trajectory comparison.

Reads:
  /tmp/traj/traj_<cond>.npz                    (positions, episode_ids, ...)
  --topdown-png + --topdown-json (optional)    floor-plan background

Writes: <out-dir>/trajectory_overlay.{pdf,png}

Usage:
    python scripts/paper_figures/make_trajectory_overlay_figure.py \\
        --traj-dir /tmp/traj \\
        --out-dir docs/NeurIPS_2026/fig \\
        --episode-id 414 --scene-id 92 \\
        --topdown-png /tmp/scene92_topdown.png \\
        --topdown-json /tmp/scene92_topdown.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = [
    # (npz_key,           label,            colour,    linestyle, marker)
    ("blind",             "Blind",          "#444444", "-",       "o"),
    ("matched",           "Coarse (1×1)",  "#377eb8", "-",       "s"),
    ("uniform",           "Uniform",        "#4daf4a", "-",       "^"),
    ("foveated",          "Foveated (fix)", "#e41a1c", "-",       "D"),
    ("foveated_learned",  "Foveated (learned)",    "#ff7f00", "-",       "v"),
]


def get_episode_traj(npz_path: Path, episode_id: int) -> dict | None:
    d = np.load(npz_path)
    eps = d["episode_ids"]
    mask = eps == episode_id
    if mask.sum() == 0:
        return None
    idx = np.where(mask)[0]
    idx = idx[np.argsort(d["step_in_episode"][idx])]
    return {
        "positions": d["positions"][idx],          # (T, 3) — x, y, z
        "goal_positions": d["goal_positions"][idx],  # (T, 3)
        "distance_to_goal": d["distance_to_goal"][idx],
        "scene_id": int(d["scene_ids"][idx[0]]),
        "n_steps": int(mask.sum()),
    }


def world_to_pixel(positions_xz: np.ndarray, sidecar: dict) -> np.ndarray:
    """Project (x, z) world-frame meters → (col, row) pixel coords.

    Returns array of shape (T, 2) where col 0 is x-pixel, col 1 is
    y-pixel (row in image, with origin at top-left).

    Habitat conventions: image col index increases with world x; image
    row index increases with world z (camera looks "down").
    """
    lo = np.asarray(sidecar["world_lower_bound"], dtype=np.float32)
    hi = np.asarray(sidecar["world_upper_bound"], dtype=np.float32)
    # Use x and z components (positions index 0 and 2 in original 3-d).
    # sidecar bounds are full 3-d; we want lo[0],hi[0] for x; lo[2],hi[2] for z.
    H, W = sidecar["topdown_height"], sidecar["topdown_width"]
    x_norm = (positions_xz[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-6)
    z_norm = (positions_xz[:, 1] - lo[2]) / max(hi[2] - lo[2], 1e-6)
    cols = x_norm * (W - 1)
    rows = z_norm * (H - 1)
    return np.column_stack([cols, rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--episode-id", type=int, required=True)
    ap.add_argument("--scene-id", type=int, default=None,
                    help="Optional sanity-check on scene")
    ap.add_argument("--topdown-png", type=Path, default=None,
                    help="If given, overlay trajectories on this floor-plan.")
    ap.add_argument("--topdown-json", type=Path, default=None,
                    help="Sidecar JSON with world-frame bounds for the topdown PNG.")
    ap.add_argument("--crop-margin", type=float, default=2.0,
                    help="Meters of margin around the trajectory bounding box "
                         "when cropping the topdown background. Set <=0 to skip cropping.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    trajs: dict[str, dict] = {}
    for cond_key, _, _, _, _ in CONDS:
        p = args.traj_dir / f"traj_{cond_key}.npz"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        ed = get_episode_traj(p, args.episode_id)
        if ed is None:
            print(f"[skip] episode {args.episode_id} not in {cond_key}")
            continue
        if args.scene_id is not None and ed["scene_id"] != args.scene_id:
            print(f"[skip] {cond_key} ep {args.episode_id} is in scene "
                  f"{ed['scene_id']}, expected {args.scene_id}")
            continue
        trajs[cond_key] = ed

    if not trajs:
        raise RuntimeError("no usable trajectories")

    # Use start position from any condition (they share start by val
    # protocol; we verify they match).
    starts = [trajs[c]["positions"][0, [0, 2]] for c in trajs]
    goals  = [trajs[c]["goal_positions"][0, [0, 2]] for c in trajs]
    start = starts[0]
    goal = goals[0]
    if not all(np.allclose(s, start, atol=0.1) for s in starts):
        print("WARNING: start positions don't match across conditions")
    if not all(np.allclose(g, goal, atol=0.1) for g in goals):
        print("WARNING: goal positions don't match across conditions")

    # ─── Determine plotting mode: topdown background or world-frame ──
    use_topdown = args.topdown_png is not None and args.topdown_json is not None
    if use_topdown:
        from PIL import Image
        sidecar = json.loads(args.topdown_json.read_text())
        bg = np.array(Image.open(args.topdown_png))
        H, W = sidecar["topdown_height"], sidecar["topdown_width"]
        # Project trajectories + start + goal to pixel coords
        traj_px = {}
        for cond_key, *_ in CONDS:
            if cond_key not in trajs:
                continue
            p = trajs[cond_key]["positions"][:, [0, 2]]
            traj_px[cond_key] = world_to_pixel(p, sidecar)
        start_px = world_to_pixel(start.reshape(1, 2), sidecar)[0]
        goal_px = world_to_pixel(goal.reshape(1, 2), sidecar)[0]

        # Crop background to a tighter view around the trajectories
        # (otherwise the full Gibson scene is mostly empty).
        if args.crop_margin > 0:
            all_pts = np.concatenate(
                [traj_px[c] for c in traj_px] +
                [start_px.reshape(1, 2), goal_px.reshape(1, 2)],
                axis=0,
            )
            # Convert margin in meters → pixels
            lo = np.asarray(sidecar["world_lower_bound"], dtype=np.float32)
            hi = np.asarray(sidecar["world_upper_bound"], dtype=np.float32)
            px_per_m_x = W / max(hi[0] - lo[0], 1e-6)
            px_per_m_z = H / max(hi[2] - lo[2], 1e-6)
            margin_px = max(args.crop_margin * px_per_m_x,
                            args.crop_margin * px_per_m_z)
            col_min = max(0, int(all_pts[:, 0].min() - margin_px))
            col_max = min(W, int(all_pts[:, 0].max() + margin_px))
            row_min = max(0, int(all_pts[:, 1].min() - margin_px))
            row_max = min(H, int(all_pts[:, 1].max() + margin_px))
            bg = bg[row_min:row_max, col_min:col_max]
            # Adjust trajectory pixel coords to match cropped origin
            for c in traj_px:
                traj_px[c] = traj_px[c] - np.array([col_min, row_min])
            start_px = start_px - np.array([col_min, row_min])
            goal_px = goal_px - np.array([col_min, row_min])

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.imshow(bg, origin="upper")
        for cond_key, label, colour, ls, marker in CONDS:
            if cond_key not in traj_px:
                continue
            p = traj_px[cond_key]
            ax.plot(p[:, 0], p[:, 1], color=colour, lw=1.7, alpha=0.92,
                    label=f"{label} ({trajs[cond_key]['n_steps']} steps)",
                    zorder=3)
            # Sparse markers along the path
            n = len(p)
            if n > 25:
                idx_sparse = np.linspace(0, n - 1, 6, dtype=int)
                ax.scatter(p[idx_sparse, 0], p[idx_sparse, 1], s=22,
                           c=colour, edgecolor="white", linewidths=0.6,
                           zorder=4, marker=marker)
        # Start / goal markers
        ax.scatter(*start_px, s=220, c="white", edgecolor="black",
                   linewidths=1.6, marker="o", zorder=6)
        ax.text(start_px[0], start_px[1], "S", ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=7)
        ax.scatter(*goal_px, s=260, c="gold", edgecolor="black",
                   linewidths=1.6, marker="*", zorder=6)
        ax.annotate("Goal", goal_px, xytext=(goal_px[0] + 12, goal_px[1] - 5),
                    fontsize=9, fontweight="bold", zorder=7,
                    color="black")
        ax.annotate("Start", start_px,
                    xytext=(start_px[0] - 10, start_px[1] - 12),
                    fontsize=9, fontweight="bold", ha="right", zorder=7,
                    color="black")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right", "bottom", "left"):
            ax.spines[s_].set_visible(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=8, frameon=False)
        scene_label = sidecar.get("scene_id", "?")
        if isinstance(scene_label, str) and "/" in scene_label:
            scene_label = Path(scene_label).stem
        ax.set_title(
            f"Same Gibson val episode (scene {scene_label}, ep {args.episode_id})",
            fontsize=9.5,
        )
    else:
        # No-background mode (original 2D scatter)
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        for cond_key, label, colour, ls, marker in CONDS:
            if cond_key not in trajs:
                continue
            p = trajs[cond_key]["positions"]
            ax.plot(p[:, 0], p[:, 2], color=colour, lw=1.6, alpha=0.85,
                    label=f"{label} ({trajs[cond_key]['n_steps']} steps)",
                    zorder=2)
            n = len(p)
            if n > 25:
                idx_sparse = np.linspace(0, n - 1, 6, dtype=int)
                ax.scatter(p[idx_sparse, 0], p[idx_sparse, 2], s=18, c=colour,
                           edgecolor="white", linewidths=0.5, zorder=3,
                           marker=marker)
        ax.scatter(start[0], start[1], s=200, c="white", edgecolor="black",
                   linewidths=1.5, marker="o", zorder=5)
        ax.text(start[0], start[1], "S", ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=6)
        ax.scatter(goal[0], goal[1], s=240, c="gold", edgecolor="black",
                   linewidths=1.5, marker="*", zorder=5)
        ax.annotate("Goal", (goal[0], goal[1]),
                    xytext=(goal[0] + 0.35, goal[1] + 0.2),
                    fontsize=9, fontweight="bold", zorder=6)
        ax.annotate("Start", (start[0], start[1]),
                    xytext=(start[0] - 0.5, start[1] + 0.4),
                    fontsize=9, fontweight="bold", ha="right", zorder=6)
        ax.set_xlabel("x (world frame, m)", fontsize=9.5)
        ax.set_ylabel("z (world frame, m)", fontsize=9.5)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=8.5)
        ax.grid(linestyle=":", alpha=0.3)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, frameon=False)
        ax.set_title(
            f"Same Gibson val episode (scene {trajs[next(iter(trajs))]['scene_id']}, "
            f"ep {args.episode_id})",
            fontsize=9.5,
        )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out_dir / f"trajectory_overlay.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
