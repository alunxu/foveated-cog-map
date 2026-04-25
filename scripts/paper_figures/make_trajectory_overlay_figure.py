"""
Trajectory overlay figure for Fig 1 setup.

For one Gibson val episode (chosen as a representative case where all 5
conditions reach the goal with SPL near each condition's per-condition
median), overlay the (x, z) world-frame trajectories of all 5 agents.
The visual contrast in trajectory shape, length, and turning behaviour
gives the reader an immediate intuition for how the input ablation
shapes navigation strategy.

Selection criterion: among 500 Gibson val episodes (deterministic
rollouts, same val split per condition), find episodes where all five
conditions succeeded and each condition's SPL is within ±0.1 of its
per-condition median SPL.  Among those, the longest-geodesic episode
gives the most interesting trajectory comparison.

Reads:  /tmp/traj/traj_<cond>.npz (positions, episode_ids, etc.)
Writes: <out-dir>/trajectory_overlay.{pdf,png}

Usage:
    python scripts/paper_figures/make_trajectory_overlay_figure.py \\
        --traj-dir /tmp/traj \\
        --out-dir docs/NeurIPS_2026/fig \\
        --episode-id 414 --scene-id 92
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = [
    # (npz_key,           label,            colour,    linestyle, marker)
    ("blind",             "Blind",          "#444444", "-",       "o"),
    ("matched",           "Matched (1×1)",  "#377eb8", "-",       "s"),
    ("uniform",           "Uniform",        "#4daf4a", "-",       "^"),
    ("foveated",          "Foveated (fix)", "#e41a1c", "-",       "D"),
    ("foveated_learned",  "Fov-learned",    "#ff7f00", "-",       "v"),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--episode-id", type=int, required=True)
    ap.add_argument("--scene-id", type=int, default=None,
                    help="Optional sanity-check on scene")
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

    # Figure setup: one wide panel
    fig, ax = plt.subplots(figsize=(6.5, 4.6))

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

    # Plot each condition's trajectory.  We invert z because Habitat z
    # axis points "into the screen" while plt y axis points up; doesn't
    # affect topology, just aesthetics.
    for cond_key, label, colour, ls, marker in CONDS:
        if cond_key not in trajs:
            continue
        p = trajs[cond_key]["positions"]
        x = p[:, 0]
        z = p[:, 2]
        ax.plot(x, z, color=colour, lw=1.6, alpha=0.85,
                label=f"{label} ({trajs[cond_key]['n_steps']} steps)",
                zorder=2)
        # Sparse markers along the path (every ~25 steps)
        n = len(x)
        if n > 25:
            idx_sparse = np.linspace(0, n - 1, 6, dtype=int)
            ax.scatter(x[idx_sparse], z[idx_sparse], s=18, c=colour,
                       edgecolor="white", linewidths=0.5, zorder=3,
                       marker=marker)

    # Start and goal markers (large, on top) — annotated directly
    ax.scatter(start[0], start[1], s=200, c="white", edgecolor="black",
               linewidths=1.5, marker="o", zorder=5)
    ax.text(start[0], start[1], "S", ha="center", va="center",
            fontsize=10, fontweight="bold", zorder=6)
    ax.scatter(goal[0], goal[1], s=240, c="gold", edgecolor="black",
               linewidths=1.5, marker="*", zorder=5)
    ax.annotate("Goal", (goal[0], goal[1]), xytext=(goal[0] + 0.35, goal[1] + 0.2),
                fontsize=9, fontweight="bold", zorder=6)
    ax.annotate("Start", (start[0], start[1]),
                xytext=(start[0] - 0.5, start[1] + 0.4),
                fontsize=9, fontweight="bold", ha="right", zorder=6)

    # Aesthetics
    ax.set_xlabel("x (world frame, m)", fontsize=9.5)
    ax.set_ylabel("z (world frame, m)", fontsize=9.5)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(linestyle=":", alpha=0.3)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    # Legend outside on the right to avoid trajectory overlap
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)

    title = (f"Same Gibson val episode (scene {trajs[next(iter(trajs))]['scene_id']}, "
             f"ep {args.episode_id})")
    ax.set_title(title, fontsize=9.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out_dir / f"trajectory_overlay.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
