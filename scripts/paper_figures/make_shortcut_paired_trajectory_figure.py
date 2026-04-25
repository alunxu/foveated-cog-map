"""
Phase B paired-trajectory figure: visualises the "memory locks onto
old goal" failure mode of persistent-memory shortcut runs.

For one carefully-chosen (scene, ep_idx) per condition, plot:
  - episode N-1 (ep_idx - 1) goal location (old goal — where the
    LSTM state was last directed before this test episode)
  - episode N start position (shared by reset + persistent)
  - episode N goal position (the NEW goal both agents should reach)
  - reset-memory trajectory (LSTM zeroed; reaches the new goal)
  - persistent-memory trajectory (LSTM carries from N-1; fails)

The dramatic contrast: reset agent solves the new task efficiently;
persistent agent wanders, often toward the OLD goal direction,
because its LSTM state is still encoding the previous task context.

Reads:  /tmp/shortcut_traj_local/<cond>_gibson_traj.npz
Writes: <out-dir>/shortcut_paired_traj.{pdf,png}

Usage:
    python scripts/paper_figures/make_shortcut_paired_trajectory_figure.py \\
        --traj-dir /tmp/shortcut_traj_local \\
        --out-dir docs/NeurIPS_2026/fig
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Hand-selected (cond, scene, ep_idx) tuples chosen for dramatic contrast:
# reset succeeds (SPL > 0.5, steps < 100) while persistent fails
# (SPL < 0.2, steps > 30 so the trajectory has substance to plot).
PANELS = [
    # (cond_key, label, scene, ep_idx, colour)
    ("blind",            "Blind",          "Adairsville", 6, "#444444"),
    ("uniform",          "Uniform",        "Allensville", 9, "#4daf4a"),
    ("foveated",         "Foveated (fix)", "7y3sRwLe3Va", 8, "#e41a1c"),
]


def load_episode_pair(traj_dir: Path, cond: str, scene: str, ep_idx: int) -> dict:
    """Pull both reset and persistent trajectories for (scene, ep_idx),
    plus ep_idx - 1's goal (old goal that LSTM persists toward)."""
    d = np.load(traj_dir / f"{cond}_gibson_traj.npz", allow_pickle=True)

    out: dict = {}
    mask = (d["scenes"] == scene) & (d["ep_idx"] == ep_idx)
    for i in np.where(mask)[0]:
        c = str(d["conditions"][i])
        out[c] = {
            "positions": d["positions"][i],
            "start": d["starts"][i],
            "goal": d["goals"][i],
            "spl": float(d["spl"][i]),
            "steps": int(d["steps"][i]),
        }
    # Old goal from ep_idx-1
    mask_prev = (d["scenes"] == scene) & (d["ep_idx"] == ep_idx - 1)
    if mask_prev.any():
        idx = np.where(mask_prev)[0][0]
        out["old_goal"] = d["goals"][idx]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6))
    if n == 1:
        axes = [axes]

    for ax, (cond, label, scene, ep, colour) in zip(axes, PANELS):
        data = load_episode_pair(args.traj_dir, cond, scene, ep)
        if "reset" not in data or "persistent" not in data:
            ax.text(0.5, 0.5, f"{cond} {scene} ep{ep}\nmissing data",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        # Plot reset trajectory (solid line, full colour)
        rp = data["reset"]["positions"][:, [0, 2]]
        ax.plot(rp[:, 0], rp[:, 1], "-", color=colour, lw=2.2,
                alpha=0.95, zorder=4)

        # Plot persistent trajectory (dashed line, faded colour)
        pp = data["persistent"]["positions"][:, [0, 2]]
        ax.plot(pp[:, 0], pp[:, 1], "--", color=colour, lw=1.4,
                alpha=0.45, zorder=3)

        # Markers
        s = data["reset"]["start"][[0, 2]]
        g = data["reset"]["goal"][[0, 2]]
        ax.scatter(*s, s=180, c="white", edgecolor="black",
                   linewidths=1.5, marker="o", zorder=6)
        ax.text(s[0], s[1], "S", ha="center", va="center",
                fontsize=9.5, fontweight="bold", zorder=7)
        ax.scatter(*g, s=260, c="gold", edgecolor="black",
                   linewidths=1.5, marker="*", zorder=6)
        ax.annotate("Goal N\n(new)", (g[0], g[1]),
                    xytext=(g[0] + 0.4, g[1] + 0.3),
                    fontsize=8, fontweight="bold", zorder=7)
        if "old_goal" in data:
            og = data["old_goal"][[0, 2]]
            ax.scatter(*og, s=160, c="lightyellow", edgecolor="black",
                       linewidths=1.0, marker="*", zorder=6)
            ax.annotate("Goal N$-$1\n(memory)", (og[0], og[1]),
                        xytext=(og[0] + 0.4, og[1] - 0.3),
                        fontsize=8, color="dimgrey", zorder=7)

        ax.set_aspect("equal")
        ax.set_title(
            f"{label} | scene {scene}, ep N={ep}\n"
            f"reset SPL={data['reset']['spl']:.2f} ({data['reset']['steps']}st) | "
            f"persistent SPL={data['persistent']['spl']:.2f} ({data['persistent']['steps']}st)",
            fontsize=9,
        )
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("z (m)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(linestyle=":", alpha=0.3)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

        # Auto-expand axis limits so labels fit
        all_x = np.concatenate([rp[:, 0], pp[:, 0], [s[0], g[0]]] +
                                ([data["old_goal"][[0, 2]][:1]] if "old_goal" in data else []))
        all_z = np.concatenate([rp[:, 1], pp[:, 1], [s[1], g[1]]] +
                                ([data["old_goal"][[0, 2]][1:]] if "old_goal" in data else []))
        x_pad = 0.6
        z_pad = 0.6
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad + 1.5)
        ax.set_ylim(all_z.min() - z_pad, all_z.max() + z_pad)

    # Single shared legend at the bottom
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.2, label="Reset memory (LSTM zeroed)"),
        Line2D([0], [0], color="black", lw=1.4, ls="--", alpha=0.55,
               label="Persistent memory (LSTM from N$-$1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=10, label="Start (N)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=14, label="New goal"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="lightyellow",
               markeredgecolor="black", markersize=12, label="Old goal (LSTM target)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=5, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.06))
    fig.subplots_adjust(bottom=0.18)

    fig.suptitle(
        "Persistent-memory failure: LSTM state from previous episode "
        "interferes with new-task navigation",
        fontsize=10.5, y=1.00,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out_dir / f"shortcut_paired_traj.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
