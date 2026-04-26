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

# Hand-selected (cond, scene, ep_idx) tuples chosen to ILLUSTRATE the
# THREE distinct persistent-memory failure modes. Selection criterion:
# reset SPL > 0.5, persistent SPL < 0.2, persistent steps > 30, same-floor
# old-vs-new goal (|Δy|<0.5m). Among those, pick the most representative:
# - blind: most negative (avg-traj → new goal closer than → old goal): tries new, can't reach.
# - uniform: most positive: locks onto old goal location.
# - foveated: ~0 margin: wanders, locks onto neither goal.
PANELS = [
    # (cond_key, label, scene, ep_idx, colour)
    ("blind",            "Blind",          "8WUmhLawc2A", 6, "#444444"),  # margin -6.91m: tries new
    ("uniform",          "Uniform",        "8WUmhLawc2A", 8, "#4daf4a"),  # margin +15.59m: LOCKED ONTO OLD
    ("foveated",         "Foveated (fix)", "1pXnuDYAj8r", 8, "#e41a1c"),  # margin -0.61m: wanders
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


# Aggregate margin per condition (from same-floor paired-episode failures
# meeting the criteria: reset SPL > 0.5, persistent SPL < 0.2, persistent
# steps >= 30, |Δy_goal| < 0.5m). Pre-computed elsewhere; hardcoded here
# to keep the figure self-contained for paper builds.
MARGIN_TABLE = [
    # (label,        n,  margin,  colour)
    ("Blind",         27, -0.38, "#444444"),
    ("Coarse",       35, -0.57, "#377eb8"),
    ("Uniform",       46, +1.83, "#4daf4a"),
    ("Foveated (fix)", 16, -0.59, "#e41a1c"),
    ("Foveated (learned)",    5, +2.30, "#ff7f00"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = len(PANELS)
    # 2×2 layout: 3 traj panels in (0,0), (0,1), (1,0); margin in (1,1)
    fig = plt.figure(figsize=(10.0, 8.4))
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.32,
                          left=0.07, right=0.97,
                          top=0.96, bottom=0.07)
    panel_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
    ]
    ax_margin = fig.add_subplot(gs[1, 1])

    for ax, (cond, label, scene, ep, colour) in zip(panel_axes, PANELS):
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
                alpha=0.50, zorder=3)

        # Markers (no inline text labels — leader lines on the side)
        s = data["reset"]["start"][[0, 2]]
        g = data["reset"]["goal"][[0, 2]]
        og = data["old_goal"][[0, 2]] if "old_goal" in data else None

        ax.scatter(*s, s=180, c="white", edgecolor="black",
                   linewidths=1.5, marker="o", zorder=6)
        ax.text(s[0], s[1], "S", ha="center", va="center",
                fontsize=9.5, fontweight="bold", zorder=7)
        ax.scatter(*g, s=300, c="gold", edgecolor="black",
                   linewidths=1.5, marker="*", zorder=6)
        if og is not None:
            ax.scatter(*og, s=200, c="white", edgecolor="black",
                       linewidths=1.2, marker="*", zorder=6)

        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("z (m)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(linestyle=":", alpha=0.3)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

        # Per-panel SPL annotation in upper-left (replaces verbose title)
        info = (f"reset SPL={data['reset']['spl']:.2f} "
                f"({data['reset']['steps']} st)\n"
                f"persistent SPL={data['persistent']['spl']:.2f} "
                f"({data['persistent']['steps']} st)")
        ax.text(0.02, 0.98, info, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5,
                color="#333", family="monospace")

        # Auto-expand axis limits and add per-marker leader-line labels
        # ON THE OUTSIDE of the trajectory cloud to avoid overlap.
        pts_x = np.concatenate([rp[:, 0], pp[:, 0], [s[0], g[0]]] +
                                ([np.array([og[0]])] if og is not None else []))
        pts_z = np.concatenate([rp[:, 1], pp[:, 1], [s[1], g[1]]] +
                                ([np.array([og[1]])] if og is not None else []))
        x_pad = 1.0
        z_pad = 1.0
        x_min, x_max = pts_x.min() - x_pad, pts_x.max() + x_pad
        z_min, z_max = pts_z.min() - z_pad, pts_z.max() + z_pad
        # Add extra room on the right for leader-line labels
        ax.set_xlim(x_min, x_max + 2.5)
        ax.set_ylim(z_min, z_max)

        # Place "New goal" label to the right of x_max with a leader line
        x_label = x_max + 0.6
        ax.annotate("new goal", xy=(g[0], g[1]),
                    xytext=(x_label, g[1]),
                    fontsize=8, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color="#888888",
                                    lw=0.5))
        if og is not None:
            ax.annotate("old goal", xy=(og[0], og[1]),
                        xytext=(x_label, og[1]),
                        fontsize=8, color="#666", va="center",
                        arrowprops=dict(arrowstyle="-", color="#888888",
                                        lw=0.5))

    # Panel (d): aggregate margin per condition
    ax_m = ax_margin
    labels = [r[0] for r in MARGIN_TABLE]
    margins = [r[2] for r in MARGIN_TABLE]
    colours = [r[3] for r in MARGIN_TABLE]
    counts = [r[1] for r in MARGIN_TABLE]
    y = np.arange(len(labels))
    ax_m.barh(y, margins, color=colours, edgecolor="black", linewidth=0.6)
    ax_m.axvline(0, color="black", lw=0.6)
    ax_m.set_yticks(y)
    ax_m.set_yticklabels(labels, fontsize=9)
    ax_m.invert_yaxis()
    ax_m.set_xlabel("dist to old $-$ dist to new (m)", fontsize=9)
    ax_m.tick_params(axis="x", labelsize=8)
    # Expand x-axis so n=N labels fit comfortably outside the bars
    pad = 0.6
    xmin = min(margins) - pad
    xmax = max(margins) + pad + 0.3
    ax_m.set_xlim(xmin, xmax)
    for i, (m, c) in enumerate(zip(margins, counts)):
        # Place n=N OUTSIDE the bar (away from zero line)
        x_pos = m + (0.10 if m > 0 else -0.10)
        ha = "left" if m > 0 else "right"
        ax_m.text(x_pos, i, f"$n{{=}}{c}$", va="center", ha=ha,
                  fontsize=8, color="#444")
    # Region labels at the TOP (not overlapping bars)
    ax_m.text(0.96, 1.02, "closer to NEW",
              transform=ax_m.transAxes,
              ha="right", va="bottom", fontsize=7.5,
              style="italic", color="#666")
    ax_m.text(0.04, 1.02, "closer to OLD",
              transform=ax_m.transAxes,
              ha="left", va="bottom", fontsize=7.5,
              style="italic", color="#666")
    ax_m.set_title("Where the persistent agent ends up",
                   fontsize=11, fontweight="bold", pad=18)
    ax_m.grid(axis="x", linestyle=":", alpha=0.3)
    for s_ in ("top", "right"):
        ax_m.spines[s_].set_visible(False)

    # Single shared legend at the bottom for traj panels
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.2, label="Reset memory"),
        Line2D([0], [0], color="black", lw=1.4, ls="--", alpha=0.55,
               label="Persistent memory"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=10, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=14, label="New goal"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=12, label="Old goal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    for ext in ("pdf", "png"):
        out = args.out_dir / f"shortcut_paired_traj.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
