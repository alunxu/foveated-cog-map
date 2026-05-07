"""
§4.4 main-text behavioural validation figure: one canonical persistent-
memory failure case per condition, overlaid on the actual Gibson scene
top-down floor plan.

Layout: 1 row × 4 cols. Each panel = one paired-episode case where
the reset-memory agent solves the new task efficiently while the
persistent-memory agent fails, illustrating the dominant failure mode
for that condition (tries-new for Blind/Coarse, locks-onto-old for
Uniform, wanders for Foveated-fix). The full 4×4 case-study catalog
spanning all margin-quartiles per condition lives in the appendix
(make_shortcut_paired_trajectory_figure.py).

Aggregate margin numbers per condition are reported in the caption.

Reads:
    --traj-dir <dir>/{cond}_gibson_traj.npz
    --topdown-dir <dir>/{scene}.{png,json}

Writes: <out-dir>/figa16_shortcut_canonical.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()
import numpy as np
from PIL import Image


# Hand-picked canonical case per condition.  Selection criterion: matches
# the condition's dominant failure-mode label and is reasonably close
# to the per-condition margin mode (not the most extreme outlier).
CANONICAL = [
    # (cond_key, label, scene, ep_idx, colour, fail_tag)
    ("blind",   "Blind",          "8WUmhLawc2A", 6, "#444444",
     "tries new, can't reach"),
    ("matched", "Coarse (1$\\times$1)", "Ackermanville", 9, "#377eb8",
     "tries new, can't reach"),
    ("foveated_logpolar", "Fov-LP", "1pXnuDYAj8r", 5, "#984ea3",
     "robust to memory carryover"),
    ("foveated", "Foveated", "1pXnuDYAj8r", 8, "#e41a1c",
     "wanders"),
    ("uniform", "Uniform",        "8WUmhLawc2A", 8, "#4daf4a",
     "locks onto old goal"),
]


def load_episode_pair(traj_dir: Path, cond: str, scene: str, ep_idx: int) -> dict:
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
    mask_prev = (d["scenes"] == scene) & (d["ep_idx"] == ep_idx - 1)
    if mask_prev.any():
        idx = np.where(mask_prev)[0][0]
        out["old_goal"] = d["goals"][idx]
    return out


def load_topdown(topdown_dir: Path, scene: str):
    p_png = topdown_dir / f"{scene}.png"
    p_json = topdown_dir / f"{scene}.json"
    if not (p_png.exists() and p_json.exists()):
        return None
    img = np.asarray(Image.open(p_png).convert("RGB"))
    meta = json.loads(p_json.read_text())
    return img, meta["world_lower_bound"], meta["world_upper_bound"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--topdown-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_cols = len(CANONICAL)
    # Cell aspect target = figure_w / (n_cols * panel_h).  We use the
    # SAME data-bounds aspect as the appendix catalog (≈1.071) so panels
    # in main and appendix figures share visual style.
    target_aspect = 1.071
    panel_w = 3.5
    panel_h = panel_w / target_aspect  # 3.27"
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(panel_w * n_cols, panel_h + 0.8),
                             gridspec_kw={"wspace": 0.18})

    for ax, (cond_key, label, scene, ep, colour, fail_tag) in zip(axes, CANONICAL):
        data = load_episode_pair(args.traj_dir, cond_key, scene, ep)
        if "reset" not in data or "persistent" not in data:
            ax.text(0.5, 0.5, f"missing\n{cond_key} {scene} ep{ep}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        # Background top-down scene map (if available).
        td = load_topdown(args.topdown_dir, scene)
        if td is not None:
            img, lo, hi = td
            # origin='lower': Habitat's get_topdown_map_from_sim writes
            # array[0, *] at world z = lower_bound[2]; with imshow
            # extent=[lo[0], hi[0], lo[2], hi[2]] this requires origin='lower'
            # so image[0,0] is placed at plot's bottom-left = world (lo[0], lo[2]).
            # origin='upper' (the matplotlib default) would vertically flip the
            # image relative to trajectory data, putting trajectories in walls.
            ax.imshow(img, extent=[lo[0], hi[0], lo[2], hi[2]],
                      origin="lower", alpha=0.55, zorder=0,
                      interpolation="bilinear")

        # Reset trajectory: light reference baseline (the "good behaviour"
        # context).  Plotted thin and translucent so it doesn't compete
        # with the focal trace.
        rp = data["reset"]["positions"][:, [0, 2]]
        ax.plot(rp[:, 0], rp[:, 1], "-", color=colour, lw=1.6,
                alpha=0.55, zorder=3)

        # Persistent trajectory: focal trace --- the ANSWER to "where does
        # persistent memory take the agent?".  Plotted thicker and fully
        # opaque, dashed to distinguish line style at a glance.
        pp = data["persistent"]["positions"][:, [0, 2]]
        ax.plot(pp[:, 0], pp[:, 1], "--", color=colour, lw=2.8,
                alpha=1.0, zorder=4)
        # End-point marker so "where did the persistent agent stop?" reads
        # at a glance even when the trajectory winds many steps.
        ax.scatter(pp[-1, 0], pp[-1, 1], s=120, marker="X",
                   color=colour, edgecolor="black",
                   linewidths=1.4, zorder=5)

        # Markers: start, new goal, old goal.
        s = data["reset"]["start"][[0, 2]]
        g = data["reset"]["goal"][[0, 2]]
        og = data["old_goal"][[0, 2]] if "old_goal" in data else None
        ax.scatter(*s, s=140, c="white", edgecolor="black",
                   linewidths=1.4, marker="o", zorder=6)
        ax.text(s[0], s[1], "S", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=7)
        ax.scatter(*g, s=240, c="gold", edgecolor="black",
                   linewidths=1.4, marker="*", zorder=6)
        if og is not None:
            ax.scatter(*og, s=170, c="white", edgecolor="black",
                       linewidths=1.2, marker="*", zorder=6)

        # Tight axis around geometry, expanded to target aspect.
        pts_x = np.concatenate([rp[:, 0], pp[:, 0], [s[0], g[0]]] +
                               ([np.array([og[0]])] if og is not None else []))
        pts_z = np.concatenate([rp[:, 1], pp[:, 1], [s[1], g[1]]] +
                               ([np.array([og[1]])] if og is not None else []))
        pad = 1.0
        x_min, x_max = pts_x.min() - pad, pts_x.max() + pad
        z_min, z_max = pts_z.min() - pad, pts_z.max() + pad
        x_center, z_center = (x_min + x_max) / 2, (z_min + z_max) / 2
        x_range, z_range = x_max - x_min, z_max - z_min
        if x_range / z_range < target_aspect:
            x_range = z_range * target_aspect
        else:
            z_range = x_range / target_aspect
        ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
        ax.set_ylim(z_center - z_range / 2, z_center + z_range / 2)
        ax.set_aspect("equal")

        # Two-row panel header above the plot:
        #   row 1: condition name (bold) -- failure-mode tag (italic gray)
        #   row 2: SPL summary (mono gray)
        ax.set_title(label, loc="left", pad=24)
        ax.text(1.0, 1.06, f"[{fail_tag}]", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10,
                style="italic", color="#666666")
        rs, ps = data["reset"]["spl"], data["persistent"]["spl"]
        ax.text(0.0, 1.01,
                f"reset SPL={rs:.2f} $\\rightarrow$ persist SPL={ps:.2f}",
                transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9,
                color="#444", family="monospace")

        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right", "bottom", "left"):
            ax.spines[s_].set_visible(True)
            ax.spines[s_].set_linewidth(0.5)
            ax.spines[s_].set_color("#888")

    # Single shared legend at the bottom.
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.6, alpha=0.55,
               label="Reset memory (baseline)"),
        Line2D([0], [0], color="black", lw=2.8, ls="--",
               label="Persistent memory (focal)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="black",
               markeredgecolor="black", markersize=10,
               label="Persistent end"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=10, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=14, label="New goal"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=12, label="Old goal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.18)
    out = args.out_dir / "figa16_shortcut_canonical.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()