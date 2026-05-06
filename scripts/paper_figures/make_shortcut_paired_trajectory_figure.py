"""
Appendix figure: extended persistent-memory failure-mode catalog.

Layout: 4 rows × 4 cols = 16 paired-episode failure cases. Rows are
omitted because only 2 paired-failure candidates exist). Cols are
4 representative cases per condition spanning the full range of
"persistent-direction" margin (most-tries-new → most-locks-onto-old).

The main-text canonical figure (1 panel per condition) is rendered by
make_shortcut_canonical_figure.py.

Each panel:
  - Top-down occupancy map (light grey navigable, dark grey obstacles)
  - Reset trajectory (solid line, condition colour)
  - Persistent trajectory (dashed line, condition colour, faded)
  - Start marker (white circle with "S")
  - New goal (gold star), old goal from previous episode (white star)

Reads:
    --traj-dir <dir>/{cond}_gibson_traj.npz  (paired-episode trajectories)
    --topdown-dir <dir>/{scene}.{png,json}   (per-scene Habitat top-down maps)

Writes: <out-dir>/fig5_shortcut_paired_traj.{pdf}

Aggregate margin numbers per condition are reported in the caption.
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


# (cond_key, label, colour, [(scene, ep_idx), ...4 picks ordered by margin])
ROWS = [
    ("blind",   "Blind",          "#444444", [
        ("17DRP5sb8fy",  2),  # margin -2.57: tries new
        ("Allensville",  5),  # margin +0.49: ambiguous
        ("29hnd4uzFmX",  1),  # margin +1.75: tilts old
        ("8WUmhLawc2A",  6),  # margin +9.15: locks onto old
    ]),
    ("matched", "Coarse (1$\\times$1)", "#377eb8", [
        ("17DRP5sb8fy",  3),  # margin -6.15
        ("759xd9YjKW5",  8),  # margin -4.40
        ("Ackermanville", 9), # margin +3.26
        ("8WUmhLawc2A",  3),  # margin +9.74
    ]),
    ("uniform", "Uniform",        "#4daf4a", [
        ("8WUmhLawc2A",  8),  # margin -16.27
        ("Almena",       4),  # margin -3.12
        ("Alfred",       7),  # margin +1.20
        ("8WUmhLawc2A",  5),  # margin +13.23
    ]),
    ("foveated","Foveated", "#e41a1c", [
        ("Ackermanville", 5), # margin -6.10
        ("1pXnuDYAj8r",  8),  # margin +0.22
        ("Aldrich",      6),  # margin +12.73
        ("Alfred",       1),  # margin +14.06
    ]),
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
    mask_prev = (d["scenes"] == scene) & (d["ep_idx"] == ep_idx - 1)
    if mask_prev.any():
        idx = np.where(mask_prev)[0][0]
        out["old_goal"] = d["goals"][idx]
    return out


def load_topdown(topdown_dir: Path, scene: str):
    """Return (rgb_array, world_lower_xyz, world_upper_xyz) for the scene's
    top-down map.  Returns None if the map files are not yet on disk
    (allows the figure to be regenerated mid-render-job)."""
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

    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.0, 14.0),
                             gridspec_kw={"wspace": 0.18, "hspace": 0.20})

    for r, (cond_key, cond_label, cond_colour, picks) in enumerate(ROWS):
        for c, (scene, ep) in enumerate(picks):
            ax = axes[r, c]
            data = load_episode_pair(args.traj_dir, cond_key, scene, ep)
            if "reset" not in data or "persistent" not in data:
                ax.text(0.5, 0.5, f"missing\n{scene} ep{ep}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#888")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # 1. Top-down map background (if available)
            td = load_topdown(args.topdown_dir, scene)
            if td is not None:
                img, lo, hi = td
                # Habitat's get_topdown_map_from_sim writes array[0, *] at
                # world z = lower_bound[2]; with imshow extent=[lo[0], hi[0],
                # lo[2], hi[2]] this requires origin='lower' so image[0,0]
                # lands at plot bottom-left (= world lo[0], lo[2]) and the
                # image+trajectory share the same y-direction. origin='upper'
                # vertically flips the image relative to trajectory data.
                ax.imshow(img, extent=[lo[0], hi[0], lo[2], hi[2]],
                          origin="lower", alpha=0.55, zorder=0,
                          interpolation="bilinear")

            # 2. Reset trajectory (solid, full colour)
            rp = data["reset"]["positions"][:, [0, 2]]
            ax.plot(rp[:, 0], rp[:, 1], "-", color=cond_colour, lw=2.0,
                    alpha=0.95, zorder=4)

            # 3. Persistent trajectory (dashed, faded)
            pp = data["persistent"]["positions"][:, [0, 2]]
            ax.plot(pp[:, 0], pp[:, 1], "--", color=cond_colour, lw=1.3,
                    alpha=0.55, zorder=3)

            # 4. Markers
            s = data["reset"]["start"][[0, 2]]
            g = data["reset"]["goal"][[0, 2]]
            og = data["old_goal"][[0, 2]] if "old_goal" in data else None
            ax.scatter(*s, s=110, c="white", edgecolor="black",
                       linewidths=1.3, marker="o", zorder=6)
            ax.text(s[0], s[1], "S", ha="center", va="center",
                    fontsize=8, fontweight="bold", zorder=7)
            ax.scatter(*g, s=200, c="gold", edgecolor="black",
                       linewidths=1.2, marker="*", zorder=6)
            if og is not None:
                ax.scatter(*og, s=140, c="white", edgecolor="black",
                           linewidths=1.0, marker="*", zorder=6)

            # 5. Tight axis around all geometry, then EXPAND smaller
            #    dimension so every panel ends up at the same data-bounds
            #    aspect ratio (matched to the figure cell shape). This
            #    keeps cells visually uniform across the grid while
            #    aspect="equal" preserves true spatial geometry inside
            #    each panel.
            pts_x = np.concatenate([rp[:, 0], pp[:, 0], [s[0], g[0]]] +
                                   ([np.array([og[0]])] if og is not None else []))
            pts_z = np.concatenate([rp[:, 1], pp[:, 1], [s[1], g[1]]] +
                                   ([np.array([og[1]])] if og is not None else []))
            pad = 1.0
            x_min, x_max = pts_x.min() - pad, pts_x.max() + pad
            z_min, z_max = pts_z.min() - pad, pts_z.max() + pad
            # Cell aspect = (figure_w / n_cols) / (figure_h / n_rows)
            # = (15 / 4) / (14 / 4) = 1.071.
            target_aspect = 15.0 / 14.0
            x_center, z_center = (x_min + x_max) / 2, (z_min + z_max) / 2
            x_range, z_range = x_max - x_min, z_max - z_min
            if x_range / z_range < target_aspect:
                x_range = z_range * target_aspect
            else:
                z_range = x_range / target_aspect
            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            ax.set_ylim(z_center - z_range / 2, z_center + z_range / 2)
            ax.set_aspect("equal")

            # 6. Compact SPL summary inside the panel, top-left
            rs = data["reset"]["spl"]
            ps = data["persistent"]["spl"]
            ax.text(0.03, 0.97,
                    f"reset SPL={rs:.2f}\npersist SPL={ps:.2f}",
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=8,
                    color="#222", family="monospace",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, boxstyle="round,pad=0.2"))

            # Strip ticks; the world-frame coordinates are not
            # information-bearing across panels (each scene has its own
            # bounds).  Keep light frame for visual containment.
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ("top", "right", "bottom", "left"):
                ax.spines[s_].set_visible(True)
                ax.spines[s_].set_linewidth(0.5)
                ax.spines[s_].set_color("#888")

        # Row label on the leftmost panel of each row.
        axes[r, 0].set_ylabel(cond_label, fontsize=14, fontweight="bold",
                               color=cond_colour, labelpad=12)

    # Single shared legend at the bottom.
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.0, label="Reset memory (succeeds)"),
        Line2D([0], [0], color="black", lw=1.3, ls="--", alpha=0.55,
               label="Persistent memory (fails)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=10, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=14, label="New goal"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=12, label="Old goal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.01))

    plt.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.05)
    out = args.out_dir / "figa17_shortcut_catalog.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()