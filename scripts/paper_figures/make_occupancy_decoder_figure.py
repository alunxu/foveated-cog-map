"""
WJ-C: occupancy decoder figure (Wijmans Fig 4 replication).

Two views:
- Left: per-condition IoU bar chart with 5-fold CV error bars. Tests
  H1 mechanism: bottleneck conditions (blind, coarse) should decode
  metric maps more accurately from final memory than rich-encoder
  conditions (uniform, foveated).
- Right: 4-panel grid showing one example episode per condition:
  ground-truth occupancy, predicted occupancy, trajectory-mask overlay.

Reads:
    --results-dir <dir>/{cond}_occupancy.json + {cond}_samples.npz
Writes: <out-dir>/fig_occupancy_decoder.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()
import numpy as np


CONDS = [
    ("blind",      "Blind",          "#444444"),
    ("matched128", "Coarse (1$\\times$1)", "#377eb8"),
    ("uniform",    "Uniform",        "#4daf4a"),
    ("foveated",   "Foveated (fix)", "#e41a1c"),
]


def _render_grid(ax, gt, pred, mask, title, colour):
    """Render side-by-side ground-truth vs prediction (small, clean)."""
    H, W = gt.shape
    # Compose RGB: ground-truth as gray scale (0=obstacle dark, 1=free
    # light), overlay prediction wrong-cells in red.
    rgb = np.dstack([gt * 0.85 + 0.15] * 3).astype(np.float32)  # base gray
    # Where we have a label (in mask) and prediction differs from GT,
    # tint that pixel toward red.
    diff = (pred != gt) & (mask > 0.5)
    rgb[diff] = np.array([0.85, 0.30, 0.30])
    # Where prediction matches GT in mask: tint slightly toward green.
    match = (pred == gt) & (mask > 0.5) & (gt > 0.5)
    rgb[match] = np.array([0.65, 0.85, 0.55])
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=9, color=colour, fontweight="bold")
    for s_ in ("top", "right", "bottom", "left"):
        ax.spines[s_].set_visible(True)
        ax.spines[s_].set_linewidth(0.4)
        ax.spines[s_].set_color("#888")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.0, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.18)
    ax_bar = fig.add_subplot(gs[0, 0])

    # --- Left: IoU bar chart per condition ---
    means = []; stds = []; labels = []; colours = []
    for key, label, colour in CONDS:
        p = args.results_dir / f"{key}_occupancy.json"
        if not p.exists():
            means.append(np.nan); stds.append(0); labels.append(label); colours.append(colour)
            continue
        d = json.loads(p.read_text())
        means.append(d["mean_iou"])
        stds.append(d.get("std_iou", 0))
        labels.append(label)
        colours.append(colour)

    x = np.arange(len(CONDS))
    valid = ~np.isnan(means)
    ax_bar.bar(x[valid], np.array(means)[valid], yerr=np.array(stds)[valid],
               color=[colours[i] for i in range(len(CONDS)) if valid[i]],
               edgecolor="black", linewidth=0.6, capsize=4)
    for i, (m, s) in enumerate(zip(means, stds)):
        if not np.isnan(m):
            ax_bar.text(x[i], m + s + 0.01, f"{m:.2f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=20, ha="right")
    ax_bar.set_ylabel("Occupancy IoU (5-fold CV)")
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_title("Allocentric occupancy decoded from $(\\mathbf{h}_T, \\mathbf{c}_T)$",
                      loc="left")
    for s_ in ("top", "right"):
        ax_bar.spines[s_].set_visible(False)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.3)

    # --- Right: 2x2 example renders per condition ---
    sub_gs = gs[0, 1].subgridspec(2, 2, wspace=0.10, hspace=0.20)
    flat_axes = [fig.add_subplot(sub_gs[r, c]) for r in range(2) for c in range(2)]

    for ax, (key, label, colour) in zip(flat_axes, CONDS):
        sp = args.results_dir / f"{key}_samples.npz"
        if not sp.exists():
            ax.text(0.5, 0.5, f"missing {key}", ha="center", va="center",
                    transform=ax.transAxes); continue
        s = np.load(sp)
        # Pick the median-IoU sample to show typical decoder behaviour.
        per_iou = []
        for i in range(s["preds"].shape[0]):
            inter = ((s["preds"][i] > 0.5) & (s["targets"][i] > 0.5)
                     & (s["masks"][i] > 0.5)).sum()
            union = (((s["preds"][i] > 0.5) | (s["targets"][i] > 0.5))
                     & (s["masks"][i] > 0.5)).sum()
            per_iou.append(inter / max(1, union))
        med_idx = int(np.argsort(per_iou)[len(per_iou) // 2])
        _render_grid(ax, s["targets"][med_idx], s["preds"][med_idx],
                     s["masks"][med_idx], f"{label}", colour)

    plt.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.15)
    out = args.out_dir / "fig_occupancy_decoder.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
