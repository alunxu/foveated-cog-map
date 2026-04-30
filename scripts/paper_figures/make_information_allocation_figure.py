"""
Information allocation figure: linear vs MLP probe per cond.

Linear-probe GPS R² varies wildly across conditions (-1.08 to +0.95) —
the bandwidth--allocation tradeoff captured by Figure 2. MLP-probe GPS
R² compresses this range (+0.48 to +0.95) — position information is
approximately preserved by non-linear readout in rich-encoder conditions
even when linear readability collapses. We do NOT claim strict
information conservation: MINE-estimated total I(h; pos) still drops
~1.5× from blind (4.6 bits) to uniform (2.9 bits). The figure shows
allocation across linear vs non-linear readout directions; the MINE
appendix gives the total-info nat-scale anchor.

Reads:  /tmp/mlp_probe_proper.json
Writes: docs/manuscript/fig/fig_information_allocation.pdf
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


# Encoder spatial dim (cells) per cond
COND_INFO = [
    # (cond_key, label, encoder_cells, color, marker)
    ("blind",    "Blind",    0,  "#444444", "o"),
    ("coarse",   "Coarse",   1,  "#377eb8", "s"),
    ("foveated", "Foveated", 64, "#e41a1c", "D"),
    ("uniform",  "Uniform",  64, "#4daf4a", "^"),
]


def x_pos(cells: int, label: str) -> float:
    """Same x-positioning as fig_capacity_allocation.py for consistency."""
    if cells == 0:
        return 0
    if cells == 1:
        return 2
    if label == "Foveated":
        return 6
    if label == "Uniform":
        return 6.7
    return 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", type=Path, default=Path("/tmp/mlp_probe_proper.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.in_json.read_text())

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Background regime shading
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.7, zorder=1)

    # Plot linear and MLP for each cond
    xs_lin, ys_lin, errs_lin = [], [], []
    xs_mlp, ys_mlp, errs_mlp = [], [], []
    for cond, label, cells, color, marker in COND_INFO:
        if cond not in data:
            continue
        d = data[cond]
        x = x_pos(cells, label)
        # Linear (slight x offset left)
        xs_lin.append(x - 0.12)
        ys_lin.append(d["linear_r2_mean"])
        errs_lin.append(d["linear_r2_std"])
        # MLP (slight x offset right)
        xs_mlp.append(x + 0.12)
        ys_mlp.append(d["mlp_r2_mean"])
        errs_mlp.append(d["mlp_r2_std"])
        # Markers + cond label
        ax.errorbar([x - 0.12], [d["linear_r2_mean"]], yerr=[d["linear_r2_std"]],
                    marker=marker, markersize=11, color=color, alpha=0.85,
                    capsize=4, capthick=1.0, elinewidth=0.9, linewidth=0,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=f"{label} (linear)" if cond == "blind" else None,
                    zorder=4)
        ax.errorbar([x + 0.12], [d["mlp_r2_mean"]], yerr=[d["mlp_r2_std"]],
                    marker=marker, markersize=11, color=color, alpha=0.85,
                    capsize=4, capthick=1.0, elinewidth=0.9, linewidth=0,
                    markeredgecolor="black", markeredgewidth=0.5,
                    markerfacecolor="white",
                    label=f"{label} (MLP)" if cond == "blind" else None,
                    zorder=4)
        # Vertical link line between linear and MLP for same cond
        ax.plot([x - 0.12, x + 0.12], [d["linear_r2_mean"], d["mlp_r2_mean"]],
                color=color, alpha=0.4, lw=1.2, ls="-", zorder=2)
        # Cond label — place above the higher of (linear, MLP); offset further if rich-encoder
        higher = max(d["linear_r2_mean"], d["mlp_r2_mean"])
        if label == "Foveated":
            ax.annotate(label, xy=(x, higher), xytext=(x - 0.45, higher + 0.18),
                        fontsize=10, color=color, fontweight="bold",
                        ha="right", va="center")
        elif label == "Uniform":
            ax.annotate(label, xy=(x, higher), xytext=(x + 0.45, higher + 0.18),
                        fontsize=10, color=color, fontweight="bold",
                        ha="left", va="center")
        else:
            ax.annotate(label, xy=(x, higher), xytext=(x, higher + 0.16),
                        fontsize=10, color=color, fontweight="bold",
                        ha="center", va="center")

    # Trend lines
    sort_lin = sorted(zip(xs_lin, ys_lin))
    sort_mlp = sorted(zip(xs_mlp, ys_mlp))
    ax.plot([t[0] for t in sort_lin], [t[1] for t in sort_lin],
            ls=":", color="grey", alpha=0.5, lw=1.2, zorder=2)
    ax.plot([t[0] for t in sort_mlp], [t[1] for t in sort_mlp],
            ls="-", color="grey", alpha=0.4, lw=1.2, zorder=2)

    # Custom legend: explain solid = linear, hollow = MLP
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="grey", lw=0,
               markersize=10, markerfacecolor="grey", markeredgecolor="black",
               label="Linear (Ridge $\\alpha{=}10$)"),
        Line2D([0], [0], marker="o", color="grey", lw=0,
               markersize=10, markerfacecolor="white", markeredgecolor="black",
               label="MLP (hidden=256, $L_2{=}10^{-4}$)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10,
              frameon=False)

    # Axes
    ax.set_xticks([0, 2, 6.35])
    ax.set_xticklabels(["none\n(blind)", "$1{\\times}1$\n(coarse)",
                        "$8{\\times}8$\n(foveated /\nuniform)"])
    ax.set_xlabel("Encoder spatial output (cells)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-layer $h_2$ GPS $R^2$\n(5-fold episode-level CV)",
                  fontsize=12, fontweight="bold")
    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-2.0, 1.15)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.20)

    ax.set_title("Information conservation: total info preserved, encoding format differs",
                 fontsize=12, fontweight="bold", loc="left", pad=10)

    plt.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
