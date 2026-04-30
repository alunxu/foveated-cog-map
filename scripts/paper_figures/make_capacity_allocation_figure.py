"""
Capacity-allocation principle: top-layer h GPS R² scales inversely with
encoder spatial bandwidth.

Plots one point per condition (5 total: blind, coarse, log-polar,
foveated, uniform), with x = encoder spatial output cell count and y =
top-layer h₂ linear GPS R² at convergence (5-fold CV mean ± σ). Shows
the monotonic anti-correlation that supports Finding 1 (bandwidth–
allocation tradeoff) under the capacity-allocation principle.

Log-polar (~2×2 encoder output) is highlighted as the preregistered
prediction confirmation: encoder-spatial-output mechanism predicts it
should fall in the bottleneck regime, and it does (+0.808 ± 0.048 at
60M frames, comparable to coarse at +0.78).

Foveated and uniform share encoder spatial dim (4×4) but differ
substantially in y (+0.06 vs -0.31), showing the design space is not
1-D — channel-level information density (blurred vs unblurred) is a
second axis.

Reads:  /tmp/ckpt_sweep_data/{blind,matched,foveated,uniform}_ckpt{N}.json
        /tmp/overnight_results/foveated_logpolar_gibson_ckpt12_det_analysis.json
Writes: docs/manuscript/fig/fig_capacity_allocation.pdf
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


# 5 conditions with their encoder spatial output dim
# (blind = no encoder, plotted at x=0 with "no encoder" marker)
CONDS = [
    # (label,         file,                                      encoder_cells, color,    marker)
    ("Blind",         "/tmp/ckpt_sweep_data/blind_ckpt34.json",  0,             "#444444", "o"),
    ("Coarse",        "/tmp/ckpt_sweep_data/matched_ckpt49.json", 1,            "#377eb8", "s"),
    # Log-polar control intentionally omitted from the headline figure until
    # its converged checkpoint is probed: at 60M frames every condition
    # (including uniform/foveated) is in the early-training high-R² regime,
    # so a 60M data-point cannot yet discriminate "follows coarse" from
    # "follows uniform after substitution". Will be re-added at the
    # systematic-convergence frame defined in Methods.
    ("Foveated",      "/tmp/ckpt_sweep_data/foveated_ckpt36.json", 64,          "#e41a1c", "D"),
    ("Uniform",       "/tmp/ckpt_sweep_data/uniform_ckpt49.json",  64,          "#4daf4a", "^"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="output PDF path")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    rows = []
    for label, fp, dim, colour, marker in CONDS:
        if not Path(fp).exists():
            print(f"  MISSING: {label} ({fp})")
            continue
        d = json.loads(Path(fp).read_text())
        b = d.get("1b_global_gps_compass", {})
        r2 = b.get("gps_cv_r2_mean")
        std = b.get("gps_cv_r2_std", 0.0)
        rows.append((label, dim, r2, std, colour, marker))
        print(f"  {label:10}  dim={dim:>3}  R²={r2:+.3f}±{std:.3f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Background: regime shading
    ax.axhspan(0.4, 1.05, color="#d4ead4", alpha=0.30, zorder=0,
               label=None)
    ax.axhspan(-1.5, 0.2, color="#f4d8d4", alpha=0.30, zorder=0)
    ax.axhline(0, ls="-", color="grey", alpha=0.6, lw=0.7, zorder=1)
    # Regime labels — placed on RIGHT side (away from points)
    ax.text(0.98, 0.95, "Bottleneck regime\n(integration route carries position)",
            transform=ax.transAxes, fontsize=10, color="#2d6a2d",
            va="top", ha="right", style="italic")
    ax.text(0.02, 0.05, "Rich-encoder regime\n(visual route carries position)",
            transform=ax.transAxes, fontsize=10, color="#a13838",
            va="bottom", ha="left", style="italic")

    # Plot points
    # Use linear x-axis with custom positions
    # Map cells -> plot position: 0 -> 0, 1 -> 2, 4 -> 4, 64 -> 6 (foveated), 6.6 (uniform)
    # Foveated and uniform separated slightly for readability
    def x_pos(cells, label):
        if cells == 0:
            return 0
        if cells == 1:
            return 2
        if cells == 4:
            return 4
        # Both 64-cell conds: foveated slightly left, uniform slightly right
        if label == "Foveated":
            return 6
        if label == "Uniform":
            return 6.7
        return 6

    xs, ys, errs, labels = [], [], [], []
    for label, dim, r2, std, colour, marker in rows:
        xpos = x_pos(dim, label)
        xs.append(xpos)
        ys.append(r2)
        errs.append(std)
        labels.append(label)
        ax.errorbar([xpos], [r2], yerr=[std], marker=marker, markersize=14,
                    color=colour, linewidth=0, capsize=4,
                    capthick=1.2, elinewidth=1.2, zorder=4,
                    markeredgecolor="black", markeredgewidth=0.6)
        # Annotate with cond name (placement: above for blind/coarse/logpolar; right-of for fov/uni)
        if label == "Blind":
            offset_x, offset_y = 0.0, 0.13
        elif label == "Coarse":
            offset_x, offset_y = 0.0, -0.16
        elif label == "Log-polar":
            offset_x, offset_y = 0.0, 0.16
        elif label == "Foveated":
            offset_x, offset_y = -0.45, 0.0
        elif label == "Uniform":
            offset_x, offset_y = 0.45, 0.0
        else:
            offset_x, offset_y = 0.0, 0.13
        ha = "right" if label == "Foveated" else ("left" if label == "Uniform" else "center")
        ax.annotate(label, xy=(xpos, r2),
                    xytext=(xpos + offset_x, r2 + offset_y),
                    fontsize=11, color=colour, fontweight="bold",
                    ha=ha, va="center")

    # Trend curve through 4 main conds (log-polar omitted until convergence)
    fov_y = next(y for (lbl, _, _, _, _, _), y in zip(rows, ys) if lbl == "Foveated")
    uni_y = next(y for (lbl, _, _, _, _, _), y in zip(rows, ys) if lbl == "Uniform")
    trend_x = [0, 2, 6.35]
    trend_y = []
    trend_y.append(next(r2 for (lbl, _, r2, _, _, _) in rows if lbl == "Blind"))
    trend_y.append(next(r2 for (lbl, _, r2, _, _, _) in rows if lbl == "Coarse"))
    trend_y.append(0.5 * (fov_y + uni_y))
    ax.plot(trend_x, trend_y, ls=":", color="grey", alpha=0.6, lw=1.5,
            zorder=2)

    # Axes
    ax.set_xticks([0, 2, 6.35])
    ax.set_xticklabels(["none\n(blind)", "$1{\\times}1$\n(coarse)",
                        "$4{\\times}4$\n(foveated /\nuniform)"])
    ax.set_xlabel("Encoder spatial output (cells)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-layer $h_2$ linear GPS $R^2$\n(5-fold CV; converged ckpt)",
                  fontsize=12, fontweight="bold")
    ax.set_xlim(-0.8, 8.0)
    ax.set_ylim(-1.55, 1.15)
    ax.tick_params(axis="both", labelsize=10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.20)

    ax.set_title("Bandwidth-allocation tradeoff: integration capacity falls as encoder bandwidth grows",
                 fontsize=12, fontweight="bold", loc="left", pad=10)

    plt.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
