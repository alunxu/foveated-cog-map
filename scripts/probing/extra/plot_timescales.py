"""Plot intrinsic timescales: tau distribution per condition (boxplot + median/IQR).
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]
NICE = {
    "blind": "blind",
    "coarse": "coarse 1×1",
    "foveated": "foveated 4×4",
    "uniform": "uniform 4×4",
    "foveated_logpolar": "fov-logpolar",
}
COLORS = {
    "blind": "#5b5b5b",
    "coarse": "#d97a35",
    "foveated": "#2c7fb8",
    "uniform": "#6a51a3",
    "foveated_logpolar": "#7fcdbb",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    d = json.load(open(args.in_path))

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5), constrained_layout=True)
    xs = np.arange(len(CONDITIONS))

    boxdata = []
    medians = []
    for c in CONDITIONS:
        if c not in d:
            boxdata.append([np.nan])
            medians.append(np.nan)
            continue
        boxdata.append(d[c]["tau_dist"])
        medians.append(d[c]["median_tau"])

    bp = ax.boxplot(boxdata, positions=xs, widths=0.55, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(COLORS[c])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    # Annotate medians on top
    for i, c in enumerate(CONDITIONS):
        if not np.isnan(medians[i]):
            ax.text(i, medians[i] + 1.5, f"{medians[i]:.1f}",
                    ha="center", fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.7))

    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right",
                        fontsize=9)
    ax.set_ylabel("intrinsic timescale τ (steps)", fontsize=10)
    ax.set_title("Murray 2014 intrinsic timescale: blind 2.1× longer than sighted",
                  fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.savefig(args.out_path, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
