"""Plot Tangling Q distributions per condition (boxplot + median line)."""
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
    medians = [d.get(c, {}).get("median", np.nan) for c in CONDITIONS]
    p25 = [d.get(c, {}).get("p25", np.nan) for c in CONDITIONS]
    p75 = [d.get(c, {}).get("p75", np.nan) for c in CONDITIONS]
    p95 = [d.get(c, {}).get("p95", np.nan) for c in CONDITIONS]

    for i, c in enumerate(CONDITIONS):
        ax.bar(i, medians[i], 0.5, color=COLORS[c], edgecolor="black",
                linewidth=0.5, label=NICE[c])
        # IQR error bars
        ax.errorbar(i, medians[i], yerr=[[medians[i] - p25[i]], [p75[i] - medians[i]]],
                     color="black", capsize=3, linewidth=1)
        # P95 line
        ax.plot([i - 0.25, i + 0.25], [p95[i], p95[i]], color="black",
                 linewidth=0.5, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right",
                        fontsize=9)
    ax.set_ylabel("Tangling Q (median, IQR; dashed = p95)", fontsize=10)
    ax.set_title("Tangling Q (Russo 2018) — recurrent-dynamics smoothness, PCA-30 h",
                  fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.savefig(args.out_path, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
