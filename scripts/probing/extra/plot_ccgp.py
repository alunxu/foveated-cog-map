"""Plot CCGP results: per-target WCV / CCGP / AI bars across 5 conditions.

Output: a 2x3 figure (2 dichotomies x 3 targets, dropping dist_bin which is
near-ceiling everywhere per the smoke result).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]
NICE = {
    "blind": "blind",
    "coarse": "coarse 1×1",
    "foveated": "foveated",
    "uniform": "uniform",
    "foveated_logpolar": "fov-logpolar",
}
COLORS = {
    "blind": "#5b5b5b",
    "coarse": "#d97a35",
    "foveated": "#2c7fb8",
    "uniform": "#6a51a3",
    "foveated_logpolar": "#7fcdbb",
}
TARGETS = ["pos_x_bin", "heading_oct", "goal_quadrant", "dist_bin"]
TARGET_LABEL = {
    "pos_x_bin": "pos_x bin (4)",
    "heading_oct": "heading octant (8)",
    "goal_quadrant": "goal quadrant (4)",
    "dist_bin": "dist-to-goal bin (3)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccgp_json", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    d = json.load(open(args.ccgp_json))

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.0), constrained_layout=True,
                              sharey="row")
    xs = np.arange(len(CONDITIONS))
    bw = 0.35

    for ti, tgt in enumerate(TARGETS):
        # Row 0: WCV bars (within-condition decoding accuracy)
        ax = axes[0, ti]
        wcv = [d.get(c, {}).get(tgt, {}).get("wcv_d1", np.nan) for c in CONDITIONS]
        ax.bar(xs, wcv, color=[COLORS[c] for c in CONDITIONS], edgecolor="black",
                linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right",
                            fontsize=8)
        ax.set_title(TARGET_LABEL[tgt], fontsize=10)
        ax.axhline(1.0/4 if "bin" in tgt or "quadrant" in tgt else 1.0/8,
                    color="black", linewidth=0.6, linestyle=":")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        if ti == 0:
            ax.set_ylabel("Within-condition CV\n(decoding acc)", fontsize=9)

        # Row 1: AI bars (abstraction index across scene parity D1)
        ax = axes[1, ti]
        ai = [d.get(c, {}).get(tgt, {}).get("ai_d1", np.nan) for c in CONDITIONS]
        ax.bar(xs, ai, color=[COLORS[c] for c in CONDITIONS], edgecolor="black",
                linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right",
                            fontsize=8)
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        if ti == 0:
            ax.set_ylabel("Abstraction index\n(CCGP / WCV) across scenes",
                            fontsize=9)

    fig.suptitle("Cross-Condition Generalisation Performance (Bernardi 2020) — abstraction across scene partitions",
                  fontsize=11, y=1.04)
    fig.savefig(args.out_path, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
