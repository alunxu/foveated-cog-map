"""
Goal-vector figure for paper §4.5 rescue disconfirmation.

Shows the key result: ego-to-goal vector is NOT linearly decodable from
hidden state in any condition, even though goal DISTANCE is. This forces
a revised explanation of the shortcut-brittleness ordering: it is not
"memory of old goal vector decays slowly" — goal vector isn't in memory
at all (it's a per-step policy input). It is "integrated trajectory /
scene history interacting with a new goal."

Usage:
    python scripts/paper_figures/make_goal_vector_figure.py \\
        --in data/goal_vector.json --out-dir docs/NeurIPS_2026/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COND_DISPLAY = {
    "blind_gibson":            ("Blind",              "#444444"),
    "uniform_gibson":          ("Uniform",            "#4daf4a"),
    "foveated_gibson":         ("Foveated (fixed)",   "#e41a1c"),
    "foveated_learned_gibson": ("Foveated (learned)", "#ff7f00"),
    "matched_gibson":          ("Matched-compute",    "#377eb8"),
}
COND_ORDER = [
    "blind_gibson",
    "uniform_gibson",
    "foveated_gibson",
    "foveated_learned_gibson",
    "matched_gibson",
]


def fig_goal_vector_grouped(in_path: Path, out_path: Path) -> None:
    with open(in_path) as f:
        d = json.load(f)

    labels, colours = [], []
    r2_vec, r2_dist, r2_dir = [], [], []
    for c in COND_ORDER:
        if c not in d:
            continue
        r = d[c]
        labels.append(COND_DISPLAY[c][0])
        colours.append(COND_DISPLAY[c][1])
        r2_vec.append(r["goal_vector_r2"])
        r2_dist.append(r["goal_dist_r2"])
        r2_dir.append(r["goal_direction_r2"])

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    n = len(labels)
    x = np.arange(n)
    w = 0.26

    # Clamp extreme negatives so bars remain visible.
    def _clip(v):
        return max(v, -0.5)

    ax.bar(x - w, [_clip(v) for v in r2_dist], w,
           color=colours, edgecolor="black", linewidth=0.6,
           label="Goal distance")
    ax.bar(x,     [_clip(v) for v in r2_dir],  w,
           color=colours, edgecolor="black", linewidth=0.6,
           hatch="///", alpha=0.7, label="Goal direction")
    ax.bar(x + w, [_clip(v) for v in r2_vec],  w,
           color=colours, edgecolor="black", linewidth=0.6,
           hatch="\\\\\\", alpha=0.5, label="Goal vector (2D)")

    # Annotate clipped bars.
    for xi, vals in zip(x, zip(r2_dist, r2_dir, r2_vec)):
        for j, v in enumerate(vals):
            if v < -0.5:
                ax.text(xi + (j - 1) * w, -0.48,
                        f"{v:+.1f}", ha="center", va="top",
                        fontsize=7, color="red")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.5, color="grey", linewidth=0.4, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=10)
    ax.set_ylabel("Ridge probe $R^2$", fontsize=10)
    ax.set_ylim(-0.5, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8, frameon=True, ncol=3)
    ax.set_title("Goal distance is decodable; goal direction and full goal vector are not",
                 fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    # Text summary for paper
    print(f"\n{'Condition':<24} {'vec-R²':<10} {'dist-R²':<10} {'dir-R²':<10}")
    for c in COND_ORDER:
        if c not in d:
            continue
        r = d[c]
        print(f"{COND_DISPLAY[c][0]:<24} "
              f"{r['goal_vector_r2']:<+10.3f} "
              f"{r['goal_dist_r2']:<+10.3f} "
              f"{r['goal_direction_r2']:<+10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_goal_vector_grouped(args.in_path, args.out_dir / "goal_vector_probe.pdf")


if __name__ == "__main__":
    main()
