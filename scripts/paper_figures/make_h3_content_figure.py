"""
Merged H3 content-evidence figure (2-panel):
  (a) Goal-vector probe R² per condition for three targets (distance,
      direction, full vector)
  (b) Shortcut-discovery bars: persistent vs. reset memory, ordered by
      relative SPL drop.

Replaces the two separate figures fig:goal_vector and fig:shortcut,
saving one figure slot while juxtaposing the two signals that jointly
support H3's content-steering claim:
  - Goal vector is NOT stored in any condition (panel a).
  - Heading-dominant representation is least shortcut-brittle (panel b).

Usage:
    python scripts/paper_figures/make_h3_content_figure.py \\
        --goal-vector data/goal_vector.json \\
        --shortcut-dir data/shortcut \\
        --out docs/NeurIPS_2026/fig/h3_content.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COND_COLS = {
    "blind": "#444444", "uniform": "#4daf4a",
    "foveated": "#e41a1c", "foveated_learned": "#ff7f00",
    "matched": "#377eb8",
}
COND_LABELS = {
    "blind_gibson": "Blind", "uniform_gibson": "Uniform",
    "foveated_gibson": "Fov-fix", "foveated_learned_gibson": "Fov-lrn",
    "matched_gibson": "Coarse",
}
COND_ORDER_GV = ["blind_gibson", "uniform_gibson", "foveated_gibson",
                 "foveated_learned_gibson", "matched_gibson"]

SHORTCUT_FILES = [
    ("blind", "blind_gibson.json", "Blind", "#444444"),
    ("uniform", "uniform_gibson.json", "Uniform", "#4daf4a"),
    ("foveated", "foveated_gibson.json", "Fov-fix", "#e41a1c"),
    ("foveated_learned", "foveated_learned_gibson.json", "Fov-lrn", "#ff7f00"),
    ("matched", "matched_gibson.json", "Coarse", "#377eb8"),
]


def load_goal_vector(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_shortcut(in_dir: Path) -> list[dict]:
    rows = []
    for cond_key, fname, label, colour in SHORTCUT_FILES:
        p = in_dir / fname
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        rows.append({
            "cond": cond_key,
            "label": label,
            "colour": colour,
            "reset": d["reset_mean_spl"],
            "persist": d["persistent_mean_spl"],
            "delta": d["cognitive_map_spl_benefit"],
            "rel_drop": 100 * d["cognitive_map_spl_benefit"] / d["reset_mean_spl"],
        })
    # Sort by ascending relative drop (least-hurt first)
    return sorted(rows, key=lambda r: r["rel_drop"], reverse=True)


def fig_h3_content(gv_path: Path, shortcut_dir: Path, out_path: Path) -> None:
    gv = load_goal_vector(gv_path)
    sc = load_shortcut(shortcut_dir)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 3.4),
                                    gridspec_kw={"width_ratios": [1.0, 1.15],
                                                 "wspace": 0.28})

    # ---------- Panel (a): goal-vector probe ----------
    conds_gv = [c for c in COND_ORDER_GV if c in gv]
    n = len(conds_gv)
    x = np.arange(n)
    w = 0.26

    r2_dist = [gv[c]["goal_dist_r2"]       for c in conds_gv]
    r2_dir  = [gv[c]["goal_direction_r2"]  for c in conds_gv]
    r2_vec  = [gv[c]["goal_vector_r2"]     for c in conds_gv]
    colours = [COND_COLS.get(c.replace("_gibson", ""), "#999") for c in conds_gv]

    # Clamp extremes so bars remain visible and comparable.
    clamp = lambda v: max(v, -0.5)

    axA.bar(x - w, [clamp(v) for v in r2_dist], w,
            color=colours, edgecolor="black", linewidth=0.5,
            label="Goal distance")
    axA.bar(x,     [clamp(v) for v in r2_dir],  w,
            color=colours, edgecolor="black", linewidth=0.5,
            hatch="///", alpha=0.7, label="Goal direction")
    axA.bar(x + w, [clamp(v) for v in r2_vec],  w,
            color=colours, edgecolor="black", linewidth=0.5,
            hatch="\\\\\\", alpha=0.5, label="Goal vector (2D)")

    # Clipped annotations
    for xi, vals in zip(x, zip(r2_dist, r2_dir, r2_vec)):
        for j, v in enumerate(vals):
            if v < -0.5:
                axA.text(xi + (j - 1) * w, -0.48,
                         f"{v:+.1f}", ha="center", va="top",
                         fontsize=7, color="red")

    axA.axhline(0, color="black", linewidth=0.5)
    axA.set_xticks(x)
    axA.set_xticklabels([COND_LABELS[c] for c in conds_gv],
                        fontsize=9, rotation=10)
    axA.set_ylabel("Ridge probe $R^2$", fontsize=9)
    axA.set_ylim(-0.5, 1.05)
    axA.grid(axis="y", linestyle=":", alpha=0.4)
    axA.legend(loc="lower left", fontsize=7, frameon=True, ncol=3,
               columnspacing=1.0, handletextpad=0.5)
    axA.set_title("(a) Goal content in the hidden state",
                  fontsize=9, pad=3)

    # ---------- Panel (b): shortcut discovery ----------
    m = len(sc)
    xb = np.arange(m)
    wb = 0.38

    axB.bar(xb - wb / 2, [r["reset"] for r in sc], wb,
            color=[r["colour"] for r in sc],
            edgecolor="black", linewidth=0.5,
            label="Reset memory")
    axB.bar(xb + wb / 2, [r["persist"] for r in sc], wb,
            color=[r["colour"] for r in sc],
            edgecolor="black", linewidth=0.5,
            hatch="///", alpha=0.7,
            label="Persistent memory")

    for i, r in enumerate(sc):
        y = max(r["reset"], r["persist"]) + 0.03
        axB.text(i, y, f"{r['rel_drop']:+.0f}\\%",
                 ha="center", va="bottom", fontsize=8,
                 fontweight="bold" if abs(r["rel_drop"]) > 30 else "normal")

    axB.set_xticks(xb)
    axB.set_xticklabels([r["label"] for r in sc], fontsize=9, rotation=10)
    axB.set_ylabel("SPL", fontsize=9)
    axB.set_ylim(0, 1.0)
    axB.grid(axis="y", linestyle=":", alpha=0.4)
    axB.legend(loc="upper right", fontsize=7, frameon=True)
    axB.set_title("(b) Shortcut: carrying memory to a new goal",
                  fontsize=9, pad=3)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-vector", type=Path, required=True)
    ap.add_argument("--shortcut-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig_h3_content(args.goal_vector, args.shortcut_dir, args.out)


if __name__ == "__main__":
    main()
