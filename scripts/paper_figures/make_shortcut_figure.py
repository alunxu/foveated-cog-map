"""
Shortcut-discovery figure for §4 behavioral results:
  - shortcut_bars.pdf: reset-memory vs. persistent-memory SPL per condition,
    with ΔSPL annotation (persistent − reset; negative = memory hurts).

A new start location in a previously-visited scene means the episode-start
memory is misaligned with the current goal. The SPL drop therefore measures
how much the agent's cognitive map locks into the *old* episode at the expense
of the new one: larger drop = more environment-specific caching.

Usage:
    python scripts/paper_figures/make_shortcut_figure.py \
        --in-dir data/shortcut --out-dir docs/NeurIPS_2026/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COND_DISPLAY = {
    "blind":            ("Blind",              "#444444"),
    "uniform":          ("Uniform",            "#4daf4a"),
    "foveated":         ("Foveated (fix)",   "#e41a1c"),
    "foveated_learned": ("Foveated (learned)", "#ff7f00"),
    "matched":          ("Matched-compute",    "#377eb8"),
}
COND_ORDER = ["blind", "uniform", "foveated", "foveated_learned", "matched"]


def _load(in_dir: Path, cond: str) -> dict | None:
    p = in_dir / f"{cond}_gibson.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fig_shortcut_bars(in_dir: Path, out_path: Path) -> None:
    rows = []
    for c in COND_ORDER:
        d = _load(in_dir, c)
        if d is None:
            continue
        rows.append({
            "cond": c,
            "label": COND_DISPLAY[c][0],
            "colour": COND_DISPLAY[c][1],
            "reset": d["reset_mean_spl"],
            "persist": d["persistent_mean_spl"],
            "delta": d["cognitive_map_spl_benefit"],
        })

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    n = len(rows)
    x = np.arange(n)
    w = 0.38

    ax.bar(x - w/2, [r["reset"] for r in rows], w,
           color=[r["colour"] for r in rows], edgecolor="black", linewidth=0.6,
           label="Reset memory each episode")
    ax.bar(x + w/2, [r["persist"] for r in rows], w,
           color=[r["colour"] for r in rows], edgecolor="black", linewidth=0.6,
           hatch="///", alpha=0.7,
           label="Persistent memory (new goal)")

    # ΔSPL annotations
    for i, r in enumerate(rows):
        y = max(r["reset"], r["persist"]) + 0.03
        ax.text(i, y, f"$\\Delta$={r['delta']:+.2f}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold" if abs(r["delta"]) > 0.15 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9, rotation=10)
    ax.set_ylabel("SPL", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_title("Carrying memory across episodes hurts more where maps are environment-specific",
                 fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    # Text dump
    print(f"\n{'Condition':<22} {'reset SPL':<10} {'persist SPL':<12} {'ΔSPL':<8} {'rel%':<6}")
    for r in rows:
        rel = 100 * r["delta"] / r["reset"] if r["reset"] > 0 else 0.0
        print(f"{r['label']:<22} {r['reset']:<10.3f} {r['persist']:<12.3f} "
              f"{r['delta']:<+8.3f} {rel:<+6.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_shortcut_bars(args.in_dir, args.out_dir / "shortcut_bars.pdf")


if __name__ == "__main__":
    main()
