"""
Memory-transplant figure for §4.4 H2:
  - transplant_bars.pdf: baseline / self-transplant / cross-transplant SPL
    grouped by (donor, recipient) pair.
  - transplant_delta.pdf: cross-baseline SPL delta summary bar chart.

Usage:
    python scripts/paper_figures/make_transplant_figure.py \
        --in-dir data/transplant --out-dir docs/manuscript/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (filename, display label, donor colour, recipient colour)
PAIRS = [
    ("foveated_to_blind.json",
     "Foveated$\\rightarrow$Blind",
     "#e41a1c", "#444444"),
    ("foveated_to_uniform.json",
     "Foveated$\\rightarrow$Uniform",
     "#e41a1c", "#4daf4a"),
]

COLS = {
    "baseline": "#bdbdbd",        # light grey
    "self":     "#7570b3",        # purple
    "cross":    "#d95f02",        # orange
}


def _load(in_dir: Path) -> list[dict]:
    out = []
    for fname, label, dc, rc in PAIRS:
        p = in_dir / fname
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        with open(p) as f:
            d = json.load(f)
        out.append({
            "label": label,
            "donor_colour": dc,
            "recipient_colour": rc,
            "baseline": d["baseline"]["mean_spl"],
            "baseline_succ": d["baseline"]["success_rate"],
            "self": d["self_transplant"]["mean_spl"],
            "self_succ": d["self_transplant"]["success_rate"],
            "cross": d["cross_transplant"]["mean_spl"],
            "cross_succ": d["cross_transplant"]["success_rate"],
            "n": d["n_episodes"],
            "midpoint": d["midpoint_step"],
        })
    return out


def fig_grouped_bars(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.4))

    n = len(rows)
    x = np.arange(n)
    w = 0.26

    bvals = [r["baseline"] for r in rows]
    svals = [r["self"] for r in rows]
    cvals = [r["cross"] for r in rows]

    ax.bar(x - w, bvals, w, color=COLS["baseline"],
           edgecolor="black", linewidth=0.6, label="Baseline (recipient only)")
    ax.bar(x,     svals, w, color=COLS["self"],
           edgecolor="black", linewidth=0.6, label="Self-transplant")
    ax.bar(x + w, cvals, w, color=COLS["cross"],
           edgecolor="black", linewidth=0.6, label="Cross-transplant")

    # Annotate deltas above the cross bars.
    for i, r in enumerate(rows):
        delta = r["cross"] - r["baseline"]
        y = max(r["baseline"], r["self"], r["cross"]) + 0.03
        ax.text(i + w, y, f"$\\Delta$={delta:+.2f}",
                ha="center", va="bottom", fontsize=8,
                color="black",
                fontweight="bold" if abs(delta) > 0.1 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_ylabel("SPL", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_title("Memory transplant: donor hidden state pasted into recipient at step 30",
                 fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(args.in_dir)
    if not rows:
        print("no data found")
        return
    fig_grouped_bars(rows, args.out_dir / "transplant_bars.pdf")

    # Quick text dump for paper.
    print("\nSummary for paper:")
    print(f"{'pair':<35} {'baseline':<10} {'self':<10} {'cross':<10} {'Δ(cross-base)':<14}")
    for r in rows:
        d = r["cross"] - r["baseline"]
        print(f"{r['label']:<35} {r['baseline']:<10.3f} {r['self']:<10.3f} "
              f"{r['cross']:<10.3f} {d:<+14.3f}")


if __name__ == "__main__":
    main()