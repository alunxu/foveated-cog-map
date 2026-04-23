"""
Transplant midpoint-sweep figure.

Replaces the single-midpoint Figure 2 (left) with a curve showing how
SPL depends on transplant midpoint. Key illustrations:
  1. self-transplant drift grows with midpoint → rollout divergence is
     real, not a protocol artefact.
  2. cross-self (right-axis, inset) isolates the pure condition-mismatch
     component and is the cleaner metric to report.

Usage:
    python scripts/paper_figures/make_transplant_sweep_figure.py \\
        --in-dir data/transplant --out-dir docs/NeurIPS_2026/fig
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAIRS = [
    ("foveated_to_uniform",          "Fov$\\rightarrow$Uniform",      "#d95f02"),
    ("foveated_to_blind",            "Fov$\\rightarrow$Blind",        "#7570b3"),
    ("foveated_learned_to_foveated", "FovLrn$\\rightarrow$Fov",       "#1b9e77"),
]
DEFAULT_MID = 30


def load_pair(in_dir: Path, stem: str) -> list[dict]:
    """Gather (midpoint, baseline_spl, self_spl, cross_spl) for this pair."""
    out = []
    pattern = re.compile(rf"^{re.escape(stem)}(?:_mid(\d+))?\.json$")
    for p in sorted(in_dir.iterdir()):
        m = pattern.match(p.name)
        if not m:
            continue
        mid = int(m.group(1)) if m.group(1) else DEFAULT_MID
        with open(p) as f:
            d = json.load(f)
        out.append({
            "midpoint": mid,
            "baseline": d["baseline"]["mean_spl"],
            "self": d["self_transplant"]["mean_spl"],
            "cross": d["cross_transplant"]["mean_spl"],
            "n": d["n_episodes"],
        })
    return sorted(out, key=lambda r: r["midpoint"])


def fig_sweep(in_dir: Path, out_path: Path) -> None:
    rows_by_pair = {stem: load_pair(in_dir, stem) for stem, _, _ in PAIRS}

    fig, (ax_spl, ax_delta) = plt.subplots(1, 2, figsize=(9.5, 3.4),
                                           gridspec_kw={"wspace": 0.32})

    # Left: SPL triad (baseline / self / cross) vs midpoint
    for stem, label, colour in PAIRS:
        rows = rows_by_pair[stem]
        if not rows:
            continue
        x = [r["midpoint"] for r in rows]
        ax_spl.plot(x, [r["baseline"] for r in rows], "o-", color=colour,
                    alpha=0.3, linewidth=1, markersize=5,
                    label=f"{label} baseline")
        ax_spl.plot(x, [r["self"] for r in rows], "s--", color=colour,
                    alpha=0.7, linewidth=1.2, markersize=5,
                    label=f"{label} self")
        ax_spl.plot(x, [r["cross"] for r in rows], "^-", color=colour,
                    alpha=1.0, linewidth=1.8, markersize=6,
                    label=f"{label} cross")
    ax_spl.set_xlabel("Transplant midpoint (steps)", fontsize=9)
    ax_spl.set_ylabel("SPL", fontsize=9)
    ax_spl.grid(linestyle=":", alpha=0.4)
    ax_spl.set_ylim(0, 1.0)
    ax_spl.legend(fontsize=6, loc="lower left", ncol=3, frameon=True)
    ax_spl.set_title("(a) Full SPL per transplant condition", fontsize=9)

    # Right: cross−self = pure condition-mismatch effect
    for stem, label, colour in PAIRS:
        rows = rows_by_pair[stem]
        if not rows:
            continue
        x = [r["midpoint"] for r in rows]
        y = [r["cross"] - r["self"] for r in rows]
        ax_delta.plot(x, y, "o-", color=colour, linewidth=1.8, markersize=6,
                      label=label)
    ax_delta.axhline(0, color="black", linewidth=0.5)
    ax_delta.set_xlabel("Transplant midpoint (steps)", fontsize=9)
    ax_delta.set_ylabel("Cross$-$Self SPL", fontsize=9)
    ax_delta.grid(linestyle=":", alpha=0.4)
    ax_delta.legend(fontsize=8, loc="lower left", frameon=True)
    ax_delta.set_title("(b) Pure condition-mismatch effect", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    # Text dump
    print("\nDump for paper:")
    for stem, label, _ in PAIRS:
        rows = rows_by_pair[stem]
        print(f"\n{label} ({stem}):")
        print(f"  {'mid':<5} {'base':<7} {'self':<7} {'cross':<7} {'cross-self':<10}")
        for r in rows:
            cs = r["cross"] - r["self"]
            print(f"  {r['midpoint']:<5} {r['baseline']:<7.3f} {r['self']:<7.3f} "
                  f"{r['cross']:<7.3f} {cs:<+10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_sweep(args.in_dir, args.out_dir / "transplant_sweep.pdf")


if __name__ == "__main__":
    main()
