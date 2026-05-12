"""
B3: Subspace evolution over training (figure).

Reads cluster-side analysis output (subspace_evolution.json) and plots:
  - Panel (a): principal angle (deg) per cond pair × training stage
  - Panel (b): position-direction cosine per cond pair × training stage

Predicts: subspaces DIVERGE during training. Early ckpts → low angles,
high cos (overlap). Late ckpts → ~90°, ~0 cos (orthogonal). This
provides DYNAMIC evidence for capacity-allocation as a training-time
process, complementing the snapshot in fig_subspace_divergence.

Reads:  /tmp/subspace_evolution.json (synced from Izar)
Writes: docs/manuscript/fig/figa7c_subspace_evolution.pdf
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


# Cond display names + colours
COND_DISPLAY = {
    "blind":    ("Blind",    "#444444"),
    "matched":  ("Coarse",   "#377eb8"),
    "foveated": ("Foveated", "#e41a1c"),
    "uniform":  ("Uniform",  "#4daf4a"),
}

# Pair colours
PAIR_COLORS = {
    ("blind", "matched"):  "#7a3a7a",
    ("blind", "foveated"): "#a13838",
    ("blind", "uniform"):  "#3a7a3a",
    ("foveated", "matched"): "#c46a2a",
    ("matched", "uniform"):  "#377e7e",
    ("foveated", "uniform"): "#bb6666",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.in_json).read_text())
    rows = data["rows"]
    print(f"Loaded {len(rows)} pair entries")

    # For each pair (cA, cB), gather points where frames_A and frames_B are similar
    # Use diagonal matching: for each pair, pick (kA, kB) where frames are closest
    pair_data: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    # Group by pair
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cA, cB = r["cond_A"], r["cond_B"]
        # Normalize pair order (alphabetic)
        if cA > cB:
            cA, cB = cB, cA
            r = dict(r); r["cond_A"], r["cond_B"] = cA, cB
            r["frames_A_M"], r["frames_B_M"] = r["frames_B_M"], r["frames_A_M"]
            r["ckpt_A"], r["ckpt_B"] = r["ckpt_B"], r["ckpt_A"]
        by_pair[(cA, cB)].append(r)

    # For each pair, build a trajectory by matching ckpts at similar frames.
    # Strategy: use the smaller cond's ckpt index as anchor, find best-matching
    # ckpt of the other cond. Average frames as the x.
    for pair, entries in by_pair.items():
        # Bucket by ckpt_A (or by frames)
        # Simple strategy: take entries where ckpt_A == ckpt_B as "matched"
        # If different ckpt schedules, fall back to closest-frames pairing.
        matched = [(e["frames_A_M"], e["principal_angle_deg"], e["pos_dir_cos"])
                   for e in entries if e["ckpt_A"] == e["ckpt_B"]]
        if not matched:
            # Fallback: closest frames
            matched = sorted(
                [(0.5 * (e["frames_A_M"] + e["frames_B_M"]),
                  e["principal_angle_deg"], e["pos_dir_cos"])
                 for e in entries],
                key=lambda t: t[0],
            )
        matched = sorted(set(matched), key=lambda t: t[0])
        pair_data[pair] = matched

    # Plot
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                     gridspec_kw={"wspace": 0.28})

    for pair, points in pair_data.items():
        cA, cB = pair
        if not points:
            continue
        xs = [p[0] for p in points]
        angles = [p[1] for p in points]
        coses = [p[2] for p in points]
        nameA, _ = COND_DISPLAY[cA]
        nameB, _ = COND_DISPLAY[cB]
        label = f"{nameA}--{nameB}"
        color = PAIR_COLORS.get(pair, PAIR_COLORS.get((pair[1], pair[0]), "#888888"))
        ax_a.plot(xs, angles, marker="o", linewidth=1.8, markersize=6,
                  color=color, label=label, alpha=0.9)
        ax_b.plot(xs, coses, marker="o", linewidth=1.8, markersize=6,
                  color=color, label=label, alpha=0.9)

    # Reference lines: orthogonal at 90° / random at ~0
    ax_a.axhline(90, ls="--", color="grey", alpha=0.5, lw=0.8)
    ax_b.axhline(0, ls="--", color="grey", alpha=0.5, lw=0.8)

    ax_a.set_xlabel("training frames (M)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Principal angle (deg)", fontsize=11, fontweight="bold")
    ax_a.set_title("(a) Subspace angle (zoomed)",
                   fontsize=11.5, loc="left", pad=8, fontweight="bold")
    ax_a.set_ylim(78, 92)
    # Reposition orthogonal annotation
    ax_a.text(0.98, 89.7, "orthogonal (90°)", transform=ax_a.get_yaxis_transform(),
              ha="right", va="center", fontsize=9, color="grey")
    ax_a.grid(linestyle=":", alpha=0.3)
    for s_ in ("top", "right"):
        ax_a.spines[s_].set_visible(False)

    ax_b.set_xlabel("training frames (M)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("cos$(\\beta_A,\\beta_B)$", fontsize=11, fontweight="bold")
    ax_b.set_title("(b) Position-direction alignment (full range)",
                   fontsize=11.5, loc="left", pad=8, fontweight="bold")
    ax_b.set_ylim(-1.0, 1.0)
    ax_b.grid(linestyle=":", alpha=0.3)
    for s_ in ("top", "right"):
        ax_b.spines[s_].set_visible(False)
    ax_b.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)

    fig.suptitle("Subspace evolution: capacity allocation established early ($\\sim 50$M frames), refined to maximal orthogonality at convergence",
                 fontsize=12.0, fontweight="bold", y=1.04)

    plt.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
