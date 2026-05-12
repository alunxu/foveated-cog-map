"""WJ-F: Excursion-forgetting figure.

Reads:  --in <wjf_segment_v2.json>   (from excursion_analyze_v2.py)
Writes: <out-dir>/figa2c_wjf_excursion.pdf

Bar chart: per-segment MAE/spread for each condition. Highlights the
warmup→recovery rise (forgetting magnitude) per condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()
import numpy as np


COND_DISPLAY = {
    "blind":      ("Blind",          "#444444"),
    "matched128": ("Coarse (1$\\times$1)", "#377eb8"),
    "matched":    ("Matched",        "#888888"),  # not in main results, skip if missing
    "uniform":    ("Uniform",        "#4daf4a"),
    "foveated":   ("Foveated", "#e41a1c"),
}
SEGMENTS = ["warmup", "detour", "recovery"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.in_path.read_text())

    # Order: drop matched128 if missing; keep blind first then alphabetical
    ordered = []
    for k in ["blind", "matched", "uniform", "foveated"]:
        if k in data:
            ordered.append(k)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    n_cond = len(ordered)
    n_seg = len(SEGMENTS)
    bar_w = 0.22
    x = np.arange(n_cond)

    for si, seg_name in enumerate(SEGMENTS):
        vals = [data[c]["segments"].get(seg_name, {}).get("mae_over_spread", np.nan)
                for c in ordered]
        offset = (si - (n_seg - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w,
               label=seg_name,
               color={"warmup": "#4DAF4A", "detour": "#FF7F00",
                      "recovery": "#E41A1C"}[seg_name],
               alpha=0.85, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_DISPLAY[c][0] for c in ordered])
    ax.set_ylabel("MAE / position-spread")
    ax.axhline(1.0, ls="--", color="grey", alpha=0.5, lw=0.7)
    ax.text(n_cond - 0.5, 1.02, "predict-mean baseline",
            color="grey", fontsize=8, ha="right")
    ax.set_ylim(0, max(1.6, max([data[c]["segments"]["recovery"]["mae_over_spread"]
                                  for c in ordered]) * 1.1))
    ax.set_title("Excursion forgetting: per-segment scale-invariant error",
                 loc="left", pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    plt.tight_layout()
    out = args.out_dir / "figa2c_wjf_excursion.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()