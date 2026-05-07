"""
Supplementary figure for the cell state c_t and per-layer h_t probing.

Shows GPS R² across 6 (state × layer) representations × 5 conditions.
Resolves §3.2 fn4 / MASTER_TRACK §3.0b O2 (top-layer h_2 only probed
in main results).

Reads:  data/extra_states/<cond>_extra_states.json (from
        scripts/probing/analyze_extra_states.py)
Writes: docs/manuscript/fig/figa3_extra_states.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()
import numpy as np


CONDS = [
    ("blind",             "Blind",    "#444444", "o"),
    ("coarse",            "Coarse",   "#377eb8", "s"),
    ("foveated_logpolar", "Fov-LP",   "#984ea3", "v"),
    ("foveated",          "Foveated", "#e41a1c", "D"),
    ("uniform",           "Uniform",  "#4daf4a", "^"),
]
CLIP_MIN = -1.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6),
                             gridspec_kw={"wspace": 0.25})

    for ax, target in zip(axes, ["gps", "compass"]):
        # x positions: h0, h1, h2, c0, c1, c2
        x_labels = [r"$\mathbf{h}_0$", r"$\mathbf{h}_1$", r"$\mathbf{h}_2$",
                    r"$\mathbf{c}_0$", r"$\mathbf{c}_1$", r"$\mathbf{c}_2$"]
        x_pos = list(range(6))

        # Visual separator h vs c
        ax.axvspan(2.5, 5.5, color="#f5f5f5", alpha=0.5, zorder=0)
        ax.axvline(2.5, color="#999", linestyle="--", lw=0.5, zorder=0)
        ax.axhline(0, ls=":", color="grey", alpha=0.5, lw=0.7, zorder=0)

        for cond_key, label, colour, marker in CONDS:
            p = args.data_dir / f"{cond_key}_extra_states.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            ys = []
            for state in ["h", "c"]:
                for layer in [0, 1, 2]:
                    key = f"{state}_layer{layer}_{target}"
                    val = d.get(key, {}).get("r2_mean")
                    ys.append(np.clip(val, CLIP_MIN, 1.05) if val is not None
                              else np.nan)
            xs_ord = [0, 1, 2, 3, 4, 5]
            ax.plot(xs_ord, ys, marker=marker, label=label,
                    color=colour, linewidth=1.6, markersize=6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel(f"{target.upper()} $R^2$ (5-fold CV)", fontsize=9)
        ax.set_xlim(-0.4, 5.4)
        ax.set_ylim(CLIP_MIN - 0.05, 1.10)
        ax.tick_params(axis="y", labelsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        # Region labels
        ax.text(1.0, 1.04, "hidden state $\\mathbf{h}$",
                ha="center", fontsize=8, style="italic", color="#444",
                transform=ax.transData)
        ax.text(4.0, 1.04, "cell state $\\mathbf{c}$",
                ha="center", fontsize=8, style="italic", color="#444",
                transform=ax.transData)
        ax.set_title(f"{target.upper()} probe across LSTM hidden + cell states",
                     fontsize=10)

    axes[0].legend(loc="lower right", fontsize=8, frameon=False, ncol=1)

    plt.tight_layout()
    out = args.out_dir / "figa3_extra_states.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()