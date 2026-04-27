"""
WJ-A: memory-length-budget sweep figure (Wijmans Fig 2 replication).

For each condition, plots top-layer GPS $R^2$ as a function of the LSTM's
effective memory length K (number of within-episode steps over which the
LSTM accumulates history before its hidden state is reset). At K=1 the
hidden state contains only the current step; at K large it carries the
agent's full episode history.

If the H1 mechanism story is right (bottleneck conditions need long-
horizon GPS-sensor integration; rich-encoder conditions don't), we
expect:
  - Bottleneck (blind, coarse): GPS R^2 grows monotonically with K and
    only saturates at K~hundreds. Long memory matters.
  - Rich-encoder (uniform, foveated): GPS R^2 stays low at every K.
    Memory length doesn't help because the LSTM isn't using sensor
    integration to encode position in the first place.

Reads:  --analysis-dir <dir>/{cond}_k{K}_det_analysis.json (analyze.py
        output for memory-length-restricted hidden-state collections)
Writes: <out-dir>/fig_memlen_sweep.pdf
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
    ("blind",      "Blind",          "#444444", "o"),
    ("matched128", "Coarse (1$\\times$1)", "#377eb8", "s"),
    ("uniform",    "Uniform",        "#4daf4a", "^"),
    ("foveated",   "Foveated (fix)", "#e41a1c", "D"),
]
KS = [1, 4, 16, 64, 256, 1000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, ls=":", color="grey", alpha=0.4, lw=0.7)

    for cond_key, label, colour, marker in CONDS:
        ks_x, r2s, r2_stds = [], [], []
        for K in KS:
            p = args.analysis_dir / f"{cond_key}_k{K}_det_analysis.json"
            if not p.exists():
                ks_x.append(K); r2s.append(np.nan); r2_stds.append(np.nan)
                continue
            d = json.loads(p.read_text())
            # analyze.py output convention: top-level "gps" key with
            # {r2_cv_mean, r2_cv_std} or similar.
            gps = d.get("gps", d.get("gps_cv_r2_mean"))
            if isinstance(gps, dict):
                r2 = float(gps.get("r2_cv_mean", gps.get("r2_mean", np.nan)))
                std = float(gps.get("r2_cv_std", gps.get("r2_std", 0)))
            else:
                r2 = float(gps) if gps is not None else np.nan
                std = 0.0
            ks_x.append(K); r2s.append(r2); r2_stds.append(std)
        ks_x = np.array(ks_x); r2s = np.array(r2s); r2_stds = np.array(r2_stds)

        # Clip extremely negative R² for plot readability (log-scale x).
        r2_clip = np.clip(r2s, -1.5, 1.05)
        ax.errorbar(ks_x, r2_clip, yerr=r2_stds,
                    marker=marker, label=label, color=colour,
                    linewidth=1.8, markersize=7, capsize=3, alpha=0.95)

    ax.set_xscale("log")
    ax.set_xlabel("LSTM memory budget $K$ (steps)")
    ax.set_ylabel("Top-layer GPS $R^2$ (5-fold CV)")
    ax.set_xticks([1, 4, 16, 64, 256, 1000])
    ax.set_xticklabels(["1", "4", "16", "64", "256", "1000"])
    ax.set_ylim(-1.55, 1.10)
    ax.set_title("Memory-length budget sweep: how much history each condition needs",
                 loc="left", pad=8)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    ax.grid(linestyle=":", alpha=0.3)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    plt.tight_layout()
    out = args.out_dir / "fig_memlen_sweep.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
