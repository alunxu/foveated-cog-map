"""
Temporal-stability figure: GPS / compass R^2 per episode-step bin, all 5
conditions.

Shows the qualitative encoder--memory race dynamic that the existing
no-cap and length-matched probes hint at:
  - Bottleneck conditions (blind, matched) maintain top-layer GPS code
    stably across episode duration (R^2 > 0.7 even at step 800+).
  - Rich-encoder conditions (uniform, foveated, foveated-learned) encode
    GPS in a typical-episode window (steps ~50-200, R^2 0.6-0.85) and
    then OVERWRITE it as the episode progresses; by step 400+ they
    decode at chance and by 800+ are catastrophically negative.

Reads: <results-dir>/temporal_probe_det.json (produced by
temporal_probe.py).
Writes: <out-dir>/temporal_probe_evolution.{pdf,png}

Usage:
    python scripts/paper_figures/make_temporal_probe_figure.py \\
        --in /scratch/izar/wxu/probing_results/temporal_probe_det.json \\
        --out-dir docs/manuscript/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDS = [
    ("blind_gibson",            "Blind",            "#444444", "o"),
    ("matched_gibson",          "Coarse (1×1)",    "#377eb8", "s"),
    ("uniform_gibson",          "Uniform",          "#4daf4a", "^"),
    ("foveated_gibson",         "Foveated (fix)",   "#e41a1c", "D"),
    ("foveated_learned_gibson", "Foveated (learned)",      "#ff7f00", "v"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clip-min", type=float, default=-1.5,
                    help="Clip R² values below this for visibility")
    args = ap.parse_args()

    data = json.loads(args.in_path.read_text())
    per_cond = data["per_condition"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=False)
    for ax, target_key, ylabel in [
        (axes[0], "gps_r2_per_bin",     "GPS $R^2$"),
        (axes[1], "compass_r2_per_bin", "Compass $R^2$"),
    ]:
        for cond_key, label, color, marker in CONDS:
            if cond_key not in per_cond:
                continue
            bins = per_cond[cond_key][target_key]
            xs = []
            ys = []
            for b in bins:
                if b["r2"] is None:
                    continue
                # X = bin midpoint (log-ish)
                xmid = (b["lo"] + min(b["hi"], 1600)) / 2.0
                xs.append(xmid)
                ys.append(max(b["r2"], args.clip_min))
            ax.plot(xs, ys, marker=marker, label=label,
                    color=color, linewidth=2, markersize=6)
        ax.axhline(0, ls=":", color="grey", alpha=0.5, lw=0.8)
        ax.set_xlabel("step in episode (bin midpoint)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_ylim(args.clip_min - 0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend on the right axis
    axes[1].legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle(
        "Top-layer GPS / compass code stability across episode duration",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = args.out_dir / f"temporal_probe_evolution.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
