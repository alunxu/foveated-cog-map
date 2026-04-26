"""
H1 substitution-mechanism figure: GPS R² across training checkpoints.

Direct mechanistic test of the §4.2 substitution claim. The claim is
that bottleneck conditions force the LSTM to *integrate* the Layer-0
GPS sensor input across time (so a linear top-layer GPS code is
maintained throughout training), while rich-encoder conditions provide
an alternative *visual route* that the policy gradually learns to
exploit, causing the integrated GPS code to fall out of linear top-
layer readability over training.

Reads:  /tmp/ckpt_sweep_data/<cond>_ckpt<N>.json  (analyze.py output)
Writes: docs/NeurIPS_2026/fig/fig3_substitution_dynamics.pdf

Each JSON corresponds to a probing run on a *single training
checkpoint* of one condition; we extract gps_cv_r2_mean and plot vs.
the checkpoint number (proxy for training frames).
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
    # (cond_key,   label,        colour,    marker, frames_per_ckpt_M)
    ("blind",            "Blind (bottleneck)",  "#444444", "o", 10.06),  # 342M / 34
    ("matched",          "Coarse (bottleneck)", "#377eb8", "s", 5.10),   # 250M / 49
    ("uniform",          "Uniform (rich-enc.)", "#4daf4a", "^", 5.10),   # 250M / 49
    ("foveated",         "Foveated (rich-enc.)", "#e41a1c", "D", 4.83),   # 174M / 36
    ("foveated_learned", "Foveated (learned)",  "#ff7f00", "v", 5.10),
]


CLIP_MIN = -2.0  # Anything below this gets clipped (uniform ckpt40 = -6.9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory with <cond>_ckpt<N>.json files")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0, ls=":", color="grey", alpha=0.5, lw=0.7, zorder=0)
    ax.axhspan(CLIP_MIN, 0, color="#ffe5e5", alpha=0.25, zorder=0)  # below-chance

    plotted_anything = False

    for cond_key, label, colour, marker, frames_per_ckpt in CONDS:
        xs_full, ys_full, errs_full = [], [], []
        xs_partial, ys_partial = [], []
        clipped_at = []
        for ck in range(0, 60):
            p = args.data_dir / f"{cond_key}_ckpt{ck}.json"
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            r2 = d.get("1b_global_gps_compass", {}).get("gps_cv_r2_mean")
            std = d.get("1b_global_gps_compass", {}).get("gps_cv_r2_std", 0.0)
            n_ep = d.get("n_episodes", 0)
            if r2 is None:
                continue
            x = ck * frames_per_ckpt
            y = float(np.clip(r2, CLIP_MIN, 1.05))
            if n_ep >= 500:
                xs_full.append(x); ys_full.append(y); errs_full.append(std)
            else:
                xs_partial.append(x); ys_partial.append(y)
            if r2 < CLIP_MIN:
                clipped_at.append((x, r2))

        if xs_full:
            plotted_anything = True
            ax.errorbar(xs_full, ys_full, yerr=errs_full, marker=marker,
                        label=label, color=colour, linewidth=1.6,
                        markersize=6, capsize=2.5, capthick=0.6,
                        elinewidth=0.6)
        if xs_partial:
            plotted_anything = True
            # Hollow / dotted marker for partial-collection data
            ax.plot(xs_partial, ys_partial, marker=marker, ls=":",
                    color=colour, markersize=5, mfc="white", mec=colour,
                    mew=1.2, alpha=0.7,
                    label=None if xs_full else f"{label} (partial)")
        for x, r2 in clipped_at:
            ax.annotate(f"{r2:.1f}", (x, CLIP_MIN + 0.08),
                        fontsize=6.5, ha="center", color="darkred")

    if not plotted_anything:
        ax.text(0.5, 0.5, "(no across-ckpt JSONs found)",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    ax.set_xlabel("training frames (millions)", fontsize=10)
    ax.set_ylabel(r"top-layer GPS $R^2$ (5-fold CV)", fontsize=10)
    ax.set_ylim(CLIP_MIN - 0.10, 1.10)
    ax.set_title(
        "Substitution mechanism: rich-encoder GPS code emerges then dissipates",
        fontsize=10.5,
    )
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(linestyle=":", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=1)

    fig.tight_layout()
    out = args.out_dir / "fig3_substitution_dynamics.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
