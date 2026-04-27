"""
H1 substitution-mechanism figure (2-panel): GPS R² + Compass R² across
training checkpoints.

Direct mechanistic test of the §4.2 substitution claim. The claim is
that bottleneck conditions force the LSTM to *integrate* the Layer-0
GPS / compass sensor inputs across time (so a linear top-layer code is
maintained throughout training), while rich-encoder conditions provide
an alternative *visual route* that the policy gradually learns to
exploit, causing the integrated codes to fall out of linear top-layer
readability over training. Showing both GPS and compass demonstrates
the substitution applies to spatial codes broadly (not just position).

Reads:  /tmp/ckpt_sweep_data/<cond>_ckpt<N>.json  (analyze.py output)
Writes: docs/NeurIPS_2026/fig/fig3_substitution_dynamics.pdf

Each JSON corresponds to a probing run on a *single training
checkpoint* of one condition; we extract {gps,compass}_cv_r2_mean and
plot each vs. the checkpoint number (proxy for training frames).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


CONDS = [
    # (cond_key,   label,        colour,    marker, frames_per_ckpt_M)
    ("blind",            "Blind (bottleneck)",  "#444444", "o", 10.06),  # 342M / 34
    ("matched",          "Coarse (bottleneck)", "#377eb8", "s", 5.10),   # 250M / 49
    ("uniform",          "Uniform (rich-enc.)", "#4daf4a", "^", 5.10),   # 250M / 49
    ("foveated",         "Foveated (rich-enc.)", "#e41a1c", "D", 4.83),   # 174M / 36
    ("foveated_learned", "Foveated (learned)",  "#ff7f00", "v", 5.10),
]


CLIP_MIN = -2.0  # Anything below this gets clipped (uniform ckpt40 = -6.9)


def _plot_panel(ax, args_data_dir, target_key_mean, target_key_std,
                show_legend=False):
    """Draw one across-training panel for a given probe target.

    target_key_mean: e.g. "gps_cv_r2_mean" or "compass_cv_r2_mean"
    target_key_std:  e.g. "gps_cv_r2_std" or "compass_cv_r2_std"
    """
    ax.axhspan(CLIP_MIN, 0, color="#f4d8d4", alpha=0.18, zorder=0)
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8, zorder=1)

    plotted_anything = False
    for cond_key, label, colour, marker, frames_per_ckpt in CONDS:
        xs_full, ys_full, errs_full = [], [], []
        xs_partial, ys_partial = [], []
        clipped_at = []
        for ck in range(0, 60):
            p = args_data_dir / f"{cond_key}_ckpt{ck}.json"
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            r2 = d.get("1b_global_gps_compass", {}).get(target_key_mean)
            std = d.get("1b_global_gps_compass", {}).get(target_key_std, 0.0)
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
                        label=label, color=colour, linewidth=2.2,
                        markersize=8, capsize=3.0, capthick=0.8,
                        elinewidth=0.8, ecolor=colour, alpha=1.0,
                        zorder=4)
        if xs_partial:
            plotted_anything = True
            ax.plot(xs_partial, ys_partial, marker=marker, ls=":",
                    color=colour, markersize=7, mfc="white", mec=colour,
                    mew=1.6, alpha=0.75, linewidth=1.6,
                    label=None if xs_full else f"{label} (partial)",
                    zorder=3)
        for x, r2 in clipped_at:
            ax.annotate(f"{r2:.1f}", (x, CLIP_MIN + 0.06),
                        fontsize=10, fontweight="bold",
                        ha="center", va="bottom",
                        color="darkred", zorder=5)

    if not plotted_anything:
        ax.text(0.5, 0.5, "(no across-ckpt JSONs found)",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    ax.set_ylim(CLIP_MIN - 0.10, 1.10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.25)

    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95),
                  fontsize=10, frameon=True, framealpha=0.92, ncol=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory with <cond>_ckpt<N>.json files")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1×2 layout: GPS (left) + Compass (right). Same x-axis range, same
    # y-axis range, sharey for tight layout. Legend lives only in
    # panel (a) since both panels show the same 5 conditions.
    fig, (ax_gps, ax_comp) = plt.subplots(
        1, 2, figsize=(11.0, 3.7), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    _plot_panel(ax_gps, args.data_dir,
                "gps_cv_r2_mean", "gps_cv_r2_std",
                show_legend=True)
    _plot_panel(ax_comp, args.data_dir,
                "compass_cv_r2_mean", "compass_cv_r2_std",
                show_legend=False)

    ax_gps.set_xlabel("training frames (M)", fontsize=12, fontweight="bold")
    ax_comp.set_xlabel("training frames (M)", fontsize=12, fontweight="bold")
    ax_gps.set_ylabel("top-layer probe $R^2$ (5-fold CV)",
                      fontsize=12, fontweight="bold")
    # Right panel: y-label dropped (sharey), but make sure ticks visible
    ax_gps.set_title("(a) GPS code",
                     fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax_comp.set_title("(b) Compass (heading) code",
                      fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)

    fig.suptitle("Substitution mechanism in rich-encoder agents",
                 fontsize=13, fontweight="bold", y=1.04)

    fig.tight_layout()
    out = args.out_dir / "fig3_substitution_dynamics.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
