"""
Replacement for the 5-bar shortcut figure: a 2-D scatter that
visualises the central claim of §4.5 (H3) — probe-readable GPS code
(from H1) vs behavioural memory reliance (shortcut SPL drop) can
dissociate.  With 5 conditions, a scatter is more compact than bars
and surfaces the 2×2 dissociation pattern (matched vs uniform are the
two opposite anomalies).

x-axis: GPS R^2 (no-cap, 5-fold CV, from <results-dir>/<cond>_gibson_det_analysis.json)
y-axis: shortcut SPL drop (%) = -(persistent - reset) / reset
        from data/shortcut/<cond>_gibson.json

Writes: <out-dir>/shortcut_scatter.{pdf,png}

Usage:
    python scripts/paper_figures/make_shortcut_scatter.py \\
        --results-dir /tmp/probing_results_local \\
        --shortcut-dir data/shortcut \\
        --out-dir docs/NeurIPS_2026/fig
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
    # (json_key,            short_label,         colour,    marker)
    ("blind",               "Blind",             "#444444", "o"),
    ("matched",             "Matched (1×1)",     "#377eb8", "s"),
    ("uniform",             "Uniform",           "#4daf4a", "^"),
    ("foveated",            "Foveated (fix)",    "#e41a1c", "D"),
    ("foveated_learned",    "Fov-learned",       "#ff7f00", "v"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--shortcut-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--x-clip-min", type=float, default=-1.5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cond_key, label, colour, marker in CONDS:
        gp = args.results_dir / f"{cond_key}_gibson_det_analysis.json"
        sp = args.shortcut_dir / f"{cond_key}_gibson.json"
        if not (gp.exists() and sp.exists()):
            print(f"[skip] {cond_key} (missing files)")
            continue
        gd = json.loads(gp.read_text())
        sd = json.loads(sp.read_text())
        gps_r2 = gd["1b_global_gps_compass"]["gps_cv_r2_mean"]
        reset = sd["reset_mean_spl"]
        persist = sd["persistent_mean_spl"]
        drop_pct = 100.0 * (reset - persist) / reset if reset > 0 else 0.0
        rows.append({
            "label": label, "colour": colour, "marker": marker,
            "gps_r2": gps_r2, "drop_pct": drop_pct,
        })

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    # Quadrant shading: 4 regions formed by GPS R² = 0.5 vertical line
    # and shortcut-drop = 30% horizontal line.  Pure aesthetic to read
    # the 2×2 dissociation off the figure.
    ax.axvspan(0.5, 1.05, ymin=0, ymax=1.0, color="#bcd4ec",
               alpha=0.18, zorder=0)
    ax.axvspan(args.x_clip_min - 0.05, 0.5, ymin=0, ymax=1.0,
               color="#dddddd", alpha=0.25, zorder=0)
    ax.axhline(30, ls=":", color="grey", alpha=0.5, lw=0.6)
    ax.axvline(0.5, ls=":", color="grey", alpha=0.5, lw=0.6)

    # Plot points + labels
    for r in rows:
        x = max(r["gps_r2"], args.x_clip_min)
        y = r["drop_pct"]
        clipped = r["gps_r2"] < args.x_clip_min
        ax.scatter(x, y, s=140, c=r["colour"], edgecolor="black",
                   linewidths=0.8, marker=r["marker"], zorder=3)
        # Label offset to avoid marker; manual nudge per point so
        # labels do not overlap.
        offsets = {
            "Blind":          (-0.06, +1.0),  # left of dot, top-right area
            "Matched (1×1)":  (-0.06, +1.5),  # left of dot
            "Uniform":        (+0.06, +1.0),  # right of dot
            "Foveated (fix)": (+0.06, +1.0),  # right of dot
            "Fov-learned":    (+0.12, +0.0),  # right of clipped dot
        }
        dx, dy = offsets.get(r["label"], (0.06, 0.0))
        ha = "right" if dx < 0 else "left"
        if clipped:
            ax.annotate(
                f"{r['label']}\n(GPS $R^2$={r['gps_r2']:.1f})",
                (x, y), xytext=(x + dx, y + dy),
                ha=ha, va="center", fontsize=8.5,
            )
            # Arrow indicating x-axis was clipped
            ax.annotate("←", (x - 0.05, y), ha="right", va="center",
                        fontsize=11, color="darkred")
        else:
            ax.annotate(r["label"], (x, y), xytext=(x + dx, y + dy),
                        ha=ha, va="center", fontsize=8.5)

    # Background-region labels (just the column headers, no quadrant
    # text — quadrant structure is conveyed by shading + caption.)
    ax.text(0.755, 0.96, "has linear\nGPS code", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#1f4e8a",
            style="italic", alpha=0.75)
    ax.text(0.25, 0.96, "no linear GPS",
            transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#444",
            style="italic", alpha=0.75)

    ax.set_xlabel("GPS $R^2$ (probe-readable from top-layer h, 5-fold CV)",
                  fontsize=9.5)
    ax.set_ylabel("Shortcut SPL drop (\\%, larger = more memory-reliant)",
                  fontsize=9.5)
    ax.set_xlim(args.x_clip_min - 0.05, 1.05)
    ax.set_ylim(0, 60)
    ax.tick_params(axis="both", labelsize=8.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out_dir / f"shortcut_scatter.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
