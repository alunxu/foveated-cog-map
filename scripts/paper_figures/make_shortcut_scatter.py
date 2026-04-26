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
    ("matched",             "Coarse (1×1)",     "#377eb8", "s"),
    ("uniform",             "Uniform",           "#4daf4a", "^"),
    ("foveated",            "Foveated (fix)",    "#e41a1c", "D"),
    ("foveated_learned",    "Foveated (learned)",       "#ff7f00", "v"),
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

    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    # 2x2 quadrants formed by GPS R² = 0.4 vertical and shortcut drop
    # = 30% horizontal. No background shading; just clean dashed lines.
    GPS_THRESH = 0.4
    DROP_THRESH = 30.0
    ax.axhline(DROP_THRESH, ls="--", color="#888888", lw=0.9, zorder=1)
    ax.axvline(GPS_THRESH, ls="--", color="#888888", lw=0.9, zorder=1)

    # Plot points + labels
    for r in rows:
        x = max(r["gps_r2"], args.x_clip_min)
        y = r["drop_pct"]
        clipped = r["gps_r2"] < args.x_clip_min
        ax.scatter(x, y, s=160, c=r["colour"], edgecolor="black",
                   linewidths=0.9, marker=r["marker"], zorder=3)
        # Label offset to avoid marker overlap.
        offsets = {
            "Blind":          (-0.08, +0.0),
            "Coarse (1×1)":  (-0.08, +0.0),
            "Uniform":        (+0.08, +0.0),
            "Foveated (fix)": (+0.08, +0.0),
            "Foveated (learned)":    (+0.10, -3.5),
        }
        dx, dy = offsets.get(r["label"], (0.08, 0.0))
        ha = "right" if dx < 0 else "left"
        label = r["label"]
        if clipped:
            label = f"{label} ($R^2{{=}}{r['gps_r2']:.1f}$)"
        ax.annotate(label, (x, y), xytext=(x + dx, y + dy),
                    ha=ha, va="center", fontsize=9, fontweight="bold")

    # Quadrant captions (corners of the 2x2)
    quad_kw = dict(transform=ax.transAxes, fontsize=8, color="#444",
                   style="italic", alpha=0.85)
    ax.text(0.97, 0.97, "has GPS,\nuses memory",
            ha="right", va="top", **quad_kw)
    ax.text(0.03, 0.97, "no GPS,\nuses memory",
            ha="left", va="top", **quad_kw)
    ax.text(0.97, 0.03, "has GPS,\nignores it",
            ha="right", va="bottom", **quad_kw)
    ax.text(0.03, 0.03, "no GPS,\nlow memory",
            ha="left", va="bottom", **quad_kw)

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
