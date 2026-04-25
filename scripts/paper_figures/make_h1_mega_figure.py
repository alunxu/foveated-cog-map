"""
H1 mega-figure for the consolidated Results section.  Three panels.

Panels:
  (a) Grouped bars: GPS $R^2$ across the 5 conditions, on Gibson (no-cap
      5-fold CV) and held-out MP3D (same checkpoints, no re-training).
      Combines what was previously two separate panels (current-state
      bars + MP3D robustness) into a single grouped bar chart.  The
      compass and DtG numbers are in Table~\ref{tab:summary} and an
      appendix companion figure.
  (b) Top-layer GPS code stability across episode duration (line plot).
  (c) Per-LSTM-layer GPS R² (line plot) — Layer-0 disambiguation.

Reads:
  - <results-dir>/{cond}_{gibson,mp3d}_det_analysis.json     (panel a)
  - <results-dir>/temporal_probe_det.json                    (panel b)
  - <results-dir>/{cond}_gibson_det_analysis.json            (panel c)

Writes: <out-dir>/h1_mega.{pdf,png}

Usage:
    python scripts/paper_figures/make_h1_mega_figure.py \\
        --results-dir /tmp/probing_results_local \\
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

# ─── Shared condition styling ─────────────────────────────────────────
CONDS = [
    # (json_key,           short_label,         colour,    marker)
    ("blind",              "Blind",             "#444444", "o"),
    ("matched",            "Matched (1×1)",     "#377eb8", "s"),
    ("uniform",            "Uniform",           "#4daf4a", "^"),
    ("foveated",           "Foveated (fix)",    "#e41a1c", "D"),
    ("foveated_learned",   "Fov-learned",       "#ff7f00", "v"),
]
CLIP_MIN = -1.5


def panel_a(ax, results_dir: Path) -> None:
    """Grouped bar chart: GPS R^2 × {Gibson, MP3D} × 5 conditions."""
    rows = []
    for cond_key, label, colour, _ in CONDS:
        gp = results_dir / f"{cond_key}_gibson_det_analysis.json"
        mp = results_dir / f"{cond_key}_mp3d_det_analysis.json"
        if not (gp.exists() and mp.exists()):
            continue
        gd = json.loads(gp.read_text())
        md = json.loads(mp.read_text())
        g = gd.get("1b_global_gps_compass", {}).get("gps_cv_r2_mean")
        m = md.get("1b_global_gps_compass", {}).get("gps_cv_r2_mean")
        gs = gd.get("1b_global_gps_compass", {}).get("gps_cv_r2_std", 0.0)
        ms = md.get("1b_global_gps_compass", {}).get("gps_cv_r2_std", 0.0)
        if g is None or m is None:
            continue
        rows.append({"label": label, "colour": colour,
                     "g": g, "gs": gs, "m": m, "ms": ms})

    if not rows:
        ax.text(0.5, 0.5, "(JSONs missing)", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        return

    # Bottleneck / pass-through shading
    ax.axvspan(-0.5, 1.5, color="#bcd4ec", alpha=0.40, zorder=0)
    ax.axvspan(1.5, 4.5, color="#dddddd", alpha=0.45, zorder=0)

    x = np.arange(len(rows))
    w = 0.36
    g_clip = [max(r["g"], CLIP_MIN) for r in rows]
    m_clip = [max(r["m"], CLIP_MIN) for r in rows]
    g_err  = [r["gs"] for r in rows]
    m_err  = [r["ms"] for r in rows]
    colours = [r["colour"] for r in rows]

    ax.bar(x - w / 2, g_clip, w, color=colours, edgecolor="black",
           linewidth=0.5, label="Gibson", yerr=g_err, capsize=2.5,
           error_kw={"linewidth": 0.6})
    ax.bar(x + w / 2, m_clip, w, color=colours, edgecolor="black",
           linewidth=0.5, hatch="///", alpha=0.7, label="MP3D (held out)",
           yerr=m_err, capsize=2.5, error_kw={"linewidth": 0.6})

    # Annotate clipped values
    for i, r in enumerate(rows):
        if r["g"] < CLIP_MIN:
            ax.annotate(f"{r['g']:.1f}", (i - w / 2, CLIP_MIN + 0.10),
                        ha="center", fontsize=6.5, color="darkred")
        if r["m"] < CLIP_MIN:
            ax.annotate(f"{r['m']:.1f}", (i + w / 2, CLIP_MIN + 0.10),
                        ha="center", fontsize=6.5, color="darkred")

    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=25,
                       ha="right", fontsize=8)
    ax.set_ylabel("GPS $R^2$ (5-fold CV)", fontsize=9)
    ax.set_ylim(CLIP_MIN - 0.05, 1.05)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("GPS encoding ordering, robust to dataset shift\n"
                 "(bottleneck | pass-through)", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=7, loc="lower right", frameon=False, ncol=1)


def panel_b(ax, results_dir: Path) -> None:
    """Top-layer GPS R² across step-in-episode bins, all 5 conditions."""
    p = results_dir / "temporal_probe_det.json"
    if not p.exists():
        ax.text(0.5, 0.5, "(temporal_probe_det.json missing)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        return
    d = json.loads(p.read_text())
    per_cond = d["per_condition"]
    for cond_key, label, colour, marker in CONDS:
        long_key = f"{cond_key}_gibson"
        if long_key not in per_cond:
            continue
        bins = per_cond[long_key]["gps_r2_per_bin"]
        xs, ys = [], []
        for b in bins:
            if b["r2"] is None:
                continue
            xmid = (b["lo"] + min(b["hi"], 1600)) / 2.0
            xs.append(xmid)
            ys.append(max(b["r2"], CLIP_MIN))
        ax.plot(xs, ys, marker=marker, label=label,
                color=colour, linewidth=1.6, markersize=4.5)
    ax.axhline(0, ls=":", color="grey", alpha=0.5, lw=0.7)
    ax.set_xlabel("step in episode (bin midpoint)", fontsize=9)
    ax.set_ylabel("GPS $R^2$", fontsize=9)
    ax.set_title("Temporal stability of top-layer GPS code", fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(CLIP_MIN - 0.05, 1.05)
    ax.tick_params(axis="both", labelsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=1)


def panel_c(ax, results_dir: Path) -> None:
    """Per-LSTM-layer GPS R², all 5 conditions."""
    layers_data: dict[str, dict[int, float]] = {}
    for cond_key, _, _, _ in CONDS:
        p = results_dir / f"{cond_key}_gibson_det_analysis.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if "1d_multilayer" not in d:
            continue
        layers_data[cond_key] = {
            e["layer"]: e["gps_r2"]
            for e in d["1d_multilayer"]
            if e.get("state") == "h"
        }

    if not layers_data:
        ax.text(0.5, 0.5, "(per-layer JSONs missing)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        return

    layers = sorted(next(iter(layers_data.values())).keys())
    for cond_key, label, colour, marker in CONDS:
        if cond_key not in layers_data:
            continue
        ys = [layers_data[cond_key][ly] for ly in layers]
        ys_clip = np.clip(ys, CLIP_MIN, 1.05)
        ax.plot(layers, ys_clip, marker=marker, label=label,
                color=colour, linewidth=1.6, markersize=5)
    ax.axhline(0, ls=":", color="grey", alpha=0.5, lw=0.7)
    ax.set_xticks(layers)
    role_names = {0: "near encoder", 1: "middle", 2: "top (policy)"}
    layer_labels = [f"L{ly}\n({role_names.get(ly, '')})" for ly in layers]
    ax.set_xticklabels(layer_labels, fontsize=8)
    ax.set_ylabel("GPS $R^2$", fontsize=9)
    ax.set_ylim(CLIP_MIN - 0.05, 1.08)
    ax.set_title("GPS code across LSTM layers", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 3 panels in one row.  Panel (a) gets a bit more horizontal real
    # estate because it carries the headline + condition labels.
    fig = plt.figure(figsize=(13.5, 3.6))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.5, 1.4, 1.0],
        wspace=0.32,
        top=0.85, bottom=0.18, left=0.05, right=0.99,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a, args.results_dir)
    panel_b(ax_b, args.results_dir)
    panel_c(ax_c, args.results_dir)

    for ax, lbl in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")]:
        ax.text(-0.12, 1.18, lbl, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

    for ext in ("pdf", "png"):
        out = args.out_dir / f"h1_mega.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
