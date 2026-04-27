"""
H1 mega-figure for the consolidated Results section. 1x3 row, three panels.

Panels:
  (a) Grouped bars: GPS $R^2$ across the 5 conditions, on Gibson (no-cap
      5-fold CV) and held-out MP3D (same checkpoints, no re-training).
  (b) Top-layer GPS code stability across episode duration (line plot).
  (c) Information-pipeline view: GPS $R^2$ at four locations along the
      agent's information pathway — Encoder feature-map (post ResNet-18,
      pre-LSTM), then LSTM Layer 0 / 1 / 2. Combines what used to be
      two separate panels (per-LSTM-layer + encoder-vs-LSTM bars) into
      a single "where in the pipeline does GPS sit" plot.

Reads:
  - <results-dir>/{cond}_{gibson,mp3d}_det_analysis.json        (a)
  - <results-dir>/temporal_probe_det.json                       (b)
  - <results-dir>/{cond}_gibson_det_analysis.json               (c, LSTM layers)
  - <results-dir>/{cond}_encoder_features_det.json              (c, encoder)

Writes: <out-dir>/fig2_h1_mega.pdf

Usage:
    python scripts/paper_figures/make_h1_mega_figure.py \\
        --results-dir /tmp/probing_results_local \\
        --out-dir docs/NeurIPS_2026/fig
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

# ─── Shared condition styling ─────────────────────────────────────────
CONDS = [
    # (json_key,           short_label,         colour,    marker)
    ("blind",              "Blind",             "#444444", "o"),
    ("matched",            "Coarse (1×1)",     "#377eb8", "s"),
    ("uniform",            "Uniform",           "#4daf4a", "^"),
    ("foveated",           "Foveated (fix)",    "#e41a1c", "D"),
    ("foveated_learned",   "Foveated (learned)",       "#ff7f00", "v"),
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

    # Bottleneck / rich-encoder shading + explicit labels (fix D)
    ax.axvspan(-0.5, 1.5, color="#bcd4ec", alpha=0.40, zorder=0)
    ax.axvspan(1.5, 4.5, color="#dddddd", alpha=0.45, zorder=0)
    ax.text(0.5, 1.04, "bottleneck",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#3a5a85",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#3a5a85", alpha=0.9, lw=0.8))
    ax.text(3.0, 1.04, "rich-encoder pass-through",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#555",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#555", alpha=0.9, lw=0.8))

    x = np.arange(len(rows))
    w = 0.38
    g_clip = [max(r["g"], CLIP_MIN) for r in rows]
    m_clip = [max(r["m"], CLIP_MIN) for r in rows]
    g_err  = [r["gs"] for r in rows]
    m_err  = [r["ms"] for r in rows]
    colours = [r["colour"] for r in rows]

    # Fix C: replace hatch with edge-distinction
    # Gibson: solid full color, dark edge.  MP3D: same color at alpha=0.40,
    # bold same-colour edge — visually distinct at any rendering size.
    ax.bar(x - w / 2, g_clip, w, color=colours, edgecolor="black",
           linewidth=0.6, label="Gibson", yerr=g_err, capsize=2.5,
           error_kw={"linewidth": 0.7})
    ax.bar(x + w / 2, m_clip, w, color=colours, edgecolor=colours,
           linewidth=2.0, alpha=0.40, label="MP3D (held out)",
           yerr=m_err, capsize=2.5, error_kw={"linewidth": 0.7})

    # Fix A: annotate clipped values prominently
    for i, r in enumerate(rows):
        if r["g"] < CLIP_MIN:
            ax.annotate(f"{r['g']:.2f}", (i - w / 2, CLIP_MIN + 0.04),
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="darkred")
        if r["m"] < CLIP_MIN:
            ax.annotate(f"{r['m']:.2f}", (i + w / 2, CLIP_MIN + 0.04),
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="darkred")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    # Use abbreviated labels for tighter layout (no rotation needed at 9pt)
    abbrev_labels = {
        "Blind": "Blind",
        "Coarse (1×1)": "Coarse",
        "Uniform": "Uniform",
        "Foveated (fix)": "Fov-fix",
        "Foveated (learned)": "Fov-learn",
    }
    ax.set_xticklabels([abbrev_labels.get(r["label"], r["label"]) for r in rows],
                       rotation=0, ha="center", fontsize=11)
    ax.set_ylabel(r"GPS $R^2$ (5-fold CV)", fontsize=12, fontweight="bold")
    ax.set_ylim(CLIP_MIN - 0.05, 1.10)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title("(a) Bottleneck retains GPS",
                 fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    # Place legend at upper-right inside the bottleneck region (where there's
    # no data competition) to avoid clipping-annotation overlap.
    ax.legend(fontsize=10, loc="upper right",
              bbox_to_anchor=(1.0, 0.92),
              frameon=True, framealpha=0.92, ncol=1)


def panel_b(ax, results_dir: Path) -> None:
    """Top-layer GPS R² across step-in-episode bins, all 5 conditions."""
    p = results_dir / "temporal_probe_det.json"
    if not p.exists():
        ax.text(0.5, 0.5, "(temporal_probe_det.json missing)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        return
    d = json.loads(p.read_text())
    per_cond = d["per_condition"]

    # Track clipped points so we can annotate them (fix B)
    clipped_pts = []  # list of (x, true_y, condition_label, colour)
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
            if b["r2"] < CLIP_MIN:
                clipped_pts.append((xmid, b["r2"], label, colour))
        ax.plot(xs, ys, marker=marker, label=label,
                color=colour, linewidth=1.8, markersize=5)

    # Fix B: annotate clipped points
    for x, true_y, _, colour in clipped_pts:
        ax.annotate(f"{true_y:.1f}", (x, CLIP_MIN + 0.04),
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="darkred")

    ax.axhline(0, ls=":", color="grey", alpha=0.6, lw=0.8)
    ax.set_xlabel("step in episode (log-binned)", fontsize=12, fontweight="bold")
    ax.set_title("(b) Long-tail collapse",
                 fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax.set_xscale("log")
    ax.set_ylim(CLIP_MIN - 0.05, 1.10)
    ax.tick_params(axis="both", labelsize=11)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="lower left", fontsize=10, frameon=False, ncol=1)


def panel_c(ax, results_dir: Path) -> None:
    """Information-pipeline view: GPS R² across {Encoder, L0, L1, L2}.

    Combines the per-LSTM-layer probe (h_0 / h_1 / h_2) with the encoder
    feature-map probe (post ResNet-18, pre-LSTM) into a single line plot
    showing where in the agent's information pipeline a linear probe
    finds GPS. Sighted conditions span all 4 x-positions; blind has no
    encoder; foveated_learned does not have encoder-probe data.

    The plot makes three readings simultaneously:
    - Encoder → L0 jump: how much GPS the LSTM input concat (with the
      Layer-0 sensor-stack GPS embedding) adds.
    - Across L0 → L1 → L2: where the H1 divergence occurs (Layer 2,
      the policy readout) for rich-encoder conditions.
    - Encoder R² ≪ L2 R² for bottleneck conditions: the "LSTM gain"
      that previously had its own panel.
    """
    # Per-condition encoder probe R²
    enc_r2: dict[str, float] = {}
    for cond_key, _, _, _ in CONDS:
        if cond_key == "blind":
            continue
        p = results_dir / f"{cond_key}_encoder_features_det.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        entry = d.get(cond_key, {})
        v = entry.get("encoder_features_gps_r2_mean")
        if v is not None:
            enc_r2[cond_key] = v

    # Per-condition per-layer LSTM R²
    lstm_layers: dict[str, dict[int, float]] = {}
    for cond_key, _, _, _ in CONDS:
        p = results_dir / f"{cond_key}_gibson_det_analysis.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if "1d_multilayer" not in d:
            continue
        lstm_layers[cond_key] = {
            e["layer"]: e["gps_r2"]
            for e in d["1d_multilayer"]
            if e.get("state") == "h"
        }

    if not lstm_layers:
        ax.text(0.5, 0.5, "(per-layer JSONs missing)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        return

    # Light shading to mark "Encoder" vs LSTM regions
    ax.axvspan(-0.4, 0.5, color="#f0f0f0", alpha=0.6, zorder=0)
    ax.axvline(0.5, color="#999", linestyle="--", lw=0.6, alpha=0.6,
               zorder=0)
    ax.axhline(0, ls=":", color="grey", alpha=0.6, lw=0.8)

    # Fix H: shaded "MLP probe recovers GPS" zone at L2 column.
    # Paper §4.2 reports MLP recovers R² ∈ [0.51, 0.73] for rich-encoder
    # conditions on the same L2 hidden states. The shaded band visualises
    # this as a hidden-but-non-linear signal the linear probe misses.
    ax.axhspan(0.51, 0.73, xmin=(2.6 + 0.4) / (3.4 + 0.4),
               xmax=(3.4 + 0.4) / (3.4 + 0.4),
               color="#a25cb4", alpha=0.20, zorder=0)
    # Place the label OUTSIDE the band, to the left, with an arrow pointing in.
    ax.annotate("MLP probe\nrecovers GPS",
                xy=(2.95, 0.62), xytext=(2.3, 0.95),
                ha="center", va="center", fontsize=10,
                color="#5a2580", style="italic", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#5a2580",
                                lw=1.0, connectionstyle="arc3,rad=-0.2"),
                zorder=5)

    # Plot one line per condition
    for cond_key, label, colour, marker in CONDS:
        xs, ys = [], []
        if cond_key in enc_r2:
            xs.append(0)
            ys.append(max(enc_r2[cond_key], CLIP_MIN))
        if cond_key in lstm_layers:
            for layer in [0, 1, 2]:
                if layer in lstm_layers[cond_key]:
                    xs.append(layer + 1)
                    val = lstm_layers[cond_key][layer]
                    ys.append(float(np.clip(val, CLIP_MIN, 1.05)))
        if xs:
            ax.plot(xs, ys, marker=marker, label=label,
                    color=colour, linewidth=1.8, markersize=6.5)

    # X-axis labels (slightly larger, single line where possible)
    x_labels = ["Encoder\n(post ResNet)", "L0\n(LSTM in)",
                "L1\n(mid)", "L2\n(top, policy)"]
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(CLIP_MIN - 0.05, 1.10)
    ax.set_title("(c) Pipeline view",
                 fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax.tick_params(axis="y", labelsize=11)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="lower left", fontsize=10, frameon=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1×3 row. Wider (a) for the bar+condition labels; (b) and (c) are
    # line plots that each take comparable horizontal space.
    fig = plt.figure(figsize=(14.5, 4.6))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.3, 1.1, 1.2],
        wspace=0.22,  # tighter — y-labels removed on (b)(c)
        top=0.82, bottom=0.20, left=0.06, right=0.99,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a, args.results_dir)
    panel_b(ax_b, args.results_dir)
    panel_c(ax_c, args.results_dir)

    # Fix F: panel labels are now baked into each panel's title via
    # set_title("(a) ...", loc="left"). No separate transAxes labels.

    out = args.out_dir / "fig2_h1_mega.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
