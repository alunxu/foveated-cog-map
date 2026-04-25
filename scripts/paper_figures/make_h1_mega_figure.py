"""
H1 mega-figure for the consolidated Results section.  Replaces the four
separate H1-supporting figures (h1_bottleneck, temporal_probe_evolution,
layerwise_decay, mp3d_generalization) with a single 4-panel composite.

Panels:
  (a) Current-state probe R² (GPS / compass / DtG) bars across the 5
      conditions — the headline H1 finding.
  (b) Top-layer GPS code stability across episode duration — the
      temporal-stability nuance (rich-encoder agents encode GPS in the
      typical-episode window then overwrite it).
  (c) Per-LSTM-layer GPS R² — disambiguates the Layer-0 sensor-pass-
      through floor from the Layer-2 policy-readout signal.
  (d) Gibson → MP3D generalisation bars for GPS R² — same checkpoints,
      no re-training; the H1 ordering is dataset-robust.

Reads:
  - Hardcoded numbers from the no-cap Gibson probes (panel a)
  - <results-dir>/temporal_probe_det.json                 (panel b)
  - <results-dir>/{cond}_gibson_det_analysis.json         (panel c)
  - <results-dir>/{cond}_{gibson,mp3d}_det_analysis.json  (panel d)

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
    # (json_key,                    short_label,         colour,    marker)
    ("blind",                       "Blind",             "#444444", "o"),
    ("matched",                     "Matched (1×1)",     "#377eb8", "s"),
    ("uniform",                     "Uniform",           "#4daf4a", "^"),
    ("foveated",                    "Foveated (fix)",    "#e41a1c", "D"),
    ("foveated_learned",            "Fov-learned",       "#ff7f00", "v"),
]
TEMPORAL_KEYS = {  # different convention used by temporal_probe_det.json
    "blind_gibson":            "blind",
    "matched_gibson":          "matched",
    "uniform_gibson":          "uniform",
    "foveated_gibson":         "foveated",
    "foveated_learned_gibson": "foveated_learned",
}
CLIP_MIN = -1.5

# ─── Panel (a): hardcoded no-cap Gibson R² values ─────────────────────
# (cond, GPS μ, GPS σ, compass μ, compass σ, DtG μ, DtG σ)  — Table 1
PANEL_A = [
    ("Blind",          +0.95, 0.02, +0.81, 0.08, +0.90, 0.03),
    ("Matched (1×1)",  +0.78, 0.10, +0.64, 0.10, +0.85, 0.12),
    ("Uniform",        -0.31, 0.86, +0.36, 0.23, +0.86, 0.09),
    ("Foveated (fix)", +0.06, 0.88, +0.07, 0.69, +0.82, 0.09),
    ("Fov-learned",    -2.43, 3.98, -1.34, 3.14, +0.81, 0.09),
]


def panel_a(ax_gps, ax_comp, ax_dtg) -> None:
    """Three sub-axes, current-state probe R² bars."""
    labels = [d[0] for d in PANEL_A]
    gps_m = np.array([d[1] for d in PANEL_A]); gps_s = np.array([d[2] for d in PANEL_A])
    com_m = np.array([d[3] for d in PANEL_A]); com_s = np.array([d[4] for d in PANEL_A])
    dtg_m = np.array([d[5] for d in PANEL_A]); dtg_s = np.array([d[6] for d in PANEL_A])
    colours = [c[2] for c in CONDS]

    x = np.arange(len(labels))
    for ax, (m, s, title) in zip(
        [ax_gps, ax_comp, ax_dtg],
        [(gps_m, gps_s, "GPS  (bottleneck | pass-through)"),
         (com_m, com_s, "Compass"),
         (dtg_m, dtg_s, "DtG (control)")],
    ):
        m_clip = np.clip(m, CLIP_MIN, 1.0)
        ax.bar(x, m_clip, yerr=s, color=colours, alpha=0.85,
               edgecolor="black", linewidth=0.5, capsize=2.5,
               error_kw={"linewidth": 0.7})
        for i, mv in enumerate(m):
            if mv < CLIP_MIN:
                ax.annotate(f"{mv:.1f}", (i, CLIP_MIN + 0.12),
                            ha="center", fontsize=6.5, color="darkred")
                ax.annotate("↓", (i, CLIP_MIN + 0.04),
                            ha="center", fontsize=10, color="darkred")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_ylim(CLIP_MIN, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("$R^2$", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

    # Background shading: bottleneck (left 2 bars) vs pass-through
    # (right 3 bars) on every (a) panel.  Group labels go in the title
    # of the GPS panel only to avoid axis-region collisions.
    for ax in (ax_gps, ax_comp, ax_dtg):
        ax.axvspan(-0.5, 1.5, color="#bcd4ec", alpha=0.40, zorder=0)
        ax.axvspan(1.5, 4.5, color="#dddddd", alpha=0.45, zorder=0)


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
        # temporal_probe uses *_gibson keys
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
    ax.set_xlabel("step in episode (bin midpoint)", fontsize=8)
    ax.set_ylabel("GPS $R^2$", fontsize=8)
    ax.set_title("Temporal stability of top-layer GPS code", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_ylim(CLIP_MIN - 0.05, 1.05)
    ax.tick_params(axis="both", labelsize=7)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=1)


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
    ax.set_xticklabels(layer_labels, fontsize=7)
    ax.set_ylabel("GPS $R^2$", fontsize=8)
    ax.set_ylim(CLIP_MIN - 0.05, 1.08)
    ax.set_title("GPS code across LSTM layers", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def panel_d(ax, results_dir: Path) -> None:
    """Gibson → MP3D generalisation bars for GPS R² (5-fold CV mean)."""
    rows = []
    for cond_key, label, colour, _ in CONDS:
        gp = results_dir / f"{cond_key}_gibson_det_analysis.json"
        mp = results_dir / f"{cond_key}_mp3d_det_analysis.json"
        if not (gp.exists() and mp.exists()):
            continue
        gd = json.loads(gp.read_text())
        md = json.loads(mp.read_text())
        # Use the 5-fold CV mean to match Table 1 / panel (a). The
        # alternative `gps_r2` field is a single train/test split and
        # disagrees in magnitude.
        g = gd.get("1b_global_gps_compass", {}).get("gps_cv_r2_mean")
        m = md.get("1b_global_gps_compass", {}).get("gps_cv_r2_mean")
        if g is None or m is None:
            continue
        rows.append({"label": label, "colour": colour, "g": g, "m": m})

    if not rows:
        ax.text(0.5, 0.5, "(MP3D JSONs missing)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        return

    x = np.arange(len(rows))
    w = 0.38
    g_clip = [max(r["g"], CLIP_MIN) for r in rows]
    m_clip = [max(r["m"], CLIP_MIN) for r in rows]
    colours = [r["colour"] for r in rows]
    ax.bar(x - w / 2, g_clip, w, color=colours, edgecolor="black",
           linewidth=0.5, label="Gibson")
    ax.bar(x + w / 2, m_clip, w, color=colours, edgecolor="black",
           linewidth=0.5, hatch="///", alpha=0.7, label="MP3D (held out)")
    # Annotate clipped values (rich-encoder very negative)
    for i, r in enumerate(rows):
        if r["g"] < CLIP_MIN:
            ax.annotate(f"{r['g']:.1f}", (i - w / 2, CLIP_MIN + 0.08),
                        ha="center", fontsize=6, color="darkred")
        if r["m"] < CLIP_MIN:
            ax.annotate(f"{r['m']:.1f}", (i + w / 2, CLIP_MIN + 0.08),
                        ha="center", fontsize=6, color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=30,
                       ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_ylabel("GPS $R^2$", fontsize=8)
    ax.set_ylim(CLIP_MIN - 0.05, 1.05)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Robustness: same checkpoints on held-out MP3D", fontsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 4 main panels: top row = (a) bars (3 sub-axes spanning), (b) temporal;
    # bottom row = (c) per-layer, (d) MP3D.
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1, 1.05],
        width_ratios=[1, 1, 1, 2.8],
        hspace=0.85, wspace=0.55,
        top=0.92, bottom=0.10, left=0.06, right=0.98,
    )
    ax_a_gps  = fig.add_subplot(gs[0, 0])
    ax_a_comp = fig.add_subplot(gs[0, 1])
    ax_a_dtg  = fig.add_subplot(gs[0, 2])
    ax_b      = fig.add_subplot(gs[0, 3])
    ax_c      = fig.add_subplot(gs[1, 0:2])
    ax_d      = fig.add_subplot(gs[1, 2:4])

    panel_a(ax_a_gps, ax_a_comp, ax_a_dtg)
    panel_b(ax_b, args.results_dir)
    panel_c(ax_c, args.results_dir)
    panel_d(ax_d, args.results_dir)

    # Panel labels (a, b, c, d) sat above each panel's title
    for ax, lbl in [(ax_a_gps, "(a)"), (ax_b, "(b)"),
                    (ax_c, "(c)"), (ax_d, "(d)")]:
        ax.text(-0.18, 1.22, lbl, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

    for ext in ("pdf", "png"):
        out = args.out_dir / f"h1_mega.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
