"""
Time-cell figure.

Four-panel figure for the time-cell analysis:

  (a) Bar — fraction of time cells per condition.
  (b) Histogram — distribution of peak times (μ) for all classified time cells
      across conditions.  A uniform distribution indicates full episode tiling.
  (c) Box/violin — FWHM distribution per condition (narrower = sharper cells).
  (d) Raster + tuning curves — for each condition, show the tuning curves of
      all classified time cells, sorted by peak time, to visualise tiling.

Input:  results/probing/time_cells.json
        results/probing/time_cells_curves.npz   (optional, for panel d)
        Both produced by scripts/probing/time_cells.py

Output: <out>  (default: docs/manuscript/fig/time_cells.pdf)

Usage:
    python scripts/paper_figures/make_time_cells_figure.py \\
        --data    results/probing/time_cells.json \\
        --curves  results/probing/time_cells_curves.npz \\
        --out     docs/manuscript/fig/time_cells.pdf
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
from matplotlib.gridspec import GridSpec

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
from _style import apply_paper_style

apply_paper_style()

COND_META = {
    "blind_gibson":            ("Blind",    "#444444"),
    "coarse_gibson":           ("Coarse",   "#377eb8"),
    "foveated_gibson":         ("Foveated", "#e41a1c"),
    "foveated_logpolar_gibson":("Log-polar","#ff7f00"),
    "uniform_gibson":          ("Uniform",  "#4daf4a"),
}


def _cond_label_color(cond: str) -> tuple[str, str]:
    return COND_META.get(cond, (cond, "#888888"))


def _load_data(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def _load_curves(npz_path: Path | None) -> dict[str, np.ndarray]:
    if npz_path is None or not npz_path.exists():
        return {}
    raw = np.load(npz_path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def make_figure(data: dict, curves_store: dict, out_path: Path) -> None:
    present_conds = [c for c in COND_META if c in data and "error" not in data[c]]
    if not present_conds:
        print("No valid conditions found in JSON — aborting figure.")
        return

    n_conds = len(present_conds)
    labels  = [_cond_label_color(c)[0] for c in present_conds]
    colors  = [_cond_label_color(c)[1] for c in present_conds]

    # Layout: 2 rows × max(n_conds, 2) cols for panels a/b/c + rasters.
    n_raster_cols = min(n_conds, 4)
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(
        2, max(4, n_raster_cols),
        figure=fig,
        height_ratios=[1.0, 1.3],
        hspace=0.45, wspace=0.35,
    )

    ax_bar   = fig.add_subplot(gs[0, 0])
    ax_hist  = fig.add_subplot(gs[0, 1])
    ax_fwhm  = fig.add_subplot(gs[0, 2])
    ax_null  = fig.add_subplot(gs[0, 3])
    raster_axes = [fig.add_subplot(gs[1, i]) for i in range(n_raster_cols)]

    # ── (a) Bar — fraction of time cells ──────────────────────────────────────
    fracs = [data[c]["frac_time_cells"] * 100 for c in present_conds]
    bars  = ax_bar.bar(range(n_conds), fracs, color=colors,
                       alpha=0.75, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, fracs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax_bar.set_xticks(range(n_conds))
    ax_bar.set_xticklabels(labels, rotation=15, ha="right")
    ax_bar.set_ylabel("Time cells (%)", fontweight="bold")
    ax_bar.set_title("(a) Fraction of time cells", loc="left", fontweight="bold", pad=4)
    ax_bar.set_ylim(0, max(fracs, default=1) * 1.35 + 1)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.3)

    # ── (b) Histogram — peak time distribution (tiling) ───────────────────────
    bin_edges = np.linspace(0, 1, 11)
    bin_width = bin_edges[1] - bin_edges[0]
    for ci, cond in enumerate(present_conds):
        peaks = data[cond].get("peak_times_time_cells", [])
        if not peaks:
            continue
        hist, _ = np.histogram(peaks, bins=bin_edges, density=True)
        ax_hist.step(bin_edges[:-1], hist, where="post",
                     color=colors[ci], linewidth=1.5, label=labels[ci], alpha=0.85)

    ax_hist.axhline(1.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7,
                    label="Uniform (tiling)")
    ax_hist.set_xlabel("Normalised episode time (peak)", fontweight="bold")
    ax_hist.set_ylabel("Density", fontweight="bold")
    ax_hist.set_title("(b) Peak-time distribution\n(uniform → full tiling)",
                       loc="left", fontweight="bold", pad=4)
    ax_hist.legend(fontsize=8, frameon=False)
    ax_hist.set_xlim(0, 1)
    ax_hist.grid(axis="y", linestyle=":", alpha=0.3)

    # ── (c) Violin — FWHM distribution ────────────────────────────────────────
    fwhm_data = []
    for cond in present_conds:
        units = data[cond].get("units", [])
        fwhms = [u["fwhm"] for u in units if u.get("is_time_cell") and u.get("fwhm") is not None]
        fwhm_data.append(fwhms if fwhms else [np.nan])

    vp = ax_fwhm.violinplot(fwhm_data, positions=range(n_conds),
                             showmedians=True, showextrema=False)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.65)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.5)

    ax_fwhm.axhline(0.45, color="crimson", linestyle="--", linewidth=1.0,
                    alpha=0.8, label="Max FWHM criterion (0.45)")
    ax_fwhm.set_xticks(range(n_conds))
    ax_fwhm.set_xticklabels(labels, rotation=15, ha="right")
    ax_fwhm.set_ylabel("FWHM (fraction of episode)", fontweight="bold")
    ax_fwhm.set_title("(c) Temporal tuning width\n(lower = sharper time cells)",
                       loc="left", fontweight="bold", pad=4)
    ax_fwhm.legend(fontsize=8, frameon=False)
    ax_fwhm.grid(axis="y", linestyle=":", alpha=0.3)

    # ── (d) Null distribution vs. observed peak amplitude ─────────────────────
    for ci, cond in enumerate(present_conds):
        null_samp = np.array(data[cond].get("null_distribution_sample", []))
        null_p99  = data[cond].get("null_p99")
        units     = data[cond].get("units", [])
        obs_amps  = [u["amp"] for u in units if u.get("is_time_cell")]

        if len(null_samp):
            ax_null.hist(null_samp, bins=30, density=True, alpha=0.25,
                         color=colors[ci], label=f"{labels[ci]} null")
        if obs_amps:
            ax_null.hist(obs_amps, bins=15, density=True, alpha=0.55,
                         color=colors[ci], histtype="step", linewidth=1.5)

    ax_null.set_xlabel("Peak amplitude (max − min activation)", fontweight="bold")
    ax_null.set_ylabel("Density", fontweight="bold")
    ax_null.set_title("(d) Null vs. observed amplitude\n(shaded = shuffle null)",
                       loc="left", fontweight="bold", pad=4)
    ax_null.legend(fontsize=7, frameon=False, ncol=2)
    ax_null.grid(axis="y", linestyle=":", alpha=0.3)

    # ── (e) Raster — tuning curves sorted by peak time ────────────────────────
    for i, cond in enumerate(present_conds[:n_raster_cols]):
        ax = raster_axes[i]
        label, color = _cond_label_color(cond)

        units = data[cond].get("units", [])
        tc_units = [u for u in units if u.get("is_time_cell")]
        bin_centers = np.array(data[cond].get("bin_centers", []))

        # Pull tuning curves from NPZ store or from JSON if embedded.
        tc_key = f"{cond}_tuning_curves"
        if tc_key in curves_store and len(tc_units) > 0:
            all_curves = curves_store[tc_key]     # (hidden_dim, n_bins)
            tc_indices = [u["unit"] for u in tc_units]
            curves     = all_curves[tc_indices]   # (n_tc, n_bins)
        elif "tuning_curves" in data[cond] and len(tc_units) > 0:
            all_curves = np.array(data[cond]["tuning_curves"])
            tc_indices = [u["unit"] for u in tc_units]
            curves     = all_curves[tc_indices]
        else:
            ax.text(0.5, 0.5,
                    f"{label}\n{len(tc_units)} time cells\n(rerun with --save-curves\nor --out-npz for raster)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="grey")
            ax.set_title(f"{label}  ({len(tc_units)} TC)", fontweight="bold", fontsize=10)
            ax.axis("off")
            continue

        if len(curves) == 0:
            ax.text(0.5, 0.5, f"{label}\n0 time cells",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="grey")
            ax.axis("off")
            continue

        # Sort by peak time.
        with np.errstate(invalid='ignore'):
            peaks = np.nanargmax(curves, axis=1)
        order = np.argsort(peaks)
        curves_sorted = curves[order]

        # Normalise each row to [0, 1] for display.
        row_min = np.nanmin(curves_sorted, axis=1, keepdims=True)
        row_max = np.nanmax(curves_sorted, axis=1, keepdims=True)
        denom   = row_max - row_min + 1e-8
        curves_norm = np.clip((curves_sorted - row_min) / denom, 0, 1)

        x = bin_centers if len(bin_centers) else np.linspace(0, 1, curves_norm.shape[1])
        ax.imshow(curves_norm, aspect="auto", cmap="hot",
                  extent=[x[0], x[-1], 0, len(curves_norm)],
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_xlabel("Normalised time", fontweight="bold", fontsize=9)
        ax.set_ylabel("Time cell (sorted by peak)", fontweight="bold", fontsize=9)
        ax.set_title(f"({chr(ord('e') + i)}) {label}  ·  {len(tc_units)} TC",
                     loc="left", fontweight="bold", fontsize=10, pad=3, color=color)
        ax.set_xlim(0, 1)

    fig.suptitle(
        "Time-cell signature in LSTM hidden states\n"
        "(Gaussian fit to per-unit activation vs. normalised episode time)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-cell figure")
    p.add_argument("--data",   required=True,
                   help="JSON from scripts/probing/time_cells.py")
    p.add_argument("--curves", default=None,
                   help="NPZ of tuning curves from time_cells.py --out-npz (optional)")
    p.add_argument("--out", default="docs/manuscript/fig/time_cells.pdf",
                   help="Output PDF/PNG path")
    return p.parse_args()


if __name__ == "__main__":
    args         = parse_args()
    data         = _load_data(Path(args.data))
    curves_store = _load_curves(Path(args.curves) if args.curves else None)
    make_figure(data, curves_store, Path(args.out))
