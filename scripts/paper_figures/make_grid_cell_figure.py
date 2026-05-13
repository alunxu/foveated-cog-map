"""
Grid-cell signature figure.

Three-panel figure for the grid-cell signature analysis:

  (a) Violin/box — gridness score distribution across all units per condition.
      Shuffle-null p99 shown as a dashed horizontal reference.
  (b) Bar — fraction of grid-like units per condition (gridness > null p99).
  (c) Rate-map + SAC panels — for the top-gridness unit in each condition,
      show the smoothed 2D rate map alongside its autocorrelogram with the
      annulus used for scoring overlaid.

Input:  results/probing/grid_cell_signature.json
        (produced by scripts/probing/grid_cell_signature.py)

Output: <out>  (default: docs/manuscript/fig/grid_cell_signature.pdf)

Usage:
    python scripts/paper_figures/make_grid_cell_figure.py \\
        --data results/probing/grid_cell_signature.json \\
        --out  docs/manuscript/fig/grid_cell_signature.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# Shared paper style — works whether run from repo root or scripts/paper_figures/
_here = Path(__file__).parent
sys.path.insert(0, str(_here))
from _style import apply_paper_style

apply_paper_style()

# ── condition display config ───────────────────────────────────────────────────

COND_META = {
    "blind_gibson":            ("Blind",    "#444444"),
    "coarse_gibson":           ("Coarse",   "#377eb8"),
    "foveated_gibson":         ("Foveated", "#e41a1c"),
    "foveated_logpolar_gibson":("Log-polar","#ff7f00"),
    "uniform_gibson":          ("Uniform",  "#4daf4a"),
}

# Annulus fractions must match grid_cell_signature.py constants.
ANNULUS_RMIN_FRAC = 0.18
ANNULUS_RMAX_FRAC = 0.70


# ── helpers ────────────────────────────────────────────────────────────────────

def _cond_label_color(cond: str) -> tuple[str, str]:
    return COND_META.get(cond, (cond, "#888888"))


def _draw_annulus(ax, sac_shape: tuple, color: str = "white", lw: float = 1.0) -> None:
    """Overlay the annulus bounds used for gridness scoring."""
    H, W = sac_shape
    cy, cx = H / 2, W / 2
    max_r = min(cy, cx)
    for frac in (ANNULUS_RMIN_FRAC, ANNULUS_RMAX_FRAC):
        circle = mpatches.Circle(
            (cx, cy), radius=max_r * frac,
            fill=False, edgecolor=color, linewidth=lw, linestyle="--")
        ax.add_patch(circle)


def _load_data(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


# ── main figure ───────────────────────────────────────────────────────────────

def make_figure(data: dict, out_path: Path) -> None:
    present_conds = [c for c in COND_META if c in data and "error" not in data[c]]
    if not present_conds:
        print("No valid conditions found in JSON — aborting figure.")
        return

    n_conds = len(present_conds)

    # Layout: row 0 = violin (wide) + bar; row 1 = rate map / SAC panels.
    # We show rate-map + SAC for at most 4 conditions.
    n_map_conds = min(n_conds, 4)
    fig = plt.figure(figsize=(13, 8))
    gs  = GridSpec(
        2, 2 + n_map_conds,
        figure=fig,
        height_ratios=[1.0, 1.1],
        width_ratios=[2.5, 1.5] + [1.0] * n_map_conds,
        hspace=0.45, wspace=0.35,
    )
    ax_violin = fig.add_subplot(gs[0, 0])
    ax_bar    = fig.add_subplot(gs[0, 1])
    map_axes  = [fig.add_subplot(gs[1, 2 + i]) for i in range(n_map_conds)]
    sac_axes  = [fig.add_subplot(gs[0, 2 + i]) for i in range(n_map_conds)]

    labels = [_cond_label_color(c)[0] for c in present_conds]
    colors = [_cond_label_color(c)[1] for c in present_conds]

    # ── (a) Violin — gridness distribution ────────────────────────────────────
    violin_data = []
    null_p99s   = []
    for cond in present_conds:
        g = np.array(data[cond]["gridness_per_unit"])
        violin_data.append(g[~np.isnan(g)])
        null_p99s.append(data[cond].get("null_p99", np.nan))

    vp = ax_violin.violinplot(violin_data, positions=range(n_conds),
                               showmedians=True, showextrema=False)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.65)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.5)

    # Reference lines.
    ax_violin.axhline(0, color="grey", linestyle=":", linewidth=0.9, alpha=0.6)
    null_mean_p99 = float(np.nanmean(null_p99s))
    ax_violin.axhline(null_mean_p99, color="crimson", linestyle="--",
                       linewidth=1.0, alpha=0.8,
                       label=f"Null p99 (mean) = {null_mean_p99:.3f}")

    ax_violin.set_xticks(range(n_conds))
    ax_violin.set_xticklabels(labels, rotation=15, ha="right")
    ax_violin.set_ylabel("Gridness score", fontweight="bold")
    ax_violin.set_title("(a) Gridness distribution\nacross all units",
                         loc="left", fontweight="bold", pad=4)
    ax_violin.legend(fontsize=9, frameon=False)
    ax_violin.grid(axis="y", linestyle=":", alpha=0.3)

    # ── (b) Bar — fraction of grid-like units ─────────────────────────────────
    fracs = [data[c]["frac_gridlike"] * 100 for c in present_conds]
    xs    = range(n_conds)
    bars  = ax_bar.bar(xs, fracs, color=colors, alpha=0.75, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, fracs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(labels, rotation=15, ha="right")
    ax_bar.set_ylabel("Grid-like units (%)", fontweight="bold")
    ax_bar.set_title("(b) Fraction grid-like\n(gridness > null p99)",
                      loc="left", fontweight="bold", pad=4)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.3)
    ax_bar.set_ylim(0, max(fracs) * 1.3 + 2)

    # ── (c) Rate-map + SAC for top unit per condition ─────────────────────────
    for i, cond in enumerate(present_conds[:n_map_conds]):
        label, color = _cond_label_color(cond)
        res = data[cond]

        top_maps = res.get("top_unit_maps", {})
        if not top_maps:
            # No maps saved — show placeholder text.
            for ax in (map_axes[i], sac_axes[i]):
                ax.text(0.5, 0.5, "Run with\n--save-maps",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="grey")
                ax.axis("off")
            sac_axes[i].set_title(f"{label}", fontweight="bold", fontsize=10)
            continue

        # Pick the highest-gridness saved unit.
        best_uid = max(top_maps.keys(), key=lambda u: top_maps[u].get("gridness", -999))
        entry    = top_maps[best_uid]
        rmap     = np.array(entry["rate_map"])
        sac_arr  = np.array(entry["sac"]) if entry["sac"] is not None else None
        g_score  = entry.get("gridness", float("nan"))

        # Rate map.
        rm_plot = np.where(np.isnan(rmap), np.nanmin(rmap), rmap)
        im = map_axes[i].imshow(rm_plot, origin="lower", cmap="hot", aspect="equal")
        map_axes[i].set_title(f"{label}\nunit {best_uid}",
                               fontsize=9, fontweight="bold", pad=3)
        map_axes[i].axis("off")
        plt.colorbar(im, ax=map_axes[i], fraction=0.04, pad=0.03,
                     label="activation")

        # Autocorrelogram.
        if sac_arr is not None:
            # Clip SAC to [−1, 1] for display (central peak already 1).
            sac_disp = np.clip(sac_arr, -1.0, 1.0)
            sac_axes[i].imshow(sac_disp, origin="lower", cmap="RdBu_r",
                                vmin=-1, vmax=1, aspect="equal")
            _draw_annulus(sac_axes[i], sac_arr.shape, color="white", lw=0.8)
            sac_axes[i].set_title(
                f"SAC  g={g_score:+.3f}",
                fontsize=9, fontweight="bold", pad=3, color=color)
            sac_axes[i].axis("off")
        else:
            sac_axes[i].text(0.5, 0.5, "SAC unavailable",
                              ha="center", va="center",
                              transform=sac_axes[i].transAxes, fontsize=9)
            sac_axes[i].axis("off")

    fig.suptitle(
        "Grid-cell signature in LSTM hidden states\n"
        "(Gridness score = min(r₆₀, r₁₂₀) − max(r₃₀, r₉₀, r₁₅₀) in SAC annulus)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-cell signature figure")
    p.add_argument("--data", required=True,
                   help="JSON from scripts/probing/grid_cell_signature.py")
    p.add_argument("--out", default="docs/manuscript/fig/grid_cell_signature.pdf",
                   help="Output PDF/PNG path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = _load_data(Path(args.data))
    make_figure(data, Path(args.out))
