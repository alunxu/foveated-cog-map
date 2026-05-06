"""
Discussion §5.1 synthesis figure: place the 5 conditions in a 2D
(H1 magnitude × H2 format isolation) plane, with marker size encoding
the third dimension (behavioural memory reliance from §4.5 shortcut).

x-axis (H1 magnitude): top-layer GPS $R^2$ on $\mathbf{h}_2$ (Gibson, 5-fold CV).
y-axis (H2 format isolation): negation of average row of the 5x5
  cross-condition memory-transplant matrix (off-diagonal cells only).
  Higher Y = this condition's hidden state is more disruptive to OTHER
  conditions' policies = more format-isolated as donor.
marker size: shortcut SPL drop % (= 100 × (reset - persist) / reset).
  Bigger marker = policy more dependent on persistent recurrent memory.

This panel REPLACES what was previously a separate H1×behaviour scatter
in §4.5, since both are 2D scatters of the same 5 conditions on the
same X-axis (top-layer GPS R²).

Caveat: coarse-as-donor has only 1 measured cell (coarse→blind);
its Y is single-cell, marked with a hollow marker + footnote in caption.

Reads:  /tmp/transplant_local/<donor>_to_<recipient>.json (mid=30)
        /tmp/probing_results_local/<cond>_gibson_det_analysis.json
        data/shortcut/<cond>_gibson.json
Writes: docs/manuscript/fig/fig8_synthesis.pdf
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
    # (key,           label,                colour,    marker)
    ("blind",            "Blind",             "#444444", "o"),
    ("matched",          "Coarse",      "#377eb8", "s"),
    ("uniform",          "Uniform",           "#4daf4a", "^"),
    ("foveated",         "Foveated",    "#e41a1c", "D"),
]



def load_h1_magnitude(probing_dir: Path) -> dict[str, float]:
    """Top-layer GPS R^2 per condition (Gibson, 5-fold CV)."""
    out = {}
    for cond_key, *_ in CONDS:
        p = probing_dir / f"{cond_key}_gibson_det_analysis.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        v = d.get("1b_global_gps_compass", {}).get("gps_cv_r2_mean")
        if v is not None:
            out[cond_key] = v
    return out


def load_donor_toxicity(transplant_dir: Path) -> tuple[dict[str, float], dict[str, int]]:
    """Average cross-transplant cost when this condition is donor.

    For each donor D, average over recipients R != D where pair JSON exists.
    The cost is (cross_transplant_spl - recipient_self_spl), where the
    self-spl is recipient-specific (collected from the same pair files).

    Returns (avg_cost dict, n_cells dict).
    """
    # First pass: recipient_self_spl per recipient
    recip_self: dict[str, float] = {}
    for fp in sorted(transplant_dir.glob("*_to_*.json")):
        # only mid=30 (default; files without _midN are mid30)
        if "_mid" in fp.stem:
            continue
        # parse donor / recipient
        parts = fp.stem.split("_to_")
        donor, recip = parts[0], parts[1]
        d = json.loads(fp.read_text())
        if recip not in recip_self:
            recip_self[recip] = d["self_transplant"]["mean_spl"]

    # Second pass: avg cost per donor
    avg_cost: dict[str, float] = {}
    n_cells: dict[str, int] = {}
    for cond_key, *_ in CONDS:
        costs = []
        for fp in sorted(transplant_dir.glob(f"{cond_key}_to_*.json")):
            if "_mid" in fp.stem:
                continue
            recip = fp.stem.split("_to_")[1]
            if recip == cond_key:  # self-transplant — skip
                continue
            d = json.loads(fp.read_text())
            cost = d["cross_transplant"]["mean_spl"] - recip_self.get(
                recip, d["self_transplant"]["mean_spl"]
            )
            costs.append(cost)
        if costs:
            avg_cost[cond_key] = float(np.mean(costs))
            n_cells[cond_key] = len(costs)
    return avg_cost, n_cells


def load_shortcut_drop(shortcut_dir: Path) -> dict[str, float]:
    """Shortcut SPL drop % per condition: 100 × (reset - persist) / reset."""
    out = {}
    for cond_key, *_ in CONDS:
        p = shortcut_dir / f"{cond_key}_gibson.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        reset = d.get("reset_mean_spl")
        persist = d.get("persistent_mean_spl")
        if reset is None or persist is None or reset <= 0:
            continue
        out[cond_key] = 100.0 * (reset - persist) / reset
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probing-dir", type=Path, required=True)
    ap.add_argument("--transplant-dir", type=Path, required=True)
    ap.add_argument("--shortcut-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gps_r2 = load_h1_magnitude(args.probing_dir)
    avg_cost, n_cells = load_donor_toxicity(args.transplant_dir)
    shortcut_drop = load_shortcut_drop(args.shortcut_dir)

    print("Loaded:")
    for k in [c[0] for c in CONDS]:
        print(f"  {k:20s} GPS R^2={gps_r2.get(k, 'N/A'):>7} | "
              f"donor toxicity={avg_cost.get(k, 'N/A'):>8} "
              f"({n_cells.get(k, 0)} cells) | "
              f"shortcut SPL drop={shortcut_drop.get(k, 'N/A')}%")

    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    GPS_THRESH = 0.4
    TOX_THRESH = 0.12  # "above" = format-isolated as donor
    CLIP_X_MIN = -0.5  # plot x-axis lower bound

    # Quadrant background tints — make the 4 regions visually distinct.
    Q_COLORS = {
        "ul": (0.97, 0.93, 1.00),   # no-GPS + isolated  (light purple)
        "ur": (0.93, 1.00, 0.93),   # GPS + isolated     (light green)
        "ll": (1.00, 0.95, 0.92),   # no-GPS + shared    (light salmon)
        "lr": (1.00, 0.99, 0.86),   # GPS + shared       (light gold)
    }
    ax.axhspan(TOX_THRESH, 1, xmin=0, xmax=(GPS_THRESH - (CLIP_X_MIN - 0.05)) /
               (1.10 - (CLIP_X_MIN - 0.05)), color=Q_COLORS["ul"], zorder=0)
    ax.axhspan(TOX_THRESH, 1, xmin=(GPS_THRESH - (CLIP_X_MIN - 0.05)) /
               (1.10 - (CLIP_X_MIN - 0.05)), xmax=1, color=Q_COLORS["ur"], zorder=0)
    ax.axhspan(-1, TOX_THRESH, xmin=0, xmax=(GPS_THRESH - (CLIP_X_MIN - 0.05)) /
               (1.10 - (CLIP_X_MIN - 0.05)), color=Q_COLORS["ll"], zorder=0)
    ax.axhspan(-1, TOX_THRESH, xmin=(GPS_THRESH - (CLIP_X_MIN - 0.05)) /
               (1.10 - (CLIP_X_MIN - 0.05)), xmax=1, color=Q_COLORS["lr"], zorder=0)

    ax.axhline(TOX_THRESH, ls="--", color="#666", lw=1.2, zorder=1)
    ax.axvline(GPS_THRESH, ls="--", color="#666", lw=1.2, zorder=1)

    # Marker-size scaling: behavioural memory reliance (shortcut SPL drop %).
    drop_min = min(shortcut_drop.values())
    drop_max = max(shortcut_drop.values())
    SIZE_MIN, SIZE_MAX = 110.0, 600.0

    def size_for(drop_pct: float) -> float:
        if drop_max == drop_min:
            return (SIZE_MIN + SIZE_MAX) / 2
        t = (drop_pct - drop_min) / (drop_max - drop_min)
        return SIZE_MIN + t * (SIZE_MAX - SIZE_MIN)

    # Plot points
    for cond_key, label, colour, marker in CONDS:
        if cond_key not in gps_r2 or cond_key not in avg_cost:
            continue
        x = max(gps_r2[cond_key], CLIP_X_MIN)
        y = -avg_cost[cond_key]
        drop = shortcut_drop.get(cond_key)
        s = size_for(drop) if drop is not None else 200
        clipped = gps_r2[cond_key] < CLIP_X_MIN
        n = n_cells[cond_key]
        if n < 3:
            facecolor = "white"
            edgecolor = colour
            edge_lw = 2.4
        else:
            facecolor = colour
            edgecolor = "black"
            edge_lw = 1.1
        ax.scatter(x, y, s=s, c=facecolor, edgecolor=edgecolor,
                   linewidths=edge_lw, marker=marker, zorder=4)
        # Label
        offsets = {
            "Blind":              (-0.06, +0.014),
            "Coarse":       (+0.07, +0.018),
            "Uniform":            (-0.06, +0.014),
            "Foveated":     (+0.06, -0.013),
        }
        dx, dy = offsets.get(label, (0.05, 0.0))
        ha = "right" if dx < 0 else "left"
        annot = label
        if clipped:
            annot = f"{label}\n($R^2{{=}}{gps_r2[cond_key]:.1f}$)"
        if n < 3:
            annot = f"{label}\n($n_{{cells}}{{=}}{n}$)"
        ax.annotate(annot, (x, y), xytext=(x + dx, y + dy),
                    ha=ha, va="center", fontsize=11, fontweight="bold")

    # Quadrant labels — bigger, clearer, with concise descriptions.
    quad_kw = dict(transform=ax.transAxes, fontsize=10, color="#222",
                   style="italic", weight="bold", alpha=0.7)
    ax.text(0.97, 0.96, "GPS-readable\nformat-isolated",
            ha="right", va="top", **quad_kw)
    ax.text(0.03, 0.96, "no GPS\nformat-isolated",
            ha="left", va="top", **quad_kw)
    ax.text(0.97, 0.04, "GPS-readable\nformat-shared",
            ha="right", va="bottom", **quad_kw)
    ax.text(0.03, 0.04, "no GPS\nformat-shared",
            ha="left", va="bottom", **quad_kw)

    ax.set_title("Three-axis synthesis of sensor-structure effects",
                 loc="left", pad=10)
    ax.set_xlabel(r"H1 magnitude: top-layer GPS $R^2$ (probe on $\mathbf{h}_2$)",
                  fontsize=12, labelpad=6)
    ax.set_ylabel("H2 format isolation\n(donor transplant cost)",
                  fontsize=12, labelpad=6)
    ax.set_xlim(CLIP_X_MIN - 0.05, 1.10)
    ax.set_ylim(-0.04, 0.32)
    ax.tick_params(axis="both", labelsize=11)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.20)

    # Size legend: 3 reference circles (small / mid / large drop) placed
    # OUTSIDE the data area to avoid covering the Coarse marker in the
    # bottom-right quadrant.
    from matplotlib.lines import Line2D
    drop_values = sorted(shortcut_drop.values())
    legend_drops = [drop_values[0], drop_values[len(drop_values) // 2],
                    drop_values[-1]]
    handles = []
    for d in legend_drops:
        s = size_for(d)
        h = Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#888", markeredgecolor="black",
                   markersize=(s ** 0.5), label=f"{d:.0f}%")
        handles.append(h)
    leg = ax.legend(handles=handles,
                    title="shortcut SPL drop\n(behavioural\nmemory reliance)",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    fontsize=9.5, title_fontsize=9.5,
                    frameon=True, framealpha=0.95, labelspacing=1.6,
                    handletextpad=1.4, borderpad=0.8)
    leg.get_frame().set_edgecolor("#999")

    fig.tight_layout()
    out = args.out_dir / "fig8_synthesis.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()