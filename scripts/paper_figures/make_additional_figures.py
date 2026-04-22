"""
Additional figures for §4.6 Additional Analyses:
  - place_cells.pdf: bar chart of units with >1 bit of spatial info.
  - multilayer_heatmap.pdf: GPS/compass probe R^2 per LSTM layer, per condition.

Usage:
    python scripts/paper_figures/make_additional_figures.py \\
        --in-dir /scratch/izar/wxu/probing_results --out-dir docs/cs503_final/fig
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COND_DISPLAY = {
    "blind": ("Blind", "#444444"),
    "uniform": ("Uniform", "#4daf4a"),
    "foveated": ("Foveated (fixed)", "#e41a1c"),
    "foveated_learned": ("Foveated (learned)", "#ff7f00"),
    "matched": ("Matched-48", "#377eb8"),
}
COND_ORDER = ["blind", "uniform", "foveated", "foveated_learned", "matched"]


def _load(in_dir: Path, cond: str) -> dict | None:
    # For foveated_learned, prefer the truncated (matched-distribution) analysis
    # so numbers are comparable to the other conditions.
    for name in ([f"{cond}_gibson_truncated_analysis.json", f"{cond}_gibson_analysis.json"]
                 if cond == "foveated_learned"
                 else [f"{cond}_gibson_analysis.json"]):
        p = in_dir / name
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[skip] {p}: {e}\n")
                return None
    sys.stderr.write(f"[skip] {cond}: no analysis JSON\n")
    return None


def fig_place_cells(in_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    names, n1bit, mean_si = [], [], []
    colours = []
    for cond in COND_ORDER:
        d = _load(in_dir, cond)
        if d is None:
            continue
        rm = d.get("5a_rate_maps")
        if rm is None:
            continue
        names.append(COND_DISPLAY[cond][0])
        n1bit.append(rm.get("n_place_cells_1bit", np.nan))
        mean_si.append(rm.get("mean_spatial_info_bits", np.nan))
        colours.append(COND_DISPLAY[cond][1])

    x = np.arange(len(names))
    ax.bar(x, n1bit, 0.6, color=colours, alpha=0.92)
    for i, (v, m) in enumerate(zip(n1bit, mean_si)):
        ax.text(i, v + 10, f"{int(v)}", ha="center", va="bottom", fontsize=9)
        ax.text(i, v / 2, f"$\\bar s$={m:.1f}", ha="center", va="center", color="white", fontsize=7)
    ax.axhline(512, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(-0.3, 512 + 5, "max possible (512)", fontsize=7, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("LSTM units with $>$1 bit spatial info")
    ax.set_title("Spatial encoding is broadly distributed except in foveated-learned")
    ax.set_ylim(0, 560)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"place_cells.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def fig_multilayer(in_dir: Path, out_dir: Path) -> None:
    # Build a (5 conds) x (6 layer/state slots) matrix for GPS R^2 and compass R^2.
    # LSTM has 3 layers, each with (h, c) → 6 slots: L0h, L0c, L1h, L1c, L2h, L2c.
    layer_labels = ["L0 $h$", "L0 $c$", "L1 $h$", "L1 $c$", "L2 $h$", "L2 $c$"]
    gps_rows, compass_rows, row_labels, row_colours = [], [], [], []
    for cond in COND_ORDER:
        d = _load(in_dir, cond)
        if d is None:
            continue
        ml = d.get("1d_multilayer")
        if ml is None:
            continue
        # Build a lookup from (layer, state) to r2
        lookup = {(r["layer"], r["state"]): (r["gps_r2"], r["compass_r2"]) for r in ml}
        gps_row = [lookup.get((L, s), (np.nan, np.nan))[0] for L in range(3) for s in ("h", "c")]
        compass_row = [lookup.get((L, s), (np.nan, np.nan))[1] for L in range(3) for s in ("h", "c")]
        gps_rows.append(gps_row)
        compass_rows.append(compass_row)
        row_labels.append(COND_DISPLAY[cond][0])
        row_colours.append(COND_DISPLAY[cond][1])

    gps_arr = np.array(gps_rows)
    compass_arr = np.array(compass_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.0), sharey=True)
    for ax, arr, title in [
        (axes[0], gps_arr, "GPS probe $R^2$"),
        (axes[1], compass_arr, "Compass probe $R^2$"),
    ]:
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.6 else "black", fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7, label="$R^2$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"multilayer_heatmap.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_place_cells(args.in_dir, args.out_dir)
    fig_multilayer(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
