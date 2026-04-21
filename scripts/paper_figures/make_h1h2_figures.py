"""
Figure generators for H1 (compensatory memory) and H2 (representational
divergence), for the CS503 final paper §4 Results.

Inputs
------
One ``*_analysis.json`` per condition (produced by
``scripts/probing/analyze.py``) plus an optional
``cross_analysis.json`` (from ``analyze_cross.py``) for H2.

Output
------
PDF + PNG figures in ``docs/cs503_final/fig/`` ready for
includegraphics.

Usage
-----
    python scripts/paper_figures/make_h1h2_figures.py \
        --in-dir  /scratch/izar/wxu/probing_results \
        --out-dir docs/cs503_final/fig

If any condition JSON is missing, the figure skips that condition and
warns to stderr rather than failing (so it can be re-run incrementally
as new training/probing finishes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Condition → display name + colour (consistent across every figure)
COND_DISPLAY = {
    "blind": ("Blind", "#444444"),
    "uniform": ("Uniform", "#4daf4a"),
    "foveated": ("Foveated (fixed)", "#e41a1c"),
    "foveated_learned": ("Foveated (learned)", "#ff7f00"),
    "matched": ("Matched-compute", "#377eb8"),
}
COND_ORDER = ["blind", "uniform", "foveated", "foveated_learned", "matched"]


def _load_analysis(in_dir: Path, cond: str) -> dict | None:
    """Load an analysis JSON; returns None and warns if missing or invalid."""
    path = in_dir / f"{cond}_gibson_analysis.json"
    if not path.exists():
        sys.stderr.write(f"[skip] {cond}: {path} not found\n")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[skip] {cond}: invalid JSON ({e})\n")
        return None


# ---------------------------------------------------------------------------
# H1.a — Path-history retention (star figure for H1)
# ---------------------------------------------------------------------------


def fig_path_history(in_dir: Path, out_dir: Path) -> None:
    """R² of decoding past GPS(t-k) from current hidden h(t), vs lag k.

    The H1 prediction is that the foveated agent retains temporal spatial
    memory for longer (slower decay, higher R² at large k) than the
    sighted uniform baseline, because its degraded visual input forces
    greater reliance on memory.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    for cond in COND_ORDER:
        d = _load_analysis(in_dir, cond)
        if d is None or "2c_path_history" not in d:
            continue
        entries = d["2c_path_history"]
        lags = [r["lag_k"] for r in entries]
        r2s = [r["r2"] for r in entries]
        label, colour = COND_DISPLAY[cond]
        ax.plot(
            lags, r2s, "-o", color=colour, label=label, linewidth=2,
            markersize=5,
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("lag $k$ (steps into the past)")
    ax.set_ylabel(r"position probe $R^2$ on GPS($t-k$)")
    ax.set_title("Path-history decoding across conditions (H1)")
    ax.set_ylim(-1.8, 1.0)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"h1_path_history.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# H1.b — Accuracy vs timestep bin
# ---------------------------------------------------------------------------


def fig_accuracy_vs_timestep(in_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    bins = ["t=0", "t=1", "t=2", "t≥3"]
    x = np.arange(len(bins))
    width = 0.15
    offsets = np.linspace(-2 * width, 2 * width, len(COND_ORDER))

    for off, cond in zip(offsets, COND_ORDER):
        d = _load_analysis(in_dir, cond)
        if d is None or "2a_accuracy_vs_timestep" not in d:
            continue
        entries = {r["timestep_bin"]: r for r in d["2a_accuracy_vs_timestep"]}
        r2s = [entries.get(b, {}).get("gps_r2", np.nan) for b in bins]
        label, colour = COND_DISPLAY[cond]
        ax.bar(x + off, r2s, width, label=label, color=colour, alpha=0.9)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("timestep bin within episode")
    ax.set_ylabel(r"GPS probe $R^2$")
    ax.set_title("Spatial encoding improves over an episode (H1 timing)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(-3.0, 1.0)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"h1_accuracy_vs_timestep.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# H1.c — Global GPS probe + selectivity summary (sanity + info content)
# ---------------------------------------------------------------------------


def fig_global_probe_summary(in_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    names, gps_r2, sel = [], [], []
    colours = []
    for cond in COND_ORDER:
        d = _load_analysis(in_dir, cond)
        if d is None or "1b_global_gps_compass" not in d:
            continue
        names.append(COND_DISPLAY[cond][0])
        gps_r2.append(d["1b_global_gps_compass"]["gps_r2"])
        sel.append(d["1ef_control_selectivity"]["gps_selectivity"])
        colours.append(COND_DISPLAY[cond][1])

    x = np.arange(len(names))
    ax.bar(x - 0.2, gps_r2, 0.38, color=colours, label=r"GPS $R^2$", alpha=0.95)
    ax.bar(x + 0.2, sel, 0.38, color=colours, alpha=0.55, hatch="//",
           label="Hewitt-Liang selectivity")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Spatial information in recurrent memory")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"h1_global_probe.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# H2 — Cross-condition CKA heatmap
# ---------------------------------------------------------------------------


def fig_cka_heatmap(in_dir: Path, out_dir: Path) -> None:
    cross_path = in_dir / "cross_analysis_5cond.json"
    if not cross_path.exists():
        cross_path = in_dir / "cross_analysis.json"
    if not cross_path.exists():
        sys.stderr.write(f"[skip] cka: no cross_analysis*.json in {in_dir}\n")
        return
    try:
        with open(cross_path) as f:
            cx = json.load(f)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[skip] cka: cannot parse {cross_path}: {e}\n")
        return

    # analyze_cross.py structure: expect a 'cka' (or '3a_cka') key with
    # condition pairs → CKA value. Try a few likely layouts.
    cka_block = None
    for k in ("3a_cka_position_aligned", "3a_cka", "cka", "position_aligned_cka"):
        if k in cx:
            cka_block = cx[k]
            break
    if cka_block is None:
        sys.stderr.write(
            f"[skip] cka: no recognised CKA key in {cross_path} "
            f"(keys = {list(cx.keys())})\n"
        )
        return

    # Build symmetric matrix from pair entries {'condA|condB': float, ...}
    conds = sorted({c for pair in cka_block for c in pair.split("|")})
    mat = np.full((len(conds), len(conds)), np.nan)
    for i, a in enumerate(conds):
        for j, b in enumerate(conds):
            if i == j:
                mat[i, j] = 1.0
            elif f"{a}|{b}" in cka_block:
                mat[i, j] = cka_block[f"{a}|{b}"]
            elif f"{b}|{a}" in cka_block:
                mat[i, j] = cka_block[f"{b}|{a}"]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(conds)))
    ax.set_yticks(range(len(conds)))
    labels = [COND_DISPLAY.get(c, (c, "#888"))[0] for c in conds]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(conds)):
        for j in range(len(conds)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    ax.set_title("Position-aligned CKA (H2)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="CKA")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"h2_cka_heatmap.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True,
                   help="directory with {cond}_gibson_analysis.json files")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="output directory for figures")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig_path_history(args.in_dir, args.out_dir)
    fig_accuracy_vs_timestep(args.in_dir, args.out_dir)
    fig_global_probe_summary(args.in_dir, args.out_dir)
    fig_cka_heatmap(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
