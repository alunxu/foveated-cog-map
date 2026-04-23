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
    "matched": ("Matched-compute", "#377eb8"),  # matched-48 is the probed run; paper text calls it Matched-compute
}
COND_ORDER = ["blind", "uniform", "foveated", "foveated_learned", "matched"]


def _load_analysis(in_dir: Path, cond: str) -> dict | None:
    """Load an analysis JSON; returns None and warns if missing or invalid.

    For foveated_learned we prefer the truncated (matched-distribution)
    analysis because its full 85k-step probe dataset covers a spatial
    range ~40x wider than the other conditions, making direct comparison
    of R^2 and selectivity values misleading.
    """
    candidates = (
        [f"{cond}_gibson_truncated_analysis.json", f"{cond}_gibson_analysis.json"]
        if cond == "foveated_learned"
        else [f"{cond}_gibson_analysis.json"]
    )
    for name in candidates:
        path = in_dir / name
        if not path.exists():
            continue
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[skip] {cond}: invalid JSON ({e})\n")
            return None
    sys.stderr.write(f"[skip] {cond}: no analysis JSON found\n")
    return None


# ---------------------------------------------------------------------------
# H1.a — Path-history retention (star figure for H1)
# ---------------------------------------------------------------------------


def fig_path_history(in_dir: Path, out_dir: Path) -> None:
    """R² of decoding past GPS(t-k) from current hidden h(t), vs lag k.

    H1 prediction: the foveated agent retains temporal spatial memory for
    longer (slower decay, higher R² at large k) than the sighted uniform
    baseline, because its degraded visual input forces greater reliance
    on memory.

    Foveated-learned is EXCLUDED from this figure. The lag-k probe
    requires k+1 consecutive within-episode steps; applying the 4-step
    cross-condition truncation (§Methods) to foveated-learned leaves
    100-200 samples at lag k>=2 with near-zero target variance (the agent
    has barely moved), which produces uninterpretable numbers (R² in
    {-10 sentinel, +1 trivial}). These are sample-count/variance
    artefacts, not representation-quality signal. Paper text §4.3
    discusses this exclusion explicitly.
    """
    # H1 comparison conditions: exclude fov-learned (see docstring)
    h1_order = [c for c in COND_ORDER if c != "foveated_learned"]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    for cond in h1_order:
        d = _load_analysis(in_dir, cond)
        if d is None or "2c_path_history" not in d:
            continue
        block = d["2c_path_history"]
        entries = block["lags"] if isinstance(block, dict) and "lags" in block else block
        entries = [r for r in entries if isinstance(r, dict) and "r2" in r]
        if not entries:
            continue
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
    # Prefer the 5-condition unaligned CKA (Kornblith 2019 standard).
    # Falls back to position-aligned (analyze_cross.py output) if only
    # that is available.
    unaligned = in_dir / "cka_unaligned_5cond.json"
    if unaligned.exists():
        try:
            with open(unaligned) as f:
                uc = json.load(f)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[skip] cka: cannot parse {unaligned}: {e}\n")
            return
        conds = uc["conditions"]
        mat = np.full((len(conds), len(conds)), np.nan)
        for i, a in enumerate(conds):
            for j, b in enumerate(conds):
                mat[i, j] = uc["cka_matrix"][a][b]
        n_samples = uc.get("n_samples_per_condition", None)
    else:
        cross_path = in_dir / "cross_analysis.json"
        if not cross_path.exists():
            sys.stderr.write("[skip] cka: no cka_unaligned or cross_analysis\n")
            return
        try:
            with open(cross_path) as f:
                cx = json.load(f)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[skip] cka: cannot parse {cross_path}: {e}\n")
            return
        cka_block = cx.get("3a_cka")
        if cka_block is None:
            sys.stderr.write(f"[skip] cka: no 3a_cka key in {cross_path}\n")
            return
        conds = cx.get("conditions") or sorted({
            p for key in cka_block for p in key.split("_vs_")
        })
        mat = np.full((len(conds), len(conds)), np.nan)
        for i, a in enumerate(conds):
            for j, b in enumerate(conds):
                if i == j:
                    mat[i, j] = 1.0
                    continue
                key = f"{a}_vs_{b}" if f"{a}_vs_{b}" in cka_block else f"{b}_vs_{a}"
                entry = cka_block.get(key, {})
                v = entry.get("cka_top_layer") or entry.get("cka_aligned")
                mat[i, j] = v if v is not None else np.nan
        n_samples = None

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    # Use a log-style colormap so the near-zero off-diagonal values are
    # visible; clip vmax to 0.05 so the diagonal (1.0) saturates.
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=0.05)
    ax.set_xticks(range(len(conds)))
    ax.set_yticks(range(len(conds)))
    labels = [COND_DISPLAY.get(c, (c, "#888"))[0] for c in conds]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(conds)):
        for j in range(len(conds)):
            v = mat[i, j]
            if not np.isnan(v):
                txt = f"{v:.3f}" if i != j else "1.00"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if v < 0.03 else "black", fontsize=7)
    title = "Unaligned linear CKA (H2)"
    if n_samples:
        title += f"  (n={n_samples})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="CKA (clipped at 0.05)")
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
