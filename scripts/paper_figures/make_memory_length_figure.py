"""
WJ-A: memory-length-budget sweep figure (Wijmans Fig 2 replication).

For each condition, plots top-layer GPS $R^2$ as a function of the LSTM's
effective memory length K (number of within-episode steps over which the
LSTM accumulates history before its hidden state is reset). At K=1 the
hidden state contains only the current step; at K large it carries the
agent's full episode history.

H1 mechanism prediction: GPS becomes recoverable from h only with enough
within-episode history; the encoder cannot substitute for temporal
integration when the GPS sensor itself is the only continuous global
spatial signal. We expect a positive R^2 at large K for every
condition; the rate at which R^2 rises with K characterises how much
sensor-integration each agent's hidden state performs.

Reads two formats (preferred order):
  1. --summary-dir/{cond}_v3.json with `ks: [{K, gps_r2, gps_r2_std}, ...]`
     (used in the paper; high-n single-split)
  2. --analysis-dir/{cond}_k{K}_det_analysis.json (per-K 5-fold CV;
     small-n, used as fallback if v3 summary missing)

Writes: <out-dir>/fig_memlen_sweep.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()
import numpy as np


CONDS = [
    ("blind",     "Blind",    "#444444", "o"),
    ("matched",   "Coarse",   "#377eb8", "s"),
    ("uniform",   "Uniform",  "#4daf4a", "^"),
    ("foveated",  "Foveated", "#e41a1c", "D"),
]
KS = [1, 4, 16, 64, 256, 1000, 100000]
KS_LABELS = ["1", "4", "16", "64", "256", "1000", "full ep."]


def _read_v3(summary_dir: Path, cond_key: str):
    """Return (ks, r2s, r2_stds) from <cond>_v3.json or (None, None, None)."""
    p = summary_dir / f"{cond_key}_v3.json"
    if not p.exists():
        return None, None, None
    d = json.loads(p.read_text())
    by_k = {entry["K"]: entry for entry in d.get("ks", [])}
    ks_x, r2s, r2_stds = [], [], []
    for K in KS:
        if K not in by_k:
            ks_x.append(K); r2s.append(np.nan); r2_stds.append(np.nan)
            continue
        e = by_k[K]
        ks_x.append(K)
        r2s.append(float(e.get("gps_r2", np.nan)))
        r2_stds.append(float(e.get("gps_r2_std", 0.0)))
    return np.asarray(ks_x), np.asarray(r2s), np.asarray(r2_stds)


def _read_per_k(analysis_dir: Path, cond_key: str):
    ks_x, r2s, r2_stds = [], [], []
    for K in KS:
        p = analysis_dir / f"{cond_key}_k{K}_det_analysis.json"
        if not p.exists():
            ks_x.append(K); r2s.append(np.nan); r2_stds.append(np.nan)
            continue
        block = json.loads(p.read_text()).get("1b_global_gps_compass", {})
        r2 = block.get("gps_cv_r2_mean", block.get("gps_r2", np.nan))
        std = block.get("gps_cv_r2_std", 0.0)
        ks_x.append(K)
        r2s.append(float(r2) if r2 is not None else np.nan)
        r2_stds.append(float(std) if std is not None else 0.0)
    return np.asarray(ks_x), np.asarray(r2s), np.asarray(r2_stds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", type=Path, default=None,
                    help="Dir with {cond}_v3.json (preferred)")
    ap.add_argument("--analysis-dir", type=Path, default=None,
                    help="Dir with {cond}_k{K}_det_analysis.json (fallback)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, ls=":", color="grey", alpha=0.4, lw=0.7)

    for cond_key, label, colour, marker in CONDS:
        ks_x = r2s = r2_stds = None
        if args.summary_dir is not None:
            ks_x, r2s, r2_stds = _read_v3(args.summary_dir, cond_key)
            if ks_x is None or np.all(np.isnan(r2s)):
                ks_x = r2s = r2_stds = None
        if ks_x is None and args.analysis_dir is not None:
            ks_x, r2s, r2_stds = _read_per_k(args.analysis_dir, cond_key)
        if ks_x is None:
            print(f"  warn: no data for {cond_key}", file=sys.stderr)
            continue

        # Drop missing points so the line connects what's there.
        mask = ~np.isnan(r2s)
        if mask.sum() == 0:
            continue
        r2_clip = np.clip(r2s[mask], -1.5, 1.05)
        ax.errorbar(ks_x[mask], r2_clip, yerr=r2_stds[mask],
                    marker=marker, label=label, color=colour,
                    linewidth=1.8, markersize=7, capsize=3, alpha=0.95)

    ax.set_xscale("log")
    ax.set_xlabel("LSTM memory budget $K$ (steps)", fontweight="bold")
    ax.set_ylabel("Top-layer GPS $R^2$", fontweight="bold")
    ax.set_xticks(KS)
    ax.set_xticklabels(KS_LABELS)
    ax.set_ylim(-0.45, 0.65)
    ax.set_title("Memory budget controls GPS recoverability across all conditions",
                 loc="left", pad=8, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    ax.grid(linestyle=":", alpha=0.3)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    plt.tight_layout()
    out = args.out_dir / "fig_memlen_sweep.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
