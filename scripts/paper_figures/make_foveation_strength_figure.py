"""
Foveation-strength sweep figure.

Probes 5 foveated agents trained at $\\sigma_{\\max} \\in \\{2, 4, 8, 12, 20\\}$
(F1-F4 + the existing $\\sigma_{\\max}=8$ checkpoint).  Plots GPS R²,
compass R², and SPL on a single x-axis ($\\sigma_{\\max}$).

The encoder--memory race predicts a continuous lever: at low blur the
foveated agent should look like uniform (encoder still resolves world-
frame, low LSTM GPS R²); at high blur it should approach matched-
compute (LSTM compensates with a stable GPS code).  The crossing point
is the testable signature.

Reads:
  - <results-dir>/foveated_sigma<S>_gibson_det_analysis.json (per-S)
    where S in {2, 4, 8, 12, 20}; for S=8 also accepts
    foveated_gibson_det_analysis.json (the existing baseline).
  - Anchor: blind / matched / uniform overall numbers (hardcoded; same
    as Table 1) drawn as horizontal reference lines.

Writes: <out-dir>/foveation_strength.{pdf,png}

Usage:
    python scripts/paper_figures/make_foveation_strength_figure.py \\
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

SIGMAS = [2, 4, 8, 12, 20]
# Reference anchors from Table 1
REF = {
    "blind":   {"gps_r2": 0.95, "compass_r2": 0.81, "label": "Blind"},
    "matched": {"gps_r2": 0.78, "compass_r2": 0.64, "label": "Coarse (1×1)"},
    "uniform": {"gps_r2": -0.31, "compass_r2": 0.36, "label": "Uniform"},
}


def load_for_sigma(results_dir: Path, sigma: int) -> dict | None:
    # Sigma=8 is the existing baseline (no special suffix in name)
    if sigma == 8:
        candidates = ["foveated_gibson_det_analysis.json",
                      "foveated_sigma8_gibson_det_analysis.json"]
    else:
        candidates = [f"foveated_sigma{sigma}_gibson_det_analysis.json"]
    for c in candidates:
        p = results_dir / c
        if p.exists():
            return json.loads(p.read_text())
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clip-min", type=float, default=-1.5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-sigma points
    rows = []
    for s in SIGMAS:
        d = load_for_sigma(args.results_dir, s)
        if d is None:
            print(f"[skip] sigma={s}: missing JSON")
            continue
        b = d.get("1b_global_gps_compass", {})
        rows.append({
            "sigma": s,
            "gps_r2": b.get("gps_cv_r2_mean"),
            "gps_std": b.get("gps_cv_r2_std", 0.0),
            "compass_r2": b.get("compass_cv_r2_mean"),
            "compass_std": b.get("compass_cv_r2_std", 0.0),
        })

    if not rows:
        raise RuntimeError("no foveated sigma JSONs found")

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # Reference horizontal lines
    palette = {
        "blind":   "#444444",
        "matched": "#377eb8",
        "uniform": "#4daf4a",
    }
    for k, v in REF.items():
        ax.axhline(v["gps_r2"], color=palette[k], linestyle=":",
                   alpha=0.65, lw=1.0,
                   label=f"{v['label']} (anchor)")

    # Foveation curve (GPS only on this primary axis; compass on a
    # twin axis would clutter — keep simple and put compass in caption).
    xs = np.array([r["sigma"] for r in rows])
    ys = np.array([r["gps_r2"] for r in rows])
    es = np.array([r["gps_std"] for r in rows])
    ys_clip = np.clip(ys, args.clip_min, 1.05)
    ax.errorbar(xs, ys_clip, yerr=es, color="#e41a1c",
                marker="D", markersize=7, linewidth=1.8,
                capsize=3.5, capthick=0.9, elinewidth=0.7,
                label="Foveated", zorder=5)
    for i, r in enumerate(rows):
        if r["gps_r2"] < args.clip_min:
            ax.annotate(f"{r['gps_r2']:.1f}", (xs[i], args.clip_min + 0.10),
                        ha="center", fontsize=7, color="darkred")

    ax.set_xlabel("$\\sigma_{\\max}$ (foveation Gaussian-blur strength)",
                  fontsize=10)
    ax.set_ylabel("GPS $R^2$ (5-fold CV mean)", fontsize=10)
    ax.set_xticks(SIGMAS)
    ax.set_ylim(args.clip_min - 0.05, 1.10)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False, ncol=2)
    ax.set_title("Foveation strength sweep: GPS encoding vs.\\ blur strength",
                 fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out_dir / f"foveation_strength.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
