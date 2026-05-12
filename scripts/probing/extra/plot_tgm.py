"""Plot TGM heatmaps for 5 conditions side-by-side, plus a cross-condition
ranking by off-diagonal extent.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]
NICE = {
    "blind": "blind",
    "coarse": "coarse 1×1",
    "foveated": "foveated 4×4",
    "uniform": "uniform 4×4",
    "foveated_logpolar": "fov-logpolar",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgm_npz", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    d = np.load(args.tgm_npz)
    available = [c for c in CONDITIONS if c in d.files]
    n = len(available)

    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0), constrained_layout=True,
                              sharey=True)
    if n == 1:
        axes = [axes]

    # Determine common color scale
    all_finite = []
    for c in available:
        m = d[c]
        all_finite.append(m[np.isfinite(m)])
    vmax = float(np.percentile(np.concatenate(all_finite), 95)) if all_finite else 1.0
    vmin = max(-0.5, float(np.percentile(np.concatenate(all_finite), 5))) if all_finite else 0.0

    for i, c in enumerate(available):
        m = d[c]
        ax = axes[i]
        im = ax.imshow(m, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                        aspect="equal")
        ax.set_xlabel("test step", fontsize=9)
        if i == 0:
            ax.set_ylabel("train step", fontsize=9)
        ax.set_title(NICE.get(c, c), fontsize=10)
        # Diagonal line for reference
        T = m.shape[0]
        ax.plot([0, T-1], [0, T-1], "w-", linewidth=0.5, alpha=0.7)

    fig.colorbar(im, ax=axes, shrink=0.8, label="goal-vec R²", aspect=20)
    fig.suptitle("Temporal Generalisation Matrix (King–Dehaene 2014) — goal-vec decoder",
                  fontsize=11, y=1.05)
    fig.savefig(args.out_path, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_path}")

    # Print summary table
    print("\nSummary (mean over diag and off-diag tri):")
    print(f"  {'cond':>20s} {'diag':>8s} {'off-tri':>8s} {'block?':>8s}")
    for c in available:
        m = d[c]
        diag = float(np.nanmean(np.diagonal(m)))
        off = float(np.nanmean(m[np.tril_indices(m.shape[0], k=-5)]))
        block = "yes" if (off > 0.5 * diag and diag > 0) else "no"
        print(f"  {c:>20s} {diag:>8.3f} {off:>8.3f} {block:>8s}")


if __name__ == "__main__":
    main()
