"""
H2 cross-condition probe-transfer heatmap.

Replaces the CKA heatmap (which sits at the noise floor for every
off-diagonal cell and conveys little) with the cross-condition probe-
transfer matrix (Table 2 in the paper). The transfer values span
[+0.99 .. -110,000+], so we plot signed log10(|x|+1) for colour and
annotate cells with the actual rounded value.

The story this figure tells: train a linear GPS probe on condition
X's hidden states, test on Y. The diagonal is high (self-probe works);
the off-diagonal is catastrophically negative (predictions are
off-manifold), with strong asymmetry: probes trained on bottleneck
conditions fail more spectacularly when applied to rich-encoder hidden
states than vice versa.

Data are hard-coded from analyze_cross.py outputs (paper Table 2).

Writes: <out-dir>/h2_probe_transfer.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

CONDS = ["Blind", "Coarse\n(1×1)", "Uniform", "Foveated\n(fix)", "Foveated\n(learned)"]

# Train (row) x Test (col) probe-transfer R²
# (paper Table 2). Order matches transplant matrix (fig 4b) for visual
# pairing: [Blind, Coarse, Uniform, Foveated-fix, Foveated-learned].
M = np.array([
    # Blind   Coarse    Uniform   Fov-fix   Fov-lrn
    [+0.99,   -55706,   -844,     -11301,   -10731],   # Blind
    [-10621,  +0.96,    -4832,    -4984,    -58169],   # Coarse
    [-1383,   -110317,  +0.90,    -7046,    -3305],    # Uniform
    [-2091,   -39100,   -2792,    +0.92,    -1263],    # Fov-fix
    [-6722,   -38842,   -2423,    -6205,    +0.89],    # Fov-lrn
], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Signed log10 transform for colour mapping (compress dynamic range
    # while preserving sign).
    M_log = np.sign(M) * np.log10(np.abs(M) + 1.0)
    vmax = float(np.nanmax(np.abs(M_log)))

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M_log, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal")

    n = len(CONDS)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(CONDS)
    ax.set_yticklabels(CONDS)
    ax.set_xlabel("Test on")
    ax.set_ylabel("Train probe on")

    # Annotate cells: diagonal with actual R², off-diagonal with
    # (potentially-clipped) integer.
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            colour = "white" if abs(M_log[i, j]) > 0.55 * vmax else "black"
            if i == j:
                txt = f"{v:+.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=colour)
            else:
                # Format as compact integer (e.g. -1.4k for -1383)
                if abs(v) >= 1000:
                    txt = f"{v / 1000:+.1f}k"
                else:
                    txt = f"{int(round(v)):+d}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9.5, color=colour)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("GPS $R^2$ (sign-log scale)")
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Cross-condition GPS probe transfer")

    plt.tight_layout()
    out = args.out_dir / "fig4_h2_probe_transfer.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
