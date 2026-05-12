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

import json

# Order matches transplant matrix (figa7a) for visual pairing.
COND_KEYS = ["blind", "coarse", "foveated_logpolar", "foveated", "uniform"]
COND_LABELS = ["Blind", "Coarse", "Log-polar", "Foveated", "Uniform"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-json", type=Path,
                    default=Path("/tmp/probe_transfer_5x5.json"),
                    help="JSON written by probe_transfer_5x5.py")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load 5×5 matrix; reorder to COND_KEYS.
    raw = json.loads(args.data_json.read_text())
    src_idx = {c: i for i, c in enumerate(raw["conds"])}
    src_M = np.array(raw["matrix"])
    M = np.array([[src_M[src_idx[r], src_idx[c]] for c in COND_KEYS]
                  for r in COND_KEYS], dtype=float)
    CONDS = COND_LABELS

    # Signed log10 transform for colour mapping (compress dynamic range
    # while preserving sign).
    M_log = np.sign(M) * np.log10(np.abs(M) + 1.0)
    vmax = float(np.nanmax(np.abs(M_log)))

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(M_log, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal")

    n = len(CONDS)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(CONDS, rotation=20, ha="right")
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
    out = args.out_dir / "figa7b_h2_probe_transfer.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()