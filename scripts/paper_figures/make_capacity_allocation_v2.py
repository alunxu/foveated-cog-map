"""
Capacity-allocation principle, v2: top-layer h GPS R² vs encoder spatial bandwidth.

Reads from `/tmp/rcp_analysis/mlp_probe.json` — full 5-condition probe results
from the unified post-retrain analyze.py + mlp_rcp.sh pipeline. Replaces v1's
ckpt-sweep-time approximations with the converged 250M values used in Table 1.

Conditions (low-bandwidth → high-bandwidth):
  blind  → coarse (1×1) → foveated-logpolar (~2×2) → foveated (4×4 blur)
                                                  → uniform (4×4 unblur)

Writes: docs/manuscript/fig/fig_capacity_allocation.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MLP_JSON = Path("/tmp/rcp_analysis/mlp_probe.json")
OUT_PDF = REPO_ROOT / "docs/manuscript/fig/fig_capacity_allocation.pdf"

# (key in mlp_probe.json, label, encoder_cells, colour, marker)
CONDS = [
    ("blind_izar",        "Blind",     0,  "#444444", "o"),
    ("coarse",            "Coarse",    1,  "#377eb8", "s"),
    ("foveated_logpolar", "Fov-logpolar", 4,  "#984ea3", "v"),
    ("foveated",          "Foveated",  16, "#e41a1c", "D"),
    ("uniform",           "Uniform",   16, "#4daf4a", "^"),
]


def main():
    data = json.loads(MLP_JSON.read_text())
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Shaded regime bands
    ax.axhspan(0.4, 1.05, color="#dceedc", alpha=0.55, zorder=0)
    ax.axhspan(-2.5, 0.4, color="#fbe0dc", alpha=0.45, zorder=0)
    ax.axhline(0, color="#888", lw=0.6, ls="--", zorder=0)

    # Data points
    for key, label, cells, col, mk in CONDS:
        d = data[key]
        r2, sd = d["linear_r2_mean"], d["linear_r2_std"]
        # Jitter foveated/uniform x slightly so they don't overlap visually
        x = cells + (0.3 if key == "uniform" else (-0.3 if key == "foveated" else 0))
        ax.errorbar(x, r2, yerr=sd, fmt=mk, color=col, markersize=11,
                    markeredgecolor="white", markeredgewidth=1.2,
                    capsize=3, lw=1.5, zorder=4, label=label)
        # In-line label
        if key in ("blind_izar", "coarse"):
            ax.annotate(label, (x + 1.2, r2), fontsize=10, color=col,
                        weight="bold", va="center")
        elif key == "foveated_logpolar":
            ax.annotate("Fov-logpolar", (x + 0.7, r2 - 0.06), fontsize=10,
                        color=col, weight="bold", va="top")
        elif key == "foveated":
            ax.annotate("Foveated", (x - 1.0, r2 + 0.05), fontsize=10,
                        color=col, weight="bold", ha="right", va="bottom")
        elif key == "uniform":
            ax.annotate("Uniform", (x + 1.2, r2), fontsize=10,
                        color=col, weight="bold", va="center")

    # Trend line through main 4 conditions (blind, coarse, foveated, uniform)
    main_x = [0, 1, 16, 16]
    main_y = [data["blind_izar"]["linear_r2_mean"],
              data["coarse"]["linear_r2_mean"],
              data["foveated"]["linear_r2_mean"],
              data["uniform"]["linear_r2_mean"]]
    ax.plot([0, 1, 16], [main_y[0], main_y[1], np.mean(main_y[2:])],
            color="#888", lw=1.0, ls=":", zorder=1)

    # Regime labels
    ax.text(13.5, 0.85, "Bottleneck regime\n(integration route\ncarries position)",
            fontsize=9, color="#3a7d3a", ha="right", va="top", style="italic")
    ax.text(13.5, -2.1, "Rich-encoder regime\n(visual route carries position)",
            fontsize=9, color="#a02528", ha="right", va="bottom", style="italic")

    ax.set_xlabel("Encoder spatial output (cells)", fontsize=11)
    ax.set_ylabel(r"Top-layer $\mathbf{h}_2$ linear GPS $R^2$" + "\n(5-fold CV; converged ckpt)",
                  fontsize=11)
    ax.set_title("Bandwidth--allocation tradeoff: integration capacity falls "
                 "as encoder bandwidth grows",
                 fontsize=11, pad=8)
    ax.set_xlim(-1.5, 18.5)
    ax.set_ylim(-2.5, 1.15)
    xt = [0, 1, 4, 16]
    ax.set_xticks(xt)
    ax.set_xticklabels(["none\n(blind)", "1×1\n(coarse)",
                        "2×2\n(logpolar)", "4×4\n(fov / uni)"], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
