"""
Information allocation figure, v2: linear vs MLP probe per cond, full 5-cond.

Reads from `/tmp/rcp_analysis/mlp_probe.json` (the post-blind, post-retrain
output from mlp_rcp.sh). Replaces v1's 4-cond hardcoded layout with the full
bandwidth axis: blind / coarse / fov-logpolar / foveated / uniform.

Linear range ≈ 2.1 across the bandwidth axis; MLP range ≈ 0.65 — position
information is approximately preserved by non-linear readout in rich-encoder
conditions even when linear readability collapses.

Writes: docs/manuscript/fig/fig_information_allocation.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MLP_JSON = Path("/tmp/rcp_analysis/mlp_probe.json")
OUT_PDF = REPO_ROOT / "docs/manuscript/fig/fig_information_allocation.pdf"

# (key, label, encoder_cells, colour, marker)
CONDS = [
    ("blind_izar",        "Blind",        0,  "#444444", "o"),
    ("coarse",            "Coarse",       1,  "#377eb8", "s"),
    ("foveated_logpolar", "Fov-logpolar", 4,  "#984ea3", "v"),
    ("foveated",          "Foveated",     16, "#e41a1c", "D"),
    ("uniform",           "Uniform",      16, "#4daf4a", "^"),
]


def x_pos(cells: int, label: str) -> float:
    if cells == 0:
        return 0.0
    if cells == 1:
        return 1.6
    if cells == 4:
        return 3.5
    if label == "Foveated":
        return 5.4
    if label == "Uniform":
        return 6.3
    return 5.5


def main():
    data = json.loads(MLP_JSON.read_text())

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.7, zorder=1)

    xs_lin, ys_lin = [], []
    xs_mlp, ys_mlp = [], []

    for cond, label, cells, color, marker in CONDS:
        if cond not in data:
            continue
        d = data[cond]
        x = x_pos(cells, label)
        # Linear (filled, slight left offset)
        ax.errorbar([x - 0.13], [d["linear_r2_mean"]], yerr=[d["linear_r2_std"]],
                    marker=marker, markersize=11, color=color, alpha=0.9,
                    capsize=4, capthick=1.0, elinewidth=0.9, linewidth=0,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=4)
        # MLP (hollow, slight right offset)
        ax.errorbar([x + 0.13], [d["mlp_r2_mean"]], yerr=[d["mlp_r2_std"]],
                    marker=marker, markersize=11, color=color, alpha=0.9,
                    capsize=4, capthick=1.0, elinewidth=0.9, linewidth=0,
                    markeredgecolor="black", markeredgewidth=0.5,
                    markerfacecolor="white", zorder=4)
        # Connector
        ax.plot([x - 0.13, x + 0.13],
                [d["linear_r2_mean"], d["mlp_r2_mean"]],
                color=color, alpha=0.5, lw=1.6, ls="-", zorder=2)
        # Label above the higher value
        higher = max(d["linear_r2_mean"], d["mlp_r2_mean"])
        if label in ("Foveated",):
            ax.annotate(label, xy=(x, higher), xytext=(x - 0.55, higher + 0.18),
                        fontsize=10, color=color, fontweight="bold",
                        ha="right", va="center")
        elif label == "Uniform":
            ax.annotate(label, xy=(x, higher), xytext=(x + 0.5, higher + 0.18),
                        fontsize=10, color=color, fontweight="bold",
                        ha="left", va="center")
        elif label == "Fov-logpolar":
            ax.annotate(label, xy=(x, higher), xytext=(x, higher + 0.18),
                        fontsize=10, color=color, fontweight="bold",
                        ha="center", va="center")
        else:
            ax.annotate(label, xy=(x, higher), xytext=(x, higher + 0.16),
                        fontsize=10, color=color, fontweight="bold",
                        ha="center", va="center")
        xs_lin.append(x - 0.13); ys_lin.append(d["linear_r2_mean"])
        xs_mlp.append(x + 0.13); ys_mlp.append(d["mlp_r2_mean"])

    # Trend lines
    pairs_lin = sorted(zip(xs_lin, ys_lin))
    pairs_mlp = sorted(zip(xs_mlp, ys_mlp))
    ax.plot([t[0] for t in pairs_lin], [t[1] for t in pairs_lin],
            ls=":", color="grey", alpha=0.55, lw=1.2, zorder=2)
    ax.plot([t[0] for t in pairs_mlp], [t[1] for t in pairs_mlp],
            ls="-", color="grey", alpha=0.40, lw=1.2, zorder=2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="grey", lw=0,
               markersize=10, markerfacecolor="grey", markeredgecolor="black",
               label=r"Linear (Ridge $\alpha{=}10$)"),
        Line2D([0], [0], marker="o", color="grey", lw=0,
               markersize=10, markerfacecolor="white", markeredgecolor="black",
               label=r"MLP (hidden=256, $L_2{=}10^{-4}$)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10, frameon=False)

    ax.set_xticks([0, 1.6, 3.5, 5.85])
    ax.set_xticklabels(["none\n(blind)", "1×1\n(coarse)",
                        "2×2\n(logpolar)", "4×4\n(fov / uni)"])
    ax.set_xlabel("Encoder spatial output (cells)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Top-layer $\mathbf{h}_2$ GPS $R^2$" + "\n(5-fold episode-level CV)",
                  fontsize=12, fontweight="bold")
    ax.set_xlim(-0.7, 7.0)
    ax.set_ylim(-2.0, 1.20)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.20)

    ax.set_title(
        "Information allocation: total info preserved, encoding format differs",
        fontsize=12, fontweight="bold", loc="left", pad=10)

    plt.tight_layout()
    fig.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
