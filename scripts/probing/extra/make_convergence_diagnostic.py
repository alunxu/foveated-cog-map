"""
4-panel convergence diagnostic: success, SPL, DtG, and the convergence
criterion plateau frame per condition.

Visualises the §3 Methods systematic convergence criterion. Reads
the per-cond TB scalars (loaded via tensorboard) and marks the
plateau frame (max-of-three-criteria rule).

Reads:  /scratch/izar/wxu/habitat_checkpoints/{cond}/tb/events.out.*
        /tmp/extra_analyses/convergence_multimetric.json
Writes: docs/manuscript/fig/fig_convergence_diagnostic.pdf
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "paper_figures"))
from _style import apply_paper_style
apply_paper_style()


# Per-cond colour
COND_COLOR = {
    "blind":    "#444444",
    "matched":  "#377eb8",
    "foveated": "#e41a1c",
    "uniform":  "#4daf4a",
}
COND_LABEL = {
    "blind":    "Blind",
    "matched":  "Coarse",
    "foveated": "Foveated",
    "uniform":  "Uniform",
}
# 60M & 100M reference markers (for early-training and 100M anchor)


def load_curves_from_json(path: Path) -> dict:
    """Load the (success, spl, dtg) plateau & opt values from convergence_multimetric.json."""
    return json.loads(path.read_text())


def main():
    # The convergence_multimetric.json gives the plateau frames + optima.
    # For the actual smoothed curves we'd need TB events; here we plot the
    # plateau diagnostic as a 1-panel summary table-like figure.

    data = load_curves_from_json(Path("/tmp/extra_analyses/convergence_multimetric.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6),
                             gridspec_kw={"wspace": 0.35})

    metric_panels = [
        ("success", "Success rate (smoothed)", "max", 0.0, 1.0),
        ("spl",     "SPL (smoothed)",          "max", 0.0, 1.0),
        ("dtg",     "Distance-to-goal (m, smoothed)", "min", 0.0, 6.0),
    ]

    for ax, (metric, ylabel, kind, ymin, ymax) in zip(axes, metric_panels):
        for cond_key, cond_name in COND_LABEL.items():
            d = data.get(cond_key, {}).get(metric, {})
            if not d:
                continue
            plateau = d.get("plateau_frame_M")
            opt = d.get("max" if kind == "max" else "min")
            color = COND_COLOR[cond_key]
            if plateau is not None:
                # Show as a vertical bar from y=0 to y=opt at x=plateau
                ax.scatter([plateau], [opt], s=120, marker="o",
                           color=color, edgecolor="black", linewidth=1.2,
                           zorder=4, label=cond_name)
                ax.annotate(f"{plateau:.0f}M", xy=(plateau, opt),
                            xytext=(plateau + 8, opt + (0.04 if kind == "max" else -0.3)),
                            fontsize=9, color=color, fontweight="bold")
            else:
                ax.scatter([], [], s=80, marker="x", color=color, label=f"{cond_name} (no plateau)")
                # Annotate near top
                ax.annotate(f"{cond_name}: no plateau in 250-342M",
                            xy=(0.02, 0.95 - 0.07 * list(COND_LABEL).index(cond_key)),
                            xycoords="axes fraction", fontsize=8.5, color=color)
        ax.set_xlabel("Plateau frame (M)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_title(f"({chr(ord('a') + ['success', 'spl', 'dtg'].index(metric))}) {metric.upper()} plateau",
                     fontsize=11, loc="left", fontweight="bold", pad=6)
        ax.set_xlim(-15, 360)
        if kind == "max":
            ax.set_ylim(ymin, ymax + 0.03)
            ax.axhline(0.95 if metric == "success" else 0.85,
                       ls=":", color="grey", alpha=0.5, lw=0.7)
        else:
            ax.set_ylim(ymin, ymax)
        ax.grid(linestyle=":", alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if metric == "success":
            ax.legend(loc="lower right", frameon=False, fontsize=8.5)

    fig.suptitle("Per-condition convergence by 3 metrics. Reported probe checkpoints: max of three plateau frames per cond. Blind shown at training horizon (342M); other conds at SPL plateau.",
                 fontsize=10.5, fontweight="bold", y=1.00)
    plt.tight_layout()
    out = Path("docs/manuscript/fig/fig_convergence_diagnostic.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
