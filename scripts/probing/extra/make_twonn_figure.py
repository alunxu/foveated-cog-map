"""Plot intrinsic dimensionality bar chart per cond."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/scripts/paper_figures")
from _style import apply_paper_style
apply_paper_style()

data = json.loads(Path("/tmp/extra_analyses/twonn_results.json").read_text())

# Order: blind, coarse, foveated (s0+s2), uniform (s0+s2)
display = [
    ("blind",       "Blind",        "#444444", "o", 0,    "s0"),
    ("coarse",      "Coarse",       "#377eb8", "s", 1,    "s0"),
    ("foveated",    "Foveated",     "#e41a1c", "D", 1.95, "s0"),
    ("foveated_s2", "Foveated (s2)","#e41a1c", "D", 2.05, "s2"),
    ("uniform",     "Uniform",      "#4daf4a", "^", 2.95, "s0"),
    ("uniform_s2",  "Uniform (s2)", "#4daf4a", "^", 3.05, "s2"),
]

fig, ax = plt.subplots(figsize=(6.5, 3.8))

for key, label, color, marker, x, sd in display:
    if key not in data:
        continue
    d = data[key]
    ax.errorbar(x, d["id_mean"], yerr=d["id_std"],
                marker=marker, markersize=12, color=color,
                capsize=4, linewidth=0,
                markerfacecolor="white" if sd == "s2" else color,
                markeredgecolor=color, markeredgewidth=2)

# Labels
for label, x in [("Blind", 0), ("Coarse", 1), ("Foveated", 2), ("Uniform", 3)]:
    ax.text(x, 0.5, label, ha="center", fontsize=10, fontweight="bold",
            color={"Blind": "#444444", "Coarse": "#377eb8",
                   "Foveated": "#e41a1c", "Uniform": "#4daf4a"}[label])

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels([])
ax.set_xlabel("Encoder bandwidth (low → high)", fontsize=11.5, fontweight="bold")
ax.set_ylabel("Intrinsic dimensionality\n(TwoNN; 5000 samples)",
              fontsize=11.5, fontweight="bold")
ax.set_ylim(0, 5.0)
ax.axhline(3.0, ls=":", color="grey", alpha=0.5, lw=1.0)
ax.text(3.4, 3.05, "3D (task-intrinsic:\nposition + heading)",
        fontsize=8, color="grey", va="bottom", ha="right")

ax.set_title("Intrinsic dim ${\\approx}$ task dim across all conds\n(capacity differs in subspace ORIENTATION, not size)",
             fontsize=11, fontweight="bold", loc="left", pad=8)

# Legend
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0], marker="o", color="grey", lw=0, markersize=10,
           markerfacecolor="grey", markeredgecolor="black", label="seed 0 (paper canonical)"),
    Line2D([0],[0], marker="o", color="grey", lw=0, markersize=10,
           markerfacecolor="white", markeredgecolor="grey", markeredgewidth=2,
           label="seed 2 (multi-seed robustness)"),
], loc="upper right", frameon=False, fontsize=9)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.3)

plt.tight_layout()
out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_intrinsic_dim.pdf")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
