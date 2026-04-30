"""Make cross-cond probe transfer heatmap figure."""
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

data = json.loads(Path("/tmp/extra_analyses/cross_cond_transfer.json").read_text())
matrix = np.array(data["matrix"])
labels = ["Blind", "Coarse", "Foveated", "Uniform"]

# For visualization, clip negative R² to -1 (catastrophic = "below 0" all the same)
mat_clipped = np.clip(matrix, -1.0, 1.0)

fig, ax = plt.subplots(figsize=(5.5, 4.6))
im = ax.imshow(mat_clipped, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

# Annotate each cell
for i in range(4):
    for j in range(4):
        val = matrix[i, j]
        if val >= 0:
            txt = f"{val:.2f}"
        elif val > -10:
            txt = f"{val:.1f}"
        else:
            txt = f"{int(val)}"
        # Choose contrasting text color
        text_color = "white" if abs(mat_clipped[i, j]) > 0.5 else "black"
        if val < -1.0:
            text_color = "white"
        ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                color=text_color, fontweight="bold")

ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticks(range(4))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Test condition (h$_2$)", fontsize=11.5, fontweight="bold")
ax.set_ylabel("Probe trained on", fontsize=11.5, fontweight="bold")
ax.set_title("Cross-cond probe transfer (Ridge $\\alpha{=}10$)\nDiagonal $\\geq 0.90$; off-diagonal $\\ll 0$ (subspaces incompatible)",
             fontsize=11.5, fontweight="bold", loc="left", pad=10)

cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("$R^2$ (clipped to [−1, 1])", fontsize=10)

# Annotation: real values for off-diagonal can be -100s to -10k
ax.text(1.05, 1.30, "off-diagonal $R^2$ values (un-clipped) range from\n"
        "$-3.5{\\times}10^4$ (foveated$\\to$coarse) to $-3.5{\\times}10^2$ (blind$\\to$uniform)",
        transform=ax.transAxes, fontsize=8, color="#555555", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"))

for s in ("top", "right"):
    ax.spines[s].set_visible(False)

plt.tight_layout()
out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_cross_cond_transfer.pdf")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
