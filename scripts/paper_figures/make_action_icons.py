"""Generate 3 stylized icons for the action-item categories on Slide 9:
  bandwidth refinements (declining R^2 curve) | cog-neuro on h2 (brain + place
  cells) | scope tests (architecture branching).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


OUT_DIR = Path(__file__).resolve().parent.parent.parent / "docs/cs503_progress/fig"
NAVY = "#1f3a68"
ACCENT = "#b03229"
DARK = "#222"
GREY = "#888"


def _save(fig, name):
    out = OUT_DIR / f"icon_action_{name}.png"
    fig.savefig(out, transparent=True, bbox_inches="tight",
                dpi=200, pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out}")


# ===== 1. Bandwidth refinements (data / curve icon) ===== #
fig, ax = plt.subplots(figsize=(2.6, 2.6))
ax.set_xlim(0, 5); ax.set_ylim(0, 5)
ax.set_aspect("equal"); ax.axis("off")

# Axes
ax.plot([0.5, 0.5, 4.6], [4.5, 0.5, 0.5], color=NAVY, lw=3,
        solid_capstyle="round")
# Curve: monotonic decline (mimics our Fig 1 finding)
xs = np.linspace(0.7, 4.4, 60)
ys = 4.0 - 1.0 * (xs - 0.7) - 0.15 * (xs - 0.7) ** 1.3
ax.plot(xs, ys, color=ACCENT, lw=4, solid_capstyle="round")
# Three big data points along the curve
for px, py, col in [(1.2, ys[int(60 * 0.13)], NAVY),
                     (2.7, ys[int(60 * 0.54)], NAVY),
                     (4.0, ys[int(60 * 0.89)], ACCENT)]:
    ax.add_patch(patches.Circle((px, py), 0.22, facecolor=col,
                                 edgecolor="white", lw=2, zorder=4))
# Subtle gridlines
for y in [1.5, 2.5, 3.5]:
    ax.plot([0.55, 4.5], [y, y], color=GREY, lw=0.6, alpha=0.35)
_save(fig, "1_bandwidth")


# ===== 2. Cog-neuro on h2 (place-cell rate map) ===== #
fig, ax = plt.subplots(figsize=(2.6, 2.6))
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal"); ax.axis("off")

# A 2D arena with two place-fields (Gaussian blobs)
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = (np.exp(-((X - 0.7) ** 2 + (Y - 0.4) ** 2) / 0.35) +
     0.85 * np.exp(-((X + 0.6) ** 2 + (Y + 0.5) ** 2) / 0.4))
ax.imshow(Z, extent=[-2, 2, -2, 2], origin="lower",
          cmap="YlOrRd", alpha=0.85)
# Arena outline
ax.add_patch(patches.Rectangle((-2, -2), 4, 4, fill=False,
                                edgecolor=NAVY, lw=3))
# Trajectory squiggle
t = np.linspace(0, 2 * np.pi, 100)
tx = 1.4 * np.cos(t) + 0.3 * np.sin(3 * t)
ty = 1.4 * np.sin(t) + 0.3 * np.cos(3 * t)
ax.plot(tx, ty, color=NAVY, lw=1.4, alpha=0.8)
# Title-like annotation
ax.text(0, -2.3, "place fields", ha="center", fontsize=10,
        color=ACCENT, style="italic", weight="bold")
_save(fig, "2_cogneuro")


# ===== 3. Scope tests (architecture branching: LSTM vs Transformer) ===== #
fig, ax = plt.subplots(figsize=(2.6, 2.6))
ax.set_xlim(0, 5); ax.set_ylim(0, 5)
ax.set_aspect("equal"); ax.axis("off")

# Source node
ax.add_patch(patches.FancyBboxPatch((1.6, 3.6), 1.8, 0.7,
                                     boxstyle="round,pad=0.05,rounding_size=0.2",
                                     facecolor=NAVY, edgecolor=NAVY))
ax.text(2.5, 3.95, "current", ha="center", va="center",
        color="white", fontsize=10, weight="bold")
# Two branch nodes
ax.add_patch(patches.FancyBboxPatch((0.3, 1.6), 1.7, 0.7,
                                     boxstyle="round,pad=0.05,rounding_size=0.2",
                                     facecolor="white", edgecolor=NAVY, lw=2))
ax.text(1.15, 1.95, "fov sweep", ha="center", va="center",
        color=NAVY, fontsize=10, weight="bold")
ax.add_patch(patches.FancyBboxPatch((3.0, 1.6), 1.7, 0.7,
                                     boxstyle="round,pad=0.05,rounding_size=0.2",
                                     facecolor="white", edgecolor=ACCENT, lw=2))
ax.text(3.85, 1.95, "transformer", ha="center", va="center",
        color=ACCENT, fontsize=10, weight="bold")
# Arrows
ax.annotate("", xy=(1.15, 2.3), xytext=(2.2, 3.6),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
ax.annotate("", xy=(3.85, 2.3), xytext=(2.8, 3.6),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
# Question mark below
ax.text(2.5, 0.7, "generalises?", ha="center", fontsize=11,
        color=ACCENT, style="italic", weight="bold")
_save(fig, "3_scope")
