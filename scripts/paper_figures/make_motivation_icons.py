"""Generate 3 stylized icons for the CS503 progress-talk motivation slide:
  echolocation (bat + sonar) | blind rodent | foveated eye.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath


OUT_DIR = Path(__file__).resolve().parent.parent.parent / "docs/cs503_progress/fig"
NAVY = "#1f3a68"
ACCENT = "#c0392b"
DARK = "#2a2a2a"
GREY = "#888888"


def _save(fig, name):
    out = OUT_DIR / f"icon_{name}.png"
    fig.savefig(out, transparent=True, bbox_inches="tight",
                dpi=200, pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out}")


# ===== 1. Bat + sonar (echolocation) ===== #
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal"); ax.axis("off")

# Sonar arcs to the right
for r, alpha in [(0.55, 0.25), (0.85, 0.40), (1.15, 0.55)]:
    arc = patches.Arc((0, 0), r * 2, r * 2, angle=0,
                      theta1=-30, theta2=30,
                      color=ACCENT, lw=2.5, alpha=alpha,
                      linestyle=(0, (4, 3)))
    ax.add_patch(arc)

# Bat silhouette (body + wings) — abstract V-shape
bat_path = MplPath([
    (-0.95, 0.10), (-0.50, 0.55), (-0.20, 0.18), (0.0, 0.30),
    (0.20, 0.18), (0.50, 0.55), (0.95, 0.10),
    (0.55, -0.10), (0.20, -0.20), (0.0, -0.10), (-0.20, -0.20),
    (-0.55, -0.10), (-0.95, 0.10)
])
ax.add_patch(patches.PathPatch(bat_path, facecolor=NAVY, edgecolor=NAVY, lw=0))
# bat ears
ax.add_patch(patches.Polygon([[-0.05, 0.32], [-0.10, 0.45], [-0.02, 0.36]],
                              color=NAVY))
ax.add_patch(patches.Polygon([[0.05, 0.32], [0.10, 0.45], [0.02, 0.36]],
                              color=NAVY))
_save(fig, "1_echolocation")


# ===== 2. Blind rodent (mole-like silhouette) ===== #
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal"); ax.axis("off")

# Body: ellipse
ax.add_patch(patches.Ellipse((0.05, -0.10), 1.7, 0.95,
                              facecolor=NAVY, edgecolor=NAVY))
# Snout: triangle on left
ax.add_patch(patches.Polygon([[-0.85, -0.05], [-1.20, -0.10],
                              [-0.85, -0.20]],
                              facecolor=NAVY))
# Tail: thin line on right
ax.plot([0.80, 1.30], [-0.08, 0.20], color=NAVY, lw=4, solid_capstyle="round")
# Eyes — closed (X marks): blind
for x in [-0.45, -0.15]:
    ax.plot([x - 0.06, x + 0.06], [0.08, -0.04], color="white", lw=2)
    ax.plot([x - 0.06, x + 0.06], [-0.04, 0.08], color="white", lw=2)
# Whiskers
for y in [0.05, -0.05]:
    ax.plot([-0.95, -1.15], [y - 0.05, y - 0.10], color=GREY, lw=1)
# Legend below
ax.text(0, -0.95, "no vision", ha="center", fontsize=14,
        color=ACCENT, style="italic", weight="bold")
_save(fig, "2_blind")


# ===== 3. Foveated eye (sharp centre, blurred periphery) ===== #
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal"); ax.axis("off")

# Eye outline (almond shape)
eye = patches.Ellipse((0, 0), 2.4, 1.3, facecolor="white",
                       edgecolor=NAVY, lw=3.5)
ax.add_patch(eye)
# Iris with gradient-like rings (sharp inner = fovea, blur outer = periphery)
for r, alpha in [(0.55, 1.0), (0.40, 1.0), (0.25, 1.0)]:
    ax.add_patch(patches.Circle((0, 0), r,
                                 facecolor=NAVY, alpha=alpha))
# Pupil
ax.add_patch(patches.Circle((0, 0), 0.18, facecolor=DARK))
# Periphery blur indicator: light dotted ring outside iris
for r in [0.75, 0.95]:
    ax.add_patch(patches.Circle((0, 0), r, fill=False,
                                 edgecolor=ACCENT, lw=1.8,
                                 linestyle=(0, (2, 3)), alpha=0.6))
# Sharp-vs-blurry label
ax.text(0, -0.95, "sharp centre, blurred periphery",
        ha="center", fontsize=11, color=ACCENT, style="italic")
_save(fig, "3_foveated")
