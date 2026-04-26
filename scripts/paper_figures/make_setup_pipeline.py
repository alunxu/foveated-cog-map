"""
Pipeline schematic for Figure 1 row 2: training (top) + memory analysis (bottom).

Cleaner layout: single horizontal flow per lane, no overlapping arrows
or boxes; encoder spatial-output and sensors as small annotations rather
than full-blown boxes.

Writes: docs/NeurIPS_2026/fig/setup_pipeline.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("docs/NeurIPS_2026/fig")

# ─── styling ──────────────────────────────────────────────────────────
TRAIN_COL = "#dbe9ff"
PROBE_COL = "#ffe8d4"
SENSOR_COL = "#ececec"
H_COL = "#fff4d4"
ARROW_COL = "#333333"
TEXT_COL = "#1a1a1a"


def box(ax, xy, w, h, text, color=TRAIN_COL, fontsize=8, weight="normal",
        rad=0.05):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rad}",
        facecolor=color, edgecolor="black", linewidth=0.7,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=TEXT_COL, fontweight=weight)


def arrow(ax, xy_from, xy_to, lw=0.9, style="->", colour=None,
          connectionstyle=None):
    a = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=11,
        color=colour or ARROW_COL, lw=lw,
        shrinkA=2, shrinkB=2,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 9.0)
    ax.set_aspect("auto")
    ax.axis("off")

    # ───── Training section (top lane, y around 6.5) ─────
    ax.text(0.2, 8.5, "Training pipeline",
            ha="left", va="center", fontsize=10, fontweight="bold",
            color="#1f4e8a")
    ax.text(3.6, 8.5, "(DD-PPO; identical across conditions, only visual variant differs)",
            ha="left", va="center", fontsize=8, color="#1f4e8a", style="italic")

    # Top lane boxes — single horizontal flow
    yT = 6.4   # baseline of training boxes
    hT = 1.0   # height of training boxes

    box(ax, (0.4, yT),  2.6, hT,
        "Visual variant\n(Fig.~1 a–e)",
        color=TRAIN_COL, fontsize=8.5, weight="bold")

    box(ax, (3.7, yT),  3.4, hT,
        "Visual encoder (ResNet-18 or none)\n"
        "spatial output: none $|$ $1{\\times}1$ $|$ $8{\\times}8$",
        color=TRAIN_COL, fontsize=8)

    box(ax, (7.8, yT),  1.0, hT, "$\\bigoplus$\nconcat",
        color="#f3e5f5", fontsize=8.5)

    box(ax, (9.6, yT),  2.4, hT, "LSTM\n3 layers, 512-d",
        color=TRAIN_COL, fontsize=9, weight="bold")

    box(ax, (12.7, yT), 1.5, hT, "policy\nhead",
        color=TRAIN_COL, fontsize=8.5)

    box(ax, (14.9, yT), 1.4, hT, "action",
        color="#e6f5e6", fontsize=8.5)

    # Sensor stack as small box hanging below concat
    box(ax, (5.6, 4.5),  3.3, 1.1,
        "Sensor stack: goal-in-start, GPS,\n"
        "compass, close-to-goal, prev-action\n(each 32-d, concat with visual)",
        color=SENSOR_COL, fontsize=7.5)

    # Training arrows (top lane, all horizontal at y = yT + hT/2)
    yc = yT + hT / 2  # centerline
    arrow(ax, (3.0, yc),  (3.7, yc))
    arrow(ax, (7.1, yc),  (7.8, yc))
    arrow(ax, (8.8, yc),  (9.6, yc))
    arrow(ax, (12.0, yc), (12.7, yc))
    arrow(ax, (14.2, yc), (14.9, yc))
    # sensor stack -> concat (curve up)
    arrow(ax, (8.3, 5.6), (8.3, yT), lw=0.9)

    # ───── Memory analysis (bottom lane, y around 1.6) ─────
    ax.text(0.2, 3.5, "Memory analysis",
            ha="left", va="center", fontsize=10, fontweight="bold",
            color="#a14a1f")
    ax.text(3.4, 3.5, "(frozen weights; deterministic rollouts on held-out scenes)",
            ha="left", va="center", fontsize=8, color="#a14a1f", style="italic")

    # h_t source directly under LSTM
    box(ax, (9.6, 1.6), 2.4, 1.0,
        "Top-layer\nhidden state $\\mathbf{h}_t$",
        color=PROBE_COL, fontsize=9, weight="bold")
    # Vertical arrow LSTM -> h_t (avoiding policy box)
    arrow(ax, (10.8, yT), (10.8, 2.6), lw=1.0)
    ax.text(11.0, 4.5, "freeze\n+ rollout",
            ha="left", va="center", fontsize=7.5, color="#a14a1f",
            style="italic")

    # 4 analysis boxes in 2x2 to the right of h_t (clean grid).
    # H-tag pills are SEPARATE boxes to the right of each analysis box
    # to avoid overlap with the box text.
    bx, by = 13.2, 1.6
    bw, bh = 3.6, 1.0
    gx = 0.5   # gap x (between col 0 box and col 0 tag, and between col 0 tag and col 1 box)
    gy = 0.4   # gap y

    analyses = [
        # row 0 (top)
        (0, 0, "Linear probe (Ridge)\n$\\to$ GPS, compass, DtG ...", "H1"),
        (1, 0, "Cross-condition CKA\n$+$ probe transfer", "H2"),
        # row 1 (bottom)
        (0, 1, "Memory transplant\n(donor $\\mathbf{h}$ $\\to$ recipient)", "H2"),
        (1, 1, "Shortcut discovery\n(reset vs persistent $\\mathbf{h}$)", "H1$\\times$H2"),
    ]
    # H3 takes a separate path: it requires retraining a 6th condition
    # (foveated-shifted) and comparing its memory analyses with foveated
    # (fix). We annotate it as a side note rather than a probe path.
    tag_w = 1.0
    col_stride = bw + tag_w + gx
    for col, row, label, tag in analyses:
        x = bx + col * col_stride
        if row == 0:
            y = by + bh + gy / 2
        else:
            y = by - bh - gy / 2
        # main box
        box(ax, (x, y), bw, bh, label,
            color=PROBE_COL, fontsize=7.5)
        # H tag pill OUTSIDE to the right
        box(ax, (x + bw + 0.1, y + 0.2), tag_w, 0.6, tag,
            color="white", fontsize=8.5, weight="bold", rad=0.05)
        # arrow from box to tag
        arrow(ax, (x + bw + 0.05, y + 0.5),
              (x + bw + 0.1, y + 0.5), lw=0.6, style="-")

    # Arrows from h_t to each of 4 analysis boxes
    htx, hty = 12.0, 2.1   # h_t right edge mid
    for col, row, *_ in analyses:
        bxe = bx + col * col_stride
        if row == 0:
            tgt = (bxe, by + bh + gy / 2 + bh / 2)
        else:
            tgt = (bxe, by - bh - gy / 2 + bh / 2)
        arrow(ax, (htx, hty), tgt, lw=0.7,
              connectionstyle="arc3,rad=0.0")

    # H3 side annotation (separate retraining experiment)
    box(ax, (0.4, 0.6), 6.5, 1.4,
        "H3 (in flight): retrain a 6th visual variant ---\n"
        "foveated-shifted, hardcoded gaze $(0.49, 0.62)$ ---\n"
        "and compare its memory analyses with foveated (fix).\n"
        "Difference $\\to$ gaze location modulates memory format.",
        color="#fff7e6", fontsize=7.5, weight="normal")
    box(ax, (7.1, 0.85), 1.0, 0.8, "H3",
        color="white", fontsize=10, weight="bold", rad=0.05)
    arrow(ax, (6.9, 1.3), (7.1, 1.25), lw=0.6, style="-")

    plt.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"setup_pipeline.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
