"""
Hero figure for the encoder-bottleneck → LSTM spatial compensation finding.

3 panels (GPS / compass / DtG), each with 5 bars ordered by bottleneck severity.
Values from scripts/probing/ det-analysis 5-fold CV.

Output: docs/manuscript/fig/h1_bottleneck.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 5-fold CV results from deterministic probe
# (condition, GPS R² mean, GPS R² std, compass mean, compass std, DtG mean, DtG std)
DATA = [
    ("Blind",           +0.95, 0.02, +0.81, 0.08, +0.90, 0.03),
    ("Coarse",   +0.78, 0.10, +0.64, 0.10, +0.85, 0.12),
    ("Uniform",         -0.31, 0.86, +0.36, 0.23, +0.86, 0.09),
    ("Foveated",  +0.06, 0.88, +0.07, 0.69, +0.82, 0.09),
]

# Clip to [-1.5, 1] for visibility; mark sentinels with arrows/patterns
CLIP_MIN = -1.5
labels = [d[0] for d in DATA]
gps_mean = np.array([d[1] for d in DATA])
gps_std  = np.array([d[2] for d in DATA])
comp_mean = np.array([d[3] for d in DATA])
comp_std  = np.array([d[4] for d in DATA])
dtg_mean = np.array([d[5] for d in DATA])
dtg_std  = np.array([d[6] for d in DATA])

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=False)

x = np.arange(len(labels))

for ax, (mean, std, tname) in zip(axes, [
    (gps_mean, gps_std, "GPS (world-frame position)"),
    (comp_mean, comp_std, "Compass (world-frame heading)"),
    (dtg_mean, dtg_std, "DtG (ego-relative, control)"),
]):
    mean_clipped = np.clip(mean, CLIP_MIN, 1.0)
    clipped_mask = mean < CLIP_MIN
    # Bars
    bars = ax.bar(x, mean_clipped, yerr=std,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.6,
                  capsize=3, error_kw={"linewidth": 0.8})
    # Mark clipped bars with downward arrow
    for i, (m, c) in enumerate(zip(mean, clipped_mask)):
        if c:
            ax.annotate(f"{m:.1f}", (i, CLIP_MIN + 0.1),
                        ha="center", fontsize=7, color="darkred")
            ax.annotate("↓", (i, CLIP_MIN + 0.05), ha="center",
                        fontsize=12, color="darkred")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_ylim(CLIP_MIN, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_title(tname, fontsize=9.5)
    ax.set_ylabel("$R^2$ (5-fold CV)", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Bracket "bottleneck" and "pass-through" groups on the GPS panel
ax0 = axes[0]
ax0.annotate("", xy=(1.4, 1.02), xytext=(-0.4, 1.02),
             arrowprops=dict(arrowstyle="-", lw=1.2, color="darkblue"))
ax0.annotate("bottleneck", xy=(0.5, 1.04), ha="center", va="bottom",
             fontsize=8, color="darkblue")
ax0.annotate("", xy=(4.4, 1.02), xytext=(1.6, 1.02),
             arrowprops=dict(arrowstyle="-", lw=1.2, color="darkgray"))
ax0.annotate("pass-through", xy=(3.0, 1.04), ha="center", va="bottom",
             fontsize=8, color="darkgray")

plt.tight_layout()

import os
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                      "docs", "manuscript", "fig")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "h1_bottleneck.pdf")
plt.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"Wrote {out_path}")