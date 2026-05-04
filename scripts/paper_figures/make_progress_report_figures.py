"""Generate two figures for the CS503 Progress Report 2 from the current
post-retrain analysis JSONs. Run locally after pulling the JSONs from RCP
into /tmp/{mlp_probe,blind_izar_100ep_preview,lagk_summary}.json.

Outputs:
  docs/cs503_progress/fig/fig1_capacity_allocation.pdf
  docs/cs503_progress/fig/fig2_lagk_stability.pdf
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Match IEEE body text (Times-like serif), Computer-Modern math.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


# ----- Data: pull from current post-retrain JSONs ----- #
MLP_JSON = Path("/tmp/mlp_probe.json")
BLIND_JSON = Path("/tmp/blind_izar_100ep_preview.json")
LAGK_JSON = Path("/tmp/lagk_summary.json")

mlp = json.loads(MLP_JSON.read_text())
blind = json.loads(BLIND_JSON.read_text())
lagk = json.loads(LAGK_JSON.read_text())

# Encoder spatial output dimension (number of spatial cells)
ENC_DIM = {
    "blind": 0,                    # no encoder
    "coarse": 1,                   # 1x1
    "foveated_logpolar": 4,        # 2x2
    "foveated": 16,                # 4x4
    "uniform": 16,                 # 4x4
}
LABEL = {
    "blind": "Blind",
    "coarse": "Coarse",
    "foveated_logpolar": "Fov-logpolar",
    "foveated": "Foveated",
    "uniform": "Uniform",
}
COLOR = {
    "blind": "#444444",
    "coarse": "#377eb8",
    "foveated_logpolar": "#984ea3",
    "foveated": "#e41a1c",
    "uniform": "#4daf4a",
}
MARKER = {
    "blind": "o",
    "coarse": "s",
    "foveated_logpolar": "P",
    "foveated": "D",
    "uniform": "^",
}

# Build dict: cond -> (linear_r2, linear_std, mlp_r2, mlp_std)
data = {}
for c in ["coarse", "foveated", "uniform", "foveated_logpolar"]:
    if c not in mlp:
        continue
    data[c] = {
        "lin_m": mlp[c]["linear_r2_mean"], "lin_s": mlp[c]["linear_r2_std"],
        "mlp_m": mlp[c]["mlp_r2_mean"], "mlp_s": mlp[c]["mlp_r2_std"],
    }
data["blind"] = {
    "lin_m": blind["linear_mlp"]["linear_r2_mean"],
    "lin_s": blind["linear_mlp"]["linear_r2_std"],
    "mlp_m": blind["linear_mlp"]["mlp_r2_mean"],
    "mlp_s": blind["linear_mlp"]["mlp_r2_std"],
}


# ===== FIGURE 1: hybrid scatter — Linear+MLP + shaded regime, inline labels ===== #
fig, ax = plt.subplots(figsize=(6.2, 3.0))

order = ["blind", "coarse", "foveated_logpolar", "foveated", "uniform"]
# x positions: blind at -0.5 for visual separation, sighted at log of encoder dim
x_pos = {"blind": -0.5, "coarse": 0.0, "foveated_logpolar": 1.0,
         "foveated": 2.0, "uniform": 2.6}  # foveated/uniform side-by-side at 4x4

lin_means = np.array([data[c]["lin_m"] for c in order])
lin_stds = np.array([data[c]["lin_s"] for c in order])
mlp_means = np.array([data[c]["mlp_m"] for c in order])
mlp_stds = np.array([data[c]["mlp_s"] for c in order])
xs = [x_pos[c] for c in order]

# Shaded regime bands
ax.axhspan(0.4, 1.05, color="#d6ebd6", alpha=0.55, zorder=0)
ax.axhspan(-2.2, -0.05, color="#f4d8d4", alpha=0.42, zorder=0)
ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8, zorder=1)

# Connecting curves
ax.plot(xs, lin_means, ls="-", color="#666", alpha=0.5, lw=1.0, zorder=2)
ax.plot(xs, mlp_means, ls=":", color="#666", alpha=0.5, lw=1.0, zorder=2)

# Linear (filled) + MLP (open) markers per condition
for i, c in enumerate(order):
    # Linear
    ax.errorbar(xs[i], lin_means[i], yerr=lin_stds[i],
                marker=MARKER[c], mfc=COLOR[c], mec=COLOR[c], ecolor=COLOR[c],
                ms=10, mew=1.5, capsize=2.5, ls="", zorder=4)
    # MLP (open marker, slight x offset)
    ax.errorbar(xs[i] + 0.18, mlp_means[i], yerr=mlp_stds[i],
                marker=MARKER[c], mfc="white", mec=COLOR[c], ecolor=COLOR[c],
                ms=10, mew=1.5, capsize=2.5, ls="", zorder=4)

# Inline condition labels (above filled marker)
label_dy = {"blind": 0.10, "coarse": 0.10, "foveated_logpolar": 0.13,
            "foveated": -0.30, "uniform": 0.13}
for c in order:
    ax.annotate(LABEL[c].replace("Foveated-logpolar", "Fov-LP"),
                xy=(x_pos[c], lin_means[order.index(c)] + label_dy[c]),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=COLOR[c])

# Regime labels
ax.text(2.45, 0.78, "Bottleneck regime\n(integration carries position)",
        fontsize=8, color="#2a6a2a", style="italic", ha="right", va="top")
ax.text(2.45, -1.85, "Rich-encoder regime\n(visual route carries position)",
        fontsize=8, color="#a04040", style="italic", ha="right", va="bottom")

# Compact Linear-vs-MLP legend
h_lin = plt.Line2D([0], [0], marker="o", color="grey", mfc="grey",
                   ms=8, ls="", label="Linear (Ridge $\\alpha{=}10$)")
h_mlp = plt.Line2D([0], [0], marker="o", color="grey", mfc="white", mec="grey",
                   ms=8, mew=1.5, ls="", label="MLP (256, $L_2{=}10^{-4}$)")
ax.legend(handles=[h_lin, h_mlp], loc="lower left", fontsize=8.5, frameon=True)

ax.set_xticks([-0.5, 0.0, 1.0, 2.3])
ax.set_xticklabels(["none\n(blind)", "$1{\\times}1$\n(coarse)",
                    "$2{\\times}2$\n(fov-LP)", "$4{\\times}4$\n(fov / uniform)"],
                   fontsize=9)
ax.set_xlabel("Encoder spatial output", fontsize=12, fontweight="bold")
ax.set_ylabel("Top-layer $\\mathbf{h}_2$ GPS $R^2$", fontsize=12, fontweight="bold")
ax.set_ylim(-2.2, 1.15)
ax.set_xlim(-0.85, 3.05)
ax.tick_params(axis="y", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title("Bandwidth-allocation: linear $R^2$ falls; MLP recovers the same info",
             fontsize=12, fontweight="bold", pad=4)
plt.tight_layout()
out1 = Path("docs/cs503_progress/fig/fig1_capacity_allocation.pdf")
out1.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out1, bbox_inches="tight")
print(f"wrote {out1}")
plt.close()


# ===== FIGURE 2: lag-k temporal stability ===== #
fig, ax = plt.subplots(figsize=(6.0, 2.4))

lags_paper = [0, 2, 5, 10, 20]
ax.axhspan(-2.0, 0, color="#f4d8d4", alpha=0.25, zorder=0)
ax.axhline(0, ls="-", color="grey", alpha=0.6, lw=0.8, zorder=1)

# Plot 4 sighted from lagk_summary; clip negative outliers at -2.0 for vis
LAGK_CLIP = -2.0
for c in ["coarse", "foveated", "uniform", "foveated_logpolar"]:
    if c not in lagk:
        continue
    means = np.array([lagk[c]["GPS"][f"k{k}"]["mean"] for k in lags_paper if lagk[c]["GPS"].get(f"k{k}")])
    stds = np.array([lagk[c]["GPS"][f"k{k}"]["std"] for k in lags_paper if lagk[c]["GPS"].get(f"k{k}")])
    valid_lags = [k for k in lags_paper if lagk[c]["GPS"].get(f"k{k}")]
    means_clip = np.clip(means, LAGK_CLIP, None)
    # Cap error bars at the clip line so they don't extend off-axis
    err_low = np.clip(means - stds, LAGK_CLIP, None)
    err_high = means + stds
    yerr_clip = np.array([means_clip - err_low, err_high - means_clip])
    ax.errorbar(valid_lags, means_clip, yerr=yerr_clip, marker=MARKER[c],
                mfc=COLOR[c], mec=COLOR[c], ecolor=COLOR[c],
                ms=7, capsize=2, ls="-", lw=1.5, label=LABEL[c], zorder=3)

# Plot blind from blind_izar_100ep_preview
blind_lags = [0, 2, 5, 10, 20]
blind_means = [blind["lagk"]["GPS"][f"k{k}"]["mean"] for k in blind_lags]
blind_stds = [blind["lagk"]["GPS"][f"k{k}"]["std"] for k in blind_lags]
ax.errorbar(blind_lags, blind_means, yerr=blind_stds, marker=MARKER["blind"],
            mfc=COLOR["blind"], mec=COLOR["blind"], ecolor=COLOR["blind"],
            ms=7, capsize=2, ls="-", lw=1.5, label="Blind$^\\dagger$", zorder=4)

ax.set_xlabel("Lag $k$", fontsize=12, fontweight="bold")
ax.set_ylabel("GPS $R^2$", fontsize=12, fontweight="bold")
ax.set_xticks(lags_paper)
ax.set_ylim(-2.2, 1.05)
ax.tick_params(axis="both", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", fontsize=7.5, ncol=2, frameon=True)
ax.set_title("Past-position decoding: $\\mathrm{GPS}_{t-k}$ from $\\mathbf{h}_t$",
             fontsize=12, fontweight="bold", pad=4)
plt.tight_layout()
out2 = Path("docs/cs503_progress/fig/fig2_lagk_stability.pdf")
plt.savefig(out2, bbox_inches="tight")
print(f"wrote {out2}")
plt.close()


# ===== FIGURE 3: substitution dynamics across training ===== #
fig, ax = plt.subplots(figsize=(6.0, 2.4))

# 5 ckpts × ~50M frames each (250M total / 49 ckpts ~= 5.1M per ckpt)
ckpts = [10, 20, 30, 40, 49]
frames_M = [c * 250.0 / 49.0 for c in ckpts]  # roughly 51, 102, 153, 204, 250

ax.axhspan(-2.0, 0, color="#f4d8d4", alpha=0.25, zorder=0)
ax.axhline(0, ls="-", color="grey", alpha=0.6, lw=0.8, zorder=1)

# 4 sighted conds (no cross-ckpt blind data; only ckpt.34 from izar)
for c in ["coarse", "foveated", "uniform", "foveated_logpolar"]:
    means, stds = [], []
    for k in ckpts:
        path = Path(f"/tmp/_subdyn_{c}_{k}.json")
        if not path.exists():
            means.append(np.nan); stds.append(np.nan); continue
        d = json.loads(path.read_text())
        gps = d.get("1b_global_gps_compass", {})
        means.append(gps.get("gps_cv_r2_mean", np.nan))
        stds.append(gps.get("gps_cv_r2_std", np.nan))
    means = np.array(means); stds = np.array(stds)
    means_clip = np.clip(means, -2.0, None)
    ax.errorbar(frames_M, means_clip, yerr=stds, marker=MARKER[c],
                mfc=COLOR[c], mec=COLOR[c], ecolor=COLOR[c],
                ms=7, capsize=2, ls="-", lw=1.5, label=LABEL[c], zorder=3)

ax.set_xlabel("Training frames (M)", fontsize=12, fontweight="bold")
ax.set_ylabel("Linear GPS $R^2$", fontsize=12, fontweight="bold")
ax.set_xticks([50, 100, 150, 200, 250])
ax.set_ylim(-2.2, 1.05)
ax.tick_params(axis="both", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", fontsize=7.5, ncol=2, frameon=True)
ax.set_title("Linear GPS $R^2$ across training",
             fontsize=12, fontweight="bold", pad=4)
plt.tight_layout()
out3 = Path("docs/cs503_progress/fig/fig3_substitution_dynamics.pdf")
plt.savefig(out3, bbox_inches="tight")
print(f"wrote {out3}")
plt.close()
