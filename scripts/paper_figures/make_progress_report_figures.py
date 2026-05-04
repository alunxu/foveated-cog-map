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


# ===== FIGURE 1: capacity allocation (encoder dim vs linear/MLP GPS R^2) ===== #
fig, ax = plt.subplots(figsize=(6.0, 2.4))

# Order: blind (dim=0), coarse (1), fov-logpolar (4), foveated (16), uniform (16)
order = ["blind", "coarse", "foveated_logpolar", "foveated", "uniform"]
xs = [ENC_DIM[c] for c in order]
lin_means = [data[c]["lin_m"] for c in order]
lin_stds = [data[c]["lin_s"] for c in order]
mlp_means = [data[c]["mlp_m"] for c in order]
mlp_stds = [data[c]["mlp_s"] for c in order]

# Light grey shaded band for "policy-readable" zone (R^2 > 0)
ax.axhspan(-2.0, 0, color="#f4d8d4", alpha=0.25, zorder=0)
ax.axhline(0, ls="-", color="grey", alpha=0.6, lw=0.8, zorder=1)

# Plot MLP curve (lighter, dashed) first so it's behind
xs_jit_mlp = [x - 0.05 if x == 16 and order[i] == "foveated" else (x + 0.05 if x == 16 else x) for i, x in enumerate(xs)]
ax.plot(xs_jit_mlp, mlp_means, ls="--", color="grey", alpha=0.6, lw=1.2, zorder=2, label="MLP probe")

for i, c in enumerate(order):
    # MLP point (open marker)
    ax.errorbar(xs_jit_mlp[i], mlp_means[i], yerr=mlp_stds[i],
                marker=MARKER[c], mfc="white", mec=COLOR[c], ecolor=COLOR[c],
                ms=7, mew=1.5, capsize=2, ls="", zorder=3)

# Linear curve (solid)
xs_jit_lin = [x - 0.05 if x == 16 and order[i] == "foveated" else (x + 0.05 if x == 16 else x) for i, x in enumerate(xs)]
ax.plot(xs_jit_lin, lin_means, ls="-", color="#222", alpha=0.4, lw=1.0, zorder=2)

for i, c in enumerate(order):
    ax.errorbar(xs_jit_lin[i], lin_means[i], yerr=lin_stds[i],
                marker=MARKER[c], mfc=COLOR[c], mec=COLOR[c], ecolor=COLOR[c],
                ms=8, mew=1.0, capsize=2, ls="", label=LABEL[c], zorder=4)

ax.set_xlabel("Encoder spatial output (# cells)", fontsize=13, fontweight="bold")
ax.set_ylabel("GPS $R^2$ (5-fold ep-CV)", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1, 4, 16])
ax.set_xticklabels(["0\n(blind)", "1\n(1×1)", "4\n(2×2)", "16\n(4×4)"], fontsize=8.5)
ax.set_ylim(-2.2, 1.05)
ax.tick_params(axis="y", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Two-column legend below the plot
handles_lin = [plt.Line2D([0], [0], marker=MARKER[c], color=COLOR[c], mfc=COLOR[c],
                          ms=7, ls="", label=LABEL[c]) for c in order]
handle_mlp = plt.Line2D([0], [0], ls="--", color="grey", lw=1.2, label="MLP (dashed)")
handle_lin = plt.Line2D([0], [0], ls="-", color="#222", alpha=0.6, lw=1.0, label="Linear (solid)")

ax.legend(handles=[handle_lin, handle_mlp] + handles_lin,
          loc="upper right", fontsize=7.5, ncol=2, frameon=True,
          bbox_to_anchor=(1.0, 1.02))

ax.set_title("Top-layer GPS $R^2$ vs.\\ encoder bandwidth",
             fontsize=13, fontweight="bold", pad=4)
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

# Plot 4 sighted from lagk_summary
for c in ["coarse", "foveated", "uniform", "foveated_logpolar"]:
    if c not in lagk:
        continue
    means = [lagk[c]["GPS"][f"k{k}"]["mean"] for k in lags_paper if lagk[c]["GPS"].get(f"k{k}")]
    stds = [lagk[c]["GPS"][f"k{k}"]["std"] for k in lags_paper if lagk[c]["GPS"].get(f"k{k}")]
    valid_lags = [k for k in lags_paper if lagk[c]["GPS"].get(f"k{k}")]
    ax.errorbar(valid_lags, means, yerr=stds, marker=MARKER[c],
                mfc=COLOR[c], mec=COLOR[c], ecolor=COLOR[c],
                ms=7, capsize=2, ls="-", lw=1.5, label=LABEL[c], zorder=3)

# Plot blind from blind_izar_100ep_preview
blind_lags = [0, 2, 5, 10, 20]
blind_means = [blind["lagk"]["GPS"][f"k{k}"]["mean"] for k in blind_lags]
blind_stds = [blind["lagk"]["GPS"][f"k{k}"]["std"] for k in blind_lags]
ax.errorbar(blind_lags, blind_means, yerr=blind_stds, marker=MARKER["blind"],
            mfc=COLOR["blind"], mec=COLOR["blind"], ecolor=COLOR["blind"],
            ms=7, capsize=2, ls="-", lw=1.5, label="Blind$^\\dagger$", zorder=4)

ax.set_xlabel("Lag $k$", fontsize=13, fontweight="bold")
ax.set_ylabel("GPS $R^2$", fontsize=13, fontweight="bold")
ax.set_xticks(lags_paper)
ax.set_ylim(-2.2, 1.05)
ax.tick_params(axis="both", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", fontsize=7.5, ncol=2, frameon=True)
ax.set_title("Past-position decoding: $\\mathrm{GPS}_{t-k}$ from $\\mathbf{h}_t$",
             fontsize=13, fontweight="bold", pad=4)
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

ax.set_xlabel("Training frames (M)", fontsize=13, fontweight="bold")
ax.set_ylabel("Linear GPS $R^2$", fontsize=13, fontweight="bold")
ax.set_xticks([50, 100, 150, 200, 250])
ax.set_ylim(-2.2, 1.05)
ax.tick_params(axis="both", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", fontsize=7.5, ncol=2, frameon=True)
ax.set_title("Linear GPS $R^2$ across training",
             fontsize=13, fontweight="bold", pad=4)
plt.tight_layout()
out3 = Path("docs/cs503_progress/fig/fig3_substitution_dynamics.pdf")
plt.savefig(out3, bbox_inches="tight")
print(f"wrote {out3}")
plt.close()
