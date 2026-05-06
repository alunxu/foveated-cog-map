"""Consolidated 3-panel §3.1 magnitude figure.

Replaces the previous trio (fig_capacity_allocation, fig3_substitution_dynamics,
fig2_h1_mega) with ONE info-dense 3-panel figure that tells the §3.1 story
across the three relevant axes:

  Panel A — cross-condition magnitude collapse:
    Top-layer h_2 linear GPS R^2 vs encoder spatial bandwidth (5 conditions).
    The bandwidth-allocation tradeoff in static form.

  Panel B — across-training substitution mechanism:
    Top-layer GPS R^2 across training checkpoints (50M -> 250M frames).
    Bottleneck conditions hold the linear code; rich-encoder conditions
    decay along trajectories whose timescale tracks encoder informativeness.

  Panel C — pipeline-view localisation:
    GPS R^2 along Encoder -> L0 -> L1 -> L2 (policy-readable).
    The L2 split is where capacity allocation lives; encoder + earlier
    layers are condition-flat.

Data sources:
  - Panel A: /tmp/rcp_analysis/mlp_probe.json
  - Panel B: <results-dir>/<cond>_gibson_ckpt<N>_det_analysis.json
  - Panel C: <results-dir>/<cond>_gibson_det_analysis.json
             <results-dir>/<cond>_encoder_features_det.json

Writes: docs/manuscript/fig/fig_magnitude.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


# ─── Shared condition styling ─────────────────────────────────────────
# (json_key,  short_label,    encoder_cells, colour,    marker, frames_per_ckpt_M)
CONDS = [
    ("blind",            "Blind",         0,  "#444444", "o", 10.06),
    ("matched",          "Coarse",        1,  "#377eb8", "s", 5.10),
    ("foveated_learned", "Fov-logpolar",  4,  "#984ea3", "v", 5.10),
    ("foveated",         "Foveated",      16, "#e41a1c", "D", 4.83),
    ("uniform",          "Uniform",       16, "#4daf4a", "^", 5.10),
]
CLIP_MIN = -2.0
X_MAX_M = 250.0


# ───────────────────── Panel A: linear GPS vs bandwidth ─────────────────────
def panel_a(ax, mlp_json: Path) -> None:
    """Linear GPS R² vs encoder spatial bandwidth across the five conditions."""
    # mlp_probe.json keys use slightly different names than CONDS
    key_map = {
        "blind": "blind_izar",
        "matched": "coarse",
        "foveated_learned": "foveated_logpolar",
        "foveated": "foveated",
        "uniform": "uniform",
    }
    data = json.loads(mlp_json.read_text())

    # Shaded regimes
    ax.axhspan(0.4, 1.05, color="#dceedc", alpha=0.55, zorder=0)
    ax.axhspan(-2.5, 0.4, color="#fbe0dc", alpha=0.45, zorder=0)
    ax.axhline(0, color="#888", lw=0.6, ls="--", zorder=0)

    # Data points
    for cond_key, label, cells, col, mk, _ in CONDS:
        d = data[key_map[cond_key]]
        r2, sd = d["linear_r2_mean"], d["linear_r2_std"]
        # Jitter foveated/uniform slightly for visual separation
        x = cells + (0.3 if cond_key == "uniform"
                      else (-0.3 if cond_key == "foveated" else 0))
        ax.errorbar(x, r2, yerr=sd, fmt=mk, color=col, markersize=10,
                    markeredgecolor="white", markeredgewidth=1.2,
                    capsize=3, lw=1.5, zorder=4, label=label)

    # Regime labels
    ax.text(13.5, 0.85, "Bottleneck regime\n(integration carries pos.)",
            fontsize=8, color="#3a7d3a", ha="right", va="top", style="italic")
    ax.text(13.5, -2.1, "Rich-encoder regime\n(visual route carries pos.)",
            fontsize=8, color="#a02528", ha="right", va="bottom", style="italic")

    ax.set_xlabel("Encoder spatial output (cells)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"top-layer linear GPS $R^2$",
                  fontsize=11, fontweight="bold")
    ax.set_title("(a) Magnitude collapse",
                 fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax.set_xlim(-1.5, 18.5)
    ax.set_ylim(-2.5, 1.15)
    ax.set_xticks([0, 1, 4, 16])
    ax.set_xticklabels(["0\n(blind)", "1×1\n(coarse)",
                        "2×2\n(fov-LP)", "4×4\n(fov / uni)"], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=1)


# ───────────────────── Panel B: cross-training substitution ─────────────────
def panel_b(ax, results_dir: Path) -> None:
    """GPS R² across training checkpoints — substitution mechanism."""
    ax.axhspan(CLIP_MIN, 0, color="#f4d8d4", alpha=0.18, zorder=0)
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8, zorder=1)

    plotted_anything = False
    for cond_key, label, _cells, col, mk, frames_per_ckpt in CONDS:
        xs_full, ys_full, errs_full = [], [], []
        xs_partial, ys_partial = [], []
        clipped_at = []
        for ck in range(0, 60):
            p = results_dir / f"{cond_key}_gibson_ckpt{ck}_det_analysis.json"
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            r2 = d.get("1b_global_gps_compass", {}).get("gps_cv_r2_mean")
            std = d.get("1b_global_gps_compass", {}).get("gps_cv_r2_std", 0.0)
            n_ep = d.get("n_episodes", 0)
            if r2 is None:
                continue
            x = ck * frames_per_ckpt
            if x > X_MAX_M:
                continue
            y = float(np.clip(r2, CLIP_MIN, 1.05))
            if n_ep >= 500:
                xs_full.append(x); ys_full.append(y); errs_full.append(std)
            else:
                xs_partial.append(x); ys_partial.append(y)
            if r2 < CLIP_MIN:
                clipped_at.append((x, r2))

        if xs_full:
            plotted_anything = True
            ax.errorbar(xs_full, ys_full, yerr=errs_full, marker=mk,
                        label=label, color=col, linewidth=2.0,
                        markersize=7, capsize=3.0, capthick=0.8,
                        elinewidth=0.8, ecolor=col, alpha=1.0, zorder=4)
        if xs_partial:
            plotted_anything = True
            ax.plot(xs_partial, ys_partial, marker=mk, ls=":",
                    color=col, markersize=6, mfc="white", mec=col,
                    mew=1.5, alpha=0.75, linewidth=1.4,
                    label=None if xs_full else f"{label} (partial)",
                    zorder=3)
        for x, r2 in clipped_at:
            ax.annotate(f"{r2:.1f}", (x, CLIP_MIN + 0.06),
                        fontsize=8, fontweight="bold",
                        ha="center", va="bottom", color="darkred", zorder=5)

    if not plotted_anything:
        ax.text(0.5, 0.5, "(no across-ckpt JSONs found)",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    ax.set_ylim(CLIP_MIN - 0.10, 1.10)
    ax.set_xlim(0, X_MAX_M + 5)
    ax.set_xlabel("training frames (M)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"top-layer GPS $R^2$", fontsize=11, fontweight="bold")
    ax.set_title("(b) Substitution mechanism",
                 fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    # Annotation pointing out the substitution
    ax.annotate("", xy=(180, -1.0), xytext=(120, 0.5),
                arrowprops=dict(arrowstyle="->", color="#a02528", lw=1.0,
                                alpha=0.7))
    ax.text(120, 0.6, "rich-encoder\nsubstitution",
            fontsize=8, color="#a02528", style="italic", ha="left", va="bottom")


# ───────────────────── Panel C: pipeline view ────────────────────────────────
def panel_c(ax, results_dir: Path) -> None:
    """GPS R² along Encoder → L0 → L1 → L2."""
    for cond_key, label, _cells, col, mk, _ in CONDS:
        # Encoder R² (only sighted conditions have an encoder)
        enc_r2 = None
        if cond_key != "blind":
            enc_p = results_dir / f"{cond_key}_encoder_features_det.json"
            if enc_p.exists():
                try:
                    enc_d = json.loads(enc_p.read_text())
                    enc_r2 = (enc_d.get("encoder_gps_compass", {})
                                   .get("gps_cv_r2_mean"))
                except Exception:
                    pass

        # L0/L1/L2 from main analysis JSON's 1d_multilayer
        layer_r2 = {0: None, 1: None, 2: None}
        main_p = results_dir / f"{cond_key}_gibson_det_analysis.json"
        if main_p.exists():
            try:
                d = json.loads(main_p.read_text())
                for entry in d.get("1d_multilayer", []):
                    if entry.get("state") == "h":
                        L = entry["layer"]
                        if L in layer_r2:
                            layer_r2[L] = float(np.clip(entry["gps_r2"],
                                                       CLIP_MIN, 1.05))
            except Exception:
                pass

        xs, ys = [], []
        if enc_r2 is not None:
            xs.append(0); ys.append(float(np.clip(enc_r2, CLIP_MIN, 1.05)))
        for L in (0, 1, 2):
            if layer_r2[L] is not None:
                xs.append(L + 1); ys.append(layer_r2[L])

        if xs:
            ax.plot(xs, ys, marker=mk, label=label,
                    color=col, linewidth=1.8, markersize=6.5)

    # L2 MLP recovery band
    ax.axhspan(0.51, 0.73, xmin=(2.85 - (-0.4)) / (3.4 - (-0.4)),
               xmax=(3.4 - (-0.4)) / (3.4 - (-0.4)),
               color="#88aaee", alpha=0.20, zorder=0)
    # Place annotation above the band, leader line down
    ax.annotate("MLP recovery\n($R^2 \\in [0.51, 0.73]$)",
                xy=(3.0, 0.62), xytext=(2.0, 0.95),
                fontsize=8, color="#445588", style="italic",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#445588",
                                lw=0.8, alpha=0.8))

    x_labels = ["Encoder", "L0\n(LSTM in)", "L1", "L2\n(top, policy)"]
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(CLIP_MIN - 0.05, 1.10)
    ax.set_xlabel("Pipeline location", fontsize=11, fontweight="bold")
    ax.set_title("(c) Pipeline view: where the divergence sits",
                 fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ───────────────────────── compose ────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-json", type=Path,
                    default=Path("/tmp/rcp_analysis/mlp_probe.json"))
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results/probing_results"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/manuscript/fig/fig_magnitude.pdf"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.0, 4.4))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.05, 1.15, 1.10],
        wspace=0.26,
        top=0.85, bottom=0.20, left=0.05, right=0.99,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a, args.mlp_json)
    panel_b(ax_b, args.results_dir)
    panel_c(ax_c, args.results_dir)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
