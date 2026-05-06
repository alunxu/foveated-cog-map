"""Consolidated 3-panel §3.1 magnitude figure (post-retrain, hp-consistent).

Replaces the previous trio (fig_capacity_allocation, fig3_substitution_dynamics,
fig2_h1_mega) with ONE info-dense 3-panel figure that tells the §3.1 story:

  Panel A — cross-condition magnitude collapse (linear GPS R^2 vs encoder
            spatial bandwidth across 5 conds; 5-fold CV).
  Panel B — across-training substitution mechanism (top-layer GPS R^2 across
            checkpoints, 50M -> 250M frames).
  Panel C — pipeline-view localisation (GPS R^2 along L0 -> L1 -> L2, where
            L2 is policy-readable).

Data sources — ALL from the post-retrain hp-consistent run on RCP, locally
mirrored under /tmp/rcp_analysis/:
  - Panel A: /tmp/rcp_analysis/mlp_probe.json
  - Panel B (sighted, 4 conds): /tmp/rcp_analysis/<cond>_det_ckpt{10,20,30,40}_analysis.json
             plus converged 250M from /tmp/rcp_analysis/<cond>_det_analysis.json
  - Panel B (blind): legacy results/probing_results/blind_gibson_ckpt<N>_det_analysis.json
             (blind kept per project memory: not re-trained on RCP)
  - Panel C: /tmp/rcp_analysis/<cond>_det_analysis.json (1d_multilayer.gps_r2 for
             L0/L1/L2 'h' state; encoder point omitted — no new encoder probing)

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


# ─── Condition styling ────────────────────────────────────────────────
# (rcp_key,            mlp_key,             label,         enc_cells, colour,    marker, frames_per_ckpt_M)
CONDS = [
    ("blind_izar",       "blind_izar",        "Blind",         0,  "#444444", "o", 10.06),
    ("coarse",           "coarse",            "Coarse",        1,  "#377eb8", "s", 5.0),
    ("foveated_logpolar","foveated_logpolar", "Fov-logpolar",  4,  "#984ea3", "v", 5.0),
    ("foveated",         "foveated",          "Foveated",      16, "#e41a1c", "D", 5.0),
    ("uniform",          "uniform",           "Uniform",       16, "#4daf4a", "^", 5.0),
]
CLIP_MIN = -2.0
X_MIN_M = 40.0   # x-axis starts before sighted's 50M ckpt for visual padding
X_MAX_M = 260.0  # capped just past sighted convergence (250M) for consistent window
RCP_DIR = Path("/tmp/rcp_analysis")
LEGACY_BLIND_DIR = Path("results/probing_results")  # blind kept per memory


# ───────────────────── Panel A: linear GPS vs bandwidth ─────────────────────
def panel_a(ax, mlp_json: Path) -> None:
    """Linear GPS R² vs encoder spatial bandwidth (5-fold CV)."""
    data = json.loads(mlp_json.read_text())

    # Shaded regimes
    ax.axhspan(0.4, 1.05, color="#dceedc", alpha=0.55, zorder=0)
    ax.axhspan(-2.5, 0.4, color="#fbe0dc", alpha=0.45, zorder=0)
    ax.axhline(0, color="#888", lw=0.6, ls="--", zorder=0)

    for _rcp, mlp_key, label, cells, col, mk, _ in CONDS:
        d = data[mlp_key]
        r2, sd = d["linear_r2_mean"], d["linear_r2_std"]
        x = cells + (0.3 if mlp_key == "uniform"
                      else (-0.3 if mlp_key == "foveated" else 0))
        ax.errorbar(x, r2, yerr=sd, fmt=mk, color=col, markersize=10,
                    markeredgecolor="white", markeredgewidth=1.2,
                    capsize=3, lw=1.5, zorder=4, label=label)

    ax.text(13.5, 0.85, "Bottleneck regime\n(integration carries pos.)",
            fontsize=8, color="#3a7d3a", ha="right", va="top", style="italic")
    ax.text(13.5, -2.1, "Rich-encoder regime\n(visual route carries pos.)",
            fontsize=8, color="#a02528", ha="right", va="bottom", style="italic")

    ax.set_xlabel("Encoder spatial output (cells)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"top-layer linear GPS $R^2$ (5-fold CV)",
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
def _read_ckpt_value(path: Path):
    """Extract gps_cv_r2_mean + std + n_episodes from a ckpt analysis file."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    blk = d.get("1b_global_gps_compass", {})
    r2 = blk.get("gps_cv_r2_mean")
    if r2 is None:
        return None
    return {
        "r2": float(r2),
        "std": float(blk.get("gps_cv_r2_std", 0.0)),
        "n_eps": int(d.get("n_episodes", 0)),
    }


def panel_b(ax) -> None:
    """GPS R² across training checkpoints — substitution mechanism."""
    ax.axhspan(CLIP_MIN, 0, color="#f4d8d4", alpha=0.18, zorder=0)
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8, zorder=1)

    plotted_anything = False
    for rcp_key, _mlp, label, _cells, col, mk, frames_per_ckpt in CONDS:
        # Build (frames_M, r2, std, n_eps) list across available ckpts
        points = []
        if rcp_key == "blind_izar":
            # Legacy blind ckpt sweep (10/20/30/34) at frames_per_ckpt=10.06.
            # We restrict to the [50M, 250M] window for visual consistency
            # with sighted; ckpts 30/34 (300M/342M) are out-of-window and
            # the values there are essentially identical to ckpt20 anyway
            # (~0.95). Probing for ckpt5 (50M) and ckpt25 (252M) is queued
            # on RCP — see scripts/cluster/submit_blind_50_250.sh.
            for ck in (10, 20):
                p = LEGACY_BLIND_DIR / f"blind_gibson_ckpt{ck}_det_analysis.json"
                v = _read_ckpt_value(p)
                if v is not None:
                    points.append((ck * frames_per_ckpt, v))
            # blind_izar ckpt.25 (252M-equivalent, the 250M-comparable
            # endpoint per submit_probe_collect_rcp.sh special-casing)
            # is stored as blind_izar_det_analysis.json (no _ckpt suffix).
            ck25_p = RCP_DIR / "blind_izar_det_analysis.json"
            v = _read_ckpt_value(ck25_p)
            if v is not None:
                points.append((25 * frames_per_ckpt, v))
            # blind_izar ckpt.5 (50M-equivalent) — pending probe-5-c5 job
            # on RCP; will land at blind_izar_det_ckpt5_analysis.json.
            ck5_p = RCP_DIR / "blind_izar_det_ckpt5_analysis.json"
            v = _read_ckpt_value(ck5_p)
            if v is not None:
                points.append((5 * frames_per_ckpt, v))
        else:
            # New RCP sweep: ckpt 10/20/30/40 at 5M each => 50/100/150/200M
            for ck in (10, 20, 30, 40):
                p = RCP_DIR / f"{rcp_key}_det_ckpt{ck}_analysis.json"
                v = _read_ckpt_value(p)
                if v is not None:
                    points.append((ck * frames_per_ckpt, v))
            # Add converged 250M point from <cond>_det_analysis.json
            conv_p = RCP_DIR / f"{rcp_key}_det_analysis.json"
            v = _read_ckpt_value(conv_p)
            if v is not None:
                points.append((250.0, v))

        # Plot all points (200-ep RCP ckpts and 500-ep converged point both
        # treated as full data; the 5-fold CV std encodes the noise).
        xs, ys, errs = [], [], []
        clipped_at = []
        for x, v in points:
            # Restrict to [50M, 250M] window for cross-condition consistency
            if x < X_MIN_M - 1 or x > X_MAX_M + 1:
                continue
            y_raw = v["r2"]
            y = float(np.clip(y_raw, CLIP_MIN, 1.05))
            xs.append(x); ys.append(y); errs.append(v["std"])
            if y_raw < CLIP_MIN:
                clipped_at.append((x, y_raw))

        if xs:
            plotted_anything = True
            ax.errorbar(xs, ys, yerr=errs, marker=mk,
                        label=label, color=col, linewidth=2.0,
                        markersize=7, capsize=3.0, capthick=0.8,
                        elinewidth=0.8, ecolor=col, alpha=1.0, zorder=4)
        for x, r2_raw in clipped_at:
            ax.annotate(f"{r2_raw:.1f}", (x, CLIP_MIN + 0.06),
                        fontsize=8, fontweight="bold",
                        ha="center", va="bottom", color="darkred", zorder=5)

    if not plotted_anything:
        ax.text(0.5, 0.5, "(no across-ckpt JSONs found)",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    ax.set_ylim(CLIP_MIN - 0.10, 1.10)
    ax.set_xlim(X_MIN_M, X_MAX_M)
    # Explicit ticks at the 5 sampled points only (no auto-generated 175/225)
    ax.set_xticks([50, 100, 150, 200, 250])
    ax.set_xticklabels(["50", "100", "150", "200", "250"],
                        rotation=0, fontsize=10)
    ax.set_xlabel("training frames (M)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"top-layer GPS $R^2$ (5-fold CV)", fontsize=11, fontweight="bold")
    ax.set_title("(b) Substitution mechanism",
                 fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.annotate("", xy=(180, -1.0), xytext=(120, 0.5),
                arrowprops=dict(arrowstyle="->", color="#a02528", lw=1.0,
                                alpha=0.7))
    ax.text(120, 0.6, "rich-encoder\nsubstitution",
            fontsize=8, color="#a02528", style="italic", ha="left", va="bottom")


# ───────────────────── Panel C: pipeline view ────────────────────────────────
def panel_c(ax, mlp_json: Path) -> None:
    """Linear GPS R² along L0 → L1 → L2 (lines), plus MLP R² at L2 (stars)
    showing the format-shift gap: how much non-linear readout recovers above
    the linear floor for each condition."""
    mlp_data = json.loads(mlp_json.read_text())
    mlp_keymap = {
        "blind_izar": "blind_izar", "coarse": "coarse",
        "foveated_logpolar": "foveated_logpolar",
        "foveated": "foveated", "uniform": "uniform",
    }

    for rcp_key, _mlp, label, _cells, col, mk, _ in CONDS:
        main_p = RCP_DIR / f"{rcp_key}_det_analysis.json"
        layer_r2 = {0: None, 1: None, 2: None}
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
        for L in (0, 1, 2):
            if layer_r2[L] is not None:
                xs.append(L); ys.append(layer_r2[L])
        if xs:
            ax.plot(xs, ys, marker=mk, label=label,
                    color=col, linewidth=1.8, markersize=6.5, zorder=3)

        # MLP star marker at L2, plus vertical line showing the linear→MLP gap
        mlp_key = mlp_keymap.get(rcp_key)
        if mlp_key in mlp_data and layer_r2[2] is not None:
            mlp_r2 = float(np.clip(mlp_data[mlp_key]["mlp_r2_mean"],
                                    CLIP_MIN, 1.05))
            linear_l2 = layer_r2[2]
            # Slight x-jitter so MLP star doesn't overlap linear marker exactly
            x_mlp = 2.10
            # Vertical gap connector
            ax.plot([x_mlp, x_mlp], [linear_l2, mlp_r2], color=col,
                    linewidth=1.0, alpha=0.5, zorder=2)
            ax.plot(x_mlp, mlp_r2, marker="*", color=col,
                    markersize=11, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=4)

    # Highlight the linear→MLP gap = format shift
    ax.text(2.55, 0.7, "MLP probe\nat L2",
            fontsize=8, color="#222", ha="left", va="center", weight="bold")
    ax.text(2.55, 0.45, "(non-linear\nrecovery)",
            fontsize=7.5, color="#444", ha="left", va="center", style="italic")
    # Annotate the largest gap (uniform: linear=0.66 single-fit, but Panel A
    # 5-fold says -1.19; mlp=0.48 → gap is huge in CV terms). Use linear
    # single-fit floor for visual consistency within Panel C.
    ax.text(2.55, -0.5, "vertical bar\n= format shift\ngap at L2",
            fontsize=7.5, color="#666", ha="left", va="center", style="italic")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["L0\n(LSTM in)", "L1", "L2\n(top, policy)"], fontsize=9)
    ax.set_xlim(-0.4, 3.3)
    ax.set_ylim(CLIP_MIN - 0.05, 1.10)
    ax.set_xlabel("Pipeline location", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"GPS $R^2$", fontsize=11, fontweight="bold")
    ax.set_title("(c) Pipeline view: where the divergence sits",
                 fontsize=12, fontweight="bold", loc="left", x=0.0, pad=8)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ───────────────────────── compose ────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-json", type=Path,
                    default=RCP_DIR / "mlp_probe.json")
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
    panel_b(ax_b)
    panel_c(ax_c, args.mlp_json)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
