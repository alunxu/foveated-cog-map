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

    # Index-based positions to keep blind and coarse from overlapping;
    # cell counts shown below in xtick labels (1×1, 2×2, 4×4 notation).
    POS_MAP = {  # mlp_key → x_position
        "blind_izar": 0,
        "coarse": 1,
        "foveated_logpolar": 2,
        "foveated": 2.85,  # share "16 cells" with uniform but stagger
        "uniform": 3.55,
    }
    for _rcp, mlp_key, label, _cells, col, mk, _ in CONDS:
        d = data[mlp_key]
        r2, sd = d["linear_r2_mean"], d["linear_r2_std"]
        x = POS_MAP[mlp_key]
        ax.errorbar(x, r2, yerr=sd, fmt=mk, color=col, markersize=14,
                    markeredgecolor="white", markeredgewidth=1.4,
                    capsize=4, lw=2.0, zorder=4, label=label)

    ax.set_xlabel("Encoder spatial output", fontsize=20, fontweight="bold")
    ax.set_ylabel(r"top-layer linear GPS $R^2$",
                  fontsize=20, fontweight="bold")
    ax.set_title("(a) Magnitude collapse",
                 fontsize=26, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.set_xlim(-0.4, 4.0)
    ax.set_ylim(-2.5, 1.15)
    ax.set_xticks([0, 1, 2, 2.85, 3.55])
    ax.set_xticklabels(["0\n(blind)", "1×1\n(coarse)", "2×2\n(fov-LP)",
                         "4×4\n(foveated)", "4×4\n(uniform)"],
                        fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=11, frameon=False, ncol=1)


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
    ax.set_xticks([50, 100, 150, 200, 250])
    ax.set_xticklabels(["50", "100", "150", "200", "250"],
                        rotation=0, fontsize=12)
    ax.set_xlabel("training frames (M)", fontsize=20, fontweight="bold")
    ax.set_ylabel(r"top-layer GPS $R^2$", fontsize=20, fontweight="bold")
    ax.set_title("(b) Substitution mechanism",
                 fontsize=26, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.25)


# ───────────────────── Panel C: predictive horizon (lag-k) ─────────────────
def panel_c(ax, mlp_json: Path) -> None:
    """Predictive horizon: GPS R^2 of decoding pos_{t+k} from h_t vs lag k.
    The Stachenfeld 2017 SR-style cognitive-map signature — blind sustains
    high R^2 over long horizons (path-integrated forward-rollout structure);
    rich-encoder conditions crash because their L2 carries scene-conditional
    visual features rather than a stable position code.
    """
    _ = mlp_json  # not used here
    LAGK_JSON = RCP_DIR / "lagk_summary.json"
    if not LAGK_JSON.exists():
        ax.text(0.5, 0.5, "(lagk_summary.json missing)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        return
    data = json.loads(LAGK_JSON.read_text())
    KS = [0, 1, 2, 5, 10, 20, 50]

    Y_DISPLAY_MIN = -1.0
    for rcp_key, _mlp, label, _cells, col, mk, _ in CONDS:
        if rcp_key not in data:
            continue
        gps = data[rcp_key].get("GPS", {})
        xs, ys, errs, clipped = [], [], [], []
        for k in KS:
            entry = gps.get(f"k{k}")
            if entry is None:
                continue
            r2 = entry.get("mean")
            if r2 is None:
                continue
            xs.append(k)
            y_raw = float(r2)
            ys.append(float(np.clip(y_raw, Y_DISPLAY_MIN, 1.05)))
            errs.append(min(float(entry.get("std", 0)), 0.4))
            clipped.append(y_raw < Y_DISPLAY_MIN)
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker=mk, label=label,
                        color=col, linewidth=2.2, markersize=10,
                        markeredgecolor="white", markeredgewidth=1.0,
                        capsize=3, elinewidth=0.8, alpha=0.95, zorder=3)
            # Downward arrows on clipped markers (uniform mostly)
            for x, y, c in zip(xs, ys, clipped):
                if c:
                    ax.annotate("", xy=(x, y - 0.12), xytext=(x, y - 0.02),
                                arrowprops=dict(arrowstyle="->", color=col,
                                                lw=1.6, alpha=0.9),
                                zorder=4)

    # Light shaded "no-signal" band below R^2 = 0 to communicate that
    # negative-R^2 = linear model worse than predict-mean (i.e. noise).
    ax.axhspan(Y_DISPLAY_MIN - 0.05, 0, color="#fbe0dc", alpha=0.30, zorder=0)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS], fontsize=12)
    ax.set_xlim(-0.3, 60)
    ax.set_ylim(Y_DISPLAY_MIN - 0.05, 1.10)
    ax.set_xlabel("predictive horizon $k$ (steps ahead)",
                  fontsize=20, fontweight="bold")
    ax.set_ylabel(r"future-position $R^2$",
                  fontsize=20, fontweight="bold")
    ax.set_title("(c) Predictive horizon",
                 fontsize=26, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.25)


# ───────────────────────── compose ────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-json", type=Path,
                    default=RCP_DIR / "mlp_probe.json")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/manuscript/fig/fig_magnitude.pdf"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18.0, 5.5))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.05, 1.15, 1.10],
        wspace=0.28,
        top=0.86, bottom=0.22, left=0.05, right=0.99,
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
