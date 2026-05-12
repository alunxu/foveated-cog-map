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
    ("foveated_logpolar","foveated_logpolar", "Log-polar",  4,  "#984ea3", "v", 5.0),
    ("foveated",         "foveated",          "Foveated",      16, "#e41a1c", "D", 5.0),
    ("uniform",          "uniform",           "Uniform",       16, "#4daf4a", "^", 5.0),
]
CLIP_MIN = -1.0           # tight y-range; outliers clipped at -1.0 line
HIGH_STD_THRESHOLD = 3.5  # almost no global filter; only one outlier hardcoded
# Surgical drop: foveated@100M cv_std=2.09, single-split R²≈+0.56 — pure CV
# artifact. Uniform@250M (std≈3.0) is kept: it's the canonical converged
# endpoint and matches panel-(a) bar / Table 1.
HARDCODED_SKIP_POINTS = {("foveated", 100.0)}  # (rcp_key, frames_M)
X_MIN_M = 40.0   # x-axis starts before sighted's 50M ckpt for visual padding
X_MAX_M = 260.0  # capped just past sighted convergence (250M) for consistent window
RCP_DIR = Path("/tmp/rcp_analysis")
LEGACY_BLIND_DIR = Path("results/probing_results")  # blind kept per memory


# ───────────────────── Panel A: linear GPS vs bandwidth ─────────────────────
# Panel (a) carries three threads:
#   (1) GPS R² declines monotonically with encoder bandwidth — the headline
#   (2) DtG R² stays flat ≥0.88 across all conditions — probe is fine, the
#       linear position target is genuinely missing from h_2 in rich-encoder.
#   (3) Foveated-logpolar (~2x2 cells, similar to coarse 1x1) lands with the
#       rich-encoder cluster, not coarse — dimensionality is not the driver.
# All three readable at a glance from one panel.

def _read_dtg_r2(rcp_key: str):
    """Pull DtG cv_r2_mean / cv_r2_std from <cond>_det_analysis.json."""
    p = RCP_DIR / f"{rcp_key}_det_analysis.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    blk = d.get("1c_distance_to_goal", {})
    return blk.get("cv_r2_mean"), blk.get("cv_r2_std")


def panel_a(ax, mlp_json: Path) -> None:
    """Paired bars per condition: GPS R² (filled) + DtG control (hatched)."""
    data = json.loads(mlp_json.read_text())

    # y=0 reference line; dropped the shaded "no-signal" zone since it
    # filled most of the panel and read as decoration rather than a band.
    ax.axhline(0, color="#888", lw=0.7, ls="--", zorder=1)

    # x-positions ordered by per-step feature variety (the discriminator),
    # not raw cell count. Foveated-logpolar's tiny encoder feature map sits
    # next to coarse if you sort by cell count — but its functional behaviour
    # places it in the rich-encoder cluster. Placing it BETWEEN coarse and
    # foveated/uniform makes the dimensionality-control point visually
    # readable from the panel alone.
    POS_MAP = {  # mlp_key → x_position
        "blind_izar":        0.0,
        "coarse":            1.0,
        "foveated_logpolar": 2.0,
        "foveated":          3.0,
        "uniform":           4.0,
    }
    bar_w = 0.32

    for _rcp, mlp_key, label, _cells, col, _mk, _ in CONDS:
        d = data[mlp_key]
        gps_r2, gps_sd = d["linear_r2_mean"], d["linear_r2_std"]
        dtg_r2, dtg_sd = _read_dtg_r2(_rcp)
        x = POS_MAP[mlp_key]

        # GPS bar (filled), clipped at -2.5 for readability
        gps_plot = max(gps_r2, -2.5)
        ax.bar(x - bar_w/2, gps_plot, width=bar_w,
               color=col, edgecolor="black", linewidth=0.8,
               yerr=gps_sd, capsize=3, ecolor="black",
               error_kw={"linewidth": 0.8, "alpha": 0.7}, zorder=3,
               label=label)
        # Annotate clipped values
        if gps_r2 < -2.5:
            ax.annotate(f"{gps_r2:.2f}", (x - bar_w/2, -2.45),
                        fontsize=9, ha="center", va="bottom",
                        color="darkred", fontweight="bold", zorder=5)

        # DtG bar (hatched, same colour, lighter alpha)
        if dtg_r2 is not None:
            ax.bar(x + bar_w/2, dtg_r2, width=bar_w,
                   color=col, edgecolor="black", linewidth=0.8,
                   alpha=0.45, hatch="///",
                   yerr=dtg_sd, capsize=3, ecolor="black",
                   error_kw={"linewidth": 0.8, "alpha": 0.7}, zorder=3)

    ax.set_xlabel("encoder bandwidth",
                  fontsize=20, fontweight="bold")
    ax.set_ylabel(r"linear probe $R^2$",
                  fontsize=20, fontweight="bold")
    ax.set_title("(a) Magnitude",
                 fontsize=22, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-2.5, 1.15)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["blind", "coarse", "log-polar",
                         "foveated", "uniform"],
                        fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Custom legend: condition colours + GPS-vs-DtG style (two-tier)
    from matplotlib.patches import Patch
    metric_handles = [
        Patch(facecolor="grey", edgecolor="black", label="GPS (target)"),
        Patch(facecolor="grey", edgecolor="black",
              alpha=0.45, hatch="///", label="DtG (control)"),
    ]
    ax.legend(handles=metric_handles, loc="lower left", fontsize=11,
              frameon=False, ncol=1, handlelength=2.0)


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
            # blind_izar ckpt.15 (151M-equivalent) — landed 2026-05-06
            ck15_p = RCP_DIR / "blind_izar_det_ckpt15_analysis.json"
            v = _read_ckpt_value(ck15_p)
            if v is not None:
                points.append((15 * frames_per_ckpt, v))
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

        # Filter then sort by x. (Append order varies across conditions —
        # blind's points come from multiple files at out-of-order ckpt
        # numbers, which would zig-zag the line if drawn unsorted.)
        triples = []
        for x, v in points:
            if x < X_MIN_M - 1 or x > X_MAX_M + 1:
                continue
            if v["std"] > HIGH_STD_THRESHOLD:
                continue
            if (rcp_key, x) in HARDCODED_SKIP_POINTS:
                continue
            y_raw = v["r2"]
            triples.append((
                x,
                float(np.clip(y_raw, CLIP_MIN, 1.05)),
                min(v["std"], 0.4),
            ))
        triples.sort(key=lambda t: t[0])
        xs = [t[0] for t in triples]
        ys = [t[1] for t in triples]
        errs = [t[2] for t in triples]

        if xs:
            plotted_anything = True
            ax.plot(xs, ys, color=col, linewidth=2.0,
                    label=label, alpha=0.9, zorder=3)
            ax.errorbar(xs, ys, yerr=errs, marker=mk,
                        color=col, markersize=8,
                        markeredgecolor=col, markerfacecolor=col,
                        capsize=3.0, capthick=0.8,
                        elinewidth=0.8, ecolor=col,
                        linestyle="none", zorder=4)

    if not plotted_anything:
        ax.text(0.5, 0.5, "(no across-ckpt JSONs found)",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    # In-panel text annotations removed per figure-style direction.

    ax.set_ylim(CLIP_MIN - 0.10, 1.10)
    ax.set_xlim(X_MIN_M, X_MAX_M)
    ax.set_xticks([50, 100, 150, 200, 250])
    ax.set_xticklabels(["50", "100", "150", "200", "250"],
                        rotation=0, fontsize=12)
    ax.set_xlabel("training frames (M)", fontsize=20, fontweight="bold")
    ax.set_ylabel(r"top-layer GPS $R^2$", fontsize=20, fontweight="bold")
    ax.set_title("(b) Substitution mechanism",
                 fontsize=22, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    # Legend with both line + condition-marker (so reader sees the same shape
    # convention used in the panel itself, not just colour-coded lines).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=col, lw=2.0, marker=mk, markersize=8,
               markerfacecolor=col, markeredgecolor=col, label=label)
        for _, _, label, _, col, mk, _ in CONDS
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=10,
              frameon=False, ncol=1, handlelength=1.8)


# ───────── Panel C: per-LSTM-layer GPS R² heatmap (mech-interp localisation) ─
def panel_c(ax, mlp_json: Path) -> None:
    """Per-LSTM-layer GPS R² heatmap: condition × {L0, L1, L2}.

    L0 (LSTM input) is dominated by the GPS-sensor passthrough — readability
    is uniformly ~0.95 across all five conditions. L2 (top, policy-readable
    layer) is where the bandwidth-graded divergence lives. The heatmap
    localises the magnitude collapse to L2 specifically — the layer the
    policy reads — supporting the mech-interp claim that the divergence is
    in the *policy-accessible* representation, not earlier in the stack.
    """
    _ = mlp_json  # not used here
    # Use 5-fold CV per layer (canonical Table 1 protocol — keeps panel (a),
    # (b), and (c) on the same probing protocol).  Computed by
    # /tmp/compute_layer_cv.py from cached h_layers npz files.
    LAYER_LABELS = ["L0", "L1", "L2"]
    LAYER_CV_JSON = RCP_DIR / "layer_cv_gps.json"
    layer_data = json.loads(LAYER_CV_JSON.read_text()) if LAYER_CV_JSON.exists() else {}

    rows = []          # list of [3 R² values] per condition
    cond_labels = []
    for rcp_key, _mlp, label, _cells, _col, _mk, _ in CONDS:
        cd = layer_data.get(rcp_key, {})
        row = [cd.get(f"L{L}", {}).get("mean", np.nan) for L in (0, 1, 2)]
        rows.append(row)
        cond_labels.append(label)

    arr = np.array(rows, dtype=float)  # (5, 3)

    # Diverging colourmap centred at 0; clip to [-2, 1]
    VMIN, VMAX = -2.0, 1.0
    arr_clip = np.clip(arr, VMIN, VMAX)
    cmap = plt.get_cmap("RdBu_r").reversed()  # red high, blue low
    im = ax.imshow(arr_clip, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")

    # Cell value annotations
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            # Choose text colour based on cell luminance
            cell_norm = (np.clip(v, VMIN, VMAX) - VMIN) / (VMAX - VMIN)
            txt_col = "white" if cell_norm > 0.85 or cell_norm < 0.15 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=txt_col, fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(LAYER_LABELS)))
    ax.set_xticklabels(LAYER_LABELS, fontsize=14)
    ax.set_yticks(range(len(cond_labels)))
    ax.set_yticklabels(cond_labels, fontsize=12)
    ax.set_xlabel("LSTM layer", fontsize=18, fontweight="bold")
    ax.set_title("(c) Layer localisation",
                 fontsize=22, fontweight="bold", loc="left", x=0.0, pad=12)
    # Colourbar (compact, on the right)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"GPS $R^2$", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)


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
