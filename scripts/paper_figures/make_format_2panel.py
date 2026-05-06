"""§3.2 Format axis: 2-panel elegant figure.

Replaces 4 main-text §3.2 figures (linear vs MLP bars, LOSO box plot,
5×5 transplant heatmap, subspace divergence) with TWO info-dense panels:

  (a) Slope chart: linear → MLP probe per condition.
      Steep slope = large format-shift gap.
      Inspired by Stachenfeld 2017's paired-condition slope plots.

  (b) Format-dichotomy scatter: LOSO median R^2 (within-condition
      scene-invariance) vs cross-condition transplant susceptibility
      (across-condition non-interchangeability).
      Each condition = one point on a 2D plane.
      Bottleneck cluster: top-left (high LOSO, low transplant cost).
      Rich-encoder cluster: bottom-right (low LOSO, high transplant cost).
      Inspired by Sanders 2020's MDS-projected evidence-ratio scatters.

Data:
  - linear/MLP R^2: /tmp/rcp_analysis/mlp_probe.json
  - LOSO median: /tmp/rcp_analysis_v3/loso_5cond.json
  - Transplant: /tmp/rcp_analysis_v3/{donor}_to_{recipient}_mid30.json
                aggregated per recipient
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


CONDS = [
    # (rcp_key,            mlp_key,             loso_key,            label,     color,    marker)
    ("blind_izar",        "blind_izar",        "blind",            "Blind",         "#444444", "o"),
    ("coarse",            "coarse",            "coarse",           "Coarse",        "#377eb8", "s"),
    ("foveated_logpolar", "foveated_logpolar", "foveated_logpolar","Fov-logpolar",  "#984ea3", "v"),
    ("foveated",          "foveated",          "foveated",          "Foveated",      "#e41a1c", "D"),
    ("uniform",           "uniform",           "uniform",          "Uniform",       "#4daf4a", "^"),
]
CLIP_MIN = -1.0
RCP_DIR = Path("/tmp/rcp_analysis")
RCP_V3 = Path("/tmp/rcp_analysis_v3")


# ───────────────────────── Panel A: Probe-depth sweep ──────────────────
PROBE_ORDER = ["linear", "MLP-1", "MLP-2", "MLP-4"]
PROBE_LABELS = {"linear": "Linear", "MLP-1": "MLP-1",
                "MLP-2": "MLP-2", "MLP-4": "MLP-4"}
# Map our internal CONDS keys to the probe_depth_sweep.json keys.
DEPTH_KEY_MAP = {
    "blind_izar":        "blind",
    "coarse":            "coarse",
    "foveated":          "foveated",
    "foveated_logpolar": "foveated_logpolar",
    "uniform":           "uniform",
}


def panel_a(ax, depth_json: Path) -> None:
    """Probe-depth sweep per condition: how much non-linearity does each
    condition require to recover position? Bottleneck conditions plateau
    near depth 0 (linear suffices); rich-encoder conditions ramp up with
    probe depth (position is non-linearly encoded). Format-shift severity
    per condition becomes visible as the slope of each curve.
    """
    data = json.loads(depth_json.read_text())
    xs = np.arange(len(PROBE_ORDER), dtype=float)
    for rcp_key, _mlp, _loso, label, col, mk in CONDS:
        depth_key = DEPTH_KEY_MAP[rcp_key]
        d = data.get(depth_key, {})
        ys = []
        for p in PROBE_ORDER:
            r = d.get(p, {})
            v = r.get("r2_mean", np.nan)
            ys.append(float(np.clip(v, CLIP_MIN, 1.05)))
        ys = np.array(ys)
        # Connecting line — steep slope = strong format shift
        ax.plot(xs, ys, color=col, linewidth=2.2, alpha=0.85, zorder=2)
        # Markers per depth
        ax.plot(xs, ys, marker=mk, color=col, markersize=11,
                markeredgecolor="white", markeredgewidth=1.4,
                linestyle="", zorder=3)
        # Right-edge label at the deepest probe
        ax.text(xs[-1] + 0.08, ys[-1], label, fontsize=11, color=col,
                va="center", weight="bold")

    ax.axhline(0, color="#888", linewidth=0.5, ls="--", zorder=0)
    # Light gray "no-signal" zone
    ax.axhspan(CLIP_MIN - 0.05, 0, color="#fbe0dc", alpha=0.30, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([PROBE_LABELS[p] for p in PROBE_ORDER],
                       fontsize=12, fontweight="bold")
    ax.set_xlim(-0.25, xs[-1] + 0.95)
    ax.set_ylim(CLIP_MIN - 0.05, 1.10)
    ax.set_ylabel(r"GPS $R^2$ at $\mathbf{h}_2$",
                  fontsize=20, fontweight="bold")
    ax.set_xlabel("probe depth", fontsize=14, fontweight="bold")
    ax.set_title("(a) Format-shift severity",
                 fontsize=26, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.25)


# ───────────────────────── Panel B: 2D scatter ──────────────────────────
def _aggregate_transplant_recipient_cost(rcp_dir: Path) -> dict:
    """For each recipient condition, mean abs(cross_spl_delta) across donors.

    Returns: {recipient_key: mean_abs_delta}
    """
    keys = ["blind_izar", "coarse", "foveated_logpolar", "foveated", "uniform"]
    # Map cluster filenames (no '_izar') to keys
    KEY_FILE_MAP = {
        "blind_izar": "blind",
        "coarse": "coarse",
        "foveated_logpolar": "foveated_logpolar",
        "foveated": "foveated",
        "uniform": "uniform",
    }
    out = {}
    for rec_key in keys:
        rec_file = KEY_FILE_MAP[rec_key]
        deltas = []
        for don_key in keys:
            if don_key == rec_key:
                continue
            don_file = KEY_FILE_MAP[don_key]
            p = rcp_dir / f"{don_file}_to_{rec_file}_mid30.json"
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                    delta = d.get("cross_spl_delta", None)
                    if delta is not None:
                        deltas.append(abs(float(delta)))
                except Exception:
                    pass
        out[rec_key] = float(np.mean(deltas)) if deltas else np.nan
    return out


def panel_b(ax, loso_json: Path, transplant_dir: Path) -> None:
    """LOSO median R^2 vs cross-condition transplant cost (recipient mean)."""
    loso = json.loads(loso_json.read_text())
    trans = _aggregate_transplant_recipient_cost(transplant_dir)

    for rcp_key, _mlp, loso_key, label, col, mk in CONDS:
        x = float(loso[loso_key]["median_r2"])
        y = float(trans.get(rcp_key, np.nan))
        if np.isnan(y):
            continue
        ax.scatter(x, y, s=320, color=col, marker=mk,
                   edgecolor="white", linewidth=1.8, zorder=3, label=label)
        # Label offset per condition
        OFFSETS = {
            "Blind":         (0.04, 0.005),
            "Coarse":        (0.04, 0.0),
            "Fov-logpolar":  (-0.05, 0.005),
            "Foveated":      (0.04, 0.0),
            "Uniform":       (-0.05, 0.0),
        }
        dx, dy = OFFSETS.get(label, (0.04, 0.005))
        ha = "right" if dx < 0 else "left"
        ax.text(x + dx, y + dy, label, fontsize=11, color=col,
                ha=ha, va="center", weight="bold")

    # Median-split guide line
    ax.axvline(0.5, color="#aaa", linewidth=0.6, ls="--", zorder=0)
    # Quadrant labels in axes-fraction coordinates
    ax.text(0.04, 0.95,
            "scene-conditional\n+ brittle to transplant",
            transform=ax.transAxes, fontsize=9, color="#a02528",
            ha="left", va="top", style="italic", weight="bold", alpha=0.9)
    ax.text(0.96, 0.05,
            "scene-invariant\n+ robust to transplant",
            transform=ax.transAxes, fontsize=9, color="#3a7d3a",
            ha="right", va="bottom", style="italic", weight="bold", alpha=0.9)

    xlo, xhi = -0.05, 1.05
    ylo, yhi = -0.01, 0.26
    ax.set_xlabel("LOSO median $R^2$ (scene-invariance, within)",
                  fontsize=20, fontweight="bold")
    ax.set_ylabel("transplant cost\n(non-interchangeability, across)",
                  fontsize=20, fontweight="bold")
    ax.set_title("(b) Format dichotomy",
                 fontsize=26, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle=":", alpha=0.25)


# ────────────────── Panel C: subspace divergence (10 pairs) ──────────────
def panel_c(ax, subspace_json: Path) -> None:
    """2D scatter: principal angle × direction cosine for all 10 condition
    pairs. Geometric corroboration of the behavioural transplant test in
    Panel B: if conditions occupy non-interchangeable subspaces, all
    pairwise tests should cluster near 'fully orthogonal' (90°, cos=0).
    """
    data = json.loads(subspace_json.read_text())
    conds = data["conds"]
    short = {"blind": "Bl", "coarse": "Co", "foveated": "Fv",
             "foveated_logpolar": "Fl", "uniform": "Un"}
    color = {"blind": "#444444", "coarse": "#377eb8",
             "foveated": "#e41a1c", "foveated_logpolar": "#984ea3",
             "uniform": "#4daf4a"}
    angle_mat = np.array(data["angle_matrix_deg"])
    cos_mat = np.array(data["cos_matrix_x"])

    # Extract upper triangle (10 pairs)
    for i in range(5):
        for j in range(i + 1, 5):
            angle = float(angle_mat[i, j])
            cos = float(cos_mat[i, j])
            # Use a 2-tone marker: outer color = condition i, inner = j
            ax.scatter(angle, cos, s=240, c=color[conds[i]],
                       edgecolor=color[conds[j]], linewidth=2.4,
                       alpha=0.92, zorder=3)
            label = f"{short[conds[i]]}--{short[conds[j]]}"
            ax.annotate(label, (angle, cos), xytext=(7, 3),
                        textcoords="offset points", fontsize=9.5,
                        color="#333", weight="bold")

    # "Fully orthogonal" target: 90°, cos=0
    ax.axvline(90, color="#3a7d3a", ls="--", lw=0.9, alpha=0.7, zorder=1)
    ax.axhline(0, color="#3a7d3a", ls="--", lw=0.9, alpha=0.7, zorder=1)
    ax.text(89.5, 0.13, "fully\northogonal\n(90°, 0)",
            fontsize=10, color="#3a7d3a", style="italic", weight="bold",
            ha="right", va="top", alpha=0.85)

    ax.set_xlabel("principal angle (deg)",
                  fontsize=20, fontweight="bold")
    ax.set_ylabel("Ridge-probe\ndirection cosine",
                  fontsize=20, fontweight="bold")
    ax.set_title("(c) Subspace divergence",
                 fontsize=26, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.set_xlim(65, 92)
    ax.set_ylim(-0.15, 0.15)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle=":", alpha=0.25)


# ──────────────────────────── compose ────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-json", type=Path,
                    default=RCP_DIR / "probe_depth_sweep.json",
                    help="probe-depth sweep JSON (linear / MLP-1 / MLP-2 / MLP-4)")
    ap.add_argument("--loso-json", type=Path, default=RCP_V3 / "loso_5cond.json")
    ap.add_argument("--transplant-dir", type=Path, default=RCP_V3)
    ap.add_argument("--subspace-json", type=Path,
                    default=RCP_V3 / "subspace_divergence_5cond.json")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/manuscript/fig/fig3_format_axis.pdf"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20.0, 5.8))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[0.85, 1.20, 1.0],
        wspace=0.32,
        top=0.86, bottom=0.16, left=0.05, right=0.99,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a, args.mlp_json)
    panel_b(ax_b, args.loso_json, args.transplant_dir)
    panel_c(ax_c, args.subspace_json)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
