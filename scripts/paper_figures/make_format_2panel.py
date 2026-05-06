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
    """Horizontal dot plot: principal angle for each of the 10 condition
    pairs, sorted high-to-low.  Drops the redundant direction-cosine
    second axis and the two-tone marker scheme of v1 — the orthogonality
    story collapses into one number per pair, and a 1D strip plot reads
    that story directly.  A reference line at 90° marks fully-orthogonal;
    the foveated/foveated-logpolar pair (sole sub-70° outlier, the two
    conditions sharing the foveated retina) is annotated separately.
    """
    data = json.loads(subspace_json.read_text())
    conds = data["conds"]
    pretty = {"blind": "Blind", "coarse": "Coarse",
              "foveated": "Foveated",
              "foveated_logpolar": "Fov-LP",
              "uniform": "Uniform"}
    angle_mat = np.array(data["angle_matrix_deg"])

    # Collect 10 unordered pairs with their angles.
    pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            label = f"{pretty[conds[i]]}–{pretty[conds[j]]}"
            pairs.append((label, float(angle_mat[i, j]),
                          conds[i], conds[j]))
    # Sort by angle, descending (most-orthogonal first).
    pairs.sort(key=lambda t: -t[1])
    n = len(pairs)
    angles = np.array([p[1] for p in pairs])

    # Highlight the within-foveated outlier (Fov-LP/Foveated).
    is_fov_pair = np.array([
        ({c1, c2} == {"foveated", "foveated_logpolar"})
        for _, _, c1, c2 in pairs
    ])

    y = np.arange(n)
    # Bars from 90° anchor to each pair's angle, so the visual length
    # encodes the *gap from orthogonal* (longer bar = less orthogonal).
    for i, ang in enumerate(angles):
        col = "#c95a3d" if is_fov_pair[i] else "#3a7d3a"
        ax.hlines(y[i], ang, 90, color=col, lw=2.5,
                  alpha=0.55 if not is_fov_pair[i] else 0.9, zorder=2)
        ax.scatter(ang, y[i], s=110, color=col,
                   edgecolor="white", linewidth=1.4, zorder=3)

    # Pair labels to the LEFT of each row (left-justified at 64°).
    for i, (label, _, _, _) in enumerate(pairs):
        ax.text(64.5, y[i], label, ha="left", va="center",
                fontsize=12, color="#222")

    # Reference line at 90° = fully orthogonal.
    ax.axvline(90, color="#222", ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.text(90, n - 0.1, "  fully\n  orthogonal",
            fontsize=11, color="#222", style="italic",
            ha="left", va="top")

    # Summary annotation: median + IQR.
    med = float(np.median(angles))
    q1, q3 = float(np.percentile(angles, 25)), float(np.percentile(angles, 75))
    ax.text(64.5, -1.0,
            f"median {med:.0f}° (IQR {q1:.0f}–{q3:.0f}°); "
            f"9/10 pairs $\\geq 75°$, only Foveated–Fov-LP at $\\sim$67°",
            fontsize=11, color="#444", style="italic")

    ax.set_xlabel("principal angle (deg)  --  larger = more orthogonal",
                  fontsize=18, fontweight="bold")
    ax.set_title("(c) Subspace divergence",
                 fontsize=26, fontweight="bold", loc="left", x=0.0, pad=12)
    ax.set_xlim(63.5, 93.5)
    ax.set_ylim(-1.6, n - 0.4)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=12)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.25)


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
