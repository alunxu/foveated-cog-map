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
    # rcp_key is also used as the transplant-file stem (blind, coarse, fnorm, ...) and the NPZ_FILES lookup.
    ("blind",             "blind_izar",        "blind",            "Blind",         "#444444", "o"),
    ("coarse",            "coarse",            "coarse",           "Coarse",        "#377eb8", "s"),
    ("fnorm",             "fnorm",             "foveated",         "Foveated",      "#e41a1c", "D"),
    ("foveated_logpolar", "foveated_logpolar", "foveated_logpolar","Log-polar",     "#984ea3", "v"),
    ("uniform",           "uniform",           "uniform",          "Uniform",       "#4daf4a", "^"),
]
# Explicit NPZ filename per condition (used by panel A PCA manifold).
NPZ_FILES = {
    "blind":             "blind_det_ckpt49.npz",
    "coarse":            "coarse_det.npz",
    "foveated_logpolar": "foveated_logpolar_det.npz",
    "fnorm":             "fnorm_det_ckpt49.npz",
    "uniform":           "uniform_det.npz",
}
CLIP_MIN = -1.0
RCP_DIR = Path("/tmp/rcp_analysis")
RCP_V3 = Path("/tmp/rcp_analysis_v3")
NPZ_DIR = Path("/tmp/rcp_analysis_v3")  # h_layers npz files


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


def panel_a_manifold(fig, parent_gs) -> None:
    """5 mini-panels: PCA-2D projection of top-layer h_2 per condition,
    coloured by ground-truth position. Reads the manifold geometry of the
    position code: blind = clean position-organized manifold (linear-readable);
    coarse = position-organized but more diffuse; rich-encoder = scene-clustered
    fragments (each scene a separate cloud). Direct visual evidence for the
    'info preserved, format shifted' story carried abstractly by panels (b, c).
    """
    sub = parent_gs.subgridspec(1, 6, wspace=0.06, hspace=0.0,
                                  width_ratios=[1, 1, 1, 1, 1, 0.06])
    rng = np.random.RandomState(42)
    SUB_N = 1500
    sc_last = None

    # Precompute global x-range (5th-95th percentile across all conditions)
    # so all panels share an absolute colorbar scale yet ignore extreme
    # outliers that would otherwise compress the visible colour range.
    pooled_x = []
    sampled = {}
    for rcp_key, *_ in CONDS:
        p = NPZ_DIR / NPZ_FILES.get(rcp_key, f"{rcp_key}_det_RCP.npz")
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        H = d["h_layers"][:, 2, :]
        pos = d["positions"][:, [0, 2]]
        n = min(SUB_N, H.shape[0])
        idx = rng.choice(H.shape[0], n, replace=False)
        sampled[rcp_key] = (H[idx], pos[idx])
        pooled_x.append(pos[idx, 0])
    pooled_x = np.concatenate(pooled_x) if pooled_x else np.array([0.0])
    vmin = float(np.percentile(pooled_x, 5))
    vmax = float(np.percentile(pooled_x, 95))

    for col_i, (rcp_key, _, _, label, col, _) in enumerate(CONDS):
        ax = fig.add_subplot(sub[0, col_i])
        if rcp_key not in sampled:
            ax.text(0.5, 0.5, "(npz missing)", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            continue
        H_s, pos_s = sampled[rcp_key]
        # Centre + PCA-2 (np SVD; no sklearn dep)
        Hc = H_s - H_s.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        H2 = Hc @ Vt[:2].T            # (n, 2)
        sc = ax.scatter(H2[:, 0], H2[:, 1], c=pos_s[:, 0],
                        cmap="viridis", s=6, alpha=0.7,
                        edgecolors="none", rasterized=True,
                        vmin=vmin, vmax=vmax)
        sc_last = sc
        # Variance-explained annotation (bottom-right)
        var_pc12 = float((S[:2] ** 2).sum() / (S ** 2).sum())
        ax.text(0.97, 0.03, f"{var_pc12*100:.0f}% PC1+2",
                transform=ax.transAxes, fontsize=11, color="#666",
                ha="right", va="bottom", style="italic")
        ax.set_title(label, fontsize=18, color=col, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sname in ("top", "right"):
            ax.spines[sname].set_visible(False)
        ax.spines["left"].set_color("#bbb")
        ax.spines["bottom"].set_color("#bbb")
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
    # Joint axis labels
    sub_axes = fig.axes[-5:]
    if len(sub_axes) >= 1:
        sub_axes[0].set_ylabel("PC2", fontsize=15, fontweight="bold")
        for ax in sub_axes:
            ax.set_xlabel("PC1", fontsize=15, fontweight="bold")
    # Compact colorbar at right of the manifold row (absolute metres).
    if sc_last is not None:
        cax = fig.add_subplot(sub[0, 5])
        cb = fig.colorbar(sc_last, cax=cax)
        cb.set_label("position $x$ (m)", fontsize=20, fontweight="bold")
        cb.ax.tick_params(labelsize=14)


# ───────────────────────── Panel B: 2D scatter ──────────────────────────
def _aggregate_transplant_recipient_cost(rcp_dir: Path) -> dict:
    """For each recipient condition, mean abs(cross_spl_delta) across donors.

    Returns: {recipient_key: mean_abs_delta}
    """
    keys = ["blind", "coarse", "fnorm", "foveated_logpolar", "uniform"]
    # Map cluster filenames — these match the transplant_results/{donor}_to_{recipient}_mid30.json naming.
    KEY_FILE_MAP = {
        "blind":             "blind",
        "coarse":            "coarse",
        "foveated_logpolar": "foveated_logpolar",
        "fnorm":             "fnorm",
        "uniform":           "uniform",
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
            "Log-polar":  (-0.05, 0.005),
            "Foveated":      (0.04, 0.0),
            "Uniform":       (-0.05, 0.0),
        }
        dx, dy = OFFSETS.get(label, (0.04, 0.005))
        ha = "right" if dx < 0 else "left"
        ax.text(x + dx, y + dy, label, fontsize=17, color=col,
                ha=ha, va="center", weight="bold")

    # Median-split guide line
    ax.axvline(0.5, color="#aaa", linewidth=0.6, ls="--", zorder=0)
    # Quadrant labels in axes-fraction coordinates
    ax.text(0.04, 0.95,
            "scene-conditional\n+ brittle to transplant",
            transform=ax.transAxes, fontsize=17, color="#a02528",
            ha="left", va="top", style="italic", weight="bold", alpha=0.9)
    ax.text(0.96, 0.05,
            "scene-invariant\n+ robust to transplant",
            transform=ax.transAxes, fontsize=17, color="#3a7d3a",
            ha="right", va="bottom", style="italic", weight="bold", alpha=0.9)

    xlo, xhi = -0.05, 1.05
    ylo, yhi = -0.01, 0.26
    ax.set_xlabel("LOSO $R^2$ (scene-invariance)",
                  fontsize=20, fontweight="bold")
    ax.set_ylabel("transplant cost",
                  fontsize=20, fontweight="bold")
    # Title placed via fig.text in main() so all three panel titles align.
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
              "foveated_logpolar": "Log-polar",
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

    # Pair labels to the LEFT of each row. Each half rendered in its own
    # condition colour so the row can be cross-referenced to the manifold/
    # phase-diagram panels at a glance. Composed via HPacker (matplotlib
    # offsetbox machinery) so the three sub-strings — name1, dash, name2 —
    # share a single anchor and mixed colours.
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
    cond_colour = {
        "Blind":     "#444444",
        "Coarse":    "#377eb8",
        "Log-polar": "#984ea3",
        "Foveated":  "#e41a1c",
        "Uniform":   "#4daf4a",
    }
    for i, (_, _, c1, c2) in enumerate(pairs):
        name1, name2 = pretty[c1], pretty[c2]
        col1 = cond_colour.get(name1, "#222")
        col2 = cond_colour.get(name2, "#222")
        t1 = TextArea(name1, textprops=dict(color=col1, fontsize=18,
                                              weight="bold"))
        t_dash = TextArea("–", textprops=dict(color="#666", fontsize=18))
        t2 = TextArea(name2, textprops=dict(color=col2, fontsize=18,
                                              weight="bold"))
        pack = HPacker(children=[t1, t_dash, t2], align="center",
                       pad=0, sep=2)
        ab = AnchoredOffsetbox(loc="center left", child=pack, pad=0,
                               frameon=False,
                               bbox_to_anchor=(48.0, y[i]),
                               bbox_transform=ax.transData)
        ax.add_artist(ab)

    # Reference line at 90° = fully orthogonal.
    ax.axvline(90, color="#222", ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.text(90, n - 0.1, "  fully\n  orthogonal",
            fontsize=17, color="#222", style="italic",
            ha="left", va="top")
    # (median/IQR summary deferred to prose / appendix; figure stays clean)

    ax.set_xlabel("principal angle (deg)",
                  fontsize=20, fontweight="bold")
    # Title placed via fig.text in main() so all three panel titles align.
    ax.set_xlim(47.0, 93.5)
    ax.set_ylim(-0.8, n - 0.4)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=14)
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

    fig = plt.figure(figsize=(22.0, 7.0))
    # 4-column outer layout with an explicit spacer between (a) and (b)
    # to push (b) further right of (a) without changing (a)'s sub-grid.
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.25, 0.18, 1.05, 1.10],  # col 1 = empty spacer
        wspace=0.10,
        top=0.84, bottom=0.13, left=0.04, right=0.99,
    )
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 3])

    # Panel (a): 5 sub-panels carved from gs[0, 0]
    panel_a_manifold(fig, gs[0, 0])
    panel_b(ax_b, args.loso_json, args.transplant_dir)
    panel_c(ax_c, args.subspace_json)

    # Panel titles: (a) sits a touch higher because it has sub-panel
    # condition labels (Blind / Coarse / ...) immediately below it that
    # need clearance; (b)/(c) drop a touch so they sit just above their
    # axes content rather than floating at the figure top.
    TITLE_FS = 26
    title_kw = dict(fontsize=TITLE_FS, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.97, "(a) Manifold Geometry", **title_kw)
    fig.text(ax_b.get_position().x0, 0.94, "(b) Format dichotomy", **title_kw)
    fig.text(ax_c.get_position().x0, 0.94, "(c) Subspace divergence", **title_kw)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
