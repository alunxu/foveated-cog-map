"""
WJ-E: per-condition + joint t-SNE of LSTM top-layer hidden states.

Wijmans 2023 Fig 1C used t-SNE of (h_t, c_t) coloured by action × collision
state to find emergent collision-detection neurons in blind agents. Here we
adapt the analysis to validate H2 (cross-condition format divergence) at
the t-SNE level, complementing the linear 1-NN purity result.

Two views:
- Joint t-SNE (left panel): pool ~1000 hidden states per condition (5×1000),
  embed to 2-D via t-SNE, color by condition. If our H2 finding holds in
  non-linear space (not just under linear similarity), each condition's
  points form a distinct cluster.
- Per-condition t-SNE (right panel grid): each condition's hidden states
  separately, coloured by distance-to-goal bin (low/medium/high). Reveals
  any within-condition spatial-information axis that linear probes might
  miss.

Reads:  --subsample-dir <dir>/{cond}_subsample.npz
Writes: <out-dir>/appfig_tsne.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()
import numpy as np
from sklearn.manifold import TSNE


CONDS = [
    ("blind",            "Blind",          "#444444"),
    ("matched",          "Coarse (1$\\times$1)", "#377eb8"),
    ("uniform",          "Uniform",        "#4daf4a"),
    ("foveated",         "Foveated (fix)", "#e41a1c"),
    ("foveated_learned", "Foveated (learned)", "#ff7f00"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-cond-n", type=int, default=1000,
                    help="samples per condition for joint t-SNE")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("=== Loading hidden states ===")
    Xs, ys, dtgs, steps = [], [], [], []
    full_hidden = {}
    full_dtg = {}
    full_step = {}
    for ci, (key, _label, _col) in enumerate(CONDS):
        p = args.subsample_dir / f"{key}_subsample.npz"
        if not p.exists():
            print(f"  skip {key}: {p} missing"); continue
        d = np.load(p)
        h = d["hidden"]               # (N, 512)
        full_hidden[key] = h
        full_dtg[key] = d["distance_to_goal"]
        full_step[key] = d["step"]
        idx = rng.choice(len(h), size=min(args.per_cond_n, len(h)), replace=False)
        Xs.append(h[idx]); ys.append(np.full(len(idx), ci, dtype=np.int32))
        dtgs.append(d["distance_to_goal"][idx]); steps.append(d["step"][idx])
        print(f"  {key}: {len(idx)} sampled (full={len(h)})")

    X = np.vstack(Xs); y = np.concatenate(ys)
    print(f"\n=== Joint t-SNE on {X.shape[0]} pooled samples ({X.shape[1]}-d) ===")
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=args.seed, n_jobs=-1)
    X2 = tsne.fit_transform(X)
    print(f"  joint t-SNE done")

    # Per-condition t-SNE for the right grid.
    per_cond_2d = {}
    per_cond_dtg = {}
    per_cond_step = {}
    n_each = 2000
    for key, _label, _col in CONDS:
        if key not in full_hidden: continue
        h = full_hidden[key]
        idx = rng.choice(len(h), size=min(n_each, len(h)), replace=False)
        h_sub = h[idx]
        ts = TSNE(n_components=2, perplexity=30, init="pca",
                  learning_rate="auto", random_state=args.seed, n_jobs=-1)
        per_cond_2d[key] = ts.fit_transform(h_sub)
        per_cond_dtg[key] = full_dtg[key][idx]
        per_cond_step[key] = full_step[key][idx]
        print(f"  per-cond t-SNE done: {key}")

    # ==================== Figure ====================
    fig = plt.figure(figsize=(15.0, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 2.0], wspace=0.18)

    # --- Left: joint t-SNE coloured by condition ---
    ax_joint = fig.add_subplot(gs[0, 0])
    for ci, (_key, label, col) in enumerate(CONDS):
        m = (y == ci)
        if m.sum() == 0: continue
        ax_joint.scatter(X2[m, 0], X2[m, 1], s=8, c=col, alpha=0.55,
                         edgecolor="none", label=label)
    ax_joint.set_title("Joint t-SNE: 5 conditions pooled", loc="left")
    ax_joint.set_xticks([]); ax_joint.set_yticks([])
    ax_joint.legend(loc="upper right", fontsize=9, frameon=False,
                    markerscale=1.6, handletextpad=0.4)
    for s_ in ("top", "right", "bottom", "left"):
        ax_joint.spines[s_].set_visible(True)
        ax_joint.spines[s_].set_linewidth(0.5)
        ax_joint.spines[s_].set_color("#888")

    # --- Right: 2x3 grid of per-condition t-SNEs coloured by DtG bin ---
    sub_gs = gs[0, 1].subgridspec(2, 3, wspace=0.12, hspace=0.20)
    flat_axes = [fig.add_subplot(sub_gs[r, c]) for r in range(2) for c in range(3)]

    for ax, (key, label, col) in zip(flat_axes, CONDS):
        if key not in per_cond_2d:
            ax.text(0.5, 0.5, f"missing {key}", transform=ax.transAxes,
                    ha="center", va="center"); continue
        emb = per_cond_2d[key]
        dtg = per_cond_dtg[key]
        # Color by DtG (distance to goal): close=green, mid=yellow, far=red
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=5, c=dtg,
                        cmap="viridis_r", alpha=0.7, edgecolor="none",
                        vmin=np.percentile(dtg, 2), vmax=np.percentile(dtg, 98))
        ax.set_title(label, loc="left", fontsize=10,
                     color=col, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right", "bottom", "left"):
            ax.spines[s_].set_visible(True)
            ax.spines[s_].set_linewidth(0.5)
            ax.spines[s_].set_color("#888")

    # 6th axis: shared colorbar for DtG
    if len(flat_axes) >= 6:
        cax = flat_axes[5]
        cax.cla()
        cax.set_xticks([]); cax.set_yticks([])
        for s_ in ("top", "right", "bottom", "left"):
            cax.spines[s_].set_visible(False)
        cb_ax = cax.inset_axes([0.10, 0.38, 0.18, 0.50])
        cb = fig.colorbar(sc, cax=cb_ax)
        cb.set_label("dist to goal (m)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cax.text(0.32, 0.65, "Per-condition t-SNE\ncoloured by\ndistance-to-goal",
                 transform=cax.transAxes, fontsize=10, va="center",
                 color="#444")

    plt.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.05)
    out = args.out_dir / "appfig_tsne.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
