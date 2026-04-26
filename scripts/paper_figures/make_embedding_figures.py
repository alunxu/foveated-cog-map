"""
Dimensionality-reduction figures for hidden-state geometry (H2 supplement).

Two panels:
  1. All-conditions PCA/t-SNE colored by condition --- visual confirmation of
     H2 (each condition occupies a distinct region of the projection plane).
  2. Per-condition PCA colored by x-position --- whether each condition's
     memory has a smoothly-varying spatial gradient in the hidden state.

Usage:
    python scripts/paper_figures/make_embedding_figures.py \\
        --in-dir /tmp/probe_results --out-dir docs/cs503_final/fig
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COND_ORDER = ["blind", "uniform", "foveated", "foveated_learned", "matched"]
COND_DISPLAY = {
    "blind": ("Blind", "#444444"),
    "uniform": ("Uniform", "#4daf4a"),
    "foveated": ("Foveated (fix)", "#e41a1c"),
    "foveated_learned": ("Foveated (learned)", "#ff7f00"),
    "matched": ("Coarse", "#377eb8"),
}


def _load_npz(in_dir: Path, cond: str) -> dict | None:
    # Prefer deterministic-rollout NPZ (post-c81352e fix). Fall back to
    # the stochastic NPZ so this script still runs on old probing_data;
    # figures from those inputs need a caveat in the paper caption.
    if cond == "foveated_learned":
        candidates = [f"{cond}_gibson_det.npz",
                      f"{cond}_gibson_truncated.npz",
                      f"{cond}_gibson.npz"]
    else:
        candidates = [f"{cond}_gibson_det.npz",
                      f"{cond}_gibson.npz"]
    for name in candidates:
        p = in_dir / name
        if p.exists():
            d = dict(np.load(p))
            return d
    sys.stderr.write(f"[skip] {cond}: no npz\n")
    return None


def _subsample(n_points: int, target: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_points <= target:
        return np.arange(n_points)
    return rng.permutation(n_points)[:target]


# ---------------------------------------------------------------------------
# Figure 1: all-conditions projection colored by condition (H2 visual)
# ---------------------------------------------------------------------------


def fig_conditions_embedding(in_dir: Path, out_dir: Path, method: str = "pca",
                             per_cond: int = 1500) -> None:
    """Concatenate top-layer hidden states from all conditions, project to
    2D, and colour by condition.

    With ``method='pca'`` the projection is linear (fast, interpretable).
    With ``method='tsne'`` it is t-SNE (nonlinear, slower).
    """
    from sklearn.decomposition import PCA

    all_H: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    colours: list[str] = []
    labels: list[str] = []

    for cond_idx, cond in enumerate(COND_ORDER):
        d = _load_npz(in_dir, cond)
        if d is None:
            continue
        H = d["h_layers"][:, -1]  # (N, 512)
        idx = _subsample(H.shape[0], per_cond, seed=cond_idx)
        all_H.append(H[idx].astype(np.float32))
        all_c.append(np.full(len(idx), cond_idx, dtype=np.int32))
        labels.append(COND_DISPLAY[cond][0])
        colours.append(COND_DISPLAY[cond][1])

    X = np.concatenate(all_H, axis=0)
    y = np.concatenate(all_c, axis=0)

    if method == "pca":
        proj = PCA(n_components=2, random_state=0)
        Z = proj.fit_transform(X)
        title_suffix = "PCA"
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init="pca", random_state=0, perplexity=30,
                    n_iter=1000)
        Z = tsne.fit_transform(X)
        title_suffix = "t-SNE"

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for cond_idx, (lbl, col) in enumerate(zip(labels, colours)):
        mask = y == cond_idx
        ax.scatter(Z[mask, 0], Z[mask, 1], s=4, alpha=0.35, color=col,
                   label=lbl, linewidths=0)
    ax.legend(loc="upper left", fontsize=8, frameon=False, markerscale=3)
    ax.set_xlabel(f"{title_suffix} 1")
    ax.set_ylabel(f"{title_suffix} 2")
    ax.set_title(f"Hidden-state geometry across conditions ({title_suffix})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    # Only the tsne version is used in the paper (appfig 7); pca is kept
    # for sanity but generated under its old name (orphan).
    if method == "tsne":
        p = out_dir / "appfig7_h2_hidden_embedding_tsne.pdf"
    else:
        p = out_dir / f"h2_hidden_embedding_{method}.pdf"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: per-condition PCA coloured by x-position (memory-position
# manifold)
# ---------------------------------------------------------------------------


def fig_per_condition_position_pca(in_dir: Path, out_dir: Path,
                                    n_show: int = 2000) -> None:
    from sklearn.decomposition import PCA

    conds_to_show = [c for c in COND_ORDER if c != "matched"]
    # skip matched-48 (deprecated); include the other 4
    n = len(conds_to_show)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False)

    for ax_i, cond in enumerate(conds_to_show):
        ax = axes[0, ax_i]
        d = _load_npz(in_dir, cond)
        if d is None:
            ax.axis("off")
            ax.set_title(f"{COND_DISPLAY[cond][0]} (no data)")
            continue
        H = d["h_layers"][:, -1].astype(np.float32)
        pos = d["positions"].astype(np.float32)
        idx = _subsample(H.shape[0], n_show, seed=42 + ax_i)
        H_sub = H[idx]
        pos_sub = pos[idx]

        proj = PCA(n_components=2, random_state=0)
        Z = proj.fit_transform(H_sub)

        # Use x-coordinate of position as colour. Use the 0th coord.
        c = pos_sub[:, 0]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap="viridis", s=6,
                        alpha=0.6, linewidths=0)
        ax.set_title(COND_DISPLAY[cond][0], fontsize=10)
        ax.set_xlabel("PC1", fontsize=8)
        if ax_i == 0:
            ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Single shared colorbar on the right
    cbar = fig.colorbar(sc, ax=axes[0, :], shrink=0.8, pad=0.02,
                        label="agent $x$-position (world frame)")
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Per-condition PCA of top-layer LSTM hidden state, "
                 "coloured by agent position", fontsize=10, y=1.03)

    # Orphan: not used in current paper. Kept for reproducibility.
    p = out_dir / "h2_position_manifold.pdf"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--method", choices=["pca", "tsne"], default="pca")
    p.add_argument("--per-cond", type=int, default=1500)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_conditions_embedding(args.in_dir, args.out_dir, method=args.method,
                             per_cond=args.per_cond)
    fig_per_condition_position_pca(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
