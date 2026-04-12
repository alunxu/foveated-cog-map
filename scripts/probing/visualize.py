"""
Visualization of probing data for Habitat navigation agents.

Generates publication-quality plots comparing hidden-state representations
across conditions (blind, uniform, foveated, matched-compute).

Plots:
  1. PCA of hidden states colored by position (per condition, 2x2)
  2. t-SNE of hidden states colored by position (per condition, 2x2)
  3. PCA colored by episode ID (per condition, 2x2)
  4. Combined PCA overlay — all conditions in one space (H2 figure)
  5. Path-history decay curves — lag vs R^2 per condition (H1 figure)

Usage:
    python scripts/probing/visualize.py \
        --data blind=/path/blind.npz uniform=/path/uniform.npz \
              foveated=/path/foveated.npz matched=/path/matched.npz \
        --results-json blind=/path/blind_analysis.json \
                       foveated=/path/foveated_analysis.json ... \
        --out-dir /path/to/figures/
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

CONDITION_COLORS = {
    "blind":    "#1f77b4",  # blue
    "uniform":  "#ff7f0e",  # orange
    "foveated": "#2ca02c",  # green
    "matched":  "#d62728",  # red
}

CONDITION_ORDER = ["blind", "uniform", "foveated", "matched"]

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize probing data across conditions")
    p.add_argument(
        "--data", nargs="+", required=True,
        help="Condition data: name=/path/to/data.npz (e.g. blind=blind.npz)")
    p.add_argument(
        "--results-json", nargs="*", default=[],
        help="Analysis JSON files: name=/path/to/analysis.json "
             "(needed for Plot 5: path-history decay)")
    p.add_argument(
        "--out-dir", required=True,
        help="Directory to save figures")
    p.add_argument(
        "--n-subsample", type=int, default=2000,
        help="Max points per condition for t-SNE (default: 2000)")
    p.add_argument(
        "--n-subsample-pca", type=int, default=5000,
        help="Max points per condition for PCA plots (default: 5000)")
    p.add_argument(
        "--seed", type=int, default=42)
    p.add_argument(
        "--tsne-perplexity", type=float, default=30.0,
        help="t-SNE perplexity (default: 30)")
    p.add_argument(
        "--skip-tsne", action="store_true",
        help="Skip t-SNE plot (slow)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_condition(spec):
    """Parse 'name=/path/to.npz' and load data."""
    name, path = spec.split("=", 1)
    data = np.load(path, allow_pickle=True)
    return name, data


def load_results_json(spec):
    """Parse 'name=/path/to.json' and load results."""
    name, path = spec.split("=", 1)
    with open(path) as f:
        results = json.load(f)
    return name, results


def subsample_arrays(arrays, n, seed):
    """Subsample multiple arrays to n rows with the same random indices."""
    rng = np.random.RandomState(seed)
    N = len(arrays[0])
    if N <= n:
        return arrays
    idx = rng.choice(N, n, replace=False)
    return [a[idx] for a in arrays]


def get_hidden_states(data):
    """Get top-layer hidden states from .npz data."""
    return data["hidden_states"]


def get_gps(data):
    """Get GPS positions, falling back to position x,z if gps unavailable."""
    if "gps" in data:
        return data["gps"]
    # Fallback: use positions[:, [0, 2]] (x, z world coords)
    return data["positions"][:, [0, 2]]


def get_episode_ids(data):
    """Get episode IDs."""
    return data["episode_ids"]


def spatial_color(gps):
    """Compute a scalar for coloring: distance from origin or x-coordinate."""
    return np.sqrt(gps[:, 0] ** 2 + gps[:, 1] ** 2)


def ordered_conditions(conditions):
    """Return condition names in canonical order, with any extras appended."""
    ordered = [c for c in CONDITION_ORDER if c in conditions]
    extras = [c for c in conditions if c not in CONDITION_ORDER]
    return ordered + extras


def condition_color(name):
    """Get color for a condition, defaulting to gray."""
    return CONDITION_COLORS.get(name, "#7f7f7f")


def save_figure(fig, out_dir, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 1: PCA colored by position (per condition)
# ═══════════════════════════════════════════════════════════════════════

def plot_pca_by_position(conditions, out_dir, n_sub, seed):
    """2x2 grid: PCA of hidden states colored by spatial distance."""
    names = ordered_conditions(conditions)
    n_cond = len(names)
    ncols = min(n_cond, 2)
    nrows = (n_cond + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_cond == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        data = conditions[name]
        H = get_hidden_states(data)
        gps = get_gps(data)

        # Subsample
        [H_sub, gps_sub] = subsample_arrays([H, gps], n_sub, seed)

        # PCA
        pca = PCA(n_components=2, random_state=seed)
        Z = pca.fit_transform(H_sub)

        # Color by distance from origin
        c = spatial_color(gps_sub)

        sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap="viridis",
                        s=3, alpha=0.5, rasterized=True)
        ax.set_title(name.capitalize())
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        fig.colorbar(sc, ax=ax, label="Dist. from origin (m)", shrink=0.8)

    # Hide unused axes
    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("PCA of Hidden States Colored by Spatial Position", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "01_pca_position")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 2: t-SNE colored by position (per condition)
# ═══════════════════════════════════════════════════════════════════════

def plot_tsne_by_position(conditions, out_dir, n_sub, seed, perplexity):
    """2x2 grid: t-SNE of hidden states colored by spatial distance."""
    names = ordered_conditions(conditions)
    n_cond = len(names)
    ncols = min(n_cond, 2)
    nrows = (n_cond + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_cond == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        data = conditions[name]
        H = get_hidden_states(data)
        gps = get_gps(data)

        # Subsample (t-SNE is O(n^2))
        [H_sub, gps_sub] = subsample_arrays([H, gps], n_sub, seed)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity,
                     random_state=seed, init="pca", learning_rate="auto")
        Z = tsne.fit_transform(H_sub)

        c = spatial_color(gps_sub)

        sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap="viridis",
                        s=3, alpha=0.5, rasterized=True)
        ax.set_title(name.capitalize())
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.colorbar(sc, ax=ax, label="Dist. from origin (m)", shrink=0.8)

    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("t-SNE of Hidden States Colored by Spatial Position", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "02_tsne_position")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 3: PCA colored by episode (per condition)
# ═══════════════════════════════════════════════════════════════════════

def plot_pca_by_episode(conditions, out_dir, n_sub, seed):
    """2x2 grid: PCA colored by episode ID to show clustering."""
    names = ordered_conditions(conditions)
    n_cond = len(names)
    ncols = min(n_cond, 2)
    nrows = (n_cond + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_cond == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        data = conditions[name]
        H = get_hidden_states(data)
        ep_ids = get_episode_ids(data)

        [H_sub, ep_sub] = subsample_arrays([H, ep_ids], n_sub, seed)

        pca = PCA(n_components=2, random_state=seed)
        Z = pca.fit_transform(H_sub)

        # Map episode IDs to sequential integers for coloring
        unique_eps = np.unique(ep_sub)
        ep_map = {eid: idx for idx, eid in enumerate(unique_eps)}
        ep_colors = np.array([ep_map[e] for e in ep_sub])

        sc = ax.scatter(Z[:, 0], Z[:, 1], c=ep_colors, cmap="tab20",
                        s=3, alpha=0.5, rasterized=True)
        ax.set_title(name.capitalize())
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        n_eps = len(unique_eps)
        ax.text(0.02, 0.98, f"{n_eps} episodes",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("PCA of Hidden States Colored by Episode", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "03_pca_episode")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 4: Combined PCA overlay (all conditions in one space)
# ═══════════════════════════════════════════════════════════════════════

def plot_pca_combined(conditions, out_dir, n_sub, seed):
    """Single PCA fitted on concatenated hidden states from all conditions.

    Color by condition to show representational divergence (H2).
    """
    names = ordered_conditions(conditions)

    # Subsample and concatenate
    all_H = []
    all_labels = []
    all_sizes = []
    for name in names:
        data = conditions[name]
        H = get_hidden_states(data)
        [H_sub] = subsample_arrays([H], n_sub, seed)
        all_H.append(H_sub)
        all_labels.extend([name] * len(H_sub))
        all_sizes.append(len(H_sub))

    H_cat = np.vstack(all_H)
    labels = np.array(all_labels)

    # Fit PCA on concatenated data
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(H_cat)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot each condition
    for name in names:
        mask = labels == name
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=condition_color(name), label=name.capitalize(),
                   s=4, alpha=0.3, rasterized=True)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Joint PCA: Representational Divergence Across Conditions")
    ax.legend(markerscale=4, frameon=True, framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, out_dir, "04_pca_combined")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 5: Path-history decay curves
# ═══════════════════════════════════════════════════════════════════════

def plot_path_history_decay(results_json, out_dir):
    """Line plot: R^2 vs lag for each condition.

    Reads from analysis JSON files (output of analyze.py), specifically
    the "2c_path_history" key.
    """
    names = ordered_conditions(results_json)
    if not names:
        print("  Skipping Plot 5 (no results JSON provided)")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for name in names:
        res = results_json[name]
        ph = res.get("2c_path_history", [])
        if not ph:
            print(f"  Warning: no path-history data for {name}, skipping")
            continue

        lags = [r["lag_k"] for r in ph]
        r2s = [r["r2"] for r in ph]

        ax.plot(lags, r2s, marker="o", color=condition_color(name),
                label=name.capitalize(), linewidth=2, markersize=5)

    ax.set_xlabel("Lag k (timesteps)")
    ax.set_ylabel("R$^2$ (GPS decoding)")
    ax.set_title("Path-History Decay: Memory of Past Positions")
    ax.set_xticks(range(0, 6))
    ax.legend(frameon=True, framealpha=0.9)
    ax.set_ylim(bottom=None, top=None)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, out_dir, "05_path_history_decay")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load condition data
    conditions = {}
    for spec in args.data:
        name, data = load_condition(spec)
        conditions[name] = data
        H = get_hidden_states(data)
        print(f"Loaded {name}: {len(H)} steps, dim={H.shape[1]}")

    # Load results JSON (for path-history decay)
    results_json = {}
    for spec in args.results_json:
        name, res = load_results_json(spec)
        results_json[name] = res
        print(f"Loaded results JSON: {name}")

    # ── Plot 1: PCA by position ──────────────────────────────────────
    print("\nPlot 1: PCA colored by position ...")
    plot_pca_by_position(conditions, args.out_dir, args.n_subsample_pca, args.seed)

    # ── Plot 2: t-SNE by position ────────────────────────────────────
    if not args.skip_tsne:
        print("\nPlot 2: t-SNE colored by position ...")
        plot_tsne_by_position(conditions, args.out_dir, args.n_subsample,
                              args.seed, args.tsne_perplexity)
    else:
        print("\nPlot 2: t-SNE skipped (--skip-tsne)")

    # ── Plot 3: PCA by episode ───────────────────────────────────────
    print("\nPlot 3: PCA colored by episode ...")
    plot_pca_by_episode(conditions, args.out_dir, args.n_subsample_pca, args.seed)

    # ── Plot 4: Combined PCA overlay ─────────────────────────────────
    print("\nPlot 4: Combined PCA overlay ...")
    plot_pca_combined(conditions, args.out_dir, args.n_subsample_pca, args.seed)

    # ── Plot 5: Path-history decay ───────────────────────────────────
    print("\nPlot 5: Path-history decay curves ...")
    plot_path_history_decay(results_json, args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
