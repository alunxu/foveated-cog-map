"""
Population coding analysis: how is spatial information structured across
LSTM units in each condition?

Question: Is the bottleneck-condition GPS code carried by a few
specialised "place cells" (sparse code) or distributed broadly across
the population (dense code)?  And do rich-encoder networks have a
similar population structure or a fundamentally different one?

Five analyses, all from existing det NPZs:

  1. Per-unit spatial-information distribution (histogram of bits per
     unit, per condition).
  2. Top-N rate-map gallery: per condition, plot the rate maps of the
     N most spatially-tuned units. Visual signature of place-cell-like
     spatial tuning.
  3. Sparse-vs-distributed decoding: how many units do we need to reach
     R^2 = X for GPS?  Curve of probe R^2 as a function of number of
     top-info units used.
  4. Per-unit GPS importance via coefficient norm under standardised
     Ridge probe (which units the linear policy can read most from).
  5. Intrinsic dimension via PCA: how many PCs to explain 95% variance
     of pooled hidden states per condition?

Usage:
    python scripts/probing/population_coding_analysis.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --suffix _det \\
        --out-json /scratch/izar/wxu/probing_results/population_coding_det.json \\
        --out-fig-dir docs/manuscript/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


CONDS = [
    ("blind_gibson",            "Blind",      "#444444"),
    ("matched_gibson",          "Coarse",     "#377eb8"),
    ("foveated_gibson",         "Foveated",   "#e41a1c"),
    ("uniform_gibson",          "Uniform",    "#4daf4a"),
]


# ---------------------------------------------------------------------------
# 1. Per-unit spatial-information distribution
# ---------------------------------------------------------------------------

def compute_per_unit_spatial_info(
    H: np.ndarray, positions: np.ndarray, scene_ids: np.ndarray,
    n_bins: int = 20, min_steps_per_scene: int = 20,
) -> np.ndarray:
    """Average per-unit spatial information across scenes.

    Returns
    -------
    bits : (hidden_dim,) ndarray
        Mean spatial information per unit (averaged across scenes).
    """
    si_per_scene = []
    for sid in np.unique(scene_ids):
        mask = scene_ids == sid
        if mask.sum() < min_steps_per_scene:
            continue
        h_s = H[mask]
        p_s = positions[mask]
        # 2-D position bin
        x = p_s[:, 0]
        y = p_s[:, 1]
        x_edges = np.linspace(x.min(), x.max() + 1e-6, n_bins + 1)
        y_edges = np.linspace(y.min(), y.max() + 1e-6, n_bins + 1)
        x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
        y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)
        bin_idx = x_idx * n_bins + y_idx  # flat bin (n_bins**2 cells)
        n_cells = n_bins * n_bins

        # Cell occupancies p(x)
        cell_count = np.bincount(bin_idx, minlength=n_cells).astype(float)
        if cell_count.sum() == 0:
            continue
        p_x = cell_count / cell_count.sum()

        # Per-unit per-cell mean activation r(x).
        # Vectorise: H is (T, D), bin_idx is (T,).
        mean_per_cell = np.zeros((n_cells, H.shape[1]), dtype=np.float64)
        for c in range(n_cells):
            m = bin_idx == c
            if m.sum() > 0:
                mean_per_cell[c] = h_s[m].mean(axis=0)
        # Overall mean firing rate r_bar = E[r(x)] over occupancy.
        r_bar = (p_x[:, None] * mean_per_cell).sum(axis=0)
        # Spatial info I(unit) = sum_x p(x) * r(x) / r_bar * log2(r(x)/r_bar)
        # only valid where r > 0 and r_bar > 0.  Guard with abs() for
        # negative firing rates (LSTM activations include negatives).
        # Use shifted-positive version: r_pos = r - min(r) + 1e-6.
        r_min = mean_per_cell.min(axis=0, keepdims=True)
        r_pos = mean_per_cell - r_min + 1e-6  # (cells, D)
        r_bar_pos = (p_x[:, None] * r_pos).sum(axis=0)  # (D,)
        ratio = r_pos / (r_bar_pos[None, :] + 1e-9)  # (cells, D)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(ratio > 0, np.log2(ratio), 0.0)
        si = (p_x[:, None] * r_pos / (r_bar_pos[None, :] + 1e-9) * log_term).sum(axis=0)
        si_per_scene.append(si)

    if not si_per_scene:
        return np.zeros(H.shape[1])
    si_matrix = np.stack(si_per_scene, axis=0)  # (scenes, D)
    return si_matrix.mean(axis=0)


# ---------------------------------------------------------------------------
# 2. Rate maps for top-N units (gallery)
# ---------------------------------------------------------------------------

def rate_map_for_unit(
    h_unit: np.ndarray, positions: np.ndarray,
    n_bins: int = 16,
) -> np.ndarray:
    """Return mean activation of one unit binned by 2-D position."""
    x = positions[:, 0]
    y = positions[:, 1]
    x_edges = np.linspace(x.min(), x.max() + 1e-6, n_bins + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-6, n_bins + 1)
    rmap = np.zeros((n_bins, n_bins), dtype=np.float64) * np.nan
    counts = np.zeros((n_bins, n_bins), dtype=int)
    sums = np.zeros((n_bins, n_bins), dtype=np.float64)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)
    np.add.at(sums, (x_idx, y_idx), h_unit)
    np.add.at(counts, (x_idx, y_idx), 1)
    valid = counts > 0
    rmap[valid] = sums[valid] / counts[valid]
    return rmap


def plot_rate_map_gallery(
    H: np.ndarray, positions: np.ndarray, scene_ids: np.ndarray,
    top_unit_ids: list[int], cond_label: str, color: str,
    out_path: Path, n_grid: int = 3,
):
    """3x3 gallery of rate maps for the top-9 most-info units (in their
    most-data scene)."""
    # Pick the largest scene for each unit so the rate map is well-sampled.
    unique_scenes, counts = np.unique(scene_ids, return_counts=True)
    big_scene = unique_scenes[counts.argmax()]
    mask = scene_ids == big_scene

    fig, axes = plt.subplots(n_grid, n_grid, figsize=(8, 8.5))
    for i, ax in enumerate(axes.flat):
        if i >= len(top_unit_ids):
            ax.axis("off")
            continue
        u = top_unit_ids[i]
        rmap = rate_map_for_unit(H[mask, u], positions[mask], n_bins=16)
        im = ax.imshow(
            rmap.T, origin="lower", aspect="auto", cmap="viridis",
        )
        ax.set_title(f"unit {u}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{cond_label} — rate maps of top-9 spatially-tuned units (largest scene)",
        fontsize=10, color=color,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# 3. Sparse-vs-distributed decoding
# ---------------------------------------------------------------------------

def sparse_decoding_curve(
    H: np.ndarray, gps: np.ndarray, ep_ids: np.ndarray,
    si_per_unit: np.ndarray,
    k_values: list[int],
    n_folds: int = 5, alpha: float = 10.0, seed: int = 42,
) -> list[dict]:
    """For each k, decode GPS from the top-k spatial-info units."""
    rng = np.random.RandomState(seed)
    unique_eps = np.unique(ep_ids)
    rng.shuffle(unique_eps)
    kf = KFold(n_splits=n_folds, shuffle=False)

    # Sort units by descending spatial info.
    sort_idx = np.argsort(-si_per_unit)

    results = []
    for k in k_values:
        sel = sort_idx[:k]
        H_sel = H[:, sel]
        r2s = []
        for tri, tei in kf.split(unique_eps):
            train_mask = np.isin(ep_ids, unique_eps[tri])
            test_mask = np.isin(ep_ids, unique_eps[tei])
            sc = StandardScaler()
            Xtr = sc.fit_transform(H_sel[train_mask])
            Xte = sc.transform(H_sel[test_mask])
            ridge = Ridge(alpha=alpha).fit(Xtr, gps[train_mask])
            pred = ridge.predict(Xte)
            r2s.append(r2_score(gps[test_mask], pred,
                                multioutput="uniform_average"))
        results.append({
            "k": k, "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        })
    return results


# ---------------------------------------------------------------------------
# 4. Per-unit GPS coefficient magnitude (under standardised Ridge)
# ---------------------------------------------------------------------------

def per_unit_gps_coef(
    H: np.ndarray, gps: np.ndarray, alpha: float = 10.0,
) -> np.ndarray:
    """L2 norm of standardised Ridge coefficients per unit (across the
    2-D GPS target).  Higher = unit more useful for GPS decoding."""
    sc = StandardScaler()
    Xs = sc.fit_transform(H)
    ridge = Ridge(alpha=alpha).fit(Xs, gps)
    # ridge.coef_ shape: (2, D)
    return np.linalg.norm(ridge.coef_, axis=0)  # (D,)


# ---------------------------------------------------------------------------
# 5. Intrinsic dimension via PCA
# ---------------------------------------------------------------------------

def intrinsic_dim_pca(H: np.ndarray, threshold: float = 0.95) -> int:
    """Number of PCs to explain `threshold` fraction of variance."""
    sc = StandardScaler().fit(H)
    Hs = sc.transform(H)
    pca = PCA().fit(Hs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cum >= threshold) + 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--suffix", type=str, default="_det")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-fig-dir", type=Path, required=True)
    ap.add_argument("--n-top-units", type=int, default=9)
    args = ap.parse_args()

    args.out_fig_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"per_condition": {}}

    # Collect per-condition spatial-info arrays for the cross-condition
    # population figure later.
    si_per_cond = {}
    intrinsic_per_cond = {}

    for cond_key, cond_label, color in CONDS:
        p = args.in_dir / f"{cond_key}{args.suffix}.npz"
        if not p.exists():
            print(f"[skip] {p}")
            continue
        d = np.load(p)
        H = d["hidden_states"].astype(np.float32)  # (T, 512)
        gps = d["gps"]                              # (T, 2)
        positions = d["positions"][:, :2] if "positions" in d.files else gps
        scene_ids = d["scene_ids"]
        ep_ids = d["episode_ids"]
        print(f"\n=== {cond_label} | T={H.shape[0]}, D={H.shape[1]} ===")

        # 1. Per-unit spatial info
        print("  computing per-unit spatial info ...")
        si = compute_per_unit_spatial_info(H, positions, scene_ids)
        si_per_cond[cond_key] = si
        n_above_1bit = int((si > 1.0).sum())

        # 2. Rate-map gallery
        print("  generating rate-map gallery ...")
        top_units = list(np.argsort(-si)[: args.n_top_units].tolist())
        plot_rate_map_gallery(
            H, positions, scene_ids,
            top_units, cond_label, color,
            args.out_fig_dir / f"rate_maps_{cond_key}.pdf",
        )

        # 3. Sparse decoding curve
        print("  computing sparse decoding curve ...")
        sparse_curve = sparse_decoding_curve(
            H, gps, ep_ids, si,
            k_values=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        )

        # 4. Per-unit GPS coefficient norm
        print("  computing per-unit Ridge coef norms ...")
        coef_norm = per_unit_gps_coef(H, gps)

        # 5. Intrinsic dim
        print("  computing intrinsic dim ...")
        idim = intrinsic_dim_pca(H, threshold=0.95)
        intrinsic_per_cond[cond_key] = idim

        results["per_condition"][cond_key] = {
            "label": cond_label,
            "hidden_dim": int(H.shape[1]),
            "n_steps": int(H.shape[0]),
            "spatial_info_mean": float(si.mean()),
            "spatial_info_max": float(si.max()),
            "n_units_above_1bit": n_above_1bit,
            "top_units": top_units,
            "spatial_info_per_unit": si.tolist(),
            "coef_norm_per_unit": coef_norm.tolist(),
            "sparse_decoding_curve": sparse_curve,
            "intrinsic_dim_pca_95pct": idim,
        }
        print(f"  spatial_info: mean={si.mean():.2f}, max={si.max():.2f}, "
              f"n_above_1bit={n_above_1bit}/{H.shape[1]}")
        print(f"  intrinsic_dim (95%): {idim}")
        print("  sparse decoding R² at k=1, 8, 64, 512:")
        for entry in sparse_curve:
            if entry["k"] in (1, 8, 64, 512):
                print(f"    k={entry['k']:4} R²={entry['r2_mean']:+.3f}")

    # ---- Cross-condition figure: spatial-info distribution + sparse curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    # Histogram (cumulative) of spatial info
    for cond_key, label, color in CONDS:
        if cond_key not in si_per_cond:
            continue
        si = si_per_cond[cond_key]
        sorted_si = np.sort(si)[::-1]
        x = np.arange(1, len(sorted_si) + 1)
        axes[0].plot(x, sorted_si, color=color, label=label, lw=1.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("unit rank (sorted by spatial info, log scale)")
    axes[0].set_ylabel("spatial information (bits)")
    axes[0].axhline(1.0, ls=":", color="grey", alpha=0.5, lw=0.8,
                    label="1-bit threshold")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].set_title("(a) Per-unit spatial-information distribution")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Sparse decoding curves
    for cond_key, label, color in CONDS:
        body = results["per_condition"].get(cond_key)
        if body is None:
            continue
        ks = [e["k"] for e in body["sparse_decoding_curve"]]
        means = [e["r2_mean"] for e in body["sparse_decoding_curve"]]
        axes[1].plot(ks, means, marker="o", color=color, label=label,
                     markersize=4, lw=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of top-spatial-info units used")
    axes[1].set_ylabel("GPS $R^2$ (5-fold CV mean)")
    axes[1].axhline(0, ls=":", color="grey", alpha=0.5, lw=0.8)
    axes[1].set_title("(b) Sparse-vs-distributed GPS decoding")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    cross_path = args.out_fig_dir / "figa9_population_coding.pdf"
    fig.savefig(cross_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {cross_path}")

    # Save results JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
