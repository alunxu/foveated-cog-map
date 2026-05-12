"""
A1+A2 v3: Proper Skaggs spatial information bits per LSTM unit.

Earlier (v2) we used MI(active; bin) where active = h > μ+σ binarised. That is
information-theoretically valid but NOT the canonical Skaggs metric, which
expects a non-negative firing rate.

This v3 uses the canonical Skaggs 1996 formula directly, with rectified
hidden-unit activations max(h, 0) as proxy "firing rate":

    I = sum_b p(b) * (lambda_b / lambda_bar) * log2(lambda_b / lambda_bar)   [bits/sample]

where p(b) = occupancy probability of bin b, lambda_b = mean rectified
activation in bin b, lambda_bar = overall mean rectified activation.
For non-negative lambda_b, the formula is non-negative (Jensen).

We report v3 alongside v2 for comparison: v3 follows canonical Skaggs;
v2 uses MI(active; bin). Both are valid spatial-info measures; v3 is
more directly comparable to neuroscience literature.

References:
- Skaggs, W. E., McNaughton, B. L., Wilson, M. A., & Barnes, C. A. (1996).
  Theta phase precession in hippocampal neuronal populations and the
  compression of temporal sequences. Hippocampus 6(2), 149-172.
- Modern shuffle-corrected pipeline: Mallory et al. (2018, Hippocampus).

Reads:  /tmp/cond_npzs/{cond}_gibson_det.npz
Writes: /tmp/extra_analyses/skaggs_proper_v3.json
        docs/manuscript/fig/figa2b_place_cells.pdf  (replaces v2 figure)
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/scripts/paper_figures")
from _style import apply_paper_style
apply_paper_style()


CONDS = [
    ("blind",    "/tmp/cond_npzs/blind_gibson_det.npz",     "Blind",    "#444444"),
    ("coarse",   "/tmp/cond_npzs/matched_gibson_det.npz",   "Coarse",   "#377eb8"),
    ("foveated", "/tmp/cond_npzs/foveated_gibson_det.npz",  "Foveated", "#e41a1c"),
    ("uniform",  "/tmp/cond_npzs/uniform_gibson_det.npz",   "Uniform",  "#4daf4a"),
]
GRID = 8
N_TOP_SCENES = 15
N_MIN_SCENE = 800
N_SHUFFLES = 100


def normalize_xy(positions):
    x, y = positions[:, 0], positions[:, 2]
    if x.max() == x.min() or y.max() == y.min():
        return None
    nx = (x - x.min()) / (x.max() - x.min() + 1e-9)
    ny = (y - y.min()) / (y.max() - y.min() + 1e-9)
    return np.column_stack([nx, ny])


def skaggs_bits(rate: np.ndarray, bin_id: np.ndarray, n_bins: int) -> float:
    """
    Canonical Skaggs spatial information bits (Skaggs 1996).
        I = sum_b p(b) * (lambda_b / lambda_bar) * log2(lambda_b / lambda_bar)

    rate: non-negative firing rate per sample (use max(h, 0) for hidden unit).
    bin_id: spatial bin index per sample.
    n_bins: total number of bins.

    Returns: bits/sample. Non-negative for non-negative `rate`.
    """
    n = len(rate)
    # Per-bin occupancy and mean rate
    p_b = np.bincount(bin_id, minlength=n_bins).astype(np.float64) / n
    sum_b = np.bincount(bin_id, weights=rate, minlength=n_bins).astype(np.float64)
    cnt_b = np.bincount(bin_id, minlength=n_bins).astype(np.float64)
    mean_b = np.where(cnt_b > 0, sum_b / np.maximum(cnt_b, 1), 0.0)
    mean_overall = float(rate.mean())
    if mean_overall < 1e-9:
        return 0.0
    # Skaggs formula
    ratio = mean_b / mean_overall
    # Avoid log(0): only sum over bins with p_b > 0 and ratio > 0
    valid = (p_b > 0) & (ratio > 0)
    bits = (p_b[valid] * ratio[valid] * np.log2(ratio[valid])).sum()
    return float(max(bits, 0.0))  # canonical Skaggs is non-neg; numerical clip


def main():
    Path("/tmp/extra_analyses").mkdir(exist_ok=True)
    results = {}
    cond_data = {}

    for cond, path, label, color in CONDS:
        print(f"\n=== {cond} (Skaggs v3, rectified rate) ===")
        d = np.load(Path(path))
        h = d["hidden_states"].astype(np.float32)
        pos = d["positions"].astype(np.float32)
        scene_ids = d["scene_ids"]

        unique, counts = np.unique(scene_ids, return_counts=True)
        idx_sorted = np.argsort(counts)[::-1]
        top_scenes = [int(s) for s in unique[idx_sorted[:N_TOP_SCENES]]
                      if (scene_ids == s).sum() >= N_MIN_SCENE]
        print(f"  using {len(top_scenes)} scenes (≥{N_MIN_SCENE} samples)")

        n_units = h.shape[1]
        bits_per_unit_scene = np.zeros((n_units, len(top_scenes)))
        n_place_units_per_scene = np.zeros(len(top_scenes), dtype=int)

        # Rectify activations: ReLU(h)
        h_pos = np.maximum(h, 0.0)

        for si, scn in enumerate(top_scenes):
            mask = scene_ids == scn
            xy = normalize_xy(pos[mask])
            if xy is None:
                continue
            bin_x = np.clip((xy[:, 0] * GRID).astype(int), 0, GRID - 1)
            bin_y = np.clip((xy[:, 1] * GRID).astype(int), 0, GRID - 1)
            bin_id = bin_x * GRID + bin_y
            n_bins = GRID * GRID
            h_scene = h_pos[mask]

            # Per-unit Skaggs bits
            for ui in range(n_units):
                bits = skaggs_bits(h_scene[:, ui], bin_id, n_bins)
                bits_per_unit_scene[ui, si] = bits

            # Shuffle null on a sample of units (first 16)
            sample_units = list(range(min(16, n_units)))
            null_bits_99 = np.zeros(len(sample_units))
            rng = np.random.default_rng(scn)
            for j, ui in enumerate(sample_units):
                nulls = []
                for _ in range(N_SHUFFLES):
                    shift = rng.integers(low=10, high=len(h_scene) - 10)
                    rate_shuf = np.roll(h_scene[:, ui], shift)
                    nulls.append(skaggs_bits(rate_shuf, bin_id, n_bins))
                null_bits_99[j] = np.percentile(nulls, 99)
            # Use sample-mean threshold as proxy for full-pop threshold
            shuffle_th = float(null_bits_99.mean())
            n_place_units_per_scene[si] = int(np.sum(bits_per_unit_scene[:, si] > shuffle_th))

        # Aggregate across scenes
        bits_pooled = bits_per_unit_scene.flatten()
        bits_per_unit_mean = bits_per_unit_scene.mean(axis=1)  # (units,)

        results[cond] = {
            "label": label,
            "color": color,
            "n_top_scenes": len(top_scenes),
            "skaggs_bits_per_unit_scene_mean": float(bits_pooled.mean()),
            "skaggs_bits_per_unit_scene_median": float(np.median(bits_pooled)),
            "skaggs_bits_per_unit_scene_p90": float(np.percentile(bits_pooled, 90)),
            "skaggs_bits_per_unit_scene_max": float(bits_pooled.max()),
            "n_place_units_per_scene_mean": float(n_place_units_per_scene.mean()),
            "n_place_units_per_scene_max": int(n_place_units_per_scene.max()),
            "method": "Skaggs 1996 canonical, rectified ReLU(h)",
        }
        cond_data[cond] = (bits_per_unit_mean, color, label, n_place_units_per_scene)
        print(f"  Skaggs bits per (unit,scene): mean={bits_pooled.mean():.4f}, "
              f"median={np.median(bits_pooled):.4f}, "
              f"p90={np.percentile(bits_pooled, 90):.4f}, max={bits_pooled.max():.4f}")
        print(f"  Place units (>shuffle 99th) per scene: mean={n_place_units_per_scene.mean():.1f}, "
              f"max={n_place_units_per_scene.max()}")

    # Plot 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8),
                             gridspec_kw={"wspace": 0.30})
    cond_order = ["blind", "coarse", "foveated", "uniform"]
    cols = [next(c[3] for c in CONDS if c[0] == k) for k in cond_order]
    labs = [next(c[2] for c in CONDS if c[0] == k) for k in cond_order]

    # Panel A: Skaggs bits per unit (averaged across scenes)
    box_a = []
    for c in cond_order:
        if c not in cond_data: continue
        bits_per_unit_mean, _, _, _ = cond_data[c]
        box_a.append(bits_per_unit_mean)
    bp = axes[0].boxplot(box_a, labels=labs, patch_artist=True, widths=0.55,
                         showmeans=True,
                         meanprops=dict(marker="D", markerfacecolor="black",
                                        markeredgecolor="white", markersize=7),
                         medianprops=dict(color="black", lw=1.3))
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.55)
    axes[0].set_ylabel("Skaggs spatial info per unit (bits)\nrectified ReLU($\\mathbf{h}_2$); mean over scenes",
                       fontsize=10.5, fontweight="bold")
    axes[0].set_title("(a) Skaggs bits per LSTM unit",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    for s in ("top", "right"): axes[0].spines[s].set_visible(False)
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)

    # Panel B: # place units per scene (mean ± std)
    place_means = [results[k]["n_place_units_per_scene_mean"] for k in cond_order]
    place_maxs = [results[k]["n_place_units_per_scene_max"] for k in cond_order]
    bars = axes[1].bar(labs, place_means, color=cols, alpha=0.7,
                       edgecolor="black", linewidth=0.8)
    for i, (m, mx) in enumerate(zip(place_means, place_maxs)):
        axes[1].text(i, m + 5, f"mean {m:.1f}", ha="center",
                     fontsize=9.5, fontweight="bold")
        axes[1].text(i, m + 2, f"max {mx}", ha="center",
                     fontsize=8, color="grey")
    axes[1].set_ylabel("# place units per scene\n($>$ 99th-pct shuffle null)",
                       fontsize=10.5, fontweight="bold")
    axes[1].set_title("(b) Spatial-coding population size",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    for s in ("top", "right"): axes[1].spines[s].set_visible(False)
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)

    # Panel C: Mean Skaggs bits per cond (mean ± std)
    means_per_cond = [results[k]["skaggs_bits_per_unit_scene_mean"] for k in cond_order]
    medians_per_cond = [results[k]["skaggs_bits_per_unit_scene_median"] for k in cond_order]
    p90_per_cond = [results[k]["skaggs_bits_per_unit_scene_p90"] for k in cond_order]
    x = np.arange(len(cond_order))
    axes[2].bar(x - 0.2, means_per_cond, width=0.18, color=cols, alpha=0.7,
                edgecolor="black", label="mean")
    axes[2].bar(x, medians_per_cond, width=0.18, color=cols, alpha=0.5,
                edgecolor="black", label="median")
    axes[2].bar(x + 0.2, p90_per_cond, width=0.18, color=cols, alpha=0.3,
                edgecolor="black", label="p90")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labs)
    axes[2].set_ylabel("Skaggs bits/unit/scene\n(distribution summary)",
                       fontsize=10.5, fontweight="bold")
    axes[2].set_title("(c) Distribution per cond",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    axes[2].legend(loc="upper right", frameon=False, fontsize=9)
    for s in ("top", "right"): axes[2].spines[s].set_visible(False)
    axes[2].grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Skaggs spatial information per LSTM unit (canonical Skaggs 1996, rectified ReLU($\\mathbf{h}_2$))",
                 fontsize=11, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/figa2b_place_cells.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")

    Path("/tmp/extra_analyses/skaggs_proper_v3.json").write_text(json.dumps(results, indent=2))
    print("wrote /tmp/extra_analyses/skaggs_proper_v3.json")


if __name__ == "__main__":
    main()
