"""
Decisive cross-scene-invariance test: leave-one-scene-out (LOSO) CV.

For each cond:
  For each test scene s_t:
    1. Train Ridge on ALL OTHER scenes' (h, gps) pairs (mean-centered globally)
    2. Predict gps on scene s_t
    3. Record R²(s_t)
  Distribution of R²(s_t) across scenes:
    - Blind: tight, all near full-cond R²
    - Uniform: wide, many failures

The variance of R²(s_t) IS the cross-scene-invariance metric.
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
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
N_MIN = 100  # min test-scene samples to bother evaluating
N_MAX = 30000  # cap total samples for speed
TOP_K_SCENES = 50  # leave-one-out over the K largest scenes


def main():
    results = {}
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.6))

    cond_results = {}
    for cond, path, label, color in CONDS:
        print(f"\n=== {cond} ===")
        d = np.load(path)
        h = d["hidden_states"].astype(np.float32)
        gps = d["gps"].astype(np.float32)
        scene_ids = d["scene_ids"]

        # Subsample for speed
        if len(h) > N_MAX:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(h), N_MAX, replace=False)
            h = h[idx]; gps = gps[idx]; scene_ids = scene_ids[idx]

        # Mean-center globally
        h = h - h.mean(axis=0, keepdims=True)
        gps = gps - gps.mean(axis=0, keepdims=True)

        # Pick top K scenes by sample count
        unique, counts = np.unique(scene_ids, return_counts=True)
        idx_sorted = np.argsort(counts)[::-1]
        top_scenes = unique[idx_sorted[:TOP_K_SCENES]]
        top_scenes = [s for s in top_scenes if (scene_ids == s).sum() >= N_MIN]
        print(f"  Top scenes: {len(top_scenes)} (≥{N_MIN} samples each)")

        loso_r2 = []
        for s_test in top_scenes:
            test_mask = scene_ids == s_test
            train_mask = ~test_mask
            X_tr, y_tr = h[train_mask], gps[train_mask]
            X_te, y_te = h[test_mask], gps[test_mask]
            ridge = Ridge(alpha=10.0).fit(X_tr, y_tr)
            r = r2_score(y_te, ridge.predict(X_te), multioutput="uniform_average")
            loso_r2.append(float(r))

        loso_r2 = np.array(loso_r2)
        cond_results[cond] = {
            "label": label,
            "n_scenes": int(len(loso_r2)),
            "r2_mean": float(np.mean(loso_r2)),
            "r2_std": float(np.std(loso_r2)),
            "r2_median": float(np.median(loso_r2)),
            "r2_p25": float(np.percentile(loso_r2, 25)),
            "r2_p75": float(np.percentile(loso_r2, 75)),
            "r2_min": float(np.min(loso_r2)),
            "r2_max": float(np.max(loso_r2)),
            "fraction_negative": float(np.mean(loso_r2 < 0)),
            "fraction_below_05": float(np.mean(loso_r2 < 0.5)),
        }
        print(f"  R² across {len(loso_r2)} scenes: {np.mean(loso_r2):+.3f} ± {np.std(loso_r2):.3f}")
        print(f"     median={np.median(loso_r2):+.3f}, [{np.min(loso_r2):+.2f}, {np.max(loso_r2):+.2f}]")
        print(f"     fraction R²<0: {cond_results[cond]['fraction_negative']:.0%}, "
              f"fraction R²<0.5: {cond_results[cond]['fraction_below_05']:.0%}")
        results[cond] = (loso_r2, color, label)

    # Box plot ordering
    cond_order = ["blind", "coarse", "foveated", "uniform"]
    box_data = []
    box_colors = []
    box_labels = []
    for c in cond_order:
        if c in results:
            r2s, color, label = results[c]
            # Clip negative for viz; record the unclipped distribution
            box_data.append(np.clip(r2s, -1.5, 1.0))
            box_colors.append(color)
            box_labels.append(label)

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="black",
                                                    markeredgecolor="white", markersize=7),
                    medianprops=dict(color="black", linewidth=1.5),
                    widths=0.55)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.7)
    ax.axhline(1, ls=":", color="grey", alpha=0.4, lw=0.7)
    ax.text(4.4, 1.0, "perfect", fontsize=8, color="grey", va="center", ha="left")
    ax.text(4.4, 0.0, "no info", fontsize=8, color="grey", va="center", ha="left")

    ax.set_ylabel("LOSO test $R^2$ per scene\n(probe trained on all OTHER scenes)",
                  fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Encoder bandwidth (low → high)", fontsize=11.5, fontweight="bold")
    ax.set_title("Leave-one-scene-out cross-validation\nBlind = scene-invariant linear readout; rich-encoder = scene-specific (linear fails)",
                 fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_ylim(-1.6, 1.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    plt.tight_layout()
    out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_loso_cv.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")

    Path("/tmp/extra_analyses/loso_cv.json").write_text(json.dumps(cond_results, indent=2))
    print("wrote /tmp/extra_analyses/loso_cv.json")


if __name__ == "__main__":
    main()
