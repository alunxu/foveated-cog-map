"""
Per-scene β stability test — decisive CAP experiment.

Hypothesis: encoder bandwidth → cross-scene β stability → linear probability.
  - Blind: β stable across scenes → cross-scene cosines HIGH → linear probe generalizes
  - Uniform: β unstable across scenes → cross-scene cosines LOW → linear probe fails

Method:
  For each cond, for each scene with >= n_min samples:
    1. Mean-center h within scene (remove scene-specific offset)
    2. Train Ridge(α=10) on (h, gps) within scene → β_scene (shape 2×512)
    3. L2-normalize β_scene rows
  Then compute pairwise cosine similarities between β's across scenes,
  separately for x and z output rows, and report mean ± std per cond.

Reads:  /tmp/cond_npzs/{cond}_gibson_det.npz
Writes: /tmp/extra_analyses/per_scene_beta_stability.json
        docs/manuscript/fig/fig_beta_stability.pdf
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
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
N_MIN = 200  # min samples per scene to fit Ridge stably
N_MAX_PER_SCENE = 1000  # cap per scene for speed


def per_scene_beta(h: np.ndarray, gps: np.ndarray, scene_ids: np.ndarray,
                   n_min: int = N_MIN, n_max: int = N_MAX_PER_SCENE,
                   alpha: float = 10.0) -> dict[int, np.ndarray]:
    """Fit Ridge per scene, return {scene_id: β (2, 512)}."""
    rng = np.random.default_rng(0)
    out = {}
    for scn in np.unique(scene_ids):
        mask = scene_ids == scn
        if mask.sum() < n_min:
            continue
        h_s = h[mask]
        gps_s = gps[mask]
        if len(h_s) > n_max:
            idx = rng.choice(len(h_s), n_max, replace=False)
            h_s = h_s[idx]; gps_s = gps_s[idx]
        # Mean-center within scene
        h_s = h_s - h_s.mean(axis=0, keepdims=True)
        gps_s = gps_s - gps_s.mean(axis=0, keepdims=True)
        ridge = Ridge(alpha=alpha).fit(h_s, gps_s)
        out[int(scn)] = ridge.coef_.astype(np.float32)  # (2, 512)
    return out


def cross_scene_cosines(betas: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Pairwise cosines between β rows across scenes."""
    scenes = sorted(betas.keys())
    n = len(scenes)
    # Stack: shape (n, 2, 512)
    B = np.stack([betas[s] for s in scenes], axis=0)
    # Per-row L2 normalize
    B_norm = B / (np.linalg.norm(B, axis=2, keepdims=True) + 1e-9)
    # Cosine matrices
    cos_x = B_norm[:, 0, :] @ B_norm[:, 0, :].T  # (n, n)
    cos_z = B_norm[:, 1, :] @ B_norm[:, 1, :].T  # (n, n)
    # Off-diagonal (i != j)
    iu = np.triu_indices(n, k=1)
    return {
        "scenes": np.array(scenes),
        "cos_x_offdiag": cos_x[iu],
        "cos_z_offdiag": cos_z[iu],
        "n_scenes": n,
    }


def main():
    results = {}
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.6))

    cond_data = {}
    for cond, path, label, color in CONDS:
        print(f"\n=== {cond} ===")
        d = np.load(path)
        h = d["hidden_states"].astype(np.float32)
        gps = d["gps"].astype(np.float32)
        scene_ids = d["scene_ids"]
        print(f"  shape: h={h.shape}, scenes={len(np.unique(scene_ids))}")

        betas = per_scene_beta(h, gps, scene_ids)
        print(f"  fitted β for {len(betas)} scenes (with ≥{N_MIN} samples)")

        if len(betas) < 5:
            print(f"  SKIP {cond}: too few scenes with enough samples")
            continue

        cos_data = cross_scene_cosines(betas)
        # Combined cosine = mean of x and z cosines (geometrically, average alignment)
        cos_combined = 0.5 * (cos_data["cos_x_offdiag"] + cos_data["cos_z_offdiag"])

        results[cond] = {
            "label": label,
            "n_scenes": int(cos_data["n_scenes"]),
            "cos_x_mean": float(np.mean(cos_data["cos_x_offdiag"])),
            "cos_x_std": float(np.std(cos_data["cos_x_offdiag"])),
            "cos_z_mean": float(np.mean(cos_data["cos_z_offdiag"])),
            "cos_z_std": float(np.std(cos_data["cos_z_offdiag"])),
            "cos_combined_mean": float(np.mean(cos_combined)),
            "cos_combined_std": float(np.std(cos_combined)),
            "cos_combined_p25": float(np.percentile(cos_combined, 25)),
            "cos_combined_p50": float(np.percentile(cos_combined, 50)),
            "cos_combined_p75": float(np.percentile(cos_combined, 75)),
        }
        cond_data[cond] = (cos_combined, color, label)
        print(f"  cosines x: {results[cond]['cos_x_mean']:+.3f} ± {results[cond]['cos_x_std']:.3f}")
        print(f"  cosines z: {results[cond]['cos_z_mean']:+.3f} ± {results[cond]['cos_z_std']:.3f}")
        print(f"  combined:   {results[cond]['cos_combined_mean']:+.3f} ± {results[cond]['cos_combined_std']:.3f}")

    # Box plot
    cond_order = ["blind", "coarse", "foveated", "uniform"]
    box_data = []
    box_colors = []
    box_labels = []
    for c in cond_order:
        if c in cond_data:
            cos, color, label = cond_data[c]
            box_data.append(cos)
            box_colors.append(color)
            box_labels.append(label)

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="black",
                                                    markeredgecolor="white", markersize=7),
                    medianprops=dict(color="black", linewidth=1.5),
                    widths=0.55)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.7)
    ax.axhline(1, ls=":", color="grey", alpha=0.4, lw=0.7)
    ax.text(4.45, 1.0, "perfectly aligned", fontsize=8, color="grey", va="center", ha="left")
    ax.text(4.45, 0.0, "orthogonal", fontsize=8, color="grey", va="center", ha="left")

    ax.set_ylabel("Cosine($\\beta_{\\rm scene_a},\\beta_{\\rm scene_b}$)\nover all scene pairs",
                  fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Encoder bandwidth (low → high)", fontsize=11.5, fontweight="bold")
    ax.set_title("Per-scene position-direction stability across conditions\n(higher = same readout works across scenes; lower = scene-specific encoding)",
                 fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_ylim(-0.5, 1.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    plt.tight_layout()
    out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_beta_stability.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")

    Path("/tmp/extra_analyses/per_scene_beta_stability.json").write_text(json.dumps(results, indent=2))
    print("wrote /tmp/extra_analyses/per_scene_beta_stability.json")


if __name__ == "__main__":
    main()
