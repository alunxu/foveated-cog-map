"""
Decisive test: is position encoded in HIGH-variance PCs or LOW-variance PCs?

Method:
  1. Train Ridge probe on full h_2 → β (shape 2×512)
  2. Take SVD of β: β = U Σ V^T → V gives 2 "position-axes" in h-space
  3. Project each position-axis onto each PC of h: |⟨v_pos, PC_i⟩|²
  4. Plot cumulative |projection|² vs PC index
  5. Compare to cumulative explained-variance (also vs PC index)

Interpretation:
  - If position-axis cumulative SAT URATES fast WITH explained-variance →
    position lives in high-variance dirs (linear-aligned)
  - If position-axis cumulative SATURATES SLOWLY (way after explained-variance) →
    position lives in LOW-variance dirs (orthogonal to manifold's main spread)
  - If position-axis spreads evenly → not concentrated anywhere

This decisively shows WHERE in h-space position-info lives.
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
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
    ("blind",    "/tmp/cond_npzs/blind_gibson_det.npz",     "Blind",    "#444444", "-"),
    ("coarse",   "/tmp/cond_npzs/matched_gibson_det.npz",   "Coarse",   "#377eb8", "-"),
    ("foveated", "/tmp/cond_npzs/foveated_gibson_det.npz",  "Foveated", "#e41a1c", "-"),
    ("uniform",  "/tmp/cond_npzs/uniform_gibson_det.npz",   "Uniform",  "#4daf4a", "-"),
]

N_SAMPLES = 15000
RNG = np.random.default_rng(42)


def main():
    results = {}
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.6))

    for cond, path, label, color, ls in CONDS:
        d = np.load(Path(path))
        h = d["hidden_states"].astype(np.float32)
        gps = d["gps"].astype(np.float32)

        if len(h) > N_SAMPLES:
            idx = RNG.choice(len(h), N_SAMPLES, replace=False)
            h = h[idx]; gps = gps[idx]
        h = h - h.mean(axis=0, keepdims=True)
        gps = gps - gps.mean(axis=0, keepdims=True)

        # PCA basis for h
        pca = PCA(n_components=200)
        pca.fit(h)
        PC = pca.components_  # shape (200, 512), each row is a unit eigenvector
        explained_var_ratio = pca.explained_variance_ratio_

        # Train Ridge probe on full h
        ridge = Ridge(alpha=10.0).fit(h, gps)
        beta = ridge.coef_  # shape (2, 512)

        # SVD of beta: beta = U @ diag(S) @ V^T  where rows of V are h-space directions
        U, S, Vt = np.linalg.svd(beta, full_matrices=False)
        V = Vt  # V[i] = i-th singular direction in h-space (shape 512)
        # Take TOP 2 singular vectors (rank of β is at most 2 since output is 2D)

        # Combined position-direction power: ||proj of V_top2||² weighted by S²
        # i.e., for each PC_i, compute (S_0² * ⟨V_0, PC_i⟩² + S_1² * ⟨V_1, PC_i⟩²) / (S_0² + S_1²)
        proj_0 = (PC @ V[0]) ** 2  # shape (200,)
        proj_1 = (PC @ V[1]) ** 2  # shape (200,)
        weight_0 = S[0] ** 2 / (S[0] ** 2 + S[1] ** 2)
        weight_1 = S[1] ** 2 / (S[0] ** 2 + S[1] ** 2)
        pos_axis_proj = weight_0 * proj_0 + weight_1 * proj_1
        pos_axis_cumulative = np.cumsum(pos_axis_proj)

        # Cumulative explained variance (for reference: how much of MANIFOLD lives in top-k)
        cum_var = np.cumsum(explained_var_ratio)

        # Note: pos_axis_cumulative might not reach 1 because the position-axis lives in 512-d
        # and we only computed 200 PCs.

        # Where does pos-axis reach 50% / 90%?
        idx_50 = int(np.searchsorted(pos_axis_cumulative, 0.5))
        idx_90 = int(np.searchsorted(pos_axis_cumulative, 0.9))
        idx_var_50 = int(np.searchsorted(cum_var, 0.5))
        idx_var_90 = int(np.searchsorted(cum_var, 0.9))

        results[cond] = {
            "label": label,
            "pos_axis_50pct_pc": idx_50,
            "pos_axis_90pct_pc": idx_90,
            "explained_var_50pct_pc": idx_var_50,
            "explained_var_90pct_pc": idx_var_90,
            "ridge_full_r2": float(r2_score(gps, ridge.predict(h),
                                            multioutput="uniform_average")),
        }
        print(f"{label:10s}  ridge full R²={results[cond]['ridge_full_r2']:+.3f}  "
              f"pos-axis 50%@PC{idx_50}/90%@PC{idx_90}  "
              f"vs explained-var 50%@PC{idx_var_50}/90%@PC{idx_var_90}")

        # Plot cumulative position-axis projection
        ax.plot(np.arange(1, 201), pos_axis_cumulative, color=color, lw=2.0,
                label=f"{label} pos-axis", linestyle=ls)
        # Plot cumulative explained variance for reference (faded)
        ax.plot(np.arange(1, 201), cum_var, color=color, lw=1.0, ls="--",
                alpha=0.4)

    ax.set_xlabel("Cumulative number of PCs", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Cumulative power", fontsize=11.5, fontweight="bold")
    ax.set_title("Position-axis projection onto PCs (solid) vs explained variance (dashed)\n"
                 "Position lives in DIFFERENT PCs than the manifold's main spread",
                 fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, ls=":", color="grey", alpha=0.4, lw=0.8)
    ax.axhline(0.5, ls=":", color="grey", alpha=0.4, lw=0.8)
    ax.text(199, 0.91, "90%", fontsize=8, color="grey", ha="right")
    ax.text(199, 0.51, "50%", fontsize=8, color="grey", ha="right")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)

    plt.tight_layout()
    out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_position_axis.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")

    Path("/tmp/extra_analyses/position_axis.json").write_text(json.dumps(results, indent=2))
    print("wrote /tmp/extra_analyses/position_axis.json")


if __name__ == "__main__":
    main()
