"""
G1: Eigenspectrum power-law fit per condition.

Following Stringer et al. 2019 (Nature, doi:10.1038/s41586-019-1346-5):
- The eigenspectrum of stimulus-driven population responses follows a
  power law λ_n ∝ n^{-α}.
- Theoretical claim: α = 1 + 2/d, where d is intrinsic stimulus dim.
- α just above (1 + 2/d) is "as compact as possible while still
  differentiable" — population code is at the smoothness/efficiency
  border.

For our PointGoal task, intrinsic stimulus dim ≈ 3 (2D position + 1D
heading), so theoretical α ≈ 1 + 2/3 ≈ 1.67.

For deterministic-rollout NPZs (no trial-to-trial noise), simple PCA on
mean-centred h_2 is sufficient — no cvPCA needed.

Reads:  /tmp/cond_npzs/{cond}_gibson_det.npz
Writes: /tmp/extra_analyses/eigenspectrum_powerlaw.json
        docs/manuscript/fig/fig_eigenspectrum.pdf
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
from scipy.stats import linregress

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


def fit_powerlaw(eigenvalues: np.ndarray, n_min: int = 5,
                 n_max: int = 200) -> tuple[float, float, float, float]:
    """
    Linear fit in log-log: log λ_n = -α log n + const.
    Returns (α, std_α, R², n_points_used).

    Skips the head (top 5 PCs) where finite-sample bias is largest, and
    the tail (PC > 200) where eigenvalues approach machine precision.
    Stringer uses similar [10, 500] bounds.
    """
    n = np.arange(1, len(eigenvalues) + 1)
    mask = (n >= n_min) & (n <= n_max) & (eigenvalues > 0)
    log_n = np.log(n[mask])
    log_lambda = np.log(eigenvalues[mask])
    slope, _, r, _, std_err = linregress(log_n, log_lambda)
    alpha = -slope  # we report positive exponent
    return float(alpha), float(std_err), float(r ** 2), int(mask.sum())


def main():
    Path("/tmp/extra_analyses").mkdir(exist_ok=True)
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                             gridspec_kw={"wspace": 0.30})

    cond_eig = {}
    for cond, path, label, color in CONDS:
        d = np.load(Path(path))
        h = d["hidden_states"].astype(np.float32)
        # subsample for speed (50k samples plenty for 512-d covariance)
        if len(h) > 50000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(h), 50000, replace=False)
            h = h[idx]
        h = h - h.mean(axis=0, keepdims=True)
        # Eigenspectrum via SVD of (n × d) → d singular values
        # For mean-centered X (n, d): cov C = X^T X / n,
        # eigenvalues of C are (s_i^2 / n) where s_i are singular values
        s = np.linalg.svd(h, compute_uv=False)
        eigvals = (s ** 2) / len(h)
        # Sort descending (already from SVD)
        eigvals = np.sort(eigvals)[::-1]
        # Trim to nonzero (last few may be at machine precision)
        eigvals = eigvals[eigvals > 1e-12]
        cond_eig[cond] = (eigvals, color, label)

        # Power-law fit
        alpha, alpha_std, r2, n_used = fit_powerlaw(eigvals)
        # Also try with wider range for robustness
        alpha_full, _, r2_full, _ = fit_powerlaw(eigvals, n_min=3, n_max=300)

        results[cond] = {
            "label": label,
            "alpha": alpha,
            "alpha_std": alpha_std,
            "r2": r2,
            "n_pcs_fit": n_used,
            "alpha_full_range": alpha_full,
            "r2_full": r2_full,
            "eigvals_top10": eigvals[:10].tolist(),
            "n_eigvals_total": int(len(eigvals)),
            "var_explained_top10": float(eigvals[:10].sum() / eigvals.sum()),
            "var_explained_top30": float(eigvals[:30].sum() / eigvals.sum()),
        }
        print(f"  {label:10}  α = {alpha:.3f} ± {alpha_std:.3f}  "
              f"R² = {r2:.3f}  n_pcs={n_used}  "
              f"var_top10={results[cond]['var_explained_top10']:.2%}")

    # Panel A: overlaid eigenspectra (log-log)
    for cond_key, cond_name in [(c[0], c[2]) for c in CONDS]:
        eigvals, color, label = cond_eig[cond_key]
        n = np.arange(1, len(eigvals) + 1)
        axes[0].loglog(n, eigvals, color=color, lw=2.0, alpha=0.85, label=label)
    # Overlay power-law reference α=1 (Stringer V1) and α=1.67 (theoretical for d=3)
    n_ref = np.arange(1, 200)
    eig_ref_1 = eigvals[5] * (n_ref / 5) ** (-1.0)
    eig_ref_167 = eigvals[5] * (n_ref / 5) ** (-1.67)
    axes[0].loglog(n_ref, eig_ref_1, ls=":", color="grey", lw=1.0, alpha=0.6)
    axes[0].text(150, eig_ref_1[-50], r"$\alpha{=}1$", color="grey",
                 fontsize=9, ha="left", va="center")
    axes[0].loglog(n_ref, eig_ref_167, ls="--", color="grey", lw=1.0, alpha=0.6)
    axes[0].text(150, eig_ref_167[-50], r"$\alpha{=}1{+}2/3$", color="grey",
                 fontsize=9, ha="left", va="center")
    axes[0].set_xlabel("PC index $n$", fontsize=11.5, fontweight="bold")
    axes[0].set_ylabel("Eigenvalue $\\lambda_n$ (log)", fontsize=11.5, fontweight="bold")
    axes[0].set_title("(a) Eigenspectrum of $\\mathbf{h}_2$ per cond",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9.5)
    axes[0].grid(linestyle=":", alpha=0.3)
    for s in ("top", "right"): axes[0].spines[s].set_visible(False)

    # Panel B: fitted α per cond
    cond_order = ["blind", "coarse", "foveated", "uniform"]
    cols = [next(c[3] for c in CONDS if c[0] == k) for k in cond_order]
    labs = [next(c[2] for c in CONDS if c[0] == k) for k in cond_order]
    alphas = [results[k]["alpha"] for k in cond_order]
    alpha_errs = [results[k]["alpha_std"] for k in cond_order]
    axes[1].bar(labs, alphas, yerr=alpha_errs, color=cols, alpha=0.7,
                edgecolor="black", linewidth=0.8, capsize=5)
    for i, (a, e) in enumerate(zip(alphas, alpha_errs)):
        axes[1].text(i, a + e + 0.02, f"{a:.2f}", ha="center",
                     fontsize=10, fontweight="bold")
    axes[1].axhline(1.0, ls=":", color="grey", alpha=0.6)
    axes[1].text(3.5, 1.02, "Stringer V1 ≈1.0", fontsize=8, color="grey", ha="right")
    axes[1].axhline(1 + 2/3, ls="--", color="grey", alpha=0.6)
    axes[1].text(3.5, 1.69, "$1{+}2/d_{\\rm task}$ (d=3)", fontsize=8, color="grey", ha="right")
    axes[1].set_ylabel("Power-law exponent $\\alpha$\n(fitted on PCs $n \\in [5, 200]$)",
                       fontsize=11.5, fontweight="bold")
    axes[1].set_title("(b) Fitted $\\alpha$ per cond",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    axes[1].set_ylim(0, max(alphas) + 0.4)
    for s in ("top", "right"): axes[1].spines[s].set_visible(False)
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Eigenspectrum power-law of $\\mathbf{h}_2$: smoothness/efficiency border per cond (Stringer et al. 2019 framework)",
                 fontsize=11, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_eigenspectrum.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")

    Path("/tmp/extra_analyses/eigenspectrum_powerlaw.json").write_text(
        json.dumps(results, indent=2))
    print("wrote /tmp/extra_analyses/eigenspectrum_powerlaw.json")


if __name__ == "__main__":
    main()
