"""
Deep-dive into intrinsic dim finding.

Q: Why do all conds have ID ≈ 3 but linear-probe R² differs?

Hypothesis: Manifold local dim is ~ task dim (3D), shared across conds.
What differs is HOW position is embedded within the manifold:
  - Blind: position ~ linear axis (1 direction in PC space)
  - Uniform: position ~ curved path (spread across many PCs)

Test 1: PCA-cumulative GPS R² — for top-k PCs (k=1..30), what's the
        Ridge probe R² using only those PCs?
  Predict: Blind saturates fast (k≤3); uniform rises slowly across many k.

Test 2: Participation ratio PR = (Σλ)² / Σλ² — # of "active" dims.
        High PR = variance spread across many dims.
  Predict: Blind low PR (compact); Uniform high PR (spread).

Test 3: Conditional variance ratio — Var(h | (x,z) bin) / Var(h).
        Low = clean position code; High = distributed.
  Predict: Blind low (each pos = tight cluster); Uniform high.

Reads:  /tmp/cond_npzs/{cond}_gibson_det.npz
Writes: /tmp/extra_analyses/pc_cumulative.json
        docs/manuscript/fig/fig_pc_cumulative.pdf
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
    ("blind",    "/tmp/cond_npzs/blind_gibson_det.npz",     "Blind",    "#444444", "o"),
    ("coarse",   "/tmp/cond_npzs/matched_gibson_det.npz",   "Coarse",   "#377eb8", "s"),
    ("foveated", "/tmp/cond_npzs/foveated_gibson_det.npz",  "Foveated", "#e41a1c", "D"),
    ("uniform",  "/tmp/cond_npzs/uniform_gibson_det.npz",   "Uniform",  "#4daf4a", "^"),
]

K_MAX = 50
N_SAMPLES = 15000
RNG = np.random.default_rng(42)


def participation_ratio(eigvals: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ²"""
    return float((eigvals.sum()) ** 2 / (eigvals ** 2).sum())


def conditional_variance_ratio(h: np.ndarray, gps: np.ndarray, n_bins: int = 10) -> float:
    """Var(h | (x,z) bin) / Var(h) — averaged over bins."""
    x_bins = np.linspace(gps[:, 0].min(), gps[:, 0].max(), n_bins + 1)
    z_bins = np.linspace(gps[:, 1].min(), gps[:, 1].max(), n_bins + 1)
    x_idx = np.clip(np.searchsorted(x_bins, gps[:, 0]) - 1, 0, n_bins - 1)
    z_idx = np.clip(np.searchsorted(z_bins, gps[:, 1]) - 1, 0, n_bins - 1)
    bin_id = x_idx * n_bins + z_idx
    total_var = h.var(axis=0).sum()
    bin_vars = []
    for b in np.unique(bin_id):
        mask = bin_id == b
        if mask.sum() < 5:
            continue
        bin_vars.append(h[mask].var(axis=0).sum())
    avg_cond_var = float(np.mean(bin_vars))
    return avg_cond_var / total_var


def run():
    results = {}
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                     gridspec_kw={"wspace": 0.32})

    for cond, path, label, color, marker in CONDS:
        p = Path(path)
        if not p.exists():
            print(f"SKIP {cond}: missing"); continue
        d = np.load(p)
        h = d["hidden_states"].astype(np.float32)
        gps = d["gps"].astype(np.float32)
        ep_ids = d["episode_ids"]

        # Subsample
        if len(h) > N_SAMPLES:
            idx = RNG.choice(len(h), N_SAMPLES, replace=False)
            h = h[idx]; gps = gps[idx]; ep_ids = ep_ids[idx]

        h = h - h.mean(axis=0, keepdims=True)
        gps_centered = gps - gps.mean(axis=0, keepdims=True)

        # PCA
        pca = PCA(n_components=K_MAX)
        h_pca = pca.fit_transform(h)
        eigvals = pca.explained_variance_

        # Test 1: cumulative GPS R² as function of k PCs
        # Episode-level CV (3 folds, fast)
        unique_eps = np.unique(ep_ids)
        RNG_local = np.random.default_rng(0)
        RNG_local.shuffle(unique_eps)
        folds = np.array_split(unique_eps, 3)
        r2_curve = np.zeros(K_MAX)
        for k in range(1, K_MAX + 1):
            r2s = []
            for te_eps in folds:
                tr_mask = ~np.isin(ep_ids, te_eps)
                te_mask = np.isin(ep_ids, te_eps)
                X_tr = h_pca[tr_mask, :k]
                X_te = h_pca[te_mask, :k]
                y_tr = gps_centered[tr_mask]
                y_te = gps_centered[te_mask]
                ridge = Ridge(alpha=10.0).fit(X_tr, y_tr)
                r = r2_score(y_te, ridge.predict(X_te), multioutput="uniform_average")
                r2s.append(r)
            r2_curve[k - 1] = np.mean(r2s)

        # Test 2: PR
        pr = participation_ratio(eigvals)

        # Test 3: conditional variance ratio
        cv_ratio = conditional_variance_ratio(h, gps, n_bins=10)

        results[cond] = {
            "label": label,
            "r2_curve": r2_curve.tolist(),
            "participation_ratio": pr,
            "conditional_variance_ratio": cv_ratio,
            "explained_variance_top10": eigvals[:10].tolist(),
            "total_variance": float(eigvals.sum()),
        }

        # Plot panel A: cumulative R²
        ax_a.plot(np.arange(1, K_MAX + 1), r2_curve, marker=marker, markersize=5,
                  markevery=3, color=color, lw=1.5, label=f"{label}")

        # Plot panel B: PR + cv_ratio annotation
        ax_b.bar([cond], [pr], color=color, alpha=0.7, edgecolor="black",
                 linewidth=0.8, label=f"{label}: PR={pr:.1f}, CVR={cv_ratio:.2f}")

        print(f"{label:10s}  PR={pr:6.2f}  CVR={cv_ratio:.3f}  R²(k=1)={r2_curve[0]:+.3f}  "
              f"R²(k=3)={r2_curve[2]:+.3f}  R²(k=10)={r2_curve[9]:+.3f}  R²(k=30)={r2_curve[29]:+.3f}")

    # Panel A annotations
    ax_a.axhline(0, ls=":", color="grey", alpha=0.5)
    ax_a.set_xlabel("# PCA components used in linear probe", fontsize=11.5, fontweight="bold")
    ax_a.set_ylabel("$R^2$ (Ridge $\\alpha{=}10$, ep-CV)", fontsize=11.5, fontweight="bold")
    ax_a.set_title("(a) Linear readout vs # PCs used\n(Blind saturates fast → linear axis;\nUniform rises slowly → distributed)",
                   fontsize=11, loc="left", fontweight="bold", pad=8)
    ax_a.set_xlim(0, K_MAX + 1)
    ax_a.legend(loc="lower right", frameon=False, fontsize=10)
    ax_a.grid(linestyle=":", alpha=0.3)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)

    # Panel B annotations
    ax_b.set_ylabel("Participation Ratio\n$(\\Sigma\\lambda)^2 / \\Sigma\\lambda^2$",
                    fontsize=11.5, fontweight="bold")
    ax_b.set_title("(b) Variance spread\n(low = compact; high = distributed)",
                   fontsize=11, loc="left", fontweight="bold", pad=8)
    ax_b.set_xticklabels([c[2] for c in CONDS], fontsize=10)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Why ID${\\approx}3$ across conds but linear $R^2$ differs: position is embedded along DIFFERENT manifold directions",
                 fontsize=12, fontweight="bold", y=1.04)

    plt.tight_layout()
    out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_pc_cumulative.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    Path("/tmp/extra_analyses/pc_cumulative.json").write_text(json.dumps(results, indent=2, default=str))
    print("wrote /tmp/extra_analyses/pc_cumulative.json")


if __name__ == "__main__":
    run()
