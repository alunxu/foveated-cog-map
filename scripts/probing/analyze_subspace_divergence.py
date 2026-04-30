"""
B2: Subspace divergence analysis (Finding 2 quantitative).

Tests the capacity-allocation principle's prediction that conditions
allocating capacity differently between routes produce hidden states in
non-interchangeable subspaces. Computes:

  (1) Per-condition top-K PCA basis on h_2 (covers ~90% variance).
  (2) Pairwise principal angles between cond_A's top-K subspace and
      cond_B's top-K subspace. Large mean angle ≈ disjoint subspaces.
  (3) Per-condition position-encoding direction: Ridge probe weight on
      GPS, then take dominant 2 directions (one per coord).
  (4) Pairwise cosine similarity between cond_A's position direction
      and cond_B's position direction. Small cos ≈ position is encoded
      in different directions in different conditions.
  (5) Per-condition fraction of total variance allocated to position
      direction: Var(beta @ h) / Var(h_total). High ratio = lots of
      capacity dedicated to position; low ratio = position info is a
      small slice of capacity.

Reads:  /tmp/cond_npzs/{blind,matched,foveated,uniform}_gibson_det.npz
Writes: docs/manuscript/fig/fig_subspace_divergence.pdf
        /tmp/subspace_divergence_results.json (numeric outputs)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "paper_figures"))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


CONDS = [
    ("blind",    "blind_gibson_det.npz",    "#444444"),
    ("coarse",   "matched_gibson_det.npz",  "#377eb8"),
    ("foveated", "foveated_gibson_det.npz", "#e41a1c"),
    ("uniform",  "uniform_gibson_det.npz",  "#4daf4a"),
]


def load_cond(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load h_2 (top-layer hidden state) + GPS labels.

    NPZ key conventions from collect.py:
      - 'rnn_hidden_states' or 'h2' : (n_steps, hidden_dim)
      - 'gps' or 'position' : (n_steps, 2) -- (x, z) world-frame
    """
    d = np.load(npz_path, allow_pickle=False)
    keys = list(d.keys())
    print(f"  NPZ keys: {keys[:10]}{'...' if len(keys)>10 else ''}")
    # Try common naming conventions
    h_keys = ["hidden_states", "h2", "rnn_hidden_states_l2", "rnn_h_l2", "rnn_h", "rnn_hidden_states"]
    gps_keys = ["gps", "position", "positions", "world_pos", "agent_pos", "xyz"]
    h = None
    gps = None
    for k in h_keys:
        if k in d.files:
            h = d[k]
            print(f"    found h={k}: shape={h.shape}")
            break
    for k in gps_keys:
        if k in d.files:
            gps = d[k]
            print(f"    found gps={k}: shape={gps.shape}")
            break
    if h is None:
        raise ValueError(f"Could not find h key in {npz_path}; keys={keys}")
    if gps is None:
        raise ValueError(f"Could not find gps key in {npz_path}; keys={keys}")
    # If h is 3D (n_layers, n_steps, hidden), take top layer
    if h.ndim == 3:
        h = h[-1]  # top layer
        print(f"    took top layer h_2: shape={h.shape}")
    # GPS world-frame x and z (drop y/height)
    if gps.shape[1] == 3:
        gps = gps[:, [0, 2]]  # x, z
    return h, gps


def principal_angles(B_A: np.ndarray, B_B: np.ndarray) -> np.ndarray:
    """Principal angles between subspaces spanned by columns of B_A and B_B.

    Both inputs (D, K_A) and (D, K_B). Returns angles in radians.
    Algorithm: SVD of B_A^T @ B_B. Singular values are cos(angle).
    """
    Q_A, _ = np.linalg.qr(B_A)
    Q_B, _ = np.linalg.qr(B_B)
    M = Q_A.T @ Q_B
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("/tmp/cond_npzs"))
    ap.add_argument("--out-fig", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=Path("/tmp/subspace_divergence_results.json"))
    ap.add_argument("--variance-threshold", type=float, default=0.90,
                    help="cumulative variance for top-K PCA cutoff")
    ap.add_argument("--max-samples", type=int, default=20000,
                    help="subsample for speed")
    args = ap.parse_args()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)

    # Load + standardize
    h_data: dict[str, np.ndarray] = {}
    gps_data: dict[str, np.ndarray] = {}
    for name, fname, _ in CONDS:
        p = args.in_dir / fname
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        print(f"Loading {name}...")
        h, gps = load_cond(p)
        # Subsample for speed
        if len(h) > args.max_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(h), args.max_samples, replace=False)
            h = h[idx]
            gps = gps[idx]
        # Mean-center within cond (subspace is invariant to mean)
        h_data[name] = h - h.mean(axis=0, keepdims=True)
        gps_data[name] = gps - gps.mean(axis=0, keepdims=True)

    cond_names = list(h_data.keys())
    n_conds = len(cond_names)

    # 1. Per-cond PCA basis (top-K covering threshold variance)
    pca_bases: dict[str, np.ndarray] = {}
    K_per_cond: dict[str, int] = {}
    for name in cond_names:
        pca = PCA(n_components=min(50, h_data[name].shape[1]))
        pca.fit(h_data[name])
        cum = np.cumsum(pca.explained_variance_ratio_)
        K = int(np.searchsorted(cum, args.variance_threshold) + 1)
        K = min(K, 50)
        pca_bases[name] = pca.components_[:K].T  # (D, K)
        K_per_cond[name] = K
        print(f"  {name}: top-{K} PCA covers {cum[K-1]*100:.1f}% variance")

    # 2. Pairwise principal angles
    angle_matrix = np.zeros((n_conds, n_conds))
    for i, A in enumerate(cond_names):
        for j, B in enumerate(cond_names):
            if i == j:
                angle_matrix[i, j] = 0.0
                continue
            angles = principal_angles(pca_bases[A], pca_bases[B])
            # Mean of all principal angles in radians; convert to degrees
            angle_matrix[i, j] = np.degrees(np.mean(angles))

    # 3. Per-cond position-encoding direction (Ridge probe on GPS)
    pos_dirs: dict[str, np.ndarray] = {}
    pos_var_ratio: dict[str, float] = {}
    for name in cond_names:
        h, gps = h_data[name], gps_data[name]
        ridge = Ridge(alpha=10.0)
        ridge.fit(h, gps)
        # ridge.coef_: (n_targets=2, n_features=512)
        beta = ridge.coef_  # (2, 512)
        pos_dirs[name] = beta  # store both x and z directions
        # Variance projected onto position dirs
        proj = h @ beta.T  # (n, 2)
        var_pos = np.var(proj, axis=0).sum()
        var_total = np.trace(h.T @ h) / len(h)
        pos_var_ratio[name] = float(var_pos / var_total)
        print(f"  {name}: position-encoding variance ratio = {pos_var_ratio[name]:.4f}")

    # 4. Pairwise cosine between position-encoding directions
    pos_cos_matrix = np.zeros((n_conds, n_conds))
    for i, A in enumerate(cond_names):
        for j, B in enumerate(cond_names):
            beta_A = pos_dirs[A]  # (2, 512)
            beta_B = pos_dirs[B]  # (2, 512)
            # Use total cosine: (||beta_A|| dot ||beta_B||) / (norms)
            # via averaging the two coord directions' cosines
            cos_x = float(np.dot(beta_A[0], beta_B[0]) /
                          (np.linalg.norm(beta_A[0]) * np.linalg.norm(beta_B[0]) + 1e-9))
            cos_z = float(np.dot(beta_A[1], beta_B[1]) /
                          (np.linalg.norm(beta_A[1]) * np.linalg.norm(beta_B[1]) + 1e-9))
            pos_cos_matrix[i, j] = 0.5 * (cos_x + cos_z)

    # Save numeric results
    results = {
        "conditions": cond_names,
        "K_per_cond": K_per_cond,
        "principal_angle_matrix_deg": angle_matrix.tolist(),
        "pos_dir_cos_matrix": pos_cos_matrix.tolist(),
        "pos_var_ratio": pos_var_ratio,
        "variance_threshold": args.variance_threshold,
        "max_samples": args.max_samples,
    }
    args.out_json.write_text(json.dumps(results, indent=2))
    print(f"\nNumeric results saved to {args.out_json}")

    # Generate figure: 2 panels (principal angles + position-direction cos)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5),
                             gridspec_kw={"wspace": 0.32})

    # Panel A: Principal angle matrix (heatmap)
    ax = axes[0]
    im = ax.imshow(angle_matrix, cmap="RdYlGn_r", vmin=0, vmax=90)
    ax.set_xticks(range(n_conds))
    ax.set_yticks(range(n_conds))
    ax.set_xticklabels([c.capitalize() for c in cond_names], fontsize=11)
    ax.set_yticklabels([c.capitalize() for c in cond_names], fontsize=11)
    for i in range(n_conds):
        for j in range(n_conds):
            v = angle_matrix[i, j]
            color = "white" if v > 45 else "black"
            ax.text(j, i, f"{v:.0f}°", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    ax.set_title("(a) Principal angles between top-${K}$ subspaces",
                 fontsize=11.5, loc="left", pad=8, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="angle (deg)")

    # Panel B: Position-encoding direction cosines
    ax = axes[1]
    im = ax.imshow(pos_cos_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n_conds))
    ax.set_yticks(range(n_conds))
    ax.set_xticklabels([c.capitalize() for c in cond_names], fontsize=11)
    ax.set_yticklabels([c.capitalize() for c in cond_names], fontsize=11)
    for i in range(n_conds):
        for j in range(n_conds):
            v = pos_cos_matrix[i, j]
            color = "black" if abs(v) < 0.5 else "white"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    ax.set_title("(b) Position-encoding direction alignment",
                 fontsize=11.5, loc="left", pad=8, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cos$(\\beta_A,\\beta_B)$")

    fig.suptitle("Subspace divergence (H2): conditions allocate capacity to mutually orthogonal subspaces",
                 fontsize=12.5, fontweight="bold", y=1.04)

    plt.tight_layout()
    fig.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
