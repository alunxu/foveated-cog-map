"""
TwoNN intrinsic dimensionality estimator (Facco et al. 2017).

For each cond's h_2 representation, estimate the intrinsic dimensionality
of the manifold using the two-nearest-neighbour distance ratio method.

ID = 1 / mean(log(r_2 / r_1)) over a sample of points,
where r_1, r_2 are the distances to the 1st and 2nd nearest neighbours.

Reads:  /tmp/cond_npzs/{cond}_gibson_det.npz
        /tmp/cond_npzs_seed2/{foveated,uniform}_seed2_det.npz
Writes: /tmp/extra_analyses/twonn_results.json
        docs/manuscript/fig/fig_intrinsic_dim.pdf
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
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Project/scripts/paper_figures"))
try:
    from _style import apply_paper_style
    apply_paper_style()
except Exception:
    pass


def two_nn_id(X: np.ndarray, n_max: int = 5000, fraction: float = 0.9,
              seed: int = 0) -> tuple[float, float]:
    """
    Two-NN intrinsic dimensionality estimator.

    Returns (estimate, std).  fraction=0.9 trims top 10% of mu
    values to mitigate noise/outliers (per Facco recommendation).
    """
    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        idx = rng.choice(len(X), n_max, replace=False)
        X = X[idx]
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)  # 3 to allow self + 2 NN
    distances, _ = nbrs.kneighbors(X)
    # Exclude self (distance 0)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    # Filter out points with degenerate distances
    mask = (r1 > 1e-9) & (r2 > r1)
    r1, r2 = r1[mask], r2[mask]
    mu = r2 / r1
    # Sort and trim top fraction
    mu = np.sort(mu)
    n_keep = int(fraction * len(mu))
    mu = mu[:n_keep]
    # Linear fit: -log(1 - F(mu)) = d * log(mu)  where F = empirical CDF
    F = np.arange(1, len(mu) + 1) / len(mu)
    # Avoid log(0)
    valid = F < 1.0
    log_mu = np.log(mu[valid])
    log_one_minus_F = -np.log(1.0 - F[valid])
    # Linear regression through origin
    d_est = float(np.sum(log_mu * log_one_minus_F) / np.sum(log_mu ** 2))
    # Bootstrap std estimate
    boot = []
    for b in range(20):
        idx_b = rng.choice(len(mu), len(mu), replace=True)
        mu_b = mu[idx_b]
        mu_b_sorted = np.sort(mu_b)
        F_b = np.arange(1, len(mu_b_sorted) + 1) / len(mu_b_sorted)
        valid_b = F_b < 1.0
        log_mu_b = np.log(mu_b_sorted[valid_b])
        log_F_b = -np.log(1.0 - F_b[valid_b])
        d_b = float(np.sum(log_mu_b * log_F_b) / np.sum(log_mu_b ** 2))
        boot.append(d_b)
    return d_est, float(np.std(boot))


def main():
    ambient_dim = 512  # h_2 dimensionality
    inputs = {
        "blind":      ("/tmp/cond_npzs/blind_gibson_det.npz",       0,  "#444444", "o"),
        "coarse":     ("/tmp/cond_npzs/matched_gibson_det.npz",     1,  "#377eb8", "s"),
        "foveated":   ("/tmp/cond_npzs/foveated_gibson_det.npz",    64, "#e41a1c", "D"),
        "uniform":    ("/tmp/cond_npzs/uniform_gibson_det.npz",     64, "#4daf4a", "^"),
        # multi-seed
        "foveated_s2":("/tmp/cond_npzs_seed2/foveated_seed2_det.npz",  64, "#e41a1c", "D"),
        "uniform_s2": ("/tmp/cond_npzs_seed2/uniform_seed2_det.npz",   64, "#4daf4a", "^"),
    }
    out = {}
    for name, (path, cells, color, marker) in inputs.items():
        p = Path(path)
        if not p.exists():
            print(f"SKIP {name}: {p} missing"); continue
        d = np.load(p)
        h = d["hidden_states"].astype(np.float32)
        # Mean-center
        h = h - h.mean(axis=0, keepdims=True)
        # Subsample for speed
        n_samples = min(20000, len(h))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(h), n_samples, replace=False)
        h = h[idx]
        d_est, d_std = two_nn_id(h, n_max=5000)
        out[name] = {
            "id_mean": d_est, "id_std": d_std,
            "encoder_cells": cells, "n_samples": n_samples,
            "ambient_dim": int(h.shape[1]),
        }
        print(f"{name:20s}  ID = {d_est:5.1f} ± {d_std:.1f}  (cells={cells})")

    Path("/tmp/extra_analyses").mkdir(exist_ok=True)
    Path("/tmp/extra_analyses/twonn_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote /tmp/extra_analyses/twonn_results.json")
    return out


if __name__ == "__main__":
    main()
