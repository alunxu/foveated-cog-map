"""Generalized shape-metric distance matrix between condition-level h-state geometries.

Following Williams, Kunz, Kornblith, Linderman, "Generalized Shape Metrics on
Neural Representations", NeurIPS 2021. We use Proposition 1 + the rotation-
invariance metric d_1 (Procrustes size-and-shape distance, eq. 7) and the
angular Riemannian distance theta_1 on Kendall's shape space (eq. 8).

Key advantages over linear CKA:
  - d_1 / theta_1 are TRUE METRICS (satisfy triangle inequality); CKA does not.
  - Resilient to known CKA failure modes (Davari et al. 2024).
  - Yield a 5x5 distance MATRIX that we can hierarchically cluster, embed in
    Euclidean space (Williams Fig 1), or run permutation tests on.

Pairing protocol (the central design choice for this analyzer):
  Williams Proposition 1 requires PAIRED stimuli -- the m rows of X_i and X_j
  must correspond to the same "stimulus" across networks. Our 5 conditions are
  5 different agents with different rollout trajectories (different actions ->
  different (position, observation) tuples), so per-step h-states are NOT
  paired across conditions. We define stimulus-level pairing as follows:

    For each Gibson eval episode (deterministic seed; same scene + start +
    goal across all conditions), compute the MEAN top-layer h_2 across the
    rollout. Result: one 512-d vector per (condition, episode). Pair across
    conditions by episode_id. Final per-condition representation:
        X_c in R^{m x 512}, m = number of common episodes (~500).

  Per-episode mean is the standard "summary representation of how this agent
  solved this episode" choice. Alternative pairings (terminal h_T only;
  step-30 only; per-scene mean) are listed in CLI flags.

Reads:  --in-dir <dir>/{cond}_gibson_det.npz   (canonical converged rollout)
Writes: <out>.json with 5x5 d_1 + theta_1 matrices, plus paired-episode count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Display name (paper convention) -> NPZ filename stem (legacy training cond)
COND_NPZ_MAP = {
    "coarse": "matched",                    # legacy alias: matched -> coarse
    "foveated": "foveated",
    "uniform": "uniform",
    "foveated_logpolar": "foveated_logpolar",
    "blind_izar": "blind_izar",             # our seed=100 izar blind, ckpt.34
}
CONDS = list(COND_NPZ_MAP.keys())


def per_episode_mean_h(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, ep_ids) where X[i] is the mean top-layer h_2 across rollout
    of episode ep_ids[i]. Sorted by ep_id ascending so cross-condition pairing
    is deterministic."""
    d = np.load(npz_path)
    H = d["hidden_states"].astype(np.float64)
    ep_ids = d["episode_ids"]
    unique = np.unique(ep_ids)
    X = np.zeros((len(unique), H.shape[1]), dtype=np.float64)
    for i, ep in enumerate(unique):
        X[i] = H[ep_ids == ep].mean(axis=0)
    return X, unique


def per_episode_terminal_h(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Terminal h_2 (last step of each episode). Alternative pairing."""
    d = np.load(npz_path)
    H = d["hidden_states"].astype(np.float64)
    ep_ids = d["episode_ids"]
    step = d["step_in_episode"]
    unique = np.unique(ep_ids)
    X = np.zeros((len(unique), H.shape[1]), dtype=np.float64)
    for i, ep in enumerate(unique):
        em = ep_ids == ep
        last = np.argmax(step[em])
        X[i] = H[em][last]
    return X, unique


def procrustes_d1(X: np.ndarray, Y: np.ndarray) -> float:
    """Williams 2021 eq. 7: d_1(X, Y) = min_{Q in O} || phi_1(X) - phi_1(Y) Q ||
    where phi_1 mean-centers columns. Closed-form via SVD of phi_1(X)^T phi_1(Y).
    This is the Procrustes size-and-shape distance with reflections."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    M = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt  # optimal orthogonal alignment
    return float(np.linalg.norm(Xc - Yc @ Q.T, ord="fro"))


def procrustes_theta1(X: np.ndarray, Y: np.ndarray) -> float:
    """Williams 2021 eq. 8: theta_1(X, Y) = min_{Q in O} arccos
    <phi_1(X), phi_1(Y) Q> / (||phi_1(X)|| ||phi_1(Y)||).
    Riemannian angular distance on Kendall shape space.
    Returns a value in [0, pi/2]."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    nx = np.linalg.norm(Xc)
    ny = np.linalg.norm(Yc)
    if nx < 1e-12 or ny < 1e-12:
        return float("nan")
    M = Xc.T @ Yc
    # max <Xc, Yc Q> over Q orthogonal = sum of singular values of M
    s = np.linalg.svd(M, compute_uv=False)
    cos_theta = s.sum() / (nx * ny)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Reference: Kornblith 2019 unaligned linear CKA, for comparison with
    Williams metrics."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    den = (np.linalg.norm(Xc.T @ Xc, ord="fro")
           * np.linalg.norm(Yc.T @ Yc, ord="fro"))
    if den < 1e-12:
        return float("nan")
    return float(num / den)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suffix", default="_gibson_det")
    ap.add_argument(
        "--pairing", default="episode_mean",
        choices=["episode_mean", "episode_terminal"],
        help="How to extract one 512-d vector per (condition, episode).",
    )
    args = ap.parse_args()

    extractor = (per_episode_mean_h
                 if args.pairing == "episode_mean"
                 else per_episode_terminal_h)

    # Load per-condition representations + episode IDs
    reps: dict[str, np.ndarray] = {}
    eps: dict[str, np.ndarray] = {}
    for c in CONDS:
        npz_stem = COND_NPZ_MAP[c]
        npz = args.in_dir / f"{npz_stem}{args.suffix}.npz"
        if not npz.exists():
            print(f"  {c} (npz '{npz_stem}{args.suffix}.npz'): missing -- skip")
            continue
        X, ep_ids = extractor(npz)
        reps[c] = X
        eps[c] = ep_ids
        print(f"  {c}: {X.shape[0]} paired (episode-level) x {X.shape[1]}d "
              f"<- {npz_stem}{args.suffix}.npz")

    # Find common episode_ids across conditions (intersection)
    if len(reps) < 2:
        print("Need >= 2 conditions; abort.")
        return
    common = None
    for c, ep in eps.items():
        common = set(ep.tolist()) if common is None else common & set(ep.tolist())
    common = sorted(common)
    print(f"\nCommon episode_ids across all loaded conditions: {len(common)}")

    # Restrict each condition to common episodes (preserve order)
    common_set = set(common)
    aligned: dict[str, np.ndarray] = {}
    for c in reps:
        idx = np.array([i for i, e in enumerate(eps[c]) if e in common_set])
        # Sort by ep_id so rows correspond across conditions
        ep_subset = eps[c][idx]
        order = np.argsort(ep_subset)
        aligned[c] = reps[c][idx][order]
    m = next(iter(aligned.values())).shape[0]
    print(f"Aligned matrix per condition: {m} x 512\n")

    # 5x5 distance matrices
    cond_names = list(aligned.keys())
    n_c = len(cond_names)
    d1_mat = np.zeros((n_c, n_c))
    theta1_mat = np.zeros((n_c, n_c))
    cka_mat = np.zeros((n_c, n_c))
    for i, ci in enumerate(cond_names):
        for j, cj in enumerate(cond_names):
            if i <= j:
                d1_mat[i, j] = procrustes_d1(aligned[ci], aligned[cj])
                theta1_mat[i, j] = procrustes_theta1(aligned[ci], aligned[cj])
                cka_mat[i, j] = linear_cka(aligned[ci], aligned[cj])
            else:
                d1_mat[i, j] = d1_mat[j, i]
                theta1_mat[i, j] = theta1_mat[j, i]
                cka_mat[i, j] = cka_mat[j, i]

    # Triangle-inequality sanity check (true metrics should satisfy it; CKA shouldn't)
    def violates_triangle(M):
        n = M.shape[0]
        v = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if M[i, j] > M[i, k] + M[k, j] + 1e-9:
                        v += 1
        return v
    n_violate_d1 = violates_triangle(d1_mat)
    n_violate_theta1 = violates_triangle(theta1_mat)
    n_violate_cka_dist = violates_triangle(1 - cka_mat)  # 1-CKA as pseudo-distance
    print(f"Triangle-inequality violations:")
    print(f"  d_1 (Procrustes Euclidean): {n_violate_d1}  (expected 0 -- true metric)")
    print(f"  theta_1 (Kendall angular):  {n_violate_theta1}  (expected 0 -- true metric)")
    print(f"  1 - CKA (pseudo):           {n_violate_cka_dist}  (CKA is not a metric;"
          f" violations expected)")

    out = {
        "pairing": args.pairing,
        "n_paired_episodes": int(m),
        "conditions": cond_names,
        "d1_matrix": d1_mat.tolist(),
        "theta1_matrix": theta1_mat.tolist(),
        "linear_cka_matrix": cka_mat.tolist(),
        "triangle_violations": {
            "d1": int(n_violate_d1),
            "theta1": int(n_violate_theta1),
            "1_minus_cka": int(n_violate_cka_dist),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}\n")

    # Summary tables
    def print_mat(name, M):
        print(f"\n  {name}:")
        print(f"    {'':<18}" + "".join(f"{c:<12}" for c in cond_names))
        for i, c in enumerate(cond_names):
            print(f"    {c:<18}" + "".join(f"{M[i,j]:<12.4f}" for j in range(n_c)))

    print_mat("d_1 (Procrustes size-shape distance, lower = more similar)", d1_mat)
    print_mat("theta_1 (Riemannian angular distance, [0, pi/2])", theta1_mat)
    print_mat("Linear CKA (higher = more similar; for cross-method comparison)",
              cka_mat)


if __name__ == "__main__":
    main()
