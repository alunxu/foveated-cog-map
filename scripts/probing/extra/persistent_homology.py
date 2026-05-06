"""Persistent homology of the cognitive-map manifold (Gardner et al. 2022, Nature).

The Gardner-Moser result: grid-cell activity in mouse MEC lives on a torus
(persistent Betti-1 = 2, Betti-2 = 1) recovered from population recordings via
Vietoris-Rips persistent homology. This is the canonical "topology of the
cognitive map" cogneuro precedent.

Pre-registered prediction (cogneuro_round2/persistent_homology.md):
  (a) median Wasserstein-2 distance between {blind, uniform} pairs exceeds
      2x the within-condition seed-to-seed Wasserstein;
  (b) persistent Betti-1 lower for {blind, coarse} than for sighted-rich;
  (c) bottleneck distance rank-orders conditions matching encoder bandwidth.

Pre-reg HPs (locked):
  - n_subsample = 1500 (smaller than the doc's 2000 for runtime budget;
    the doc-suggested fallback to 1000 + PCA-32 used here as PCA-32 + 1500
    on cosine distance to retain enough signal)
  - n_seeds = 4 (smaller than doc's 5 for budget)
  - persistence threshold = 0.1
  - max_dim = 1 (Betti-2 is too compute-heavy at 1500 pts in 32-d)
  - PCA-32 pre-projection (validated on round-1 TGM that PCA-30 preserves
    cross-condition signal)
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

DATA_ROOT = Path("/tmp/rcp_analysis_v3")
FILES = {
    "blind": "blind_izar_det_RCP.npz",
    "coarse": "coarse_det_RCP.npz",
    "foveated": "foveated_det_RCP.npz",
    "foveated_logpolar": "foveated_logpolar_det_RCP.npz",
    "uniform": "uniform_det_RCP.npz",
}


def normalise_to_sphere(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def compute_persistence(H_pcs: np.ndarray, n_subsample: int, seed: int,
                          max_dim: int = 1) -> dict:
    """Subsample H_pcs uniformly, compute Vietoris-Rips persistence diagram on
    cosine distance (via L2-normalised Euclidean)."""
    from ripser import ripser
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(H_pcs), n_subsample, replace=False)
    X = normalise_to_sphere(H_pcs[idx])
    res = ripser(X, maxdim=max_dim)
    return {f"dgm_{d}": np.asarray(res["dgms"][d], dtype=np.float32)
             for d in range(max_dim + 1)}


def persistent_count(dgm: np.ndarray, threshold: float = 0.1) -> int:
    """Count features with persistence (death - birth) > threshold."""
    if len(dgm) == 0:
        return 0
    persistences = dgm[:, 1] - dgm[:, 0]
    finite = np.isfinite(persistences)
    return int((persistences[finite] > threshold).sum())


def wasserstein_safe(dgm_a: np.ndarray, dgm_b: np.ndarray, p: int = 2) -> float:
    from persim import wasserstein
    a = np.asarray(dgm_a, dtype=np.float64)
    b = np.asarray(dgm_b, dtype=np.float64)
    a = a[np.isfinite(a).all(1)]
    b = b[np.isfinite(b).all(1)]
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0:
        return float(np.linalg.norm(b[:, 1] - b[:, 0]))
    if len(b) == 0:
        return float(np.linalg.norm(a[:, 1] - a[:, 0]))
    try:
        return float(wasserstein(a, b))
    except Exception:
        return float("nan")


def bottleneck_safe(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    from persim import bottleneck
    a = np.asarray(dgm_a, dtype=np.float64)
    b = np.asarray(dgm_b, dtype=np.float64)
    a = a[np.isfinite(a).all(1)]
    b = b[np.isfinite(b).all(1)]
    try:
        return float(bottleneck(a, b))
    except Exception:
        return float("nan")


def analyse_condition(npz_path: Path, n_subsample: int, n_seeds: int,
                        n_pcs: int, max_dim: int) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    mu = h.mean(0, keepdims=True); sd = h.std(0, keepdims=True) + 1e-6
    h = (h - mu) / sd
    pca = PCA(n_components=n_pcs, random_state=0).fit(h)
    h_pcs = pca.transform(h).astype(np.float32)

    print(f"  PCA top-{n_pcs} cum var = {pca.explained_variance_ratio_.sum():.3f}")
    diagrams = []
    for seed in range(n_seeds):
        d_seed = compute_persistence(h_pcs, n_subsample=n_subsample, seed=seed,
                                       max_dim=max_dim)
        diagrams.append(d_seed)
        print(f"    seed {seed}: dgm0 size={len(d_seed['dgm_0'])}  "
               f"dgm1 size={len(d_seed.get('dgm_1', []))}", flush=True)
    return diagrams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/persistence_results.npz")
    ap.add_argument("--summary_path", default="/tmp/persistence_summary.json")
    ap.add_argument("--n_subsample", type=int, default=1500)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--n_pcs", type=int, default=32)
    ap.add_argument("--max_dim", type=int, default=1)
    ap.add_argument("--persistence_thresh", type=float, default=0.1)
    args = ap.parse_args()

    all_dgms = {}
    for cond, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f"  MISSING {path}; skip {cond}")
            continue
        print(f"\n=== {cond} ===")
        all_dgms[cond] = analyse_condition(path, n_subsample=args.n_subsample,
                                              n_seeds=args.n_seeds,
                                              n_pcs=args.n_pcs,
                                              max_dim=args.max_dim)

    # Save raw diagrams (compressed npz)
    save_dict = {}
    for cond, seeds in all_dgms.items():
        for i, d in enumerate(seeds):
            for dim_key, arr in d.items():
                save_dict[f"{cond}_{i}_{dim_key}"] = arr
    np.savez_compressed(args.out_path, **save_dict)

    # Summary statistics
    summary = {"per_cond": {}, "wasserstein_within": {}, "wasserstein_across": {},
                "bottleneck_across": {}}
    for cond, seeds in all_dgms.items():
        summary["per_cond"][cond] = {}
        for dim_key in seeds[0]:
            counts = [persistent_count(d[dim_key], args.persistence_thresh) for d in seeds]
            persistence_max = []
            for d in seeds:
                arr = d[dim_key]
                pers = arr[:, 1] - arr[:, 0] if len(arr) > 0 else np.array([0.0])
                pers = pers[np.isfinite(pers)]
                if len(pers) == 0:
                    persistence_max.append(0.0)
                else:
                    persistence_max.append(float(np.max(pers)))
            summary["per_cond"][cond][dim_key] = {
                "persistent_count_mean": float(np.mean(counts)),
                "persistent_count_std": float(np.std(counts)),
                "max_persistence_mean": float(np.mean(persistence_max)),
            }

    # Within-condition seed-to-seed Wasserstein (dgm_1 only, primary signal)
    for cond in all_dgms:
        ws = []
        for i, j in combinations(range(args.n_seeds), 2):
            for dim in [1] if args.max_dim >= 1 else [0]:
                ws.append(wasserstein_safe(all_dgms[cond][i][f"dgm_{dim}"],
                                              all_dgms[cond][j][f"dgm_{dim}"]))
        summary["wasserstein_within"][cond] = {
            "mean": float(np.nanmean(ws)),
            "std": float(np.nanstd(ws)),
        }

    # Across-condition Wasserstein (using seed-0 of each)
    for c1, c2 in combinations(all_dgms.keys(), 2):
        ws = []
        for i in range(args.n_seeds):
            for dim in [1] if args.max_dim >= 1 else [0]:
                ws.append(wasserstein_safe(all_dgms[c1][i][f"dgm_{dim}"],
                                              all_dgms[c2][i][f"dgm_{dim}"]))
        summary["wasserstein_across"][f"{c1}_vs_{c2}"] = float(np.nanmean(ws))
        bottlenecks = []
        for i in range(args.n_seeds):
            for dim in [1] if args.max_dim >= 1 else [0]:
                bottlenecks.append(bottleneck_safe(all_dgms[c1][i][f"dgm_{dim}"],
                                                       all_dgms[c2][i][f"dgm_{dim}"]))
        summary["bottleneck_across"][f"{c1}_vs_{c2}"] = float(np.nanmean(bottlenecks))

    json.dump(summary, open(args.summary_path, "w"), indent=2)
    print(f"\nsaved diagrams to {args.out_path}")
    print(f"saved summary to {args.summary_path}")
    print("\n=== Summary ===")
    print("Per-condition persistent Betti counts (>{}):".format(args.persistence_thresh))
    for cond in summary["per_cond"]:
        print(f"  {cond:>20s}:", end="")
        for dim_key in summary["per_cond"][cond]:
            c = summary["per_cond"][cond][dim_key]
            print(f"  {dim_key} ct={c['persistent_count_mean']:.1f} max_p={c['max_persistence_mean']:.3f}", end="")
        print()


if __name__ == "__main__":
    main()
