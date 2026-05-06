"""Intrinsic Dimension + Participation Ratio analyzer for LSTM top-layer h-states.

Following:
  - Ansuini, Laio, Macke, Zoccolan, "Intrinsic dimension of data representations
    in deep neural networks", NeurIPS 2019 (TwoNN method).
  - Facco, d'Errico, Rodriguez, Laio, "Estimating the intrinsic dimension of
    datasets by a minimal neighborhood information", Sci. Rep. 2017
    (TwoNN original derivation; CDF + linear-regression variant).

Two architecture-agnostic scalars per condition:
  - Participation Ratio (PR): soft count of effective LINEAR dimensions,
        PR = (sum eigval)^2 / sum(eigval^2)
    on the centered hidden-state covariance.
  - TwoNN intrinsic dimension via two estimators (Ansuini eq. 1 derivation):
      (a) MLE:  d_hat = N / sum(log mu_i),  mu_i = r_i^(2) / r_i^(1)
      (b) CDF + linear regression on (-log(1 - F_emp(mu))) vs log(mu);
          slope = d. (Facco recommends; less sensitive to outliers.)
    We report (a) and (b) per bootstrap resample so we can flag disagreement.

  Caveat (Ansuini sec. 2): TwoNN UNDERESTIMATES when d > 20 with non-uniform
  density. For LSTM top-layer states we expect d << 20 so this caveat
  does not bind; we still log the (a)-vs-(b) gap as a sanity check.

  Reliability check (Ansuini Fig 2B / Facco): subsampling analysis.
  Compute ID at sample sizes {500, 1000, 2500, 5000, 10000}, flag if curve
  is unstable -- "scale invariance" is the published reliability signature.

Reads:  --in-dir <dir>/{cond}_gibson_det.npz   (canonical converged rollout)
Writes: <out>.json with PR + TwoNN-ID(MLE) + TwoNN-ID(CDF) + subsample curve
        per condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


# Display name (paper convention) -> NPZ filename stem (legacy training cond name)
# "coarse" was historically named "matched" (compute-matched to other sighted
# conditions); paper §3.1 uses "Coarse" to emphasize the 1x1 spatial collapse.
# Both refer to the SAME agent: 48x48 RGB -> ResNet-18 -> 1x1 feature map.
COND_NPZ_MAP = {
    "blind": "blind",
    "coarse": "matched",
    "uniform": "uniform",
    "foveated": "foveated",
    "foveated_learned": "foveated_learned",
}
CONDS = list(COND_NPZ_MAP.keys())
H_KEY = "hidden_states"  # top-layer h_2


def participation_ratio(X: np.ndarray) -> float:
    """PR = (Σ λ)^2 / Σ λ^2 on the unbiased sample covariance (n-1)."""
    X = X - X.mean(axis=0)
    cov = (X.T @ X) / max(len(X) - 1, 1)
    lam = np.linalg.eigvalsh(cov)
    lam = lam[lam > 1e-12]
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def _twonn_mu(X: np.ndarray) -> np.ndarray:
    """Compute mu_i = r_i^(2) / r_i^(1) for all i. Filter degenerate (mu <= 1)."""
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)  # d[:, 0] = self distance = 0; d[:, 1] = r1; d[:, 2] = r2
    r1, r2 = d[:, 1], d[:, 2]
    mu = r2 / np.maximum(r1, 1e-12)
    return mu[mu > 1.0]


def twonn_id_mle(X: np.ndarray) -> float:
    """Ansuini eq. 1 closed-form MLE: d = N / Σ log(mu)."""
    mu = _twonn_mu(X)
    if len(mu) == 0:
        return float("nan")
    return float(len(mu) / np.sum(np.log(mu)))


def twonn_id_cdf(X: np.ndarray) -> float:
    """Facco 2017 CDF method: linear regression slope of -log(1 - F_emp(mu))
    against log(mu). Slope = d. Less sensitive to outliers than MLE."""
    mu = _twonn_mu(X)
    if len(mu) < 10:
        return float("nan")
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)
    # F_emp at the i-th sorted mu is i/n; use (i-1)/n to avoid log(0)
    F = np.arange(1, n + 1) / (n + 1)  # plug-in shift to keep F < 1
    x = np.log(mu_sorted)
    y = -np.log(1 - F)
    # Drop first/last 5% to avoid CDF tail noise (Facco standard)
    lo, hi = int(0.05 * n), int(0.95 * n)
    x, y = x[lo:hi], y[lo:hi]
    # Least-squares slope through origin (theoretical: y = d*x)
    slope = float(np.sum(x * y) / np.sum(x * x))
    return slope


def subsample_id(X: np.ndarray, sizes=(500, 1000, 2500, 5000), n_boot=10, seed=0):
    """Reliability check (Ansuini Fig 2B): ID across sample sizes."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in sizes:
        if s > len(X):
            continue
        ids_mle = []
        ids_cdf = []
        for b in range(n_boot):
            idx = rng.choice(len(X), s, replace=False)
            Xs = X[idx]
            ids_mle.append(twonn_id_mle(Xs))
            ids_cdf.append(twonn_id_cdf(Xs))
        out[str(s)] = {
            "id_mle_mean": float(np.nanmean(ids_mle)),
            "id_mle_std": float(np.nanstd(ids_mle)),
            "id_cdf_mean": float(np.nanmean(ids_cdf)),
            "id_cdf_std": float(np.nanstd(ids_cdf)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suffix", default="_gibson_det")
    ap.add_argument("--n-bootstrap", type=int, default=10)
    ap.add_argument("--final-size", type=int, default=5000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    results = {}
    for c in CONDS:
        npz_stem = COND_NPZ_MAP[c]
        npz = args.in_dir / f"{npz_stem}{args.suffix}.npz"
        if not npz.exists():
            print(f"  {c} (npz stem '{npz_stem}'): missing -- skip")
            continue
        d = np.load(npz)
        H = d[H_KEY].astype(np.float64)
        print(f"\n  {c} <- {npz_stem}{args.suffix}.npz: H shape {H.shape}, dtype {H.dtype}")

        # PR (uses all samples)
        pr = participation_ratio(H)

        # Final ID estimate: bootstrap at final-size, both estimators
        ids_mle = []
        ids_cdf = []
        for b in range(args.n_bootstrap):
            idx = rng.choice(len(H), min(args.final_size, len(H)), replace=False)
            Xs = H[idx]
            ids_mle.append(twonn_id_mle(Xs))
            ids_cdf.append(twonn_id_cdf(Xs))

        # Reliability subsample curve
        sub = subsample_id(H, sizes=(500, 1000, 2500, 5000), n_boot=args.n_bootstrap)

        results[c] = {
            "n": int(len(H)),
            "d_embed": int(H.shape[1]),
            "pr": pr,
            "twonn_id_mle_mean": float(np.nanmean(ids_mle)),
            "twonn_id_mle_std": float(np.nanstd(ids_mle)),
            "twonn_id_cdf_mean": float(np.nanmean(ids_cdf)),
            "twonn_id_cdf_std": float(np.nanstd(ids_cdf)),
            "subsample_curve": sub,
            "n_bootstrap": args.n_bootstrap,
            "final_size": args.final_size,
        }
        print(f"    PR = {pr:.2f}")
        print(f"    TwoNN-ID (MLE) = {np.nanmean(ids_mle):.3f} +/- {np.nanstd(ids_mle):.3f}")
        print(f"    TwoNN-ID (CDF) = {np.nanmean(ids_cdf):.3f} +/- {np.nanstd(ids_cdf):.3f}")
        print(f"    subsample curve (CDF, mean +/- std):")
        for s_str, r in sub.items():
            print(f"      n={s_str:>5}: {r['id_cdf_mean']:.3f} +/- {r['id_cdf_std']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}\n")

    # Summary
    print(f"{'cond':<18} {'n':<10} {'PR':<10} {'ID-MLE':<14} {'ID-CDF':<14}")
    for c, r in results.items():
        print(
            f"{c:<18} {r['n']:<10} {r['pr']:<10.2f} "
            f"{r['twonn_id_mle_mean']:.2f} +/- {r['twonn_id_mle_std']:.2f}  "
            f"{r['twonn_id_cdf_mean']:.2f} +/- {r['twonn_id_cdf_std']:.2f}"
        )
    print(
        "\nPrediction (Ansuini-style + H1 substitution): if encoder--memory race "
        "shapes representation complexity, expect rich-encoder PR/ID > bottleneck "
        "PR/ID. Disagreement between MLE and CDF estimators flags potential "
        "scale-dependence or outlier contamination."
    )


if __name__ == "__main__":
    main()
