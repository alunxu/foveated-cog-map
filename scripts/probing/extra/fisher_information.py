"""Linear Fisher information about position (Beck/Brunel/Kanitscheider).

I_F(s) = (d_mu/ds)^T Sigma^-1 (d_mu/ds)

where mu(s) = E[h | pos = s] and Sigma is the within-bin noise covariance.

Pre-registered prediction (cogneuro_round2/fisher_info.md):
  Bias-corrected linear Fisher info I_F(pos) is monotone-decreasing across
  {blind, coarse, fov-LP, foveated, uniform}; relative ratio blind/uniform
  >= 2x. Decomposition shows the magnitude difference is signal-dominated
  (d_mu/ds variation) rather than noise-dominated (Sigma variation).

Procedure:
  1. PCA-50 reduce h_t (Sigma-inverse needs to be well-conditioned).
  2. Bin pos into 20 quantile bins per axis.
  3. Estimate mu(s) (per-bin mean) and Sigma (within-bin pooled cov).
  4. Bias-correction via Kanitscheider 2015 train/test harmonic-mean estimator.
  5. Report I_F across regularisation grid lambda ∈ {1e-4, 1e-3, 1e-2}; pre-reg
     the median value; require ordering to be stable across the grid.
"""
from __future__ import annotations

import argparse
import json
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


def estimate_signal_and_noise(h: np.ndarray, s: np.ndarray, n_bins: int = 20):
    """Bin s into n_bins quantile bins; return per-bin mean (mu) and pooled
    within-bin covariance (Sigma)."""
    qs = np.quantile(s, np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_idx = np.digitize(s, qs)
    mus = []
    sigma_acc = np.zeros((h.shape[1], h.shape[1]), dtype=np.float64)
    n_total = 0
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() < 50:
            mus.append(None)
            continue
        h_b = h[m].astype(np.float64)
        mu_b = h_b.mean(0)
        mus.append(mu_b)
        # pooled within-bin covariance
        diff = h_b - mu_b
        sigma_acc += diff.T @ diff
        n_total += len(h_b)
    if n_total == 0:
        return None
    Sigma = sigma_acc / max(n_total - n_bins, 1)
    s_centers = np.array([(qs[max(0, b - 1)] + qs[min(len(qs) - 1, b)]) / 2.0
                           if b > 0 and b < n_bins
                           else (s.min() if b == 0 else s.max())
                           for b in range(n_bins)])
    valid = [m is not None for m in mus]
    mu_arr = np.stack([m for m in mus if m is not None])
    s_centers_v = s_centers[valid]
    return mu_arr, Sigma, s_centers_v


def linear_fisher(h: np.ndarray, s: np.ndarray, n_bins: int = 20,
                   regularise: float = 1e-3) -> dict:
    """Returns dict with linear Fisher + decomposition into signal / noise."""
    res = estimate_signal_and_noise(h, s, n_bins=n_bins)
    if res is None:
        return {"fisher": np.nan, "signal_norm": np.nan, "noise_eig_max": np.nan}
    mu_arr, Sigma, s_centers = res
    # Regularise + invert Sigma
    Sigma_reg = Sigma + regularise * np.eye(Sigma.shape[0]) * np.trace(Sigma) / Sigma.shape[0]
    Sigma_inv = np.linalg.inv(Sigma_reg)
    # Signal direction: gradient of mu w.r.t. s
    if len(s_centers) < 3:
        return {"fisher": np.nan, "signal_norm": np.nan, "noise_eig_max": np.nan}
    dmu_ds = np.gradient(mu_arr, s_centers, axis=0).mean(axis=0)
    fisher = float(dmu_ds @ Sigma_inv @ dmu_ds)
    return {
        "fisher": fisher,
        "signal_norm": float(np.linalg.norm(dmu_ds)),
        "noise_trace": float(np.trace(Sigma)),
        "noise_eig_max": float(np.linalg.eigvalsh(Sigma)[-1]),
    }


def bias_corrected_fisher(h: np.ndarray, s: np.ndarray, ep_id: np.ndarray,
                            n_bins: int = 20, regularise: float = 1e-3,
                            n_bootstrap: int = 10) -> dict:
    """Kanitscheider 2015 train/test split, harmonic-mean estimator."""
    eps_unique = np.unique(ep_id)
    rng = np.random.default_rng(0)
    fishers = []
    sigs = []
    noises = []
    for b in range(n_bootstrap):
        rng.shuffle(eps_unique)
        n_tr = len(eps_unique) // 2
        train_eps = set(eps_unique[:n_tr].tolist())
        train = np.array([e in train_eps for e in ep_id])
        I_tr = linear_fisher(h[train], s[train], n_bins=n_bins, regularise=regularise)
        I_te = linear_fisher(h[~train], s[~train], n_bins=n_bins, regularise=regularise)
        if not (np.isnan(I_tr["fisher"]) or np.isnan(I_te["fisher"])):
            harmonic = 2 * I_tr["fisher"] * I_te["fisher"] / (I_tr["fisher"] + I_te["fisher"] + 1e-9)
            fishers.append(harmonic)
            sigs.append((I_tr["signal_norm"] + I_te["signal_norm"]) / 2)
            noises.append((I_tr["noise_eig_max"] + I_te["noise_eig_max"]) / 2)
    if not fishers:
        return {"fisher_mean": np.nan}
    return {
        "fisher_mean": float(np.mean(fishers)),
        "fisher_std": float(np.std(fishers)),
        "signal_norm": float(np.mean(sigs)),
        "noise_eig_max": float(np.mean(noises)),
        "n_bootstrap": len(fishers),
    }


def analyse_one_condition(npz_path: Path, n_pcs: int = 50, n_max: int = 80000,
                            n_bins: int = 20, n_bootstrap: int = 10) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    pos = d["positions"].astype(np.float32)
    ep_id = d["episode_ids"]

    if len(h) > n_max:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(h), n_max, replace=False))
        h, pos, ep_id = h[idx], pos[idx], ep_id[idx]

    mu = h.mean(0, keepdims=True); sd = h.std(0, keepdims=True) + 1e-6
    h = (h - mu) / sd
    pca = PCA(n_components=n_pcs, random_state=0).fit(h)
    h_pcs = pca.transform(h).astype(np.float32)

    results = {}
    # Habitat positions: axis 0 = x (horizontal), axis 1 = y (height, ~constant),
    # axis 2 = z (horizontal). Fisher about ground-plane position uses axes 0, 2.
    for axis_name, axis_idx in [("x", 0), ("z", 2)]:
        s = pos[:, axis_idx]
        per_lambda = {}
        for reg in [1e-4, 1e-3, 1e-2]:
            r = bias_corrected_fisher(h_pcs, s, ep_id, n_bins=n_bins,
                                        regularise=reg, n_bootstrap=n_bootstrap)
            per_lambda[f"{reg:.0e}"] = r
        results[axis_name] = per_lambda
        med = np.median([per_lambda[k]["fisher_mean"] for k in per_lambda
                          if not np.isnan(per_lambda[k]["fisher_mean"])])
        results[f"{axis_name}_median"] = float(med)

    results["total_median"] = float(results["x_median"] + results["z_median"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/fisher_results.json")
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--n_max", type=int, default=80000)
    ap.add_argument("--n_bins", type=int, default=20)
    ap.add_argument("--n_bootstrap", type=int, default=10)
    args = ap.parse_args()

    results = {}
    for cond, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            continue
        print(f"\n=== {cond} ===")
        results[cond] = analyse_one_condition(path, n_pcs=args.n_pcs,
                                                 n_max=args.n_max, n_bins=args.n_bins,
                                                 n_bootstrap=args.n_bootstrap)
        print(f"  Fisher: x median={results[cond]['x_median']:.2f}  "
               f"z median={results[cond]['z_median']:.2f}  "
               f"total={results[cond]['total_median']:.2f}")

    json.dump(results, open(args.out_path, "w"), indent=2)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
