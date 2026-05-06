"""Intrinsic timescales (Murray et al. 2014, Nat Neuro) — per-unit autocorr decay tau.

The cortical hierarchy from Murray 2014: tau (autocorr decay) grows monotonically
from sensory regions (V1 ~50ms) through parietal (~100ms) to prefrontal (~300+ms).
Sensors fast, integrators slow.

For us, this asks: do conditions with more sensory bandwidth show faster intrinsic
timescales (more "sensor-like" units), and conditions with less bandwidth show
slower (more "integrator-like")?

Pre-registered prediction (cogneuro_frameworks/timescales_murray.md):
  blind: longest median tau (must integrate proprioception)
  coarse 1x1: second longest (weak per-step signal)
  foveated: shortest or bimodal
  uniform: shortest (richest per-step signal)
  fov-LP: bimodal

For each unit (PCA-30 reduced), compute single-exponential ACF fit
  ACF(lag) ~ A * exp(-lag/tau) + c
on within-episode autocorrelation averaged across episodes.

Output: per-condition distribution of tau values + median + IQR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

DATA_ROOT = Path("/tmp/rcp_analysis_v3")
FILES = {
    "blind": "blind_izar_det_RCP.npz",
    "coarse": "coarse_det_RCP.npz",
    "foveated": "foveated_det_RCP.npz",
    "foveated_logpolar": "foveated_logpolar_det_RCP.npz",
    "uniform": "uniform_det_RCP.npz",
}


def autocorr(x: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """ACF of 1-D series, lags 0..max_lag, normalised to 1 at lag 0."""
    x = x - x.mean()
    n = len(x)
    var = (x * x).mean() + 1e-9
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        if k >= n:
            break
        out[k] = (x[:-k] * x[k:]).mean() / var
    return out


def per_unit_tau(h: np.ndarray, ep_id: np.ndarray, sip: np.ndarray, max_lag: int = 30,
                  min_ep_len: int = 50) -> np.ndarray:
    """Returns (D,) array of fitted tau (NaN for units that don't fit)."""
    eps = np.unique(ep_id)
    D = h.shape[1]
    taus = np.full(D, np.nan)
    fit_r2 = np.full(D, np.nan)

    # Build per-episode ordered slices once, indexed by unit
    ep_slices = []
    for e in eps:
        m = ep_id == e
        if m.sum() < min_ep_len:
            continue
        order = np.argsort(sip[m])
        ep_slices.append(h[m][order])

    if len(ep_slices) < 5:
        return taus, fit_r2

    for d in range(D):
        acfs = []
        for slc in ep_slices:
            if len(slc) < max_lag + 5:
                continue
            acfs.append(autocorr(slc[:, d], max_lag))
        if len(acfs) < 5:
            continue
        acf_avg = np.mean(acfs, axis=0)
        try:
            (A, tau, c), _ = curve_fit(
                lambda d_, A_, t_, c_: A_ * np.exp(-d_ / t_) + c_,
                np.arange(max_lag + 1), acf_avg,
                p0=[1.0, 5.0, 0.0],
                bounds=([0, 0.1, -1], [2, 200, 1]),
                maxfev=2000,
            )
            taus[d] = tau
            ss_res = ((acf_avg - (A * np.exp(-np.arange(max_lag + 1) / tau) + c)) ** 2).sum()
            ss_tot = ((acf_avg - acf_avg.mean()) ** 2).sum() + 1e-9
            fit_r2[d] = 1.0 - ss_res / ss_tot
        except Exception:
            continue
    return taus, fit_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/timescales.json")
    ap.add_argument("--max_lag", type=int, default=30)
    ap.add_argument("--n_pcs", type=int, default=30)
    ap.add_argument("--min_ep_len", type=int, default=50)
    args = ap.parse_args()

    out = {}
    for cond, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f"  MISSING {path}; skip {cond}")
            continue
        d = np.load(path, allow_pickle=True)
        h = d["hidden_states"].astype(np.float32)
        ep_id = d["episode_ids"]
        sip = d["step_in_episode"]
        # Standardise + PCA
        mu = h.mean(0, keepdims=True); sd = h.std(0, keepdims=True) + 1e-6
        h = (h - mu) / sd
        if args.n_pcs > 0:
            h = PCA(n_components=args.n_pcs, random_state=0).fit_transform(h).astype(np.float32)

        taus, fit_r2 = per_unit_tau(h, ep_id, sip, max_lag=args.max_lag,
                                       min_ep_len=args.min_ep_len)
        valid = (~np.isnan(taus)) & (fit_r2 > 0.5)
        v_taus = taus[valid]
        if len(v_taus) == 0:
            print(f"  {cond}: no units fit (max_lag={args.max_lag}, n_pcs={args.n_pcs})")
            continue
        out[cond] = {
            "n_fit_units": int(len(v_taus)),
            "n_total_units": int(len(taus)),
            "median_tau": float(np.median(v_taus)),
            "mean_tau": float(np.mean(v_taus)),
            "iqr_low": float(np.quantile(v_taus, 0.25)),
            "iqr_high": float(np.quantile(v_taus, 0.75)),
            "p10": float(np.quantile(v_taus, 0.1)),
            "p90": float(np.quantile(v_taus, 0.9)),
            "tau_dist": v_taus.tolist(),
        }
        print(f"\n=== {cond} ===")
        print(f"  fit {len(v_taus)}/{len(taus)} units (R^2 > 0.5)")
        print(f"  median tau = {out[cond]['median_tau']:.2f}  iqr = [{out[cond]['iqr_low']:.2f}, "
               f"{out[cond]['iqr_high']:.2f}]  p10/p90 = {out[cond]['p10']:.2f}/{out[cond]['p90']:.2f}")

    json.dump(out, open(args.out_path, "w"), indent=2)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
