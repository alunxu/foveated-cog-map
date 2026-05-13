"""
Time-cell analysis for LSTM hidden units.

Tests whether individual units develop sharp, non-overlapping temporal tuning
within episodes — the single-unit signature of hippocampal time cells
(Eichenbaum 2014; MacDonald et al. 2011).

A time cell fires at a specific, reproducible point in an episode regardless
of the agent's location. In a population, time cells tile the episode with
their peaks, providing an internal clock that is distinct from place coding.

Method:
  1. Normalise episode time: t_norm = step_in_episode / episode_length ∈ [0, 1).
  2. Bin into N_TIME_BINS bins and compute the mean activation per bin for
     each unit → temporal tuning curve T(t).
  3. Fit a Gaussian to T(t):  A·exp(−(t − μ)² / 2σ²) + B.
  4. Declare a unit a "time cell" if all of:
       (a) Gaussian R² ≥ FIT_R2_THRESH  (good fit to a single peak)
       (b) FWHM ≤ MAX_FWHM_FRAC of the epoch  (narrow temporal tuning)
       (c) Peak / (baseline + ε) ≥ MIN_SELECTIVITY  (significant modulation)
       (d) Amplitude above shuffle-null 99th percentile  (not noise)
  5. Population-level: check whether time cells tile the episode (uniform
     distribution of peak times) vs. cluster at the start/end.

Per-condition output:
  - Fraction of units classified as time cells
  - Distribution of peak times μ (tests tiling)
  - Distribution of FWHM (sharpness of temporal tuning)
  - Mean Gaussian R² across all units
  - Shuffle null for peak amplitude
  - Per-unit tuning curves (saved as NPZ for figure generation)

Usage:
    python scripts/probing/time_cells.py \\
        --in-dir /scratch/izar/$USER/probing_data \\
        --conds blind_gibson uniform_gibson foveated_gibson \\
        --out results/probing/time_cells.json \\
        --out-npz results/probing/time_cells_curves.npz

Reads:  <in-dir>/<cond>.npz   (output of scripts/probing/collect.py)
Writes: <out> JSON + optional <out-npz> (tuning curves for all units/conds)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kstest, uniform

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────

N_TIME_BINS       = 20     # bins over normalised episode time [0, 1)
MIN_EPISODE_STEPS = 20     # skip episodes shorter than this
MIN_BINS_WITH_DATA = 10    # unit must have data in at least this many bins
N_SHUFFLES        = 500    # shuffles for amplitude null distribution
NULL_PERCENTILE   = 99     # significance threshold

# Time-cell classification criteria
FIT_R2_THRESH    = 0.60    # Gaussian goodness-of-fit
MAX_FWHM_FRAC    = 0.45    # peak width ≤ 45 % of epoch
MIN_SELECTIVITY  = 1.8     # peak / (baseline + ε) ratio


# ── Gaussian fitting ───────────────────────────────────────────────────────────

def _gaussian(t: np.ndarray, A: float, mu: float, sigma: float, B: float) -> np.ndarray:
    """Gaussian with baseline: A·exp(−(t−μ)²/2σ²) + B."""
    return A * np.exp(-0.5 * ((t - mu) / (sigma + 1e-8)) ** 2) + B


def fit_gaussian_tuning(tuning_curve: np.ndarray,
                        bin_centers: np.ndarray) -> dict | None:
    """Fit a Gaussian to a temporal tuning curve.

    Returns a dict with {A, mu, sigma, B, r2, fwhm, selectivity} or None
    if the fit fails or the curve has no data in enough bins.
    """
    valid = ~np.isnan(tuning_curve)
    if valid.sum() < MIN_BINS_WITH_DATA:
        return None

    t_v = bin_centers[valid]
    y_v = tuning_curve[valid]

    # Initial guess: amplitude at peak bin, width = 0.15 epoch, zero baseline.
    peak_idx = int(np.argmax(y_v))
    A0  = float(y_v[peak_idx] - y_v.min())
    mu0 = float(t_v[peak_idx])
    B0  = float(y_v.min())

    try:
        popt, _ = curve_fit(
            _gaussian, t_v, y_v,
            p0=[A0, mu0, 0.15, B0],
            bounds=([0, 0, 0.01, -np.inf], [np.inf, 1.0, 0.5, np.inf]),
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return None

    A, mu, sigma, B = popt
    if A < 0 or sigma <= 0:
        return None

    # Goodness-of-fit R².
    pred = _gaussian(t_v, *popt)
    ss_res = np.sum((y_v - pred) ** 2)
    ss_tot = np.sum((y_v - y_v.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    fwhm       = float(2.355 * sigma)              # full-width at half-max
    baseline   = float(B)
    selectivity = float(A / (abs(baseline) + 1e-8))

    return {
        "A": float(A), "mu": float(mu), "sigma": float(sigma), "B": float(B),
        "r2": r2, "fwhm": fwhm, "selectivity": selectivity,
        "peak_value": float(A + B),
    }


# ── tuning curve computation ───────────────────────────────────────────────────

def compute_tuning_curves(
    H: np.ndarray,
    step_in_ep: np.ndarray,
    ep_ids: np.ndarray,
    n_bins: int = N_TIME_BINS,
    min_ep_steps: int = MIN_EPISODE_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute temporal tuning curves for all units.

    Returns:
        tuning: (hidden_dim, n_bins) — mean activation per normalised-time bin.
        bin_centers: (n_bins,) — bin centre positions in [0, 1).
    """
    hidden_dim = H.shape[1]
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    sum_act  = np.zeros((hidden_dim, n_bins), dtype=np.float64)
    count    = np.zeros((hidden_dim, n_bins), dtype=np.float64)

    for ep in np.unique(ep_ids):
        ep_mask  = ep_ids == ep
        steps    = step_in_ep[ep_mask]
        ep_len   = steps.max() + 1
        if ep_len < min_ep_steps:
            continue

        t_norm   = steps / ep_len                  # in [0, 1)
        bin_idx  = np.clip(
            np.digitize(t_norm, bin_edges) - 1, 0, n_bins - 1)

        H_ep = H[ep_mask]
        for bi in range(n_bins):
            bmask = bin_idx == bi
            if bmask.sum() == 0:
                continue
            sum_act[:, bi] += H_ep[bmask].sum(axis=0)
            count[:, bi]   += bmask.sum()

    with np.errstate(invalid='ignore'):
        tuning = np.where(count > 0, sum_act / count, np.nan)  # (D, n_bins)

    return tuning, bin_centers


def shuffle_peak_amplitudes(
    H: np.ndarray,
    step_in_ep: np.ndarray,
    ep_ids: np.ndarray,
    n_bins: int = N_TIME_BINS,
    n_shuffles: int = N_SHUFFLES,
    min_ep_steps: int = MIN_EPISODE_STEPS,
    n_sample_units: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute null distribution of peak amplitudes via episode-time shuffling.

    For each shuffle, randomly permute step indices within every episode
    (destroys temporal structure while keeping spatial coverage and hidden-state
    marginal distributions) then recompute the tuning curve peak amplitude.

    Returns a (n_shuffles × n_sample_units,) array of null peak amplitudes.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    hidden_dim = H.shape[1]
    sample_ui  = rng.choice(hidden_dim, size=min(n_sample_units, hidden_dim), replace=False)
    null_peaks: list[float] = []

    ep_groups: dict[int, np.ndarray] = {}
    for ep in np.unique(ep_ids):
        ep_mask = ep_ids == ep
        if step_in_ep[ep_mask].max() + 1 >= min_ep_steps:
            ep_groups[ep] = np.where(ep_mask)[0]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for _ in range(n_shuffles):
        sum_null = np.zeros((len(sample_ui), n_bins), dtype=np.float64)
        cnt_null = np.zeros((len(sample_ui), n_bins), dtype=np.float64)

        for ep, indices in ep_groups.items():
            steps  = step_in_ep[indices]
            ep_len = steps.max() + 1
            # Permute time within episode.
            perm_steps = rng.permutation(len(indices))
            t_norm = perm_steps / ep_len
            bin_idx = np.clip(np.digitize(t_norm, bin_edges) - 1, 0, n_bins - 1)

            H_ep = H[indices][:, sample_ui]   # (ep_steps, n_sample_units)
            for bi in range(n_bins):
                bmask = bin_idx == bi
                if bmask.sum() == 0:
                    continue
                sum_null[:, bi] += H_ep[bmask].sum(axis=0)
                cnt_null[:, bi] += bmask.sum()

        with np.errstate(invalid='ignore'):
            tc_null = np.where(cnt_null > 0, sum_null / cnt_null, np.nan)

        # Peak amplitude = max(tuning curve) − min(tuning curve) per unit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peaks = np.nanmax(tc_null, axis=1) - np.nanmin(tc_null, axis=1)
        null_peaks.extend(peaks[~np.isnan(peaks)].tolist())

    return np.array(null_peaks)


# ── per-condition analysis ─────────────────────────────────────────────────────

def analyse_condition(
    H: np.ndarray,
    step_in_ep: np.ndarray,
    ep_ids: np.ndarray,
    n_bins: int = N_TIME_BINS,
    n_shuffles: int = N_SHUFFLES,
    min_ep_steps: int = MIN_EPISODE_STEPS,
    save_curves: bool = False,
) -> dict:
    """Time-cell analysis for one condition.

    Returns a dict ready for JSON serialisation.  If save_curves=True, also
    returns "tuning_curves" (hidden_dim, n_bins) embedded in the dict.
    """
    hidden_dim = H.shape[1]
    n_eps = len(np.unique(ep_ids))

    # Long enough episodes only.
    valid_eps = [
        ep for ep in np.unique(ep_ids)
        if step_in_ep[ep_ids == ep].max() + 1 >= min_ep_steps
    ]
    if not valid_eps:
        return {"error": f"no episode with >= {min_ep_steps} steps", "n_units": hidden_dim}

    valid_mask = np.isin(ep_ids, valid_eps)
    H_v       = H[valid_mask]
    step_v    = step_in_ep[valid_mask]
    ep_v      = ep_ids[valid_mask]

    print(f"    {len(valid_eps)}/{n_eps} episodes ({valid_mask.sum()} steps), "
          f"{hidden_dim} units")

    tuning, bin_centers = compute_tuning_curves(
        H_v, step_v, ep_v, n_bins=n_bins, min_ep_steps=min_ep_steps)

    # Shuffle null for peak amplitudes.
    print("    computing shuffle null …")
    null_peaks = shuffle_peak_amplitudes(
        H_v, step_v, ep_v, n_bins=n_bins, n_shuffles=n_shuffles,
        min_ep_steps=min_ep_steps)

    null_p99   = float(np.percentile(null_peaks, NULL_PERCENTILE)) if len(null_peaks) else np.nan
    null_mean  = float(np.mean(null_peaks))  if len(null_peaks) else None
    null_std   = float(np.std(null_peaks))   if len(null_peaks) else None

    # Classify each unit.
    unit_results: list[dict] = []
    n_time_cells = 0

    for ui in range(hidden_dim):
        tc = tuning[ui]                     # (n_bins,)
        amp = float(np.nanmax(tc) - np.nanmin(tc))

        fit = fit_gaussian_tuning(tc, bin_centers)
        if fit is None:
            unit_results.append({"unit": ui, "is_time_cell": False,
                                  "amp": amp, "fit": None})
            continue

        amp_sig  = amp > null_p99 if not np.isnan(null_p99) else True
        good_fit = fit["r2"] >= FIT_R2_THRESH
        narrow   = fit["fwhm"] <= MAX_FWHM_FRAC
        selective = fit["selectivity"] >= MIN_SELECTIVITY

        is_tc = amp_sig and good_fit and narrow and selective
        if is_tc:
            n_time_cells += 1

        unit_results.append({
            "unit": ui,
            "is_time_cell": bool(is_tc),
            "amp": float(amp),
            "amp_sig": bool(amp_sig),
            **fit,
        })

    # Population-level tiling: are peak times uniformly distributed?
    tc_peaks = [r["mu"] for r in unit_results if r.get("is_time_cell")]
    if len(tc_peaks) >= 8:
        ks_stat, ks_p = kstest(tc_peaks, uniform(loc=0, scale=1).cdf)
        tiling_uniform = bool(ks_p > 0.05)   # fail to reject uniformity → tiling
    else:
        ks_stat, ks_p, tiling_uniform = np.nan, np.nan, None

    fwhm_all  = [r["fwhm"] for r in unit_results if r.get("fwhm") is not None]
    r2_all    = [r["r2"]   for r in unit_results if r.get("r2")   is not None]

    out: dict = {
        "n_episodes":     len(valid_eps),
        "n_steps":        int(valid_mask.sum()),
        "n_units":        int(hidden_dim),
        "n_time_cells":   n_time_cells,
        "frac_time_cells": float(n_time_cells / hidden_dim),
        "null_p99":       float(null_p99),
        "null_mean":      null_mean,
        "null_std":       null_std,
        "criteria": {
            "fit_r2_thresh":   FIT_R2_THRESH,
            "max_fwhm_frac":   MAX_FWHM_FRAC,
            "min_selectivity": MIN_SELECTIVITY,
            "null_percentile": NULL_PERCENTILE,
        },
        "peak_times_time_cells": tc_peaks,
        "tiling_ks_stat":   float(ks_stat) if not np.isnan(ks_stat) else None,
        "tiling_ks_p":      float(ks_p)    if not np.isnan(ks_p)    else None,
        "tiling_uniform":   tiling_uniform,
        "mean_fwhm":        float(np.mean(fwhm_all)) if fwhm_all else None,
        "median_fwhm":      float(np.median(fwhm_all)) if fwhm_all else None,
        "mean_r2":          float(np.mean(r2_all)) if r2_all else None,
        "null_distribution_sample": null_peaks[:500].tolist(),
        "bin_centers":      bin_centers.tolist(),
        "units":            unit_results,
    }
    if save_curves:
        out["tuning_curves"] = tuning.tolist()   # (hidden_dim, n_bins)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-cell analysis for LSTM hidden units")
    p.add_argument("--in-dir",  required=True,
                   help="Directory containing <cond>.npz files from collect.py")
    p.add_argument("--conds", nargs="+",
                   default=["blind_gibson", "uniform_gibson",
                            "foveated_gibson", "foveated_logpolar_gibson"],
                   help="Condition name(s) matching <cond>.npz in --in-dir")
    p.add_argument("--out", required=True,
                   help="Output JSON, e.g. results/probing/time_cells.json")
    p.add_argument("--out-npz", default=None,
                   help="Optional NPZ for per-unit tuning curves (all conditions)")
    p.add_argument("--n-bins",          type=int,   default=N_TIME_BINS)
    p.add_argument("--n-shuffles",      type=int,   default=N_SHUFFLES)
    p.add_argument("--min-ep-steps",    type=int,   default=MIN_EPISODE_STEPS)
    p.add_argument("--layer", type=int, default=-1,
                   help="LSTM layer index: -1=top (default), 0=bottom, 1=mid")
    p.add_argument("--save-curves", action="store_true",
                   help="Embed tuning curves in JSON (large; prefer --out-npz)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    in_dir = Path(args.in_dir)
    out_p  = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    results:   dict = {}
    npz_store: dict = {}

    for cond in args.conds:
        npz = in_dir / f"{cond}.npz"
        if not npz.exists():
            print(f"\n[{cond}] {npz} not found — skipping")
            results[cond] = {"error": "npz not found"}
            continue

        print(f"\n{'='*60}\n  {cond}\n{'='*60}")
        data = np.load(npz, allow_pickle=True)

        if "h_layers" in data:
            H = data["h_layers"][:, args.layer, :]
        else:
            H = data["hidden_states"]

        step_in_ep = data["step_in_episode"]
        ep_ids     = data["episode_ids"]
        N, D = H.shape
        print(f"  N={N}, hidden_dim={D}, episodes={len(np.unique(ep_ids))}")

        save_curves = args.save_curves or (args.out_npz is not None)
        res = analyse_condition(
            H=H, step_in_ep=step_in_ep, ep_ids=ep_ids,
            n_bins=args.n_bins, n_shuffles=args.n_shuffles,
            min_ep_steps=args.min_ep_steps,
            save_curves=save_curves,
        )

        # Move tuning curves to separate NPZ store to keep JSON lightweight.
        if args.out_npz and "tuning_curves" in res:
            npz_store[f"{cond}_tuning_curves"] = np.array(res.pop("tuning_curves"))
            npz_store[f"{cond}_bin_centers"]   = np.array(res["bin_centers"])

        results[cond] = res

        if "error" not in res:
            print(f"\n  Time cells   : {res['n_time_cells']} / {res['n_units']}  "
                  f"({res['frac_time_cells']*100:.1f}%)")
            print(f"  Null p99 amp : {res['null_p99']:.4f}")
            print(f"  Mean FWHM    : {res.get('mean_fwhm', 'n/a')}")
            print(f"  Tiling (KS p): {res.get('tiling_ks_p', 'n/a')}  "
                  f"→ {'uniform (tiling ✓)' if res.get('tiling_uniform') else 'non-uniform'}")

    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON → {out_p}")

    if args.out_npz and npz_store:
        np.savez_compressed(args.out_npz, **npz_store)
        print(f"Saved tuning curves → {args.out_npz}")


if __name__ == "__main__":
    main()
