"""Splitter / journey-coding cells (Wood, Dudchenko, Robitsek, Eichenbaum 2000, Neuron).

Hippocampal place cells often show *journey-dependent* firing: cell i fires at
location (x, y) only on a subset of trajectories that pass through (x, y).
"Splitter cells" encode trajectory identity in addition to / instead of pure
spatial position.

DL analogue: for each unit i in cached h_t, fit a 2-way ANOVA
    h_i ~ pos_bin + traj_feature + pos_bin × traj_feature
A unit is a "splitter" if the interaction term is significant after FDR.

Pre-registered prediction (cogneuro_round2/splitter_cells.md):
  Fraction of splitter units (FDR p < 0.01) is monotone-DECREASING with
  encoder bandwidth: blind > coarse > {foveated_logpolar, foveated} > uniform.
  Relative effect blind/uniform >= 2x. eta^2-partial follows the same order.

Trajectory features tested (all pre-registered before running):
  - prev_action: most recent action (6-class discrete)
  - dir_5: rolling 5-step heading mean, octant-binned (8-class)

We report BOTH; pre-reg requires direction-stability across the two definitions.
If they disagree we report the inconsistency, not pick the friendlier one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.stats as st
from statsmodels.stats.multitest import multipletests

DATA_ROOT = Path("/tmp/rcp_analysis_v3")
FILES = {
    "blind": "blind_izar_det_RCP.npz",
    "coarse": "coarse_det_RCP.npz",
    "foveated": "foveated_det_RCP.npz",
    "foveated_logpolar": "foveated_logpolar_det_RCP.npz",
    "uniform": "uniform_det_RCP.npz",
}


def discretise_pos(pos: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretise (N, 2) positions into 1-D pos-bin index using a 2-D grid then flat index."""
    qsx = np.quantile(pos[:, 0], np.linspace(0, 1, n_bins + 1)[1:-1])
    qsy = np.quantile(pos[:, 1], np.linspace(0, 1, n_bins + 1)[1:-1])
    bx = np.digitize(pos[:, 0], qsx)
    by = np.digitize(pos[:, 1], qsy)
    return bx * n_bins + by  # 0..n_bins^2-1


def heading_octant(heading: np.ndarray) -> np.ndarray:
    """Bin 1-D heading (radians) into 8 octants 0..7."""
    h = np.where(heading < 0, heading + 2 * np.pi, heading)
    return (h / (2 * np.pi) * 8).astype(int) % 8


def rolling_dir_octant(heading: np.ndarray, ep_id: np.ndarray, sip: np.ndarray,
                        window: int = 5) -> np.ndarray:
    """Rolling mean of heading (cos, sin then atan2), within-episode, octant-binned."""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    out = np.zeros_like(heading)
    eps = np.unique(ep_id)
    for e in eps:
        m = ep_id == e
        if m.sum() < 1:
            continue
        order = np.argsort(sip[m])
        idx = np.where(m)[0][order]
        cos_ep = cos_h[idx]
        sin_ep = sin_h[idx]
        # rolling mean (window steps before, including current)
        cum_c = np.concatenate([[0], np.cumsum(cos_ep)])
        cum_s = np.concatenate([[0], np.cumsum(sin_ep)])
        T = len(cos_ep)
        roll_c = np.zeros(T)
        roll_s = np.zeros(T)
        for t in range(T):
            lo = max(0, t - window + 1)
            roll_c[t] = (cum_c[t + 1] - cum_c[lo]) / (t - lo + 1)
            roll_s[t] = (cum_s[t + 1] - cum_s[lo]) / (t - lo + 1)
        roll_h = np.arctan2(roll_s, roll_c)
        out[idx] = np.where(roll_h < 0, roll_h + 2 * np.pi, roll_h)
    return (out / (2 * np.pi) * 8).astype(int) % 8


def two_way_anova_unit(h_u: np.ndarray, pos_bin: np.ndarray, traj: np.ndarray):
    """Pure-numpy 2-way ANOVA for one unit. Returns (F_int, p_int, eta2_int).

    Uses the type-I sum-of-squares decomposition: SS_total = SS_pos + SS_traj +
    SS_int + SS_resid; F_int = (SS_int / df_int) / (SS_resid / df_resid).
    """
    # Drop combinations that are too rare
    pairs = pos_bin * 100 + traj
    unique_pairs, counts = np.unique(pairs, return_counts=True)
    valid = pairs.copy()
    too_rare = unique_pairs[counts < 5]
    keep = ~np.isin(pairs, too_rare)
    if keep.sum() < 100:
        return np.nan, 1.0, 0.0
    h_u = h_u[keep]
    pos_bin = pos_bin[keep]
    traj = traj[keep]

    n = len(h_u)
    grand = h_u.mean()
    ss_total = ((h_u - grand) ** 2).sum()

    # Marginal means for pos
    pos_means = {}
    for p in np.unique(pos_bin):
        m = pos_bin == p
        pos_means[p] = h_u[m].mean()
    ss_pos = sum((pos_means[p] - grand) ** 2 * (pos_bin == p).sum()
                  for p in pos_means)

    traj_means = {}
    for t in np.unique(traj):
        m = traj == t
        traj_means[t] = h_u[m].mean()
    ss_traj = sum((traj_means[t] - grand) ** 2 * (traj == t).sum()
                   for t in traj_means)

    # Cell means
    cell_means = {}
    for p in np.unique(pos_bin):
        for t in np.unique(traj):
            m = (pos_bin == p) & (traj == t)
            if m.sum() > 0:
                cell_means[(p, t)] = h_u[m].mean()

    ss_cells = 0.0
    for (p, t), mu_pt in cell_means.items():
        m = (pos_bin == p) & (traj == t)
        ss_cells += (mu_pt - grand) ** 2 * m.sum()
    ss_int = ss_cells - ss_pos - ss_traj

    # Residual
    pred = np.zeros_like(h_u)
    for (p, t), mu_pt in cell_means.items():
        m = (pos_bin == p) & (traj == t)
        pred[m] = mu_pt
    ss_resid = ((h_u - pred) ** 2).sum()

    df_pos = len(np.unique(pos_bin)) - 1
    df_traj = len(np.unique(traj)) - 1
    df_int = df_pos * df_traj
    df_resid = n - df_pos - df_traj - df_int - 1
    if df_resid < 1 or df_int < 1 or ss_resid < 1e-9:
        return np.nan, 1.0, 0.0
    F_int = (ss_int / df_int) / (ss_resid / df_resid)
    p_int = 1.0 - st.f.cdf(F_int, df_int, df_resid)
    eta2_int = ss_int / ss_total if ss_total > 1e-9 else 0.0
    return float(F_int), float(p_int), float(eta2_int)


def splitter_fraction(H: np.ndarray, pos_bin: np.ndarray, traj: np.ndarray,
                       alpha: float = 0.01):
    """Run 2-way ANOVA per unit; FDR-correct p-values; return splitter stats."""
    n_units = H.shape[1]
    Fs = np.zeros(n_units)
    ps = np.ones(n_units)
    eta2s = np.zeros(n_units)
    for u in range(n_units):
        F, p, e = two_way_anova_unit(H[:, u], pos_bin, traj)
        Fs[u] = F if not np.isnan(F) else 0.0
        ps[u] = p
        eta2s[u] = e
    rej, _, _, _ = multipletests(ps, alpha=alpha, method="fdr_bh")
    return {
        "fraction": float(rej.mean()),
        "n_significant": int(rej.sum()),
        "n_total": int(n_units),
        "median_eta2_split": float(np.median(eta2s[rej])) if rej.any() else 0.0,
        "mean_eta2_all": float(np.mean(eta2s)),
        "median_eta2_all": float(np.median(eta2s)),
        "F_p99": float(np.quantile(Fs[~np.isnan(Fs)], 0.99)),
    }


def analyse_one_condition(npz_path: Path, n_max: int = 60000) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    pos = d["positions"].astype(np.float32)
    head = d["headings"].astype(np.float32)
    eps_id = d["episode_ids"]
    sip = d["step_in_episode"]

    # Subsample for speed
    if len(h) > n_max:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(h), n_max, replace=False))
        h, pos, head, eps_id, sip = h[idx], pos[idx], head[idx], eps_id[idx], sip[idx]

    # Standardise hidden states unit-wise
    mu = h.mean(0, keepdims=True)
    sd = h.std(0, keepdims=True) + 1e-6
    h = (h - mu) / sd

    pos_bin = discretise_pos(pos, n_bins=8)
    out = {}
    # Trajectory feature 1: previous heading octant (instantaneous direction)
    traj_dir = heading_octant(head)
    print(f"  trajectory feature: prev_dir_octant ({len(np.unique(traj_dir))} classes)")
    out["prev_dir"] = splitter_fraction(h, pos_bin, traj_dir)
    # Trajectory feature 2: rolling 5-step direction octant
    traj_roll = rolling_dir_octant(head, eps_id, sip, window=5)
    print(f"  trajectory feature: rolling_5_dir_octant ({len(np.unique(traj_roll))} classes)")
    out["roll5_dir"] = splitter_fraction(h, pos_bin, traj_roll)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/splitter_results.json")
    ap.add_argument("--n_max", type=int, default=60000)
    args = ap.parse_args()

    results = {}
    for cond, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f"  MISSING {path}; skip {cond}")
            continue
        print(f"\n=== condition: {cond} ===")
        results[cond] = analyse_one_condition(path, n_max=args.n_max)
        for feat_name, r in results[cond].items():
            print(f"  {feat_name}: splitter fraction = {r['fraction']:.3f} "
                   f"({r['n_significant']}/{r['n_total']}) "
                   f"median eta^2 (sig units) = {r['median_eta2_split']:.4f}")

    json.dump(results, open(args.out_path, "w"), indent=2)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
