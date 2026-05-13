"""
Grid-cell signature analysis for LSTM hidden units.

Tests whether any units in the top-layer LSTM develop hexagonally-periodic
spatial tuning — the defining signature of entorhinal grid cells in mammals.

Method (Barry et al. 2007 / Hafting et al. 2005):
  1. Build a 2D firing-rate map: mean unit activation binned over (x, z)
     position, then smoothed with a Gaussian kernel.
  2. Compute the 2D spatial autocorrelogram (SAC) of each rate map.
  3. Rotate the SAC at 30°, 60°, 90°, 120°, 150° and measure Pearson r
     within an annular mask that excludes the central peak.
  4. Gridness score = min(r60, r120) − max(r30, r90, r150).
     Hexagonal symmetry (60°/120° matches) gives positive scores;
     square/random maps give near-zero or negative scores.
  5. Shuffle null: circular-shift unit activations along time; repeat
     steps 1–4.  A unit is "grid-like" if its gridness > 99th-pct null.

Per-condition output:
  - Gridness score distribution (all units × scenes)
  - Fraction of grid-like units and their peak-gridness values
  - Mean ± std gridness vs. shuffle baseline
  - Top-10 unit indices by gridness (for figure generation)
  - Per-unit rate-map and SAC saved to JSON when --save-maps is set

Usage (local / Izar interactive):
    python scripts/probing/grid_cell_signature.py \\
        --in-dir /scratch/izar/$USER/probing_data \\
        --conds blind_gibson uniform_gibson foveated_gibson \\
        --out results/probing/grid_cell_signature.json

Reads:  <in-dir>/<cond>.npz  (output of scripts/probing/collect.py)
Writes: <out> JSON
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate as ndrotate
from scipy.signal import correlate2d

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────

RATE_MAP_BINS    = 24      # spatial bins per axis
SMOOTH_SIGMA     = 1.5     # Gaussian smoothing (bins)
MIN_OCCUPANCY    = 3       # min timesteps per spatial bin
MIN_SCENE_STEPS  = 500     # min steps per scene to include
N_TOP_SCENES     = 10      # scenes to aggregate per condition
N_SHUFFLES       = 200     # circular-shift shuffles for null
NULL_PERCENTILE  = 99      # significance cutoff
GRIDNESS_FLOOR   = 0.0     # minimum gridness to be "grid-like" (standard)

# Annulus bounds as fraction of SAC half-width — excludes central peak,
# includes first ring of off-centre peaks where grid structure appears.
ANNULUS_RMIN_FRAC = 0.18
ANNULUS_RMAX_FRAC = 0.70


# ── rate-map helpers ───────────────────────────────────────────────────────────

def build_rate_map(activations: np.ndarray, positions: np.ndarray,
                   n_bins: int = RATE_MAP_BINS,
                   min_occ: int = MIN_OCCUPANCY) -> np.ndarray | None:
    """Smoothed 2D rate map for one unit.

    Returns (n_bins, n_bins) float, NaN where unvisited, or None if
    spatial coverage is too sparse.
    """
    xz = positions[:, [0, 2]]
    x_min, z_min = xz.min(0)
    x_max, z_max = xz.max(0)
    if x_max == x_min or z_max == z_min:
        return None

    x_bin = np.clip(
        ((xz[:, 0] - x_min) / (x_max - x_min) * n_bins).astype(int), 0, n_bins - 1)
    z_bin = np.clip(
        ((xz[:, 1] - z_min) / (z_max - z_min) * n_bins).astype(int), 0, n_bins - 1)

    occupancy = np.zeros((n_bins, n_bins), dtype=np.float64)
    sum_act   = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(occupancy, (x_bin, z_bin), 1.0)
    np.add.at(sum_act,   (x_bin, z_bin), activations.astype(np.float64))

    visited = occupancy >= min_occ
    if visited.sum() < n_bins:       # less than one full row covered — skip
        return None

    rate_map = np.full((n_bins, n_bins), np.nan)
    rate_map[visited] = sum_act[visited] / occupancy[visited]

    # Fill NaN bins with scene mean before Gaussian smoothing so the kernel
    # does not propagate NaNs into occupied bins.
    fill_val = np.nanmean(rate_map)
    filled = np.where(np.isnan(rate_map), fill_val, rate_map)
    smoothed = gaussian_filter(filled, sigma=SMOOTH_SIGMA)

    # Restore NaN mask after smoothing.
    smoothed[~visited] = np.nan
    return smoothed


def spatial_autocorrelogram(rate_map: np.ndarray) -> np.ndarray | None:
    """Normalised 2D spatial autocorrelogram (SAC).

    The zero-lag peak is normalised to 1.  Unvisited (NaN) bins are
    replaced by the scene mean before correlation.

    Returns None if the rate map has zero variance.
    """
    filled = rate_map.copy()
    fill_val = np.nanmean(filled)
    filled = np.where(np.isnan(filled), fill_val, filled)

    r = filled - filled.mean()
    if r.std() < 1e-10:
        return None

    sac = correlate2d(r, r, mode='full')
    zero_lag = sac[sac.shape[0] // 2, sac.shape[1] // 2]
    if abs(zero_lag) < 1e-10:
        return None
    return sac / zero_lag


def _annulus_mask(shape: tuple[int, int],
                  rmin_frac: float = ANNULUS_RMIN_FRAC,
                  rmax_frac: float = ANNULUS_RMAX_FRAC) -> np.ndarray:
    H, W = shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return (dist >= max_r * rmin_frac) & (dist <= max_r * rmax_frac)


def compute_gridness(sac: np.ndarray,
                     rmin_frac: float = ANNULUS_RMIN_FRAC,
                     rmax_frac: float = ANNULUS_RMAX_FRAC) -> float:
    """Gridness score from a spatial autocorrelogram.

    Rotates the SAC at 30°, 60°, 90°, 120°, 150° and correlates each
    with the original within an annular mask.

      Gridness = min(r₆₀, r₁₂₀) − max(r₃₀, r₉₀, r₁₅₀)

    Hexagonal grid → r₆₀, r₁₂₀ ≈ high → positive score.
    Returns NaN if the SAC is too flat to score.
    """
    mask = _annulus_mask(sac.shape, rmin_frac, rmax_frac)
    if mask.sum() < 30:
        return np.nan

    ref = sac[mask]
    if ref.std() < 1e-10:
        return np.nan

    r_at: dict[int, float] = {}
    for angle in (30, 60, 90, 120, 150):
        rot = ndrotate(sac, angle=angle, reshape=False, order=1)
        vals = rot[mask]
        if vals.std() < 1e-10:
            return np.nan
        r_at[angle] = float(np.corrcoef(ref, vals)[0, 1])

    return min(r_at[60], r_at[120]) - max(r_at[30], r_at[90], r_at[150])


# ── per-condition analysis ─────────────────────────────────────────────────────

def analyse_condition(
    H: np.ndarray,
    positions: np.ndarray,
    scene_ids: np.ndarray,
    n_bins: int = RATE_MAP_BINS,
    n_shuffles: int = N_SHUFFLES,
    n_top_scenes: int = N_TOP_SCENES,
    min_scene_steps: int = MIN_SCENE_STEPS,
    save_maps: bool = False,
) -> dict:
    """Grid-cell signature for one condition.

    Scores every unit across the top scenes, computes a shuffle null,
    and returns a dict ready for JSON serialisation.
    """
    hidden_dim = H.shape[1]

    # Select scenes with best coverage.
    unique_scenes, scene_counts = np.unique(scene_ids, return_counts=True)
    order = np.argsort(scene_counts)[::-1]
    top_scenes = [
        int(s) for s, c in zip(unique_scenes[order], scene_counts[order])
        if c >= min_scene_steps
    ][:n_top_scenes]

    if not top_scenes:
        return {"error": f"no scene with >= {min_scene_steps} steps", "n_scenes": 0}

    print(f"    {len(top_scenes)} scenes  ×  {hidden_dim} units")

    g_per_scene: list[np.ndarray] = []
    null_samples: list[float] = []
    top_unit_maps: dict = {}

    for si, scn in enumerate(top_scenes):
        mask   = scene_ids == scn
        H_sc   = H[mask]
        pos_sc = positions[mask]

        g_sc = np.full(hidden_dim, np.nan)
        for ui in range(hidden_dim):
            rmap = build_rate_map(H_sc[:, ui], pos_sc, n_bins=n_bins)
            if rmap is None:
                continue
            sac = spatial_autocorrelogram(rmap)
            if sac is None:
                continue
            g_sc[ui] = compute_gridness(sac)

        g_per_scene.append(g_sc)

        # Shuffle null — circularly shift activations, subsample 32 units.
        rng = np.random.default_rng(42 + si)
        sample_ui = rng.choice(hidden_dim, size=min(32, hidden_dim), replace=False)
        for _ in range(n_shuffles):
            shift = rng.integers(50, max(51, len(H_sc) - 50))
            for ui in sample_ui:
                rmap = build_rate_map(np.roll(H_sc[:, ui], shift), pos_sc, n_bins=n_bins)
                if rmap is None:
                    continue
                sac = spatial_autocorrelogram(rmap)
                if sac is None:
                    continue
                g = compute_gridness(sac)
                if not np.isnan(g):
                    null_samples.append(float(g))

        n_scored = int((~np.isnan(g_sc)).sum())
        print(f"      scene {si+1}/{len(top_scenes)}: {n_scored} units scored")

    if not g_per_scene:
        return {"error": "no scoreable units in any scene", "n_scenes": 0}

    # Mean gridness per unit across scenes.
    g_matrix = np.array(g_per_scene)          # (n_scenes, hidden_dim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_g = np.nanmean(g_matrix, axis=0) # (hidden_dim,)

    null_arr  = np.array(null_samples)
    null_p99  = float(np.percentile(null_arr, NULL_PERCENTILE)) if len(null_arr) else np.nan
    threshold = max(GRIDNESS_FLOOR, null_p99 if not np.isnan(null_p99) else 0.0)

    is_gridlike = (~np.isnan(mean_g)) & (mean_g > threshold)
    n_gridlike  = int(is_gridlike.sum())
    n_scoreable = int((~np.isnan(mean_g)).sum())

    # Top-10 by gridness.
    valid   = np.where(~np.isnan(mean_g))[0]
    top10   = valid[np.argsort(mean_g[valid])[::-1][:10]]

    # Save rate maps / SACs for top-5 units in first scene, if requested.
    if save_maps and len(top10) > 0:
        sc0_mask = scene_ids == top_scenes[0]
        for ui in top10[:5]:
            rmap = build_rate_map(H[sc0_mask, ui], positions[sc0_mask], n_bins=n_bins)
            if rmap is not None:
                sac = spatial_autocorrelogram(rmap)
                top_unit_maps[int(ui)] = {
                    "rate_map": rmap.tolist(),
                    "sac": sac.tolist() if sac is not None else None,
                    "gridness": float(mean_g[ui]),
                }

    out = {
        "n_scenes":       len(top_scenes),
        "n_units":        int(hidden_dim),
        "n_scoreable":    int(n_scoreable),
        "n_gridlike":     n_gridlike,
        "frac_gridlike":  float(n_gridlike / max(n_scoreable, 1)),
        "mean_gridness":  float(np.nanmean(mean_g)),
        "std_gridness":   float(np.nanstd(mean_g)),
        "median_gridness":float(np.nanmedian(mean_g)),
        "p90_gridness":   float(np.nanpercentile(mean_g, 90)),
        "max_gridness":   float(np.nanmax(mean_g)),
        "null_p99":       float(null_p99),
        "null_mean":      float(np.mean(null_arr)) if len(null_arr) else None,
        "null_std":       float(np.std(null_arr))  if len(null_arr) else None,
        "threshold_used": float(threshold),
        "top10_units":    [{"unit": int(u), "gridness": float(mean_g[u])} for u in top10],
        "gridness_per_unit": mean_g.tolist(),
        "null_distribution": null_arr[:500].tolist(),
    }
    if save_maps:
        out["top_unit_maps"] = top_unit_maps
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-cell signature analysis for LSTM units")
    p.add_argument("--in-dir",  required=True,
                   help="Directory containing <cond>.npz files from collect.py")
    p.add_argument("--conds", nargs="+",
                   default=["blind_gibson", "uniform_gibson",
                            "foveated_gibson", "foveated_logpolar_gibson"],
                   help="Condition name(s) matching <cond>.npz in --in-dir")
    p.add_argument("--out", required=True,
                   help="Output JSON, e.g. results/probing/grid_cell_signature.json")
    p.add_argument("--n-bins",        type=int,   default=RATE_MAP_BINS)
    p.add_argument("--n-shuffles",    type=int,   default=N_SHUFFLES)
    p.add_argument("--n-top-scenes",  type=int,   default=N_TOP_SCENES)
    p.add_argument("--min-scene-steps", type=int, default=MIN_SCENE_STEPS)
    p.add_argument("--layer", type=int, default=-1,
                   help="LSTM layer index: -1=top (default), 0=bottom, 1=mid")
    p.add_argument("--save-maps", action="store_true",
                   help="Embed rate maps and SACs for top units in JSON output")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    in_dir = Path(args.in_dir)
    out_p  = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    for cond in args.conds:
        npz = in_dir / f"{cond}.npz"
        if not npz.exists():
            print(f"\n[{cond}] {npz} not found — skipping")
            results[cond] = {"error": "npz not found"}
            continue

        print(f"\n{'='*60}\n  {cond}\n{'='*60}")
        data = np.load(npz, allow_pickle=True)

        # Use the requested LSTM layer (h state only).
        if "h_layers" in data:
            H = data["h_layers"][:, args.layer, :]   # (N, hidden_dim)
        else:
            H = data["hidden_states"]                  # (N, hidden_dim)

        positions = data["positions"]
        scene_ids = data["scene_ids"]
        N, D = H.shape
        print(f"  N={N}, hidden_dim={D}, scenes={len(np.unique(scene_ids))}")

        res = analyse_condition(
            H=H, positions=positions, scene_ids=scene_ids,
            n_bins=args.n_bins, n_shuffles=args.n_shuffles,
            n_top_scenes=args.n_top_scenes,
            min_scene_steps=args.min_scene_steps,
            save_maps=args.save_maps,
        )
        results[cond] = res

        if "error" not in res:
            print(f"\n  Scoreable: {res['n_scoreable']}/{res['n_units']}  "
                  f"Grid-like: {res['n_gridlike']} ({res['frac_gridlike']*100:.1f}%)  "
                  f"[thresh={res['threshold_used']:+.3f}]")
            print(f"  Mean gridness: {res['mean_gridness']:+.4f} ± {res['std_gridness']:.4f}  "
                  f"Max: {res['max_gridness']:+.4f}  Null p99: {res['null_p99']:+.4f}")
            print(f"  Top-3 units: {[u['unit'] for u in res['top10_units'][:3]]}")

    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_p}")


if __name__ == "__main__":
    main()
