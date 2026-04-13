"""
Cross-condition analysis for Habitat navigation agents.

Compares representations across conditions (blind, uniform, foveated,
matched-compute) using CKA and cross-condition probe transfer.

Experiments:
  Phase 3 — H2: Representational Divergence
    3a. Position-aligned CKA between conditions (per-layer)
        - Bins agent states by (scene, discretized-position) so that CKA
          compares hidden states from the *same spatial context*, not random
          unaligned samples.
        - Split-half within-condition CKA as upper-bound control.
        - Permutation null (shuffled rows) as lower-bound control.
    3b. Cross-condition probe transfer (train on A, test on B)
    3c. Comparative summary table

Usage:
    python scripts/probing/analyze_cross.py \
        --data blind=/path/blind.npz uniform=/path/uniform.npz \
              foveated=/path/foveated.npz matched=/path/matched.npz \
        --out /path/cross_analysis.json
"""

import argparse
import json
import os
import sys
import itertools

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.probing import fit_probe, prepare_features


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Cross-condition representation analysis")
    p.add_argument("--data", nargs="+", required=True,
                   help="Condition data in format: name=/path/to/data.npz")
    p.add_argument("--out", required=True, help="Output .json path")
    p.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Max samples per condition for CKA (memory bound)")
    p.add_argument("--grid-res", type=float, default=0.5,
                   help="Spatial bin size in meters for position-aligned CKA")
    p.add_argument("--min-bin-count", type=int, default=2,
                   help="Min samples per spatial bin in both conditions")
    p.add_argument("--n-permutations", type=int, default=100,
                   help="Number of permutations for CKA null distribution")
    return p.parse_args()


def load_condition(spec):
    """Parse 'name=/path/to.npz' and load data."""
    name, path = spec.split("=", 1)
    data = np.load(path, allow_pickle=True)
    return name, data


def linear_cka(X, Y):
    """Compute linear CKA between two feature matrices.

    Kornblith et al. (ICML 2019):
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    X: (n, p) and Y: (n, q) must have the same number of samples,
    and rows must be aligned (same sample identity).
    """
    # Center columns
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices via dot products
    XtX = X.T @ X  # (p, p)
    YtY = Y.T @ Y  # (q, q)
    YtX = Y.T @ X  # (q, p)

    num = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    if denom < 1e-12:
        return 0.0
    return float(num / denom)


# ═══════════════════════════════════════════════════════════════════════
#  Spatial binning for position-aligned CKA
# ═══════════════════════════════════════════════════════════════════════

def _make_bin_key(gps, grid_res):
    """Discretize a 2-D GPS coordinate into a bin tuple."""
    return (int(np.floor(gps[0] / grid_res)),
            int(np.floor(gps[1] / grid_res)))


def align_by_position(data_a, data_b, feature_key, grid_res, min_count, max_samples, seed):
    """Align two datasets by spatial bin, returning paired feature matrices.

    For each spatial bin that has >= min_count samples in BOTH conditions,
    randomly pick one sample from each condition. This ensures row i in
    the returned X_a and X_b corresponds to the same spatial context.

    Returns:
        X_a: (n_aligned, D_a)
        X_b: (n_aligned, D_b)
        n_bins_used: int
    """
    rng = np.random.RandomState(seed)

    gps_a = data_a["gps"]
    gps_b = data_b["gps"]
    H_a = data_a[feature_key]
    H_b = data_b[feature_key]

    # Handle per-layer features: (N, n_layers, D) → select layer
    if H_a.ndim == 3:
        # Will be called per-layer from the caller
        pass

    # Build bin → indices maps
    bins_a = {}
    for i in range(len(gps_a)):
        k = _make_bin_key(gps_a[i], grid_res)
        bins_a.setdefault(k, []).append(i)

    bins_b = {}
    for i in range(len(gps_b)):
        k = _make_bin_key(gps_b[i], grid_res)
        bins_b.setdefault(k, []).append(i)

    # Find shared bins with enough samples
    shared_bins = set(bins_a.keys()) & set(bins_b.keys())
    valid_bins = [k for k in shared_bins
                  if len(bins_a[k]) >= min_count and len(bins_b[k]) >= min_count]

    if not valid_bins:
        return None, None, 0

    # Sample one from each condition per bin
    idx_a, idx_b = [], []
    for k in valid_bins:
        idx_a.append(rng.choice(bins_a[k]))
        idx_b.append(rng.choice(bins_b[k]))

    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)

    # Cap at max_samples
    if len(idx_a) > max_samples:
        sel = rng.choice(len(idx_a), max_samples, replace=False)
        idx_a = idx_a[sel]
        idx_b = idx_b[sel]

    return H_a[idx_a], H_b[idx_b], len(valid_bins)


# ═══════════════════════════════════════════════════════════════════════
#  3a. Position-aligned CKA between conditions
# ═══════════════════════════════════════════════════════════════════════

def compute_pairwise_cka(conditions, max_samples, seed, grid_res, min_bin_count, n_permutations):
    """Compute position-aligned CKA for all pairs of conditions.

    For each pair:
      1. Align samples by spatial bin (scene-agnostic GPS discretization).
      2. Compute CKA on aligned rows → cross-condition CKA.
      3. Split-half within each condition → upper-bound control CKA.
      4. Permutation null → shuffled-row CKA (lower-bound baseline).
    """
    names = list(conditions.keys())
    results = {}
    rng = np.random.RandomState(seed)

    for a, b in itertools.combinations(names, 2):
        pair_key = f"{a}_vs_{b}"
        data_a, data_b = conditions[a], conditions[b]

        # ── Position-aligned cross-condition CKA (top layer h) ──
        X_a, X_b, n_bins = align_by_position(
            data_a, data_b, "hidden_states",
            grid_res, min_bin_count, max_samples, seed,
        )

        if X_a is None or len(X_a) < 50:
            results[pair_key] = {
                "cka_aligned": None,
                "n_aligned": 0 if X_a is None else len(X_a),
                "n_bins": n_bins,
                "error": "too few aligned samples (<50)",
            }
            continue

        cka_cross = linear_cka(X_a, X_b)

        # ── Permutation null: shuffle rows of X_b ──
        perm_ckas = []
        for _ in range(n_permutations):
            perm_idx = rng.permutation(len(X_b))
            perm_ckas.append(linear_cka(X_a, X_b[perm_idx]))
        perm_mean = float(np.mean(perm_ckas))
        perm_std = float(np.std(perm_ckas))
        perm_p = float(np.mean(np.array(perm_ckas) >= cka_cross))

        # ── Split-half within-condition controls ──
        split_half = {}
        for name, data in [(a, data_a), (b, data_b)]:
            H = data["hidden_states"]
            n = len(H)
            if n < 100:
                split_half[name] = None
                continue
            idx = rng.permutation(n)
            half = n // 2
            cka_sh = linear_cka(H[idx[:half]], H[idx[half:half * 2]])
            split_half[name] = float(cka_sh)

        # ── Per-layer CKA (position-aligned) ──
        layer_ckas = {}
        if "h_layers" in data_a and "h_layers" in data_b:
            n_layers = min(data_a["h_layers"].shape[1], data_b["h_layers"].shape[1])
            for li in range(n_layers):
                # We need to re-align using the same bin logic but extract per-layer
                # Use the same aligned indices (GPS-based) but pull layer features
                gps_a, gps_b = data_a["gps"], data_b["gps"]
                # Rebuild aligned indices
                rng_layer = np.random.RandomState(seed)
                bins_a_map = {}
                for i in range(len(gps_a)):
                    k = _make_bin_key(gps_a[i], grid_res)
                    bins_a_map.setdefault(k, []).append(i)
                bins_b_map = {}
                for i in range(len(gps_b)):
                    k = _make_bin_key(gps_b[i], grid_res)
                    bins_b_map.setdefault(k, []).append(i)
                shared = set(bins_a_map.keys()) & set(bins_b_map.keys())
                valid = [k for k in shared
                         if len(bins_a_map[k]) >= min_bin_count and len(bins_b_map[k]) >= min_bin_count]
                if len(valid) < 50:
                    continue
                ia, ib = [], []
                for k in valid:
                    ia.append(rng_layer.choice(bins_a_map[k]))
                    ib.append(rng_layer.choice(bins_b_map[k]))
                ia, ib = np.array(ia), np.array(ib)
                if len(ia) > max_samples:
                    sel = rng_layer.choice(len(ia), max_samples, replace=False)
                    ia, ib = ia[sel], ib[sel]

                ha = data_a["h_layers"][ia, li, :]
                hb = data_b["h_layers"][ib, li, :]
                layer_ckas[f"h_layer_{li}"] = linear_cka(ha, hb)

                ca = data_a["c_layers"][ia, li, :]
                cb = data_b["c_layers"][ib, li, :]
                layer_ckas[f"c_layer_{li}"] = linear_cka(ca, cb)

        results[pair_key] = {
            "cka_aligned": cka_cross,
            "n_aligned": int(len(X_a)),
            "n_bins": n_bins,
            "permutation_null_mean": perm_mean,
            "permutation_null_std": perm_std,
            "permutation_p_value": perm_p,
            "n_permutations": n_permutations,
            "split_half_cka": split_half,
            "layer_ckas": layer_ckas,
            "grid_res_m": grid_res,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
#  3b. Cross-condition probe transfer
# ═══════════════════════════════════════════════════════════════════════

def cross_condition_probe_transfer(conditions, alpha, seed):
    """Train GPS probe on condition A, test on condition B.

    If representations are similar, a probe trained on A should decode
    GPS from B's hidden states. Low transfer R² indicates divergent codes.
    """
    names = list(conditions.keys())
    results = {}

    for a in names:
        data_a = conditions[a]
        H_a = data_a["hidden_states"]
        gps_a = data_a.get("gps")
        if gps_a is None:
            continue

        # Fit scaler and probe on A
        scaler = StandardScaler()
        H_a_scaled = scaler.fit_transform(H_a)
        reg = Ridge(alpha=alpha)
        reg.fit(H_a_scaled, gps_a)

        # Self-test (sanity)
        pred_self = reg.predict(H_a_scaled)
        r2_self = float(r2_score(gps_a, pred_self, multioutput="uniform_average"))

        for b in names:
            if a == b:
                results[f"train_{a}_test_{b}"] = {"r2": r2_self, "type": "self"}
                continue

            data_b = conditions[b]
            H_b = data_b["hidden_states"]
            gps_b = data_b.get("gps")
            if gps_b is None:
                continue

            # Apply A's scaler to B's hidden states and predict
            H_b_scaled = scaler.transform(H_b)
            pred_b = reg.predict(H_b_scaled)
            r2_transfer = float(r2_score(gps_b, pred_b, multioutput="uniform_average"))

            results[f"train_{a}_test_{b}"] = {"r2": r2_transfer, "type": "transfer"}

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Load all conditions
    conditions = {}
    for spec in args.data:
        name, data = load_condition(spec)
        conditions[name] = data
        print(f"Loaded {name}: {len(data['hidden_states'])} steps, "
              f"dim={data['hidden_states'].shape[1]}")

    results = {"conditions": list(conditions.keys())}

    # 3a. Position-aligned pairwise CKA
    print(f"\n{'═'*60}")
    print(f"  3a. Position-aligned CKA (grid_res={args.grid_res}m)")
    print(f"{'═'*60}")
    cka_results = compute_pairwise_cka(
        conditions, args.max_samples, args.seed,
        args.grid_res, args.min_bin_count, args.n_permutations,
    )
    results["3a_cka"] = cka_results
    for pair, res in cka_results.items():
        if res.get("error"):
            print(f"  {pair}: {res['error']}")
            continue
        print(f"  {pair}:")
        print(f"    CKA (aligned)  = {res['cka_aligned']:.4f}  "
              f"(n={res['n_aligned']}, bins={res['n_bins']})")
        print(f"    Perm null      = {res['permutation_null_mean']:.4f} "
              f"± {res['permutation_null_std']:.4f}  "
              f"(p={res['permutation_p_value']:.3f})")
        for name, sh in res["split_half_cka"].items():
            if sh is not None:
                print(f"    Split-half({name}) = {sh:.4f}")
        for lk, lv in res["layer_ckas"].items():
            print(f"    {lk}: {lv:.4f}")

    # 3b. Cross-condition probe transfer
    print(f"\n{'═'*60}")
    print("  3b. Cross-condition probe transfer (GPS)")
    print(f"{'═'*60}")
    transfer_results = cross_condition_probe_transfer(conditions, args.alpha, args.seed)
    results["3b_probe_transfer"] = transfer_results
    for key, res in sorted(transfer_results.items()):
        marker = "●" if res["type"] == "self" else "○"
        print(f"  {marker} {key}: R²={res['r2']:+.4f}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
