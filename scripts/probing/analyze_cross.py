"""
Cross-condition analysis for Habitat navigation agents.

Compares representations across conditions (blind, uniform, foveated,
matched-compute) using CKA and cross-condition probe transfer.

Experiments:
  Phase 3 — H2: Representational Divergence
    3a. Linear CKA between conditions (per-layer)
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

    X: (n, p) and Y: (n, q) must have the same number of samples.
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


def subsample(arrays, n, seed):
    """Subsample multiple arrays to n rows (same indices)."""
    rng = np.random.RandomState(seed)
    N = len(arrays[0])
    if N <= n:
        return arrays
    idx = rng.choice(N, n, replace=False)
    return [a[idx] for a in arrays]


# ═══════════════════════════════════════════════════════════════════════
#  3a. CKA between conditions
# ═══════════════════════════════════════════════════════════════════════

def compute_pairwise_cka(conditions, max_samples, seed):
    """Compute CKA for all pairs of conditions.

    Since agents take different trajectories, we cannot align by timestep.
    Instead, we subsample hidden states from each condition and compute
    CKA on the (unaligned) feature matrices. This measures whether the
    *geometry* of the representation space is similar, not whether
    specific inputs produce similar outputs.

    For a fairer comparison, we also compute CKA after aligning by
    binned (scene, position) when possible.
    """
    names = list(conditions.keys())
    results = {}

    for a, b in itertools.combinations(names, 2):
        pair_key = f"{a}_vs_{b}"
        data_a, data_b = conditions[a], conditions[b]

        # Top-layer h
        H_a = data_a["hidden_states"]
        H_b = data_b["hidden_states"]

        # Subsample to same size
        n = min(len(H_a), len(H_b), max_samples)
        [H_a_sub] = subsample([H_a], n, seed)
        [H_b_sub] = subsample([H_b], n, seed + 1)

        cka_top = linear_cka(H_a_sub, H_b_sub)

        # Per-layer CKA if available
        layer_ckas = {}
        if "h_layers" in data_a and "h_layers" in data_b:
            hl_a, hl_b = data_a["h_layers"], data_b["h_layers"]
            n_layers = min(hl_a.shape[1], hl_b.shape[1])
            for li in range(n_layers):
                ha = hl_a[:, li, :]
                hb = hl_b[:, li, :]
                n_l = min(len(ha), len(hb), max_samples)
                [ha_sub] = subsample([ha], n_l, seed)
                [hb_sub] = subsample([hb], n_l, seed + 1)
                layer_ckas[f"h_layer_{li}"] = linear_cka(ha_sub, hb_sub)

            # Cell states
            cl_a, cl_b = data_a["c_layers"], data_b["c_layers"]
            for li in range(n_layers):
                ca = cl_a[:, li, :]
                cb = cl_b[:, li, :]
                n_l = min(len(ca), len(cb), max_samples)
                [ca_sub] = subsample([ca], n_l, seed)
                [cb_sub] = subsample([cb], n_l, seed + 1)
                layer_ckas[f"c_layer_{li}"] = linear_cka(ca_sub, cb_sub)

        results[pair_key] = {
            "cka_top_layer": cka_top,
            "n_samples": n,
            "layer_ckas": layer_ckas,
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

    # 3a. Pairwise CKA
    print(f"\n{'═'*60}")
    print("  3a. Pairwise CKA (linear)")
    print(f"{'═'*60}")
    cka_results = compute_pairwise_cka(conditions, args.max_samples, args.seed)
    results["3a_cka"] = cka_results
    for pair, res in cka_results.items():
        print(f"  {pair}: CKA={res['cka_top_layer']:.4f} (n={res['n_samples']})")
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
