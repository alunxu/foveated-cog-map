"""
Standalone unaligned linear-CKA across conditions.

Motivation
----------
``scripts/probing/analyze_cross.py`` uses position-aligned CKA, which
buckets hidden states by spatial bin and only computes CKA if >50 bins
are shared. With 500-episode probing samples (≈1.9k steps per
condition), that threshold is almost never met. Unaligned linear CKA
(Kornblith et al., 2019) operates on arbitrary-order samples and is the
standard representational-similarity metric; we use it as the primary
metric for H2 pending more probe data.

This script:
  1. Loads ``h_layers[:, -1]`` (top-layer hidden) from each
     ``{condition}.npz``.
  2. Computes pairwise linear CKA on matched numbers of rows
     (min-length truncation — each condition contributes the same
     number of rows).
  3. Writes a 5×5 CKA matrix JSON.

Usage
-----
    python scripts/probing/unaligned_cka.py \\
        --data blind=/path/to/blind.npz ... \\
        --out /path/to/cka_unaligned.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between feature matrices X, Y (both shape (n, d_i))."""
    assert X.shape[0] == Y.shape[0], f"row mismatch: {X.shape} vs {Y.shape}"
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # Frobenius norms of cross/inner products
    hsic_xy = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Yc.T @ Yc, ord="fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return float("nan")
    return float(hsic_xy / denom)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="name=/path/to/cond.npz (one per condition)",
    )
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument(
        "--layer",
        default=-1,
        type=int,
        help="which LSTM layer to compare (default: -1 = top layer)",
    )
    args = p.parse_args()

    # Load
    feats: dict[str, np.ndarray] = {}
    for entry in args.data:
        name, _, path = entry.partition("=")
        if not path:
            raise ValueError(f"expected name=path, got {entry!r}")
        d = np.load(path)
        H = d["h_layers"][:, args.layer]  # (N, hidden)
        feats[name] = H.astype(np.float32)
        print(f"[load] {name}: {H.shape[0]} steps, hidden={H.shape[1]}")

    # Truncate to common N for fair unaligned comparison
    common_n = min(H.shape[0] for H in feats.values())
    print(f"[truncate] common N = {common_n}")
    rng = np.random.default_rng(0)
    for name, H in feats.items():
        idx = rng.permutation(H.shape[0])[:common_n]
        feats[name] = H[idx]

    # Compute pairwise CKA
    names = list(feats.keys())
    matrix = {}
    for i, a in enumerate(names):
        matrix[a] = {}
        for j, b in enumerate(names):
            if i == j:
                matrix[a][b] = 1.0
                continue
            v = linear_cka(feats[a], feats[b])
            matrix[a][b] = v
            print(f"  CKA({a}, {b}) = {v:.4f}")

    out = {
        "conditions": names,
        "layer_index": args.layer,
        "n_samples_per_condition": common_n,
        "cka_matrix": matrix,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
