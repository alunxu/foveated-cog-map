"""
B3: Subspace evolution over training — cluster-side analysis.

For each (cond, ckpt) NPZ in /scratch/izar/wxu/probing_data/, computes:
  - Top-K PCA basis (K covering 90% variance, capped at 30)
  - Ridge probe weight β for GPS direction

Then computes pairwise principal angles + position-direction cosines for
all cross-cond pairs at EACH training stage. Outputs a single JSON with
the full (cond_A, cond_B, ckpt_t) → (angle, cos) mapping.

Plot locally afterwards: angle/cos evolution vs training frames, one
line per cond pair. Predicts subspaces DIVERGE during training (early =
overlap, late = orthogonal), supporting "capacity reallocation as
training-time process".

Reads:  /scratch/izar/wxu/probing_data/<cond>_gibson_ckpt<N>_det.npz
Writes: /scratch/izar/wxu/probing_results/subspace_evolution.json

Usage (on Izar):
    python scripts/probing/run_subspace_evolution.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --out /scratch/izar/wxu/probing_results/subspace_evolution.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


# Frames per ckpt (per-cond convention from main paper)
FRAMES_PER_CKPT = {
    "blind":    10.06e6,   # 342M / 34
    "matched":  5.10e6,    # 250M / 49
    "foveated": 4.83e6,    # 174M / 36
    "uniform":  5.10e6,    # 250M / 49
}

CONDS = ["blind", "matched", "foveated", "uniform"]


def load_h_gps(npz_path: Path, max_samples: int = 20000):
    """Load top-layer h + GPS labels, mean-center h."""
    d = np.load(npz_path, allow_pickle=False)
    h = d["hidden_states"].astype(np.float32)
    gps = d["gps"].astype(np.float32)
    # Subsample
    if len(h) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(h), max_samples, replace=False)
        h = h[idx]
        gps = gps[idx]
    h = h - h.mean(axis=0, keepdims=True)
    gps = gps - gps.mean(axis=0, keepdims=True)
    return h, gps


def fit_pca_basis(h: np.ndarray, variance_threshold: float = 0.90,
                  K_cap: int = 30) -> tuple[np.ndarray, int]:
    """Return (D × K) basis matrix and K."""
    pca = PCA(n_components=min(50, h.shape[1]))
    pca.fit(h)
    cum = np.cumsum(pca.explained_variance_ratio_)
    K = int(np.searchsorted(cum, variance_threshold) + 1)
    K = min(K, K_cap)
    return pca.components_[:K].T, K  # (D, K)


def fit_pos_direction(h: np.ndarray, gps: np.ndarray,
                       alpha: float = 10.0) -> np.ndarray:
    """Return (2, D) Ridge weight on GPS."""
    ridge = Ridge(alpha=alpha)
    ridge.fit(h, gps)
    return ridge.coef_  # (2, D)


def principal_angles_deg(B_A: np.ndarray, B_B: np.ndarray) -> float:
    """Mean principal angle in degrees between subspaces of B_A and B_B."""
    Q_A, _ = np.linalg.qr(B_A)
    Q_B, _ = np.linalg.qr(B_B)
    M = Q_A.T @ Q_B
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(s))))


def pos_dir_cosine(beta_A: np.ndarray, beta_B: np.ndarray) -> float:
    """Mean cosine between position-direction vectors (averaged over x and z)."""
    cos_x = float(np.dot(beta_A[0], beta_B[0]) /
                  (np.linalg.norm(beta_A[0]) * np.linalg.norm(beta_B[0]) + 1e-9))
    cos_z = float(np.dot(beta_A[1], beta_B[1]) /
                  (np.linalg.norm(beta_A[1]) * np.linalg.norm(beta_B[1]) + 1e-9))
    return 0.5 * (cos_x + cos_z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=20000)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pat = re.compile(r"(blind|matched|foveated|uniform)_gibson_ckpt(\d+)_det\.npz$")
    files: list[tuple[str, int, Path]] = []
    for p in sorted(args.in_dir.glob("*_det.npz")):
        m = pat.match(p.name)
        if not m:
            continue
        cond, ckpt = m.group(1), int(m.group(2))
        files.append((cond, ckpt, p))
    print(f"found {len(files)} (cond, ckpt) NPZs")

    # Step 1: per-(cond, ckpt) compute PCA basis + GPS direction
    bases: dict[tuple[str, int], np.ndarray] = {}
    pos_dirs: dict[tuple[str, int], np.ndarray] = {}
    K_used: dict[tuple[str, int], int] = {}
    for cond, ckpt, p in files:
        print(f"  loading {cond} ckpt{ckpt}...")
        try:
            h, gps = load_h_gps(p, max_samples=args.max_samples)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue
        basis, K = fit_pca_basis(h)
        bases[(cond, ckpt)] = basis
        K_used[(cond, ckpt)] = K
        pos_dirs[(cond, ckpt)] = fit_pos_direction(h, gps)
        print(f"    K={K}, h.shape={h.shape}")

    # Step 2: pairwise principal angles + cosines per ckpt-stage
    # Strategy: for each cond pair (A, B), for each ckpt of A and matching ckpt of B,
    # report angle. We use ckpt index (not absolute frames) because conds use different
    # frames-per-ckpt. We bin by relative training stage (early/mid/late/final).
    # Output: list of per-pair-per-stage entries.
    rows = []
    for (cA, kA), basis_A in bases.items():
        for (cB, kB), basis_B in bases.items():
            if cA == cB:
                continue
            # Only output one direction (cA <= cB alphabetically) to halve output size
            if cA >= cB:
                continue
            # Match: same ckpt index? Use k that exists in both. Closest match.
            # For now, keep all (kA, kB) pairs — let post-processing align.
            angle = principal_angles_deg(basis_A, basis_B)
            cos = pos_dir_cosine(pos_dirs[(cA, kA)], pos_dirs[(cB, kB)])
            frames_A = (kA + 1) * FRAMES_PER_CKPT[cA] / 1e6  # M frames
            frames_B = (kB + 1) * FRAMES_PER_CKPT[cB] / 1e6
            rows.append({
                "cond_A": cA, "ckpt_A": kA, "frames_A_M": round(frames_A, 1),
                "cond_B": cB, "ckpt_B": kB, "frames_B_M": round(frames_B, 1),
                "K_A": K_used[(cA, kA)], "K_B": K_used[(cB, kB)],
                "principal_angle_deg": angle,
                "pos_dir_cos": cos,
            })

    # Save
    out = {
        "n_pairs": len(rows),
        "max_samples": args.max_samples,
        "frames_per_ckpt_M": {k: round(v / 1e6, 2) for k, v in FRAMES_PER_CKPT.items()},
        "rows": rows,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out} ({len(rows)} pair entries)")


if __name__ == "__main__":
    main()
