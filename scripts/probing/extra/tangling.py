"""Tangling Q (Russo et al. 2018, Neuron) — population-dynamics tangling metric.

For two state-space points x(t), x(t') with derivatives x_dot(t), x_dot(t'):
    Q(t) = max_{t'} ||x_dot(t) - x_dot(t')||^2 / (||x(t) - x(t')||^2 + eps)

A high Q at point t means the derivative is multivalued there: the trajectory
"cannot have come from an autonomous dynamical system" because nearby states
are evolving in different directions. Russo et al. showed motor cortex
activity has lower Q than the muscle commands it generates — i.e., M1 has
*untangled* the muscle output.

For us, Q answers: how autonomous is the recurrent code per condition? Pure-
input agents (where every step's h is determined by the new input) should
have HIGH Q (input pushes nearby h's in different directions). Pure-recurrent
agents (where recurrence dominates) should have LOW Q (smooth flow field).

Pre-registered prediction (filed in tangling_russo.md):
  blind: low Q (most autonomous, dynamics dominate)
  coarse: low-medium
  foveated / uniform: medium-high (rich input drives fresh derivatives)
  fov-logpolar: medium

Adds a consumption-axis test complementary to the TGM result: blind has slow
basis rotation (TGM diagonal) AND smooth flow (low Q) means it lives on an
autonomous dynamical-system manifold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/tmp/rcp_analysis_v3")
FILES = {
    "blind": "blind_izar_det_RCP.npz",
    "coarse": "coarse_det_RCP.npz",
    "foveated": "foveated_det_RCP.npz",
    "foveated_logpolar": "foveated_logpolar_det_RCP.npz",
    "uniform": "uniform_det_RCP.npz",
}


def compute_deltas(h: np.ndarray, ep_id: np.ndarray, sip: np.ndarray):
    """Return delta_h[t] = h[t+1] - h[t] within episode; valid mask for non-boundary."""
    deltas = np.zeros_like(h)
    valid = np.zeros(len(h), dtype=bool)
    eps_unique = np.unique(ep_id)
    # Sort by ep then sip
    order = np.lexsort((sip, ep_id))
    h_o, ep_o, sip_o = h[order], ep_id[order], sip[order]
    n = len(h_o)
    for k in range(n - 1):
        if ep_o[k] == ep_o[k + 1] and sip_o[k + 1] - sip_o[k] == 1:
            deltas[order[k]] = h_o[k + 1] - h_o[k]
            valid[order[k]] = True
    return deltas, valid


def tangling_q(h: np.ndarray, deltas: np.ndarray, max_pairs: int = 50000,
                seed: int = 0) -> dict:
    """Sample random pairs and report Q distribution."""
    rng = np.random.default_rng(seed)
    n = len(h)
    # Random pairs of indices
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    same = i == j
    if same.any():
        j[same] = (j[same] + 1) % n
    h_diff = h[i] - h[j]
    d_diff = deltas[i] - deltas[j]
    h_dist2 = (h_diff ** 2).sum(-1)
    d_dist2 = (d_diff ** 2).sum(-1)
    Q = d_dist2 / (h_dist2 + 1e-6)
    return {
        "median": float(np.median(Q)),
        "p25": float(np.quantile(Q, 0.25)),
        "p75": float(np.quantile(Q, 0.75)),
        "p95": float(np.quantile(Q, 0.95)),
        "mean": float(np.mean(Q)),
        "n_pairs": int(max_pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/tangling_results.json")
    ap.add_argument("--max_pairs", type=int, default=200000)
    ap.add_argument("--n_pcs", type=int, default=0,
                     help="If >0, PCA-reduce hidden states first.")
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
        # Standardise
        mu = h.mean(0, keepdims=True); sd = h.std(0, keepdims=True) + 1e-6
        h = (h - mu) / sd
        if args.n_pcs > 0:
            from sklearn.decomposition import PCA
            h = PCA(n_components=args.n_pcs, random_state=0).fit_transform(h).astype(np.float32)

        deltas, valid = compute_deltas(h, ep_id, sip)
        h_v = h[valid]
        d_v = deltas[valid]
        print(f"\n=== {cond} ===  n_valid={len(h_v)}  d_dim={h_v.shape[1]}")
        result = tangling_q(h_v, d_v, max_pairs=args.max_pairs)
        out[cond] = result
        print(f"  Q: median={result['median']:.4f}  mean={result['mean']:.4f}  "
               f"p25={result['p25']:.4f}  p75={result['p75']:.4f}  p95={result['p95']:.4f}")

    json.dump(out, open(args.out_path, "w"), indent=2)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
