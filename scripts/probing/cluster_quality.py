"""
Cluster-quality metrics for the H2 representational divergence claim.

We claim in §4.3 that the five conditions occupy "non-overlapping
regions" of the t-SNE embedding (visual confirmation of near-zero CKA).
This script quantifies that claim with two off-the-shelf metrics:

  - Silhouette score (sklearn): how well-separated are the clusters?
    Range [-1, +1]; higher = better separation.  Computed on the raw
    top-layer LSTM hidden states pooled across conditions, treating
    each condition as a cluster label.

  - 1-NN purity: for each pooled hidden state, find its nearest
    neighbour by Euclidean distance among the OTHER pooled samples; ask
    whether that neighbour came from the same condition.  Proportion
    correct = 1-NN purity.  Random baseline = 1/5 = 0.20.

Both metrics expect cluster-like geometry; high values indicate the
representational subspaces really are disjoint (not merely near-zero
in CKA).

Usage:
    python scripts/probing/cluster_quality.py \\
        --in-dir /scratch/izar/wxu/probing_data \\
        --suffix _det \\
        --out /scratch/izar/wxu/probing_results/cluster_quality_det.json \\
        --n-per-cond 1500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


CONDITIONS = [
    "blind_gibson",
    "uniform_gibson",
    "foveated_gibson",
    "foveated_learned_gibson",
    "matched_gibson",
]


def load_top_layer(path: Path) -> np.ndarray:
    d = np.load(path)
    if "h_layers" in d.files:
        # h_layers shape: (T, n_layers, hidden) -> top layer
        h = d["h_layers"]
        return h[:, -1, :]
    return d["hidden_states"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-cond", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    Xs, ys, labels = [], [], []
    for i, cond in enumerate(CONDITIONS):
        p = args.in_dir / f"{cond}{args.suffix}.npz"
        if not p.exists():
            print(f"[skip] {p}")
            continue
        H = load_top_layer(p).astype(np.float32)
        n = min(args.n_per_cond, H.shape[0])
        idx = rng.choice(H.shape[0], n, replace=False)
        Xs.append(H[idx])
        ys.append(np.full(n, i, dtype=np.int32))
        labels.append(cond)
        print(f"  loaded {cond}: sampled {n} of {H.shape[0]}")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(f"\nPooled X: {X.shape}, y: {y.shape}, n_clusters: {len(labels)}")

    # ---- Silhouette ----
    # On 5 × 1500 = 7500 samples this is fine in seconds.
    sil = float(silhouette_score(X, y, metric="euclidean", random_state=args.seed))
    print(f"\nSilhouette (Euclidean): {sil:+.4f}")
    print("  random-baseline silhouette: ~0; perfect separation: +1; overlapping: <0")

    # ---- 1-NN purity ----
    nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    nbrs.fit(X)
    _, nn_idx = nbrs.kneighbors(X)
    # nn_idx[:, 0] is the point itself; nn_idx[:, 1] is its 1-NN.
    pred = y[nn_idx[:, 1]]
    purity = float((pred == y).mean())
    print(f"\n1-NN purity: {purity:.4f}")
    print(f"  random-baseline 1-NN purity: {1.0/len(labels):.4f}")
    print("  perfect separation: 1.00")

    # Per-condition breakdown
    per_cond = {}
    for i, cond in enumerate(labels):
        mask = y == i
        n = int(mask.sum())
        correct = int((pred[mask] == y[mask]).sum())
        per_cond[cond] = {"n": n, "correct": correct, "purity": correct / n}
        print(f"  {cond}: {correct}/{n} = {correct/n:.4f}")

    out = {
        "n_per_cond": args.n_per_cond,
        "n_total": int(X.shape[0]),
        "n_dim": int(X.shape[1]),
        "conditions": labels,
        "silhouette_euclidean": sil,
        "one_nn_purity_overall": purity,
        "one_nn_purity_per_cond": per_cond,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
