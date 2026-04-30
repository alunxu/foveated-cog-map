"""
Cross-cond probe-transfer matrix.

For each pair (cond_A, cond_B): train Ridge regression on h_2 of cond_A
predicting GPS, then test on h_2 of cond_B. Diagonal entries = own-probe
R² (already known); off-diagonal = transfer R² (small/negative if
subspaces orthogonal).

Reads:  /tmp/cond_npzs/{cond}_gibson_det.npz
Writes: /tmp/extra_analyses/cross_cond_transfer.json
        docs/manuscript/fig/fig_cross_cond_transfer.pdf
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

CONDS = [
    ("blind",     "/tmp/cond_npzs/blind_gibson_det.npz"),
    ("coarse",    "/tmp/cond_npzs/matched_gibson_det.npz"),
    ("foveated",  "/tmp/cond_npzs/foveated_gibson_det.npz"),
    ("uniform",   "/tmp/cond_npzs/uniform_gibson_det.npz"),
]


def load(p, n_max=10000, seed=0):
    d = np.load(p)
    h = d["hidden_states"].astype(np.float32)
    gps = d["gps"].astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(h) > n_max:
        idx = rng.choice(len(h), n_max, replace=False)
        h = h[idx]; gps = gps[idx]
    return h, gps


def main():
    cond_data = {c: load(p) for c, p in CONDS}
    results = {}
    matrix = np.zeros((4, 4))
    for i, (cA, pA) in enumerate(CONDS):
        hA, gpsA = cond_data[cA]
        # 80/20 train/test on cond A (but probe is trained on full A)
        hA_centered = hA - hA.mean(axis=0, keepdims=True)
        gpsA_centered = gpsA - gpsA.mean(axis=0, keepdims=True)
        ridge = Ridge(alpha=10.0).fit(hA_centered, gpsA_centered)
        for j, (cB, pB) in enumerate(CONDS):
            hB, gpsB = cond_data[cB]
            # Use the test set of cond B (same probe, applied to cond B)
            # Mean-center using cond A's stats (= the probe's training stats)
            hB_centered = hB - hA.mean(axis=0, keepdims=True)  # use A's mean
            gpsB_centered = gpsB - gpsA.mean(axis=0, keepdims=True)
            yhat = ridge.predict(hB_centered)
            r2 = r2_score(gpsB_centered, yhat, multioutput="uniform_average")
            matrix[i, j] = r2
            results[f"{cA}->{cB}"] = float(r2)
            print(f"  probe({cA})  →  test({cB})  R² = {r2:+.3f}")

    # Save results
    out = {
        "matrix": matrix.tolist(),
        "row_labels": [c for c, _ in CONDS],
        "col_labels": [c for c, _ in CONDS],
        "by_pair": results,
    }
    Path("/tmp/extra_analyses").mkdir(exist_ok=True)
    Path("/tmp/extra_analyses/cross_cond_transfer.json").write_text(json.dumps(out, indent=2))
    print("wrote /tmp/extra_analyses/cross_cond_transfer.json")
    return matrix


if __name__ == "__main__":
    main()
