"""Cross-condition GPS probe-transfer 5×5: train Ridge on cond X's
hidden states, test on cond Y's.  Computes the 25-cell matrix used in
the §E format-divergence appendix figure.

Runs on RCP via env vars:
  PROBE_NPZ_DIR (default /tmp/cond_npzs)
  PROBE_RESULTS_OUT (default /tmp/extra_analyses)

Writes: <out>/probe_transfer_5x5.json with the 5×5 matrix.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

NPZ_DIR = os.environ.get("PROBE_NPZ_DIR", "/tmp/cond_npzs")
RESULTS_OUT = os.environ.get("PROBE_RESULTS_OUT", "/tmp/extra_analyses")

CONDS = [
    ("blind",             f"{NPZ_DIR}/blind_izar_det.npz"),
    ("coarse",            f"{NPZ_DIR}/coarse_det.npz"),
    ("foveated_logpolar", f"{NPZ_DIR}/foveated_logpolar_det.npz"),
    ("foveated",          f"{NPZ_DIR}/foveated_det.npz"),
    ("uniform",           f"{NPZ_DIR}/uniform_det.npz"),
]
N_SUB = 30000   # subsample steps per condition for speed


def load(cond, path):
    d = np.load(path)
    h = d["hidden_states"].astype(np.float32)
    gps = d["gps"].astype(np.float32)
    n = len(h)
    if n > N_SUB:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, N_SUB, replace=False)
        h = h[idx]; gps = gps[idx]
    return h, gps


def main():
    Path(RESULTS_OUT).mkdir(exist_ok=True, parents=True)
    print(f"Loading {len(CONDS)} conditions ...")
    data = {}
    for cond, path in CONDS:
        if not Path(path).exists():
            print(f"  MISSING {cond}: {path}")
            continue
        h, gps = load(cond, path)
        # Center per-condition (probe is fit on centered features)
        h = h - h.mean(axis=0, keepdims=True)
        gps = gps - gps.mean(axis=0, keepdims=True)
        data[cond] = (h, gps)
        print(f"  {cond}: {h.shape}")

    keys = [c[0] for c in CONDS if c[0] in data]
    M = np.full((len(keys), len(keys)), np.nan)
    for i, kx in enumerate(keys):
        hx, gx = data[kx]
        ridge = Ridge(alpha=10.0).fit(hx, gx)
        for j, ky in enumerate(keys):
            hy, gy = data[ky]
            r2 = float(r2_score(gy, ridge.predict(hy),
                                multioutput="uniform_average"))
            M[i, j] = r2
            print(f"  train {kx} → test {ky}: R^2 = {r2:+.3f}")

    out = {
        "conds": keys,
        "matrix": M.tolist(),
    }
    Path(f"{RESULTS_OUT}/probe_transfer_5x5.json").write_text(
        json.dumps(out, indent=2))
    print(f"wrote {RESULTS_OUT}/probe_transfer_5x5.json")


if __name__ == "__main__":
    main()
