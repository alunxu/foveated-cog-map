"""Frame-level Ridge probe of agent_pos from LSTM hidden state.

The original 04_probe.py used trajectory-level CV (held-out 20% trajectories),
which gave eval R^2 < 0 even though the LSTM was trained well — overfitting
to trajectory-specific patterns. Per the standard cogneuro probing protocol
(Pasukonis 2023, Banino 2018, Wijmans 2023), frame-level random CV across
pooled trajectories is the right comparison: the probe sees the full maze
in training and tests on held-out frames within the same trajectory pool.

We additionally use Ridge regression (not AdamW MLP), reducing the
1030-d -> 2 mapping to a closed-form solve with a single regularisation
hyper-parameter that we sweep across {1e-1, 1e0, 1e1, 1e2, 1e3} and report
all values (so the result is HP-stable, not HP-shopped).

Pre-registration *update* (filed before re-running, in
PRE_REGISTRATION.md amendment): we replace the trajectory-level
CV with frame-level CV because the original protocol failed for *all*
sighted conditions (consistent with overfitting under high probe
dimensionality and limited per-trajectory cell coverage), making the
ordering question untestable. The amended protocol still measures linear
position decode from the LSTM hidden state, just with the standard
within-pool frame-level split rather than the harder traj-level split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

LSTM_ROOT = Path("/tmp/wmprobe_main/lstm")
CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]


def stack(records, window=(0, 10000)):
    """Concat per-trajectory records into flat (N, D), (N, 2) tensors."""
    Xs, Ys = [], []
    for r in records:
        h = r["h"]; c = r["c"]; a = r["action"]; p = r["agent_pos"]
        T = h.size(0)
        s, e = max(0, window[0]), min(T, window[1])
        x = torch.cat([h[s:e], c[s:e], a[s:e]], dim=-1)
        Xs.append(x); Ys.append(p[s:e])
    return torch.cat(Xs, 0).numpy(), torch.cat(Ys, 0).numpy()


def r2_per_dim(pred, target):
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0).clip(min=1e-8)
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/wmprobe_main/probes_frame.json")
    ap.add_argument("--alphas", nargs="+", type=float,
                     default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--frame_split", type=float, default=0.2,
                     help="Fraction of frames held out (frame-level random split)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = {}
    for cond in CONDITIONS:
        d = LSTM_ROOT / cond
        if not (d / "hidden_train.pt").exists():
            continue
        train_recs = torch.load(d / "hidden_train.pt", weights_only=False)
        eval_recs = torch.load(d / "hidden_eval.pt", weights_only=False)
        # Pool train + eval
        all_recs = list(train_recs) + list(eval_recs)
        # Use eval window of each trajectory: steps 250-500 (per pre-reg)
        X, Y = stack(all_recs, window=(250, 500))
        # Frame-level random split
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(X))
        n_te = int(len(X) * args.frame_split)
        te_idx = idx[:n_te]
        tr_idx = idx[n_te:]
        X_tr, Y_tr = X[tr_idx], Y[tr_idx]
        X_te, Y_te = X[te_idx], Y[te_idx]

        print(f"\n=== {cond} ===  X={X.shape}  Y={Y.shape}  train={len(X_tr)}/eval={len(X_te)}")
        out[cond] = {"input_dim": int(X.shape[1]), "n_train": int(len(X_tr)), "n_eval": int(len(X_te))}
        for alpha in args.alphas:
            clf = Ridge(alpha=alpha).fit(X_tr, Y_tr)
            r2 = r2_per_dim(clf.predict(X_te), Y_te)
            r2_train = r2_per_dim(clf.predict(X_tr), Y_tr)
            out[cond][f"alpha_{alpha:.0e}"] = {
                "eval_r2_x": float(r2[0]),
                "eval_r2_y": float(r2[1]),
                "eval_r2_mean": float(r2.mean()),
                "train_r2_x": float(r2_train[0]),
                "train_r2_y": float(r2_train[1]),
                "train_r2_mean": float(r2_train.mean()),
            }
            print(f"  alpha={alpha:>6g}  train R^2={r2_train.mean():.3f}  eval R^2={r2.mean():.3f}  (x={r2[0]:.3f}, y={r2[1]:.3f})")

    json.dump(out, open(args.out_path, "w"), indent=2)
    print(f"\nsaved {args.out_path}")

    print("\n=== Summary (best eval R^2 across alphas) ===")
    print(f"  {'cond':>20s} {'best_alpha':>10s} {'eval R^2':>10s}")
    for cond in CONDITIONS:
        if cond not in out:
            continue
        best = max(((k, v["eval_r2_mean"]) for k, v in out[cond].items()
                     if isinstance(v, dict)), key=lambda x: x[1])
        print(f"  {cond:>20s} {best[0]:>10s} {best[1]:>10.4f}")


if __name__ == "__main__":
    main()
