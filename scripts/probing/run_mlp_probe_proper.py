"""
Proper MLP probe sweep with hidden=256, L2=1e-4, 5-fold episode-level CV.

Matches the protocol described in main.tex Appendix probe-details, which
gave foveated R² = 0.51, uniform = 0.55. Re-run for ALL 4 conds to get
clean (linear, MLP) pairs for information-conservation analysis.

Reads:  /tmp/cond_npzs/{cond}_gibson_det.npz
Writes: /tmp/mlp_probe_proper.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


CONDS = [
    ("blind",    "blind_gibson_det.npz"),
    ("coarse",   "matched_gibson_det.npz"),
    ("foveated", "foveated_gibson_det.npz"),
    ("uniform",  "uniform_gibson_det.npz"),
]


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)


def fit_mlp_probe(X_train, y_train, X_test, y_test, hidden=256, l2=1e-4,
                  epochs=50, batch_size=512, lr=1e-3, device="cpu"):
    """Fit 2-layer MLP, return test R² (averaged over targets)."""
    model = MLP(X_train.shape[1], y_train.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_test, dtype=torch.float32, device=device)
    n = len(Xt)
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            batch = idx[i:i+batch_size]
            yhat = model(Xt[batch])
            loss = loss_fn(yhat, yt[batch])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        yhat_test = model(Xv).cpu().numpy()
    return r2_score(y_test, yhat_test, multioutput="uniform_average")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("/tmp/cond_npzs"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/mlp_probe_proper.json"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=20000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    results = {}
    for cond, fname in CONDS:
        p = args.in_dir / fname
        if not p.exists():
            print(f"MISSING {p}"); continue
        print(f"\n=== {cond} ===")
        d = np.load(p)
        h = d["hidden_states"].astype(np.float32)
        gps = d["gps"].astype(np.float32)
        ep_ids = d["episode_ids"]
        # Subsample
        if len(h) > args.max_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(h), args.max_samples, replace=False)
            h = h[idx]; gps = gps[idx]; ep_ids = ep_ids[idx]
        # Mean-center h (linear probe baseline)
        h = h - h.mean(axis=0, keepdims=True)
        gps = gps - gps.mean(axis=0, keepdims=True)
        # 5-fold episode-level CV
        unique_eps = np.unique(ep_ids)
        rng = np.random.default_rng(42)
        rng.shuffle(unique_eps)
        folds = np.array_split(unique_eps, args.n_folds)

        linear_r2_per_fold = []
        mlp_r2_per_fold = []
        for fi, test_eps in enumerate(folds):
            train_mask = ~np.isin(ep_ids, test_eps)
            test_mask = np.isin(ep_ids, test_eps)
            X_tr, y_tr = h[train_mask], gps[train_mask]
            X_te, y_te = h[test_mask], gps[test_mask]
            # Linear (Ridge alpha=10)
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr, y_tr)
            lin_r2 = r2_score(y_te, ridge.predict(X_te), multioutput="uniform_average")
            # MLP (hidden=256, L2=1e-4)
            mlp_r2 = fit_mlp_probe(X_tr, y_tr, X_te, y_te,
                                    hidden=256, l2=1e-4,
                                    epochs=50, batch_size=512, lr=1e-3,
                                    device=device)
            linear_r2_per_fold.append(lin_r2)
            mlp_r2_per_fold.append(mlp_r2)
            print(f"  fold {fi}: linear={lin_r2:+.3f}, mlp={mlp_r2:+.3f}")
        lin_mean = float(np.mean(linear_r2_per_fold))
        lin_std = float(np.std(linear_r2_per_fold))
        mlp_mean = float(np.mean(mlp_r2_per_fold))
        mlp_std = float(np.std(mlp_r2_per_fold))
        results[cond] = {
            "linear_r2_mean": lin_mean, "linear_r2_std": lin_std,
            "mlp_r2_mean": mlp_mean, "mlp_r2_std": mlp_std,
            "n_samples": int(len(h)), "n_folds": args.n_folds,
        }
        print(f"  → linear: {lin_mean:+.3f} ± {lin_std:.3f}")
        print(f"  → MLP:    {mlp_mean:+.3f} ± {mlp_std:.3f}")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
