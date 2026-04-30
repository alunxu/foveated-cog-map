"""
100M same-frame anchor: linear+MLP probe on ckpt 20 NPZs for all 4 conds.

Tests claim: rank ordering Blind > Coarse > Foveated > Uniform on linear R²
is preserved when comparing AT THE SAME FRAME COUNT (~100M), not just at
respective convergence frames.

Also computes LOSO R² for each cond at 100M, to confirm the scene-invariance
finding is consistent at 100M (not just at convergence).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")


CONDS = [
    ("blind",    "/tmp/cond_npzs_100M/blind_gibson_ckpt20_det.npz"),
    ("coarse",   "/tmp/cond_npzs_100M/matched_gibson_ckpt20_det.npz"),
    ("foveated", "/tmp/cond_npzs_100M/foveated_gibson_ckpt20_det.npz"),
    ("uniform",  "/tmp/cond_npzs_100M/uniform_gibson_ckpt20_det.npz"),
]


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
    def forward(self, x): return self.net(x)


def fit_mlp(X_tr, y_tr, X_te, y_te, hidden=256, l2=1e-4, epochs=50, bs=512, lr=1e-3):
    m = MLP(X_tr.shape[1], y_tr.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=l2)
    Xt = torch.tensor(X_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_te, dtype=torch.float32)
    n = len(Xt)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            yhat = m(Xt[b])
            loss = nn.MSELoss()(yhat, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): yhp = m(Xv).numpy()
    return r2_score(y_te, yhp, multioutput="uniform_average")


def main():
    out = {}
    for cond, path in CONDS:
        p = Path(path)
        if not p.exists():
            print(f"SKIP {cond}: {p} missing"); continue
        print(f"\n=== {cond} ===")
        d = np.load(p)
        h = d["hidden_states"].astype(np.float32)
        gps = d["gps"].astype(np.float32)
        ep_ids = d["episode_ids"]
        scene_ids = d["scene_ids"]

        if len(h) > 20000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(h), 20000, replace=False)
            h = h[idx]; gps = gps[idx]; ep_ids = ep_ids[idx]; scene_ids = scene_ids[idx]
        h = h - h.mean(axis=0, keepdims=True)
        gps = gps - gps.mean(axis=0, keepdims=True)

        # Episode-level 5-fold CV
        unique_eps = np.unique(ep_ids)
        rng = np.random.default_rng(42); rng.shuffle(unique_eps)
        folds = np.array_split(unique_eps, 5)
        lin_r2, mlp_r2 = [], []
        for te_eps in folds:
            tr_mask = ~np.isin(ep_ids, te_eps); te_mask = np.isin(ep_ids, te_eps)
            ridge = Ridge(alpha=10.0).fit(h[tr_mask], gps[tr_mask])
            lin_r2.append(r2_score(gps[te_mask], ridge.predict(h[te_mask]),
                                    multioutput="uniform_average"))
            mlp_r2.append(fit_mlp(h[tr_mask], gps[tr_mask], h[te_mask], gps[te_mask]))

        # LOSO CV on top scenes
        unique_scenes, counts = np.unique(scene_ids, return_counts=True)
        idx_sorted = np.argsort(counts)[::-1]
        top_scenes = [s for s in unique_scenes[idx_sorted[:30]]
                      if (scene_ids == s).sum() >= 100]
        loso_r2 = []
        for s_test in top_scenes:
            te = scene_ids == s_test
            tr = ~te
            ridge = Ridge(alpha=10.0).fit(h[tr], gps[tr])
            loso_r2.append(r2_score(gps[te], ridge.predict(h[te]),
                                     multioutput="uniform_average"))
        loso_r2 = np.array(loso_r2)

        out[cond] = {
            "linear_r2_mean": float(np.mean(lin_r2)),
            "linear_r2_std": float(np.std(lin_r2)),
            "mlp_r2_mean": float(np.mean(mlp_r2)),
            "mlp_r2_std": float(np.std(mlp_r2)),
            "loso_median": float(np.median(loso_r2)),
            "loso_mean": float(np.mean(loso_r2)),
            "loso_std": float(np.std(loso_r2)),
            "loso_frac_neg": float(np.mean(loso_r2 < 0)),
            "n_loso_scenes": int(len(loso_r2)),
            "n_samples": int(len(h)),
        }
        print(f"  Linear: {np.mean(lin_r2):+.3f} ± {np.std(lin_r2):.3f}")
        print(f"  MLP:    {np.mean(mlp_r2):+.3f} ± {np.std(mlp_r2):.3f}")
        print(f"  LOSO median: {np.median(loso_r2):+.3f}, frac<0: {np.mean(loso_r2 < 0):.0%}")

    Path("/tmp/extra_analyses/results_100M.json").write_text(json.dumps(out, indent=2))
    print("\nwrote /tmp/extra_analyses/results_100M.json")


if __name__ == "__main__":
    main()
