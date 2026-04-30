"""
C1: Predictive-horizon probe (Stachenfeld 2017 SR / Whittington 2020 TEM).

Tests whether h_t encodes FUTURE positions (x_{t+k}, y_{t+k}) for
k = 0, 1, 5, 10, 20, 50 steps ahead. Following successor representation
theory, a place-cell-like code at s_t encodes a vector of expected
discounted future occupancies; under linear regression, this manifests
as a linear map from h_t to expected (x_{t+k}, y_{t+k}) decaying with k.

Predictions:
- Bottleneck conds (Blind, Coarse): integrated GPS code is smooth in
  time; h_t linearly decodes near-future positions; R²(k) decays slowly.
- Rich-encoder conds (Uniform): reactive to current visual input;
  h_t encodes mostly current state; R²(k) drops fast at k=1 or stays
  low for all k.

We use BOTH linear (Ridge α=10) and MLP (hidden=256, L²=1e-4) probes,
because uniform's linear R²(k=0) is already negative — linear signal
would be uninformative for uniform across all k.

DEFERRED: foveated condition (174M is below SPL plateau; will run with
seed0pre 250M-effective tomorrow).

Reads:  /tmp/cond_npzs/{blind,matched,uniform}_gibson_det.npz
Writes: /tmp/extra_analyses/predictive_horizon.json
        docs/manuscript/fig/fig_predictive_horizon.pdf
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/scripts/paper_figures")
from _style import apply_paper_style
apply_paper_style()


# Defer foveated; run on 3 properly-converged conds
CONDS = [
    ("blind",    "/tmp/cond_npzs/blind_gibson_det.npz",     "Blind",    "#444444"),
    ("coarse",   "/tmp/cond_npzs/matched_gibson_det.npz",   "Coarse",   "#377eb8"),
    ("uniform",  "/tmp/cond_npzs/uniform_gibson_det.npz",   "Uniform",  "#4daf4a"),
]
HORIZONS = [0, 1, 5, 10, 20, 50]


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
    def forward(self, x): return self.net(x)


def fit_mlp(X_tr, y_tr, X_te, y_te, hidden=256, l2=1e-4, epochs=30, bs=512, lr=1e-3):
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
    return r2_score(y_te, yhp, multioutput='uniform_average')


def build_pairs(h, gps, ep_ids, k):
    """For each episode, build (h_t, gps_{t+k}) pairs respecting boundaries."""
    pairs_h = []
    pairs_g = []
    pairs_ep = []
    for ep in np.unique(ep_ids):
        mask = ep_ids == ep
        ep_h = h[mask]
        ep_g = gps[mask]
        T = len(ep_h)
        if T <= k + 5:
            continue
        # h_t for t=0..T-k-1 paired with gps_{t+k}
        pairs_h.append(ep_h[:T - k])
        pairs_g.append(ep_g[k:])
        pairs_ep.extend([ep] * (T - k))
    if not pairs_h:
        return None, None, None
    return np.concatenate(pairs_h), np.concatenate(pairs_g), np.array(pairs_ep)


def main():
    Path("/tmp/extra_analyses").mkdir(exist_ok=True)
    results = {}

    for cond, path, label, color in CONDS:
        print(f"\n=== {cond} ===")
        d = np.load(Path(path))
        h_full = d["hidden_states"].astype(np.float32)
        gps_full = d["gps"].astype(np.float32)
        ep_ids_full = d["episode_ids"]

        # Subsample episodes for speed: keep all samples within ~200 episodes
        unique_eps = np.unique(ep_ids_full)
        rng = np.random.default_rng(0)
        kept_eps = rng.choice(unique_eps, min(200, len(unique_eps)), replace=False)
        ep_mask = np.isin(ep_ids_full, kept_eps)
        h_full = h_full[ep_mask]
        gps_full = gps_full[ep_mask]
        ep_ids_full = ep_ids_full[ep_mask]
        print(f"  {len(np.unique(ep_ids_full))} episodes, {len(h_full)} samples")

        results[cond] = {"label": label, "color": color, "horizons": HORIZONS,
                         "linear_r2": [], "mlp_r2": [], "n_samples": []}

        for k in HORIZONS:
            h_k, g_k, ep_k = build_pairs(h_full, gps_full, ep_ids_full, k)
            if h_k is None:
                results[cond]["linear_r2"].append(None)
                results[cond]["mlp_r2"].append(None)
                results[cond]["n_samples"].append(0)
                continue

            # Mean-center
            h_k = h_k - h_k.mean(axis=0, keepdims=True)
            g_k = g_k - g_k.mean(axis=0, keepdims=True)

            # Episode-level 5-fold CV
            unique_test_eps = np.unique(ep_k)
            rng2 = np.random.default_rng(42); rng2.shuffle(unique_test_eps)
            folds = np.array_split(unique_test_eps, 5)
            lin_r2 = []; mlp_r2 = []
            for te_eps in folds:
                tr_mask = ~np.isin(ep_k, te_eps); te_mask = np.isin(ep_k, te_eps)
                Xtr, ytr = h_k[tr_mask], g_k[tr_mask]
                Xte, yte = h_k[te_mask], g_k[te_mask]
                # Linear (Ridge)
                rg = Ridge(alpha=10.0).fit(Xtr, ytr)
                lin_r2.append(r2_score(yte, rg.predict(Xte), multioutput="uniform_average"))
                # MLP
                mlp_r2.append(fit_mlp(Xtr, ytr, Xte, yte))

            lin_mean = float(np.mean(lin_r2))
            mlp_mean = float(np.mean(mlp_r2))
            results[cond]["linear_r2"].append(lin_mean)
            results[cond]["mlp_r2"].append(mlp_mean)
            results[cond]["n_samples"].append(int(len(h_k)))
            print(f"  k={k:3d}: linear={lin_mean:+.3f}, mlp={mlp_mean:+.3f}, n={len(h_k):5d}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                             gridspec_kw={"wspace": 0.30})

    # Panel A: Linear R²(k)
    for cond, info in results.items():
        ax = axes[0]
        ax.plot(info["horizons"], info["linear_r2"], "o-",
                color=info["color"], lw=2.0, markersize=8, label=info["label"])
    axes[0].axhline(0, ls=":", color="grey", alpha=0.5, lw=0.7)
    axes[0].set_xlabel("Prediction horizon $k$ (steps ahead)",
                       fontsize=11.5, fontweight="bold")
    axes[0].set_ylabel("Linear $R^2$ predicting $(x, y)_{t+k}$ from $\\mathbf{h}_t$",
                       fontsize=11, fontweight="bold")
    axes[0].set_title("(a) Linear probe (Ridge $\\alpha{=}10$)",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    axes[0].grid(linestyle=":", alpha=0.3)
    for s in ("top", "right"): axes[0].spines[s].set_visible(False)

    # Panel B: MLP R²(k)
    for cond, info in results.items():
        ax = axes[1]
        ax.plot(info["horizons"], info["mlp_r2"], "o-",
                color=info["color"], lw=2.0, markersize=8, label=info["label"])
    axes[1].axhline(0, ls=":", color="grey", alpha=0.5, lw=0.7)
    axes[1].set_xlabel("Prediction horizon $k$ (steps ahead)",
                       fontsize=11.5, fontweight="bold")
    axes[1].set_ylabel("MLP $R^2$ predicting $(x, y)_{t+k}$ from $\\mathbf{h}_t$",
                       fontsize=11, fontweight="bold")
    axes[1].set_title("(b) MLP probe (hidden $=256$, $L^2 = 10^{-4}$)",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].legend(loc="upper right", frameon=False, fontsize=10)
    axes[1].grid(linestyle=":", alpha=0.3)
    for s in ("top", "right"): axes[1].spines[s].set_visible(False)

    fig.suptitle("Predictive horizon: does $\\mathbf{h}_t$ encode future positions $(x, y)_{t+k}$? (foveated deferred to 250M re-probe)",
                 fontsize=11, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_predictive_horizon.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")

    Path("/tmp/extra_analyses/predictive_horizon.json").write_text(
        json.dumps(results, indent=2))
    print("wrote /tmp/extra_analyses/predictive_horizon.json")


if __name__ == "__main__":
    main()
