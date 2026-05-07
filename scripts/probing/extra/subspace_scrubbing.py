"""
F3: GPS-subspace causal scrubbing (Blind, Coarse, Uniform — foveated deferred).

Tests whether the linear position-axis (Ridge β) is the dominant direction
encoding position in h_2, OR if position is redundantly encoded across many
non-linear directions:

  1. Compute Ridge β probe (β: 2 × 512 — direction in h_2 that linearly
     predicts (x, y) GPS).
  2. Project h_2 onto null-space of β: h_scrub = (I - β^+ β) h_2.
     This removes the 2-d β subspace from h_2 while preserving 510 dims.
  3. Re-evaluate the MLP probe on h_scrub. Compare to MLP R^2 on full h_2.

Predictions:
  - Bottleneck conds (Blind, Coarse): position concentrated in β; scrubbing
    β should cause MLP R^2 to DROP substantially.
  - Rich-encoder conds (Uniform): position encoded non-linearly in non-β
    directions; scrubbing β should leave MLP R^2 LARGELY UNCHANGED.

This is "noising" patching (Heimersheim & Nanda 2024 §2.3): tests whether
β was NECESSARY to maintain (linearly + non-linearly) recoverable position.

DEFERRED: foveated (174M below SPL plateau; will run with seed0pre 250M-
effective tomorrow).

Reads:  /tmp/cond_npzs/{blind,matched,uniform}_gibson_det.npz
Writes: /tmp/extra_analyses/subspace_scrubbing.json
        docs/manuscript/fig/figa14_subspace_scrubbing.pdf
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
# Optional paper-style import (skip if not available, e.g. on RCP).
try:
    from pathlib import Path as _Pth
    _local_style = _Pth("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/scripts/paper_figures")
    if _local_style.exists():
        sys.path.insert(0, str(_local_style))
        from _style import apply_paper_style
        apply_paper_style()
except Exception:
    pass


import os
NPZ_DIR = os.environ.get("SCRUB_NPZ_DIR", "/tmp/cond_npzs")
RESULTS_OUT = os.environ.get("SCRUB_RESULTS_OUT", "/tmp/extra_analyses")
CONDS = [
    ("blind",             f"{NPZ_DIR}/blind_izar_det.npz",        "Blind",    "#444444"),
    ("coarse",            f"{NPZ_DIR}/coarse_det.npz",            "Coarse",   "#377eb8"),
    ("foveated_logpolar", f"{NPZ_DIR}/foveated_logpolar_det.npz", "Fov-LP",   "#984ea3"),
    ("foveated",          f"{NPZ_DIR}/foveated_det.npz",          "Foveated", "#e41a1c"),
    ("uniform",           f"{NPZ_DIR}/uniform_det.npz",           "Uniform",  "#4daf4a"),
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


_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fit_mlp(X_tr, y_tr, X_te, y_te, hidden=256, l2=1e-4, epochs=40, bs=512, lr=1e-3):
    m = MLP(X_tr.shape[1], y_tr.shape[1], hidden).to(_DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=l2)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=_DEV)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=_DEV)
    Xv = torch.tensor(X_te, dtype=torch.float32, device=_DEV)
    n = len(Xt)
    for _ in range(epochs):
        idx = torch.randperm(n, device=_DEV)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            yhat = m(Xt[b])
            loss = nn.MSELoss()(yhat, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): yhp = m(Xv).cpu().numpy()
    return r2_score(y_te, yhp, multioutput='uniform_average')


def main():
    Path(RESULTS_OUT).mkdir(exist_ok=True, parents=True)
    results = {}

    for cond, path, label, color in CONDS:
        print(f"\n=== {cond} ===")
        d = np.load(Path(path))
        h = d["hidden_states"].astype(np.float32)
        gps = d["gps"].astype(np.float32)
        ep_ids = d["episode_ids"]
        if len(h) > 20000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(h), 20000, replace=False)
            h = h[idx]; gps = gps[idx]; ep_ids = ep_ids[idx]
        h = h - h.mean(axis=0, keepdims=True)
        gps = gps - gps.mean(axis=0, keepdims=True)

        # Step 1: Ridge probe → β (linear GPS direction)
        ridge = Ridge(alpha=10.0).fit(h, gps)
        beta = ridge.coef_  # (2, 512)

        # Step 2: Null-space projector P = I - β^T (β β^T)^-1 β
        beta_t_beta = beta @ beta.T  # (2, 2)
        beta_pinv = beta.T @ np.linalg.inv(beta_t_beta + 1e-9 * np.eye(2))
        P_nullspace = np.eye(beta.shape[1]) - beta_pinv @ beta

        # Step 3: 5-fold ep-CV
        unique_eps = np.unique(ep_ids)
        rng2 = np.random.default_rng(42); rng2.shuffle(unique_eps)
        folds = np.array_split(unique_eps, 5)
        mlp_orig_r2 = []; mlp_scrub_r2 = []
        lin_orig_r2 = []; lin_scrub_r2 = []
        for te_eps in folds:
            tr_mask = ~np.isin(ep_ids, te_eps); te_mask = np.isin(ep_ids, te_eps)
            X_tr, y_tr = h[tr_mask], gps[tr_mask]
            X_te, y_te = h[te_mask], gps[te_mask]
            X_tr_scrub = X_tr @ P_nullspace
            X_te_scrub = X_te @ P_nullspace

            r1 = Ridge(alpha=10.0).fit(X_tr, y_tr)
            r2 = Ridge(alpha=10.0).fit(X_tr_scrub, y_tr)
            lin_orig_r2.append(r2_score(y_te, r1.predict(X_te), multioutput="uniform_average"))
            lin_scrub_r2.append(r2_score(y_te, r2.predict(X_te_scrub), multioutput="uniform_average"))
            mlp_orig_r2.append(fit_mlp(X_tr, y_tr, X_te, y_te))
            mlp_scrub_r2.append(fit_mlp(X_tr_scrub, y_tr, X_te_scrub, y_te))

        results[cond] = {
            "label": label, "color": color,
            "linear_r2_orig": float(np.mean(lin_orig_r2)),
            "linear_r2_orig_std": float(np.std(lin_orig_r2)),
            "linear_r2_scrub": float(np.mean(lin_scrub_r2)),
            "linear_r2_scrub_std": float(np.std(lin_scrub_r2)),
            "mlp_r2_orig": float(np.mean(mlp_orig_r2)),
            "mlp_r2_orig_std": float(np.std(mlp_orig_r2)),
            "mlp_r2_scrub": float(np.mean(mlp_scrub_r2)),
            "mlp_r2_scrub_std": float(np.std(mlp_scrub_r2)),
            "mlp_drop": float(np.mean(mlp_orig_r2) - np.mean(mlp_scrub_r2)),
        }
        print(f"  Linear: orig={np.mean(lin_orig_r2):+.3f}, scrub={np.mean(lin_scrub_r2):+.3f}, drop={np.mean(lin_orig_r2)-np.mean(lin_scrub_r2):+.3f}")
        print(f"  MLP:    orig={np.mean(mlp_orig_r2):+.3f}, scrub={np.mean(mlp_scrub_r2):+.3f}, drop={np.mean(mlp_orig_r2)-np.mean(mlp_scrub_r2):+.3f}")

    # Save JSON results so we can replot anywhere later.
    Path(f"{RESULTS_OUT}/subspace_scrubbing.json").write_text(
        json.dumps(results, indent=2))
    print(f"wrote {RESULTS_OUT}/subspace_scrubbing.json")

    # Plot only if a figure path is set (skipped on RCP runs that just
    # produce the JSON).
    fig_out = os.environ.get("SCRUB_FIG_OUT")
    if not fig_out:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.4))
    cond_order = [c[0] for c in CONDS if c[0] in results]
    cols = [next(c[3] for c in CONDS if c[0] == k) for k in cond_order]
    labs = [next(c[2] for c in CONDS if c[0] == k) for k in cond_order]
    orig = [results[k]["mlp_r2_orig"] for k in cond_order]
    scrub = [results[k]["mlp_r2_scrub"] for k in cond_order]
    orig_err = [results[k]["mlp_r2_orig_std"] for k in cond_order]
    scrub_err = [results[k]["mlp_r2_scrub_std"] for k in cond_order]

    x = np.arange(len(cond_order))
    w = 0.35
    ax.bar(x - w/2, orig, w, yerr=orig_err, color=cols, alpha=0.85,
           label="Original $\\mathbf{h}_2$",
           edgecolor="black", linewidth=0.8, capsize=4)
    ax.bar(x + w/2, scrub, w, yerr=scrub_err, color=cols, alpha=0.4,
           label="$\\beta$-scrubbed $\\mathbf{h}_2$",
           edgecolor="black", linewidth=0.8, capsize=4, hatch="///")
    for i, (o, s) in enumerate(zip(orig, scrub)):
        drop = o - s
        ax.text(i, max(o, s) + 0.05, f"$\\Delta {drop:+.2f}$", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.7)
    ax.set_ylabel("MLP probe GPS $R^2$\n(5-fold episode-level CV)",
                  fontsize=11.5, fontweight="bold")
    ax.set_title("$\\beta$-subspace scrubbing: does removing the linear $\\beta$ direction kill MLP-recoverable position?",
                 fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_out, dpi=200, bbox_inches="tight")
    print(f"wrote {fig_out}")


if __name__ == "__main__":
    main()
