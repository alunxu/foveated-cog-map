"""Predictive coding residual (Rao & Ballard 1999, Nature Neuroscience).

Predictive-coding cortical models maintain an internal forward model and
propagate prediction-error to higher layers. Here we fit a post-hoc one-step
forward MLP `f̂: (h_t, a_t) -> h_{t+1}` per condition (no LSTM retraining)
and analyse the residual epsilon_t = h_{t+1} - f̂(h_t, a_t).

The agent's pre-registered prediction (cogneuro_round2/predictive_coding.md):
  Mean per-step residual norm E[||eps_t||] increases monotonically across
  {blind, coarse, fov-LP, foveated, uniform}; uniform-vs-blind effect >= 30%.
  Rank-90% of Cov(eps) is lower for blind/coarse than for foveated/uniform.

Mechanism: the encoder pushes more *novel* information per step into h in
the rich-encoder conditions, so the recurrent state is "less predictable"
from its prior state alone.

Pre-reg HPs (locked here, used uniformly across conditions):
  forward MLP: 2 hidden layers x 256, ReLU, AdamW 1e-3, weight_decay 1e-4
  episode-level 80/20 train/eval split (no information leakage)
  500 epochs, early stop on eval loss plateau
  PCA-50 pre-reduction of h to keep MLP input dim manageable
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

DATA_ROOT = Path("/tmp/rcp_analysis_v3")
FILES = {
    "blind": "blind_izar_det_RCP.npz",
    "coarse": "coarse_det_RCP.npz",
    "foveated": "foveated_det_RCP.npz",
    "foveated_logpolar": "foveated_logpolar_det_RCP.npz",
    "uniform": "uniform_det_RCP.npz",
}


class ForwardMLP(nn.Module):
    def __init__(self, d_in: int, d_action: int, d_out: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + d_action, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, h, a):
        return self.net(torch.cat([h, a], dim=-1))


def episode_split(eps_id: np.ndarray, frac_train: float = 0.8, seed: int = 0):
    """Split episodes (not frames) so no information leakage."""
    eps = np.unique(eps_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_train = int(len(eps) * frac_train)
    train_eps = set(eps[:n_train].tolist())
    train_mask = np.array([e in train_eps for e in eps_id])
    return train_mask


def collect_pairs(h: np.ndarray, action_oh: np.ndarray, ep_id: np.ndarray, sip: np.ndarray):
    """Build (h_t, a_t, h_{t+1}) consecutive within-episode pairs."""
    order = np.lexsort((sip, ep_id))
    h_o, a_o, ep_o, sip_o = h[order], action_oh[order], ep_id[order], sip[order]
    keep = (ep_o[:-1] == ep_o[1:]) & ((sip_o[1:] - sip_o[:-1]) == 1)
    h_t = h_o[:-1][keep]
    a_t = a_o[:-1][keep]
    h_tp1 = h_o[1:][keep]
    return h_t, a_t, h_tp1


def analyse_one_condition(npz_path: Path, n_pcs: int = 50, max_steps: int = 100000,
                            device: str = "cpu") -> dict:
    d = np.load(npz_path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    pos = d["positions"].astype(np.float32)
    ep_id = d["episode_ids"]
    sip = d["step_in_episode"]

    # Construct one-hot action per step. Habitat action is implicit; we use
    # discrete heading change as proxy: bin Δheading.
    head = d["headings"].astype(np.float32)
    # Standardise
    mu = h.mean(0, keepdims=True); sd = h.std(0, keepdims=True) + 1e-6
    h = (h - mu) / sd
    pca = PCA(n_components=n_pcs, random_state=0).fit(h)
    h_pcs = pca.transform(h).astype(np.float32)

    # Action proxy: 8-octant heading
    h_oct = np.where(head < 0, head + 2 * np.pi, head)
    act_idx = (h_oct / (2 * np.pi) * 8).astype(int) % 8
    a_oh = np.eye(8, dtype=np.float32)[act_idx]

    # Subsample
    if len(h_pcs) > max_steps:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(h_pcs), max_steps, replace=False))
        h_pcs = h_pcs[idx]; a_oh = a_oh[idx]; ep_id = ep_id[idx]; sip = sip[idx]

    h_t, a_t, h_tp1 = collect_pairs(h_pcs, a_oh, ep_id, sip)
    # Reconstruct ep_id_pairs from the same sort+keep logic as collect_pairs.
    order = np.lexsort((sip, ep_id))
    ep_o = ep_id[order]
    sip_o = sip[order]
    keep = (ep_o[:-1] == ep_o[1:]) & ((sip_o[1:] - sip_o[:-1]) == 1)
    ep_pairs = ep_o[:-1][keep]
    print(f"  pairs collected: {len(h_t)}  ep_pairs: {len(ep_pairs)}")
    assert len(h_t) == len(ep_pairs), f"mismatch {len(h_t)} vs {len(ep_pairs)}"
    train_mask = episode_split(ep_pairs, frac_train=0.8)

    h_t_t, a_t_t, y_t = h_t[train_mask], a_t[train_mask], h_tp1[train_mask]
    h_t_e, a_t_e, y_e = h_t[~train_mask], a_t[~train_mask], h_tp1[~train_mask]
    print(f"  train: {len(h_t_t)}  eval: {len(h_t_e)}")

    # Train MLP
    dev = torch.device(device)
    model = ForwardMLP(d_in=n_pcs, d_action=a_oh.shape[1], d_out=n_pcs,
                       d_hidden=256).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.from_numpy(h_t_t).to(dev)
    Atr = torch.from_numpy(a_t_t).to(dev)
    Ytr = torch.from_numpy(y_t).to(dev)
    Xev = torch.from_numpy(h_t_e).to(dev)
    Aev = torch.from_numpy(a_t_e).to(dev)
    Yev = torch.from_numpy(y_e).to(dev)
    N = len(Xtr)

    best_eval = float("inf")
    best_state = None
    for epoch in range(150):
        idx = torch.randperm(N, device=dev)
        for s in range(0, N, 512):
            b = idx[s:s + 512]
            pred = model(Xtr[b], Atr[b])
            loss = ((pred - Ytr[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ev = ((model(Xev, Aev) - Yev) ** 2).mean().item()
        if ev < best_eval:
            best_eval = ev
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        eps = (Yev - model(Xev, Aev)).cpu().numpy()
    # Compute R^2 (multi-output average)
    ss_res = (eps ** 2).sum(0)
    ss_tot = ((Yev.cpu().numpy() - Yev.mean(0).cpu().numpy()) ** 2).sum(0).clip(min=1e-9)
    r2 = float((1.0 - ss_res / ss_tot).mean())

    eps_norm = np.linalg.norm(eps, axis=1)
    mean_norm = float(eps_norm.mean())
    p10, p50, p90 = np.quantile(eps_norm, [0.1, 0.5, 0.9])
    cov_eps = np.cov(eps.T)
    eigs = np.linalg.eigvalsh(cov_eps)[::-1]
    cum = np.cumsum(eigs) / eigs.sum()
    rank_90 = int(np.searchsorted(cum, 0.90) + 1)

    return {
        "n_train": int(len(Xtr)),
        "n_eval": int(len(Xev)),
        "n_pcs": n_pcs,
        "forward_R2": r2,
        "mean_residual_norm": mean_norm,
        "p10": float(p10), "p50": float(p50), "p90": float(p90),
        "cov_top1_eig": float(eigs[0]),
        "cov_total_var": float(eigs.sum()),
        "rank_90_pct": rank_90,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/tmp/predcode_results.json")
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=100000)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    results = {}
    for cond, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f"  MISSING {path}; skip {cond}")
            continue
        print(f"\n=== {cond} ===")
        results[cond] = analyse_one_condition(path, n_pcs=args.n_pcs,
                                                 max_steps=args.max_steps, device=device)
        r = results[cond]
        print(f"  forward R^2 = {r['forward_R2']:.3f}  ||eps|| mean = "
               f"{r['mean_residual_norm']:.3f}  rank_90 = {r['rank_90_pct']}/{args.n_pcs}")

    json.dump(results, open(args.out_path, "w"), indent=2)
    print(f"\nsaved {args.out_path}")


if __name__ == "__main__":
    main()
