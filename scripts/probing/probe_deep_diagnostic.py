"""
Deep diagnostic on fov-fix_det / fov-learned_det probe failure.

Key questions:
  1. Is compass still pass-through? (Should be trivially R²≈1 for sighted
     conditions since compass is a direct observation each step.)
  2. Is distance-to-goal probeable? (Ego-relative, should work better than
     absolute GPS.)
  3. What's the per-lag R² behaviour? (paper's H1 probe.)
  4. What's the variance of the target? (If var is truly low across 200-step
     episodes we have bigger problems.)
  5. Does scaling / standardization matter?
  6. Does a much smaller ridge α recover the signal?

Run on Izar (has access to the npz files):
    python3 /tmp/probe_deep_diagnostic.py
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

PATHS = {
    "foveated_det": "/scratch/izar/wxu/probing_data/foveated_gibson_det.npz",
    "foveated_learned_det": "/scratch/izar/wxu/probing_data/foveated_learned_gibson_det.npz",
    "foveated_stoch": "/scratch/izar/wxu/probing_data/foveated_gibson.npz",
    "foveated_learned_stoch": "/scratch/izar/wxu/probing_data/foveated_learned_gibson.npz",
}


def episode_split(ep_ids, train_frac=0.8, seed=42):
    unique_eps = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    n_tr = int(len(unique_eps) * train_frac)
    train_eps, test_eps = unique_eps[:n_tr], unique_eps[n_tr:]
    return np.isin(ep_ids, train_eps), np.isin(ep_ids, test_eps)


def fit(Xtr, Xte, ytr, yte, alpha):
    if ytr.ndim == 1:
        ytr = ytr[:, None]; yte = yte[:, None]
    R = Ridge(alpha=alpha)
    R.fit(Xtr, ytr)
    yp = R.predict(Xte)
    r2 = r2_score(yte, yp, multioutput="uniform_average")
    mae = np.mean(np.abs(yp - yte))
    return r2, mae


def analyze(path, tag):
    d = np.load(path, allow_pickle=True)
    H = d["hidden_states"]      # (N, 512)
    positions = d["positions"]  # (N, 3) world xyz
    ep_ids = d["episode_ids"]
    gps = d.get("gps")           # (N, 2) episodic GPS
    compass = d.get("compass")   # (N,) episodic compass
    d2g = d["distance_to_goal"]
    gpos = d["goal_positions"]

    N = len(H)
    print(f"\n=== {tag}  (N={N:,} steps, {len(np.unique(ep_ids))} episodes) ===")

    train_mask, test_mask = episode_split(ep_ids)
    Htr, Hte = H[train_mask], H[test_mask]
    print(f"  train/test = {Htr.shape[0]:,}/{Hte.shape[0]:,}")

    # ---- Target variance (is there signal to decode?) ----
    if gps is not None:
        print(f"\n  GPS (episodic): range [{gps.min():.2f}, {gps.max():.2f}] m, "
              f"std {gps.std():.2f} m, var per-dim {gps.var(axis=0).mean():.2f} m²")
    if compass is not None:
        print(f"  Compass (episodic): range [{compass.min():.3f}, {compass.max():.3f}] rad, "
              f"std {compass.std():.3f}, var {compass.var():.3f}")
    print(f"  DtG: range [{d2g.min():.2f}, {d2g.max():.2f}] m, std {d2g.std():.2f}")

    # ---- Probe 1: GPS at different α ----
    if gps is not None:
        print(f"\n  Probe α-sweep (GPS):")
        for a in [1e-3, 1e-1, 1.0, 10.0, 100.0]:
            r2, mae = fit(Htr, Hte, gps[train_mask], gps[test_mask], a)
            print(f"    α={a:>7.3f}: R²={r2:+.3f}, MAE={mae:.3f} m")

    # ---- Probe 2: compass sin/cos, different α ----
    if compass is not None:
        c_sc = np.column_stack([np.sin(compass), np.cos(compass)])
        print(f"\n  Probe α-sweep (compass sin/cos):")
        for a in [1e-3, 1e-1, 1.0, 10.0, 100.0]:
            r2, mae = fit(Htr, Hte, c_sc[train_mask], c_sc[test_mask], a)
            print(f"    α={a:>7.3f}: R²={r2:+.3f}")

    # ---- Probe 3: DtG (should work — ego-relative) ----
    print(f"\n  Probe DtG (ego-relative, should be robust):")
    for a in [1.0, 10.0]:
        r2, mae = fit(Htr, Hte, d2g[train_mask], d2g[test_mask], a)
        print(f"    α={a}: R²={r2:+.3f}, MAE={mae:.3f} m")

    # ---- Probe 4: past positions at lag k ----
    print(f"\n  Path-history lag-k probe (GPS@t-k from H@t):")
    # Build lag-k pairs within each episode
    for k in [0, 1, 2, 5]:
        Xs, ys, eps = [], [], []
        for e in np.unique(ep_ids):
            idx = np.where(ep_ids == e)[0]
            if len(idx) <= k:
                continue
            # X = H[t], y = positions[t-k] (or gps[t-k])
            target_idx = idx[:len(idx) - k]
            X_idx = idx[k:]
            Xs.append(H[X_idx])
            if gps is not None:
                ys.append(gps[target_idx])
            else:
                ys.append(positions[target_idx, [0, 2]])
            eps.append(np.full(len(target_idx), e))
        if not Xs:
            continue
        X = np.vstack(Xs); y = np.vstack(ys); e_arr = np.concatenate(eps)
        tm, um = episode_split(e_arr)
        r2, mae = fit(X[tm], X[um], y[tm], y[um], 10.0)
        print(f"    lag={k}: N={len(X):,}, R²={r2:+.3f}, MAE={mae:.3f}")

    # ---- Probe 5: Per-episode test — pick one episode, check hidden-state stats ----
    print(f"\n  One sample episode sanity check:")
    ep0 = np.unique(ep_ids)[0]
    idx = np.where(ep_ids == ep0)[0]
    print(f"    Episode {ep0}: {len(idx)} steps")
    if gps is not None:
        print(f"    GPS trajectory: start {gps[idx[0]]}, end {gps[idx[-1]]}, path len "
              f"{np.abs(np.diff(gps[idx], axis=0)).sum():.2f} m")
    print(f"    H norm: mean {np.linalg.norm(H[idx], axis=1).mean():.2f}, "
          f"std {np.linalg.norm(H[idx], axis=1).std():.2f}")
    print(f"    H variance (top-5 dims): {sorted(H[idx].var(axis=0))[-5:]}")


for tag, path in PATHS.items():
    try:
        analyze(path, tag)
    except FileNotFoundError:
        print(f"\n{tag}: skip (not ready)")
    except Exception as e:
        print(f"\n{tag}: ERROR {type(e).__name__}: {e}")
