"""
Test whether StandardScaler on hidden states destroys LSTM probe signal.

Compare 4 configurations for fov-learned_det (which should be probeable):
  (A) raw H, no scaling           (my diagnostic)
  (B) StandardScaler on H         (analyze.py's prepare_features)
  (C) Mean-centered only (no std div)
  (D) StandardScaler + drop low-variance dims

If (B) >> others: scaler is fine, my diagnostic had a bug
If (A) >> (B): StandardScaler destroys signal
If (A) ~ (B) but analyze.py still reports -10: analyze.py has a separate bug
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def episode_split(ep_ids, train_frac=0.8, seed=42):
    unique_eps = np.unique(ep_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_eps)
    split = int(len(unique_eps) * train_frac)
    train_eps = set(unique_eps[:split].tolist())
    train_mask = np.array([e in train_eps for e in ep_ids])
    return train_mask, ~train_mask


def fit_r2(Xtr, Xte, Ytr, Yte, alpha=10.0):
    R = Ridge(alpha=alpha)
    R.fit(Xtr, Ytr)
    pred = R.predict(Xte)
    r2 = r2_score(Yte, pred, multioutput="uniform_average")
    return r2


def run(path, tag):
    d = np.load(path, allow_pickle=True)
    H = d["hidden_states"].astype(np.float32)
    ep = d["episode_ids"]
    gps = d["gps"]
    comp = d["compass"]
    d2g = d["distance_to_goal"]

    tm, um = episode_split(ep)
    print(f"\n=== {tag} (N={len(H):,}) ===")
    print(f"  H stats: mean {H.mean():+.3f}, std {H.std():.3f}, "
          f"min-dim-std {H.std(axis=0).min():.2e}, max-dim-std {H.std(axis=0).max():.2e}")
    print(f"  How many H dims have std < 1e-6?  {(H.std(axis=0) < 1e-6).sum()} / {H.shape[1]}")
    print(f"  How many H dims have std < 1e-3?  {(H.std(axis=0) < 1e-3).sum()} / {H.shape[1]}")

    c_sc = np.column_stack([np.sin(comp), np.cos(comp)])

    print(f"\n  {'config':<40} {'GPS R²':>10} {'compass R²':>12} {'DtG R²':>10}")
    print(f"  {'-' * 78}")

    # (A) Raw H, no scaling
    r_gps = fit_r2(H[tm], H[um], gps[tm], gps[um])
    r_comp = fit_r2(H[tm], H[um], c_sc[tm], c_sc[um])
    r_dtg = fit_r2(H[tm], H[um], d2g[tm, None], d2g[um, None])
    print(f"  (A) raw H, no scaling                   {r_gps:+10.3f} {r_comp:+12.3f} {r_dtg:+10.3f}")

    # (B) StandardScaler on H (analyze.py style)
    sc = StandardScaler()
    Htr = sc.fit_transform(H[tm])
    Hte = sc.transform(H[um])
    r_gps = fit_r2(Htr, Hte, gps[tm], gps[um])
    r_comp = fit_r2(Htr, Hte, c_sc[tm], c_sc[um])
    r_dtg = fit_r2(Htr, Hte, d2g[tm, None], d2g[um, None])
    print(f"  (B) StandardScaler (analyze.py)         {r_gps:+10.3f} {r_comp:+12.3f} {r_dtg:+10.3f}")

    # (C) Mean-centered only
    mu = H[tm].mean(axis=0)
    Htr_c = H[tm] - mu
    Hte_c = H[um] - mu
    r_gps = fit_r2(Htr_c, Hte_c, gps[tm], gps[um])
    r_comp = fit_r2(Htr_c, Hte_c, c_sc[tm], c_sc[um])
    r_dtg = fit_r2(Htr_c, Hte_c, d2g[tm, None], d2g[um, None])
    print(f"  (C) mean-centered only                  {r_gps:+10.3f} {r_comp:+12.3f} {r_dtg:+10.3f}")

    # (D) StandardScaler + drop low-variance dims (<1e-6)
    keep = H[tm].std(axis=0) > 1e-6
    sc2 = StandardScaler()
    Htr2 = sc2.fit_transform(H[tm][:, keep])
    Hte2 = sc2.transform(H[um][:, keep])
    r_gps = fit_r2(Htr2, Hte2, gps[tm], gps[um])
    r_comp = fit_r2(Htr2, Hte2, c_sc[tm], c_sc[um])
    r_dtg = fit_r2(Htr2, Hte2, d2g[tm, None], d2g[um, None])
    print(f"  (D) Scaler + drop low-var dims ({keep.sum():3d}/512) {r_gps:+10.3f} {r_comp:+12.3f} {r_dtg:+10.3f}")


run("/scratch/izar/wxu/probing_data/foveated_gibson_det.npz", "FOV-FIX det")
run("/scratch/izar/wxu/probing_data/foveated_learned_gibson_det.npz", "FOV-LEARNED det")
run("/scratch/izar/wxu/probing_data/foveated_gibson.npz", "FOV-FIX stoch (buggy)")
run("/scratch/izar/wxu/probing_data/foveated_learned_gibson.npz", "FOV-LEARNED stoch")
