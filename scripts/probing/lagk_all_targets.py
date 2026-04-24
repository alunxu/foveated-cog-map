import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def kfold_r2(H, Y, ep, n_folds=5, alpha=10.0, seed=42):
    ue = np.unique(ep)
    rng = np.random.RandomState(seed)
    rng.shuffle(ue)
    kf = KFold(n_splits=n_folds, shuffle=False)
    r2s = []
    for tri, tei in kf.split(ue):
        tr = np.isin(ep, ue[tri]); te = np.isin(ep, ue[tei])
        sc = StandardScaler()
        Xtr = sc.fit_transform(H[tr])
        Xte = sc.transform(H[te])
        Ytr, Yte = Y[tr], Y[te]
        if Ytr.ndim == 1:
            Ytr = Ytr[:, None]; Yte = Yte[:, None]
        R = Ridge(alpha=alpha)
        R.fit(Xtr, Ytr)
        p = R.predict(Xte)
        r2s.append(r2_score(Yte, p, multioutput="uniform_average"))
    return float(np.mean(r2s)), float(np.std(r2s))


def lag_pairs(v, ep, k):
    Xi, Yv, E = [], [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        if len(idx) <= k:
            continue
        Xi.extend(idx[k:])
        Yv.extend(v[idx[:len(idx) - k]])
        E.extend([e] * (len(idx) - k))
    return np.array(Xi), np.array(Yv), np.array(E)


names = [
    "foveated_gibson_det",
    "foveated_learned_gibson_det",
    "uniform_gibson_det",
    "matched_gibson_det",
    "blind_gibson_det",
]

header = f"{'condition':<26} {'target':<10} {'lag0':>12} {'lag2':>12} {'lag5':>12} {'lag10':>12}"
print(header)
print("-" * len(header))

for name in names:
    path = f"/scratch/izar/wxu/probing_data/{name}.npz"
    try:
        d = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        print(f"{name:<26} (not ready)")
        continue
    H = d["hidden_states"].astype(np.float32)
    ep = d["episode_ids"]
    gps = d["gps"]
    comp = d["compass"]
    comp_sc = np.column_stack([np.sin(comp), np.cos(comp)])
    dtg = d["distance_to_goal"]

    for tgt_name, tgt_vals in [("GPS", gps), ("compass", comp_sc), ("DtG", dtg)]:
        rs = []
        for k in [0, 2, 5, 10]:
            Xi, Yv, E = lag_pairs(tgt_vals, ep, k)
            if len(Xi) < 100:
                rs.append("     -     ")
                continue
            m, s = kfold_r2(H[Xi], Yv, E)
            rs.append(f"{m:+.2f}\u00b1{s:.2f}")
        print(f"{name:<26} {tgt_name:<10} {rs[0]:>12} {rs[1]:>12} {rs[2]:>12} {rs[3]:>12}")
    print()
