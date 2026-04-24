"""
Explore alternative probe targets that may be robust under det data.

Hypothesis: PointNav LSTM encodes ego-relative navigation state (DtG,
goal-vector, past path) but not world-absolute GPS/compass.

Test which targets give high R² with LOW CV variance on det data for
fov-fix, fov-learned. If we find a target that's (a) conceptually
interesting, (b) high R² across folds, (c) discriminates between
conditions — we have a rescue.

Candidate targets:
  T1. dtg (current distance to goal) — already shown robust
  T2. dtg@lag-k (past distance to goal) — path-history compensatory memory
  T3. goal_vector (ego-frame g - p) — 2-D ego-relative goal direction
  T4. goal_vector@lag-k — past ego-frame goal (= memory of where goal was
      relative to agent k steps ago)
  T5. heading_change_since_start (cumulative rotation) — how much have I
      turned so far
  T6. path_displacement (|g_t - g_0|, scalar) — how far have I moved
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def kfold_r2(H, Y, ep_ids, n_folds=5, alpha=10.0, seed=42, scale=True):
    """Episode-level KFold; return per-fold R²."""
    unique_eps = np.unique(ep_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_eps)
    kf = KFold(n_splits=n_folds, shuffle=False)
    r2s = []
    for tr_idx, te_idx in kf.split(unique_eps):
        tr_set = set(unique_eps[tr_idx].tolist())
        tr_mask = np.array([e in tr_set for e in ep_ids])
        te_mask = ~tr_mask
        if tr_mask.sum() < 100 or te_mask.sum() < 20:
            continue
        Xtr, Xte = H[tr_mask], H[te_mask]
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        Ytr, Yte = Y[tr_mask], Y[te_mask]
        if Ytr.ndim == 1:
            Ytr = Ytr[:, None]; Yte = Yte[:, None]
        R = Ridge(alpha=alpha)
        R.fit(Xtr, Ytr)
        pred = R.predict(Xte)
        r2 = r2_score(Yte, pred, multioutput="uniform_average")
        r2s.append(r2)
    return np.array(r2s) if r2s else np.array([np.nan])


def build_lag_pairs(values, ep_ids, k):
    """Return (values_at_t_minus_k, index_of_H_at_t, ep_ids_at_t) for within-episode pairs."""
    X_idx, Y_vals, Eps = [], [], []
    for e in np.unique(ep_ids):
        idx = np.where(ep_ids == e)[0]
        if len(idx) <= k:
            continue
        X_idx.extend(idx[k:])
        Y_vals.extend(values[idx[:len(idx) - k]])
        Eps.extend([e] * (len(idx) - k))
    return np.array(X_idx), np.array(Y_vals), np.array(Eps)


def analyze(path, tag):
    d = np.load(path, allow_pickle=True)
    H = d["hidden_states"].astype(np.float32)
    ep = d["episode_ids"]
    pos = d["positions"]  # (N, 3)
    head = d["headings"]  # (N,)
    gps = d["gps"]
    comp = d["compass"]
    d2g = d["distance_to_goal"]
    goal = d["goal_positions"]

    N = len(H)
    print(f"\n=== {tag}  (N={N:,}) ===")

    # T1: DtG (current)
    r = kfold_r2(H, d2g, ep)
    print(f"  T1 DtG (current)            : R² = {r.mean():+.3f} ± {r.std():.3f}  [folds: {np.array2string(r, precision=2)}]")

    # T3: goal_vector (ego-frame g - p)
    # agent at pos, heading h; ego-frame goal = R(-h) * (goal_xz - pos_xz)
    pos_xz = np.stack([pos[:, 0], pos[:, 2]], axis=1)
    goal_xz = np.stack([goal[:, 0], goal[:, 2]], axis=1)
    rel = goal_xz - pos_xz  # (N, 2) world-frame
    cos_h, sin_h = np.cos(-head), np.sin(-head)
    ego_goal = np.stack([
        cos_h * rel[:, 0] - sin_h * rel[:, 1],
        sin_h * rel[:, 0] + cos_h * rel[:, 1],
    ], axis=1)
    r = kfold_r2(H, ego_goal, ep)
    print(f"  T3 ego-frame goal-vector    : R² = {r.mean():+.3f} ± {r.std():.3f}  [folds: {np.array2string(r, precision=2)}]")

    # T5: heading change since episode start
    cum_rot = []
    start_heads = {}
    for i, e in enumerate(ep):
        if e not in start_heads:
            start_heads[e] = head[i]
        dh = head[i] - start_heads[e]
        dh = np.arctan2(np.sin(dh), np.cos(dh))
        cum_rot.append(dh)
    cum_rot = np.array(cum_rot)
    cum_rot_sc = np.column_stack([np.sin(cum_rot), np.cos(cum_rot)])
    r = kfold_r2(H, cum_rot_sc, ep)
    print(f"  T5 heading-since-start      : R² = {r.mean():+.3f} ± {r.std():.3f}  [folds: {np.array2string(r, precision=2)}]")

    # T6: path displacement (scalar distance traveled from start)
    path_disp = np.zeros(N)
    start_pos = {}
    for i in range(N):
        e = ep[i]
        if e not in start_pos:
            start_pos[e] = pos[i, [0, 2]]
        path_disp[i] = np.linalg.norm(pos[i, [0, 2]] - start_pos[e])
    r = kfold_r2(H, path_disp, ep)
    print(f"  T6 path-displacement scalar : R² = {r.mean():+.3f} ± {r.std():.3f}  [folds: {np.array2string(r, precision=2)}]")

    # T2/T4: lag-k DtG and lag-k goal-vector (H1 compensatory-memory test)
    print(f"\n  Lag-k compensatory memory tests:")
    for k in [0, 1, 2, 3, 5, 8]:
        X_idx, Yv, Eps = build_lag_pairs(d2g, ep, k)
        if len(X_idx) < 100:
            continue
        r = kfold_r2(H[X_idx], Yv, Eps)
        X_idx2, Yg, Eps2 = build_lag_pairs(ego_goal, ep, k)
        r_gv = kfold_r2(H[X_idx2], Yg, Eps2)
        print(f"    lag {k}: DtG R²={r.mean():+.3f}±{r.std():.3f} | ego-goal R²={r_gv.mean():+.3f}±{r_gv.std():.3f}  (N={len(X_idx):,})")


for tag, path in [
    ("FOV-FIX det",          "/scratch/izar/wxu/probing_data/foveated_gibson_det.npz"),
    ("FOV-LEARNED det",      "/scratch/izar/wxu/probing_data/foveated_learned_gibson_det.npz"),
    ("UNIFORM stoch (buggy)", "/scratch/izar/wxu/probing_data/uniform_gibson.npz"),
    ("BLIND stoch (buggy)",  "/scratch/izar/wxu/probing_data/blind_gibson.npz"),
]:
    try:
        analyze(path, tag)
    except FileNotFoundError:
        print(f"\n{tag}: not ready")
    except Exception as e:
        print(f"\n{tag}: ERROR {type(e).__name__}: {e}")
