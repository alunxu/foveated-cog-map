"""Re-run analyze_extra_states (per-layer h+c) + goal_vector probes for
all 5 post-retrain conditions, on the {cond}_det.npz NPZs (consistent
with the rest of the §C/§D pipeline).

Outputs:
  $RESULTS_OUT/extra_states/{cond}_extra_states.json
  $RESULTS_OUT/goal_vector_5cond.json
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

NPZ_DIR = Path("/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp")
RESULTS = Path("/scratch/wxu/habitat_checkpoints_rcp/extra_analyses_5cond")
RESULTS.mkdir(parents=True, exist_ok=True)
EXTRA_DIR = RESULTS / "extra_states"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)

# (key in figure scripts, npz file stem on RCP)
CONDS = [
    ("blind",             "blind_izar_det"),
    ("coarse",            "coarse_det"),
    ("foveated_logpolar", "foveated_logpolar_det"),
    ("foveated",          "foveated_det"),
    ("uniform",           "uniform_det"),
]


# ---- analyze_extra_states-style: 6 (h+c × 3 layers) × {gps, compass}
def fit_per_state(X, y, groups, alpha=10.0, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []
    for tr, te in gkf.split(X, y, groups):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        fold_r2.append(r2_score(y[te], m.predict(X[te]), multioutput="uniform_average"))
    return {"r2_mean": float(np.mean(fold_r2)), "r2_std": float(np.std(fold_r2))}


def compute_extra_states(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    has_h = "h_layers" in d.files
    has_c = "c_layers" in d.files
    if not has_h:
        print(f"  no h_layers in {npz_path.name}; skipping extra_states")
        return None
    h_layers = d["h_layers"]   # (N, 3, 512)
    c_layers = d["c_layers"] if has_c else None
    gps = d["gps"][:, :2] if d["gps"].ndim == 2 else d["gps"]
    compass = d.get("compass") if "compass" in d.files else None
    if compass is not None:
        compass = np.stack([np.sin(compass), np.cos(compass)], axis=-1) \
            if compass.ndim == 1 else compass
    ep = d["episode_ids"]
    out = {}
    for state, arr in [("h", h_layers), ("c", c_layers)]:
        if arr is None:
            continue
        for L in range(arr.shape[1]):
            X = arr[:, L, :]
            out[f"{state}_layer{L}_gps"] = fit_per_state(X, gps, ep)
            if compass is not None:
                out[f"{state}_layer{L}_compass"] = fit_per_state(X, compass, ep)
    return out


# ---- goal_vector_probe-style
def ego_goal_vector(positions, goals, headings):
    dxyz = goals - positions
    dx, dz = dxyz[:, 0], dxyz[:, 2]
    cos_h = np.cos(-headings); sin_h = np.sin(-headings)
    forward = cos_h * (-dz) - sin_h * dx
    lateral = sin_h * (-dz) + cos_h * dx
    return np.stack([forward, lateral], axis=1)


def split_by_episode(ep_ids, seed=0, test_frac=0.2):
    unique = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(test_frac * len(unique)))
    test_eps = set(unique[:n_test].tolist())
    test_mask = np.isin(ep_ids, list(test_eps))
    return ~test_mask, test_mask


def compute_goal_vector(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    needed = ("hidden_states", "positions", "goal_positions",
              "headings", "episode_ids", "gps")
    if not all(k in d.files for k in needed):
        miss = [k for k in needed if k not in d.files]
        return {"error": f"missing {miss}"}
    X = d["hidden_states"].astype(np.float32)
    positions = d["positions"].astype(np.float32)
    goals = d["goal_positions"].astype(np.float32)
    headings = d["headings"].astype(np.float32)
    if headings.ndim > 1:
        headings = headings.squeeze()
    ep = d["episode_ids"]
    y = ego_goal_vector(positions, goals, headings)
    dist = np.linalg.norm(y, axis=1)
    direction = np.arctan2(y[:, 1], y[:, 0])
    tr, te = split_by_episode(ep)
    # 2-D
    clf = Ridge(alpha=10.0).fit(X[tr], y[tr])
    r2_vec = r2_score(y[te], clf.predict(X[te]), multioutput="uniform_average")
    # dist
    clf_d = Ridge(alpha=10.0).fit(X[tr], dist[tr])
    r2_dist = r2_score(dist[te], clf_d.predict(X[te]))
    # dir (sin, cos)
    dir_sc = np.stack([np.sin(direction), np.cos(direction)], axis=1)
    clf_dir = Ridge(alpha=10.0).fit(X[tr], dir_sc[tr])
    r2_dir = r2_score(dir_sc[te], clf_dir.predict(X[te]),
                      multioutput="uniform_average")
    # GPS reference
    clf_gps = Ridge(alpha=10.0).fit(X[tr], d["gps"][tr])
    r2_gps = r2_score(d["gps"][te], clf_gps.predict(X[te]),
                      multioutput="uniform_average")
    return {
        "n_steps": int(X.shape[0]),
        "n_episodes": int(len(np.unique(ep))),
        "goal_vector_r2": float(r2_vec),
        "goal_dist_r2": float(r2_dist),
        "goal_direction_r2": float(r2_dir),
        "gps_r2_reference": float(r2_gps),
    }


print("=== analyze_extra_states (per-layer h+c × {gps, compass}) ===")
extra_summary = {}
for k, stem in CONDS:
    p = NPZ_DIR / f"{stem}.npz"
    if not p.exists():
        print(f"  MISSING {k}: {p}")
        continue
    print(f"  {k}: {p.name}")
    out = compute_extra_states(p)
    if out is None:
        print("    (no h_layers field; skipped)")
        continue
    (EXTRA_DIR / f"{k}_extra_states.json").write_text(json.dumps(out, indent=2))
    extra_summary[k] = out
    # print h2 GPS & compass
    h2g = out.get("h_layer2_gps", {}).get("r2_mean")
    h2c = out.get("h_layer2_compass", {}).get("r2_mean")
    print(f"    h2 GPS R²={h2g}  h2 compass R²={h2c}")
(RESULTS / "extra_states_summary.json").write_text(
    json.dumps(extra_summary, indent=2))

print("\n=== goal_vector_probe (5-cond) ===")
gv = {}
for k, stem in CONDS:
    p = NPZ_DIR / f"{stem}.npz"
    if not p.exists():
        print(f"  MISSING {k}: {p}")
        continue
    print(f"  {k}: {p.name}")
    r = compute_goal_vector(p)
    if "error" in r:
        print(f"    ERROR: {r['error']}")
        continue
    gv[k] = r
    print(f"    vec R²={r['goal_vector_r2']:+.3f}  "
          f"dist R²={r['goal_dist_r2']:+.3f}  "
          f"dir R²={r['goal_direction_r2']:+.3f}  "
          f"GPS-ref R²={r['gps_r2_reference']:+.3f}")
(RESULTS / "goal_vector_5cond.json").write_text(json.dumps(gv, indent=2))

print("\nDONE")
