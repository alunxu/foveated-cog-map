"""
LOSO + per-scene R² distribution for 5 conditions.
Tests whether sighted-internal has a sub-gradient (coarse vs foveated vs log-polar vs uniform),
or whether all 4 sighted truly collapse to ~0 (true binary).
"""
import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

NPZ_DIR = Path("/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp")
CONDS = [
    ("blind", "blind_det_ckpt49.npz"),
    ("coarse", "coarse_det.npz"),
    ("fnorm", "fnorm_det_ckpt49.npz"),                       # foveated F2
    ("foveated_logpolar", "foveated_logpolar_det.npz"),
    ("uniform", "uniform_det.npz"),
]
N_MAX = 30000
TOP_K_SCENES = 50
N_MIN = 100
ALPHA = 10.0
SEED = 0


def loso_one_condition(npz_path: Path):
    d = np.load(npz_path)
    h = d["hidden_states"].astype(np.float32)
    gps = d["gps"].astype(np.float32)
    scene_ids = d["scene_ids"]

    # Subsample
    rng = np.random.default_rng(SEED)
    if len(h) > N_MAX:
        idx = rng.choice(len(h), N_MAX, replace=False)
        h, gps, scene_ids = h[idx], gps[idx], scene_ids[idx]

    h = h - h.mean(axis=0, keepdims=True)
    gps = gps - gps.mean(axis=0, keepdims=True)

    # Top-K scenes
    unique, counts = np.unique(scene_ids, return_counts=True)
    idx_sorted = np.argsort(counts)[::-1]
    top_scenes = unique[idx_sorted[:TOP_K_SCENES]]
    top_scenes = [s for s in top_scenes if (scene_ids == s).sum() >= N_MIN]

    # Also fit ID probe (all-scenes 5-fold) for reference
    id_ridge = Ridge(alpha=ALPHA).fit(h, gps)
    id_r2 = float(r2_score(gps, id_ridge.predict(h), multioutput="uniform_average"))

    loso_r2 = []
    for s_test in top_scenes:
        test_mask = scene_ids == s_test
        train_mask = ~test_mask
        if train_mask.sum() < 1000:
            continue
        X_tr, y_tr = h[train_mask], gps[train_mask]
        X_te, y_te = h[test_mask], gps[test_mask]
        ridge = Ridge(alpha=ALPHA).fit(X_tr, y_tr)
        r = r2_score(y_te, ridge.predict(X_te), multioutput="uniform_average")
        loso_r2.append(float(r))

    loso_r2 = np.array(loso_r2)
    return {
        "n_scenes": int(len(loso_r2)),
        "id_r2": id_r2,
        "loso_r2_mean": float(loso_r2.mean()),
        "loso_r2_std": float(loso_r2.std()),
        "loso_r2_median": float(np.median(loso_r2)),
        "loso_r2_p25": float(np.percentile(loso_r2, 25)),
        "loso_r2_p75": float(np.percentile(loso_r2, 75)),
        "loso_r2_p10": float(np.percentile(loso_r2, 10)),
        "loso_r2_p90": float(np.percentile(loso_r2, 90)),
        "loso_r2_min": float(loso_r2.min()),
        "loso_r2_max": float(loso_r2.max()),
        "frac_negative": float((loso_r2 < 0).mean()),
        "frac_below_05": float((loso_r2 < 0.5).mean()),
        "gap_to_id": float(id_r2 - np.median(loso_r2)),
        "per_scene_r2": loso_r2.tolist(),
    }


def main():
    out = {}
    for cond, fname in CONDS:
        p = NPZ_DIR / fname
        if not p.exists():
            print(f"SKIP {cond}: missing {p}")
            continue
        print(f"=== {cond} ({fname}) ===")
        res = loso_one_condition(p)
        out[cond] = res
        print(f"  ID R²={res['id_r2']:+.3f}, LOSO median={res['loso_r2_median']:+.3f}, gap={res['gap_to_id']:+.3f}, frac<0={res['frac_negative']:.0%}")
        print(f"  LOSO p10/p25/p50/p75/p90: {res['loso_r2_p10']:+.2f} / {res['loso_r2_p25']:+.2f} / {res['loso_r2_median']:+.2f} / {res['loso_r2_p75']:+.2f} / {res['loso_r2_p90']:+.2f}")

    Path("/scratch/wxu/dh-spatial/results/probing_results/loso_5cond_v4.json").write_text(json.dumps(out, indent=2))
    print("\nwrote /scratch/wxu/dh-spatial/results/probing_results/loso_5cond_v4.json")


if __name__ == "__main__":
    main()
