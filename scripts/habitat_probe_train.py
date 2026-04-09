"""
Habitat linear probe trainer.

Loads probing data collected by habitat_probe_collect.py and trains
linear probes to predict agent pose from frozen LSTM hidden states.

Two probe modes:
  A. **Global probing** — pool data across all scenes, split by episode.
     Uses GPS (relative displacement from start) and compass as targets.
     Scene-independent; works with any amount of data.
  B. **Per-scene probing** — Wijmans et al. methodology. Temporal split
     within each scene. Requires enough per-scene data (≥50 steps).

Probes:
  1. Position (x, z) or GPS — Ridge regression → R², MAE in meters
  2. Heading / compass — Ridge regression → R², angular MAE in degrees
  3. Combined — Ridge regression → overall R²

Usage:
    python scripts/habitat_probe_train.py \
        --data /scratch/izar/$USER/probing_data/blind_gibson.npz \
        --out  /scratch/izar/$USER/probing_results/blind_gibson.json
"""

import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description="Train linear probes on Habitat agent hidden states")
    p.add_argument("--data", required=True, help="Path to .npz from habitat_probe_collect.py")
    p.add_argument("--out", default=None, help="Output .json path")
    p.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization")
    p.add_argument("--pca-dim", type=int, default=32, help="PCA dims (0=skip)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--min-steps-scene", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fit_probe(X_tr, X_te, Y_tr, Y_te, alpha):
    """Generic Ridge probe: fit on train, eval on test."""
    reg = Ridge(alpha=alpha)
    reg.fit(X_tr, Y_tr)
    pred = reg.predict(X_te)
    r2 = float(np.clip(r2_score(Y_te, pred, multioutput="uniform_average"), -10, 1))
    mae = float(mean_absolute_error(Y_te, pred))
    return r2, mae, pred


def angular_mae(pred_sincos, true_headings):
    """Compute angular MAE from predicted sin/cos vs true headings."""
    pred_angle = np.arctan2(pred_sincos[:, 0], pred_sincos[:, 1])
    diff = np.abs(np.arctan2(np.sin(pred_angle - true_headings),
                              np.cos(pred_angle - true_headings)))
    return float(np.degrees(np.mean(diff)))


def prepare_features(H_tr, H_te, pca_dim):
    """Standardize + optional PCA."""
    scaler = StandardScaler()
    H_tr = scaler.fit_transform(H_tr)
    H_te = scaler.transform(H_te)
    if pca_dim > 0 and H_tr.shape[1] > pca_dim:
        n_comp = min(pca_dim, H_tr.shape[0], H_tr.shape[1])
        pca = PCA(n_components=n_comp)
        H_tr = pca.fit_transform(H_tr)
        H_te = pca.transform(H_te)
    return H_tr, H_te


# ─────────────── Global probing (GPS-based, scene-independent) ───────────────

def global_probe(H, gps, compass, ep_ids, alpha, pca_dim, train_frac, seed):
    """Pool all data, split by episode, probe GPS+compass from hidden state."""
    rng = np.random.RandomState(seed)
    unique_eps = np.unique(ep_ids)
    rng.shuffle(unique_eps)
    split = int(len(unique_eps) * train_frac)
    train_eps = set(unique_eps[:split].tolist())
    train_mask = np.array([e in train_eps for e in ep_ids])

    H_tr, H_te = H[train_mask], H[~train_mask]
    gps_tr, gps_te = gps[train_mask], gps[~train_mask]
    comp_tr, comp_te = compass[train_mask], compass[~train_mask]

    H_tr, H_te = prepare_features(H_tr, H_te, pca_dim)

    # GPS probe (relative x, z displacement)
    gps_r2, gps_mae, _ = fit_probe(H_tr, H_te, gps_tr, gps_te, alpha)

    # Compass probe (sin/cos of heading)
    comp_sincos_tr = np.hstack([np.sin(comp_tr), np.cos(comp_tr)])
    comp_sincos_te = np.hstack([np.sin(comp_te), np.cos(comp_te)])
    comp_r2, _, comp_pred = fit_probe(H_tr, H_te, comp_sincos_tr, comp_sincos_te, alpha)
    comp_mae_deg = angular_mae(comp_pred, comp_te.ravel())

    # Combined
    Y_tr = np.hstack([gps_tr, comp_sincos_tr])
    Y_te = np.hstack([gps_te, comp_sincos_te])
    comb_r2, _, _ = fit_probe(H_tr, H_te, Y_tr, Y_te, alpha)

    return {
        "gps_r2": float(gps_r2),
        "gps_mae_m": float(gps_mae),
        "compass_r2": float(comp_r2),
        "compass_mae_deg": float(comp_mae_deg),
        "combined_r2": float(comb_r2),
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "n_train_eps": int(split),
        "n_test_eps": int(len(unique_eps) - split),
    }


# ─────────────── Per-scene probing (absolute position) ───────────────────────

def per_scene_probe(H, P, theta, scene_ids, alpha, train_frac, min_steps, pca_dim):
    """Wijmans-style per-scene probing with temporal split."""
    unique_scenes = np.unique(scene_ids)
    scene_details = []

    for sid in unique_scenes:
        mask = scene_ids == sid
        n_steps = mask.sum()
        if n_steps < min_steps:
            continue

        h_s, p_s, t_s = H[mask], P[mask], theta[mask]
        split = int(len(h_s) * train_frac)
        if split < 15 or (len(h_s) - split) < 8:
            continue

        H_tr, H_te = prepare_features(h_s[:split], h_s[split:], pca_dim)
        P_tr, P_te = p_s[:split], p_s[split:]
        t_tr, t_te = t_s[:split], t_s[split:]

        # Position probe (x, z)
        pos_r2, pos_mae, _ = fit_probe(H_tr, H_te, P_tr[:, [0, 2]], P_te[:, [0, 2]], alpha)

        # Heading probe (sin/cos)
        Y_tr_h = np.stack([np.sin(t_tr), np.cos(t_tr)], axis=1)
        Y_te_h = np.stack([np.sin(t_te), np.cos(t_te)], axis=1)
        head_r2, _, head_pred = fit_probe(H_tr, H_te, Y_tr_h, Y_te_h, alpha)
        head_mae = angular_mae(head_pred, t_te)

        scene_details.append({
            "scene_id": int(sid), "n_steps": int(n_steps),
            "pos_r2": float(pos_r2), "pos_mae": float(pos_mae),
            "head_r2": float(head_r2), "head_mae": float(head_mae),
        })

    return scene_details


def main():
    args = parse_args()
    if args.out is None:
        args.out = args.data.replace(".npz", "_results.json")

    print(f"Loading {args.data} ...")
    data = np.load(args.data)
    H = data["hidden_states"]
    P = data["positions"]
    theta = data["headings"]
    ep_ids = data["episode_ids"]
    scene_ids = data["scene_ids"]
    gps = data["gps"]
    compass = data["compass"]

    n_eps = len(np.unique(ep_ids))
    n_scenes = len(np.unique(scene_ids))
    print(f"  Steps: {len(H)}, Hidden dim: {H.shape[1]}")
    print(f"  Episodes: {n_eps}, Scenes: {n_scenes}")
    print(f"  PCA dim: {args.pca_dim}, Alpha: {args.alpha}")

    # ═══════ Mode A: Global GPS/Compass probing ═══════
    print(f"\n{'='*70}")
    print("  MODE A: Global probing (GPS + Compass, episode-level split)")
    print(f"{'='*70}")
    global_res = global_probe(
        H, gps, compass, ep_ids,
        alpha=args.alpha, pca_dim=args.pca_dim,
        train_frac=args.train_frac, seed=args.seed,
    )
    print(f"  Train: {global_res['n_train']} steps ({global_res['n_train_eps']} eps)")
    print(f"  Test:  {global_res['n_test']} steps ({global_res['n_test_eps']} eps)")
    print(f"  GPS R²={global_res['gps_r2']:+.4f}  MAE={global_res['gps_mae_m']:.3f}m")
    print(f"  Compass R²={global_res['compass_r2']:+.4f}  MAE={global_res['compass_mae_deg']:.1f}°")
    print(f"  Combined R²={global_res['combined_r2']:+.4f}")

    # ═══════ Mode B: Per-scene absolute position probing ═══════
    print(f"\n{'='*70}")
    print(f"  MODE B: Per-scene probing (absolute pos, temporal split, min_steps={args.min_steps_scene})")
    print(f"{'='*70}")
    scene_details = per_scene_probe(
        H, P, theta, scene_ids,
        alpha=args.alpha, train_frac=args.train_frac,
        min_steps=args.min_steps_scene, pca_dim=args.pca_dim,
    )

    if scene_details:
        for sd in scene_details:
            print(f"  Scene {sd['scene_id']:3d}: {sd['n_steps']:4d} steps  |  "
                  f"Pos R²={sd['pos_r2']:+.3f} MAE={sd['pos_mae']:.2f}m  |  "
                  f"Head R²={sd['head_r2']:+.3f} MAE={sd['head_mae']:.1f}°")
        med_pos_r2 = float(np.median([s["pos_r2"] for s in scene_details]))
        med_pos_mae = float(np.median([s["pos_mae"] for s in scene_details]))
        med_head_r2 = float(np.median([s["head_r2"] for s in scene_details]))
        med_head_mae = float(np.median([s["head_mae"] for s in scene_details]))
    else:
        print("  No scenes with enough data.")
        med_pos_r2 = med_pos_mae = med_head_r2 = med_head_mae = None

    # ═══════ Save results ═══════
    results = {
        "source": args.data,
        "n_steps": int(len(H)),
        "hidden_dim": int(H.shape[1]),
        "n_episodes": int(n_eps),
        "n_scenes": int(n_scenes),
        "alpha": args.alpha,
        "pca_dim": args.pca_dim,
        "global_probe": global_res,
        "per_scene": {
            "n_scenes_probed": len(scene_details),
            "median_pos_r2": med_pos_r2,
            "median_pos_mae": med_pos_mae,
            "median_head_r2": med_head_r2,
            "median_head_mae": med_head_mae,
            "scenes": scene_details,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  GLOBAL:    GPS R²={global_res['gps_r2']:+.4f}  MAE={global_res['gps_mae_m']:.3f}m  |  "
          f"Compass R²={global_res['compass_r2']:+.4f}  MAE={global_res['compass_mae_deg']:.1f}°")
    if med_pos_r2 is not None:
        print(f"  PER-SCENE: Pos R²={med_pos_r2:+.4f}  MAE={med_pos_mae:.3f}m  |  "
              f"Head R²={med_head_r2:+.4f}  MAE={med_head_mae:.1f}°  ({len(scene_details)} scenes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
