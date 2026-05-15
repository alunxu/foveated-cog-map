from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

# Import build_dataset from the decoder script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_occupancy_decoder import build_dataset  # noqa: E402


def iou_score(preds, Y, M):
    preds = preds > 0.5
    gt = Y > 0.5
    mask = M > 0.5

    inter = (preds & gt & mask).sum(axis=(1, 2))
    union = ((preds | gt) & mask).sum(axis=(1, 2))

    return inter / np.maximum(union, 1)


def eval_condition(cond_name, hidden_npz, scene_occ_dir, grid_size, grid_res, dilate_m, n_folds, seed):
    class Args:
        pass

    args = Args()
    args.hidden_npz = Path(hidden_npz)
    args.scene_occ_dir = Path(scene_occ_dir)
    args.grid_size = grid_size
    args.grid_res = grid_res
    args.dilate_m = dilate_m

    print("=" * 70)
    print(f"Baseline evaluation: {cond_name}")
    print("=" * 70)

    X, Y, M = build_dataset(args)
    n = len(Y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results = {
        "condition": cond_name,
        "n_episodes": int(n),
        "baselines": {
            "all_free": [],
            "train_prior_map": [],
            "random_global_rate": [],
        },
    }

    rng = np.random.default_rng(seed)

    for fi, (tr, te) in enumerate(kf.split(np.arange(n))):
        Ytr, Mtr = Y[tr], M[tr]
        Yte, Mte = Y[te], M[te]

        # Baseline 1: predict every evaluated cell as free/navigable.
        pred_all_free = np.ones_like(Yte, dtype=np.float32)
        all_free_iou = iou_score(pred_all_free, Yte, Mte).mean()

        # Baseline 2: train-set average occupancy map, thresholded at 0.5.
        # This tests whether a fixed spatial prior explains the result.
        denom = Mtr.sum(axis=0)
        global_rate = (Ytr * Mtr).sum() / max(Mtr.sum(), 1)
        prior_map = np.where(
            denom > 0,
            (Ytr * Mtr).sum(axis=0) / np.maximum(denom, 1),
            global_rate,
        )
        pred_prior = np.broadcast_to((prior_map > 0.5).astype(np.float32), Yte.shape)
        prior_iou = iou_score(pred_prior, Yte, Mte).mean()

        # Baseline 3: random Bernoulli using global train free-space rate.
        pred_random = rng.binomial(1, global_rate, size=Yte.shape).astype(np.float32)
        random_iou = iou_score(pred_random, Yte, Mte).mean()

        results["baselines"]["all_free"].append(float(all_free_iou))
        results["baselines"]["train_prior_map"].append(float(prior_iou))
        results["baselines"]["random_global_rate"].append(float(random_iou))

        print(
            f"fold {fi}: "
            f"all_free={all_free_iou:.3f} | "
            f"prior={prior_iou:.3f} | "
            f"random={random_iou:.3f}"
        )

    for name, vals in results["baselines"].items():
        results["baselines"][name] = {
            "folds": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-occ-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--grid-res", type=float, default=0.5)
    ap.add_argument("--dilate-m", type=float, default=2.5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    user = os.environ["USER"]
    conds = {
        "blind": f"/scratch/izar/{user}/probing_data/blind_gibson_det.npz",
        "matched128": f"/scratch/izar/{user}/probing_data/matched_gibson_det.npz",
        "uniform": f"/scratch/izar/{user}/probing_data/uniform_gibson_det.npz",
        "foveated": f"/scratch/izar/{user}/probing_data/foveated_gibson_det.npz",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for cond, hidden_npz in conds.items():
        res = eval_condition(
            cond,
            hidden_npz,
            args.scene_occ_dir,
            args.grid_size,
            args.grid_res,
            args.dilate_m,
            args.n_folds,
            args.seed,
        )
        all_results[cond] = res

        out = args.out_dir / f"{cond}_occupancy_baselines.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"Wrote {out}")

    summary = args.out_dir / "occupancy_baselines_summary.json"
    summary.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
