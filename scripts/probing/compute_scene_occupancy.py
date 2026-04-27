r"""
Pre-compute allocentric occupancy grid (free-space mask) per scene by
querying habitat-sim's pathfinder. WJ-C stage 1.

For each unique scene present in the trained-condition rollouts, this
script queries ``env.sim.pathfinder.is_navigable((x, y, z))`` over a
regular 2-D grid in the scene's navigable bounding box (x and z in
world meters; y is fixed at the agent's typical floor height). Output
is a binary occupancy grid (1 = navigable, 0 = obstacle / void).

Used by build_occupancy_targets.py to assemble the per-episode
"traversed-within-2.5m" target masks for the WJ-C decoder training.

Usage:
    python scripts/probing/compute_scene_occupancy.py \
        --config-name pointnav/ddppo_pointnav_blind_gibson \
        --scene-ids-file /scratch/izar/wxu/scene_list.txt \
        --out-dir /scratch/izar/wxu/scene_occupancy \
        --grid-res 0.20

Each scene takes ~10-30 s. ~70 scenes total → ~15-30 min single-V100.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

import src.habitat  # noqa: F401

import habitat
from src.utils.habitat_env import load_habitat_config


def query_scene_occupancy(env: habitat.Env, grid_res: float = 0.20) -> dict:
    """Query the navigable bounds of the loaded scene and produce a
    binary occupancy grid at ``grid_res`` (m/cell)."""
    sim = env.sim
    bounds = sim.pathfinder.get_bounds()  # (lower, upper) → 3-vectors
    lo = np.asarray(bounds[0], dtype=np.float64)
    hi = np.asarray(bounds[1], dtype=np.float64)

    # We need a y-coord to query at. Take the navmesh's "y_min + ε" so
    # the query is on the ground floor. Habitat has multi-floor scenes
    # in some Gibson maps; this gets the lowest floor (acceptable for
    # PointNav agents that start at ground level).
    y_query = float(lo[1]) + 0.10

    x_vals = np.arange(lo[0], hi[0], grid_res)
    z_vals = np.arange(lo[2], hi[2], grid_res)
    H, W = len(z_vals), len(x_vals)
    occ = np.zeros((H, W), dtype=np.uint8)

    for i, z in enumerate(z_vals):
        for j, x in enumerate(x_vals):
            occ[i, j] = 1 if sim.pathfinder.is_navigable([float(x), y_query, float(z)]) else 0

    return {
        "occupancy": occ,            # (H, W) uint8
        "world_lower": [float(lo[0]), float(lo[1]), float(lo[2])],
        "world_upper": [float(hi[0]), float(hi[1]), float(hi[2])],
        "y_query": y_query,
        "grid_res": grid_res,
        "shape": [int(H), int(W)],
    }


def find_episode_for_scene(ds, target_scene: str):
    """Pick any episode in the dataset that uses target_scene (we only
    need it to load the scene mesh; the episode start/goal don't matter)."""
    for ep in ds.episodes:
        if Path(ep.scene_id).stem == target_scene or ep.scene_id == target_scene:
            return ep
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--scene-ids-file", type=Path, required=True,
                    help="One scene id (basename) per line, e.g. 'Adairsville'")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--grid-res", type=float, default=0.20,
                    help="Grid resolution in meters (default 0.20)")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenes = [s.strip() for s in args.scene_ids_file.read_text().splitlines() if s.strip()]
    print(f"Will process {len(scenes)} scenes")

    # Load dataset once.
    cfg_for_lookup = load_habitat_config(args.config_name, "", overrides=[
        f"habitat.dataset.split={args.split}",
        "habitat.environment.iterator_options.shuffle=False",
    ])
    from habitat.datasets import make_dataset
    full_ds = make_dataset(
        id_dataset=cfg_for_lookup.habitat.dataset.type,
        config=cfg_for_lookup.habitat.dataset,
    )
    print(f"Dataset: {len(full_ds.episodes)} episodes total")

    for si, scene in enumerate(scenes):
        out_npz = args.out_dir / f"{scene}.npz"
        out_json = args.out_dir / f"{scene}.json"
        if out_npz.exists() and out_json.exists():
            print(f"  [{si+1}/{len(scenes)}] skip {scene} (exists)")
            continue
        ep = find_episode_for_scene(full_ds, scene)
        if ep is None:
            print(f"  [{si+1}/{len(scenes)}] WARN: no episode for {scene}; skip")
            continue
        target_scene_id = ep.scene_id

        # Build env with this scene pinned.
        config = load_habitat_config(args.config_name, "", overrides=[
            f"habitat.dataset.split={args.split}",
            "habitat.environment.iterator_options.shuffle=False",
            f"habitat.simulator.scene={target_scene_id}",
        ])
        # Pass single-episode dataset so habitat.Env doesn't iterate.
        ds = make_dataset(
            id_dataset=cfg_for_lookup.habitat.dataset.type,
            config=cfg_for_lookup.habitat.dataset,
        )
        ds.episodes = [ep]
        env = habitat.Env(config=config.habitat, dataset=ds)
        _ = env.reset()

        result = query_scene_occupancy(env, grid_res=args.grid_res)
        np.savez_compressed(out_npz, occupancy=result["occupancy"])
        with open(out_json, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "occupancy"},
                      f, indent=2)
        env.close()
        print(f"  [{si+1}/{len(scenes)}] {scene} → shape={result['shape']}, "
              f"navigable={result['occupancy'].mean():.2f}")


if __name__ == "__main__":
    main()
