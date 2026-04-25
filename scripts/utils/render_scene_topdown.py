"""
Render the Habitat top-down occupancy map for a chosen Gibson val
scene + episode, save as PNG plus a JSON sidecar containing the
world-frame extent (lower / upper bounds) so we can later project
world-frame trajectories onto the pixel grid.

Used to produce the floor-plan background for the §1 setup figure
(Fig 1d, trajectory_overlay_topdown.pdf).

Usage:
    python scripts/utils/render_scene_topdown.py \\
        --config-name pointnav/ddppo_pointnav_blind_gibson \\
        --episode-id 414 \\
        --out-png /tmp/scene92_topdown.png \\
        --out-json /tmp/scene92_topdown.json
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

import src.habitat  # noqa: F401  registers configs / policies

import habitat
from habitat.utils.visualizations import maps

from src.utils.habitat_env import load_habitat_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--episode-id", type=int, default=None,
                    help="Episode id within the split (numerical).")
    ap.add_argument("--start-x", type=float, default=None,
                    help="Match by start_position; use this together with "
                         "--start-y, --start-z (most robust for cross-data "
                         "consistency).")
    ap.add_argument("--start-y", type=float, default=None)
    ap.add_argument("--start-z", type=float, default=None)
    ap.add_argument("--split", default="train",
                    help="Dataset split (default: train, matches probing data).")
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--map-resolution", type=int, default=1024)
    args = ap.parse_args()

    # Step 1: load the dataset WITHOUT creating an env (avoids
    # habitat.Env binding to the first scene at construction time).
    cfg_for_lookup = load_habitat_config(args.config_name, "", overrides=[
        f"habitat.dataset.split={args.split}",
        "habitat.environment.iterator_options.shuffle=False",
    ])

    from habitat.datasets import make_dataset
    ds = make_dataset(
        id_dataset=cfg_for_lookup.habitat.dataset.type,
        config=cfg_for_lookup.habitat.dataset,
    )
    print(f"Dataset loaded: {len(ds.episodes)} episodes total")

    # Step 2: pick the target episode.
    if args.start_x is not None:
        target = np.array([args.start_x, args.start_y, args.start_z],
                          dtype=np.float32)
        best, best_d = None, float("inf")
        for e in ds.episodes:
            sp = np.asarray(e.start_position, dtype=np.float32)
            d = float(np.linalg.norm(sp - target))
            if d < best_d:
                best_d, best = d, e
        if best is None or best_d > 0.5:
            raise RuntimeError(
                f"No episode with start_position near {target.tolist()} "
                f"in split={args.split} (closest dist {best_d:.3f}m)"
            )
        matching = [best]
        print(f"Matched start_position {target.tolist()} → "
              f"episode_id={best.episode_id}, scene={best.scene_id}, "
              f"distance={best_d:.4f}m")
    else:
        target_id = str(args.episode_id)
        matching = [e for e in ds.episodes if str(e.episode_id) == target_id]
        if not matching:
            raise RuntimeError(
                f"Episode {args.episode_id} not found in split={args.split}"
            )

    # Step 3: rebuild config with the SCENE pointed at the matching
    # episode's scene_id.  This forces habitat.Env.__init__ to load
    # the right scene mesh from the start, bypassing the multi-scene
    # iterator entirely.
    target_scene = matching[0].scene_id
    target_scene_basename = Path(target_scene).stem  # e.g. "E9uDoFAP3SH"
    print(f"Target scene basename: {target_scene_basename}")

    config = load_habitat_config(args.config_name, "", overrides=[
        f"habitat.dataset.split={args.split}",
        "habitat.environment.iterator_options.shuffle=False",
        f"habitat.simulator.scene={target_scene}",
    ])

    # Filter dataset to just our target episode and pass it explicitly
    # so habitat.Env doesn't re-load the full multi-scene episode list.
    ds.episodes = matching

    env = habitat.Env(config=config.habitat, dataset=ds)
    obs = env.reset()
    ep = env.current_episode
    print(f"Loaded episode {ep.episode_id} in scene {ep.scene_id}")

    print(f"Loaded episode {ep.episode_id} in scene {ep.scene_id}")
    print(f"  start: {ep.start_position}, goal: {ep.goals[0].position}")

    # Top-down map (1 = navigable, 0 = obstacle)
    top_down = maps.get_topdown_map_from_sim(
        env.sim, map_resolution=args.map_resolution
    )
    print(f"Top-down map shape: {top_down.shape}")

    # World-frame bounds.  Habitat maps API uses (lower, upper) in y/z
    # depending on version.  Use sim.pathfinder.get_bounds() as a safe
    # fallback for the world-frame extent of the navigable mesh.
    bounds = env.sim.pathfinder.get_bounds()
    lower = list(map(float, bounds[0]))
    upper = list(map(float, bounds[1]))
    print(f"Sim bounds (lower → upper): {lower} → {upper}")

    # Colourise: maps.colorize_topdown_map produces an (H, W, 3) RGB
    # image.  Default colourmap: dark grey = obstacle, light grey =
    # navigable, blue/green for path overlays (we don't draw any here).
    colour_map = maps.colorize_topdown_map(top_down)

    # Save PNG via PIL (avoid matplotlib dependency in the compute env)
    try:
        from PIL import Image
        Image.fromarray(colour_map).save(str(args.out_png))
    except ImportError:
        import imageio
        imageio.imwrite(str(args.out_png), colour_map)
    print(f"Wrote {args.out_png}")

    # JSON sidecar.  We deliberately store everything we might need for
    # world-frame → pixel-frame projection (height, width, world bounds).
    sidecar = {
        "scene_id": ep.scene_id,
        "episode_id": ep.episode_id,
        "start_position": list(map(float, ep.start_position)),
        "goal_position": list(map(float, ep.goals[0].position)),
        "topdown_height": int(colour_map.shape[0]),
        "topdown_width": int(colour_map.shape[1]),
        "world_lower_bound": lower,   # (x, y, z) world-frame meters
        "world_upper_bound": upper,
        "map_resolution": int(args.map_resolution),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Wrote {args.out_json}")

    env.close()


if __name__ == "__main__":
    main()
