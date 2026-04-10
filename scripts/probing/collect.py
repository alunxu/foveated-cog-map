"""
Probing data collector.

Runs eval rollouts on a trained PointNav checkpoint and records
LSTM hidden states (all layers, h and c) paired with ground-truth
agent pose at every timestep.

Output: an .npz file containing:
    h_layers       (N, num_lstm_layers, hidden_size) — h states for all layers
    c_layers       (N, num_lstm_layers, hidden_size) — c states for all layers
    hidden_states  (N, hidden_size)     — top-layer h (backward compat)
    positions      (N, 3)               — absolute (x, y, z) world coords
    headings       (N,)                 — heading angle in radians
    gps            (N, 2)               — episodic GPS sensor readout
    compass        (N,)                 — episodic compass sensor readout
    distance_to_goal (N,)               — Euclidean distance to goal
    goal_positions (N, 3)               — goal location in world coords
    step_in_episode (N,)                — timestep index within episode
    episode_ids    (N,)                 — which episode each step belongs to
    scene_ids      (N,)                 — scene id string index per step
    local_occupancy (N, G, G)           — [optional] local navigability grid (1=free, 0=wall)

Usage on cluster:
    python scripts/probing/collect.py \
        --config-name pointnav/ddppo_pointnav_blind_gibson \
        --ckpt /scratch/izar/$USER/habitat_checkpoints/blind_gibson/ckpt.9.pth \
        --episodes 500 \
        --out /scratch/izar/$USER/probing_data/blind_gibson.npz

This script does NOT use the habitat-baselines evaluator. It directly
creates a single Habitat environment, loads the policy, and steps
through episodes — much simpler and gives us full access to the
simulator state.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add project root so src.habitat is importable (registers custom policies)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

import src.habitat  # noqa: F401 — registers custom policies + sensors
from src.utils.habitat_env import heading_from_quaternion, load_habitat_config, load_policy

import habitat


def query_local_occupancy(sim, agent_pos, grid_size=5.0, grid_res=0.25):
    """Query navigability around the agent to build a local occupancy grid.

    Returns a binary 2D array (1=navigable, 0=obstacle) centered on the
    agent's (x, z) position in world coordinates.

    Args:
        sim: Habitat simulator instance (has sim.pathfinder)
        agent_pos: (3,) array — agent position [x, y, z]
        grid_size: side length of the grid in meters
        grid_res: resolution in meters per cell

    Returns:
        occ: (n_cells, n_cells) float32 array, 1=navigable, 0=obstacle
    """
    n = int(grid_size / grid_res)
    half = grid_size / 2.0
    y = agent_pos[1]  # keep agent's height

    occ = np.zeros((n, n), dtype=np.float32)
    for ix in range(n):
        wx = agent_pos[0] - half + (ix + 0.5) * grid_res
        for iz in range(n):
            wz = agent_pos[2] - half + (iz + 0.5) * grid_res
            point = np.array([wx, y, wz])
            if sim.pathfinder.is_navigable(point):
                occ[ix, iz] = 1.0
    return occ


def parse_args():
    p = argparse.ArgumentParser(description="Collect probing data from a Habitat agent")
    p.add_argument("--config-name", required=True, help="Hydra config name (e.g. pointnav/ddppo_pointnav_blind_gibson)")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pth file")
    p.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--collect-occupancy", action="store_true",
                   help="Collect local occupancy grids (5m×5m, 0.25m res = 20×20)")
    p.add_argument("--occ-grid-size", type=float, default=5.0,
                   help="Occupancy grid side length in meters")
    p.add_argument("--occ-grid-res", type=float, default=0.25,
                   help="Occupancy grid resolution in meters")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Config + Environment + Policy ----
    config = load_habitat_config(args.config_name, args.ckpt, overrides=[
        "habitat.dataset.split=train",
        "habitat.environment.iterator_options.shuffle=True",
        "habitat.environment.iterator_options.group_by_scene=False",
    ])

    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.out}")
    print(f"Device: {device}")

    env = habitat.Env(config=config.habitat)
    _ = env.reset()  # needed to infer observation spaces

    policy, hidden_size, num_recurrent_layers, rnn_is_lstm = load_policy(
        config, env, args.ckpt, device,
    )

    print(f"Hidden size: {hidden_size}, num_recurrent_layers: {num_recurrent_layers}")
    print(f"LSTM: {rnn_is_lstm}")

    # Number of actual LSTM layers (e.g. 3 for num_recurrent_layers=6)
    if rnn_is_lstm:
        n_lstm_layers = num_recurrent_layers // 2
    else:
        n_lstm_layers = num_recurrent_layers
    print(f"Actual LSTM layers: {n_lstm_layers}")

    # ---- Collect data ----
    all_h_layers = []       # (N, n_lstm_layers, hidden_size)
    all_c_layers = []       # (N, n_lstm_layers, hidden_size) — LSTM only
    all_hidden = []         # (N, hidden_size) — top-layer h for backward compat
    all_positions = []
    all_headings = []
    all_gps = []
    all_compass = []
    all_distance_to_goal = []
    all_goal_positions = []
    all_step_in_episode = []
    all_episode_ids = []
    all_scene_ids = []
    all_local_occupancy = []  # optional: (N, grid_h, grid_w)
    scene_id_to_idx = {}

    for ep in range(args.episodes):
        obs = env.reset()
        episode = env.current_episode

        scene_id = episode.scene_id
        episode_id = episode.episode_id
        if scene_id not in scene_id_to_idx:
            scene_id_to_idx[scene_id] = len(scene_id_to_idx)
        scene_idx = scene_id_to_idx[scene_id]

        # Goal position from episode definition
        goal_pos = np.array(episode.goals[0].position, dtype=np.float32)

        # Init recurrent hidden states — batch-first: (batch, num_layers, hidden)
        rnn_hidden = torch.zeros(
            1, num_recurrent_layers, hidden_size, device=device
        )
        prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
        not_done_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

        done = False
        step = 0
        while not done:
            # Build batched observation
            batch = {k: torch.from_numpy(np.expand_dims(v, 0)).to(device)
                     for k, v in obs.items()}

            with torch.no_grad():
                action_data = policy.act(
                    batch,
                    rnn_hidden,
                    prev_action,
                    not_done_mask,
                    deterministic=False,
                )

            # Extract ALL LSTM layer states.
            # new_rnn_hidden is batch-first: (1, num_recurrent_layers, hidden).
            # For LSTM with num_recurrent_layers=6 (= 3 actual layers × 2):
            #   h indices: [0, 2, 4],  c indices: [1, 3, 5]
            new_rnn_hidden = action_data.rnn_hidden_states
            if rnn_is_lstm:
                h_all = new_rnn_hidden[0, 0::2].cpu().numpy()   # (n_lstm_layers, hidden)
                c_all = new_rnn_hidden[0, 1::2].cpu().numpy()   # (n_lstm_layers, hidden)
                top_h = h_all[-1]                                # (hidden,)
            else:
                h_all = new_rnn_hidden[0].cpu().numpy()
                c_all = np.zeros_like(h_all)
                top_h = h_all[-1]

            all_h_layers.append(h_all)
            all_c_layers.append(c_all)
            all_hidden.append(top_h)

            # Ground-truth pose from simulator
            agent_state = env.sim.get_agent_state()
            pos = agent_state.position  # (3,) numpy: x, y, z
            heading = heading_from_quaternion(agent_state.rotation)
            all_positions.append(pos.copy())
            all_headings.append(heading)

            # Distance to goal
            dist = np.linalg.norm(pos - goal_pos)
            all_distance_to_goal.append(dist)
            all_goal_positions.append(goal_pos.copy())

            # Local occupancy grid (optional — adds ~20ms/step)
            if args.collect_occupancy:
                occ = query_local_occupancy(
                    env.sim, pos,
                    grid_size=args.occ_grid_size,
                    grid_res=args.occ_grid_res,
                )
                all_local_occupancy.append(occ)

            # Episodic sensor readouts
            if "gps" in obs:
                all_gps.append(obs["gps"].copy())
            if "compass" in obs:
                all_compass.append(obs["compass"].copy().flatten())

            all_step_in_episode.append(step)
            all_episode_ids.append(ep)
            all_scene_ids.append(scene_idx)

            # Step environment
            action_int = action_data.env_actions[0].item()
            obs = env.step(action_int)
            done = env.episode_over

            # Update RNN state and prev action
            rnn_hidden = new_rnn_hidden
            prev_action = action_data.actions
            not_done_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

            step += 1

        if (ep + 1) % 10 == 0:
            total_steps = len(all_hidden)
            print(f"  Episode {ep+1}/{args.episodes} done ({step} steps, {total_steps} total steps)")

    env.close()

    # ---- Save ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    N = len(all_hidden)
    save_dict = {
        # All-layer hidden states
        "h_layers": np.array(all_h_layers, dtype=np.float32),          # (N, n_layers, hidden)
        "c_layers": np.array(all_c_layers, dtype=np.float32),          # (N, n_layers, hidden)
        "hidden_states": np.array(all_hidden, dtype=np.float32),       # (N, hidden) — compat
        # Spatial ground truth
        "positions": np.array(all_positions, dtype=np.float32),
        "headings": np.array(all_headings, dtype=np.float32),
        "distance_to_goal": np.array(all_distance_to_goal, dtype=np.float32),
        "goal_positions": np.array(all_goal_positions, dtype=np.float32),
        # Episode metadata
        "step_in_episode": np.array(all_step_in_episode, dtype=np.int32),
        "episode_ids": np.array(all_episode_ids, dtype=np.int32),
        "scene_ids": np.array(all_scene_ids, dtype=np.int32),
    }
    if all_gps:
        save_dict["gps"] = np.array(all_gps, dtype=np.float32)
    if all_compass:
        save_dict["compass"] = np.array(all_compass, dtype=np.float32)
    if all_local_occupancy:
        save_dict["local_occupancy"] = np.array(all_local_occupancy, dtype=np.float32)
        save_dict["occ_grid_size"] = np.float32(args.occ_grid_size)
        save_dict["occ_grid_res"] = np.float32(args.occ_grid_res)

    np.savez_compressed(args.out, **save_dict)

    print(f"\nSaved {N} steps from {args.episodes} episodes to {args.out}")
    print(f"  h_layers:      ({N}, {n_lstm_layers}, {hidden_size})")
    print(f"  c_layers:      ({N}, {n_lstm_layers}, {hidden_size})")
    print(f"  hidden_states: ({N}, {hidden_size})")
    print(f"  positions:     ({N}, 3)")
    if all_local_occupancy:
        gs = int(args.occ_grid_size / args.occ_grid_res)
        print(f"  occupancy:     ({N}, {gs}, {gs})")
    print(f"  scenes:        {len(scene_id_to_idx)} unique")

    # Save scene_id mapping for reference
    mapping_path = args.out.replace(".npz", "_scenes.txt")
    with open(mapping_path, "w") as f:
        for sid, idx in sorted(scene_id_to_idx.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{sid}\n")
    print(f"  scene mapping: {mapping_path}")


if __name__ == "__main__":
    main()
