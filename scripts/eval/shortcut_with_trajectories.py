"""
Shortcut discovery eval that ALSO saves per-episode trajectories.
Companion to shortcut.py — same protocol, but writes per-episode
(positions, dtg) for both reset and persistent conditions.

We need this for the paper's "having vs using" trajectory figure
(Phase B): visualize a same-scene pair where the persistent-memory
LSTM stays "locked" on the previous goal even though the new goal
has been provided as input, vs the reset-memory case where the agent
solves the second episode efficiently.

Output: <out-dir>/<run_name>_trajectories.npz containing per-scene
per-episode-pair trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

import src.habitat  # noqa: F401
from src.utils.habitat_env import (
    heading_from_quaternion, compute_spl, load_habitat_config, load_policy,
)

import habitat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes-per-scene", type=int, default=10)
    p.add_argument("--max-scenes", type=int, default=20)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-traj-npz", required=True,
                   help="NPZ with per-episode trajectories")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_episode(env, policy, rnn_hidden, prev_action, not_done_mask, device,
                rnn_is_lstm, num_recurrent_layers):
    obs = env.reset()
    episode = env.current_episode
    start_pos = np.array(episode.start_position)
    goal_pos = np.array(episode.goals[0].position)
    geodesic = env.sim.geodesic_distance(start_pos, goal_pos)

    positions: list[np.ndarray] = []
    dtg_list: list[float] = []
    path_length = 0.0
    prev_pos = start_pos.copy()
    done = False
    steps = 0
    max_steps = 2000

    while not done and steps < max_steps:
        # Record state at the *start* of the step
        agent_state = env.sim.get_agent_state()
        positions.append(agent_state.position.copy())
        dtg = env.sim.geodesic_distance(agent_state.position, goal_pos)
        dtg_list.append(float(dtg))

        batch = {k: torch.from_numpy(np.expand_dims(v, 0)).to(device)
                 for k, v in obs.items()}

        with torch.no_grad():
            action_data = policy.act(
                batch, rnn_hidden, prev_action, not_done_mask,
                deterministic=True,
            )

        rnn_hidden = action_data.rnn_hidden_states
        prev_action = action_data.actions
        not_done_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

        action_int = action_data.env_actions[0].item()
        obs = env.step(action_int)

        cur_pos = env.sim.get_agent_state().position
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos = cur_pos.copy()

        done = env.episode_over
        steps += 1

    final_pos = env.sim.get_agent_state().position
    dist_to_goal = float(np.linalg.norm(final_pos - goal_pos))
    success = bool((action_int == 0) and (dist_to_goal < 0.2))
    spl = compute_spl(success, path_length, geodesic)

    return {
        "success": success,
        "spl": float(spl),
        "path_length": float(path_length),
        "geodesic": float(geodesic),
        "steps": steps,
        "dist_to_goal": dist_to_goal,
        "positions": np.asarray(positions, dtype=np.float32),  # (T, 3)
        "dtg": np.asarray(dtg_list, dtype=np.float32),
        "start_position": start_pos.astype(np.float32),
        "goal_position": goal_pos.astype(np.float32),
    }, rnn_hidden, prev_action, not_done_mask


def main():
    args = parse_args()
    device = torch.device(args.device)

    config = load_habitat_config(args.config_name, args.ckpt, overrides=[
        "habitat.dataset.split=train",
        "habitat.environment.iterator_options.shuffle=False",
        "habitat.environment.max_episode_steps=2000",
    ])

    env = habitat.Env(config=config.habitat)
    _ = env.reset()

    policy, hidden_size, num_recurrent_layers, rnn_is_lstm = load_policy(
        config, env, args.ckpt, device,
    )

    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Episodes/scene: {args.episodes_per_scene}, max scenes: {args.max_scenes}")

    all_episodes = env.episodes
    scene_episodes = defaultdict(list)
    for ep in all_episodes:
        scene_episodes[ep.scene_id].append(ep)

    scenes = list(scene_episodes.keys())
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]

    print(f"Scenes: {len(scenes)}")

    all_results = []
    # We collect trajectories in a flat list, one entry per (scene, episode_index, condition).
    traj_records: list[dict] = []

    for si, scene_id in enumerate(scenes):
        eps = scene_episodes[scene_id][:args.episodes_per_scene]
        if len(eps) < 2:
            continue

        scene_name = os.path.basename(scene_id).split(".")[0]
        print(f"\n  Scene {si+1}/{len(scenes)}: {scene_name} ({len(eps)} episodes)")

        for condition in ["reset", "persistent"]:
            env._episode_iterator = iter(eps)
            rnn_hidden = torch.zeros(1, num_recurrent_layers, hidden_size, device=device)
            prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
            not_done_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

            ep_results = []
            for ei in range(len(eps)):
                if condition == "reset":
                    rnn_hidden = torch.zeros(1, num_recurrent_layers, hidden_size, device=device)
                    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
                    not_done_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

                metrics, rnn_hidden, prev_action, not_done_mask = run_episode(
                    env, policy, rnn_hidden, prev_action, not_done_mask,
                    device, rnn_is_lstm, num_recurrent_layers,
                )
                ep_results.append(metrics)

                # Save trajectory
                traj_records.append({
                    "scene": scene_name,
                    "scene_idx": si,
                    "ep_idx": ei,
                    "condition": condition,
                    "positions": metrics["positions"],
                    "dtg": metrics["dtg"],
                    "start": metrics["start_position"],
                    "goal": metrics["goal_position"],
                    "success": int(metrics["success"]),
                    "spl": float(metrics["spl"]),
                    "geodesic": float(metrics["geodesic"]),
                    "path_length": float(metrics["path_length"]),
                    "steps": int(metrics["steps"]),
                })

            successes = [r["success"] for r in ep_results]
            spls = [float(r["spl"]) for r in ep_results]
            all_results.append({
                "scene": scene_name,
                "condition": condition,
                "n_episodes": len(eps),
                "success_rate": float(np.mean(successes)),
                "mean_spl": float(np.mean(spls)),
                "per_episode_spl": spls,
                "per_episode_success": successes,
            })
            print(f"    [{condition[:1].upper()}] success={np.mean(successes):.2f}  spl={np.mean(spls):.3f}")

    env.close()

    # ---- Summary JSON ----
    reset_spls = [r["mean_spl"] for r in all_results if r["condition"] == "reset"]
    persist_spls = [r["mean_spl"] for r in all_results if r["condition"] == "persistent"]
    reset_success = [r["success_rate"] for r in all_results if r["condition"] == "reset"]
    persist_success = [r["success_rate"] for r in all_results if r["condition"] == "persistent"]

    summary = {
        "config": args.config_name,
        "checkpoint": args.ckpt,
        "n_scenes": len(scenes),
        "episodes_per_scene": args.episodes_per_scene,
        "reset_mean_spl": float(np.mean(reset_spls)) if reset_spls else None,
        "persistent_mean_spl": float(np.mean(persist_spls)) if persist_spls else None,
        "reset_mean_success": float(np.mean(reset_success)) if reset_success else None,
        "persistent_mean_success": float(np.mean(persist_success)) if persist_success else None,
        "cognitive_map_spl_benefit": (
            float(np.mean(persist_spls) - np.mean(reset_spls)) if persist_spls else 0.0
        ),
        "per_scene": all_results,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {args.out_json}")

    # ---- Save trajectories NPZ ----
    # Padded arrays, one entry per record.  Variable-length sequences
    # are stored as object arrays (dtype=object) for flexibility.
    os.makedirs(os.path.dirname(args.out_traj_npz) or ".", exist_ok=True)
    n = len(traj_records)
    save_dict = {
        "scenes":     np.array([r["scene"] for r in traj_records]),
        "scene_idx":  np.array([r["scene_idx"] for r in traj_records], dtype=np.int32),
        "ep_idx":     np.array([r["ep_idx"] for r in traj_records], dtype=np.int32),
        "conditions": np.array([r["condition"] for r in traj_records]),
        "positions":  np.array([r["positions"] for r in traj_records], dtype=object),
        "dtg":        np.array([r["dtg"] for r in traj_records], dtype=object),
        "starts":     np.stack([r["start"] for r in traj_records]),
        "goals":      np.stack([r["goal"] for r in traj_records]),
        "success":    np.array([r["success"] for r in traj_records], dtype=np.int32),
        "spl":        np.array([r["spl"] for r in traj_records], dtype=np.float32),
        "geodesic":   np.array([r["geodesic"] for r in traj_records], dtype=np.float32),
        "path_length":np.array([r["path_length"] for r in traj_records], dtype=np.float32),
        "steps":      np.array([r["steps"] for r in traj_records], dtype=np.int32),
    }
    np.savez_compressed(args.out_traj_npz, **save_dict)
    print(f"Wrote {n} trajectories to {args.out_traj_npz}")


if __name__ == "__main__":
    main()
