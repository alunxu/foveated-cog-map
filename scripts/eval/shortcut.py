"""
Shortcut discovery / cognitive-map behavioral evaluation.

Inspired by SPACE's shortcut discovery task (Ramakrishnan, Wijmans et al.,
ICLR 2025) — the classic Tolman test for cognitive maps.

Instead of testing route memorization, this evaluates whether accumulated
spatial experience within a scene improves navigation efficiency — the
behavioral signature of a cognitive map.

Protocol:
  1. Group evaluation episodes by scene.
  2. Run episodes within each scene SEQUENTIALLY, in two conditions:
     (a) PERSISTENT memory: LSTM hidden state carries over between episodes
         (not reset). The agent accumulates spatial knowledge.
     (b) RESET memory: Hidden state zeroed between episodes (standard eval).
  3. Compare SPL across conditions. If persistent > reset, the agent's
     accumulated spatial representations functionally help navigation.
  4. Compare the persistent-memory benefit across agent types (blind,
     uniform, foveated, matched) to test which agents build the most
     useful cognitive maps.

This is a FUNCTIONAL test — unlike probing, which shows what information
is present, this shows whether that information is actually *used*.

Usage:
    python scripts/eval/shortcut.py \
        --config-name pointnav/ddppo_pointnav_blind_gibson \
        --ckpt /scratch/izar/$USER/habitat_checkpoints/blind_gibson/ckpt.16.pth \
        --episodes-per-scene 10 \
        --out /scratch/izar/$USER/shortcut_results/blind_gibson.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

import src.habitat  # noqa: F401
from src.utils.habitat_env import (
    heading_from_quaternion, compute_spl, load_habitat_config, load_policy,
)

import habitat


def parse_args():
    p = argparse.ArgumentParser(description="Shortcut discovery / cognitive-map behavioral eval")
    p.add_argument("--config-name", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes-per-scene", type=int, default=10,
                   help="Number of episodes to run per scene (sequential)")
    p.add_argument("--max-scenes", type=int, default=20,
                   help="Max number of scenes to evaluate (0=all)")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_episode(env, policy, rnn_hidden, prev_action, not_done_mask, device, rnn_is_lstm, num_recurrent_layers):
    """Run one episode, return (success, spl, path_length, steps, new_rnn_hidden)."""
    obs = env.reset()
    episode = env.current_episode

    # Geodesic distance for SPL
    start_pos = np.array(episode.start_position)
    goal_pos = np.array(episode.goals[0].position)
    geodesic = env.sim.geodesic_distance(start_pos, goal_pos)

    path_length = 0.0
    prev_pos = start_pos.copy()
    done = False
    steps = 0
    max_steps = 500

    while not done and steps < max_steps:
        batch = {k: torch.from_numpy(np.expand_dims(v, 0)).to(device)
                 for k, v in obs.items()}

        with torch.no_grad():
            action_data = policy.act(
                batch, rnn_hidden, prev_action, not_done_mask,
                deterministic=True,  # deterministic for eval
            )

        rnn_hidden = action_data.rnn_hidden_states
        prev_action = action_data.actions
        not_done_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

        action_int = action_data.env_actions[0].item()
        obs = env.step(action_int)

        # Track path length
        agent_state = env.sim.get_agent_state()
        cur_pos = agent_state.position
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos = cur_pos.copy()

        done = env.episode_over
        steps += 1

    # Check success: agent called STOP within 0.2m of goal
    final_pos = env.sim.get_agent_state().position
    dist_to_goal = np.linalg.norm(final_pos - goal_pos)
    success = (action_int == 0) and (dist_to_goal < 0.2)  # STOP action = 0

    spl = compute_spl(success, path_length, geodesic)

    return {
        "success": success,
        "spl": spl,
        "path_length": path_length,
        "geodesic": geodesic,
        "steps": steps,
        "dist_to_goal": dist_to_goal,
    }, rnn_hidden, prev_action, not_done_mask


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Config + Environment + Policy ----
    config = load_habitat_config(args.config_name, args.ckpt, overrides=[
        "habitat.dataset.split=train",
        "habitat.environment.iterator_options.shuffle=False",
        "habitat.environment.iterator_options.group_by_scene=True",
    ])

    env = habitat.Env(config=config.habitat)
    _ = env.reset()

    policy, hidden_size, num_recurrent_layers, rnn_is_lstm = load_policy(
        config, env, args.ckpt, device,
    )

    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Episodes/scene: {args.episodes_per_scene}")
    print(f"LSTM: {rnn_is_lstm}, layers: {num_recurrent_layers}, hidden: {hidden_size}")

    # ---- Group episodes by scene ----
    # Collect all available episodes and group by scene_id
    all_episodes = env.episodes
    scene_episodes = defaultdict(list)
    for ep in all_episodes:
        scene_episodes[ep.scene_id].append(ep)

    scenes = list(scene_episodes.keys())
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]

    print(f"Scenes: {len(scenes)}, episodes/scene: {args.episodes_per_scene}")

    # ---- Run evaluation in both conditions ----
    all_results = []

    for si, scene_id in enumerate(scenes):
        eps = scene_episodes[scene_id][:args.episodes_per_scene]
        if len(eps) < 2:
            continue

        scene_name = os.path.basename(scene_id).split(".")[0]
        print(f"\n  Scene {si+1}/{len(scenes)}: {scene_name} ({len(eps)} episodes)")

        for condition in ["reset", "persistent"]:
            # Initialize fresh hidden state for each condition
            rnn_hidden = torch.zeros(1, num_recurrent_layers, hidden_size, device=device)
            prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
            not_done_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

            ep_results = []
            for ei, ep in enumerate(eps):
                # Override current episode
                env._current_episode = ep
                env._episode_over = False

                if condition == "reset":
                    # Standard eval: reset hidden state between episodes
                    rnn_hidden = torch.zeros(1, num_recurrent_layers, hidden_size, device=device)
                    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
                    not_done_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
                # else: persistent — keep rnn_hidden from previous episode

                metrics, rnn_hidden, prev_action, not_done_mask = run_episode(
                    env, policy, rnn_hidden, prev_action, not_done_mask,
                    device, rnn_is_lstm, num_recurrent_layers,
                )
                ep_results.append(metrics)

            # Aggregate
            successes = [r["success"] for r in ep_results]
            spls = [r["spl"] for r in ep_results]

            all_results.append({
                "scene": scene_name,
                "condition": condition,
                "n_episodes": len(eps),
                "success_rate": float(np.mean(successes)),
                "mean_spl": float(np.mean(spls)),
                "per_episode_spl": spls,
                "per_episode_success": successes,
            })

            tag = "P" if condition == "persistent" else "R"
            print(f"    [{tag}] Success={np.mean(successes):.2f}  "
                  f"SPL={np.mean(spls):.3f}")

    env.close()

    # ---- Compute summary ----
    reset_spls = [r["mean_spl"] for r in all_results if r["condition"] == "reset"]
    persist_spls = [r["mean_spl"] for r in all_results if r["condition"] == "persistent"]
    reset_success = [r["success_rate"] for r in all_results if r["condition"] == "reset"]
    persist_success = [r["success_rate"] for r in all_results if r["condition"] == "persistent"]

    # Cognitive map benefit = persistent - reset
    spl_benefit = float(np.mean(persist_spls) - np.mean(reset_spls)) if persist_spls else 0
    success_benefit = float(np.mean(persist_success) - np.mean(reset_success)) if persist_success else 0

    # Early vs. late analysis (within persistent condition)
    # Compare first-half vs. second-half episode SPLs within each scene
    early_spls, late_spls = [], []
    for r in all_results:
        if r["condition"] == "persistent" and r["n_episodes"] >= 4:
            mid = r["n_episodes"] // 2
            early_spls.extend(r["per_episode_spl"][:mid])
            late_spls.extend(r["per_episode_spl"][mid:])

    summary = {
        "config": args.config_name,
        "checkpoint": args.ckpt,
        "n_scenes": len(scenes),
        "episodes_per_scene": args.episodes_per_scene,
        "reset_mean_spl": float(np.mean(reset_spls)) if reset_spls else None,
        "persistent_mean_spl": float(np.mean(persist_spls)) if persist_spls else None,
        "reset_mean_success": float(np.mean(reset_success)) if reset_success else None,
        "persistent_mean_success": float(np.mean(persist_success)) if persist_success else None,
        "cognitive_map_spl_benefit": spl_benefit,
        "cognitive_map_success_benefit": success_benefit,
        "persistent_early_spl": float(np.mean(early_spls)) if early_spls else None,
        "persistent_late_spl": float(np.mean(late_spls)) if late_spls else None,
        "map_building_spl_gain": float(np.mean(late_spls) - np.mean(early_spls)) if early_spls else None,
        "per_scene": all_results,
    }

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Reset (standard):    SPL={summary['reset_mean_spl']:.3f}  "
          f"Success={summary['reset_mean_success']:.2f}")
    print(f"  Persistent (cogmap): SPL={summary['persistent_mean_spl']:.3f}  "
          f"Success={summary['persistent_mean_success']:.2f}")
    print(f"  Cognitive-map SPL benefit:     {spl_benefit:+.3f}")
    print(f"  Cognitive-map success benefit: {success_benefit:+.3f}")
    if summary["map_building_spl_gain"] is not None:
        print(f"  Map-building gain (late−early): {summary['map_building_spl_gain']:+.3f}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
