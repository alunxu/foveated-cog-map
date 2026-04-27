r"""
Probe-agent experiment (Wijmans 2023 Fig 3 replication, ported to our
5-condition setup).

For each condition X (blind / coarse / uniform / foveated / foveated-learned):

  1. Run trained agent on episode S→T → record agent SPL +
     final hidden state (h_T, c_T).
  2. Reset env to the SAME episode (same start, same goal).
  3. Initialise the probe (structurally identical to agent: same config +
     same checkpoint) with (h_T, c_T) as its starting LSTM state instead
     of zeros, and run S→T → record probe SPL.

Comparisons reported:
  - agent_spl: standard zero-init run (= AllZeroMemory in Wijmans Fig 3B).
  - probe_trained_spl: probe initialised with trained agent's final memory.

Story for §4.5 dissociation:
  - Bottleneck conditions (blind / coarse) are expected to show
    probe_trained > agent because their memory linearly carries GPS
    (already established by §4.2 H1). This is a sanity check.
  - Rich-encoder conditions (uniform / foveated) are the interesting
    cases: if probe_trained > agent, the memory IS useful for navigation
    DESPITE not being linearly readable — direct evidence that "probe-
    readable" and "policy-used" can dissociate (the §4.5 candidate
    dissociation, currently \uncertain-flagged, would gain a tighter
    behavioural validation).

Reads:
    --config <hydra-config-name>      e.g. pointnav/ddppo_pointnav_blind_gibson
    --ckpt <ckpt-path>                e.g. /scratch/.../blind_gibson/ckpt.34.pth

Writes:
    <out-path>.json with:
      {
        "condition": "<cond>",
        "n_episodes": int,
        "per_episode": [
          {"episode_id": ..., "agent": {...}, "probe_trained": {...}},
          ...
        ],
        "agent":          {"mean_spl": ..., "success_rate": ..., "n": ...},
        "probe_trained":  {"mean_spl": ..., "success_rate": ..., "n": ...},
        "delta":          {"mean_spl": probe - agent, ...}
      }

Usage:
    python scripts/eval/probe_agent.py \
        --config pointnav/ddppo_pointnav_blind_gibson \
        --ckpt /scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.34.pth \
        --episodes 150 \
        --out /scratch/izar/wxu/probe_agent_results/blind.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

import src.habitat  # noqa: F401  registers configs / policies

import habitat
from src.utils.habitat_env import (
    compute_spl,
    load_habitat_config,
    load_policy,
)


# ---------------------------------------------------------------------------
# Helpers (mirror transplant.py to keep semantics consistent)
# ---------------------------------------------------------------------------

def _init_zero_state(num_recurrent_layers: int, hidden_size: int, device):
    rnn_hidden = torch.zeros(1, num_recurrent_layers, hidden_size, device=device)
    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
    not_done_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    return rnn_hidden, prev_action, not_done_mask


def _step_policy(env, obs, policy, rnn_hidden, prev_action, not_done_mask, device):
    batch = {
        k: torch.from_numpy(np.expand_dims(v, 0)).to(device)
        for k, v in obs.items()
    }
    with torch.no_grad():
        action_data = policy.act(
            batch, rnn_hidden, prev_action, not_done_mask,
            deterministic=True,
        )
    rnn_hidden = action_data.rnn_hidden_states
    prev_action = action_data.actions
    not_done_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    action_int = action_data.env_actions[0].item()
    next_obs = env.step(action_int)
    done = env.episode_over
    return next_obs, done, rnn_hidden, prev_action, not_done_mask, action_int


def _run_full_episode(env, obs, policy, rnn_hidden, prev_action, not_done_mask,
                      device, max_steps: int = 2000) -> tuple[dict, torch.Tensor]:
    """Run a full episode; return (metrics dict, final rnn_hidden).

    The final rnn_hidden is what we feed into the probe as its initial state.
    """
    episode = env.current_episode
    start_pos = np.array(episode.start_position)
    goal_pos = np.array(episode.goals[0].position)
    geodesic = float(env.sim.geodesic_distance(start_pos, goal_pos))

    path_length = 0.0
    prev_pos = env.sim.get_agent_state().position.copy()
    done = False
    steps = 0
    action_int = -1

    while not done and steps < max_steps:
        obs, done, rnn_hidden, prev_action, not_done_mask, action_int = _step_policy(
            env, obs, policy, rnn_hidden, prev_action, not_done_mask, device,
        )
        cur_pos = env.sim.get_agent_state().position
        path_length += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos.copy()
        steps += 1

    final_pos = env.sim.get_agent_state().position
    dist_to_goal = float(np.linalg.norm(final_pos - goal_pos))
    success = (action_int == 0) and (dist_to_goal < 0.2)
    spl = compute_spl(success, path_length, geodesic)
    return ({
        "success": bool(success),
        "spl": float(spl),
        "path_length": float(path_length),
        "geodesic": float(geodesic),
        "steps": int(steps),
        "dist_to_goal": float(dist_to_goal),
    }, rnn_hidden.detach().clone())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Hydra config name, e.g. pointnav/ddppo_pointnav_blind_gibson")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "mean_spl": None, "success_rate": None}
    return {
        "n": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_spl": float(np.mean([r["spl"] for r in rows])),
        "mean_path_length": float(np.mean([r["path_length"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_dist_to_goal": float(np.mean([r["dist_to_goal"] for r in rows])),
    }


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  Probe-agent experiment (Wijmans Fig 3 replication)")
    print(f"  Config:    {args.config}")
    print(f"  Ckpt:      {args.ckpt}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Out:       {args.out}")
    print("=" * 72)

    print("\n=== Building env ===")
    config = load_habitat_config(args.config, str(args.ckpt), overrides=[
        "habitat.dataset.split=train",
        "habitat.environment.iterator_options.shuffle=True",
        "habitat.environment.iterator_options.group_by_scene=False",
    ])
    env = habitat.Env(config=config.habitat)
    _ = env.reset()

    print("\n=== Loading policy ===")
    policy, hidden, layers, is_lstm = load_policy(
        config, env, str(args.ckpt), device,
    )
    print(f"  hidden={hidden}, layers={layers}, LSTM={is_lstm}")

    # Sample episodes deterministically (use seed for repro across runs).
    rng = np.random.default_rng(args.seed)
    all_eps = list(env.episodes)
    n = min(args.episodes, len(all_eps))
    sampled_eps = list(rng.choice(all_eps, size=n, replace=False))
    print(f"\nEvaluating {n} episodes")

    def _pin(ep):
        env._episode_iterator = iter([ep])
        env._episode_over = False
        obs = env.reset()
        assert env.current_episode.episode_id == ep.episode_id, (
            f"pinning failed: wanted {ep.episode_id}, got {env.current_episode.episode_id}"
        )
        return obs

    per_episode = []
    agent_rows: list[dict] = []
    probe_rows: list[dict] = []

    for ei, ep in enumerate(sampled_eps):
        # ----- 1. Agent run (zero-init) -----
        obs = _pin(ep)
        rnn_h, prev_a, mask = _init_zero_state(layers, hidden, device)
        agent_metrics, final_state = _run_full_episode(
            env, obs, policy, rnn_h, prev_a, mask, device, args.max_steps,
        )
        agent_rows.append(agent_metrics)

        # ----- 2. Probe run (init with agent's final memory) -----
        obs = _pin(ep)
        probe_rnn_h = final_state  # already (1, layers, hidden) on device
        # Treat the probe as if it has just finished an episode at S,
        # so feed the same not_done_mask convention as a fresh run does
        # AFTER the first step: mask=1 (so RNN consumes hidden state).
        probe_prev_a = torch.zeros(1, 1, dtype=torch.long, device=device)
        probe_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
        probe_metrics, _ = _run_full_episode(
            env, obs, policy, probe_rnn_h, probe_prev_a, probe_mask,
            device, args.max_steps,
        )
        probe_rows.append(probe_metrics)

        per_episode.append({
            "episode_id": str(ep.episode_id),
            "scene_id": str(ep.scene_id),
            "agent": agent_metrics,
            "probe_trained": probe_metrics,
        })

        if (ei + 1) % 25 == 0:
            ag = _agg(agent_rows)
            pr = _agg(probe_rows)
            print(f"  [{ei+1}/{n}]  agent SPL={ag['mean_spl']:.3f}  "
                  f"probe SPL={pr['mean_spl']:.3f}  "
                  f"Δ={pr['mean_spl']-ag['mean_spl']:+.3f}")

    env.close()

    agent_agg = _agg(agent_rows)
    probe_agg = _agg(probe_rows)
    delta = {
        "mean_spl": probe_agg["mean_spl"] - agent_agg["mean_spl"],
        "success_rate": probe_agg["success_rate"] - agent_agg["success_rate"],
        "mean_steps": agent_agg["mean_steps"] - probe_agg["mean_steps"],
    }

    print("\n=== Final ===")
    print(f"  Agent (AllZeroMemory)     SPL={agent_agg['mean_spl']:.3f}  "
          f"succ={agent_agg['success_rate']:.3f}  "
          f"steps={agent_agg['mean_steps']:.1f}")
    print(f"  Probe (TrainedAgentMem)   SPL={probe_agg['mean_spl']:.3f}  "
          f"succ={probe_agg['success_rate']:.3f}  "
          f"steps={probe_agg['mean_steps']:.1f}")
    print(f"  Δ probe − agent           SPL={delta['mean_spl']:+.3f}")

    out: dict[str, Any] = {
        "condition_config": args.config,
        "ckpt": str(args.ckpt),
        "n_episodes": n,
        "agent": agent_agg,           # AllZeroMemory baseline
        "probe_trained": probe_agg,   # TrainedAgentMemory
        "delta": delta,
        "per_episode": per_episode,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
