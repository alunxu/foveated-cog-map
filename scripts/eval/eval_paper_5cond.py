r"""
Clean SPL/Success evaluation for one condition.

Built on the same pattern as probe_agent.py (load_habitat_config +
habitat.Env() + load_policy()), bypassing habitat_baselines.run.main()
which has stubborn dataset-config resolution issues against our merged
Gibson+MP3D pool.

For each sampled episode:
  - reset env to that episode (pinned via _episode_iterator)
  - run trained policy from zero-init memory, deterministic
  - compute SPL / Success / steps / path_length / geodesic
  - aggregate at the end

Usage:
    python scripts/eval/eval_paper_5cond.py \
        --config pointnav/ddppo_pointnav_blind_gibson \
        --ckpt /scratch/wxu/habitat_checkpoints_rcp/blind_izar/ckpt.34.pth \
        --episodes 500 \
        --out /scratch/wxu/habitat_checkpoints_rcp/eval_5cond/blind.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
# Helpers (reuse probe_agent.py semantics)
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
                      device, max_steps: int = 2000) -> dict:
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
    return {
        "success": bool(success),
        "spl": float(spl),
        "path_length": float(path_length),
        "geodesic": float(geodesic),
        "steps": int(steps),
        "dist_to_goal": float(dist_to_goal),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Hydra config name, e.g. pointnav/ddppo_pointnav_blind_gibson")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--split", default="train",
                   help="Dataset split. Default 'train' uses the training pool with "
                        "rng-shuffled (start,goal) pairs; 'test' (when --data-path "
                        "points at a held-out dataset) uses the standard test set.")
    p.add_argument("--data-path", default=None,
                   help="Optional override for habitat.dataset.data_path. Use to "
                        "point at MP3D test (Wijmans 2023 protocol): "
                        "'data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz'.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-sample", action="store_true",
                   help="If set, evaluate ALL episodes in pool (no rng sub-sample). "
                        "Use this for MP3D-test runs to evaluate full 1008-episode set.")
    return p.parse_args()


def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "mean_spl": None, "success_rate": None}
    return {
        "n": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_spl": float(np.mean([r["spl"] for r in rows])),
        "std_spl": float(np.std([r["spl"] for r in rows])),
        "sem_spl": float(np.std([r["spl"] for r in rows]) / np.sqrt(len(rows))),
        "mean_path_length": float(np.mean([r["path_length"] for r in rows])),
        "mean_geodesic": float(np.mean([r["geodesic"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_dist_to_goal": float(np.mean([r["dist_to_goal"] for r in rows])),
    }


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  Paper-level eval (clean SPL/Success, single-env probe-style)")
    print(f"  Config:    {args.config}")
    print(f"  Ckpt:      {args.ckpt}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Split:     {args.split}")
    print(f"  Out:       {args.out}")
    print("=" * 72)

    print("\n=== Building env ===")
    overrides = [
        f"habitat.dataset.split={args.split}",
        "habitat.environment.iterator_options.shuffle=True",
        "habitat.environment.iterator_options.group_by_scene=False",
    ]
    if args.data_path is not None:
        overrides.insert(0, f"habitat.dataset.data_path={args.data_path}")
        print(f"  data_path:  {args.data_path}")
    config = load_habitat_config(args.config, str(args.ckpt), overrides=overrides)
    env = habitat.Env(config=config.habitat)
    _ = env.reset()

    print("\n=== Loading policy ===")
    policy, hidden, layers, is_lstm = load_policy(
        config, env, str(args.ckpt), device,
    )
    print(f"  hidden={hidden}, layers={layers}, LSTM={is_lstm}")

    all_eps = list(env.episodes)
    if args.no_sample:
        # Evaluate ALL episodes in pool (Wijmans-style: deterministic full eval).
        sampled_eps = list(all_eps)
        n = len(sampled_eps)
        print(f"\nEvaluating ALL {n} episodes (no rng sub-sample)")
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.episodes, len(all_eps))
        sampled_eps = list(rng.choice(all_eps, size=n, replace=False))
        print(f"\nEvaluating {n} episodes (pool size {len(all_eps)}, sampled with seed={args.seed})")

    def _pin(ep):
        env._episode_iterator = iter([ep])
        env._episode_over = False
        obs = env.reset()
        assert env.current_episode.episode_id == ep.episode_id, (
            f"pinning failed: wanted {ep.episode_id}, got {env.current_episode.episode_id}"
        )
        return obs

    per_episode = []
    rows: list[dict] = []
    t0 = time.time()

    for ei, ep in enumerate(sampled_eps):
        obs = _pin(ep)
        rnn_h, prev_a, mask = _init_zero_state(layers, hidden, device)
        metrics = _run_full_episode(
            env, obs, policy, rnn_h, prev_a, mask, device, args.max_steps,
        )
        rows.append(metrics)
        per_episode.append({
            "episode_id": str(ep.episode_id),
            "scene_id": str(ep.scene_id),
            **metrics,
        })

        if (ei + 1) % 25 == 0:
            ag = _agg(rows)
            elapsed = time.time() - t0
            eta = elapsed / (ei + 1) * (n - ei - 1)
            print(f"  [{ei+1}/{n}]  SPL={ag['mean_spl']:.3f}  "
                  f"succ={ag['success_rate']:.3f}  "
                  f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min")

    env.close()
    agg = _agg(rows)

    print("\n=== Final ===")
    print(f"  SPL          {agg['mean_spl']:.4f}  (std {agg['std_spl']:.4f}, sem {agg['sem_spl']:.4f})")
    print(f"  Success rate {agg['success_rate']:.4f}")
    print(f"  Mean steps   {agg['mean_steps']:.1f}")

    out: dict[str, Any] = {
        "condition_config": args.config,
        "ckpt": str(args.ckpt),
        "split": args.split,
        "n_episodes": n,
        "summary": agg,
        "per_episode": per_episode,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
