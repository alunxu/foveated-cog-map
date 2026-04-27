r"""
WJ-F Excursion Forgetting (Wijmans 2023 \S 3.4 / Fig 4 replication).

Tests whether the LSTM hidden state's GPS code survives a forced detour
in which the policy's actions are temporarily replaced by random
(non-stop) actions, pushing the agent off-policy.

Protocol per episode:
  1. Run trained agent for ``--warmup-steps`` (default 50) steps.
     Hidden state evolves under normal policy rollout.
  2. Inject ``--detour-steps`` (default 25) RANDOM actions to push the
     agent off-course. Hidden state continues to update; the GPS sensor
     is still the agent's true position throughout.
  3. Resume policy rollout for up to ``--recovery-steps`` (default 100)
     steps or until episode ends.

Save per-step hidden state h_2, ground-truth position, and a segment id
(0=warmup, 1=detour, 2=recovery). Downstream: probe each segment
separately and compare GPS R^2 across segments per condition.

Decision rule (per the no-overoptimism convention):
  - Result paper-integratable iff per-segment R^2 shows a clear
    bottleneck-vs-rich-encoder split that survives 100+ episode CV.
  - Otherwise it's an open question; document but don't paper-write.

Reads:
    --config         hydra config (any condition)
    --ckpt           checkpoint path
    --warmup-steps   default 50
    --detour-steps   default 25
    --recovery-steps default 100
    --episodes       default 100 (target valid; sampled 3x to allow rejection)
    --out            NPZ path

Writes:
    <out>.npz with the same schema as scripts/probing/collect.py output
    plus a `segments` (T,) int8 array (0=warmup, 1=detour, 2=recovery)
    and `wjf_meta` containing the warmup/detour/recovery step counts.

Usage:
    python scripts/eval/excursion_forgetting.py \
        --config pointnav/ddppo_pointnav_blind_gibson \
        --ckpt /scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.34.pth \
        --episodes 100 \
        --out /scratch/izar/wxu/excursion_results/blind_det.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

import src.habitat  # noqa: F401  registers configs / policies

import habitat
from src.utils.habitat_env import heading_from_quaternion, load_habitat_config, load_policy


# Habitat default action ids
ACT_STOP = 0
ACT_FORWARD = 1
ACT_LEFT = 2
ACT_RIGHT = 3


def _init_zero_state(num_layers, hidden_size, device):
    rnn = torch.zeros(1, num_layers, hidden_size, device=device)
    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
    not_done = torch.zeros(1, 1, dtype=torch.bool, device=device)
    return rnn, prev_action, not_done


def _step(env, obs, policy, rnn, prev_action, not_done, device,
          forced_action: int | None = None):
    """One env step. If forced_action is not None, override the policy's
    action choice (but still do the policy forward pass to keep rnn
    state evolving with the same observation)."""
    batch = {
        k: torch.from_numpy(np.expand_dims(v, 0)).to(device)
        for k, v in obs.items()
    }
    with torch.no_grad():
        action_data = policy.act(
            batch, rnn, prev_action, not_done, deterministic=True,
        )
    rnn = action_data.rnn_hidden_states  # post-act h
    if forced_action is not None:
        action_int = int(forced_action)
        prev_action = torch.full((1, 1), action_int, dtype=torch.long, device=device)
    else:
        prev_action = action_data.actions
        action_int = action_data.env_actions[0].item()
    not_done = torch.ones(1, 1, dtype=torch.bool, device=device)
    next_obs = env.step(action_int)
    done = env.episode_over
    return next_obs, done, rnn, prev_action, not_done, action_int


def _collect_step_state(env, rnn):
    """Snapshot per-step state for probing: top-layer hidden, all hidden
    layers, position, heading, distance-to-goal, GPS, compass."""
    h_top = rnn[0, -1].detach().cpu().numpy()  # (hidden_size,)
    h_all = rnn[0].detach().cpu().numpy()  # (num_layers, hidden_size)
    agent_state = env.sim.get_agent_state()
    pos = np.array(agent_state.position, dtype=np.float32)
    heading = float(heading_from_quaternion(agent_state.rotation))
    goal_pos = np.array(env.current_episode.goals[0].position, dtype=np.float32)
    dtg = float(np.linalg.norm(pos - goal_pos))
    return h_top, h_all, pos, heading, dtg, goal_pos


def run_episode(env, obs, policy, num_layers, hidden_size, device,
                warmup_steps, detour_steps, recovery_steps,
                rng) -> dict | None:
    """Run a single episode through warmup -> detour -> recovery.

    Returns dict with per-step arrays, or None if episode ended before
    detour completed (so the segment structure is incomplete).
    """
    rnn, prev_action, not_done = _init_zero_state(num_layers, hidden_size, device)
    h_top_buf, h_all_buf = [], []
    pos_buf, head_buf, dtg_buf, goal_buf = [], [], [], []
    seg_buf, step_buf = [], []

    step_idx = 0

    def _record(seg_id):
        h_top, h_all, pos, heading, dtg, goal = _collect_step_state(env, rnn)
        h_top_buf.append(h_top)
        h_all_buf.append(h_all)
        pos_buf.append(pos)
        head_buf.append(heading)
        dtg_buf.append(dtg)
        goal_buf.append(goal)
        seg_buf.append(seg_id)
        step_buf.append(step_idx)

    # --- warmup ---
    for _ in range(warmup_steps):
        next_obs, done, rnn, prev_action, not_done, act = _step(
            env, obs, policy, rnn, prev_action, not_done, device,
        )
        _record(0)
        step_idx += 1
        obs = next_obs
        if done or act == ACT_STOP:
            return None  # episode ended too early

    # --- detour: random forced actions (non-STOP) ---
    for _ in range(detour_steps):
        forced = int(rng.choice([ACT_FORWARD, ACT_LEFT, ACT_RIGHT]))
        next_obs, done, rnn, prev_action, not_done, act = _step(
            env, obs, policy, rnn, prev_action, not_done, device,
            forced_action=forced,
        )
        _record(1)
        step_idx += 1
        obs = next_obs
        if done:
            return None  # rare; navmesh closed loop on us

    # --- recovery: free policy ---
    for _ in range(recovery_steps):
        next_obs, done, rnn, prev_action, not_done, act = _step(
            env, obs, policy, rnn, prev_action, not_done, device,
        )
        _record(2)
        step_idx += 1
        obs = next_obs
        if done or act == ACT_STOP:
            break

    return {
        "h_top": np.stack(h_top_buf, axis=0),
        "h_all": np.stack(h_all_buf, axis=0),
        "positions": np.stack(pos_buf, axis=0),
        "headings": np.array(head_buf, dtype=np.float32),
        "dtg": np.array(dtg_buf, dtype=np.float32),
        "goal": np.stack(goal_buf, axis=0),
        "segments": np.array(seg_buf, dtype=np.int8),
        "step_in_episode": np.array(step_buf, dtype=np.int32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=100,
                    help="Target number of valid episodes")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--detour-steps", type=int, default=25)
    ap.add_argument("--recovery-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  WJ-F Excursion Forgetting")
    print(f"  Config: {args.config}")
    print(f"  Ckpt:   {args.ckpt}")
    print(f"  Eps:    target {args.episodes}")
    print(f"  W/D/R:  {args.warmup_steps}/{args.detour_steps}/{args.recovery_steps}")
    print(f"  Out:    {args.out}")
    print("=" * 64)

    config = load_habitat_config(args.config, "", overrides=[
        "habitat.dataset.split=train",
        "habitat.environment.iterator_options.shuffle=True",
        "habitat.environment.iterator_options.group_by_scene=False",
    ])
    env = habitat.Env(config=config.habitat)

    rng = np.random.default_rng(args.seed)
    all_eps = list(env.episodes)
    n_pool = min(args.episodes * 3, len(all_eps))
    sampled = list(rng.choice(all_eps, size=n_pool, replace=False))
    print(f"Sampled {n_pool} candidate episodes (target valid: {args.episodes})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, num_layers, hidden_size = load_policy(config, args.ckpt, device)
    policy.eval()

    def _pin(ep):
        env._episode_iterator = iter([ep])
        env._episode_over = False
        return env.reset()

    all_h_top, all_h_all = [], []
    all_pos, all_head, all_dtg, all_goal = [], [], [], []
    all_seg, all_step = [], []
    all_eid, all_sid = [], []
    n_valid = 0

    for ei, ep in enumerate(sampled):
        if n_valid >= args.episodes:
            break
        try:
            obs = _pin(ep)
        except StopIteration:
            continue
        result = run_episode(
            env, obs, policy, num_layers, hidden_size, device,
            args.warmup_steps, args.detour_steps, args.recovery_steps,
            rng,
        )
        if result is None:
            continue
        T = len(result["segments"])
        all_h_top.append(result["h_top"])
        all_h_all.append(result["h_all"])
        all_pos.append(result["positions"])
        all_head.append(result["headings"])
        all_dtg.append(result["dtg"])
        all_goal.append(result["goal"])
        all_seg.append(result["segments"])
        all_step.append(result["step_in_episode"])
        all_eid.append(np.full(T, n_valid, dtype=np.int32))
        all_sid.append(np.full(T, hash(ep.scene_id) % (2**31), dtype=np.int32))
        n_valid += 1
        if n_valid % 10 == 0:
            print(f"  [{n_valid}/{args.episodes}] valid")

    env.close()

    if n_valid == 0:
        print("ERROR: no valid episodes (every episode ended before detour completed)")
        sys.exit(1)

    h_top = np.concatenate(all_h_top, axis=0)
    h_all = np.concatenate(all_h_all, axis=0)
    positions = np.concatenate(all_pos, axis=0)
    headings = np.concatenate(all_head, axis=0)
    dtg = np.concatenate(all_dtg, axis=0)
    goal = np.concatenate(all_goal, axis=0)
    segments = np.concatenate(all_seg, axis=0)
    step_in_episode = np.concatenate(all_step, axis=0)
    episode_ids = np.concatenate(all_eid, axis=0)
    scene_ids = np.concatenate(all_sid, axis=0)

    print(f"\nFinal: {n_valid} valid episodes, total {h_top.shape[0]} steps")
    counts = {0: int((segments == 0).sum()), 1: int((segments == 1).sum()), 2: int((segments == 2).sum())}
    print(f"  segments: warmup={counts[0]}, detour={counts[1]}, recovery={counts[2]}")
    print(f"  h_top shape: {h_top.shape}, positions: {positions.shape}")

    save_dict = {
        "hidden_states": h_top,
        "h_layers": h_all,
        "positions": positions,
        "headings": headings,
        "distance_to_goal": dtg,
        "goal_positions": goal,
        "segments": segments,
        "step_in_episode": step_in_episode,
        "episode_ids": episode_ids,
        "scene_ids": scene_ids,
        "wjf_warmup_steps": np.int32(args.warmup_steps),
        "wjf_detour_steps": np.int32(args.detour_steps),
        "wjf_recovery_steps": np.int32(args.recovery_steps),
    }

    tmp_path = str(args.out) + ".tmp.npz"
    np.savez_compressed(tmp_path, **save_dict)
    os.replace(tmp_path, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
