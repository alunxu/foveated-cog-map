"""Generate offline trajectories from Memory-Maze 9x9 with a simple explorer policy.

Why locally generated rather than the published offline dataset?
  Pasukonis 2023's published Drive folder is rate-limited (24h cooldown). We
  reproduce the same NPZ schema (image / action / agent_pos / agent_dir /
  maze_layout / target_pos / targets_pos / target_color / target_vec /
  targets_vec) on Mac via `MUJOCO_GL=glfw`. ~183 steps/s on the M-series CPU,
  so 1k trajectories × 1001 steps ≈ 1.5h.

Policy: prefers forward, turns randomly when stuck (position hasn't changed in
N steps). Not the published scripted explorer but covers the maze adequately
for our probing question (linear position decode from LSTM hidden state).

Output: one .npz per trajectory in $OUT_DIR. Compatible with the
`MemoryMazeDataset` class in `02_cache_features.py`.
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import numpy as np
import gym
import memory_maze  # noqa: F401  (registers gym envs)

# ---- numpy alias patches needed by gym 0.26 + numpy 2.0 ----
np.bool8 = np.bool_  # type: ignore[attr-defined]


def make_env(size: str = "9x9", seed: int = 0):
    """Create the ExtraObs Memory-Maze env at the requested grid size."""
    env_id = f"memory_maze:MemoryMaze-{size}-ExtraObs-v0"
    env = gym.make(env_id)
    env.reset()  # GymWrapper.reset() doesn't accept seed kwarg
    return env


def explorer_action(rng: np.random.Generator, last_pos: np.ndarray, cur_pos: np.ndarray,
                     stuck_steps: int, action_space_n: int = 6) -> tuple[int, int]:
    """Forward-biased policy that turns when stuck.

    Action map for memory-maze (6-discrete one-hot):
      0 = noop, 1 = forward, 2 = forward-left, 3 = forward-right,
      4 = turn-left, 5 = turn-right (typical dm_lab order).

    We bias toward forward (action 1) but if position hasn't changed for 4+
    steps, sample a turn (4 or 5) with probability 0.7. This avoids getting
    stuck against walls forever.
    """
    if stuck_steps >= 4:
        if rng.random() < 0.7:
            return int(rng.choice([4, 5])), stuck_steps
    # 60% forward, 15% diagonal, 25% turn
    p = rng.random()
    if p < 0.6:
        return 1, stuck_steps
    if p < 0.75:
        return int(rng.choice([2, 3])), stuck_steps
    return int(rng.choice([4, 5])), stuck_steps


def rollout(env, T: int, seed: int) -> dict:
    """Run T steps in the env with the explorer policy. Return dict matching
    Pasukonis 2023 NPZ schema."""
    rng = np.random.default_rng(seed)
    obs = env.reset()
    if isinstance(obs, tuple):  # gym 0.26+ returns (obs, info)
        obs = obs[0]

    images = np.empty((T, 64, 64, 3), dtype=np.uint8)
    actions_oh = np.zeros((T, 6), dtype=np.uint8)
    agent_pos = np.empty((T, 2), dtype=np.float32)
    agent_dir = np.empty((T, 2), dtype=np.float32)
    target_color = np.empty((T, 3), dtype=np.float32)
    target_pos = np.empty((T, 2), dtype=np.float32)
    target_vec = np.empty((T, 2), dtype=np.float32)
    targets_pos = np.empty((T, 3, 2), dtype=np.float32)
    targets_vec = np.empty((T, 3, 2), dtype=np.float32)
    rewards = np.zeros((T,), dtype=np.float32)
    maze_layout = obs["maze_layout"].copy()

    last_pos = obs["agent_pos"].astype(np.float32)
    stuck_steps = 0
    for t in range(T):
        # Record obs at step t
        images[t] = obs["image"]
        agent_pos[t] = obs["agent_pos"]
        agent_dir[t] = obs["agent_dir"]
        target_color[t] = obs["target_color"]
        target_pos[t] = obs["target_pos"]
        target_vec[t] = obs["target_vec"]
        # Pad targets_pos/targets_vec to 3 if needed
        tp = obs["targets_pos"]
        tv = obs["targets_vec"]
        if tp.shape[0] < 3:
            pad = ((0, 3 - tp.shape[0]), (0, 0))
            tp = np.pad(tp, pad)
            tv = np.pad(tv, pad)
        targets_pos[t] = tp[:3]
        targets_vec[t] = tv[:3]

        # Step
        cur_pos = obs["agent_pos"]
        if np.linalg.norm(cur_pos - last_pos) < 0.05:
            stuck_steps += 1
        else:
            stuck_steps = 0
            last_pos = cur_pos

        a, stuck_steps = explorer_action(rng, last_pos, cur_pos, stuck_steps)
        actions_oh[t, a] = 1
        out = env.step(a)
        if len(out) == 4:
            obs, r, done, info = out
        else:  # gym 0.26+ returns (obs, r, terminated, truncated, info)
            obs, r, terminated, truncated, info = out
            done = terminated or truncated
        rewards[t] = r
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            stuck_steps = 0
            last_pos = obs["agent_pos"].astype(np.float32)

    return dict(
        image=images,
        action=actions_oh,
        reward=rewards,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        maze_layout=maze_layout,
        target_color=target_color,
        target_pos=target_pos,
        target_vec=target_vec,
        targets_pos=targets_pos,
        targets_vec=targets_vec,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_traj", type=int, default=200)
    ap.add_argument("--T", type=int, default=1001)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--size", type=str, default="9x9")
    ap.add_argument("--start_idx", type=int, default=0,
                     help="Starting trajectory index (for resuming).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Generating {args.num_traj} trajectories of T={args.T} steps to {args.out_dir}")
    env = make_env(size=args.size, seed=args.seed_offset)
    import time
    t0 = time.time()
    for i in range(args.start_idx, args.start_idx + args.num_traj):
        out_path = os.path.join(args.out_dir, f"traj_{i:05d}.npz")
        if os.path.exists(out_path):
            continue
        traj = rollout(env, T=args.T, seed=args.seed_offset + i)
        np.savez_compressed(out_path, **traj)
        if (i - args.start_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            done_n = i - args.start_idx + 1
            rate = done_n / elapsed
            eta = (args.num_traj - done_n) / rate
            xs = traj["agent_pos"][:, 0]
            ys = traj["agent_pos"][:, 1]
            cov = (xs.max() - xs.min()) * (ys.max() - ys.min())
            print(
                f"  [{done_n}/{args.num_traj}] {rate:.2f} traj/s "
                f"(eta {eta/60:.1f}m, cov bbox area={cov:.2f})",
                flush=True,
            )
    env.close()
    print(f"Done. Wrote to {args.out_dir} in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
