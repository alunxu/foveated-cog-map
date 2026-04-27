r"""
Classical Bug-style navigation baseline (Wijmans 2023 Table 1 / Fig 1B).

Provides a non-learning point of comparison for the 5 PointNav agents:
how well does a hand-designed reactive controller with goal-distance and
heading sensors do on the same evaluation scenes? Reviewers will ask
whether 95--99 % success means the task is too easy. The Bug baseline
answers: "with the same sensor stack but no learning, this is what a
greedy controller achieves."

Algorithm (greedy + always-right wall-recovery, single pass):
  1. While not at goal and step budget remaining:
     a. Read pointgoal-with-gps-compass: (rho, theta) in agent frame.
     b. If rho < success threshold, emit STOP and finish.
     c. If |theta| > 0.087 rad ($\approx 5^\circ$), TURN toward goal.
     d. Else MOVE_FORWARD.
     e. If the previous MOVE_FORWARD collided, switch to wall-recovery
        mode: TURN_RIGHT once, MOVE_FORWARD once. Loop until forward
        succeeds (max ``recovery_budget`` turns), then resume goal-
        seeking.

Notes:
  - Pure reactive controller; no map / no memory.
  - "Clairvoyant" variant of the Bug algorithm in Wijmans uses an oracle
    to pick the wall-following direction; the always-right variant here
    is the simpler reactive approximation.

Reads: standard habitat config + dataset (no checkpoint).
Writes: <out>/bug_baseline.json with per-episode metrics + aggregate.

Usage:
    python scripts/eval/bug_baseline.py \
        --config pointnav/ddppo_pointnav_blind_gibson \
        --episodes 200 \
        --out /scratch/izar/wxu/bug_baseline_results/bug.json
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
from src.utils.habitat_env import compute_spl, load_habitat_config


# Habitat default action ids: 0=stop, 1=forward, 2=left, 3=right
ACT_STOP = 0
ACT_FORWARD = 1
ACT_LEFT = 2
ACT_RIGHT = 3

# Tuning constants
SUCCESS_DIST = 0.2
HEADING_TOLERANCE_RAD = 0.087  # ~5°
RECOVERY_BUDGET = 20


def _pointgoal_rho_theta(env, obs):
    """Compute (rho, theta) of goal in agent frame.

    Try standard PointGoalWithGPSCompass sensor first; fall back to
    computing directly from env.sim agent state + episode goal — this
    works for the Wijmans-faithful sensor stack (goal_in_start_frame +
    GPS + compass) which doesn't expose pointgoal_with_gps_compass.
    """
    g = obs.get("pointgoal_with_gps_compass")
    if g is None:
        g = obs.get("pointgoal")
    if g is not None:
        g = np.asarray(g)
        if g.shape == (2,):
            return float(g[0]), float(g[1])
        elif g.shape == (3,):
            dx, _, dz = g
            rho = float(np.sqrt(dx * dx + dz * dz))
            theta = float(np.arctan2(-dx, -dz))
            return rho, theta

    # Fallback: compute from env state. Agent frame: +z = forward, +x = right.
    agent_state = env.sim.get_agent_state()
    pos = np.asarray(agent_state.position, dtype=np.float64)
    goal = np.asarray(env.current_episode.goals[0].position, dtype=np.float64)
    delta_world = goal - pos
    rho = float(np.linalg.norm(delta_world))

    # Rotate world delta into agent frame using agent rotation quaternion.
    # Habitat convention: agent looks along -z when heading=0; heading
    # increases counter-clockwise around y axis. Use a robust quaternion
    # rotation via quaternion arithmetic.
    import quaternion as quat
    rot = agent_state.rotation
    if not isinstance(rot, quat.quaternion):
        rot = quat.quaternion(rot.real, *rot.imag)
    inv = rot.inverse()
    # Rotate delta_world into agent frame.
    delta_q = quat.quaternion(0, *delta_world)
    rotated = inv * delta_q * rot
    dx_agent = rotated.x
    dz_agent = rotated.z  # forward axis
    # theta: angle from agent forward (-z) to goal direction. Positive = left.
    theta = float(np.arctan2(-dx_agent, -dz_agent))
    return rho, theta


def bug_step(env, obs, in_recovery: bool, recovery_steps: int) -> tuple[int, bool, int]:
    """Decide next action.

    Returns (action, new_in_recovery, new_recovery_steps).
    """
    rho, theta = _pointgoal_rho_theta(env, obs)

    # 1. Close enough → STOP.
    if rho < SUCCESS_DIST:
        return ACT_STOP, False, 0

    # 2. In wall-recovery mode: keep turning right + trying forward, with
    #    bounded budget. Exit when collision count drops or budget done.
    if in_recovery:
        if recovery_steps >= RECOVERY_BUDGET:
            return ACT_FORWARD, False, 0
        # Alternate right-turn and forward attempt
        if recovery_steps % 2 == 0:
            return ACT_RIGHT, True, recovery_steps + 1
        else:
            return ACT_FORWARD, True, recovery_steps + 1

    # 3. Not in recovery: align toward goal.
    if abs(theta) > HEADING_TOLERANCE_RAD:
        # In habitat: theta > 0 means goal is to the LEFT; turn left.
        # theta < 0 means goal to the right; turn right.
        return (ACT_LEFT, False, 0) if theta > 0 else (ACT_RIGHT, False, 0)

    # 4. Aligned → move forward.
    return ACT_FORWARD, False, 0


def run_episode(env, obs, max_steps: int = 2000) -> dict:
    """Run a Bug episode given starting obs (env should already be reset/pinned)."""
    episode = env.current_episode
    start_pos = np.array(episode.start_position)
    goal_pos = np.array(episode.goals[0].position)
    geodesic = float(env.sim.geodesic_distance(start_pos, goal_pos))

    path_length = 0.0
    prev_pos = env.sim.get_agent_state().position.copy()

    in_recovery = False
    recovery_steps = 0
    consecutive_collisions = 0
    done = False
    last_action = -1
    steps = 0

    while not done and steps < max_steps:
        action, in_recovery, recovery_steps = bug_step(
            env, obs, in_recovery, recovery_steps,
        )
        last_action = action
        if action == ACT_STOP:
            obs = env.step(action)
            done = True
            break
        obs = env.step(action)
        # Track collisions: switch to recovery on forward collide.
        if action == ACT_FORWARD and env.sim.previous_step_collided:
            consecutive_collisions += 1
            in_recovery = True
            recovery_steps = 0
        else:
            consecutive_collisions = 0
            if env.sim.previous_step_collided:
                # Turning collisions should be rare; ignore.
                pass
        cur_pos = env.sim.get_agent_state().position
        path_length += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos.copy()
        steps += 1
        done = env.episode_over

    final_pos = env.sim.get_agent_state().position
    dist_to_goal = float(np.linalg.norm(final_pos - goal_pos))
    success = (last_action == ACT_STOP) and (dist_to_goal < SUCCESS_DIST)
    spl = compute_spl(success, path_length, geodesic)

    return {
        "episode_id": str(episode.episode_id),
        "scene_id": str(episode.scene_id),
        "success": bool(success),
        "spl": float(spl),
        "path_length": float(path_length),
        "geodesic": float(geodesic),
        "steps": int(steps),
        "dist_to_goal": float(dist_to_goal),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Hydra config name; the agent / encoder doesn't "
                         "matter, we only need PointNav sensors and the env. "
                         "Pass blind for fastest sim.")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Bug-baseline classical controller")
    print(f"  Config:    {args.config}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Out:       {args.out}")
    print("=" * 64)

    config = load_habitat_config(args.config, "", overrides=[
        "habitat.dataset.split=train",
        "habitat.environment.iterator_options.shuffle=True",
        "habitat.environment.iterator_options.group_by_scene=False",
    ])
    env = habitat.Env(config=config.habitat)

    rng = np.random.default_rng(args.seed)
    all_eps = list(env.episodes)
    n = min(args.episodes, len(all_eps))
    sampled = list(rng.choice(all_eps, size=n, replace=False))
    print(f"\nSampled {n} episodes")

    def _pin(ep):
        env._episode_iterator = iter([ep])
        env._episode_over = False
        return env.reset()

    results = []
    for ei, ep in enumerate(sampled):
        obs = _pin(ep)
        m = run_episode(env, obs, max_steps=args.max_steps)
        results.append(m)
        if (ei + 1) % 25 == 0:
            spl = float(np.mean([r["spl"] for r in results]))
            succ = float(np.mean([r["success"] for r in results]))
            print(f"  [{ei+1}/{n}]  SPL={spl:.3f}  succ={succ:.3f}")

    env.close()

    succ = float(np.mean([r["success"] for r in results]))
    spl = float(np.mean([r["spl"] for r in results]))
    steps_avg = float(np.mean([r["steps"] for r in results]))
    print("\n=== Final ===")
    print(f"  Success rate:  {succ:.3f}")
    print(f"  Mean SPL:      {spl:.3f}")
    print(f"  Mean steps:    {steps_avg:.1f}")

    out = {
        "n_episodes": n,
        "success_rate": succ,
        "mean_spl": spl,
        "mean_steps": steps_avg,
        "mean_path_length": float(np.mean([r["path_length"] for r in results])),
        "mean_dist_to_goal": float(np.mean([r["dist_to_goal"] for r in results])),
        "per_episode": results,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
