"""
Memory transplant evaluation (Wijmans et al. inspired).

Tests whether LSTM hidden-state representations are compatible across
agent conditions (blind, uniform, foveated, matched-compute).  If
two agents learn a shared memory format, swapping the donor's hidden
state into the recipient at mid-episode should preserve navigation
performance.  If not, performance degrades — revealing that the
representations are structurally different.

Protocol:
  For each episode, evaluate three conditions:
    1. **Baseline**:        recipient runs the full episode alone.
    2. **Self-transplant**:  recipient runs the first half, its own
                             hidden state is "transplanted" back at the
                             midpoint (sanity check; should equal baseline).
    3. **Cross-transplant**: donor runs the first half, its hidden state
                             is injected into the recipient at the midpoint;
                             the recipient then finishes the episode.

  The donor and recipient may have different architectures (e.g., blind
  vs sighted).  Because hidden-state dimensionality must match for a raw
  transplant, the script checks compatibility and aborts early if shapes
  differ.

Usage:
    python scripts/eval/transplant.py \
        --donor-config    pointnav/ddppo_pointnav_foveated_gibson \
        --donor-ckpt      /scratch/izar/$USER/habitat_checkpoints/foveated/ckpt.16.pth \
        --recipient-config pointnav/ddppo_pointnav_uniform_gibson \
        --recipient-ckpt   /scratch/izar/$USER/habitat_checkpoints/uniform/ckpt.16.pth \
        --episodes 200 \
        --midpoint-frac 0.5 \
        --out /scratch/izar/$USER/transplant_results/foveated_to_uniform.json
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
    compute_spl,
    load_habitat_config,
    load_policy,
)

import habitat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_rnn_state(num_recurrent_layers, hidden_size, device):
    """Return zeroed (rnn_hidden, prev_action, not_done_mask)."""
    rnn_hidden = torch.zeros(1, num_recurrent_layers, hidden_size, device=device)
    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
    not_done_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    return rnn_hidden, prev_action, not_done_mask


def _step_policy(env, obs, policy, rnn_hidden, prev_action, not_done_mask, device):
    """Run one policy step. Returns (next_obs, done, rnn_hidden, prev_action, not_done_mask, action_int)."""
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
                      device, max_steps=500):
    """Run a complete episode from current state. Returns metrics dict."""
    episode = env.current_episode
    start_pos = np.array(episode.start_position)
    goal_pos = np.array(episode.goals[0].position)
    geodesic = env.sim.geodesic_distance(start_pos, goal_pos)

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
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos = cur_pos.copy()
        steps += 1

    final_pos = env.sim.get_agent_state().position
    dist_to_goal = np.linalg.norm(final_pos - goal_pos)
    success = (action_int == 0) and (dist_to_goal < 0.2)
    spl = compute_spl(success, path_length, geodesic)

    return {
        "success": bool(success),
        "spl": float(spl),
        "path_length": float(path_length),
        "geodesic": float(geodesic),
        "steps": steps,
        "dist_to_goal": float(dist_to_goal),
    }


def _run_first_half(env, obs, policy, rnn_hidden, prev_action, not_done_mask,
                    device, n_steps):
    """Run the first *n_steps* of an episode (or until done).

    Returns (obs, done, rnn_hidden, prev_action, not_done_mask, path_length,
             steps_taken, last_action_int).
    """
    prev_pos = env.sim.get_agent_state().position.copy()
    path_length = 0.0
    action_int = -1

    for step in range(n_steps):
        if env.episode_over:
            break
        obs, done, rnn_hidden, prev_action, not_done_mask, action_int = _step_policy(
            env, obs, policy, rnn_hidden, prev_action, not_done_mask, device,
        )
        cur_pos = env.sim.get_agent_state().position
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos = cur_pos.copy()

    return obs, env.episode_over, rnn_hidden, prev_action, not_done_mask, path_length, action_int


def _run_second_half(env, obs, policy, rnn_hidden, prev_action, not_done_mask,
                     device, path_length_so_far, max_steps=500):
    """Continue running from current state to episode end.

    Returns metrics dict for the FULL episode (including the first-half
    path length).
    """
    episode = env.current_episode
    start_pos = np.array(episode.start_position)
    goal_pos = np.array(episode.goals[0].position)
    geodesic = env.sim.geodesic_distance(start_pos, goal_pos)

    path_length = path_length_so_far
    prev_pos = env.sim.get_agent_state().position.copy()
    done = env.episode_over
    steps = 0
    action_int = -1

    while not done and steps < max_steps:
        obs, done, rnn_hidden, prev_action, not_done_mask, action_int = _step_policy(
            env, obs, policy, rnn_hidden, prev_action, not_done_mask, device,
        )
        cur_pos = env.sim.get_agent_state().position
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos = cur_pos.copy()
        steps += 1

    final_pos = env.sim.get_agent_state().position
    dist_to_goal = np.linalg.norm(final_pos - goal_pos)
    success = (action_int == 0) and (dist_to_goal < 0.2)
    spl = compute_spl(success, path_length, geodesic)

    return {
        "success": bool(success),
        "spl": float(spl),
        "path_length": float(path_length),
        "geodesic": float(geodesic),
        "steps": steps,  # second-half steps only
        "dist_to_goal": float(dist_to_goal),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Memory transplant evaluation (Wijmans et al. inspired)"
    )
    # Donor agent (provides hidden state for transplant)
    p.add_argument("--donor-config", required=True,
                   help="Hydra config name for donor agent")
    p.add_argument("--donor-ckpt", required=True,
                   help="Checkpoint path for donor agent")

    # Recipient agent (receives transplanted hidden state)
    p.add_argument("--recipient-config", required=True,
                   help="Hydra config name for recipient agent")
    p.add_argument("--recipient-ckpt", required=True,
                   help="Checkpoint path for recipient agent")

    # Episode control
    p.add_argument("--episodes", type=int, default=200,
                   help="Number of episodes to evaluate")
    p.add_argument("--midpoint-step", type=int, default=None,
                   help="Fixed step number for transplant (overrides --midpoint-frac)")
    p.add_argument("--midpoint-frac", type=float, default=0.5,
                   help="Fraction of max_episode_steps for transplant (default: 0.5)")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Max steps per episode")

    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Compute midpoint step
    if args.midpoint_step is not None:
        midpoint = args.midpoint_step
    else:
        midpoint = int(args.max_steps * args.midpoint_frac)
    print(f"Transplant midpoint: step {midpoint}")

    # ---- Load recipient env + policy ----
    # The recipient's environment is the one we actually run episodes in.
    # For the donor, we only need the policy (to generate hidden states),
    # not a separate environment — both agents observe the same scene via
    # the recipient's env.  However, a blind vs sighted policy may expect
    # different observation keys.  We handle this by loading each policy
    # with its own config but sharing the same Habitat environment.
    #
    # Approach: use the RECIPIENT's config/env for the shared environment
    # (since the recipient must run the full episode in baseline).  For the
    # donor, we construct its policy from its own config but feed it the
    # observations available from the recipient's env.  If the donor is
    # blind, it ignores visual observations; if sighted, the env must
    # provide RGB (so the recipient config should include visual sensors).
    #
    # For the simplest case (both configs use the same sensor suite, e.g.,
    # both sighted or both blind), this works directly.  For mixed cases
    # (blind donor + sighted recipient), the blind policy only reads
    # non-visual sensors which are always present.

    # Load both configs, then use the DONOR's config to build the single
    # shared env. The donor's config is typically a superset (sighted
    # donor -> has RGB; blind-only recipient can still read the non-visual
    # stack from the same env). Running two simultaneous habitat.Env
    # instances in one process caused segfaults during sim init.
    print("\n=== Loading configs ===")
    donor_config = load_habitat_config(args.donor_config, args.donor_ckpt, overrides=[
        "habitat.dataset.split=train",
        "habitat.environment.iterator_options.shuffle=True",
        "habitat.environment.iterator_options.group_by_scene=False",
    ])
    recip_config = load_habitat_config(args.recipient_config, args.recipient_ckpt, overrides=[
        "habitat.dataset.split=train",
    ])

    # Pick the env config from whichever side has visual sensors. If donor is
    # blind and recipient is sighted, the env MUST come from recipient (else
    # the recipient policy is built thinking it's blind and ckpt load fails on
    # compression-layer shape mismatch).
    donor_has_rgb = "rgb" in str(donor_config.habitat.simulator.agents).lower()
    recip_has_rgb = "rgb" in str(recip_config.habitat.simulator.agents).lower()
    if donor_has_rgb or not recip_has_rgb:
        env_source = "donor"
        env_config = donor_config.habitat
    else:
        env_source = "recipient"
        env_config = recip_config.habitat
    print(f"\n=== Building shared env (from {env_source} config; donor_rgb={donor_has_rgb} recip_rgb={recip_has_rgb}) ===")
    env = habitat.Env(config=env_config)
    _ = env.reset()

    print("\n=== Loading recipient policy ===")
    recip_policy, recip_hidden, recip_layers, recip_lstm = load_policy(
        recip_config, env, args.recipient_ckpt, device,
    )
    print(f"  config:  {args.recipient_config}")
    print(f"  hidden:  {recip_hidden}, layers: {recip_layers}, LSTM: {recip_lstm}")

    print("\n=== Loading donor policy ===")
    donor_policy, donor_hidden, donor_layers, donor_lstm = load_policy(
        donor_config, env, args.donor_ckpt, device,
    )
    # No second env created. Cross-sensor transplant: donor consumes full
    # observations (RGB + non-visual); blind recipient simply ignores RGB
    # when its policy forward-passes through obs_transforms_batch.
    print(f"  config:  {args.donor_config}")
    print(f"  hidden:  {donor_hidden}, layers: {donor_layers}, LSTM: {donor_lstm}")

    # ---- Compatibility check ----
    if recip_hidden != donor_hidden or recip_layers != donor_layers:
        print(f"\nERROR: Hidden-state shapes are incompatible!")
        print(f"  Donor:     hidden={donor_hidden}, layers={donor_layers}")
        print(f"  Recipient: hidden={recip_hidden}, layers={recip_layers}")
        print("A raw transplant requires identical LSTM dimensions.")
        print("Consider using a learned projection layer for mismatched sizes.")
        env.close()
        sys.exit(1)

    hidden_size = recip_hidden
    num_recurrent_layers = recip_layers

    print(f"\nHidden-state shape: ({num_recurrent_layers}, {hidden_size}) -- compatible")

    # ---- Run evaluation ----
    all_episodes = env.episodes
    n_episodes = min(args.episodes, len(all_episodes))
    print(f"\nEvaluating {n_episodes} episodes (midpoint={midpoint}, max_steps={args.max_steps})")

    results_baseline = []
    results_self = []
    results_cross = []

    # Use the iterator-injection pattern from shortcut.py: set
    # env._episode_iterator to yield exactly the episode we want. This
    # is robust across habitat-lab versions; writing _current_episode
    # directly is not reliable (some versions rebuild it from the
    # iterator on reset() regardless of the attribute).
    def _pin_episode(ep):
        env._episode_iterator = iter([ep])
        env._episode_over = False
        obs = env.reset()
        assert env.current_episode.episode_id == ep.episode_id, (
            f"Episode pinning failed: wanted {ep.episode_id}, "
            f"got {env.current_episode.episode_id}"
        )
        return obs

    for ei in range(n_episodes):
        # ================================================================
        # Condition 1: BASELINE — recipient runs full episode
        # ================================================================
        ep = all_episodes[ei]
        obs = _pin_episode(ep)

        rnn_h, prev_a, mask = _init_rnn_state(num_recurrent_layers, hidden_size, device)
        metrics_baseline = _run_full_episode(
            env, obs, recip_policy, rnn_h, prev_a, mask, device, args.max_steps,
        )
        results_baseline.append(metrics_baseline)

        # ================================================================
        # Condition 2: SELF-TRANSPLANT — recipient first half, then
        #              re-inject its own hidden state (sanity check)
        # ================================================================
        obs = _pin_episode(ep)

        rnn_h, prev_a, mask = _init_rnn_state(num_recurrent_layers, hidden_size, device)

        # Run recipient for the first half
        obs, done_early, rnn_h, prev_a, mask, path_len, _ = _run_first_half(
            env, obs, recip_policy, rnn_h, prev_a, mask, device, midpoint,
        )

        if done_early:
            # Episode ended before midpoint; use whatever we got
            episode_obj = env.current_episode
            goal_pos = np.array(episode_obj.goals[0].position)
            final_pos = env.sim.get_agent_state().position
            dist = np.linalg.norm(final_pos - goal_pos)
            # The last action is unknown (episode terminated), mark as failure
            metrics_self = {
                "success": False, "spl": 0.0,
                "path_length": float(path_len),
                "geodesic": float(env.sim.geodesic_distance(
                    np.array(episode_obj.start_position), goal_pos)),
                "steps": midpoint, "dist_to_goal": float(dist),
            }
        else:
            # "Transplant" — just re-use the same rnn_h (identity operation)
            self_rnn_h = rnn_h.clone()
            metrics_self = _run_second_half(
                env, obs, recip_policy, self_rnn_h, prev_a, mask,
                device, path_len, args.max_steps - midpoint,
            )
        results_self.append(metrics_self)

        # ================================================================
        # Condition 3: CROSS-TRANSPLANT — donor takes first-half actions
        #              in the shared env; at midpoint, donor's hidden
        #              state is injected into recipient and recipient
        #              continues the second half from donor's physical
        #              end-position.
        # ================================================================

        # Reset env to this episode (same starting conditions as baseline).
        recip_obs = _pin_episode(ep)

        donor_rnn_h, donor_prev_a, donor_mask = _init_rnn_state(
            num_recurrent_layers, hidden_size, device,
        )

        # Track path length starting from baseline start.
        recip_path_len = 0.0
        done_donor = False

        # Donor drives first half in shared env.
        for step in range(midpoint):
            if not done_donor:
                recip_obs, done_donor, donor_rnn_h, donor_prev_a, donor_mask, step_path, _ = (
                    _run_first_half(
                        env, recip_obs, donor_policy,
                        donor_rnn_h, donor_prev_a, donor_mask,
                        device, 1,
                    )
                )
                recip_path_len += step_path
            if done_donor:
                break

        done_recip = done_donor
        if done_recip:
            # Episode ended before midpoint
            episode_obj = env.current_episode
            goal_pos = np.array(episode_obj.goals[0].position)
            final_pos = env.sim.get_agent_state().position
            dist = np.linalg.norm(final_pos - goal_pos)
            metrics_cross = {
                "success": False, "spl": 0.0,
                "path_length": float(recip_path_len),
                "geodesic": float(env.sim.geodesic_distance(
                    np.array(episode_obj.start_position), goal_pos)),
                "steps": step + 1, "dist_to_goal": float(dist),
            }
        else:
            # === THE TRANSPLANT ===
            # Donor has taken first-half actions in the shared env; the
            # agent is now at donor's midpoint physical position. Inject
            # donor's hidden state into recipient, carry over prev-action
            # and mask, and let recipient drive the second half from here.
            transplanted_rnn_h = donor_rnn_h.clone()

            metrics_cross = _run_second_half(
                env, recip_obs, recip_policy,
                transplanted_rnn_h, donor_prev_a, donor_mask,
                device, recip_path_len, args.max_steps - midpoint,
            )
        results_cross.append(metrics_cross)

        # ---- Progress ----
        if (ei + 1) % 25 == 0:
            _print_progress(ei, n_episodes, results_baseline, results_self, results_cross)

    env.close()

    # ---- Aggregate results ----
    def _agg(results_list):
        valid = [r for r in results_list if r is not None]
        if not valid:
            return {"n": 0, "success_rate": None, "mean_spl": None}
        return {
            "n": len(valid),
            "success_rate": float(np.mean([r["success"] for r in valid])),
            "mean_spl": float(np.mean([r["spl"] for r in valid])),
            "mean_dist_to_goal": float(np.mean([r["dist_to_goal"] for r in valid])),
            "mean_path_length": float(np.mean([r["path_length"] for r in valid])),
            "mean_steps": float(np.mean([r["steps"] for r in valid])),
        }

    agg_baseline = _agg(results_baseline)
    agg_self = _agg(results_self)
    agg_cross = _agg(results_cross)

    # Compute degradation from transplant
    cross_spl_delta = None
    cross_success_delta = None
    if agg_baseline["mean_spl"] is not None and agg_cross["mean_spl"] is not None:
        cross_spl_delta = agg_cross["mean_spl"] - agg_baseline["mean_spl"]
        cross_success_delta = agg_cross["success_rate"] - agg_baseline["success_rate"]

    summary = {
        "donor_config": args.donor_config,
        "donor_ckpt": args.donor_ckpt,
        "recipient_config": args.recipient_config,
        "recipient_ckpt": args.recipient_ckpt,
        "n_episodes": n_episodes,
        "midpoint_step": midpoint,
        "max_steps": args.max_steps,
        "hidden_size": hidden_size,
        "num_recurrent_layers": num_recurrent_layers,
        "baseline": agg_baseline,
        "self_transplant": agg_self,
        "cross_transplant": agg_cross,
        "cross_spl_delta": cross_spl_delta,
        "cross_success_delta": cross_success_delta,
        "per_episode": {
            "baseline": results_baseline,
            "self_transplant": results_self,
            "cross_transplant": [r for r in results_cross],  # may contain None
        },
    }

    # ---- Print summary ----
    print(f"\n{'='*65}")
    print(f"  MEMORY TRANSPLANT RESULTS")
    print(f"{'='*65}")
    print(f"  Donor:     {args.donor_config}")
    print(f"  Recipient: {args.recipient_config}")
    print(f"  Episodes:  {n_episodes}  |  Midpoint: step {midpoint}")
    print(f"{'='*65}")
    print(f"  {'Condition':<20s}  {'Success':>8s}  {'SPL':>8s}  {'Dist→Goal':>10s}")
    print(f"  {'-'*50}")
    for label, agg in [("Baseline", agg_baseline),
                       ("Self-transplant", agg_self),
                       ("Cross-transplant", agg_cross)]:
        if agg["mean_spl"] is not None:
            print(f"  {label:<20s}  {agg['success_rate']:>8.3f}  "
                  f"{agg['mean_spl']:>8.3f}  {agg['mean_dist_to_goal']:>10.3f}")
        else:
            print(f"  {label:<20s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>10s}")
    print(f"  {'-'*50}")
    if cross_spl_delta is not None:
        print(f"  Cross-transplant SPL delta:     {cross_spl_delta:+.3f}")
        print(f"  Cross-transplant success delta: {cross_success_delta:+.3f}")
    print(f"{'='*65}")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.out}")


def _print_progress(ei, n_episodes, results_baseline, results_self, results_cross):
    """Print running averages."""
    def _sr(lst):
        valid = [r for r in lst if r is not None]
        if not valid:
            return "N/A"
        return f"{np.mean([r['success'] for r in valid]):.3f}"

    def _spl(lst):
        valid = [r for r in lst if r is not None]
        if not valid:
            return "N/A"
        return f"{np.mean([r['spl'] for r in valid]):.3f}"

    print(f"  [{ei+1}/{n_episodes}]  baseline SR={_sr(results_baseline)} SPL={_spl(results_baseline)}  |  "
          f"self SR={_sr(results_self)} SPL={_spl(results_self)}  |  "
          f"cross SR={_sr(results_cross)} SPL={_spl(results_cross)}")


if __name__ == "__main__":
    main()
