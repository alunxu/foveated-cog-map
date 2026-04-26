"""Quick diagnostic for shortcut eval 0% success bug."""
import os, sys, numpy as np, torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))
import src.habitat  # noqa: F401
from src.utils.habitat_env import load_habitat_config, load_policy, compute_spl
import habitat

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    config = load_habitat_config(args.config_name, args.ckpt, overrides=[
        "habitat.dataset.split=train",
        "habitat.environment.iterator_options.shuffle=False",
        "habitat.environment.max_episode_steps=2000",
    ])
    env = habitat.Env(config=config.habitat)
    obs = env.reset()

    # --- Check checkpoint key format ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    all_keys = list(sd.keys())
    ac_keys = [k for k in all_keys if k.startswith("actor_critic.")]
    print(f"\nCheckpoint: {len(all_keys)} total keys, {len(ac_keys)} with 'actor_critic.' prefix")
    print(f"First 5 keys: {all_keys[:5]}")
    stripped = {k.replace('actor_critic.', ''): v for k, v in sd.items() if k.startswith('actor_critic.')}
    print(f"After stripping: {len(stripped)} keys")

    policy, hidden_size, num_recurrent_layers, rnn_is_lstm = load_policy(
        config, env, args.ckpt, device,
    )

    # --- Check which weights loaded ---
    policy_keys = set(policy.state_dict().keys())
    loaded_keys = set(stripped.keys())
    matched = policy_keys & loaded_keys
    missing = policy_keys - loaded_keys
    unexpected = loaded_keys - policy_keys
    print(f"\nWeight loading: {len(matched)} matched, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"  MISSING (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"  UNEXPECTED (first 5): {list(unexpected)[:5]}")

    print(f"\nObs keys: {list(obs.keys())}")
    for k, v in obs.items():
        print(f"  {k}: shape={np.array(v).shape}, dtype={np.array(v).dtype}, "
              f"min={np.array(v).min():.4f}, max={np.array(v).max():.4f}")

    ep = env.current_episode
    start_pos = np.array(ep.start_position)
    goal_pos = np.array(ep.goals[0].position)
    geodesic = env.sim.geodesic_distance(start_pos, goal_pos)
    print(f"\nEpisode: scene={os.path.basename(ep.scene_id)}")
    print(f"  start={start_pos}, goal={goal_pos}")
    print(f"  geodesic={geodesic:.3f}")

    # Init hidden state
    rnn_hidden = torch.zeros(1, num_recurrent_layers, hidden_size, device=device)
    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
    not_done_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

    action_counts = {}
    path_length = 0.0
    prev_pos = start_pos.copy()

    for step_i in range(50):  # Just 50 steps
        batch = {k: torch.from_numpy(np.expand_dims(v, 0)).to(device)
                 for k, v in obs.items()}

        with torch.no_grad():
            action_data = policy.act(batch, rnn_hidden, prev_action, not_done_mask, deterministic=True)

        rnn_hidden = action_data.rnn_hidden_states
        prev_action = action_data.actions
        not_done_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

        # Print raw action data for first 5 steps
        if step_i < 5:
            print(f"\n  Step {step_i}:")
            print(f"    actions={action_data.actions}")
            print(f"    env_actions={action_data.env_actions}")
            print(f"    actions.shape={action_data.actions.shape}")
            print(f"    env_actions.shape={action_data.env_actions.shape}")
            act_val = action_data.actions[0].item()
            env_act_val = action_data.env_actions[0].item()
            print(f"    actions[0].item()={act_val}, env_actions[0].item()={env_act_val}")

        action_int = action_data.env_actions[0].item()
        action_counts[action_int] = action_counts.get(action_int, 0) + 1

        obs = env.step(action_int)
        agent_state = env.sim.get_agent_state()
        cur_pos = agent_state.position
        path_length += np.linalg.norm(cur_pos - prev_pos)
        prev_pos = cur_pos.copy()

        dist = np.linalg.norm(cur_pos - goal_pos)

        if step_i < 5 or step_i % 10 == 0:
            print(f"    pos={cur_pos}, dist_to_goal={dist:.3f}, episode_over={env.episode_over}")

        if env.episode_over:
            print(f"\n  Episode ended at step {step_i}")
            print(f"  Last action: {action_int}")
            print(f"  Distance to goal: {dist:.3f}")
            print(f"  Success (action==0 & dist<0.2): {action_int==0 and dist<0.2}")
            break

    print(f"\nAction distribution: {action_counts}")
    print(f"Path length: {path_length:.3f}")
    print(f"Final dist to goal: {np.linalg.norm(env.sim.get_agent_state().position - goal_pos):.3f}")
    env.close()

if __name__ == "__main__":
    main()
