"""
CS503 Project — Train Navigation Agent

Usage:
    python scripts/train.py --config cfgs/blind.yaml
    python scripts/train.py --config cfgs/uniform.yaml
    python scripts/train.py --config cfgs/foveated.yaml
    python scripts/train.py --config cfgs/matched_compute.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
from gymnasium.vector import SyncVectorEnv
from loguru import logger
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.nav_env import NavigationEnv
from src.envs.wrappers import (
    PointGoalWrapper, BlindWrapper, FoveatedWrapper,
    UniformWrapper, MatchedComputeWrapper, N_ACTIONS,
)
from src.models import NavigationAgent
from src.training.rollout import RolloutBuffer
from src.training.ppo import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train navigation agent")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides (key=value)")
    return parser.parse_args()


def load_config(config_path: str, overrides: list[str] = None) -> OmegaConf:
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        base_path = Path(config_path).parent
        for default in cfg.defaults:
            base_cfg = OmegaConf.load(base_path / f"{default}.yaml")
            cfg = OmegaConf.merge(base_cfg, cfg)
        del cfg["defaults"]
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def make_env_fn(cfg, seed_offset=0):
    """Return a callable that creates a single wrapped environment."""
    def _init():
        encoder_type = cfg.model.encoder.get("type", "cnn")
        is_matched_compute = (
            not cfg.foveation.enabled
            and encoder_type != "vector"
            and cfg.env.get("matched_compute", False)
        )

        # Matched-compute: always render at 64×64, then downsample via wrapper
        nav_image_size = 64 if is_matched_compute else cfg.env.image_size

        env = NavigationEnv(
            env_id=cfg.env.id,
            image_size=nav_image_size,
            max_steps=cfg.env.max_steps,
            seed=cfg.get("seed", 42) + seed_offset,
        )
        env = PointGoalWrapper(
            env,
            success_reward=cfg.env.success_reward,
            time_penalty=cfg.env.time_penalty,
        )

        if encoder_type == "vector":
            env = BlindWrapper(env, pointgoal_dim=cfg.env.pointgoal_dim)
        elif cfg.foveation.enabled:
            env = FoveatedWrapper(
                env,
                fovea_radius=cfg.foveation.fovea_radius,
                blur_sigma_max=cfg.foveation.blur_sigma_max,
                falloff=cfg.foveation.falloff,
            )
        elif is_matched_compute:
            env = MatchedComputeWrapper(env, target_size=cfg.env.image_size)
        else:
            env = UniformWrapper(env)

        return env
    return _init


def preprocess_obs(obs, encoder_type, device):
    """Convert numpy observations to torch tensors."""
    if encoder_type == "vector":
        return torch.from_numpy(obs).float().to(device)
    else:
        # (N, H, W, C) -> (N, C, H, W), normalize to [0,1]
        return torch.from_numpy(obs).float().permute(0, 3, 1, 2).to(device) / 255.0


def evaluate(agent, cfg, device, n_episodes=20):
    """Evaluate agent on fresh episodes."""
    encoder_type = cfg.model.encoder.get("type", "cnn")
    env_fn = make_env_fn(cfg, seed_offset=10000)
    env = env_fn()

    successes = []
    spls = []
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        hidden = None
        prev_action = torch.zeros(1, dtype=torch.long, device=device)
        total_reward = 0.0
        steps = 0
        initial_geodesic = info["initial_geodesic"]

        done = False
        while not done:
            obs_t = preprocess_obs(obs[np.newaxis], encoder_type, device)
            pg = torch.from_numpy(info["pointgoal"][np.newaxis]).float().to(device)

            with torch.no_grad():
                action, gaze, _, _, new_hidden = agent.act(
                    obs_t, hidden, pointgoal=pg, prev_action=prev_action,
                    deterministic=True,
                )

            act_int = action.item()

            if cfg.foveation.get("gaze_action", False) and gaze is not None:
                gaze_np = gaze.cpu().numpy().squeeze()
                env_action = np.array([float(act_int), gaze_np[0], gaze_np[1]], dtype=np.float32)
            else:
                env_action = act_int

            obs, reward, terminated, truncated, info = env.step(env_action)
            hidden = new_hidden
            prev_action = action.unsqueeze(0) if action.dim() == 0 else action
            total_reward += reward
            steps += 1
            done = terminated or truncated

        success = float(terminated)
        successes.append(success)
        if success and initial_geodesic > 0:
            spl = success * (initial_geodesic / max(initial_geodesic, steps))
        else:
            spl = 0.0
        spls.append(spl)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    return {
        "eval/success_rate": np.mean(successes),
        "eval/spl": np.mean(spls),
        "eval/mean_reward": np.mean(episode_rewards),
        "eval/mean_length": np.mean(episode_lengths),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    encoder_type = cfg.model.encoder.get("type", "cnn")
    is_foveated = cfg.foveation.get("gaze_action", False)

    # Run name
    run_name = cfg.get("run_name", "auto")
    if run_name == "auto":
        run_name = f"{Path(args.config).stem}_{int(time.time())}"

    output_dir = Path(cfg.get("output_dir", "./outputs")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info("=" * 60)
    logger.info(f"  Agentic Cognitive Maps — Training")
    logger.info(f"  Config:       {args.config}")
    logger.info(f"  Run name:     {run_name}")
    logger.info(f"  Output:       {output_dir}")
    logger.info(f"  Device:       {device}")
    logger.info(f"  Encoder:      {encoder_type}")
    logger.info(f"  Foveated:     {cfg.foveation.enabled}")
    logger.info(f"  Gaze ctrl:    {is_foveated}")
    logger.info("=" * 60)

    # Create vectorized environment
    num_envs = cfg.env.num_envs
    vec_env = SyncVectorEnv([make_env_fn(cfg, i) for i in range(num_envs)])

    # Create agent
    n_actions = cfg.env.get("n_actions", N_ACTIONS)
    agent = NavigationAgent(
        encoder_type=encoder_type,
        image_size=cfg.env.image_size,
        encoder_channels=list(cfg.model.encoder.get("channels", [16, 32, 64])),
        hidden_size=cfg.model.memory.hidden_size,
        num_memory_layers=cfg.model.memory.get("num_layers", 1),
        n_actions=n_actions,
        gaze_enabled=is_foveated,
        pointgoal_dim=cfg.env.get("pointgoal_dim", 4),
    ).to(device)

    n_params = agent.count_parameters()
    logger.info(f"  Parameters:   {n_params:,} ({n_params/1e6:.2f}M)")

    # Create trainer and buffer
    trainer = PPOTrainer(
        agent,
        lr=cfg.training.learning_rate,
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
        clip_range=cfg.training.clip_range,
        entropy_coef=cfg.training.entropy_coef,
        value_coef=cfg.training.value_coef,
        max_grad_norm=cfg.training.max_grad_norm,
        n_epochs=cfg.training.n_epochs,
        batch_size=cfg.training.batch_size,
    )

    obs_space = vec_env.single_observation_space
    obs_dtype = np.float32 if encoder_type == "vector" else np.uint8

    buffer = RolloutBuffer(
        n_steps=cfg.training.n_steps,
        n_envs=num_envs,
        obs_shape=obs_space.shape,
        hidden_size=cfg.model.memory.hidden_size,
        num_memory_layers=cfg.model.memory.get("num_layers", 1),
        pointgoal_dim=cfg.env.get("pointgoal_dim", 4),
        has_gaze=is_foveated,
        obs_dtype=obs_dtype,
    )

    # W&B
    wandb_run = None
    if cfg.get("log_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    # ---- Training loop ----
    n_steps = cfg.training.n_steps
    total_timesteps = cfg.training.total_timesteps
    n_updates = total_timesteps // (n_steps * num_envs)

    obs, info = vec_env.reset()
    hidden = torch.zeros(
        cfg.model.memory.get("num_layers", 1), num_envs,
        cfg.model.memory.hidden_size, device=device,
    )
    prev_actions = torch.zeros(num_envs, dtype=torch.long, device=device)

    total_steps = 0
    episode_rewards_buffer = []  # track per-episode returns
    episode_lengths_buffer = []
    running_rewards = np.zeros(num_envs)
    running_lengths = np.zeros(num_envs, dtype=np.int64)

    logger.info(f"  Updates:      {n_updates}")
    logger.info(f"  Steps/update: {n_steps * num_envs}")
    logger.info("")

    for update in range(1, n_updates + 1):
        buffer.reset()

        for step in range(n_steps):
            obs_t = preprocess_obs(obs, encoder_type, device)
            pointgoals = torch.from_numpy(
                np.stack(info["pointgoal"])
            ).float().to(device)

            with torch.no_grad():
                action, gaze, log_prob, value, new_hidden = agent.act(
                    obs_t, hidden, pointgoal=pointgoals, prev_action=prev_actions,
                )

            # Format actions for env
            act_np = action.cpu().numpy()
            if is_foveated and gaze is not None:
                gaze_np = gaze.cpu().numpy()
                # Pack as flat array [movement, gaze_x, gaze_y] per env
                env_actions = np.column_stack([
                    act_np.astype(np.float32),
                    gaze_np,
                ])  # (N, 3)
            else:
                env_actions = act_np

            next_obs, rewards, terminated, truncated, next_info = vec_env.step(env_actions)
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            # Extract per-env info
            agent_pos = np.stack(next_info["agent_pos"]).astype(np.float32) if "agent_pos" in next_info else None
            agent_dir = np.array(next_info["agent_dir"], dtype=np.int64) if "agent_dir" in next_info else None
            collisions = np.array(next_info["collision"], dtype=np.float32) if "collision" in next_info else None

            # Store in buffer
            buffer.add(
                obs=obs,
                action=act_np,
                prev_action=prev_actions.cpu().numpy(),
                log_prob=log_prob.cpu().numpy(),
                value=value.cpu().numpy(),
                reward=rewards,
                done=dones,
                hidden_state=hidden.cpu().numpy(),  # (num_layers, N, H)
                pointgoal=np.stack(info["pointgoal"]),
                gaze_action=gaze_np if is_foveated and gaze is not None else None,
                agent_pos=agent_pos,
                agent_dir=agent_dir,
                collision=collisions,
            )

            # Track episode stats
            running_rewards += rewards
            running_lengths += 1
            for i in range(num_envs):
                if dones[i]:
                    episode_rewards_buffer.append(running_rewards[i])
                    episode_lengths_buffer.append(running_lengths[i])
                    running_rewards[i] = 0.0
                    running_lengths[i] = 0

            # Reset hidden for done envs
            hidden = new_hidden.clone()
            for i in range(num_envs):
                if dones[i]:
                    hidden[:, i, :] = 0.0

            obs = next_obs
            info = next_info
            prev_actions = action
            total_steps += num_envs

        # Bootstrap value
        with torch.no_grad():
            obs_t = preprocess_obs(obs, encoder_type, device)
            pointgoals = torch.from_numpy(
                np.stack(info["pointgoal"])
            ).float().to(device)
            next_value = agent.get_value(obs_t, hidden, pointgoal=pointgoals, prev_action=prev_actions)

        # PPO update
        rollout_data = buffer.get_as_tensors(device)
        metrics = trainer.update(rollout_data, next_value)

        # Logging
        if len(episode_rewards_buffer) > 0:
            mean_reward = np.mean(episode_rewards_buffer[-100:])
            mean_length = np.mean(episode_lengths_buffer[-100:])
        else:
            mean_reward = 0.0
            mean_length = 0.0

        if update % 10 == 0 or update == 1:
            logger.info(
                f"Update {update:5d}/{n_updates} | "
                f"Steps {total_steps:>10,d} | "
                f"Reward {mean_reward:7.2f} | "
                f"EpLen {mean_length:6.1f} | "
                f"PL {metrics['policy_loss']:.4f} | "
                f"VL {metrics['value_loss']:.4f} | "
                f"Ent {metrics['entropy']:.4f}"
            )

        if wandb_run:
            log_dict = {
                "train/mean_reward": mean_reward,
                "train/mean_length": mean_length,
                "train/total_steps": total_steps,
                **{f"train/{k}": v for k, v in metrics.items()},
            }
            wandb_run.log(log_dict, step=total_steps)

        # Evaluation
        eval_freq = cfg.eval.get("freq", 50_000)
        if total_steps % eval_freq < n_steps * num_envs:
            eval_metrics = evaluate(agent, cfg, device, cfg.eval.n_episodes)
            logger.info(
                f"  EVAL | Success {eval_metrics['eval/success_rate']:.2%} | "
                f"SPL {eval_metrics['eval/spl']:.3f} | "
                f"Reward {eval_metrics['eval/mean_reward']:.2f}"
            )
            if wandb_run:
                wandb_run.log(eval_metrics, step=total_steps)

        # Checkpointing
        ckpt_freq = cfg.eval.get("save_ckpt_freq", 200_000)
        if total_steps % ckpt_freq < n_steps * num_envs:
            ckpt_path = output_dir / f"checkpoint_{total_steps}.pt"
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "timesteps": total_steps,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    ckpt_path = output_dir / "checkpoint_final.pt"
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "timesteps": total_steps,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, ckpt_path)
    logger.info(f"Training complete. Final checkpoint: {ckpt_path}")

    # Final eval
    eval_metrics = evaluate(agent, cfg, device, cfg.eval.n_episodes)
    logger.info(
        f"  FINAL EVAL | Success {eval_metrics['eval/success_rate']:.2%} | "
        f"SPL {eval_metrics['eval/spl']:.3f}"
    )

    vec_env.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
