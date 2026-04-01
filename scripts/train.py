"""
CS503 Project — Train Navigation Agent

Usage:
    python scripts/train.py --config cfgs/foveated.yaml
    python scripts/train.py --config cfgs/uniform.yaml
    python scripts/train.py --config cfgs/matched_compute.yaml

For cluster:
    sbatch submit_job.sh cfgs/foveated.yaml <WANDB_KEY> 1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.nav_env import NavigationEnv
from src.envs.wrappers import FoveatedWrapper, UniformWrapper, MatchedComputeWrapper
from src.models import FoveatedNavigationAgent
from src.training.rollout import RolloutBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train navigation agent")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides (key=value)")
    return parser.parse_args()


def load_config(config_path: str, overrides: list[str] = None) -> OmegaConf:
    """Load config with optional base inheritance."""
    cfg = OmegaConf.load(config_path)

    # Handle 'defaults' inheritance
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


def make_env(cfg):
    """Create environment with appropriate wrapper based on config."""
    env = NavigationEnv(
        env_id=cfg.env.id,
        image_size=cfg.env.image_size,
        max_steps=cfg.env.max_steps,
        seed=cfg.get("seed", None),
    )

    if cfg.foveation.enabled:
        env = FoveatedWrapper(
            env,
            fovea_radius=cfg.foveation.fovea_radius,
            blur_sigma_max=cfg.foveation.blur_sigma_max,
            falloff=cfg.foveation.falloff,
        )
    elif cfg.env.get("image_size", 64) < 64:
        env = MatchedComputeWrapper(env, target_size=cfg.env.image_size)
    else:
        env = UniformWrapper(env)

    return env


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    # Run name
    run_name = cfg.get("run_name", "auto")
    if run_name == "auto":
        run_name = f"{Path(args.config).stem}_{int(time.time())}"

    output_dir = Path(cfg.get("output_dir", "./outputs")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info("=" * 60)
    logger.info(f"  Agentic Cognitive Maps — Training")
    logger.info(f"  Config:     {args.config}")
    logger.info(f"  Run name:   {run_name}")
    logger.info(f"  Output:     {output_dir}")
    logger.info(f"  Device:     {device}")
    logger.info(f"  Foveated:   {cfg.foveation.enabled}")
    logger.info(f"  Gaze ctrl:  {cfg.foveation.gaze_action}")
    logger.info("=" * 60)

    # Create environment
    env = make_env(cfg)

    # Create agent
    agent = FoveatedNavigationAgent(
        image_size=cfg.env.image_size,
        encoder_channels=list(cfg.model.encoder.channels),
        hidden_size=cfg.model.memory.hidden_size,
        n_actions=env.action_space.n,
        gaze_enabled=cfg.foveation.gaze_action,
    ).to(device)

    n_params = agent.count_parameters()
    logger.info(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # W&B
    if cfg.get("log_wandb", False):
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    # TODO: Implement training loop
    # This is where Member B connects the pieces:
    # 1. Create PPOTrainer
    # 2. Create RolloutBuffer
    # 3. Collect rollouts (with hidden states for probing)
    # 4. PPO update
    # 5. Evaluate periodically
    # 6. Save checkpoints

    logger.info("")
    logger.info("⚠️  Training loop not yet implemented.")
    logger.info("   Member B: implement the rollout collection + PPO update loop.")
    logger.info("   The environment, agent, and buffer are ready.")

    env.close()


if __name__ == "__main__":
    main()
