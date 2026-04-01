"""
CS503 Project — Main Training Script

Usage:
    python scripts/train.py --config cfgs/example.yaml
    torchrun --nproc_per_node=2 scripts/train.py --config cfgs/example.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="CS503 Project Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in key=value format (e.g. lr=0.001 batch_size=64)",
    )
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        logger.info(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
        return rank, world_size, local_rank
    else:
        logger.info("Single-GPU training")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0


def setup_wandb(cfg, rank):
    """Initialize W&B logging (rank 0 only)."""
    if cfg.get("log_wandb", False) and rank == 0:
        try:
            import wandb

            run_name = cfg.get("wandb_run_name", "auto")
            if run_name == "auto":
                run_name = cfg.get("run_name", f"run_{int(time.time())}")

            wandb.init(
                project=cfg.get("wandb_project", "CS503_Project"),
                entity=cfg.get("wandb_entity", None),
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info(f"W&B initialized: {wandb.run.url}")
            return wandb
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    return None


def build_model(cfg):
    """Instantiate model from config.
    
    Replace this with your actual model construction.
    """
    model_cfg = cfg.get("model_config", {})
    
    # TODO: Replace with your model instantiation
    # Example using Hydra-style instantiation:
    # from hydra.utils import instantiate
    # model = instantiate(model_cfg)
    
    logger.info(f"Model config: {OmegaConf.to_yaml(model_cfg)}")
    raise NotImplementedError(
        "Replace build_model() with your actual model construction. "
        "See src/models/ for where to define your models."
    )


def build_dataloaders(cfg, rank, world_size):
    """Build train and eval dataloaders.
    
    Replace this with your actual data loading.
    """
    # TODO: Replace with your dataloader construction
    # Example:
    # from src.data import MyDataset
    # train_dataset = MyDataset(**cfg.train_dataset_config)
    # ...
    
    raise NotImplementedError(
        "Replace build_dataloaders() with your actual data loading. "
        "See src/data/ for where to define your datasets."
    )


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, epoch, rank, wandb_run):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # TODO: Implement your training step
        # Example:
        # images, labels = batch
        # images = images.cuda(non_blocking=True)
        # labels = labels.cuda(non_blocking=True)
        #
        # with torch.cuda.amp.autocast(dtype=torch.float16):
        #     output = model(images)
        #     loss = criterion(output, labels)
        #
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()
        
        pass

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, eval_loader, cfg, rank):
    """Evaluate the model."""
    model.eval()
    
    # TODO: Implement your evaluation logic
    # Return a dict of metrics
    
    return {"eval_loss": 0.0}


def main():
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        override_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    # Setup output directory
    output_dir = Path(cfg.get("output_dir", "./outputs/default"))
    if str(output_dir).endswith("auto"):
        run_name = cfg.get("run_name", f"run_{int(time.time())}")
        if run_name == "auto":
            run_name = f"run_{int(time.time())}"
        output_dir = output_dir.parent / run_name
    
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        logger.info(f"Output directory: {output_dir}")

    # Setup W&B
    wandb_run = setup_wandb(cfg, rank)

    # Build model, data, optimizer
    # model = build_model(cfg)
    # train_loader, eval_loader = build_dataloaders(cfg, rank, world_size)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logger.info("=" * 60)
    logger.info("  CS503 Project — Training Script")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  GPUs: {world_size}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  This is a scaffold. Implement your model, data, and training loop.")
    logger.info("   See: src/models/, src/data/, and this file's TODO comments.")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    if wandb_run and rank == 0:
        wandb_run.finish()


if __name__ == "__main__":
    main()
