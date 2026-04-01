"""
CS503 Project — Evaluation & Inference Script

Usage:
    python scripts/evaluate.py --config cfgs/example.yaml --checkpoint outputs/run_name/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="CS503 Project Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    # checkpoint = torch.load(args.checkpoint, map_location=device)

    # Build model and load weights
    # model = build_model(cfg)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)
    # model.eval()

    # Build eval dataloader
    # eval_loader = build_eval_loader(cfg, split=args.split)

    # Run evaluation
    # metrics = evaluate(model, eval_loader, device)

    # Save results
    output_dir = Path(args.output_dir or f"outputs/eval_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  CS503 Project — Evaluation Script")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Split: {args.split}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  This is a scaffold. Implement your evaluation logic.")


if __name__ == "__main__":
    main()
