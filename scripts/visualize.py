"""
CS503 Project — Visualization Script

Generate qualitative results, figures, and visualizations for the report.

Usage:
    python scripts/visualize.py --config cfgs/example.yaml --checkpoint outputs/run_name/best.pt --output_dir docs/assets
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import torch
from loguru import logger
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="CS503 Project Visualization")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="docs/assets", help="Where to save figures")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  CS503 Project — Visualization Script")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  This is a scaffold. Add your visualization code here.")
    logger.info("   Generated figures will be saved to docs/assets/ for the project webpage.")

    # TODO: Add visualization code
    # Examples:
    # - Plot training curves from W&B or local logs
    # - Generate sample predictions vs ground truth
    # - Create attention/feature visualizations
    # - Produce ablation study comparison plots
    # - Generate videos or animations

    # Example placeholder:
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.set_title("Training Loss")
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Loss")
    # fig.savefig(output_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    # plt.close(fig)


if __name__ == "__main__":
    main()
