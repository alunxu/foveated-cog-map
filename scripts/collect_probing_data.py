"""
Collect probing data from a trained agent.

Runs the trained agent through episodes, collecting at each timestep:
- GRU hidden states (the "cognitive map")
- Ground truth (occupancy, target, agent position)
- Gaze history and uncertainty maps

Saves everything to disk for offline probing analysis.

Usage:
    python scripts/collect_probing_data.py \
        --config cfgs/foveated.yaml \
        --checkpoint outputs/foveated_agent/best.pt \
        --output_dir outputs/foveated_agent/probing_data \
        --n_episodes 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect probing data")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()

    # TODO: Member C implements this script
    # 1. Load config and trained agent from checkpoint
    # 2. Create environment (same condition as training)
    # 3. For each episode:
    #    a. Reset environment
    #    b. At each step, run agent forward pass
    #    c. Store: hidden_state, ground_truth, gaze_pos, observation
    # 4. Save all collected data as .npz files

    logger.info("=" * 60)
    logger.info("  Probing Data Collection")
    logger.info(f"  Config:      {args.config}")
    logger.info(f"  Checkpoint:  {args.checkpoint}")
    logger.info(f"  Episodes:    {args.n_episodes}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  Not yet implemented — Member C's responsibility.")
    logger.info("   Collect hidden_states + ground_truth at each timestep.")


if __name__ == "__main__":
    main()
