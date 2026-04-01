"""
Gaze-memory coupling analysis.

Tests H3 using collected probing data:
- Correlation between gaze direction and memory uncertainty
- Comparison to Bayesian ideal observer

Usage:
    python scripts/analyze_gaze.py --data outputs/foveated_agent/probing_data/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.gaze_memory import gaze_uncertainty_correlation, gaze_optimality_score


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze-memory coupling analysis")
    parser.add_argument("--data", type=str, required=True, help="Path to probing data dir")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)

    logger.info("=" * 60)
    logger.info("  Gaze-Memory Coupling Analysis (H3)")
    logger.info(f"  Data dir: {data_dir}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  Load probing data and run gaze analysis.")
    logger.info("   See src/analysis/gaze_memory.py for the metrics.")

    # TODO: Member D implements:
    # 1. Load gaze positions, uncertainty maps from probing data
    # 2. correlation_results = gaze_uncertainty_correlation(gaze, uncertainty_maps)
    # 3. optimality_results = gaze_optimality_score(gaze, uncertainty_maps, fov_transform, ...)
    # 4. Save results


if __name__ == "__main__":
    main()
