"""
Train linear probes on collected hidden states.

Loads probing data (hidden states + ground truth) and trains the
four linear probes: occupancy, target, foveation history, uncertainty.

Usage:
    python scripts/train_probes.py --data outputs/foveated_agent/probing_data/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.probing.analysis import run_probing_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Train linear probes")
    parser.add_argument("--data", type=str, required=True, help="Path to probing data dir")
    parser.add_argument("--output", type=str, default=None, help="Where to save results")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)

    # TODO: Load probing data from disk (saved by collect_probing_data.py)
    # hidden_states = np.load(data_dir / "hidden_states.npy")
    # ground_truth_data = ...
    # gaze_histories = ...

    logger.info("=" * 60)
    logger.info("  Linear Probe Training")
    logger.info(f"  Data dir: {data_dir}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  Load probing data and call run_probing_analysis().")
    logger.info("   See src/probing/analysis.py for the pipeline.")

    # results = run_probing_analysis(hidden_states, ground_truth_data, gaze_histories, ...)
    #
    # output_path = Path(args.output or data_dir / "probe_results.json")
    # with open(output_path, "w") as f:
    #     json.dump(results, f, indent=2)
    # logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
