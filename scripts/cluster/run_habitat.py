"""
Custom entry point for habitat-baselines training.

Registers our custom policies (e.g. FoveatedPointNavResNetPolicy) before
invoking the standard habitat_baselines.run main function.
"""

import sys
import os

# Add project root to path so 'src.habitat' is importable
project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(project_root))

# Register custom policies with habitat-baselines
import src.habitat  # noqa: F401

# Run habitat-baselines
from habitat_baselines.run import main

if __name__ == "__main__":
    main()
