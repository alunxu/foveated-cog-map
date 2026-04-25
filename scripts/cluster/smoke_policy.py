"""
Smoke-test: instantiate each new foveated policy + run one forward pass
on synthetic input.  Catches import/registration/encoder-construction
errors without launching a full training job.

Usage:  python scripts/cluster/smoke_policy.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from gym import spaces

# Trigger registration side effects.
sys.path.insert(0, ".")
import src.habitat  # noqa: F401

from habitat_baselines.common.baseline_registry import baseline_registry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_obs_space(image_size: int = 256) -> spaces.Dict:
    return spaces.Dict({
        "rgb": spaces.Box(
            low=0, high=255,
            shape=(image_size, image_size, 3), dtype=np.uint8,
        ),
        "episodic_gps": spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32,
        ),
        "episodic_compass": spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32,
        ),
        "goal_in_start_frame": spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32,
        ),
        "close_to_goal_indicator": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        ),
    })


def _make_action_space():
    return spaces.Discrete(4)


def smoke_one(name: str) -> bool:
    """Construct policy by registry name + run one forward pass."""
    print(f"\n=== smoke {name} ===", flush=True)
    cls = baseline_registry.get_policy(name)
    if cls is None:
        print(f"  MISSING policy class for {name}")
        return False

    obs_space = _make_obs_space()
    act_space = _make_action_space()

    try:
        policy = cls(
            observation_space=obs_space,
            action_space=act_space,
            hidden_size=512,
            num_recurrent_layers=3,
            rnn_type="LSTM",
            backbone="resnet18",
            normalize_visual_inputs="rgb" in obs_space.spaces,
        ).to(DEVICE).eval()
    except Exception as e:
        print(f"  CONSTRUCTION FAILED: {type(e).__name__}: {e}")
        return False

    # Dummy obs batch
    B = 2
    obs = {
        "rgb": torch.randint(
            0, 255, (B, 256, 256, 3), dtype=torch.uint8, device=DEVICE,
        ),
        "episodic_gps": torch.randn(B, 2, device=DEVICE),
        "episodic_compass": torch.randn(B, 1, device=DEVICE),
        "goal_in_start_frame": torch.randn(B, 2, device=DEVICE),
        "close_to_goal_indicator": torch.zeros(B, 1, device=DEVICE),
    }
    rnn_h = torch.zeros(B, 6, 512, device=DEVICE)
    prev_a = torch.zeros(B, 1, dtype=torch.long, device=DEVICE)
    masks = torch.ones(B, 1, dtype=torch.bool, device=DEVICE)

    try:
        with torch.no_grad():
            _ = policy.act(obs, rnn_h, prev_a, masks, deterministic=True)
    except Exception as e:
        print(f"  FORWARD FAILED: {type(e).__name__}: {e}")
        return False

    print("  OK")
    return True


def main() -> int:
    fails = 0
    for name in [
        "FoveatedWijmansPolicy",        # baseline (already known to work)
        "FoveatedNormalisedWijmansPolicy",  # F2
        "FoveatedStrongWijmansPolicy",      # F4
        "FoveatedLogPolarWijmansPolicy",    # F3
    ]:
        if not smoke_one(name):
            fails += 1
    print(f"\n=== summary: {fails} fail(s) ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
