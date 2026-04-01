"""
Rollout buffer — stores trajectories for PPO and probing.

Crucially, stores GRU hidden states at each timestep so that
Member C can later train linear probes on them.

Member B is responsible for this module.
"""

import torch
import numpy as np


class RolloutBuffer:
    """Stores rollout data for PPO training and probing analysis.

    Stores at each timestep:
        - observations (for recomputing forward pass)
        - actions, log_probs (for PPO ratio)
        - values, rewards, dones (for GAE)
        - hidden_states (for probing — the cognitive map!)
        - gaze_positions (for gaze-memory coupling analysis)
        - ground_truth (occupancy, target_pos — probe targets)

    Args:
        n_steps: Number of steps per rollout.
        n_envs: Number of parallel environments.
        obs_shape: Shape of a single observation.
        hidden_size: GRU hidden state dimension.
        store_ground_truth: Whether to store GT for probing.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple,
        hidden_size: int,
        store_ground_truth: bool = True,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.store_ground_truth = store_ground_truth
        self.pos = 0

        # Core PPO data
        self.observations = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Hidden states (for probing)
        self.hidden_states = np.zeros(
            (n_steps, n_envs, hidden_size), dtype=np.float32
        )

        # Gaze positions (for gaze-memory coupling, Member D)
        self.gaze_positions = np.zeros((n_steps, n_envs, 2), dtype=np.float32)

        # Ground truth (for probe targets, Member C)
        if store_ground_truth:
            self.agent_positions = np.zeros((n_steps, n_envs, 2), dtype=np.float32)
            self.agent_directions = np.zeros((n_steps, n_envs), dtype=np.int64)
            # Variable-size ground truth stored as list
            self.occupancy_grids = []
            self.target_positions = []

    def add(
        self,
        obs, action, log_prob, value, reward, done,
        hidden_state, gaze_pos=None, ground_truth=None,
    ):
        """Add one timestep of data."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.hidden_states[self.pos] = hidden_state

        if gaze_pos is not None:
            self.gaze_positions[self.pos] = gaze_pos

        if self.store_ground_truth and ground_truth is not None:
            self.agent_positions[self.pos] = ground_truth.get("agent_pos", [0, 0])
            self.agent_directions[self.pos] = ground_truth.get("agent_dir", 0)
            self.occupancy_grids.append(ground_truth.get("occupancy_grid"))
            self.target_positions.append(ground_truth.get("target_pos"))

        self.pos += 1

    def reset(self):
        """Reset buffer position for new rollout."""
        self.pos = 0
        if self.store_ground_truth:
            self.occupancy_grids = []
            self.target_positions = []

    def get_as_tensors(self, device: torch.device) -> dict:
        """Convert buffer to torch tensors for PPO update."""
        return {
            "observations": torch.from_numpy(self.observations).to(device),
            "actions": torch.from_numpy(self.actions).to(device),
            "log_probs": torch.from_numpy(self.log_probs).to(device),
            "values": torch.from_numpy(self.values).to(device),
            "rewards": torch.from_numpy(self.rewards).to(device),
            "dones": torch.from_numpy(self.dones).to(device),
            "hidden_states": torch.from_numpy(self.hidden_states).to(device),
        }
