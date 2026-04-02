"""
Rollout buffer — stores trajectories for PPO and probing.

Stores GRU hidden states at each timestep for later probing analysis.
"""

import torch
import numpy as np


class RolloutBuffer:
    """Stores rollout data for PPO training and probing analysis.

    Args:
        n_steps: Number of steps per rollout.
        n_envs: Number of parallel environments.
        obs_shape: Shape of a single observation.
        hidden_size: GRU hidden state dimension.
        num_memory_layers: Number of GRU layers.
        pointgoal_dim: Dimension of pointgoal vector (0 to disable).
        has_gaze: Whether to store gaze actions.
        obs_dtype: Numpy dtype for observations.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple,
        hidden_size: int,
        num_memory_layers: int = 1,
        pointgoal_dim: int = 4,
        has_gaze: bool = False,
        obs_dtype=np.uint8,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.has_gaze = has_gaze
        self.pointgoal_dim = pointgoal_dim
        self.num_memory_layers = num_memory_layers
        self.pos = 0

        # Core PPO data
        self.observations = np.zeros((n_steps, n_envs, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.prev_actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Hidden states for probing and PPO re-evaluation
        # Shape: (n_steps, num_layers, n_envs, hidden_size)
        self.hidden_states = np.zeros(
            (n_steps, num_memory_layers, n_envs, hidden_size), dtype=np.float32
        )

        # Pointgoal
        if pointgoal_dim > 0:
            self.pointgoals = np.zeros((n_steps, n_envs, pointgoal_dim), dtype=np.float32)

        # Gaze actions (for foveated agent PPO recomputation)
        if has_gaze:
            self.gaze_actions = np.zeros((n_steps, n_envs, 2), dtype=np.float32)

        # Ground truth for probing
        self.agent_positions = np.zeros((n_steps, n_envs, 2), dtype=np.float32)
        self.agent_directions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.collisions = np.zeros((n_steps, n_envs), dtype=np.float32)

    def add(
        self,
        obs, action, prev_action, log_prob, value, reward, done,
        hidden_state, pointgoal=None, gaze_action=None,
        agent_pos=None, agent_dir=None, collision=None,
    ):
        """Add one timestep of data."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.prev_actions[self.pos] = prev_action
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.hidden_states[self.pos] = hidden_state  # (num_layers, n_envs, hidden_size)

        if pointgoal is not None and self.pointgoal_dim > 0:
            self.pointgoals[self.pos] = pointgoal

        if gaze_action is not None and self.has_gaze:
            self.gaze_actions[self.pos] = gaze_action

        if agent_pos is not None:
            self.agent_positions[self.pos] = agent_pos
        if agent_dir is not None:
            self.agent_directions[self.pos] = agent_dir
        if collision is not None:
            self.collisions[self.pos] = collision

        self.pos += 1

    def reset(self):
        """Reset buffer position for new rollout."""
        self.pos = 0

    def get_as_tensors(self, device: torch.device) -> dict:
        """Convert buffer to torch tensors for PPO update."""
        data = {
            "observations": torch.from_numpy(self.observations).to(device),
            "actions": torch.from_numpy(self.actions).to(device),
            "prev_actions": torch.from_numpy(self.prev_actions).to(device),
            "log_probs": torch.from_numpy(self.log_probs).to(device),
            "values": torch.from_numpy(self.values).to(device),
            "rewards": torch.from_numpy(self.rewards).to(device),
            "dones": torch.from_numpy(self.dones).to(device),
            "hidden_states": torch.from_numpy(self.hidden_states).to(device),
        }
        if self.pointgoal_dim > 0:
            data["pointgoals"] = torch.from_numpy(self.pointgoals).to(device)
        if self.has_gaze:
            data["gaze_actions"] = torch.from_numpy(self.gaze_actions).to(device)
        return data
