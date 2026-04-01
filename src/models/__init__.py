"""
Agent architecture: CNN encoder → GRU memory → policy head.

This module ties together the encoder, memory, and policy into a
single agent that can be trained with PPO.
"""

import torch
import torch.nn as nn

from src.models.encoder import CNNEncoder
from src.models.memory import RecurrentMemory
from src.models.policy import NavigationPolicy


class FoveatedNavigationAgent(nn.Module):
    """Complete navigation agent with recurrent memory.

    Pipeline per timestep:
        observation (H, W, 3) → CNN encoder → features (D,)
        features + prev_hidden → GRU → hidden (H,)
        hidden → policy → action, gaze, value

    Args:
        image_size: Observation size.
        encoder_channels: CNN channel list.
        hidden_size: GRU hidden dimension.
        n_actions: Number of discrete movement actions.
        gaze_enabled: Whether agent controls gaze direction.
    """

    def __init__(
        self,
        image_size: int = 64,
        encoder_channels: list[int] = (16, 32, 64),
        hidden_size: int = 256,
        n_actions: int = 7,
        gaze_enabled: bool = False,
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            image_size=image_size,
            channels=encoder_channels,
        )
        self.memory = RecurrentMemory(
            input_dim=self.encoder.feature_dim,
            hidden_size=hidden_size,
        )
        self.policy = NavigationPolicy(
            hidden_size=hidden_size,
            n_actions=n_actions,
            gaze_enabled=gaze_enabled,
        )

        self.hidden_size = hidden_size

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None):
        """Forward pass for a single timestep.

        Args:
            obs: (B, C, H, W) normalised observation.
            hidden: (1, B, hidden_size) previous GRU hidden state.

        Returns:
            action_dist, gaze_dist, value, new_hidden
        """
        features = self.encoder(obs)
        memory_out, new_hidden = self.memory(features, hidden)
        action_dist, gaze_dist, value = self.policy(memory_out)
        return action_dist, gaze_dist, value, new_hidden

    def act(self, obs: torch.Tensor, hidden: torch.Tensor = None, deterministic: bool = False):
        """Sample action for environment interaction.

        Returns:
            action, gaze, log_prob, value, new_hidden
        """
        features = self.encoder(obs)
        memory_out, new_hidden = self.memory(features, hidden)
        action, gaze, log_prob, value = self.policy.act(memory_out, deterministic)
        return action, gaze, log_prob, value, new_hidden

    def get_hidden_state(self, obs: torch.Tensor, hidden: torch.Tensor = None):
        """Get the GRU hidden state (for probing).

        Returns:
            hidden_state: (B, hidden_size) — the cognitive map.
            new_hidden: updated hidden for next step.
        """
        features = self.encoder(obs)
        memory_out, new_hidden = self.memory(features, hidden)
        return memory_out, new_hidden

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
