"""
Agent architecture: encoder -> GRU memory -> policy head.

Supports both visual agents (CNN encoder) and blind agents (vector encoder).
For visual agents, the pointgoal vector is projected and concatenated with
CNN features before the GRU.
"""

import torch
import torch.nn as nn

from src.models.encoder import CNNEncoder, VectorEncoder
from src.models.memory import RecurrentMemory
from src.models.policy import NavigationPolicy


class NavigationAgent(nn.Module):
    """Complete navigation agent with recurrent memory.

    Pipeline per timestep (visual):
        observation (B, C, H, W) -> CNN -> features (B, D_cnn)
        pointgoal (B, 4) -> Linear -> goal_feat (B, 32)
        [features; goal_feat] -> GRU -> hidden (B, H)
        hidden -> policy -> action, gaze, value

    Pipeline per timestep (blind):
        pointgoal (B, 4) + prev_action -> VectorEncoder -> features (B, D_vec)
        features -> GRU -> hidden (B, H)
        hidden -> policy -> action, gaze, value

    Args:
        encoder_type: 'cnn' for visual agents, 'vector' for blind.
        image_size: Observation size (only for CNN).
        encoder_channels: CNN channel list (only for CNN).
        hidden_size: GRU hidden dimension.
        num_memory_layers: Number of GRU layers.
        n_actions: Number of discrete movement actions.
        gaze_enabled: Whether agent controls gaze direction.
        pointgoal_dim: Dimension of pointgoal vector (0 to disable for visual).
    """

    def __init__(
        self,
        encoder_type: str = "cnn",
        image_size: int = 64,
        encoder_channels: list[int] = (16, 32, 64),
        hidden_size: int = 256,
        num_memory_layers: int = 1,
        n_actions: int = 4,
        gaze_enabled: bool = False,
        pointgoal_dim: int = 4,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_size = hidden_size
        self.pointgoal_dim = pointgoal_dim

        if encoder_type == "vector":
            self.encoder = VectorEncoder(
                input_dim=pointgoal_dim,
                n_actions=n_actions,
            )
            self.visual_projection = None
            self.goal_projections = None
            memory_input_dim = self.encoder.feature_dim
        else:
            self.encoder = CNNEncoder(
                image_size=image_size,
                channels=encoder_channels,
            )
            # Compress CNN features before GRU to avoid bottleneck
            # (raw CNN output is 4096-d, far too large for a 256/512-d GRU)
            cnn_out_dim = self.encoder.feature_dim
            compressed_dim = hidden_size  # match GRU width
            self.visual_projection = nn.Sequential(
                nn.Linear(cnn_out_dim, compressed_dim),
                nn.ReLU(),
            )
            # Visual agents also receive pointgoal via per-component projections
            # (same pattern as VectorEncoder — gives pointgoal proper representation)
            if pointgoal_dim > 0:
                goal_proj_dim = 32  # per-component projection dim
                self.goal_projections = nn.ModuleList([
                    nn.Sequential(nn.Linear(1, goal_proj_dim), nn.ReLU())
                    for _ in range(pointgoal_dim)
                ])
                goal_feat_dim = pointgoal_dim * goal_proj_dim  # 4 * 32 = 128
                memory_input_dim = compressed_dim + goal_feat_dim
            else:
                self.goal_projections = None
                memory_input_dim = compressed_dim

        self.memory = RecurrentMemory(
            input_dim=memory_input_dim,
            hidden_size=hidden_size,
            num_layers=num_memory_layers,
        )
        self.policy = NavigationPolicy(
            hidden_size=hidden_size,
            n_actions=n_actions,
            gaze_enabled=gaze_enabled,
        )

    def _encode(self, obs, pointgoal=None, prev_action=None):
        """Encode observation into feature vector for the GRU."""
        if self.encoder_type == "vector":
            return self.encoder(obs, prev_action)
        else:
            features = self.encoder(obs)
            if self.visual_projection is not None:
                features = self.visual_projection(features)
            if self.goal_projections is not None and pointgoal is not None:
                goal_parts = [
                    proj(pointgoal[:, i:i+1])
                    for i, proj in enumerate(self.goal_projections)
                ]
                goal_feat = torch.cat(goal_parts, dim=-1)  # (B, 128)
                features = torch.cat([features, goal_feat], dim=-1)
            return features

    def forward(self, obs, hidden=None, pointgoal=None, prev_action=None):
        """Forward pass for a single timestep.

        Returns:
            action_dist, gaze_dist, value, new_hidden
        """
        features = self._encode(obs, pointgoal, prev_action)
        memory_out, new_hidden = self.memory(features, hidden)
        action_dist, gaze_dist, value = self.policy(memory_out)
        return action_dist, gaze_dist, value, new_hidden

    def act(self, obs, hidden=None, pointgoal=None, prev_action=None,
            deterministic=False):
        """Sample action for environment interaction.

        Returns:
            action, gaze, log_prob, value, new_hidden
        """
        features = self._encode(obs, pointgoal, prev_action)
        memory_out, new_hidden = self.memory(features, hidden)
        action, gaze, log_prob, value = self.policy.act(memory_out, deterministic)
        return action, gaze, log_prob, value, new_hidden

    def evaluate_actions(self, obs, actions, hidden, pointgoal=None,
                         prev_action=None, gaze_actions=None):
        """Re-evaluate actions for PPO update.

        Args:
            obs: (B, ...) observations.
            actions: (B,) movement actions taken.
            hidden: (num_layers, B, hidden_size) stored hidden states.
            pointgoal: (B, 4) or None.
            prev_action: (B,) or None.
            gaze_actions: (B, 2) or None.

        Returns:
            log_prob, value, entropy
        """
        features = self._encode(obs, pointgoal, prev_action)
        memory_out, _ = self.memory(features, hidden)
        action_dist, gaze_dist, value = self.policy(memory_out)

        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        if gaze_dist is not None and gaze_actions is not None:
            log_prob = log_prob + gaze_dist.log_prob(gaze_actions).sum(dim=-1)
            entropy = entropy + gaze_dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy

    def get_value(self, obs, hidden=None, pointgoal=None, prev_action=None):
        """Get value estimate only (for GAE bootstrap)."""
        features = self._encode(obs, pointgoal, prev_action)
        memory_out, _ = self.memory(features, hidden)
        _, _, value = self.policy(memory_out)
        return value.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Keep backward compat alias
FoveatedNavigationAgent = NavigationAgent
