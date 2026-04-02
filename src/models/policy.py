"""
Policy head — action and gaze direction.

Takes GRU hidden state and outputs:
  - Movement action (discrete: left, right, forward, toggle, etc.)
  - Gaze direction (continuous: x, y in [0, 1]) — foveated agent only
  - Value estimate (for PPO)

Member B is responsible for this module.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class NavigationPolicy(nn.Module):
    """Actor-critic policy for the navigation agent.

    Args:
        hidden_size: Input dimension from GRU memory.
        n_actions: Number of discrete movement actions.
        action_hidden: Hidden dim for action MLP.
        gaze_hidden: Hidden dim for gaze MLP (0 to disable gaze).
        gaze_enabled: Whether the agent controls gaze direction.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        n_actions: int = 7,  # MiniGrid default: left, right, forward, pickup, drop, toggle, done
        action_hidden: int = 64,
        gaze_hidden: int = 64,
        gaze_enabled: bool = False,
    ):
        super().__init__()
        self.gaze_enabled = gaze_enabled

        # Actor: movement actions (discrete)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, n_actions),
        )

        # Actor: gaze direction (continuous, 2D)
        if gaze_enabled:
            self.gaze_head = nn.Sequential(
                nn.Linear(hidden_size, gaze_hidden),
                nn.ReLU(),
            )
            self.gaze_mean = nn.Linear(gaze_hidden, 2)
            self.gaze_log_std = nn.Parameter(torch.full((2,), -1.0))  # σ ≈ 0.37

        # Critic: value estimate
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, 1),
        )

    def forward(self, hidden_state: torch.Tensor):
        """Compute action distribution and value.

        Args:
            hidden_state: (B, hidden_size) from GRU.

        Returns:
            action_dist: Categorical distribution over movement actions.
            gaze_dist: Normal distribution over gaze (x, y), or None.
            value: (B, 1) value estimate.
        """
        # Movement action
        action_logits = self.action_head(hidden_state)
        action_dist = Categorical(logits=action_logits)

        # Gaze direction
        gaze_dist = None
        if self.gaze_enabled:
            gaze_features = self.gaze_head(hidden_state)
            gaze_mean = torch.sigmoid(self.gaze_mean(gaze_features))  # [0, 1]
            gaze_std = self.gaze_log_std.clamp(-3.0, 0.0).exp().expand_as(gaze_mean)  # σ ∈ [0.05, 1.0]
            gaze_dist = Normal(gaze_mean, gaze_std)

        # Value
        value = self.value_head(hidden_state)

        return action_dist, gaze_dist, value

    def act(self, hidden_state: torch.Tensor, deterministic: bool = False):
        """Sample an action.

        Returns:
            action: int movement action.
            gaze: (2,) gaze position or None.
            log_prob: log probability of the action(s).
            value: value estimate.
        """
        action_dist, gaze_dist, value = self(hidden_state)

        if deterministic:
            action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)

        gaze = None
        if gaze_dist is not None:
            if deterministic:
                gaze = gaze_dist.mean
            else:
                gaze = gaze_dist.sample()
            gaze = gaze.clamp(0, 1)
            log_prob = log_prob + gaze_dist.log_prob(gaze).sum(dim=-1)

        return action, gaze, log_prob, value.squeeze(-1)
