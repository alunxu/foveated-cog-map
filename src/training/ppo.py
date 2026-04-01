"""
PPO trainer for the navigation agent.

Handles the training loop, advantage estimation, and policy updates.
Stores hidden states alongside rollout data for later probing.

Member B is responsible for this module.
"""

import torch
import torch.nn as nn
import numpy as np
from loguru import logger


class PPOTrainer:
    """Proximal Policy Optimization trainer.

    Args:
        agent: FoveatedNavigationAgent instance.
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping norm.
        n_epochs: PPO update epochs per rollout.
        batch_size: Minibatch size for PPO updates.
    """

    def __init__(
        self,
        agent,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 256,
    ):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: (T, N) rewards.
            values: (T, N) value estimates.
            dones: (T, N) episode termination flags.
            next_value: (N,) value estimate at T+1.

        Returns:
            advantages: (T, N) GAE advantages.
            returns: (T, N) discounted returns.
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout_buffer: dict) -> dict:
        """Run PPO update on collected rollout data.

        Args:
            rollout_buffer: dict containing rollout data.
                Required keys: observations, actions, log_probs, values,
                rewards, dones, hidden_states.
                Optional: gaze_actions.

        Returns:
            metrics: dict of training metrics for logging.
        """
        # TODO: Implement PPO update loop
        # 1. Compute GAE advantages
        # 2. Flatten rollout data
        # 3. For n_epochs:
        #    a. Shuffle and create minibatches
        #    b. Recompute log_probs and values with current policy
        #    c. Compute clipped policy loss
        #    d. Compute value loss
        #    e. Compute entropy bonus
        #    f. Backprop and clip gradients
        
        raise NotImplementedError(
            "Implement PPO update. This is Member B's primary responsibility. "
            "See stable-baselines3 PPO for reference implementation."
        )
