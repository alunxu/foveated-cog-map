"""
PPO trainer for the navigation agent.

Handles advantage estimation and policy updates.
"""

import torch
import torch.nn as nn
import numpy as np
from loguru import logger


class PPOTrainer:
    """Proximal Policy Optimization trainer.

    Args:
        agent: NavigationAgent instance.
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

    def update(self, rollout_data: dict, next_value: torch.Tensor) -> dict:
        """Run PPO update on collected rollout data.

        Args:
            rollout_data: dict of tensors from RolloutBuffer.get_as_tensors().
            next_value: (N,) bootstrapped value for the last step.

        Returns:
            metrics: dict of training metrics.
        """
        rewards = rollout_data["rewards"]      # (T, N)
        values = rollout_data["values"]        # (T, N)
        dones = rollout_data["dones"]          # (T, N)
        old_log_probs = rollout_data["log_probs"]  # (T, N)
        observations = rollout_data["observations"]  # (T, N, ...)
        actions = rollout_data["actions"]      # (T, N)
        prev_actions = rollout_data["prev_actions"]  # (T, N)
        hidden_states = rollout_data["hidden_states"]  # (T, num_layers, N, H)

        pointgoals = rollout_data.get("pointgoals")  # (T, N, 4) or None
        gaze_actions = rollout_data.get("gaze_actions")  # (T, N, 2) or None

        T, N = rewards.shape

        # 1. Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # 2. Flatten (T, N, ...) -> (T*N, ...)
        def flatten(x):
            return x.reshape(T * N, *x.shape[2:])

        flat_obs = flatten(observations)
        flat_actions = flatten(actions)
        flat_prev_actions = flatten(prev_actions)
        flat_old_log_probs = flatten(old_log_probs)
        flat_advantages = flatten(advantages)
        flat_returns = flatten(returns)
        # Hidden states: (T, num_layers, N, H) -> need (T*N, num_layers, H)
        # Permute to (T, N, num_layers, H) then flatten
        flat_hidden = hidden_states.permute(0, 2, 1, 3).reshape(T * N, hidden_states.shape[1], -1)
        # -> (num_layers, T*N, H) for GRU
        flat_hidden = flat_hidden.permute(1, 0, 2).contiguous()

        flat_pointgoals = flatten(pointgoals) if pointgoals is not None else None
        flat_gaze_actions = flatten(gaze_actions) if gaze_actions is not None else None

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # Preprocess observations for CNN
        if self.agent.encoder_type == "cnn":
            flat_obs = flat_obs.float().permute(0, 3, 1, 2) / 255.0

        total_samples = T * N
        all_metrics = []

        # 3. PPO epochs
        for epoch in range(self.n_epochs):
            indices = torch.randperm(total_samples, device=flat_obs.device)

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                mb_idx = indices[start:end]

                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_prev_actions = flat_prev_actions[mb_idx]
                mb_old_log_probs = flat_old_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_hidden = flat_hidden[:, mb_idx, :]
                mb_pointgoals = flat_pointgoals[mb_idx] if flat_pointgoals is not None else None
                mb_gaze = flat_gaze_actions[mb_idx] if flat_gaze_actions is not None else None

                # Re-evaluate actions under current policy
                new_log_probs, new_values, entropy = self.agent.evaluate_actions(
                    mb_obs, mb_actions, mb_hidden,
                    pointgoal=mb_pointgoals,
                    prev_action=mb_prev_actions,
                    gaze_actions=mb_gaze,
                )

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()

                all_metrics.append({
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": -entropy_loss.item(),
                    "approx_kl": approx_kl,
                })

        # Average metrics across all minibatches
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics
