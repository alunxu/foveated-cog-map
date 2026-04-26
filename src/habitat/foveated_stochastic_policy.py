"""
Foveated PointNav policy with a STOCHASTIC learned gaze controller.

This is the gaze-collapse-fix variant of ``FoveatedLearnedGazePolicy``.
Instead of a deterministic sigmoid MLP that the navigation loss alone
cannot keep diverse (collapses to a fixed point at (0.49, 0.62) under
pure task supervision), the gaze decoder outputs a *bounded Gaussian
distribution* from which the actual gaze position is sampled.

The architecture differences vs. ``FoveatedLearnedGazePolicy``:

  - Decoder output dim: 2 (μ alone)  →  4 (μ_x_logit, μ_y_logit, σ_x_raw, σ_y_raw)
  - μ = sigmoid(μ_logit) ∈ [0, 1]
  - σ = σ_min + (σ_max - σ_min) · sigmoid(σ_raw)   bounded in [0.05, 0.30]
  - Train: gaze = (μ + σ · ε).clamp(0.05, 0.95)    where  ε ~ N(0, 1)
  - Eval:  gaze = μ   (deterministic; cleaner for probing / analysis)

Reparameterization (gaze = μ + σ · ε) lets gradient flow through the
sample, so PPO's task loss reaches both μ and σ. The bounded σ replaces
the auxiliary "gaze_diversity" hinge loss: σ cannot collapse below 0.05
(persistent ~5% of image-range exploration) and cannot exceed 0.30
(prevents random gaze that would destroy useful foveation).

Why bounded-σ instead of entropy-bonus aux loss
-----------------------------------------------
The existing ``GazeDiversityLoss`` hinge-penalises minibatch variance
falling below 0.01. In the deterministic policy, two stable gaze points
across envs can give minibatch variance > 0.01 while each env's gaze is
still essentially constant — the hinge thinks the job is done. Bounded
σ prevents this by enforcing per-env stochasticity directly, not via a
batch-level statistic.

Same slow-gaze approximation as the deterministic learned-gaze policy:
one gaze sample per env per rollout segment (broadcast across timesteps
within the segment). See ``foveated_learned_policy.py`` for rationale.

Hyperparameters
---------------
The σ range and clamp window are exposed via the policy ``__init__``:
  - ``sigma_min`` (default 0.05): minimum exploration std
  - ``sigma_max`` (default 0.30): maximum exploration std
  - ``gaze_clip_lo`` / ``gaze_clip_hi`` (default 0.05 / 0.95): clamp
    window for the sampled gaze (avoid sigmoid extremes that produce
    no-spatial-variation foveation)
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from gym import spaces
from torch import nn

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import NetPolicy

from src.habitat.foveated_learned_policy import FoveatedLearnedGazeNet
from src.habitat.wijmans_policy import _wrap_action_distribution_with_clamp

if TYPE_CHECKING:
    from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Net: stochastic-gaze extension of FoveatedLearnedGazeNet
# ---------------------------------------------------------------------------


class FoveatedStochasticGazeNet(FoveatedLearnedGazeNet):
    """``FoveatedLearnedGazeNet`` with reparameterized-Gaussian gaze sampling.

    Replaces the parent's deterministic ``gaze_decoder`` (2-dim sigmoid
    output for ``(x, y)``) with a 4-dim head that parameterises a bounded
    Gaussian over gaze position. See module docstring for the rationale.

    The forward pass replicates the parent's logic for everything except
    the gaze computation block; we inherit the foveated visual encoder,
    the Wijmans sensor stack, the action embedding, and the LSTM.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone: str,
        resnet_baseplanes: int,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        fovea_radius: int = 16,
        blur_sigma_max: float = 8.0,
        falloff: str = "quadratic",
        gaze_hidden: int = 64,
        # Stochastic-gaze hyperparameters
        sigma_min: float = 0.05,
        sigma_max: float = 0.30,
        gaze_clip_lo: float = 0.05,
        gaze_clip_hi: float = 0.95,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            normalize_visual_inputs=normalize_visual_inputs,
            force_blind_policy=force_blind_policy,
            discrete_actions=discrete_actions,
            fovea_radius=fovea_radius,
            blur_sigma_max=blur_sigma_max,
            falloff=falloff,
            gaze_hidden=gaze_hidden,
        )

        # Replace the parent's deterministic 2-dim sigmoid decoder with a
        # 4-dim head: (μ_x_logit, μ_y_logit, σ_x_raw, σ_y_raw).  We
        # do NOT apply sigmoid here — μ uses sigmoid in forward, σ uses a
        # bounded sigmoid transform (see below).
        self.gaze_decoder = nn.Sequential(
            nn.Linear(self._hidden_size, gaze_hidden),
            nn.ReLU(),
            nn.Linear(gaze_hidden, 4),
        )

        # σ-bounding parameters (registered as buffers so they save/load).
        self.register_buffer(
            "_sigma_min", torch.tensor(float(sigma_min))
        )
        self.register_buffer(
            "_sigma_max", torch.tensor(float(sigma_max))
        )
        self.register_buffer(
            "_gaze_clip_lo", torch.tensor(float(gaze_clip_lo))
        )
        self.register_buffer(
            "_gaze_clip_hi", torch.tensor(float(gaze_clip_hi))
        )

    def _decode_gaze(self, prev_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (μ, σ) for the gaze distribution given a per-env hidden state.

        Args:
            prev_h: (num_envs, hidden_size)

        Returns:
            mu:    (num_envs, 2) in [0, 1]
            sigma: (num_envs, 2) in [σ_min, σ_max]
        """
        params = self.gaze_decoder(prev_h)  # (num_envs, 4)
        mu_logit = params[..., :2]
        sigma_raw = params[..., 2:]

        mu = torch.sigmoid(mu_logit)
        # σ = σ_min + (σ_max − σ_min) · sigmoid(σ_raw)
        sigma_span = self._sigma_max - self._sigma_min
        sigma = self._sigma_min + sigma_span * torch.sigmoid(sigma_raw)

        return mu, sigma

    def _sample_gaze(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterized Gaussian sample, clamped to a safe range.

        Args:
            mu:    (num_envs, 2)
            sigma: (num_envs, 2)

        Returns:
            gaze: (num_envs, 2) in [gaze_clip_lo, gaze_clip_hi]
        """
        if self.training:
            eps = torch.randn_like(mu)
            gaze = mu + sigma * eps
            gaze = gaze.clamp(self._gaze_clip_lo, self._gaze_clip_hi)
        else:
            # Deterministic at eval (cleaner for downstream probing /
            # analysis pipelines).  Equivalent to setting ε=0.
            gaze = mu

        return gaze

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # The vast majority of this method mirrors
        # ``FoveatedLearnedGazeNet.forward`` exactly; the only differences
        # are in the gaze block (we sample from a bounded Gaussian rather
        # than apply a deterministic sigmoid). Duplicating the body here
        # avoids monkey-patching the parent and keeps the diff explicit.
        x = []
        aux_loss_state: Dict[str, torch.Tensor] = {}

        if not self.is_blind:
            # ---- batch-shape discovery (same as parent) --------------------
            first_visual_key = next(
                k for k, v in observations.items() if v.dim() == 4
            )
            first_visual = observations[first_visual_key]
            total_batch = first_visual.shape[0]

            # ---- per-env μ, σ from initial hidden state -------------------
            rnn_state_input = rnn_hidden_states.detach()
            if rnn_state_input.dim() == 3:
                prev_h = rnn_state_input.mean(dim=1)  # (num_envs, hidden_size)
            else:
                prev_h = rnn_state_input  # GRU or unusual shape
            num_envs = prev_h.shape[0]

            mu_per_env, sigma_per_env = self._decode_gaze(prev_h)

            # ---- sample gaze (reparameterized, train) / use μ (eval) -------
            gaze_per_env = self._sample_gaze(mu_per_env, sigma_per_env)

            # ---- broadcast across timesteps in the segment ---------------
            if total_batch == num_envs:
                gaze = gaze_per_env
            else:
                num_steps = total_batch // num_envs
                assert total_batch == num_steps * num_envs, (
                    f"Cannot broadcast gaze: total_batch={total_batch} is not "
                    f"divisible by num_envs={num_envs}"
                )
                # T-major repeat (same convention as parent).
                gaze = gaze_per_env.repeat(num_steps, 1)

            # Expose gaze + distribution params for analysis / aux losses.
            aux_loss_state["gaze"] = gaze
            aux_loss_state["gaze_mu"] = mu_per_env
            aux_loss_state["gaze_sigma"] = sigma_per_env

            # ---- visual features (with foveation at sampled gaze) --------
            visual_feats = self.visual_encoder(observations, gaze=gaze)
            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # ---- Wijmans sensor block + previous-action embedding + LSTM -----
        # Identical to parent. Importing here keeps the module dependency
        # surface explicit; the parent's namespace already has these symbols.
        from habitat.tasks.nav.nav import (
            EpisodicCompassSensor,
            EpisodicGPSSensor,
        )

        from src.habitat.wijmans_sensors import (
            GoalInStartFrameSensor,
            CloseToGoalSensor,
        )

        if self.g_embedding is not None:
            x.append(
                self.g_embedding(
                    observations[GoalInStartFrameSensor.cls_uuid].float()
                )
            )

        if self.gps_embedding is not None:
            x.append(
                self.gps_embedding(
                    observations[EpisodicGPSSensor.cls_uuid].float()
                )
            )

        if self.compass_embedding is not None:
            compass_raw = observations[EpisodicCompassSensor.cls_uuid]
            compass_cs = torch.stack(
                [torch.cos(compass_raw), torch.sin(compass_raw)], dim=-1
            ).squeeze(dim=-2)
            x.append(self.compass_embedding(compass_cs.float()))

        if self.close_embedding is not None:
            x.append(
                self.close_embedding(
                    observations[CloseToGoalSensor.cls_uuid].float()
                )
            )

        if self.discrete_actions:
            pa = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(pa)
            prev_action_embed = self.prev_action_embedding(
                torch.where(masks.view(-1), pa + 1, start_token)
            )
        else:
            prev_action_embed = self.prev_action_embedding(
                masks * prev_actions.float()
            )
        x.append(prev_action_embed)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


@baseline_registry.register_policy(name="FoveatedStochasticGazePolicy")
class FoveatedStochasticGazePolicy(NetPolicy):
    """Foveated PointNav policy with reparameterized-Gaussian gaze sampling.

    Drop-in alternative to ``FoveatedLearnedGazePolicy`` (deterministic
    learned gaze). Uses bounded σ ∈ [0.05, 0.30] in the gaze sample
    distribution to prevent collapse without an auxiliary diversity
    regulariser. See module docstring for the design rationale.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 3,
        rnn_type: str = "LSTM",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        fovea_radius: int = 16,
        blur_sigma_max: float = 8.0,
        falloff: str = "quadratic",
        gaze_hidden: int = 64,
        sigma_min: float = 0.05,
        sigma_max: float = 0.30,
        gaze_clip_lo: float = 0.05,
        gaze_clip_hi: float = 0.95,
        **kwargs,
    ):
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
        else:
            discrete_actions = True

        super().__init__(
            FoveatedStochasticGazeNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                fovea_radius=fovea_radius,
                blur_sigma_max=blur_sigma_max,
                falloff=falloff,
                gaze_hidden=gaze_hidden,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                gaze_clip_lo=gaze_clip_lo,
                gaze_clip_hi=gaze_clip_hi,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

        _wrap_action_distribution_with_clamp(self)

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = kwargs.get("agent_name")
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            agent_name = config.habitat.simulator.agents_order[0]

        # Stochastic-gaze hyperparameters are hardcoded in __init__ defaults
        # rather than read from the policy config: Hydra's structured
        # config validation rejects unknown fields on PolicyConfig.  Edit
        # the FoveatedStochasticGazePolicy.__init__ defaults to tune them.
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
        )
