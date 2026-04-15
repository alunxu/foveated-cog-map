"""
Wijmans-faithful PointNav policy for Habitat DD-PPO.

This policy implements the architecture described in Section A.1 of Wijmans
et al. (ICLR 2023) "Emergence of Maps in the Memories of Blind Navigation
Agents" for the blind PointGoal navigation agent. It also generalizes to
sighted agents (with a visual encoder) and is the base class for the
foveated agent.

Architecture (Section A.1, paraphrased):

  > The agent is parameterized by a 3-layer LSTM with 512-d hidden dimension.
  > At each time step, the agent receives observations:
  >   g       : the location of the goal relative to start
  >   GPS     : its current position relative to start
  >   compass : its current heading relative to start
  >   close   : an indicator close to goal: min(||g - GPS||, 0.5)
  > All 4 inputs are projected to 32-d using separated fully-connected layers.
  > These are concatenated with a learned 32-d embedding of the previous
  > action to form a 160-d input to the LSTM. The output of the LSTM is then
  > processed by a fully-connected layer to produce a softmax distribution
  > over the action space and an estimate of the value function.

For sighted agents, the visual features (after the ResNet encoder + FC to
hidden_size) are concatenated with the 160-d sensor block before the LSTM.

The two Wijmans-specific sensors (``GoalInStartFrameSensor`` and
``CloseToGoalSensor``) are defined in ``src/habitat/wijmans_sensors.py``.
GPS and compass come from Habitat's standard ``EpisodicGPSSensor`` and
``EpisodicCompassSensor``.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn

from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy

# Importing this also registers the sensors with Habitat.
from src.habitat.wijmans_sensors import (
    GoalInStartFrameSensor,
    CloseToGoalSensor,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Logit clamping (prevents NaN crashes in DD-PPO)
# ---------------------------------------------------------------------------

_LOGIT_CLAMP = 10.0


def _wrap_action_distribution_with_clamp(policy):
    """Replace policy.action_distribution.forward to clamp OUTPUT logits.

    CategoricalNet.forward does: logits = self.linear(x) → Categorical(logits).
    We must clamp *logits* (the linear output), not *x* (the LSTM features).
    The previous version clamped x, which still allowed the linear layer to
    produce extreme logits from clamped inputs → NaN in multinomial.
    """
    action_dist = policy.action_distribution
    original_linear = action_dist.linear

    class ClampedLinear(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)

        # Delegate weight/bias access for optimizer
        @property
        def weight(self):
            return self.inner.weight

        @property
        def bias(self):
            return self.inner.bias

    action_dist.linear = ClampedLinear(original_linear)


# ---------------------------------------------------------------------------
# Net
# ---------------------------------------------------------------------------


class WijmansPointNavNet(Net):
    """PointNav net that matches Wijmans et al. (ICLR 2023) Section A.1.

    The four sensor inputs (g, GPS, compass, close) are each projected
    independently to 32-d. They are concatenated with a 32-d previous-action
    embedding to form a 160-d feature, optionally augmented with a 512-d
    visual feature (for sighted agents), and fed to a 3-layer LSTM-512.
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"

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
    ):
        super().__init__()
        self.discrete_actions = discrete_actions
        self._hidden_size = hidden_size
        self._n_prev_action = 32

        # ---- Previous action embedding ----
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = sum(s.shape[0] for s in action_space.spaces.values())
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )

        # ---- Sensor embeddings (Wijmans-style: each input → its own 32-d FC) ----
        sensor_block_dim = self._n_prev_action  # always include prev_action

        # g: goal in start frame (2-D)
        if GoalInStartFrameSensor.cls_uuid in observation_space.spaces:
            self.g_embedding = nn.Linear(2, 32)
            sensor_block_dim += 32
        else:
            self.g_embedding = None

        # GPS: current position in start frame (2-D)
        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            sensor_block_dim += 32
        else:
            self.gps_embedding = None

        # compass: current heading in start frame (1-D scalar; we'll convert
        # to (cos θ, sin θ) before the FC, matching the parent class).
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            self.compass_embedding = nn.Linear(2, 32)
            sensor_block_dim += 32
        else:
            self.compass_embedding = None

        # close-to-goal indicator (1-D)
        if CloseToGoalSensor.cls_uuid in observation_space.spaces:
            self.close_embedding = nn.Linear(1, 32)
            sensor_block_dim += 32
        else:
            self.close_embedding = None

        # ---- Visual encoder (only for sighted agents) ----
        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if len(v.shape) == 3
                }
            )

        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    int(np.prod(self.visual_encoder.output_shape)), hidden_size
                ),
                nn.ReLU(True),
            )
            visual_dim = hidden_size
        else:
            self.visual_fc = None
            visual_dim = 0

        # ---- Recurrent state encoder (3-layer LSTM-512) ----
        rnn_input_size = visual_dim + sensor_block_dim
        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    # ----------- Property API expected by NetPolicy -----------

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    # ----------- Forward -----------

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state: Dict[str, torch.Tensor] = {}

        # Visual features (only for sighted agents).
        if not self.is_blind:
            if (
                self.PRETRAINED_VISUAL_FEATURES_KEY in observations  # noqa: SIM401
            ):
                visual_feats = observations[
                    self.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)
            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # g: goal in start frame
        if self.g_embedding is not None:
            g_obs = observations[GoalInStartFrameSensor.cls_uuid].float()
            x.append(self.g_embedding(g_obs))

        # GPS
        if self.gps_embedding is not None:
            gps_obs = observations[EpisodicGPSSensor.cls_uuid].float()
            x.append(self.gps_embedding(gps_obs))

        # compass: convert scalar angle to (cos, sin) before the FC
        if self.compass_embedding is not None:
            compass_raw = observations[
                EpisodicCompassSensor.cls_uuid
            ]
            compass_cs = torch.stack(
                [torch.cos(compass_raw), torch.sin(compass_raw)],
                dim=-1,
            ).squeeze(dim=-2)  # (B, 2)
            x.append(self.compass_embedding(compass_cs.float()))

        # close-to-goal indicator
        if self.close_embedding is not None:
            close_obs = observations[CloseToGoalSensor.cls_uuid].float()
            x.append(self.close_embedding(close_obs))

        # Previous action embedding
        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )
        x.append(prev_actions)

        # Concatenate and run through the LSTM
        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


@baseline_registry.register_policy(name="WijmansPointNavPolicy")
class WijmansPointNavPolicy(NetPolicy):
    """NetPolicy wrapper around ``WijmansPointNavNet``.

    Drop-in replacement for ``PointNavResNetPolicy`` that uses the
    Wijmans et al. sensor architecture.
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
        **kwargs,
    ):
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
        else:
            discrete_actions = True

        super().__init__(
            WijmansPointNavNet(
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
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

        # Wrap action_distribution to clamp logits and prevent NaN crashes
        # during DD-PPO training (logit overflow → inf → NaN in multinomial).
        _wrap_action_distribution_with_clamp(self)

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Filter out eval-only sensors (like high-res cameras for video).
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
