"""
Foveated PointNav policy with a LEARNED gaze controller.

This is the "learned-gaze" foveated condition. It is paired with the
``FoveatedWijmansPolicy`` (fixed-center) variant to form the gaze ablation
that the project proposal promises:

  - Foveated, fixed-center gaze   -> ``FoveatedWijmansPolicy``
  - Foveated, learned gaze        -> ``FoveatedLearnedGazePolicy`` (this file)
  - Foveated, random gaze         -> (TODO, optional third leg)

How learned gaze works under the rollout-buffer constraint
----------------------------------------------------------
Habitat-baselines processes a whole rollout segment of length ``num_steps``
in one batched forward pass during PPO updates. The observations have shape
``(num_envs * num_steps, ...)``, but the recurrent hidden state is only the
INITIAL state of the segment, with shape
``(num_layers * 2, num_envs, hidden_size)`` (for an LSTM).

A truly per-timestep gaze would require unrolling the LSTM step-by-step,
which is much slower than batched processing. Instead, we use a
"slow-gaze" approximation:

  1. Compute gaze ONCE from the initial hidden state of the segment.
  2. Broadcast that gaze across all timesteps in the segment.
  3. Apply the foveation transform with the broadcast gaze and run the
     visual encoder, then the LSTM, on the whole batch.

This means the gaze updates only at segment boundaries (every 256 steps in
our config), not every step. The gaze decoder still receives gradient from
the PPO loss because gaze affects the foveation, which affects the visual
features, which affect the LSTM output and ultimately the action log-probs
and value estimate. The decoder learns "where to look on average over the
next N steps" rather than per-step gaze, which is a coarser but valid
approximation.

If the learned-gaze agent shows a meaningful gain over the fixed-center
agent, we may revisit this with sequential per-step processing as a
follow-up.
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
from habitat_baselines.rl.ppo import NetPolicy

# NOTE: To re-enable autograd anomaly detection during debugging, uncomment:
#   torch.autograd.set_detect_anomaly(True)
# (significant overhead, do not leave on in production)

from src.habitat.foveated_policy import FoveatedResNetEncoder, FoveatedWijmansNet
from src.habitat.wijmans_policy import _wrap_action_distribution_with_clamp
from src.habitat.wijmans_sensors import (
    GoalInStartFrameSensor,
    CloseToGoalSensor,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class FoveatedLearnedGazeNet(FoveatedWijmansNet):
    """FoveatedWijmansNet variant with a learned gaze decoder.

    Inherits the foveated visual encoder from ``FoveatedWijmansNet`` and adds
    a small MLP that decodes a 2-D gaze position in [0, 1] from the previous
    LSTM hidden state. The same gaze is broadcast across all timesteps in a
    rollout segment (see module docstring for the rationale).
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
        blur_sigma_max: float = 8.0,  # PoC default, see docs/foveation_design.md §2.1
        falloff: str = "quadratic",
        gaze_hidden: int = 64,
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
        )

        # Gaze decoder: hidden state -> 2-D gaze position in [0, 1]
        self.gaze_decoder = nn.Sequential(
            nn.Linear(self._hidden_size, gaze_hidden),
            nn.ReLU(),
            nn.Linear(gaze_hidden, 2),
            nn.Sigmoid(),
        )

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

        if not self.is_blind:
            # Find any visual key to discover the total batch size and device.
            first_visual_key = next(
                k for k, v in observations.items() if v.dim() == 4
            )
            first_visual = observations[first_visual_key]
            total_batch = first_visual.shape[0]

            # ---- Compute gaze from the initial hidden state of this segment ----
            # Habitat-baselines' rollout buffer stores the LSTM hidden state
            # with shape (num_envs, num_recurrent_layers * 2, hidden_size) -
            # env-major, NOT the (num_layers * 2, num_envs, hidden_size) used
            # by vanilla PyTorch LSTM. To get a per-env representation that
            # is robust to the (h0, c0, h1, c1, ...) ordering, we average
            # across the layer/cell dimension.
            # Detach so the gaze decoder doesn't try to backprop through
            # rnn_hidden_states; the LSTM state_encoder modifies it in place
            # later in the forward pass, which would otherwise cause a
            # "variable modified by inplace operation" error during backward.
            # The gaze decoder's own parameters still receive gradient via
            # the forward path: decoder -> gaze -> foveation -> visual feats
            # -> LSTM -> loss.
            rnn_state_input = rnn_hidden_states.detach()
            if rnn_state_input.dim() == 3:
                # shape (num_envs, num_layers * 2, hidden_size)
                prev_h = rnn_state_input.mean(dim=1)  # (num_envs, hidden_size)
            else:
                prev_h = rnn_state_input  # GRU or unusual shape
            num_envs = prev_h.shape[0]

            gaze_per_env = self.gaze_decoder(prev_h)  # (num_envs, 2)

            # During act(), total_batch == num_envs, so no broadcast needed.
            # During evaluate_actions(), total_batch == num_envs * num_steps;
            # we expand the per-env gaze to a per-(env*timestep) tensor.
            #
            # Habitat's rollout buffer is T-major: positions [0, num_envs)
            # are env_0..env_{num_envs-1} at timestep 0, [num_envs, 2*num_envs)
            # are the same envs at timestep 1, etc. The expand-and-reshape
            # below produces the same gaze at every timestep, which matches
            # the "slow-gaze" semantics described in the module docstring.
            if total_batch == num_envs:
                gaze = gaze_per_env
            else:
                num_steps = total_batch // num_envs
                assert total_batch == num_steps * num_envs, (
                    f"Cannot broadcast gaze: total_batch={total_batch} is not "
                    f"divisible by num_envs={num_envs}"
                )
                # Use .repeat() (which copies) rather than .expand().reshape()
                # (which creates a view). A view here can confuse autograd
                # because the foveation transform's gather/scatter operations
                # may share storage with the broadcast gaze tensor.
                # Tile in dim 0: num_steps copies of gaze_per_env stacked.
                # Result is T-major: positions [0:num_envs] = gaze_per_env,
                # positions [num_envs:2*num_envs] = same gaze_per_env, etc.
                gaze = gaze_per_env.repeat(num_steps, 1)

            aux_loss_state["gaze"] = gaze

            # ---- Visual features (with foveation at the broadcast gaze) ----
            visual_feats = self.visual_encoder(observations, gaze=gaze)
            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # ---- Wijmans sensor block (g, GPS, compass, close-to-goal) ----
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

        # ---- Previous-action embedding ----
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

        # ---- LSTM ----
        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state


@baseline_registry.register_policy(name="FoveatedLearnedGazePolicy")
class FoveatedLearnedGazePolicy(NetPolicy):
    """Wijmans-faithful PointNav policy with foveated vision and learned gaze.

    Drop-in alternative to ``FoveatedWijmansPolicy`` (fixed-center gaze).
    Use the policy name ``FoveatedLearnedGazePolicy`` in the YAML config.
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
        blur_sigma_max: float = 8.0,  # PoC default, see docs/foveation_design.md §2.1
        falloff: str = "quadratic",
        gaze_hidden: int = 64,
        **kwargs,
    ):
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
        else:
            discrete_actions = True

        super().__init__(
            FoveatedLearnedGazeNet(
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

        # Foveation hyperparameters are hardcoded in __init__ defaults; Hydra's
        # structured config validation rejects unknown fields on PolicyConfig.
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
