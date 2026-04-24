"""
Foveated PointNav policy for Habitat DD-PPO.

Combines the Wijmans-faithful sensor stack (g, GPS, compass, close-to-goal
indicator) with foveated visual perception and a learned gaze controller.

Key design:
  - The agent receives the same Wijmans sensors as the blind / uniform /
    matched conditions, so the only varied factor across the four conditions
    is the structure of the visual input.
  - Foveation is applied to RGB observations BEFORE the ResNet encoder.
  - Gaze direction is decoded from the PREVIOUS LSTM hidden state by a small
    MLP, so the agent "decides where to look" based on its memory, then
    processes the foveated view.
  - Gaze is deterministic given the hidden state (no separate action / entropy
    bonus). Gradients flow end-to-end through the foveation transform:
    navigation loss → LSTM → gaze decoder → foveation → visual features.

Member C is responsible for this module.
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
from habitat_baselines.rl.ppo import NetPolicy

from src.habitat.torch_foveation import TorchFoveationTransform
from src.habitat.wijmans_policy import WijmansPointNavNet, _wrap_action_distribution_with_clamp
from src.habitat.wijmans_sensors import (
    GoalInStartFrameSensor,
    CloseToGoalSensor,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Foveated visual encoder
# ---------------------------------------------------------------------------


class FoveatedResNetEncoder(ResNetEncoder):
    """ResNet encoder with foveation applied before the backbone.

    Wraps the standard ResNetEncoder by applying a spatially-varying Gaussian
    blur (controlled by a gaze position) to the input image. The output shape
    is identical to the parent ResNetEncoder, so downstream layers
    (visual_fc, etc.) need no changes.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
        fovea_radius: int = 16,
        blur_sigma_max: float = 8.0,  # PoC default, see docs/foveation_design.md §2.1
        falloff: str = "quadratic",
    ):
        super().__init__(
            observation_space=observation_space,
            baseplanes=baseplanes,
            ngroups=ngroups,
            spatial_size=spatial_size,
            make_backbone=make_backbone,
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.is_blind:
            first_key = self.visual_keys[0]
            img_h = observation_space.spaces[first_key].shape[0]
            # Foveation operates after the avg_pool2d that halves the input.
            self.foveation = TorchFoveationTransform(
                image_size=img_h // 2,
                fovea_radius=max(fovea_radius // 2, 1),
                blur_sigma_max=blur_sigma_max,
                falloff=falloff,
            )
        else:
            self.foveation = None

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        gaze: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.is_blind:
            return None

        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k].permute(0, 3, 1, 2)
            if self.key_needs_rescaling[k] is not None:
                obs_k = obs_k.float() * self.key_needs_rescaling[k]
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        x = torch.nn.functional.avg_pool2d(x, 2)

        # Apply foveation BEFORE the running mean/var and backbone.
        if self.foveation is not None and gaze is not None:
            x = self.foveation(x, gaze)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


# ---------------------------------------------------------------------------
# Net: foveated extension of the Wijmans-faithful net
# ---------------------------------------------------------------------------


class FoveatedShiftedGazeNet(WijmansPointNavNet):
    """Wijmans-faithful PointNav net with foveated vision at a hardcoded shifted gaze.

    Inherits the Wijmans sensor stack from ``WijmansPointNavNet`` and swaps
    the standard ResNet encoder for ``FoveatedResNetEncoder``.  The gaze
    position is *not* learned; it is fixed at ``(0.49, 0.62)`` (normalised
    image coordinates) to match the collapsed-gaze position of the
    foveated-learned agent.  There is no gaze MLP in this policy.

    The ``gaze_hidden`` constructor argument is accepted for API
    compatibility but is unused.
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
        gaze_hidden: int = 64,  # accepted for API compatibility; unused (no gaze MLP)
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
        )

        # Replace the standard visual encoder with the foveated one.
        if not self.is_blind:
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

            # NOTE: we disable normalize_visual_inputs (RunningMeanAndVar) for
            # the foveated encoder. The Habitat-baselines RunningMeanAndVar
            # module mutates a buffer (self._count) in place during forward,
            # which conflicts with autograd when there is an additional
            # gradient-producing path through the gaze decoder. Replacing it
            # with an identity module (the default when normalize_visual_inputs
            # is False) avoids the in-place version conflict. The standard
            # sighted agents are unaffected by this bug because they have only
            # one gradient path through the visual encoder.
            self.visual_encoder = FoveatedResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=False,
                fovea_radius=fovea_radius,
                blur_sigma_max=blur_sigma_max,
                falloff=falloff,
            )

            # Re-create visual_fc to match the (possibly different) output shape.
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        int(np.prod(self.visual_encoder.output_shape)),
                        hidden_size,
                    ),
                    nn.ReLU(True),
                )

        # NOTE: gaze_decoder is currently disabled because it has shape
        # mismatch issues between act() and evaluate_actions(). We use a
        # fixed center gaze for now (the "fixed-center" ablation). When
        # learned gaze is added back, the decoder should also output gaze
        # for each timestep stored in the rollout buffer.
        # self.gaze_decoder = nn.Sequential(
        #     nn.Linear(hidden_size, gaze_hidden),
        #     nn.ReLU(),
        #     nn.Linear(gaze_hidden, 2),
        #     nn.Sigmoid(),
        # )

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

        # ---- Visual features (with foveation at fixed shifted gaze) ----
        # This is the foveated-shifted causal control: gaze is hardcoded at
        # (0.49, 0.62) — 30 px below image centre in 256×256 pixels, matching
        # the position where fov-learned's gaze decoder converged.  No gaze
        # MLP exists in this policy; the shift is purely a static constant.
        if not self.is_blind:
            # Get the batch dimension from the first visual key.
            first_visual_key = next(
                k for k, v in observations.items() if v.dim() == 4
            )
            batch_size = observations[first_visual_key].shape[0]
            # Hardcoded gaze at (0.49, 0.62): a systematic downward shift of
            # ~30 pixels from image center, matched to where fov-learned's
            # learned-gaze module converged. Controlled test of whether
            # gaze position (not gaze dynamics) drives the H3 content shift.
            gaze = torch.tensor(
                [[0.49, 0.62]],
                device=observations[first_visual_key].device,
                dtype=torch.float32,
            ).expand(batch_size, 2).contiguous()
            aux_loss_state["gaze"] = gaze

            visual_feats = self.visual_encoder(observations, gaze=gaze)
            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # ---- Wijmans sensor block (same as parent) ----
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
                [torch.cos(compass_raw), torch.sin(compass_raw)],
                dim=-1,
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
            prev_actions_squeezed = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions_squeezed)
            prev_action_embed = self.prev_action_embedding(
                torch.where(
                    masks.view(-1), prev_actions_squeezed + 1, start_token
                )
            )
        else:
            prev_action_embed = self.prev_action_embedding(
                masks * prev_actions.float()
            )
        x.append(prev_action_embed)

        # ---- Concatenate and pass through the LSTM ----
        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


@baseline_registry.register_policy(name="FoveatedShiftedGazePolicy")
class FoveatedShiftedGazePolicy(NetPolicy):
    """Wijmans-faithful PointNav policy with foveated vision and learned gaze."""

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
            FoveatedShiftedGazeNet(
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

        policy_cfg = config.habitat_baselines.rl.policy[agent_name]

        # Foveation hyperparameters are hardcoded in __init__ defaults rather
        # than read from the policy config, because Hydra's structured config
        # validation rejects unknown fields on PolicyConfig. Edit the
        # FoveatedShiftedGazePolicy.__init__ defaults to tune them.
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=policy_cfg,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
        )
