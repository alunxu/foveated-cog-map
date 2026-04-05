"""
Foveated PointNav policy for Habitat DD-PPO.

Extends PointNavResNetPolicy with learnable gaze control:
- Gaze direction is decoded from the PREVIOUS LSTM hidden state
- Foveation is applied to RGB observations BEFORE the ResNet encoder
- This means the agent "decides where to look" based on memory, then processes
  the foveated view through the visual pipeline
- Gaze is deterministic given hidden state (no separate action/entropy needed)
- Gradients flow: navigation loss → LSTM → gaze decoder → foveation → visual features

Member C is responsible for this module.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn

from habitat.tasks.nav.nav import ImageGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    PointNavResNetNet,
    PointNavResNetPolicy,
    ResNetEncoder,
)
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.ppo import NetPolicy
from habitat_baselines.utils.common import get_num_actions

from src.habitat.torch_foveation import TorchFoveationTransform

if TYPE_CHECKING:
    from omegaconf import DictConfig


class FoveatedResNetEncoder(ResNetEncoder):
    """ResNet encoder with foveation applied before the backbone.

    Wraps the standard ResNetEncoder to apply spatially-varying blur
    controlled by a gaze position before feeding to the CNN.
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
        blur_sigma_max: float = 6.0,
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
            # Get the image spatial size from observation space
            first_key = self.visual_keys[0]
            img_h = observation_space.spaces[first_key].shape[0]
            # Foveation operates on the half-size image (after avg_pool2d)
            self.foveation = TorchFoveationTransform(
                image_size=img_h // 2,
                fovea_radius=fovea_radius // 2,  # scale with downsampling
                blur_sigma_max=blur_sigma_max,
                falloff=falloff,
            )
        else:
            self.foveation = None

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        gaze: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_blind:
            return None

        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            obs_k = obs_k.permute(0, 3, 1, 2)  # (B, C, H, W)
            if self.key_needs_rescaling[k] is not None:
                obs_k = obs_k.float() * self.key_needs_rescaling[k]
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        x = torch.nn.functional.avg_pool2d(x, 2)  # standard habitat half-size

        # Apply foveation BEFORE running_mean_and_var and backbone
        if self.foveation is not None and gaze is not None:
            x = self.foveation(x, gaze)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class FoveatedPointNavResNetNet(PointNavResNetNet):
    """PointNav network with foveated vision.

    Key design: gaze is decoded from the PREVIOUS step's LSTM hidden,
    then used to foveate the current observation before visual encoding.
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
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        fovea_radius: int = 16,
        blur_sigma_max: float = 6.0,
        falloff: str = "quadratic",
        gaze_hidden: int = 64,
    ):
        # Initialize parent (this creates self.visual_encoder as standard ResNetEncoder)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            normalize_visual_inputs=normalize_visual_inputs,
            fuse_keys=fuse_keys,
            force_blind_policy=force_blind_policy,
            discrete_actions=discrete_actions,
        )

        # Replace the visual encoder with foveated version
        if not self.is_blind:
            if force_blind_policy:
                use_obs_space = spaces.Dict({})
            else:
                use_obs_space = spaces.Dict(
                    {
                        k: observation_space.spaces[k]
                        for k in observation_space.spaces
                        if len(observation_space.spaces[k].shape) == 3
                        and k != ImageGoalSensor.cls_uuid
                    }
                )

            self.visual_encoder = FoveatedResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
                fovea_radius=fovea_radius,
                blur_sigma_max=blur_sigma_max,
                falloff=falloff,
            )

            # Re-create visual_fc since output shape might differ
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )

        # Gaze decoder: hidden state → gaze position (2D, normalized [0,1])
        self.gaze_decoder = nn.Sequential(
            nn.Linear(hidden_size, gaze_hidden),
            nn.ReLU(),
            nn.Linear(gaze_hidden, 2),
            nn.Sigmoid(),  # output in [0, 1]
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
        aux_loss_state = {}

        # Decode gaze from PREVIOUS hidden state
        # rnn_hidden_states: (num_layers, B, hidden_size) for LSTM: (num_layers*2, B, hidden_size)
        # Use the last layer's hidden state
        if rnn_hidden_states.dim() == 3:
            # Take the last layer's h (not c for LSTM)
            prev_hidden = rnn_hidden_states[-2]  # second-to-last for LSTM (h of last layer)
        else:
            prev_hidden = rnn_hidden_states

        gaze = self.gaze_decoder(prev_hidden.detach().clone())
        # On episode reset (mask=0), default gaze to center
        gaze = gaze * masks.unsqueeze(-1) + 0.5 * (1 - masks.unsqueeze(-1))
        aux_loss_state["gaze"] = gaze

        if not self.is_blind:
            if (
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                # Pass gaze to foveated encoder
                visual_feats = self.visual_encoder(observations, gaze=gaze)

            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # Process non-visual inputs (same as parent)
        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())

        # Goal sensor processing (copied from parent to avoid calling super().forward())
        from habitat.tasks.nav.nav import (
            EpisodicCompassSensor,
            EpisodicGPSSensor,
            HeadingSensor,
            IntegratedPointGoalGPSAndCompassSensor,
            PointGoalSensor,
            ProximitySensor,
            ObjectGoalSensor,
            ImageGoalSensor,
        )
        from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert goal_observations.shape[1] == 3
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )
            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [torch.cos(sensor_observations[0]), torch.sin(sensor_observations[0])],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if hasattr(self, "obj_categories_embedding") and ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

        for uuid in [ImageGoalSensor.cls_uuid, InstanceImageGoalSensor.cls_uuid]:
            if uuid in observations:
                goal_image = observations[uuid]
                goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                goal_visual_output = goal_visual_encoder({"rgb": goal_image})
                goal_visual_fc = getattr(self, f"{uuid}_fc")
                x.append(goal_visual_fc(goal_visual_output))

        # Previous action embedding
        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(masks * prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state


@baseline_registry.register_policy(name="FoveatedPointNavResNetPolicy")
class FoveatedPointNavResNetPolicy(NetPolicy):
    """PointNav policy with foveated vision and learnable gaze control."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        fovea_radius: int = 16,
        blur_sigma_max: float = 6.0,
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
            FoveatedPointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
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

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Filter out eval cameras
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

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        # Read foveation params from policy config (with defaults)
        policy_cfg = config.habitat_baselines.rl.policy[agent_name]
        fovea_radius = getattr(policy_cfg, "fovea_radius", 16)
        blur_sigma_max = getattr(policy_cfg, "blur_sigma_max", 6.0)
        falloff = getattr(policy_cfg, "falloff", "quadratic")
        gaze_hidden = getattr(policy_cfg, "gaze_hidden", 64)

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
            fovea_radius=fovea_radius,
            blur_sigma_max=blur_sigma_max,
            falloff=falloff,
            gaze_hidden=gaze_hidden,
        )
