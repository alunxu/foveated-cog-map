"""
Foveated PointNav policy with log-polar resampling (F3 experiment).

Tests whether foveation \emph{model strength} matters for the
encoder--memory race finding.  The default foveation in our paper
(TorchFoveationTransform: uniform pixel grid, eccentricity-dependent
Gaussian blur) only degrades peripheral spatial frequency; the spatial
sample structure is preserved at full 256×256, and the ResNet-18 encoder
still has plenty of spatial cells to extract position from.  Real
primate foveation, by contrast, has variable spatial sampling — acuity
falls roughly as 1/eccentricity — so the periphery actually has fewer
spatial samples, not just blurrier ones.

This policy uses ``LogPolarFoveationTransform`` instead of
``TorchFoveationTransform``.  The output of the foveation step is
``(B, C, n_rho, n_theta)``, with default 64×64 (foveation samples reduced
from 128×128 by 4× total).  The ResNet-18 encoder then operates on this
smaller, gaze-centred grid.  Whether GPS now decodes from the LSTM
hidden state — closer to matched-compute (R²≈0.78) than to fov-fix
(R²≈0.06) — is the empirical question this experiment answers.

Same training pipeline (DD-PPO, 250M frames target).
Same downstream LSTM + sensor stack.
Difference: foveation transform now actually creates an encoder-input
bottleneck.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder

from src.habitat.foveated_policy import (
    FoveatedWijmansNet,
    FoveatedWijmansPolicy,
)
from src.habitat.torch_foveation import LogPolarFoveationTransform


class FoveatedLogPolarResNetEncoder(ResNetEncoder):
    """ResNet encoder with log-polar foveation applied before the backbone.

    Same shape contract as ``FoveatedResNetEncoder`` for the downstream
    LSTM, but the foveation output is ``(B, C, n_rho, n_theta)`` instead
    of ``(B, C, image_size//2, image_size//2)``.  The ResNet backbone
    handles the rectangular input transparently.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
        n_rho: int = 64,
        n_theta: int = 64,
        rho_min: float = 4.0,
        rho_max: Optional[float] = None,
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
            # Match ``FoveatedResNetEncoder``'s convention: avg-pool the
            # input by 2 before applying foveation, so the foveation
            # operates on an ``img_h // 2`` square.
            self.foveation = LogPolarFoveationTransform(
                image_size=img_h // 2,
                n_rho=n_rho,
                n_theta=n_theta,
                rho_min=rho_min,
                rho_max=rho_max,
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

        # Apply log-polar foveation BEFORE running mean/var and backbone.
        # If ``gaze`` is None (e.g., this is fov-fix-style with centred
        # gaze) the LogPolarFoveationTransform defaults to image centre.
        if self.foveation is not None:
            x = self.foveation(x, gaze)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class FoveatedLogPolarWijmansNet(FoveatedWijmansNet):
    """Wijmans-faithful PointNav net with log-polar foveation.

    Replaces the visual encoder with ``FoveatedLogPolarResNetEncoder``;
    the rest of the architecture (LSTM, sensor stack) is unchanged from
    ``FoveatedWijmansNet``.  We rebuild the visual_encoder after the
    parent ``__init__`` sets up the (Gaussian-blur) variant — at the cost
    of one wasted construction, but it keeps the diff small.
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
        # Log-polar-specific kwargs
        n_rho: int = 64,
        n_theta: int = 64,
        rho_min: float = 4.0,
        rho_max: Optional[float] = None,
        # Unused but accepted for parent-class compat:
        fovea_radius: int = 16,
        blur_sigma_max: float = 8.0,
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
            gaze_hidden=gaze_hidden,
        )

        # Replace the visual encoder with the log-polar version.
        # The parent already constructed a Gaussian-blur encoder; we
        # discard it.  This is a small inefficiency at construction time
        # but keeps the inheritance simple.
        if not self.is_blind:
            from habitat_baselines.rl.ddppo.policy import resnet
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

            _enable = type(self)._force_enable_normaliser and normalize_visual_inputs

            self.visual_encoder = FoveatedLogPolarResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=_enable,
                n_rho=n_rho,
                n_theta=n_theta,
                rho_min=rho_min,
                rho_max=rho_max,
            )

            # Re-build visual_fc to match the (likely smaller) encoder
            # output shape.
            import numpy as np
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    int(np.prod(self.visual_encoder.output_shape)),
                    hidden_size,
                ),
                nn.ReLU(True),
            )


@baseline_registry.register_policy(name="FoveatedLogPolarWijmansPolicy")
class FoveatedLogPolarWijmansPolicy(FoveatedWijmansPolicy):
    """F3 experiment: foveated agent with log-polar resampling.

    Tests whether stronger foveation (variable spatial sampling, not just
    blur) creates a real encoder-input bottleneck — and whether that
    moves LSTM GPS encoding closer to the bottleneck conditions
    (blind, matched-compute) or leaves it at the rich-encoder pass-through
    level (current fov-fix).
    """

    _net_cls = FoveatedLogPolarWijmansNet
