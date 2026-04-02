"""
Visual and vector encoders.

- CNNEncoder: processes egocentric RGB observations into a flat feature vector.
- VectorEncoder: processes pointgoal vector + previous action for blind agents.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Simple CNN encoder for 64x64 RGB observations.

    Architecture: 3 conv blocks (Conv -> BN -> ReLU -> MaxPool) -> flatten.

    Args:
        image_size: Input image dimension (assumes square).
        in_channels: Number of input channels (3 for RGB).
        channels: List of output channels per conv layer.
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        channels: list[int] = (16, 32, 64),
        kernel_size: int = 3,
    ):
        super().__init__()

        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([
                nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            c_in = c_out

        self.conv = nn.Sequential(*layers)

        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            out = self.conv(dummy)
            self.feature_dim = out.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) float tensor, normalised to [0, 1].

        Returns:
            features: (B, feature_dim) flat feature vector.
        """
        return self.conv(x).flatten(1)


class VectorEncoder(nn.Module):
    """Encoder for the blind agent (pointgoal vector + previous action).

    Following Wijmans et al.: each pointgoal component gets its own
    Linear(1, proj_dim) -> ReLU, plus an action embedding. All are
    concatenated to produce the feature vector.

    Args:
        input_dim: Dimension of the pointgoal vector (default 4).
        proj_dim: Projection dimension per component (default 32).
        n_actions: Number of possible actions for the action embedding.
        action_embed_dim: Dimension of the action embedding (default 32).
    """

    def __init__(
        self,
        input_dim: int = 4,
        proj_dim: int = 32,
        n_actions: int = 4,
        action_embed_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim

        # One projection per input component
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(1, proj_dim), nn.ReLU())
            for _ in range(input_dim)
        ])

        # Previous action embedding
        self.action_embed = nn.Embedding(n_actions + 1, action_embed_dim)  # +1 for "no action"
        self.no_action_idx = n_actions

        self.feature_dim = input_dim * proj_dim + action_embed_dim

    def forward(self, x: torch.Tensor, prev_action: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) pointgoal vector.
            prev_action: (B,) long tensor of previous actions, or None.

        Returns:
            features: (B, feature_dim) flat feature vector.
        """
        parts = []
        for i, proj in enumerate(self.projections):
            parts.append(proj(x[:, i:i+1]))

        if prev_action is None:
            prev_action = torch.full((x.shape[0],), self.no_action_idx,
                                     dtype=torch.long, device=x.device)
        parts.append(self.action_embed(prev_action))

        return torch.cat(parts, dim=-1)
