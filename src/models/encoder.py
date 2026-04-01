"""
CNN visual encoder.

Processes egocentric RGB observations into a flat feature vector.
Designed to be lightweight (~1M parameters total for the full agent).

Member B is responsible for this module.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Simple CNN encoder for 64×64 RGB observations.

    Architecture: 3 conv blocks (Conv → BN → ReLU → MaxPool) → flatten.
    Output is a flat feature vector fed into the GRU memory.

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
