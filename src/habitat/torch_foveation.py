"""
GPU-native foveation transform for Habitat DD-PPO.

Operates entirely on CUDA tensors — no numpy round-trips.
Ports the multi-scale Gaussian blur approach from src/envs/foveation.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TorchFoveationTransform(nn.Module):
    """Differentiable foveation transform for batched GPU images.

    Applies eccentricity-dependent blur: sharp at gaze center, progressively
    blurred towards periphery. Uses pre-computed multi-scale blur stack with
    per-pixel interpolation.

    Args:
        image_size: Input image spatial dimension (assumes square).
        fovea_radius: Radius of sharp foveal region (pixels).
        blur_sigma_max: Maximum Gaussian sigma at the periphery.
        falloff: Eccentricity-to-sigma mapping ('linear', 'quadratic', 'exponential').
        n_levels: Number of blur levels in the multi-scale stack.
    """

    def __init__(
        self,
        image_size: int = 128,
        fovea_radius: int = 16,
        blur_sigma_max: float = 6.0,
        falloff: str = "quadratic",
        n_levels: int = 5,
    ):
        super().__init__()
        self.image_size = image_size
        self.fovea_radius = fovea_radius
        self.blur_sigma_max = blur_sigma_max
        self.falloff = falloff
        self.n_levels = n_levels

        # Pre-compute pixel coordinate grids (registered as buffers for device movement)
        y, x = torch.meshgrid(
            torch.arange(image_size, dtype=torch.float32),
            torch.arange(image_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("_y_grid", y.clone())  # (H, W)
        self.register_buffer("_x_grid", x.clone())  # (H, W)

        # Pre-compute blur kernels for each level
        sigma_levels = torch.linspace(0, blur_sigma_max, n_levels)
        self.register_buffer("_sigma_levels", sigma_levels)

        # Max possible distance from any point to image corner
        self._max_dist = (2 ** 0.5) * image_size / 2

        # Build and register Gaussian kernels
        self._kernels = nn.ParameterList()  # not trainable, but stored with model
        self._paddings = []
        for i, sigma in enumerate(sigma_levels):
            if i == 0 or sigma < 0.5:
                self._kernels.append(None)
                self._paddings.append(0)
            else:
                ks = int(2 * (3 * sigma.item()) + 1)
                if ks < 3:
                    ks = 3
                if ks % 2 == 0:
                    ks += 1
                padding = ks // 2
                kernel_1d = self._make_gaussian_kernel_1d(ks, sigma.item())
                # Store as non-trainable parameter
                param = nn.Parameter(kernel_1d, requires_grad=False)
                self._kernels.append(param)
                self._paddings.append(padding)

    @staticmethod
    def _make_gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        return kernel / kernel.sum()

    def _blur_at_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Apply pre-computed Gaussian blur at a specific level."""
        if self._kernels[level] is None:
            return x  # level 0 = no blur

        C = x.shape[1]
        kernel_1d = self._kernels[level]
        padding = self._paddings[level]

        # Separable 2D convolution (depthwise).
        # Vertical kernel  : shape (C, 1, K, 1)
        # Horizontal kernel: shape (C, 1, 1, K)
        K = kernel_1d.shape[0]
        kh = kernel_1d.view(1, 1, K, 1).expand(C, 1, K, 1).contiguous()
        kw = kernel_1d.view(1, 1, 1, K).expand(C, 1, 1, K).contiguous()

        # F.pad uses (left, right, top, bottom) ordering for the spatial dims.
        out = F.pad(x, [0, 0, padding, padding], mode="reflect")
        out = F.conv2d(out, kh, groups=C)
        out = F.pad(out, [padding, padding, 0, 0], mode="reflect")
        out = F.conv2d(out, kw, groups=C)
        return out

    def _compute_eccentricity(
        self, gaze: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-pixel eccentricity given batched gaze positions.

        Args:
            gaze: (B, 2) gaze positions in [0, 1] normalized coordinates.

        Returns:
            ecc: (B, H, W) eccentricity values in [0, 1].
        """
        B = gaze.shape[0]
        # Convert normalized [0,1] to pixel coordinates
        gaze_x = gaze[:, 0] * self.image_size  # (B,)
        gaze_y = gaze[:, 1] * self.image_size  # (B,)

        # Distance from each pixel to gaze center
        dx = self._x_grid.unsqueeze(0) - gaze_x.view(B, 1, 1)  # (B, H, W)
        dy = self._y_grid.unsqueeze(0) - gaze_y.view(B, 1, 1)  # (B, H, W)
        dist = torch.sqrt(dx ** 2 + dy ** 2)

        # Per-sample max eccentricity: farthest image corner from
        # this gaze position. Replaces the previous static image-centre
        # max (valid only when gaze is at (0.5, 0.5)). For off-centre
        # gazes the static version over-saturates peripheral
        # eccentricity, producing systematically stronger peripheral
        # blur than the formula nominally specifies. This fix keeps
        # the foveation transform consistent across gaze locations so
        # cross-condition comparisons (e.g., fov-shifted vs.\ fov-fix)
        # differ only in the gaze location.
        corner_x = torch.tensor(
            [0.0, 0.0, self.image_size - 1.0, self.image_size - 1.0],
            device=gaze.device, dtype=gaze.dtype,
        )
        corner_y = torch.tensor(
            [0.0, self.image_size - 1.0, 0.0, self.image_size - 1.0],
            device=gaze.device, dtype=gaze.dtype,
        )
        corner_dx = corner_x.view(1, 4) - gaze_x.view(B, 1)
        corner_dy = corner_y.view(1, 4) - gaze_y.view(B, 1)
        corner_dist = torch.sqrt(corner_dx ** 2 + corner_dy ** 2)
        max_dist = corner_dist.max(dim=1).values.view(B, 1, 1)  # (B, 1, 1)

        ecc = (dist - self.fovea_radius) / (max_dist - self.fovea_radius)
        ecc = ecc.clamp(0.0, 1.0)

        return ecc

    def _eccentricity_to_sigma(self, ecc: torch.Tensor) -> torch.Tensor:
        """Map eccentricity to blur sigma using the falloff function."""
        if self.falloff == "linear":
            return ecc * self.blur_sigma_max
        elif self.falloff == "quadratic":
            return ecc ** 2 * self.blur_sigma_max
        elif self.falloff == "exponential":
            return (torch.exp(ecc) - 1) / (2.718281828 - 1) * self.blur_sigma_max
        else:
            raise ValueError(f"Unknown falloff: {self.falloff}")

    def forward(
        self,
        image: torch.Tensor,
        gaze: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply foveation to a batch of images.

        Args:
            image: (B, C, H, W) float tensor, already scaled to [0, 1].
            gaze: (B, 2) normalized gaze positions in [0, 1]. Defaults to center.

        Returns:
            Foveated image (B, C, H, W).
        """
        B, C, H, W = image.shape

        if gaze is None:
            gaze = torch.full((B, 2), 0.5, device=image.device)

        # Compute per-pixel sigma map
        ecc = self._compute_eccentricity(gaze)  # (B, H, W)
        sigma_map = self._eccentricity_to_sigma(ecc)  # (B, H, W)

        # Map sigma to fractional level index
        level_frac = sigma_map / self.blur_sigma_max * (self.n_levels - 1)  # (B, H, W)
        level_low = level_frac.long().clamp(0, self.n_levels - 2)
        level_high = level_low + 1
        alpha = (level_frac - level_low.float()).unsqueeze(1)  # (B, 1, H, W)

        # Build blur stack
        blur_stack = []
        for lvl in range(self.n_levels):
            blur_stack.append(self._blur_at_level(image, lvl))
        blur_stack = torch.stack(blur_stack, dim=0)  # (L, B, C, H, W)

        # Gather low and high levels per pixel
        # level_low: (B, H, W) -> expand to (B, C, H, W) for gather
        idx_low = level_low.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, H, W)
        idx_high = level_high.unsqueeze(1).expand(-1, C, -1, -1)

        # blur_stack is (L, B, C, H, W) -> rearrange to (B, L, C, H, W) for gather on dim=1
        blur_stack = blur_stack.permute(1, 0, 2, 3, 4)  # (B, L, C, H, W)

        # Gather: select level per pixel
        low_vals = torch.gather(
            blur_stack, 1, idx_low.unsqueeze(1)
        ).squeeze(1)  # (B, C, H, W)
        high_vals = torch.gather(
            blur_stack, 1, idx_high.unsqueeze(1)
        ).squeeze(1)  # (B, C, H, W)

        # Interpolate
        result = (1 - alpha) * low_vals + alpha * high_vals

        return result


class LogPolarFoveationTransform(nn.Module):
    """Log-polar foveation transform — a stronger foveation model than
    Gaussian-blur-only.

    Real primate foveation has variable spatial sampling (acuity falls as
    1/eccentricity), not just spatial-frequency degradation.  This module
    approximates that by:

      1. Sampling each input frame on a log-polar grid centred at the
         current gaze position, with ``n_rho`` rings exponentially spaced
         along radius and ``n_theta`` angular bins around the gaze.
      2. Producing an output tensor of shape ``(B, C, n_rho, n_theta)``
         that the encoder consumes as if it were a small uniform image.

    Compared to ``TorchFoveationTransform`` (uniform pixel grid +
    eccentricity-dependent Gaussian blur), this transform actually
    *removes* peripheral spatial samples rather than just blurring them,
    creating a true encoder-input bottleneck that scales with eccentricity.
    The downstream encoder therefore sees fewer and fewer spatial cells
    for content far from gaze.

    Defaults:
      - ``image_size``: input frame side length (square).
      - ``n_rho``:    number of radial rings (effective output height).
      - ``n_theta``:  number of angular bins (effective output width).
      - ``rho_min``:  inner radius (pixels) — anything inside is sampled
                     as the foveal centre.
      - ``rho_max``:  outer radius (pixels) — anything outside is clipped
                     to the boundary.

    Args (default values picked to roughly match human-foveation acuity):
      image_size = 256, n_rho = 32, n_theta = 64, rho_min = 4, rho_max = 128
    """

    def __init__(
        self,
        image_size: int = 256,
        n_rho: int = 32,
        n_theta: int = 64,
        rho_min: float = 4.0,
        rho_max: Optional[float] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_rho = n_rho
        self.n_theta = n_theta
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max if rho_max is not None
                             else image_size * (2 ** 0.5) / 2)

        # Pre-compute the log-spaced radial sampling grid.
        # log_rho values in [log(rho_min), log(rho_max)] inclusive.
        log_rho = torch.linspace(
            float(torch.log(torch.tensor(self.rho_min))),
            float(torch.log(torch.tensor(self.rho_max))),
            n_rho,
            dtype=torch.float32,
        )
        rho = torch.exp(log_rho)  # (n_rho,)
        self.register_buffer("_rho", rho)

        theta = torch.linspace(
            0.0, 2 * 3.141592653589793, n_theta + 1,
            dtype=torch.float32,
        )[:-1]  # exclude the duplicate at 2π
        self.register_buffer("_theta", theta)

        # Sampling offsets (dx, dy) per (rho, theta) cell — fixed in the
        # gaze-centred coordinate frame.  Final pixel coords are
        # gaze + offset.  Stored as (n_rho, n_theta, 2) for convenience.
        rr, tt = torch.meshgrid(rho, theta, indexing="ij")
        offsets = torch.stack(
            [rr * torch.cos(tt), rr * torch.sin(tt)], dim=-1
        )  # (n_rho, n_theta, 2)
        self.register_buffer("_offsets_xy", offsets)

    @property
    def output_shape(self) -> tuple:
        """The (H', W') shape the downstream encoder will see."""
        return (self.n_rho, self.n_theta)

    def forward(
        self,
        image: torch.Tensor,
        gaze: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply log-polar resampling to a batch of images.

        Args:
            image: (B, C, H, W) float tensor in [0, 1].
            gaze:  (B, 2) gaze positions in [0, 1] normalised coords;
                   defaults to image centre.

        Returns:
            Resampled tensor of shape (B, C, n_rho, n_theta).
        """
        B, C, H, W = image.shape
        if gaze is None:
            gaze = torch.full((B, 2), 0.5, device=image.device,
                              dtype=image.dtype)

        # Convert normalised gaze to pixel coords (B, 2).
        gaze_px = gaze * torch.tensor([W - 1.0, H - 1.0],
                                      device=image.device, dtype=image.dtype)

        # Build the per-batch sampling grid in pixel coords:
        #   target_xy[b, i, j] = gaze_px[b] + _offsets_xy[i, j]
        offsets_xy = self._offsets_xy.to(image.dtype)  # (n_rho, n_theta, 2)
        target_xy = (
            gaze_px.view(B, 1, 1, 2)
            + offsets_xy.view(1, self.n_rho, self.n_theta, 2)
        )  # (B, n_rho, n_theta, 2)

        # F.grid_sample expects the sampling grid in *normalised* coords
        # in [-1, 1], with the last-dim ordering (x, y) referring to W, H.
        # Build that explicitly to avoid an off-by-one ambiguity.
        target_x_norm = (target_xy[..., 0] / (W - 1.0)) * 2.0 - 1.0
        target_y_norm = (target_xy[..., 1] / (H - 1.0)) * 2.0 - 1.0
        grid = torch.stack([target_x_norm, target_y_norm], dim=-1)
        # (B, n_rho, n_theta, 2)

        out = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, C, n_rho, n_theta)

        return out
