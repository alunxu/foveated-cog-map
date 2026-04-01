"""
Foveation transform — eccentricity-dependent resolution degradation.

Simulates biological foveated vision: high resolution at the centre of gaze,
progressively blurred towards the periphery.

Member A is responsible for this module.
"""

import numpy as np
import torch
import torch.nn.functional as F


class FoveationTransform:
    """Apply foveation to an image given a gaze position.

    The image is divided into concentric rings around the gaze point.
    Each ring is blurred with a Gaussian kernel whose sigma increases
    with eccentricity (distance from gaze centre).

    Args:
        image_size: Size of the (square) input image.
        fovea_radius: Radius of the high-resolution foveal region (pixels).
        blur_sigma_max: Maximum Gaussian blur sigma at the image periphery.
        falloff: How blur increases with eccentricity ('linear', 'quadratic', 'exponential').
    """

    def __init__(
        self,
        image_size: int = 64,
        fovea_radius: int = 8,
        blur_sigma_max: float = 4.0,
        falloff: str = "quadratic",
    ):
        self.image_size = image_size
        self.fovea_radius = fovea_radius
        self.blur_sigma_max = blur_sigma_max
        self.falloff = falloff

        # Pre-compute distance grid (centred at image centre by default)
        y, x = np.mgrid[:image_size, :image_size].astype(np.float32)
        self._y_grid = y
        self._x_grid = x

    def compute_sigma_map(self, gaze_x: float, gaze_y: float) -> np.ndarray:
        """Compute per-pixel blur sigma given gaze position.

        Args:
            gaze_x, gaze_y: Gaze position in pixel coordinates [0, image_size).

        Returns:
            sigma_map: (H, W) array of blur sigma at each pixel.
        """
        # Distance from gaze centre
        dist = np.sqrt((self._x_grid - gaze_x) ** 2 + (self._y_grid - gaze_y) ** 2)

        # Normalised eccentricity: 0 at fovea edge, 1 at max distance
        max_dist = np.sqrt(2) * self.image_size / 2  # corner distance from centre
        eccentricity = np.clip((dist - self.fovea_radius) / (max_dist - self.fovea_radius), 0.0, 1.0)

        # Apply falloff function
        if self.falloff == "linear":
            sigma = eccentricity * self.blur_sigma_max
        elif self.falloff == "quadratic":
            sigma = eccentricity ** 2 * self.blur_sigma_max
        elif self.falloff == "exponential":
            sigma = (np.exp(eccentricity) - 1) / (np.e - 1) * self.blur_sigma_max
        else:
            raise ValueError(f"Unknown falloff: {self.falloff}")

        return sigma

    def __call__(self, image: np.ndarray, gaze_x: float = None, gaze_y: float = None) -> np.ndarray:
        """Apply foveation to an image.

        Args:
            image: (H, W, C) uint8 or float image.
            gaze_x, gaze_y: Gaze position. Defaults to image centre.

        Returns:
            Foveated image of same shape and dtype.
        """
        H, W = image.shape[:2]
        if gaze_x is None:
            gaze_x = W / 2
        if gaze_y is None:
            gaze_y = H / 2

        sigma_map = self.compute_sigma_map(gaze_x, gaze_y)

        # Apply spatially-varying blur using multi-scale approach:
        # Pre-blur image at several sigma levels, then select per-pixel.
        # This is an efficient approximation of true spatially-varying blur.
        original_dtype = image.dtype
        img_float = image.astype(np.float32) / 255.0 if original_dtype == np.uint8 else image.astype(np.float32)

        # Convert to torch for efficient Gaussian blur
        img_t = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        # Create blurred versions at different sigma levels
        n_levels = 5
        sigma_levels = np.linspace(0, self.blur_sigma_max, n_levels)
        blurred_stack = [img_t]  # level 0 = original

        for sigma in sigma_levels[1:]:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            if kernel_size < 3:
                kernel_size = 3
            padding = kernel_size // 2
            blurred = _gaussian_blur(img_t, kernel_size, sigma, padding)
            blurred_stack.append(blurred)

        blurred_stack = torch.cat(blurred_stack, dim=0)  # (n_levels, C, H, W)

        # For each pixel, interpolate between blur levels based on sigma_map
        sigma_normalized = sigma_map / self.blur_sigma_max * (n_levels - 1)  # [0, n_levels-1]
        level_low = np.floor(sigma_normalized).astype(int).clip(0, n_levels - 2)
        level_high = level_low + 1
        alpha = sigma_normalized - level_low  # interpolation weight

        # Gather and blend
        result = np.zeros_like(img_float)
        for c in range(img_float.shape[2]):
            low_vals = blurred_stack[level_low, c, np.arange(H)[:, None], np.arange(W)[None, :]].numpy()
            high_vals = blurred_stack[level_high, c, np.arange(H)[:, None], np.arange(W)[None, :]].numpy()
            result[:, :, c] = (1 - alpha) * low_vals + alpha * high_vals

        if original_dtype == np.uint8:
            result = (result * 255).clip(0, 255).astype(np.uint8)

        return result

    def get_uncertainty_map(self, gaze_history: list[tuple[float, float]]) -> np.ndarray:
        """Compute per-location perceptual uncertainty from gaze history.

        Areas that were foveated many times have low uncertainty;
        areas only seen peripherally have high uncertainty.

        Args:
            gaze_history: List of (gaze_x, gaze_y) positions.

        Returns:
            uncertainty_map: (H, W) float array, higher = more uncertain.
        """
        # Accumulate inverse-sigma (clarity) across all fixations
        clarity = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        for gx, gy in gaze_history:
            sigma_map = self.compute_sigma_map(gx, gy)
            # Clarity is inversely proportional to blur sigma
            # sigma=0 → perfect clarity, sigma=max → minimal clarity
            fixation_clarity = 1.0 / (1.0 + sigma_map)
            clarity += fixation_clarity

        # Uncertainty = inverse of accumulated clarity
        uncertainty = 1.0 / (1.0 + clarity)
        return uncertainty


def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float, padding: int) -> torch.Tensor:
    """Apply Gaussian blur to a batch of images."""
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable 2D convolution
    C = x.shape[1]
    kernel_h = kernel_1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    kernel_w = kernel_1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)

    x = F.pad(x, [0, 0, padding, padding], mode="reflect")
    x = F.conv2d(x, kernel_h, groups=C)
    x = F.pad(x, [padding, padding, 0, 0], mode="reflect")
    x = F.conv2d(x, kernel_w, groups=C)
    return x
