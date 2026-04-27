"""
Generate the 5 setup-figure panels (Figure 1 row 1) showing what each
condition's encoder receives.

Conditions:
  - blind: existing schematic (no visual input + sensor stack illustration)
  - matched: 48x48 downsample of base scene, upscaled to display size with
    nearest-neighbour interpolation so the pixelation is visible
  - uniform: base scene at full resolution
  - foveated (fix): base scene with eccentricity-dependent Gaussian blur
    centred at (0.5, 0.5); gaze marker drawn for clarity
  - foveated (learned): same blur model but gaze at (0.49, 0.62), the
    collapsed-gaze location of the trained learned-gaze module; gaze
    marker drawn

Reads:  docs/manuscript/fig/fig_uniform.png  (base scene)
Writes: docs/manuscript/fig/setup_<cond>.{png,pdf}

Run from project root:
    python scripts/paper_figures/make_setup_panels.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

OUT_DIR = Path("docs/manuscript/fig")
BASE = OUT_DIR / "fig_uniform.png"  # the existing 256x256 RGB scene


def foveated(img: np.ndarray, gx: float, gy: float, sigma_max: float = 8.0) -> np.ndarray:
    """Eccentricity-dependent Gaussian blur centred at gaze (gx, gy) in [0,1]^2.

    sigma(r) = sigma_max * r^2  with r = normalised distance from gaze.
    Approximated by mixing N pre-blurred copies via per-pixel weights.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = gx * (w - 1)
    cy = gy * (h - 1)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = np.sqrt(max(cx, w - 1 - cx) ** 2 + max(cy, h - 1 - cy) ** 2)
    r_norm = r / r_max
    sigma = sigma_max * r_norm ** 2  # 0 at gaze, sigma_max at far edge

    # Pre-compute blurred copies at log-spaced sigmas
    sigmas = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    blurred = [img.astype(np.float32)] + [
        gaussian_filter(img.astype(np.float32), sigma=(s, s, 0))
        for s in sigmas[1:]
    ]
    out = np.zeros_like(img, dtype=np.float32)
    # For each pixel pick the two adjacent sigmas + linear-interpolate
    for i in range(h):
        for j in range(w):
            s = sigma[i, j]
            # find bracketing sigmas
            k = np.searchsorted(sigmas, s)
            k = min(max(k, 1), len(sigmas) - 1)
            s_lo, s_hi = sigmas[k - 1], sigmas[k]
            t = 0.0 if s_hi == s_lo else (s - s_lo) / (s_hi - s_lo)
            out[i, j] = (1 - t) * blurred[k - 1][i, j] + t * blurred[k][i, j]
    return np.clip(out, 0, 255).astype(np.uint8)


def save_panel(arr: np.ndarray, name: str, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(arr)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333")
    fig.tight_layout(pad=0.1)
    out = OUT_DIR / f"fig1_{name}.pdf"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
    plt.close(fig)


def add_gaze_marker(arr: np.ndarray, gx: float, gy: float,
                    color: tuple = (255, 200, 0),
                    radius: int = 6) -> np.ndarray:
    """Draw a small filled circle + ring at gaze location."""
    h, w = arr.shape[:2]
    cx, cy = int(gx * (w - 1)), int(gy * (h - 1))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    out = arr.copy()
    # outer ring
    ring = (d >= radius - 1) & (d <= radius + 1)
    out[ring] = (0, 0, 0)
    # inner dot
    inner = d <= radius - 2
    out[inner] = color
    return out


def main() -> None:
    assert BASE.exists(), f"base image not found: {BASE}"
    img = np.array(Image.open(BASE).convert("RGB"))  # 256x256 typical

    # uniform: keep as-is
    save_panel(img, "setup_uniform")

    # matched: downsample to 48x48, then upscale with NEAREST so
    # the pixelation (encoder bottleneck visualisation) is visible
    pil = Image.fromarray(img).resize((48, 48), Image.BILINEAR)
    pil_up = pil.resize((img.shape[1], img.shape[0]), Image.NEAREST)
    save_panel(np.array(pil_up), "setup_matched")

    # foveated (fix): gaze at (0.5, 0.5)
    fov_fix = foveated(img, 0.5, 0.5, sigma_max=8.0)
    fov_fix_marked = add_gaze_marker(fov_fix, 0.5, 0.5)
    save_panel(fov_fix_marked, "setup_foveated_fix")

    # foveated (learned): gaze at (0.49, 0.62) — the collapsed-gaze
    # location of the trained learned-gaze module
    fov_learned = foveated(img, 0.49, 0.62, sigma_max=8.0)
    fov_learned_marked = add_gaze_marker(fov_learned, 0.49, 0.62,
                                          color=(120, 180, 255))
    save_panel(fov_learned_marked, "setup_foveated_learned")

    # blind: keep as-is (existing schematic in fig_blind.png is informative)
    # We just re-export with consistent border + sizing for visual
    # parity with the photo panels.
    blind_src = OUT_DIR / "fig_blind.png"
    if blind_src.exists():
        blind = np.array(Image.open(blind_src).convert("RGB"))
        save_panel(blind, "setup_blind")
    else:
        print("WARN: fig_blind.png missing, skipping setup_blind.")


if __name__ == "__main__":
    main()
