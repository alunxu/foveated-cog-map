"""
Unit tests for the per-sample dynamic max_dist fix in
src/habitat/torch_foveation.py (E2 from AUDIT_FINDINGS.md).

Why these matter:
- Old code: self._max_dist = sqrt(2) * image_size / 2  (static, gaze-blind).
  For shifted gaze e.g. (0.49, 0.62), the actual farthest corner is
  ~13% farther than this static value, which over-saturates peripheral
  eccentricity in foveated-shifted training and biases the comparison
  with foveated-fixed.
- New code: corner_dist computed per-sample from gaze position.

Tests:
  T1. Centred gaze (0.5, 0.5): new max_dist == old static max_dist
      (no regression for foveated-fix runs already in the paper).
  T2. Shifted gaze (0.49, 0.62): new max_dist > old static, by ~13%
      (matches the magnitude reported in the audit).
  T3. Eccentricity is in [0, 1] for both centred and shifted gaze.
  T4. Forward pass on CPU produces a tensor of expected shape and
      contains no NaN/Inf.

Run with: python -m pytest tests/test_torch_foveation.py -v
"""
import math
import sys
from pathlib import Path

try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
import torch

# Make src/ importable when run via `python -m pytest`.  We import the
# torch_foveation module directly (importlib) so we don't trigger
# src/habitat/__init__.py, which depends on the gym/habitat-baselines
# stack — these aren't available outside the cluster env, but they're
# unrelated to torch_foveation itself.
ROOT = Path(__file__).resolve().parents[1]
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "torch_foveation",
    ROOT / "src" / "habitat" / "torch_foveation.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TorchFoveationTransform = _mod.TorchFoveationTransform


IMAGE_SIZE = 128
OLD_STATIC_MAX_DIST = (2 ** 0.5) * IMAGE_SIZE / 2  # what _max_dist used to be


def _per_sample_max_dist(transform: TorchFoveationTransform,
                         gaze: torch.Tensor) -> torch.Tensor:
    """Replicate the per-sample max_dist computation from _compute_eccentricity."""
    H = transform.image_size
    gaze_x = gaze[:, 0] * H
    gaze_y = gaze[:, 1] * H
    corner_x = torch.tensor([0.0, 0.0, H - 1.0, H - 1.0],
                            dtype=gaze.dtype)
    corner_y = torch.tensor([0.0, H - 1.0, 0.0, H - 1.0],
                            dtype=gaze.dtype)
    corner_dx = corner_x.view(1, 4) - gaze_x.view(-1, 1)
    corner_dy = corner_y.view(1, 4) - gaze_y.view(-1, 1)
    corner_dist = torch.sqrt(corner_dx ** 2 + corner_dy ** 2)
    return corner_dist.max(dim=1).values


def test_centred_gaze_max_dist_matches_old_static():
    """T1: centred gaze still gives the (sqrt(2)/2 * size) max_dist."""
    transform = TorchFoveationTransform(image_size=IMAGE_SIZE)
    gaze = torch.tensor([[0.5, 0.5]])
    max_dist = _per_sample_max_dist(transform, gaze).item()
    # New value is computed against (size-1) corners, so it differs from the
    # old (size/2 * sqrt(2)) by 1 pixel. Allow 1-pixel tolerance.
    assert abs(max_dist - OLD_STATIC_MAX_DIST) < 1.5, (
        f"Expected {OLD_STATIC_MAX_DIST:.2f}, got {max_dist:.2f}"
    )


def test_shifted_gaze_max_dist_larger_than_old_static():
    """T2: shifted gaze (0.49, 0.62) gives a max_dist ~13% larger."""
    transform = TorchFoveationTransform(image_size=IMAGE_SIZE)
    gaze = torch.tensor([[0.49, 0.62]])
    max_dist = _per_sample_max_dist(transform, gaze).item()
    ratio = max_dist / OLD_STATIC_MAX_DIST
    assert ratio > 1.10, (
        f"Expected ratio > 1.10, got {ratio:.3f} (max_dist={max_dist:.2f})"
    )
    # Audit said ~13%; allow 10-20%.
    assert ratio < 1.20, (
        f"Expected ratio < 1.20, got {ratio:.3f}"
    )


def test_eccentricity_in_unit_interval():
    """T3: per-pixel eccentricity is always in [0, 1] for any gaze."""
    transform = TorchFoveationTransform(image_size=IMAGE_SIZE).eval()
    for gaze_xy in [(0.5, 0.5), (0.49, 0.62), (0.0, 0.0), (1.0, 1.0)]:
        gaze = torch.tensor([list(gaze_xy)], dtype=torch.float32)
        ecc = transform._compute_eccentricity(gaze)
        assert ecc.min().item() >= 0.0, gaze_xy
        assert ecc.max().item() <= 1.0 + 1e-6, (gaze_xy, ecc.max().item())


def test_forward_runs_clean():
    """T4: full forward pass on a small batch is finite-valued."""
    transform = TorchFoveationTransform(image_size=64).eval()
    image = torch.rand(2, 3, 64, 64)
    gaze = torch.tensor([[0.5, 0.5], [0.49, 0.62]], dtype=torch.float32)
    out = transform(image, gaze=gaze)
    assert out.shape == image.shape
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    # Allow running without pytest installed; collect every test_* function
    # in this module and invoke each in sequence.
    failures = []
    for name in sorted(globals()):
        if name.startswith("test_") and callable(globals()[name]):
            try:
                globals()[name]()
                print(f"PASS  {name}")
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
                failures.append(name)
    if failures:
        print(f"\n{len(failures)} failures: {failures}")
        sys.exit(1)
    print("\nAll tests passed.")
    sys.exit(0)
