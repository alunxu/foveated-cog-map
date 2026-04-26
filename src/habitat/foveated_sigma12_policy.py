"""
Foveated PointNav policy at sigma_max=12 (F1c experiment, mid-strong blur).

Identical to FoveatedWijmansPolicy except blur_sigma_max=12 (vs 8 in
fov-fix and 20 in foveated_strong).  Fills the gap between the existing
fov-fix (sigma=8) and the strong-end (sigma=20).

Together with fov-fix (sigma=8) and foveated_strong (sigma=20), this
gives a 3-point monotonic strength sweep on the high side.
"""
from habitat_baselines.common.baseline_registry import baseline_registry

from src.habitat.foveated_policy import (
    FoveatedWijmansNet,
    FoveatedWijmansPolicy,
)


class FoveatedSigma12WijmansNet(FoveatedWijmansNet):
    """Marker class — sigma is set at policy __init__ time."""


@baseline_registry.register_policy(name="FoveatedSigma12WijmansPolicy")
class FoveatedSigma12WijmansPolicy(FoveatedWijmansPolicy):
    """F1c experiment: foveated agent with sigma_max=12 (mid-strong blur)."""

    _net_cls = FoveatedSigma12WijmansNet

    def __init__(self, *args, blur_sigma_max: float = 12.0, **kwargs):
        super().__init__(
            *args,
            blur_sigma_max=blur_sigma_max,
            **kwargs,
        )
