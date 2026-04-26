"""
Foveated PointNav policy at sigma_max=4 (F1b experiment, weak blur).

Identical to FoveatedWijmansPolicy except blur_sigma_max=4 (vs 8 in
fov-fix).  Half-strength blur, fills the gap between F1 (sigma=2) and
the existing fov-fix (sigma=8) in the strength sweep.

Expected outcome (encoder--memory race prediction): top-layer GPS R²
should remain close to uniform's (~0); the encoder--memory race claims
are about the rich-vs-bottleneck axis, and sigma=4 stays well within the
rich-encoder regime.
"""
from habitat_baselines.common.baseline_registry import baseline_registry

from src.habitat.foveated_policy import (
    FoveatedWijmansNet,
    FoveatedWijmansPolicy,
)


class FoveatedSigma4WijmansNet(FoveatedWijmansNet):
    """Marker class — sigma is set at policy __init__ time."""


@baseline_registry.register_policy(name="FoveatedSigma4WijmansPolicy")
class FoveatedSigma4WijmansPolicy(FoveatedWijmansPolicy):
    """F1b experiment: foveated agent with sigma_max=4 (weak blur)."""

    _net_cls = FoveatedSigma4WijmansNet

    def __init__(self, *args, blur_sigma_max: float = 4.0, **kwargs):
        super().__init__(
            *args,
            blur_sigma_max=blur_sigma_max,
            **kwargs,
        )
