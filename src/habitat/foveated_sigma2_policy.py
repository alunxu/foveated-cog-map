"""
Foveated PointNav policy at sigma_max=2 (F1 experiment, very weak blur).

Identical to FoveatedWijmansPolicy except blur_sigma_max=2.  This is the
weak-end calibration anchor of the strength sweep: at sigma_max=2 the
periphery is barely blurred, so the encoder sees an image close to
uniform-RGB.

Expected outcome (encoder--memory race prediction): top-layer GPS R²
should be close to uniform's (~0), not close to coarse's (~0.78).  If
F1 instead shows a strong GPS code, the encoder--spatial-feature-variety
mechanism is wrong (something other than peripheral acuity is driving
the H1 ordering in the existing fov-fix vs. uniform comparison).
"""
from habitat_baselines.common.baseline_registry import baseline_registry

from src.habitat.foveated_policy import (
    FoveatedWijmansNet,
    FoveatedWijmansPolicy,
)


class FoveatedSigma2WijmansNet(FoveatedWijmansNet):
    """Marker class — sigma is set at policy __init__ time."""


@baseline_registry.register_policy(name="FoveatedSigma2WijmansPolicy")
class FoveatedSigma2WijmansPolicy(FoveatedWijmansPolicy):
    """F1 experiment: foveated agent with sigma_max=2 (very weak blur)."""

    _net_cls = FoveatedSigma2WijmansNet

    def __init__(self, *args, blur_sigma_max: float = 2.0, **kwargs):
        super().__init__(
            *args,
            blur_sigma_max=blur_sigma_max,
            **kwargs,
        )
