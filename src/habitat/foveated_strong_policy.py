"""
Foveated PointNav policy with stronger blur (F4 experiment).

Identical to FoveatedWijmansPolicy except blur_sigma_max defaults to 20
instead of 8.  This tests whether foveation strength matters for the
encoder--memory race: at sigma_max=8 the periphery is moderately blurred
(retaining most spatial structure); at sigma_max=20 the periphery is
nearly featureless, approaching the regime where the visual encoder
cannot resolve world-frame structure.

Same network architecture as FoveatedWijmansPolicy.
Same training pipeline (DD-PPO, 250M frames target).
Only difference: blur_sigma_max in the foveation transform.

Expected outcomes:
- F4 GPS R² close to current fov-fix (≈ 0): the bottleneck framing is
  robust to blur strength within the Gaussian-blur model class — and a
  stronger foveation model (e.g., ``LogPolarFoveationTransform``) is
  needed to test the foveation-as-bottleneck hypothesis.
- F4 GPS R² closer to matched-compute (≈ 0.7): foveation strength
  matters, and our paper's "fov-fix ≈ uniform pass-through" finding is
  a Gaussian-blur-strength artefact rather than a true property of
  foveation.
"""
from habitat_baselines.common.baseline_registry import baseline_registry

from src.habitat.foveated_policy import (
    FoveatedWijmansNet,
    FoveatedWijmansPolicy,
)


class FoveatedStrongWijmansNet(FoveatedWijmansNet):
    """Same as FoveatedWijmansNet (kept as a marker class for clarity)."""

    # No override needed — sigma is set at policy __init__ time, not at
    # net class level.  Class exists for symmetry with
    # FoveatedNormalisedWijmansNet and for grep-ability.


@baseline_registry.register_policy(name="FoveatedStrongWijmansPolicy")
class FoveatedStrongWijmansPolicy(FoveatedWijmansPolicy):
    """F4 experiment: foveated agent with sigma_max=20 (stronger blur)."""

    _net_cls = FoveatedStrongWijmansNet

    def __init__(self, *args, blur_sigma_max: float = 20.0, **kwargs):
        # F4 default: 20.0 (vs 8.0 in fov-fix).  Override via kwargs if needed.
        super().__init__(
            *args,
            blur_sigma_max=blur_sigma_max,
            **kwargs,
        )
