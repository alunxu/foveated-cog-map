"""
Foveated PointNav policy with RunningMeanAndVar input normaliser ENABLED
(F2 experiment).

The default ``FoveatedWijmansPolicy`` disables ``normalize_visual_inputs``
because the in-place ``RunningMeanAndVar._count`` buffer update conflicts
with autograd along the gaze-decoder gradient path.  Foveated-fixed has
no gaze decoder, so this conflict does not arise — but historically the
flag was hardcoded False to keep all foveated variants consistent.  This
is a confound when comparing fov-fix to uniform (which uses the
normaliser) on representational metrics.

The F2 experiment trains a fov-fix variant with the normaliser ENABLED,
to test whether the H1 "rich-encoder pass-through" finding is robust to
this implementation difference.

Expected outcomes:
- F2 GPS R² ≈ 0 (matches current fov-fix at R²=0.06): normaliser is not
  a confound; the bottleneck framing stands.
- F2 GPS R² substantially > 0: the normaliser was hiding spatial
  information from the LSTM in the original fov-fix; rich-encoder vs
  bottleneck comparisons need to be redone with the normaliser flipped
  consistently.
"""
from habitat_baselines.common.baseline_registry import baseline_registry

from src.habitat.foveated_policy import (
    FoveatedWijmansNet,
    FoveatedWijmansPolicy,
)


class FoveatedNormalisedWijmansNet(FoveatedWijmansNet):
    """Same as FoveatedWijmansNet but with RunningMeanAndVar enabled.

    The override is via the ``_force_enable_normaliser`` class attribute,
    which the parent ``__init__`` reads to decide whether to construct the
    encoder with the normaliser turned on.  See
    ``foveated_policy.py:FoveatedWijmansNet`` for the gating logic.
    """

    _force_enable_normaliser: bool = True


@baseline_registry.register_policy(name="FoveatedNormalisedWijmansPolicy")
class FoveatedNormalisedWijmansPolicy(FoveatedWijmansPolicy):
    """F2 experiment: foveated agent with input normaliser enabled.

    Identical to ``FoveatedWijmansPolicy`` except the underlying net is
    ``FoveatedNormalisedWijmansNet`` (see above), which sets
    ``_force_enable_normaliser=True``.
    """

    _net_cls = FoveatedNormalisedWijmansNet
