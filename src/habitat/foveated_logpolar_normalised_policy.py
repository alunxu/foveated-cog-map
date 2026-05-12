"""
Foveated-LogPolar PointNav policy with RunningMeanAndVar input normaliser
ENABLED (F-LP2 / F-norm-LP experiment — companion to F2 for the log-polar
condition).

The default ``FoveatedLogPolarWijmansPolicy`` disables
``normalize_visual_inputs`` for the same reason as the fov-fix variant
(autograd conflict with a never-actually-used gaze decoder).  The log-polar
condition has no gaze decoder either, so the flag can be safely flipped.

F-LP2 trains a log-polar variant with the normaliser ENABLED to test
whether the H1 "rich-encoder pass-through" finding (and the more general
format-shift dichotomy) is robust to this implementation choice for the
log-polar condition.  Together with F2 (the foveated counterpart) this
removes the disabled-normaliser confound from the entire rich-encoder
side of the 5-condition comparison.

Expected outcomes:
- F-LP2 GPS R² ≈ 0 (matches current fov-LP at R²=-0.03): normaliser is
  not a confound; bottleneck framing stands.
- F-LP2 GPS R² substantially > 0: the normaliser was hiding spatial
  information from the LSTM; rich-encoder vs bottleneck comparisons
  need to be redone with the normaliser flipped consistently.
"""
from habitat_baselines.common.baseline_registry import baseline_registry

from src.habitat.foveated_logpolar_policy import (
    FoveatedLogPolarWijmansNet,
    FoveatedLogPolarWijmansPolicy,
)


class FoveatedLogPolarNormalisedWijmansNet(FoveatedLogPolarWijmansNet):
    """Same as FoveatedLogPolarWijmansNet but with RunningMeanAndVar enabled.

    The override is via the ``_force_enable_normaliser`` class attribute,
    which the parent ``__init__`` reads to decide whether to construct the
    encoder with the normaliser turned on.  See
    ``foveated_logpolar_policy.py:FoveatedLogPolarWijmansNet`` for the
    gating logic.
    """

    _force_enable_normaliser: bool = True


@baseline_registry.register_policy(name="FoveatedLogPolarNormalisedWijmansPolicy")
class FoveatedLogPolarNormalisedWijmansPolicy(FoveatedLogPolarWijmansPolicy):
    """F-LP2 experiment: log-polar foveated agent with input normaliser enabled.

    Identical to ``FoveatedLogPolarWijmansPolicy`` except the underlying
    net is ``FoveatedLogPolarNormalisedWijmansNet`` (see above), which
    sets ``_force_enable_normaliser=True``.
    """

    _net_cls = FoveatedLogPolarNormalisedWijmansNet
