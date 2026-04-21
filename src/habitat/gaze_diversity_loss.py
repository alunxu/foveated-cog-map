"""
Gaze-diversity auxiliary loss for the learned-gaze foveated policy.

Background
----------
Our first ``FoveatedLearnedGazePolicy`` run (job 2839306) trained to 250M
frames and achieved SPL ≈ 0.82, but the gaze decoder collapsed: across
85k evaluation steps, the sigmoid-squashed gaze output varied by only
~0.005 on each dimension (see ``/scratch/izar/wxu/probing_results/h3_analysis.json``).
Functionally, the learned agent became an offset fixed-gaze agent. The
navigation loss alone was not a strong enough signal to prevent
collapse — either because of sigmoid saturation early in training or
because there is no task reward for varying where to look.

Loss
----
We add a minibatch-level negative variance penalty on ``aux_loss_state["gaze"]``
(already exposed by the learned-gaze policy's ``forward``):

    L_aux = -coef * mean(Var(gaze, dim=batch))

Maximising the variance across a minibatch directly fights collapse
without forcing any particular gaze trajectory. The upper bound of
variance for a uniform distribution on [0,1] is 1/12 ≈ 0.083, so a
coefficient of 0.01 puts the maximum possible aux-loss contribution at
-8e-4, which is ~1% of the typical PPO value-loss magnitude. This is
small enough to let navigation dominate but large enough to provide a
persistent anti-collapse pressure.

Registration
------------
The aux loss is registered under the name ``gaze_diversity`` and a
matching Hydra config node is stored at
``habitat_baselines/rl/auxiliary_losses/gaze_diversity`` so the usual
config override syntax works:

    habitat_baselines.rl.auxiliary_losses.gaze_diversity.coef=0.05

Activate by listing it in the yaml's ``auxiliary_losses`` field (see
``habitat_configs/ddppo_pointnav_foveated_learned_div_gibson.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default_structured_configs import AuxLossConfig


# ---------------------------------------------------------------------------
# Hydra structured config
# ---------------------------------------------------------------------------


@dataclass
class GazeDiversityLossConfig(AuxLossConfig):
    """Target-variance gaze diversity regulariser.

    ``coef`` is the penalty magnitude when variance is below ``target_var``;
    the loss is zero once the per-batch variance exceeds the target, so
    the task loss dominates once collapse is avoided.
    """

    coef: float = 0.01
    target_var: float = 0.01  # std ≈ 0.1 → gaze spans ~10% of image range


ConfigStore.instance().store(
    package="habitat_baselines.rl.auxiliary_losses.gaze_diversity",
    group="habitat_baselines/rl/auxiliary_losses",
    name="gaze_diversity",
    node=GazeDiversityLossConfig,
)


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------


@baseline_registry.register_auxiliary_loss(name="gaze_diversity")
class GazeDiversityLoss(nn.Module):
    """Target-variance anti-collapse regulariser for the learned-gaze policy.

    Reads ``aux_loss_state["gaze"]`` (shape ``(batch, 2)`` in [0,1]²) and
    returns ``coef * relu(target_var - Var_batch(gaze))``. When the
    per-batch variance of the gaze output falls below ``target_var``, a
    positive penalty is applied whose gradient pushes the gaze decoder
    to increase variance. Once variance exceeds ``target_var``, the loss
    is zero and the task loss is the only signal on the gaze decoder.

    This avoids the pathology observed in the uncapped variant (coef 0.01,
    pilot job 2840256): the aux loss kept pushing variance even after
    gaze was well-spread, eventually driving gaze toward uniform-random
    behaviour and destroying the consistent-foveation → visual-feature
    pipeline (SPL collapsed to 0.05, metric NaN at 8.3M frames).

    ``target_var = 0.01`` corresponds to per-dim std ≈ 0.1 — the gaze
    distribution covers roughly 10% of the image range. Baseline
    (collapsed) gaze had std ≈ 5e-4, i.e. three orders of magnitude
    below target; random gaze has std ≈ 0.29 (well above target).

    No-op when ``aux_loss_state`` lacks a ``gaze`` entry.
    """

    def __init__(
        self,
        action_space,
        net,
        coef: float = 0.01,
        target_var: float = 0.01,
    ):
        super().__init__()
        self.coef = coef
        self.target_var = target_var

    def forward(self, aux_loss_state, batch):
        gaze = aux_loss_state.get("gaze")
        if gaze is None or gaze.shape[0] < 2:
            device = (
                gaze.device if isinstance(gaze, torch.Tensor) else torch.device("cpu")
            )
            return {"loss": torch.zeros((), device=device)}

        # Mean per-dim variance across the minibatch.
        var = gaze.var(dim=0, unbiased=False).mean()
        # Hinge: only penalise when below target, zero otherwise.
        shortfall = torch.clamp(self.target_var - var, min=0.0)
        loss = self.coef * shortfall
        return {"loss": loss}
