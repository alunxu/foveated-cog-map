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
    """Coefficient for the per-batch gaze variance penalty."""

    coef: float = 0.01


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
    """Anti-collapse regulariser for the learned-gaze policy.

    Reads ``aux_loss_state["gaze"]`` (shape ``(batch, 2)`` in [0,1]²) and
    returns a loss equal to ``-coef * Var_batch(gaze).mean()``. A
    minibatch with diverse gazes produces a negative loss (pulls total
    loss down); a minibatch with collapsed gaze produces a loss of ~0.
    Minimising the total loss therefore encourages variance. No-op when
    ``aux_loss_state`` lacks a ``gaze`` entry (non-learned-gaze policies).
    """

    def __init__(self, action_space, net, coef: float = 0.01):
        super().__init__()
        self.coef = coef

    def forward(self, aux_loss_state, batch):
        gaze = aux_loss_state.get("gaze")
        if gaze is None or gaze.shape[0] < 2:
            # Return a zero tensor on an appropriate device so the
            # aggregator can stack without warnings.
            device = (
                gaze.device if isinstance(gaze, torch.Tensor) else torch.device("cpu")
            )
            return {"loss": torch.zeros((), device=device)}

        # Per-dim variance across the minibatch, averaged over the 2
        # gaze dims. Negated because PPO minimises the sum of losses.
        var_per_dim = gaze.var(dim=0, unbiased=False)
        loss = -self.coef * var_per_dim.mean()
        return {"loss": loss}
