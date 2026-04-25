"""
Habitat extensions for the Agentic Cognitive Maps project.

Importing this module registers all custom Habitat sensors and policies with
the global Habitat / habitat-baselines registries:

  - Sensors (Wijmans-faithful sensor stack):
      * GoalInStartFrameSensor      ("goal_in_start_frame")
      * CloseToGoalSensor           ("close_to_goal_indicator")

  - Policies:
      * WijmansPointNavPolicy       — Wijmans-faithful PointNav policy
                                       (blind / uniform / matched agents)
      * FoveatedWijmansPolicy       — foveated, fixed-center gaze
                                       (the "fixed-center" leg of our
                                        gaze ablation)
      * FoveatedLearnedGazePolicy   — foveated, learned gaze decoder
                                       (the "learned" leg of the ablation;
                                        slow-gaze approximation, see
                                        foveated_learned_policy.py)
"""

# Import order matters: sensors first (so the policy modules can refer to
# them), then the base Wijmans policy, then the foveated extensions. Each
# import has a registration side effect.

from src.habitat import wijmans_sensors  # noqa: F401  (registers sensors)
from src.habitat.wijmans_policy import WijmansPointNavPolicy  # noqa: F401
from src.habitat.foveated_policy import FoveatedWijmansPolicy  # noqa: F401
from src.habitat.foveated_shifted_policy import FoveatedShiftedGazePolicy  # noqa: F401
from src.habitat.foveated_learned_policy import (  # noqa: F401
    FoveatedLearnedGazePolicy,
)

# F2/F3/F4 foveation-strength experiments (see experiments/
# foveation_strength_ablation.md and foveation_normaliser_invariance.md).
from src.habitat.foveated_normalised_policy import (  # noqa: F401
    FoveatedNormalisedWijmansPolicy,
)
from src.habitat.foveated_strong_policy import (  # noqa: F401
    FoveatedStrongWijmansPolicy,
)
from src.habitat.foveated_logpolar_policy import (  # noqa: F401
    FoveatedLogPolarWijmansPolicy,
)

# Auxiliary losses (ablation experiments):
#  - gaze_diversity: anti-collapse regulariser for learned-gaze policies.
#    Importing the module has side effects (config store + baseline
#    registry registration). Safe to import even when not used; the aux
#    loss only activates if it is listed in the yaml's
#    habitat_baselines.rl.auxiliary_losses section.
from src.habitat import gaze_diversity_loss  # noqa: F401
