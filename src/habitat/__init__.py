"""
Habitat extensions for the Agentic Cognitive Maps project.

Importing this module registers all custom Habitat sensors and policies with
the global Habitat / habitat-baselines registries:

  - Sensors (Wijmans-faithful sensor stack):
      * GoalInStartFrameSensor   ("goal_in_start_frame")
      * CloseToGoalSensor        ("close_to_goal_indicator")

  - Policies:
      * WijmansPointNavPolicy   — Wijmans-faithful PointNav policy
                                  (used by blind / uniform / matched agents)
      * FoveatedWijmansPolicy   — foveated extension with learned gaze
                                  (used by the foveated agent)
"""

# Import order matters: sensors first (so the policy modules can refer to them),
# then the base Wijmans policy, then the foveated extension.
# Each import has a registration side effect.

from src.habitat import wijmans_sensors  # noqa: F401  (registers sensors)
from src.habitat.wijmans_policy import WijmansPointNavPolicy  # noqa: F401
from src.habitat.foveated_policy import FoveatedWijmansPolicy  # noqa: F401
