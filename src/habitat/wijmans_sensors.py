"""
Custom Habitat sensors that match the input format of Wijmans et al. (ICLR 2023)
"Emergence of Maps in the Memories of Blind Navigation Agents".

Per Section A.1 of the paper, the agent receives at every timestep:
  - g           : the goal location relative to start (constant per episode, 2-D)
  - GPS         : current position relative to start (changes each step, 2-D)
  - compass     : current heading relative to start (changes each step, 1-D)
  - close_to_goal_indicator : min(||g - GPS||, 0.5) (1-D)

The standard Habitat `EpisodicGPSSensor` and `EpisodicCompassSensor` already
provide GPS and compass. This file adds the two missing pieces:

  - GoalInStartFrameSensor : the static goal in the episode's start frame
  - CloseToGoalSensor      : the close-to-goal indicator

These follow the exact same coordinate convention as `EpisodicGPSSensor`
(world (x,y,z) projected to start-frame (x,y) via [-z, x]).

Importing this module triggers `@registry.register_sensor` so the sensors
become available in Habitat configs as:
  - lab_sensors:
      - goal_in_start_frame_sensor
      - close_to_goal_indicator_sensor
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import spaces
from hydra.core.config_store import ConfigStore

from habitat.config.default_structured_configs import LabSensorConfig
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector


@registry.register_sensor
class GoalInStartFrameSensor(Sensor):
    r"""Returns the goal position in the episode's start coordinate frame.

    This is the variable Wijmans et al. call ``g``: a constant 2-D vector for
    the duration of an episode, giving the goal's location relative to where
    the agent started, in the start orientation. The agent must combine this
    with its (changing) GPS and compass readings to know where to go.

    cls_uuid: ``goal_in_start_frame``
    """

    cls_uuid: str = "goal_in_start_frame"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        goal_position = np.array(
            episode.goals[0].position, dtype=np.float32
        )

        # Express the goal in the start frame: rotate (goal - origin) by the
        # inverse of the start rotation, then project to the (x, y) horizontal
        # plane via [-z, x] -- same convention as EpisodicGPSSensor.
        goal_in_start = quaternion_rotate_vector(
            rotation_world_start.inverse(), goal_position - origin
        )
        return np.array(
            [-goal_in_start[2], goal_in_start[0]], dtype=np.float32
        )


@registry.register_sensor
class CloseToGoalSensor(Sensor):
    r"""Indicator of whether the agent is close to the goal.

    Returns ``min(||g - GPS||, 0.5)`` where ``g`` is the goal position in the
    start frame and ``GPS`` is the agent's current position in the start frame.
    Wijmans et al. include this explicitly because their agents do not learn
    a robust stopping logic without it.

    cls_uuid: ``close_to_goal_indicator``
    """

    cls_uuid: str = "close_to_goal_indicator"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=0.5,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        # Goal in start frame.
        goal_position = np.array(
            episode.goals[0].position, dtype=np.float32
        )
        goal_in_start = quaternion_rotate_vector(
            rotation_world_start.inverse(), goal_position - origin
        )
        g = np.array([-goal_in_start[2], goal_in_start[0]], dtype=np.float32)

        # Agent (GPS) in start frame.
        agent_state = self._sim.get_agent_state()
        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_state.position - origin
        )
        gps = np.array([-agent_position[2], agent_position[0]], dtype=np.float32)

        distance = float(np.linalg.norm(g - gps))
        return np.array([min(distance, 0.5)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Hydra config-store registration
#
# Habitat uses Hydra structured configs. To make our sensors usable from a
# YAML config (under `habitat/task/lab_sensors`), we need to register a
# dataclass for each one with the global ConfigStore.
# ---------------------------------------------------------------------------


@dataclass
class GoalInStartFrameSensorConfig(LabSensorConfig):
    type: str = "GoalInStartFrameSensor"


@dataclass
class CloseToGoalSensorConfig(LabSensorConfig):
    type: str = "CloseToGoalSensor"


cs = ConfigStore.instance()

cs.store(
    package="habitat.task.lab_sensors.goal_in_start_frame_sensor",
    group="habitat/task/lab_sensors",
    name="goal_in_start_frame_sensor",
    node=GoalInStartFrameSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.close_to_goal_indicator_sensor",
    group="habitat/task/lab_sensors",
    name="close_to_goal_indicator_sensor",
    node=CloseToGoalSensorConfig,
)
