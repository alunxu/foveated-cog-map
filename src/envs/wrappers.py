"""
Gymnasium wrappers for the four experimental conditions.

- PointGoalWrapper: provides goal-relative coordinates, reward shaping, reduced actions
- BlindWrapper: discards visual observations, returns only pointgoal vector
- FoveatedWrapper: applies eccentricity-dependent blur, adds gaze action
- UniformWrapper: passes through full-resolution observations
- MatchedComputeWrapper: downscales to match foveated agent's info budget
"""

from collections import deque

import gymnasium as gym
import numpy as np
from src.envs.foveation import FoveationTransform


# MiniGrid action codes
_MG_LEFT = 0
_MG_RIGHT = 1
_MG_FORWARD = 2
_MG_TOGGLE = 5  # used as "done" signal
_MG_DONE = 6

# Our reduced action space: 0=left, 1=right, 2=forward
# No "done" action — MiniGrid terminates automatically when agent reaches goal
ACTION_MAP = [_MG_LEFT, _MG_RIGHT, _MG_FORWARD]
N_ACTIONS = len(ACTION_MAP)

# Direction vectors for MiniGrid directions (0=right, 1=down, 2=left, 3=up)
DIR_TO_VEC = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)


def _bfs_geodesic(occupancy: np.ndarray, start: tuple, end: tuple) -> float:
    """BFS shortest path on occupancy grid. Returns path length or inf."""
    h, w = occupancy.shape
    if start == end:
        return 0.0
    visited = set()
    queue = deque([(start[0], start[1], 0)])
    visited.add((start[0], start[1]))
    while queue:
        x, y, dist = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                if occupancy[ny, nx] < 0.5:  # not a wall
                    if (nx, ny) == (end[0], end[1]):
                        return float(dist + 1)
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
    return float("inf")


class PointGoalWrapper(gym.Wrapper):
    """Provides PointGoal-style observations and reward shaping.

    Applied to ALL conditions before the vision/blind wrapper.

    Adds to info:
        pointgoal: 4-d vector [relative_dist, relative_angle, cos_heading, sin_heading]
        collision: whether the last forward action hit a wall
        initial_geodesic: geodesic distance at episode start (for SPL)
        geodesic_dist: current geodesic distance to goal

    Reshapes reward: -delta_geodesic + success_bonus - time_penalty
    Reduces action space to 4: left, right, forward, done
    """

    def __init__(self, env, success_reward=10.0, time_penalty=0.001):
        super().__init__(env)
        self.success_reward = success_reward
        self.time_penalty = time_penalty
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        self._prev_geodesic = None
        self._initial_geodesic = None
        self._prev_pos = None
        self._prev_action = None
        self._occupancy = None
        self._target_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        gt = info["ground_truth"]
        self._occupancy = gt["occupancy_grid"]
        self._target_pos = gt["target_pos"]
        agent_pos = info["agent_pos"]

        geodesic = _bfs_geodesic(self._occupancy, agent_pos, self._target_pos)
        self._initial_geodesic = geodesic
        self._prev_geodesic = geodesic
        self._prev_pos = agent_pos
        self._prev_action = None

        pointgoal = self._compute_pointgoal(agent_pos, info["agent_dir"])
        info["pointgoal"] = pointgoal
        info["collision"] = False
        info["initial_geodesic"] = self._initial_geodesic
        info["geodesic_dist"] = geodesic

        return obs, info

    def step(self, action):
        mg_action = ACTION_MAP[action]
        prev_pos = self._prev_pos

        obs, _, terminated, truncated, info = self.env.step(mg_action)

        agent_pos = info["agent_pos"]
        agent_dir = info["agent_dir"]

        # Collision detection: forward action but didn't move
        collision = (mg_action == _MG_FORWARD) and (agent_pos == prev_pos)

        # Geodesic-based reward shaping
        cur_geodesic = _bfs_geodesic(self._occupancy, agent_pos, self._target_pos)

        if terminated and cur_geodesic <= 1.0:
            reward = self.success_reward
        elif terminated:
            # Terminated but not at goal (e.g., MiniGrid "done" action elsewhere)
            reward = -self.time_penalty
        else:
            delta = self._prev_geodesic - cur_geodesic
            reward = delta - self.time_penalty

        self._prev_geodesic = cur_geodesic
        self._prev_pos = agent_pos
        self._prev_action = action

        pointgoal = self._compute_pointgoal(agent_pos, agent_dir)
        info["pointgoal"] = pointgoal
        info["collision"] = collision
        info["initial_geodesic"] = self._initial_geodesic
        info["geodesic_dist"] = cur_geodesic

        return obs, reward, terminated, truncated, info

    def _compute_pointgoal(self, agent_pos, agent_dir):
        """Compute 4-d pointgoal vector."""
        if self._target_pos is None:
            return np.zeros(4, dtype=np.float32)

        # Relative vector from agent to goal in world coordinates
        dx = self._target_pos[0] - agent_pos[0]
        dy = self._target_pos[1] - agent_pos[1]

        # Distance
        dist = np.sqrt(dx**2 + dy**2)

        # Agent heading angle (MiniGrid: 0=right, 1=down, 2=left, 3=up)
        heading_angle = agent_dir * (np.pi / 2)  # 0, pi/2, pi, 3pi/2

        # Angle to goal in world frame
        goal_angle = np.arctan2(dy, dx)

        # Relative angle (how much agent needs to turn to face goal)
        rel_angle = goal_angle - heading_angle
        # Normalize to [-pi, pi]
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        return np.array([
            dist,
            rel_angle,
            np.cos(heading_angle),
            np.sin(heading_angle),
        ], dtype=np.float32)


class BlindWrapper(gym.Wrapper):
    """Discards visual observations, returns only pointgoal vector.

    For replicating the blind agent condition from Wijmans et al.
    """

    def __init__(self, env, pointgoal_dim=4):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(pointgoal_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        info["gaze_pos"] = None
        info["gaze_history"] = []
        return info["pointgoal"].copy(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        info["gaze_pos"] = None
        info["gaze_history"] = []
        return info["pointgoal"].copy(), reward, terminated, truncated, info


class FoveatedWrapper(gym.Wrapper):
    """Wraps a navigation env with foveated vision and gaze control.

    Action is a flat array: [movement_float, gaze_x, gaze_y]
      - movement_float is rounded to int for the discrete movement action
      - gaze_x, gaze_y are in [0, 1] normalised coordinates

    Using a flat Box action space instead of Dict for SyncVectorEnv compatibility.
    """

    def __init__(self, env, fovea_radius=8, blur_sigma_max=4.0, falloff="quadratic"):
        super().__init__(env)

        img_size = env.observation_space.shape[0]
        self.foveation = FoveationTransform(
            image_size=img_size,
            fovea_radius=fovea_radius,
            blur_sigma_max=blur_sigma_max,
            falloff=falloff,
        )

        self.img_size = img_size
        self.gaze_history = []
        self.current_gaze = (img_size / 2, img_size / 2)

        # Flat action space: [movement, gaze_x, gaze_y]
        n_act = env.action_space.n if hasattr(env.action_space, 'n') else 4
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([float(n_act - 1), 1.0, 1.0], dtype=np.float32),
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gaze_history = []
        self.current_gaze = (self.img_size / 2, self.img_size / 2)

        foveated_obs = self.foveation(obs, *self.current_gaze)
        info["gaze_pos"] = self.current_gaze
        info["gaze_history"] = list(self.gaze_history)
        return foveated_obs, info

    def step(self, action):
        """Step with movement + gaze action.

        Args:
            action: array-like [movement_float, gaze_x, gaze_y].
        """
        action = np.asarray(action, dtype=np.float32)
        movement = int(np.round(action[0]))
        gaze_x = float(np.clip(action[1], 0, 1)) * self.img_size
        gaze_y = float(np.clip(action[2], 0, 1)) * self.img_size
        self.current_gaze = (gaze_x, gaze_y)
        self.gaze_history.append(self.current_gaze)

        obs, reward, terminated, truncated, info = self.env.step(movement)
        foveated_obs = self.foveation(obs, gaze_x, gaze_y)

        info["gaze_pos"] = self.current_gaze
        info["gaze_history"] = list(self.gaze_history)
        info["uncertainty_map"] = self.foveation.get_uncertainty_map(self.gaze_history)

        return foveated_obs, reward, terminated, truncated, info


class UniformWrapper(gym.Wrapper):
    """Pass-through wrapper for uniform-vision baseline.

    No foveation applied. Observations are full resolution.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["gaze_pos"] = None
        info["gaze_history"] = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["gaze_pos"] = None
        info["gaze_history"] = []
        return obs, reward, terminated, truncated, info


class MatchedComputeWrapper(gym.Wrapper):
    """Downscales observations to match the foveated agent's information budget.

    Uses low uniform resolution instead of spatially-varying resolution.
    The CNN receives actual target_size × target_size images (no upscaling).
    """

    def __init__(self, env, target_size: int = 32):
        super().__init__(env)
        self.target_size = target_size
        # Update observation space to reflect the actual output size
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(target_size, target_size, 3),
            dtype=np.uint8,
        )

    def _downscale(self, obs: np.ndarray) -> np.ndarray:
        from PIL import Image
        img = Image.fromarray(obs)
        small = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        return np.array(small)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["gaze_pos"] = None
        info["gaze_history"] = []
        return self._downscale(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["gaze_pos"] = None
        info["gaze_history"] = []
        return self._downscale(obs), reward, terminated, truncated, info
