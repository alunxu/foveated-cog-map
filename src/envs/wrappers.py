"""
Gymnasium wrappers for the three experimental conditions.

- FoveatedWrapper: applies eccentricity-dependent blur, adds gaze action
- UniformWrapper: passes through full-resolution observations
- MatchedComputeWrapper: downscales to match foveated agent's info budget

Member A is responsible for this module.
"""

import gymnasium as gym
import numpy as np
from src.envs.foveation import FoveationTransform


class FoveatedWrapper(gym.Wrapper):
    """Wraps a navigation env with foveated vision and gaze control.

    The action space becomes a Dict:
        - 'movement': original discrete action
        - 'gaze': continuous (gaze_x, gaze_y) in [0, 1] normalised coordinates

    The observation is the foveated image.
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
        self.current_gaze = (img_size / 2, img_size / 2)  # Start at centre

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gaze_history = []
        self.current_gaze = (self.img_size / 2, self.img_size / 2)

        foveated_obs = self.foveation(obs, *self.current_gaze)
        info["gaze_pos"] = self.current_gaze
        info["gaze_history"] = list(self.gaze_history)
        return foveated_obs, info

    def step(self, action: dict):
        """Step with movement + gaze action.

        Args:
            action: dict with 'movement' (int) and 'gaze' (array of [gx, gy] in [0,1]).
        """
        movement = action["movement"]
        gaze_norm = action["gaze"]  # [0, 1] normalised

        # Convert normalised gaze to pixel coordinates
        gaze_x = gaze_norm[0] * self.img_size
        gaze_y = gaze_norm[1] * self.img_size
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
    """

    def __init__(self, env, target_size: int = 32):
        super().__init__(env)
        self.target_size = target_size
        self._orig_size = env.observation_space.shape[0]

    def _downscale(self, obs: np.ndarray) -> np.ndarray:
        from PIL import Image
        img = Image.fromarray(obs)
        small = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        # Scale back up to original size (so the model input shape is the same)
        upscaled = small.resize((self._orig_size, self._orig_size), Image.BILINEAR)
        return np.array(upscaled)

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
