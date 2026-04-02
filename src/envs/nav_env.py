"""
Navigation environment wrapper.

Wraps MiniGrid (or similar) environments to provide:
- RGB egocentric observations at configurable resolution
- Ground-truth occupancy grid and target location (for probing)
- Step-level metadata (agent position, orientation) for analysis

Member A is responsible for this module.
"""

import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper


class NavigationEnv(gym.Env):
    """Wrapper around MiniGrid that provides RGB observations and ground truth.

    Args:
        env_id: Gymnasium environment ID (e.g., 'MiniGrid-FourRooms-v0').
        image_size: Target observation size (will be resized).
        max_steps: Maximum steps per episode.
        render_mode: 'rgb_array' for headless, 'human' for display.
        seed: Random seed.
    """

    def __init__(
        self,
        env_id: str = "MiniGrid-FourRooms-v0",
        image_size: int = 64,
        max_steps: int = 500,
        render_mode: str = "rgb_array",
        seed: int = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.env_id = env_id
        self._seed = seed
        self._seeded = False  # Track whether RNG has been initialized

        # Create base environment
        self._base_env = gym.make(env_id, render_mode=render_mode, max_steps=max_steps)

        # Wrap to get RGB image observations instead of symbolic grid
        self._base_env = RGBImgObsWrapper(self._base_env, tile_size=8)
        self._base_env = ImgObsWrapper(self._base_env)

        self.action_space = self._base_env.action_space
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )

    @property
    def unwrapped_env(self):
        return self._base_env.unwrapped

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset environment.

        Returns:
            obs: (H, W, 3) uint8 RGB observation.
            info: dict with ground truth for probing.

        Note: Only the first reset uses the stored seed (to initialize the RNG).
        Subsequent resets with seed=None let MiniGrid's internal RNG produce
        diverse layouts, which is critical for generalization.
        """
        if seed is not None:
            reset_seed = seed
        elif not self._seeded:
            reset_seed = self._seed
        else:
            reset_seed = None
        self._seeded = True
        obs, info = self._base_env.reset(seed=reset_seed, options=options)
        obs = self._resize(obs)

        info["ground_truth"] = self._extract_ground_truth()
        info["agent_pos"] = tuple(self.unwrapped_env.agent_pos)
        info["agent_dir"] = int(self.unwrapped_env.agent_dir)

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step.

        Args:
            action: Integer action from the action space.

        Returns:
            obs, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        obs = self._resize(obs)

        info["ground_truth"] = self._extract_ground_truth()
        info["agent_pos"] = tuple(self.unwrapped_env.agent_pos)
        info["agent_dir"] = int(self.unwrapped_env.agent_dir)

        return obs, reward, terminated, truncated, info

    def _resize(self, obs: np.ndarray) -> np.ndarray:
        """Resize observation to target size."""
        if obs.shape[0] != self.image_size or obs.shape[1] != self.image_size:
            from PIL import Image
            img = Image.fromarray(obs)
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            obs = np.array(img)
        return obs

    def _extract_ground_truth(self) -> dict:
        """Extract ground-truth spatial information for probing.

        Returns dict with:
            occupancy_grid: (grid_h, grid_w) binary array — 1 = wall/obstacle
            target_pos: (x, y) position of the goal
            grid_size: (h, w) of the environment grid
        """
        grid = self.unwrapped_env.grid
        width, height = grid.width, grid.height

        # Occupancy grid: 1 where wall or obstacle
        occupancy = np.zeros((height, width), dtype=np.float32)
        target_pos = None

        for j in range(height):
            for i in range(width):
                cell = grid.get(i, j)
                if cell is not None:
                    if cell.type == "wall":
                        occupancy[j, i] = 1.0
                    elif cell.type == "goal":
                        target_pos = (i, j)

        return {
            "occupancy_grid": occupancy,
            "target_pos": target_pos,
            "grid_size": (height, width),
        }

    def close(self):
        self._base_env.close()

    def render(self):
        return self._base_env.render()
