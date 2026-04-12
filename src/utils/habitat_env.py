"""
Shared Habitat environment utilities used by collect_probes.py and
eval_shortcut.py.

Consolidates Hydra configuration loading, policy checkpoint loading,
and geometric helpers (heading extraction, SPL computation).
"""

import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def heading_from_quaternion(q):
    """Extract yaw heading (radians) from a Habitat np.quaternion.

    Uses quaternion components directly to avoid importing the `quaternion`
    package (which pulls in numba/llvmlite and may not load on all cluster
    nodes).

    The forward vector [0, 0, -1] is rotated by the quaternion, and we
    compute atan2 on the XZ projection.
    """
    w, x, y, z = q.w, q.x, q.y, q.z
    fx = -(2.0 * (x * z + w * y))
    fz = -(1.0 - 2.0 * (x * x + y * y))
    return np.arctan2(fx, fz)


def compute_spl(success, agent_path_length, geodesic_distance):
    """Compute Success weighted by (inverse) Path Length for a single episode.

    SPL = S · (geodesic / max(path, geodesic))
    """
    if not success:
        return 0.0
    return max(geodesic_distance, 1e-6) / max(agent_path_length, geodesic_distance, 1e-6)


# ---------------------------------------------------------------------------
# Hydra config loading
# ---------------------------------------------------------------------------

def load_habitat_config(config_name, ckpt_path, overrides=None):
    """Compose a Habitat config via Hydra with standard eval defaults.

    Args:
        config_name: Hydra config name (e.g. 'pointnav/ddppo_pointnav_blind_gibson')
        ckpt_path: path to checkpoint .pth file
        overrides: optional list of extra Hydra overrides

    Returns:
        config: OmegaConf config object (writable)
    """
    from habitat.config.default_structured_configs import register_hydra_plugin
    from habitat_baselines.config.default_structured_configs import (
        HabitatBaselinesConfigPlugin,
    )
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    GlobalHydra.instance().clear()

    import habitat_baselines
    config_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(habitat_baselines.__file__)), "config")
    )

    default_overrides = [
        "habitat_baselines.evaluate=True",
        "habitat_baselines.load_resume_state_config=False",
        "habitat_baselines.num_environments=1",
        f"habitat_baselines.eval_ckpt_path_dir={ckpt_path}",
        "habitat.environment.max_episode_steps=2000",
    ]
    if overrides:
        default_overrides.extend(overrides)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_name, overrides=default_overrides)

    OmegaConf.set_readonly(config, False)

    # Fill in mandatory defaults that Hydra compose() alone may leave MISSING
    sim_cfg = config.habitat.simulator
    if OmegaConf.is_missing(sim_cfg, "agents_order"):
        sim_cfg.agents_order = list(sim_cfg.agents.keys())

    # Point scenes_dir at the cluster data
    data_dir = os.environ.get(
        "HABITAT_DATA_DIR",
        f"/scratch/izar/{os.environ['USER']}/habitat_data",
    )
    config.habitat.dataset.scenes_dir = os.path.join(data_dir, "scene_datasets")

    return config


def load_policy(config, env, ckpt_path, device):
    """Instantiate and load a policy from a checkpoint.

    Args:
        config: Habitat config (from load_habitat_config)
        env: Habitat environment instance
        ckpt_path: path to checkpoint .pth file
        device: torch.device

    Returns:
        policy: loaded policy in eval mode
        hidden_size: LSTM hidden dimension
        num_recurrent_layers: total recurrent layers (for LSTM: 2 × n_layers)
        rnn_is_lstm: bool
    """
    import habitat_baselines.rl.ddppo.policy  # noqa: F401 — register PointNavResNetPolicy
    from habitat_baselines.common.baseline_registry import baseline_registry

    policy_name = config.habitat_baselines.rl.policy.main_agent.name
    policy_cls = baseline_registry.get_policy(policy_name)
    assert policy_cls is not None, f"Policy '{policy_name}' not found in registry"

    policy = policy_cls.from_config(
        config=config,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    policy.load_state_dict(
        {k.replace("actor_critic.", ""): v for k, v in state_dict.items()
         if k.startswith("actor_critic.")},
        strict=False,
    )
    policy.to(device)
    policy.eval()

    hidden_size = config.habitat_baselines.rl.ppo.hidden_size
    num_recurrent_layers = policy.net.num_recurrent_layers
    rnn_is_lstm = "LSTM" in config.habitat_baselines.rl.ddppo.rnn_type

    return policy, hidden_size, num_recurrent_layers, rnn_is_lstm


def ensure_project_on_path():
    """Add the project root to sys.path if not already present."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
