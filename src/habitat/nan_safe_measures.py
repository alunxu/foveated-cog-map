"""
NaN/Inf-safe wrappers for Habitat's DistanceToGoal and
DistanceToGoalReward measures.

Why this exists
---------------
Habitat-sim's ``geodesic_distance(agent_pos, goal_pos, episode)`` returns
``+inf`` when the agent has reached a navmesh component disconnected from
the goal (a rare but real scene-edge case — Gibson + MP3D scenes have
isolated navmesh islands at thresholds, balconies, etc.). Habitat-lab's
``DistanceToGoal.update_metric`` stores that ``+inf`` directly into
``self._metric``; ``DistanceToGoalReward.update_metric`` then computes
``-(new_distance - previous_distance)`` and at the next step ``-inf − inf
= NaN``. The NaN per-step reward then poisons three downstream things:

  1. ``current_episode_reward`` (per-step accumulator) → episode-total NaN.
  2. ``running_episode_stats[k]`` cumulative across-all-episodes-ever:
     once polluted, **never** recovers because the cumulative is
     monotonically additive. Survives across SLURM/RunAI restarts via
     ``.habitat-resume-state.pth`` (see habitat-baselines
     ``ppo_trainer.py:686``).
  3. Window-aggregated logged reward / distance metrics:
     ``(window[-1] - window[0]).sum() / count`` → NaN forever after a
     single contaminated episode.

The actual learning impact is small (per inspection: 1 of 16 envs in our
foveated-seed0pre run had cumulative NaN, contaminating event rate
≈ 0.01% of episodes; per-step rewards entering the PPO rollout buffer
remain finite for the other 15/16 envs and >99.99% of steps in the
contaminated env). Weights stay clean (verified: 0/97 NaN tensors across
multiple latest.pth scans during training). However, the cumulative-stat
log is permanently misleading until training is restarted.

What this module does
---------------------
Monkey-patches ``DistanceToGoal.update_metric`` and
``DistanceToGoalReward.update_metric`` at import time so that:

  * ``DistanceToGoal._metric`` is clamped to the last finite value (or a
    large finite default if no finite value is available yet) when
    ``geodesic_distance`` returns a non-finite value. This keeps the
    cumulative ``running_episode_stats["distance_to_goal"]`` finite.

  * ``DistanceToGoalReward._metric`` is clamped to ``0.0`` whenever its
    underlying delta would be non-finite. A zero per-step delta gives
    no signal about progress in that step (rather than an arbitrary
    spurious reward), and this matches the intuition that the agent's
    geodesic information is briefly unavailable.

This is safer than letting NaN propagate (no learning signal corruption,
no permanent log poisoning) and safer than picking a fixed large default
(which would create a spurious huge reward when the agent walks back
on-navmesh and ``-(small_finite - large_default)`` becomes a large
positive reward).

Importing this module has the side effect of patching the global Habitat
classes; do it ONCE per process, before any task is constructed.
"""
from __future__ import annotations

import math


def _patch_distance_to_goal():
    """Apply the monkey-patch. Called at import time."""
    from habitat.tasks.nav.nav import DistanceToGoal, DistanceToGoalReward

    if getattr(DistanceToGoal, "_nan_safe_patched", False):
        return  # idempotent

    _orig_dtg_update = DistanceToGoal.update_metric

    def _nan_safe_dtg_update(self, episode, *args, **kwargs):
        _orig_dtg_update(self, episode, *args, **kwargs)
        m = self._metric
        if m is None or not math.isfinite(float(m)):
            # Use the last finite reading if we have one; otherwise fall back
            # to a large finite default (1000m is far beyond any plausible
            # PointGoal scene diameter — Gibson scenes are <100m).
            fallback = getattr(self, "_last_finite_metric", None)
            self._metric = fallback if fallback is not None else 1000.0
        else:
            self._last_finite_metric = float(m)

    DistanceToGoal.update_metric = _nan_safe_dtg_update
    DistanceToGoal._nan_safe_patched = True

    _orig_dtgr_update = DistanceToGoalReward.update_metric

    def _nan_safe_dtgr_update(self, episode, task, *args, **kwargs):
        _orig_dtgr_update(self, episode, task, *args, **kwargs)
        m = self._metric
        if m is None or not math.isfinite(float(m)):
            # No reliable signal about progress this step; emit zero delta.
            self._metric = 0.0

    DistanceToGoalReward.update_metric = _nan_safe_dtgr_update
    DistanceToGoalReward._nan_safe_patched = True


_patch_distance_to_goal()
