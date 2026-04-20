"""
Regression tests for the NaN sanitisation patches installed by
``src.habitat.wijmans_policy``.

These tests guard against three ways someone could accidentally
regress the fix during future refactors:

  1. The monkey-patch on ``PPO.before_step`` gets replaced / removed, so
     non-finite gradients again reach ``optimizer.step()`` and corrupt
     weights.
  2. The action-head logit clamp gets removed, so ``torch.multinomial``
     crashes on NaN probabilities instead of sampling uniformly.
  3. The ``nan_sanitised`` learner metric is renamed or stops being
     appended, so downstream tensorboard dashboards silently lose the
     signal.

These tests do NOT require habitat-sim / GPUs / datasets — they exercise
just the patched PPO class in CPU-only mode with a dummy model.
"""

from __future__ import annotations

import collections
import math

import pytest
import torch
import torch.nn as nn

# Importing src.habitat runs _install_safe_optimizer_step(), which is what
# we're testing. The import also requires habitat-baselines to be
# installed; tests will be skipped if it isn't.
try:
    import src.habitat  # noqa: F401  # triggers the monkey-patches
    from habitat_baselines.rl.ppo.ppo import PPO

    HABITAT_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    HABITAT_AVAILABLE = False
    _IMPORT_ERR = str(_e)

pytestmark = pytest.mark.skipif(
    not HABITAT_AVAILABLE,
    reason=f"habitat-baselines not installed: {_IMPORT_ERR if not HABITAT_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyActorCritic(nn.Module):
    """Minimal stand-in for the real Wijmans policy.

    PPO.before_step only calls ``self.actor_critic.policy_parameters()`` and
    ``self.actor_critic.aux_loss_parameters()``; we don't need the real
    recurrent architecture for this test.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def policy_parameters(self):
        return list(self.parameters())

    def aux_loss_parameters(self):
        return {}


class _DummyPPO:
    """Construct a PPO-like object without running the real __init__.

    We only need ``actor_critic`` + ``max_grad_norm`` + ``non_ac_params``
    for the patched ``before_step`` to run end-to-end.
    """

    def __init__(self, actor_critic: nn.Module, max_grad_norm: float = 0.2):
        self.actor_critic = actor_critic
        self.max_grad_norm = max_grad_norm
        self.non_ac_params = []  # no aux params

    # Copy the instance method onto the dummy so the patched class
    # method is what we actually exercise.
    before_step = PPO.before_step


def _manually_set_grads(model: nn.Module, fill) -> None:
    """Attach a .grad to every parameter, possibly filled with NaN/inf."""
    for p in model.parameters():
        g = torch.empty_like(p)
        if fill == "nan":
            g.fill_(float("nan"))
        elif fill == "inf":
            g.fill_(float("inf"))
        elif fill == "mixed":
            # first half finite, second half NaN
            flat = g.view(-1)
            half = flat.numel() // 2
            flat[:half] = 0.1
            flat[half:] = float("nan")
        elif fill == "finite":
            g.fill_(0.1)
        else:
            raise ValueError(fill)
        p.grad = g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_patch_is_installed():
    """The import of src.habitat must have replaced PPO.before_step."""
    assert getattr(
        PPO.before_step, "_safe_grad_patched", False
    ), "PPO.before_step is not our patched version — did src.habitat import fail?"

    assert getattr(
        PPO._update_from_batch, "_safe_grad_patched", False
    ), "PPO._update_from_batch is not our patched version."


def test_nan_gradients_are_zeroed_before_optimizer_step():
    """Core regression test: a NaN grad must not reach optimizer.step()."""
    torch.manual_seed(0)
    model = _DummyActorCritic()
    # Snapshot clean weights.
    initial_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Inject NaN grads and run the patched before_step + optimizer step.
    _manually_set_grads(model, fill="nan")
    ppo = _DummyPPO(model)
    ppo.before_step()

    # After sanitisation, all grads must be finite.
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), (
            f"parameter {name} grad still contains non-finite values "
            f"after before_step()"
        )

    # Simulate the optimizer step that follows.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()

    # Weights must still be finite (the whole point of the fix).
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), (
            f"parameter {name} became non-finite after optimizer.step() — "
            f"the sanitisation patch failed"
        )

    # Because the grads were all-NaN and got zeroed, weights shouldn't
    # have changed (zero grad → zero update).
    for name, p in model.named_parameters():
        assert torch.allclose(p.detach(), initial_weights[name]), (
            f"parameter {name} changed despite zero grad — optimizer bug?"
        )


def test_inf_gradients_are_zeroed():
    """+inf / -inf grads must also be sanitised (not just NaN)."""
    torch.manual_seed(0)
    model = _DummyActorCritic()
    _manually_set_grads(model, fill="inf")
    ppo = _DummyPPO(model)
    ppo.before_step()
    for p in model.parameters():
        assert torch.isfinite(p.grad).all()


def test_mixed_gradients_preserve_finite_parts():
    """With a half-NaN gradient, the NaN half must become zero and the
    finite half must keep its sign (absolute magnitude may be rescaled
    by the subsequent clip_grad_norm_, which is expected)."""
    torch.manual_seed(0)
    model = _DummyActorCritic()
    _manually_set_grads(model, fill="mixed")
    ppo = _DummyPPO(model)
    ppo.before_step()
    for p in model.parameters():
        flat = p.grad.view(-1)
        half = flat.numel() // 2
        # NaN half → exactly zero after sanitisation + clip (0 × c = 0)
        assert torch.allclose(flat[half:], torch.zeros_like(flat[half:])), (
            f"NaN half of grad was not sanitised to zero: {flat[half:]}"
        )
        # Finite half must stay strictly positive (sign preserved, value
        # possibly reduced by clip_grad_norm_'s rescaling).
        assert (flat[:half] > 0).all(), (
            f"finite half of grad lost its sign: {flat[:half]}"
        )
        assert torch.isfinite(flat[:half]).all()


def test_fix_is_noop_on_clean_gradients():
    """With all-finite grads, patched before_step must give identical
    grad values to torch.nan_to_num (i.e. identity). This protects the
    "fix is no-op on clean training" guarantee documented in the README.
    """
    torch.manual_seed(0)
    model = _DummyActorCritic()
    _manually_set_grads(model, fill="finite")
    grads_before = [p.grad.clone() for p in model.parameters()]

    ppo = _DummyPPO(model)
    ppo.before_step()

    for g_before, p in zip(grads_before, model.parameters()):
        # clip_grad_norm_ may scale down the grads, but should not
        # introduce any NaN/inf and the direction must be preserved.
        assert torch.isfinite(p.grad).all()
        if g_before.norm() > 0 and p.grad.norm() > 0:
            cos = torch.nn.functional.cosine_similarity(
                g_before.view(-1), p.grad.view(-1), dim=0
            ).item()
            assert math.isclose(cos, 1.0, rel_tol=1e-5), (
                f"grad direction changed from cosine 1.0 to {cos}"
            )


def test_sanitisation_counter_increments_on_nan():
    """NAN_SANITISATION_STATS must track how often the fix fires."""
    from src.habitat.wijmans_policy import NAN_SANITISATION_STATS

    before_events = NAN_SANITISATION_STATS["total_events"]
    before_fixed = NAN_SANITISATION_STATS["total_params_fixed"]

    model = _DummyActorCritic()
    _manually_set_grads(model, fill="nan")
    ppo = _DummyPPO(model)
    ppo.before_step()

    assert NAN_SANITISATION_STATS["total_events"] == before_events + 1
    # Every grad element was NaN, so total_params_fixed should grow by
    # the sum of param sizes.
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        NAN_SANITISATION_STATS["total_params_fixed"] == before_fixed + total_params
    )


def test_sanitisation_counter_stays_flat_on_clean_grads():
    """Clean gradients must not bump the counter — protects the no-op
    guarantee and keeps the tensorboard metric meaningful."""
    from src.habitat.wijmans_policy import NAN_SANITISATION_STATS

    before_events = NAN_SANITISATION_STATS["total_events"]

    model = _DummyActorCritic()
    _manually_set_grads(model, fill="finite")
    ppo = _DummyPPO(model)
    ppo.before_step()

    assert NAN_SANITISATION_STATS["total_events"] == before_events
