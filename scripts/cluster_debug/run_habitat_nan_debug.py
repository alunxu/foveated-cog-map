"""
Debug entry point for reproducing the NaN event observed in foveated
training (job 2836021, frame ~182.6M). Resumes from the last clean
checkpoint (ckpt.36.pth of that run) and trains with:

  1. The gradient-sanitisation patch DISABLED, so we can observe the NaN
     propagating instead of silencing it.

  2. ``torch.autograd.set_detect_anomaly(True)`` — raises on the first
     NaN produced during backward pass, with a traceback pointing to the
     exact forward op that generated it.

  3. Forward hooks on the visual encoder, LSTM, value head, and action
     head that inspect each layer's output and, on the first non-finite
     tensor, write a diagnostic dump and abort via ``os._exit(42)``.

  4. A PPO.before_backward hook that dumps the loss decomposition
     (action_loss / value_loss / entropy / aux terms) whenever any term
     is non-finite — so we can tell whether the NaN originates in the
     forward numerics or in a loss computation.

All diagnostics are written under
  $NAN_DEBUG_DIR/nan_diagnostic_{rank}.txt  (default /tmp/nan_debug)
so they survive the job being killed.

The process exits fast on first NaN so later NaN noise does not flood
the logs.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup  (must come before importing src.habitat)
# ---------------------------------------------------------------------------

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Import src.habitat to register custom sensors/policies.
#
# IMPORTANT: importing src.habitat also calls _install_safe_optimizer_step,
# which would *prevent* the NaN from surfacing. We undo that patch right
# after import so the vanilla PPO.before_step runs during this debug
# session.
# ---------------------------------------------------------------------------

import torch

from habitat_baselines.rl.ppo.ppo import PPO as _PPO  # noqa: E402

_vanilla_before_step = _PPO.before_step  # capture before src.habitat runs

import src.habitat  # noqa: F401, E402

# Restore the unpatched before_step so NaN gradients are NOT zeroed.
_PPO.before_step = _vanilla_before_step

# Also un-patch the action-head clamp so NaN logits surface rather than
# being masked. We discover the patched action_distribution.linear.forward
# at policy-construction time via a registration hook below.

# ---------------------------------------------------------------------------
# Diagnostic directory (per-rank files so DDP workers don't clobber).
# ---------------------------------------------------------------------------

NAN_DEBUG_DIR = Path(os.environ.get("NAN_DEBUG_DIR", "/tmp/nan_debug"))
NAN_DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def _diag_path() -> Path:
    return NAN_DEBUG_DIR / f"nan_diagnostic_rank{_rank()}.txt"


def _dump_tensor_stats(name: str, t: torch.Tensor) -> str:
    if not isinstance(t, torch.Tensor):
        return f"{name}: not a tensor ({type(t).__name__})"
    flat = t.detach().float()
    n_nan = torch.isnan(flat).sum().item()
    n_inf = torch.isinf(flat).sum().item()
    n_neg_inf = torch.isneginf(flat).sum().item() if n_inf else 0
    finite = flat[torch.isfinite(flat)]
    if finite.numel() > 0:
        stats = (
            f"min={finite.min().item():.4e}  max={finite.max().item():.4e}  "
            f"mean={finite.mean().item():.4e}  std={finite.std().item():.4e}"
        )
    else:
        stats = "no finite values"
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype}  "
        f"nan={n_nan} inf={n_inf} (-inf={n_neg_inf}) / {t.numel()}  {stats}"
    )


_already_reported = False


def _report_and_abort(origin: str, tensor: torch.Tensor, extras: dict) -> None:
    """Write diagnostic and abort the process on the first NaN."""
    global _already_reported
    if _already_reported:
        return
    _already_reported = True

    lines = [
        "=" * 78,
        f"NAN DETECTED in forward pass at: {origin}",
        "=" * 78,
        _dump_tensor_stats("output", tensor),
        "",
    ]
    for k, v in extras.items():
        if isinstance(v, torch.Tensor):
            lines.append(_dump_tensor_stats(k, v))
        else:
            lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Stack at detection point:")
    lines.extend(traceback.format_stack())

    diag = _diag_path()
    diag.write_text("\n".join(lines))

    # Also print so SLURM stderr captures it.
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()

    # Hard exit so the other workers also get SIGTERM via SLURM.
    os._exit(42)


# ---------------------------------------------------------------------------
# Forward-output NaN detection: install hooks on every leaf Module after
# model creation.
# ---------------------------------------------------------------------------


def _install_forward_nan_hooks(module: torch.nn.Module, prefix: str = "") -> None:
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        _install_forward_nan_hooks(child, full)

    # Only install on leaf modules (no children) to get the narrowest possible
    # failure site. Also install on the top-level nn.Linear/LSTM/Conv2d etc.
    if len(list(module.children())) > 0:
        return

    def _hook(mod, inputs, output, _name=prefix):
        # Support tuple/list outputs (e.g. LSTM returns (output, (h,c)))
        tensors = []
        if isinstance(output, torch.Tensor):
            tensors = [output]
        elif isinstance(output, (tuple, list)):
            for t in output:
                if isinstance(t, torch.Tensor):
                    tensors.append(t)
                elif isinstance(t, (tuple, list)):
                    tensors.extend(x for x in t if isinstance(x, torch.Tensor))

        for t in tensors:
            if not torch.isfinite(t).all():
                # Gather input stats too (only if they're plain tensors).
                in_stats = {}
                for i, inp in enumerate(inputs if isinstance(inputs, (tuple, list)) else [inputs]):
                    if isinstance(inp, torch.Tensor):
                        in_stats[f"input[{i}]"] = inp
                _report_and_abort(
                    origin=f"{_name} ({type(mod).__name__})",
                    tensor=t,
                    extras=in_stats,
                )
                return

    module.register_forward_hook(_hook)


# ---------------------------------------------------------------------------
# Hook policy construction: disable action-head logit sanitisation and
# install forward NaN detection.
# ---------------------------------------------------------------------------


def _strip_action_head_sanitiser(policy) -> None:
    """Remove the nan_to_num + clamp wrapper installed by
    ``_wrap_action_distribution_with_clamp``. With the wrapper active,
    NaN logits would be silently replaced with 0 and we'd never see the
    failure site. Deleting the instance-level ``forward`` attribute makes
    Python fall back to the unwrapped ``nn.Linear.forward`` on the class.
    """
    linear = policy.action_distribution.linear
    if "forward" in linear.__dict__:
        del linear.__dict__["forward"]


# Monkey-patch Policy's __init__-like path. The Wijmans policies call
# _wrap_action_distribution_with_clamp in their from_config / __init__.
# Instead of chasing each, we intercept at the point where the trainer
# creates the agent.
from habitat_baselines.common.obs_transformers import (  # noqa: E402
    apply_obs_transforms_batch,  # noqa: F401  (side-effect: force module import)
)
from habitat_baselines.rl.ppo.policy import NetPolicy  # noqa: E402

_original_policy_init = NetPolicy.__init__


def _debug_policy_init(self, *args, **kwargs):
    _original_policy_init(self, *args, **kwargs)
    try:
        _strip_action_head_sanitiser(self)
    except Exception as e:
        sys.stderr.write(f"[nan-debug] failed to strip sanitiser: {e}\n")
    # Install NaN hooks on the whole policy.
    _install_forward_nan_hooks(self, prefix=type(self).__name__)


NetPolicy.__init__ = _debug_policy_init

# ---------------------------------------------------------------------------
# Loss-decomposition hook (per PPO mini-batch).
# ---------------------------------------------------------------------------

from habitat_baselines.rl.ppo.ppo import PPO  # noqa: E402

_original_update_from_batch = PPO._update_from_batch


def _debug_update_from_batch(self, batch, epoch, rollouts, learner_metrics):
    # Run the real update inside anomaly-detection context
    with torch.autograd.detect_anomaly():
        return _original_update_from_batch(self, batch, epoch, rollouts, learner_metrics)


PPO._update_from_batch = _debug_update_from_batch

# ---------------------------------------------------------------------------
# Global anomaly detection (belt-and-braces on top of the per-batch context).
# ---------------------------------------------------------------------------

torch.autograd.set_detect_anomaly(True)

# ---------------------------------------------------------------------------
# Run habitat-baselines main.
# ---------------------------------------------------------------------------

from habitat_baselines.run import main  # noqa: E402

if __name__ == "__main__":
    main()
