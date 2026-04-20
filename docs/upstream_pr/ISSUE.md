# Silent weight corruption: `PPO.before_step` lets NaN gradients reach `optimizer.step()`

**Target repo**: `facebookresearch/habitat-lab`
**File**: `habitat-baselines/habitat_baselines/rl/ppo/ppo.py`

## Summary

In long-horizon DD-PPO training runs, a single NaN gradient on one worker can corrupt every parameter in the network. The cause is that `torch.nn.utils.clip_grad_norm_` does not sanitise non-finite gradients, so after a rare NaN event the current `PPO.before_step` passes NaN gradients straight into `optimizer.step()`. The resulting corruption is silent: the training loop keeps running, metrics turn to NaN, and the final checkpoint is all zeros / NaN.

## Observed behaviour

One of our training runs (`habitat_baselines.rl.ppo.PPO` with default `max_grad_norm=0.2`, 2 GPUs × 4 environments, ResNet18 + 3-layer LSTM) reproduced this reliably:

| When | What |
|---|---|
| Frame 0 – 182,599,680 | Training stable, SPL ≈ 0.83, reward ≈ 10 |
| Frame 182,599,680 | One update step produces NaN gradient on one worker |
| Frame 182,599,680 + ε | DDP all-reduce averages NaN with finite → all workers see NaN grad |
| Frame 182,599,680 + 2ε | `clip_grad_norm_(grads, 0.2)` with any NaN in grads returns NaN |
| Frame 182,599,680 + 3ε | `optimizer.step()` writes NaN into every parameter |
| All subsequent frames | Every forward output is NaN, metrics are NaN, but `torch.multinomial` eventually raises |
| Final checkpoint | 90 / 97 parameter tensors are entirely NaN |

The training loop did not crash until `torch.multinomial` hit the NaN probability tensor **2.5 days later**. By that point all saved checkpoints for that condition were corrupt.

Log excerpt (one metric row before, one after):

```
2026-04-16 18:58:39   Num frames 182598720  reward: 10.341  spl: 0.829  success: 0.983
2026-04-16 19:03:20   Num frames 182599680  reward:    nan  spl: 0.000  success: 0.000
```

## Root cause

`torch.nn.utils.clip_grad_norm_` is NaN-unsafe:

```python
total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in params]))
# If any p.grad contains NaN:  total_norm = NaN
clip_coef = max_norm / (total_norm + 1e-6)
# clip_coef = max_norm / NaN = NaN
for p in params:
    p.grad.mul_(clip_coef)   # grad stays NaN
```

The NaN gradient then flows into `optimizer.step()`, which updates every parameter with `p -= lr * NaN`, producing NaN weights. From that point on the network is dead.

The originating NaN on a single mini-batch can come from standard float32 edge cases in PPO — softmax underflow + `log(0)`, value-head overflow, zero-variance advantage normalisation, LSTM cell blow-up combined with gate saturation — that long training runs hit only after tens of millions of updates. See https://github.com/pytorch/pytorch/issues/19222 for the PyTorch-side discussion.

The problem is not limited to foveated inputs or unusual policies: any DD-PPO training run long enough to hit these float32 edge cases is vulnerable.

## Fix

Sanitise non-finite gradient elements before `clip_grad_norm_`. This turns a single bad mini-batch into a no-op update instead of corrupting weights. The fix is:

- **No-op on clean training paths**: `torch.nan_to_num_` of a finite tensor is identity. Runs with zero NaN events produce bitwise-identical weights.
- **One line per parameter group**: minimal intrusion on the existing code.
- **Exposes a counter**: a `nan_sanitised` metric is added to `learner_metrics` so users can see whether the safety net fired.

See `patch.diff` in this directory for the concrete change.

## Minimal reproduction

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Simulate a NaN gradient (e.g. from log(0) in policy entropy)
for p in model.parameters():
    p.grad = torch.full_like(p, float("nan"))

# This is what PPO.before_step does today:
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
print("grad norm:", norm)                              # nan
print("grads finite:", all(
    torch.isfinite(p.grad).all() for p in model.parameters()
))                                                     # False
opt.step()
print("weights finite:", all(
    torch.isfinite(p).all() for p in model.parameters()
))                                                     # False
```

## Scope

The fix only touches `PPO.before_step`. It does not change any algorithm, hyperparameter, or public API. Anyone who wants the old behaviour (crash visibly on NaN instead of absorbing it) can set `self._skip_nan_sanitise = True` on the trainer — the patch checks that flag and falls through.

## Alternatives considered

- **Fix upstream in PyTorch**: `clip_grad_norm_` should arguably sanitise NaN, but changing that default behaviour is a larger discussion and unlikely to ship soon. This PR is a local fix in habitat-baselines that users benefit from immediately.
- **Detect and skip the whole optimizer step**: equivalent outcome (no weight change on bad batch) but more intrusive to the existing control flow.
- **Clip by value instead of by norm**: does not address the NaN case; `clip_grad_value_` with NaN input is also NaN-unsafe.
