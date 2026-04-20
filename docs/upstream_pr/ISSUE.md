# Silent weight corruption: `PPO.before_step` lets NaN gradients reach `optimizer.step()`

**Target repo**: `facebookresearch/habitat-lab`
**File**: `habitat-baselines/habitat_baselines/rl/ppo/ppo.py`

## Summary

In long-horizon DD-PPO training runs, a single NaN gradient on one worker can corrupt every parameter in the network. The cause is that `torch.nn.utils.clip_grad_norm_` does not sanitise non-finite gradients, so after a rare NaN event the current `PPO.before_step` passes NaN gradients straight into `optimizer.step()`. The resulting corruption is silent: the training loop keeps running, metrics turn to NaN, and the checkpoint is eventually saved with NaN weights.

## Observed behaviour

Running a standard DD-PPO PointGoal navigation setup (ResNet + recurrent policy, default `max_grad_norm=0.2`, 2 GPUs) reproduced this reliably on a multi-day run:

| When | What |
|---|---|
| Training start → ~180M frames | Training stable, SPL ≈ 0.83, reward ≈ 10 |
| One update step, ~180M frames | One worker's mini-batch produces a NaN gradient |
| Next step (ε later) | DDP all-reduce averages NaN with finite → all workers see NaN grad |
| Next step (2ε later) | `clip_grad_norm_(grads, 0.2)` with any NaN in grads returns NaN |
| Next step (3ε later) | `optimizer.step()` writes NaN into every parameter |
| All subsequent frames | Every forward output is NaN, metrics are NaN |
| ~2.5 days later | `torch.multinomial` finally raises on the NaN probability tensor |
| Final checkpoint | The vast majority of parameter tensors are entirely NaN |

The training loop did not crash at the moment of corruption; it kept running with a dead model for days. By the time `torch.multinomial` raised, every recent checkpoint was unusable.

Log excerpt (one metric row just before and just after the event):

```
Num frames 182598720   reward: 10.341   spl: 0.829   success: 0.983
Num frames 182599680   reward:    nan   spl: 0.000   success: 0.000
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

The problem is not specific to any policy architecture: any DD-PPO run long enough to hit these float32 edge cases is vulnerable.

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

## Proposed fix

Sanitise non-finite gradient elements before `clip_grad_norm_` runs. This turns a single bad mini-batch into a no-op update instead of corrupting weights:

```python
# In PPO.before_step, before clip_grad_norm_:
for p in self.actor_critic.parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
```

Properties:

- **No-op on clean training paths**: `torch.nan_to_num_` of a finite tensor is identity. Runs with zero NaN events produce bitwise-identical weights.
- **One small addition, no API change, no new dependency.**
- **Can be paired with a `nan_sanitised` scalar in `learner_metrics`** so users can see in tensorboard whether the safety net ever fired.

I'm happy to open a PR with the full patch (including an optional metric) and a regression test if the maintainers are open to merging this.

## Alternatives considered

- **Fix upstream in PyTorch**: `clip_grad_norm_` should arguably sanitise NaN, but changing that default behaviour is a larger discussion and unlikely to ship soon. A local fix in habitat-baselines gives users immediate benefit.
- **Detect and skip the whole optimizer step on NaN**: equivalent outcome (no weight change on bad batch) but more intrusive to the existing control flow.
- **Clip by value instead of by norm**: does not address the NaN case; `clip_grad_value_` with NaN input is also NaN-unsafe.
