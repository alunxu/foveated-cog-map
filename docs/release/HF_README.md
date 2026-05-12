# spatial-memory-checkpoints

Trained PointGoal-navigation agents from a 5-condition visual-sensor study.
Each condition is a separate Habitat agent (DD-PPO, ResNet-18 + 3-layer LSTM-512)
trained for 250M frames on Gibson-0+. The five conditions vary only in the
visual front-end fed to the encoder; everything downstream (sensors, LSTM,
policy head, training pipeline) is held fixed.

## Repository layout

```
blind/              ckpt.0.pth ... ckpt.49.pth, latest.pth
coarse/             ckpt.0.pth ... ckpt.49.pth, latest.pth
foveated/           ckpt.0.pth ... ckpt.49.pth, latest.pth
foveated_logpolar/  ckpt.0.pth ... ckpt.49.pth, latest.pth
uniform/            ckpt.0.pth ... ckpt.49.pth, latest.pth
```

Each `ckpt.{i}.pth` is a `torch.save` of a dict with at least the key
`state_dict` (DD-PPO `actor_critic.*` weights). `latest.pth` is the trainer's
last-saved checkpoint pointer; `ckpt.49.pth` is the converged model used for
all paper-table numbers.

## Per-condition spec

| Condition           | Visual input            | Encoder spatial output | Policy class                    |
|---------------------|-------------------------|------------------------|---------------------------------|
| `blind`             | none (encoder bypassed) | n/a                    | `WijmansPointNavPolicy`         |
| `coarse`            | 48x48 RGB               | 1x1                    | `WijmansPointNavPolicy`         |
| `foveated`          | 256x256 RGB + Gaussian foveation (sigma_max=8) | 4x4 | `FoveatedWijmansPolicy`         |
| `foveated_logpolar` | 256x256 RGB -> 64x64 log-polar resampling | ~2x2 | `FoveatedLogPolarWijmansPolicy` |
| `uniform`           | 256x256 RGB             | 4x4                    | `WijmansPointNavPolicy`         |

## Loading a checkpoint and extracting hidden states

The agent must be reconstructed via the corresponding Habitat config; you
cannot instantiate the policy from the `.pth` alone. The companion code is at
<https://github.com/alunxu/foveated-cog-map>.

```python
import os, sys
import numpy as np
import torch
import habitat

# --- companion code (foveated-cog-map repo) provides these helpers --------
import src.habitat  # noqa: F401  registers all WijmansPointNavPolicy variants
from src.utils.habitat_env import load_habitat_config, load_policy

CONDITION_CONFIGS = {
    "blind":             "pointnav/ddppo_pointnav_blind_gibson",
    "coarse":            "pointnav/ddppo_pointnav_coarse_gibson",
    "foveated":          "pointnav/ddppo_pointnav_foveated_gibson",
    "foveated_logpolar": "pointnav/ddppo_pointnav_foveated_logpolar_gibson",
    "uniform":           "pointnav/ddppo_pointnav_uniform_gibson",
}

condition = "coarse"          # one of: blind / coarse / foveated / foveated_logpolar / uniform
ckpt_path = f"./{condition}/ckpt.49.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hydra-compose the matched eval config and instantiate env + policy.
config = load_habitat_config(CONDITION_CONFIGS[condition], ckpt_path, overrides=[
    "habitat.dataset.split=val",
    "habitat.environment.iterator_options.shuffle=False",
    "habitat.environment.max_episode_steps=2000",
])
env = habitat.Env(config=config.habitat)
policy, hidden_size, num_recurrent_layers, rnn_is_lstm = load_policy(
    config, env, ckpt_path, device,
)
policy.eval()

# --- roll out one deterministic episode and collect top-layer LSTM states ---
obs = env.reset()
rnn_hidden = torch.zeros(1, num_recurrent_layers, hidden_size, device=device)
prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
not_done = torch.zeros(1, 1, dtype=torch.bool, device=device)

h2_history = []   # top-layer hidden state h_2 per step
positions = []

while not env.episode_over:
    batch = {k: torch.from_numpy(np.asarray(v))[None].to(device) for k, v in obs.items()}
    with torch.no_grad():
        out = policy.act(batch, rnn_hidden, prev_action, not_done, deterministic=True)
    rnn_hidden = out.rnn_hidden_states            # shape: (1, num_recurrent_layers, hidden_size)
    prev_action = out.actions
    not_done = torch.ones(1, 1, dtype=torch.bool, device=device)

    # rnn_hidden layout (LSTM, num_layers=3): rows = [h_0, c_0, h_1, c_1, h_2, c_2]
    # The policy-readable top-layer hidden state is h_2 = rnn_hidden[:, 4, :].
    h2_history.append(rnn_hidden[0, 4].cpu().numpy())
    positions.append(np.array(env.sim.get_agent_state().position))
    obs = env.step(out.env_actions[0].item())

h2_history = np.stack(h2_history)   # shape: (T, 512)
positions  = np.stack(positions)    # shape: (T, 3)
```

The 512-d top-layer hidden state `h_2` is what the linear policy head reads
and is the substrate analysed across all probe / transplant / shape-metric
experiments in the accompanying paper.

## Indexing convention

- `ckpt.{i}.pth` for `i in [0, 49]` are 50 evenly-spaced training-stage
  checkpoints saved by the DD-PPO trainer; `i = 49` is the converged
  endpoint (~250M frames).
- `latest.pth` is the trainer's latest-pointer file (typically identical to
  `ckpt.49.pth` for converged runs).

## License

Trained model weights released under CC BY 4.0. The companion code
(<https://github.com/alunxu/foveated-cog-map>) is released under its own
license; see that repo for details.
