# Memory Maze (Pasukonis et al. 2023, ICLR)

* Paper: <https://openreview.net/pdf?id=yHLvIlE9RGN>
* Repo: <https://github.com/jurgisp/memory-maze>
* Project: <https://danijar.com/project/memorymaze/>

This is the **environment + offline dataset + probing protocol** — not a model. It is
the spine of our Top-1 plan.

## Public availability — verified

* `pip install memory-maze` (PyPI 1.0.2). Provides gym envs:
  `memory_maze:MemoryMaze-{9x9,11x11,13x13,15x15}-v0` (image-only) and
  `memory_maze:MemoryMaze-{...}-ExtraObs-v0` (adds agent_pos, agent_dir,
  maze_layout, target info to obs dict).
* Offline dataset on Google Drive:
  <https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB>
  — folders `memory-maze-9x9` and `memory-maze-15x15`. ~100GB per dataset.
  30k trajectories × 1001 steps each = 30M steps.

## NPZ trajectory schema (verified from paper Table 2)

| key | shape | dtype | meaning |
|-----|-------|-------|---------|
| `image` | (1001, 64, 64, 3) | uint8 | first-person RGB |
| `action` | (1001, 6) | binary | one-hot |
| `reward` | (1001,) | float | sparse |
| `agent_pos` | (1001, 2) | float | global xy ← **our probe target** |
| `agent_dir` | (1001, 2) | float | unit vector |
| `maze_layout` | (9,9) or (15,15) | binary | walls |
| `target_color` | (1001, 3) | float | RGB |
| `target_pos` | (1001, 2) | float | global xy of current target |
| `target_vec` | (1001, 2) | float | agent-centric current target |
| `targets_pos` | (1001, 3, 2) or (1001, 6, 2) | float | global xy of all objects |
| `targets_vec` | (1001, 3, 2) or (1001, 6, 2) | float | agent-centric all objects |

## Published probing protocol (Pasukonis 2023, §3.3 + §5)

* Probe network: **4-layer MLP, 1024 units, layer norm, ELU activation**.
* Probe input: model latent (1D vector, e.g., concat(h_t, z_t) for RSSM = 2048-d) +
  agent position + agent orientation.
* Probe targets:
  * **Walls** (BCE loss, accuracy %): predict `maze_layout` per cell. Constant
    baseline 80.8% (9x9), 78.3% (15x15). Supervised oracle 99.7% / 88.5%. RSSM
    (TBTT) reaches 94.0% / 80.5%.
  * **Objects** (MSE loss → RMSE in grid-cell units): predict `targets_vec`. Constant
    baseline 4.9 / 8.0. Supervised oracle 0.5 / 3.0. RSSM (TBTT) reaches 2.5 / 5.8.
* Eval window: average over second half of each trajectory (steps 500-1000) — the
  initial burn-in is when the agent has not yet seen enough of the maze for any
  model to predict layout.
* Constant baseline = always predict training-set mean.
* No-memory baseline = VAE that only sees current frame, no history.

## Why this is the right environment for our paper's prediction

* "Encoder bandwidth → linear-readable position in memory" requires both spatial
  variation in input and a memory-relevant task. Memory Maze is *designed* for this:
  agent must integrate visual evidence over hundreds of steps to localize itself.
* Position labels are released as ground truth, in the same NPZ as the input frames.
* Probe protocol is published, so we can match Pasukonis 2023 directly and avoid
  rolling our own.
* No retraining required for the encoder (we use frozen pretrained encoders, see
  vc1_r3m.md and dinov2 in EXPERIMENT_PLAN).

## Killer caveats

1. **Image resolution is 64×64**, smaller than VC-1/DINOv2's native 224. Standard
   embodied-AI practice is to upsample, which is mildly information-free but is
   what the field does (CortexBench uses this convention).
2. **Discrete action space (6)** vs Habitat's discrete-3 + GPS+Compass — slight
   mismatch in modality but the integration claim is action-modality-independent.
3. **Scripted-policy data** (random-walk-ish), not learned-policy rollouts. Could be
   a confound: the LSTM trained on this data may not learn the same integration
   strategy a goal-driven RL agent would.
4. **agent_pos is given to the probe as input** in the published protocol — we
   *flip* this and use it as the *target*. This is a deliberate variation; our
   probing question is different from theirs.

## Recommended subset for 8h budget

* 1000 train trajectories + 200 eval trajectories from Memory Maze 9x9 (smaller is
  fine for capacity purposes; 15x15 is harder to fit in 8h).
* Estimated download time: ~30-60 min for ~3GB of NPZ files via gdown (Google Drive
  rate limits permitting).
