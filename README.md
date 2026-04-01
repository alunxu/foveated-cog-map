# Agentic Cognitive Maps
## How Foveation Shapes Memory Content in Navigation Agents

**CS503 Visual Intelligence — EPFL, Spring 2026**

> When a visual agent navigates through a foveated sensor, what does it choose to remember, and does the difference reflect a rational strategy for managing perceptual uncertainty?

---

## Project Structure

```
Project/
├── cfgs/                       # Experiment configs
│   ├── base.yaml               #   Shared defaults
│   ├── foveated.yaml           #   Foveated agent (core)
│   ├── uniform.yaml            #   Uniform-vision baseline
│   └── matched_compute.yaml    #   Matched info budget control
│
├── src/                        # Main package
│   ├── envs/                   #   Environment + foveation
│   │   ├── foveation.py        #     Eccentricity-dependent blur
│   │   ├── nav_env.py          #     MiniGrid navigation wrapper
│   │   └── wrappers.py         #     Foveated / Uniform / Matched wrappers
│   ├── models/                 #   Agent architecture
│   │   ├── encoder.py          #     CNN visual encoder
│   │   ├── memory.py           #     GRU recurrent memory
│   │   └── policy.py           #     Action + gaze policy head
│   └── training/               #   RL training
│       ├── ppo.py              #     PPO algorithm
│       └── rollout.py          #     Rollout buffer
│
├── scripts/
│   └── train.py                # Main training entry point
│
├── setup_env.sh                # Cluster environment setup
├── submit_job.sh               # SLURM batch submission
├── sync_to_cluster.sh          # Upload code → SCITAS
└── sync_from_cluster.sh        # Download results ← SCITAS
```

## Agents

| Condition | Input | Gaze control | Purpose |
|-----------|-------|-------------|---------|
| **Foveated** | Eccentricity-degraded 64×64 | Yes (action + gaze) | Core agent |
| **Uniform** | Full-resolution 64×64 | No | Baseline |
| **Matched-compute** | Uniform low-res 64×64 | No | Controls for info quantity vs. distribution |

## Quick Start

```bash
# 1. Clone and setup (on cluster or locally)
bash setup_env.sh

# 2. Set your SCITAS username (for sync scripts)
export SCITAS_USER=<your-username>

# 3. Train foveated agent
python scripts/train.py --config cfgs/foveated.yaml

# 4. Train baseline
python scripts/train.py --config cfgs/uniform.yaml
```

## Storage (SCITAS Izar)

| What | Where | Notes |
|------|-------|-------|
| Code | `/home/<username>/CS503_Project/` | 100 GB, backed up |
| Data & checkpoints | `/scratch/<username>/CS503_Project/` | Large, ephemeral (>2 weeks deleted) |

## W&B

Set `wandb_entity` in `cfgs/base.yaml` to your W&B username or team.
