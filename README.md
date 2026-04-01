# Agentic Cognitive Maps
## How Foveation Shapes Memory Content in Navigation Agents

**CS503 Visual Intelligence — EPFL, Spring 2026**

> When a visual agent navigates through a foveated sensor, what does it choose to remember? Does its memory over-represent uncertain (peripherally-seen) regions? Does it encode location-dependent confidence — a *belief map* rather than a plain map?

---

## Project Structure

```
Project/
├── cfgs/                           # Experiment configs
│   ├── base.yaml                   #   Shared defaults
│   ├── foveated.yaml               #   Foveated agent (core)
│   ├── uniform.yaml                #   Uniform-vision baseline
│   ├── matched_compute.yaml        #   Matched information budget control
│   └── ablations/                  #   Ablation studies
│       ├── fov_mild.yaml           #     Mild foveation falloff
│       ├── fov_sharp.yaml          #     Sharp foveation falloff
│       ├── memory_small.yaml       #     Small GRU hidden size
│       ├── memory_large.yaml       #     Large GRU hidden size
│       └── clutter_high.yaml       #     High environment clutter
│
├── src/                            # Main package
│   ├── envs/                       #   Member A: environments
│   │   ├── foveation.py            #     Foveation transform (eccentricity-dependent blur)
│   │   ├── nav_env.py              #     Navigation environment wrapper
│   │   └── wrappers.py             #     Gym wrappers (foveated / uniform / matched)
│   ├── models/                     #   Member B: agent architecture
│   │   ├── encoder.py              #     CNN visual encoder
│   │   ├── memory.py               #     GRU recurrent memory
│   │   └── policy.py               #     Policy head (action + gaze direction)
│   ├── training/                   #   Member B: RL training
│   │   ├── ppo.py                  #     PPO algorithm
│   │   └── rollout.py              #     Rollout buffer (stores hidden states)
│   ├── probing/                    #   Member C: representational analysis
│   │   ├── probes.py               #     Linear probe definitions
│   │   ├── targets.py              #     Ground-truth target computation
│   │   └── analysis.py             #     Probe training & evaluation
│   ├── analysis/                   #   Member D: information-theoretic analysis
│   │   ├── information.py          #     Information gain, detectability maps
│   │   ├── gaze_memory.py          #     Gaze-memory coupling metrics
│   │   └── bayesian.py             #     Bayesian ideal observer reference
│   └── utils/                      #     Shared utilities
│       ├── config.py               #     Config loading (OmegaConf)
│       ├── logging.py              #     W&B + local logging
│       └── checkpointing.py        #     Save / load models
│
├── scripts/                        # Entry points
│   ├── train.py                    #   Train an agent (foveated/uniform/matched)
│   ├── evaluate.py                 #   Evaluate navigation performance
│   ├── collect_probing_data.py     #   Run trained agent, collect hidden states + GT
│   ├── train_probes.py             #   Train linear probes on collected data
│   ├── analyze_gaze.py             #   Gaze-memory coupling analysis
│   └── visualize.py                #   Generate figures for the report
│
├── notebooks/                      # Jupyter analysis notebooks
├── tests/                          # Unit tests
├── outputs/                        # Training outputs (gitignored)
├── docs/                           # Project webpage (GitHub Pages)
│
├── setup_env.sh                    # Cluster environment setup
├── submit_job.sh                   # SLURM batch submission
├── sync_to_cluster.sh              # Upload code → SCITAS
└── sync_from_cluster.sh            # Download results ← SCITAS
```

## Agents

| Condition | Input | Gaze control | Purpose |
|-----------|-------|-------------|---------|
| **Foveated** | Eccentricity-degraded 64×64 | Yes (action + gaze) | Core agent |
| **Uniform** | Full-resolution 64×64 | No | Baseline |
| **Matched-compute** | Uniform low-res 64×64 | No | Controls for info quantity vs. distribution |

## Hypotheses

- **H1**: Foveated agent's memory *over-represents* peripherally-observed (high-uncertainty) regions
- **H2**: Hidden state encodes *location-dependent confidence* (belief map, not just spatial map)
- **H3**: Gaze and memory are *jointly adapted* — agent looks where memory is weakest

## Team

| Member | Role | Package |
|--------|------|---------|
| A | Environment + foveation | `src/envs/` |
| B | RL training pipeline (PPO, architecture) | `src/models/`, `src/training/` |
| C | Probing and representational analysis | `src/probing/` |
| D | Information-theoretic analysis + gaze-memory coupling | `src/analysis/` |

## Quick Start

```bash
# 1. Setup (cluster or local)
bash setup_env.sh

# 2. Train foveated agent
python scripts/train.py --config cfgs/foveated.yaml

# 3. Collect probing data from trained agent
python scripts/collect_probing_data.py --config cfgs/foveated.yaml --checkpoint outputs/<run>/best.pt

# 4. Train probes on hidden states
python scripts/train_probes.py --data outputs/<run>/probing_data/

# 5. Analyze gaze-memory coupling
python scripts/analyze_gaze.py --data outputs/<run>/probing_data/

# 6. Generate figures
python scripts/visualize.py --results outputs/<run>/
```

## Storage (SCITAS Izar)

| What | Where | Quota |
|------|-------|-------|
| Code | `/home/<username>/CS503_Project/` | 100 GB (backed up) |
| Datasets | `/scratch/<username>/CS503_Project/data/` | Large (ephemeral) |
| Checkpoints | `/scratch/<username>/CS503_Project/checkpoints/` | Large (ephemeral) |
| Probing data | `/scratch/<username>/CS503_Project/probing_data/` | Large (ephemeral) |
| Best models + logs | `/home/<username>/CS503_Project/outputs/` | 100 GB (backed up) |

## W&B

- **Entity**: `<your-wandb-username>`
- **Project**: `agentic-cognitive-map`
