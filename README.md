# Agentic Cognitive Maps
## How Foveation Shapes Memory Content in Navigation Agents

**CS503 Visual Intelligence — EPFL, Spring 2026**

> When a visual agent navigates through a foveated sensor, what does it choose to remember, and does the difference reflect a rational strategy for managing perceptual uncertainty?

---

## Motivation

Wijmans et al. (ICLR 2023) showed that blind navigation agents — receiving only GPS+Compass — develop internal spatial maps in their recurrent memory, detectable via linear probing. We extend this finding by asking: **how does foveated vision change what the agent remembers?**

Biological vision is foveated: high resolution at the center of gaze, degraded in the periphery. When an agent must navigate with such a sensor and actively control where it looks, its memory must selectively encode spatial information under perceptual uncertainty. This project probes whether foveation produces qualitatively different memory representations than uniform vision.

## Approach

We replicate and extend the Wijmans et al. framework:

1. **Train navigation agents** under multiple vision conditions using DD-PPO
2. **Probe recurrent hidden states** for spatial information (occupancy maps, agent position, collision detection)
3. **Compare memory content** across conditions to understand how sensor type shapes internal representations

### Experimental Conditions

| Condition | Input | Gaze control | Purpose |
|-----------|-------|-------------|---------|
| **Blind** | GPS+Compass only | No | Core replication of Wijmans et al. |
| **Uniform** | Full-resolution RGB-D | No | Visual baseline |
| **Foveated** | Eccentricity-degraded RGB-D | Yes (learned) | Core experiment |
| **Matched-compute** | Low-resolution uniform RGB-D | No | Controls for info quantity vs. distribution |

### Environments

| Environment | Purpose | Status |
|-------------|---------|--------|
| **Habitat + Gibson** (492 scenes) | Main results — photorealistic 3D navigation | Training in progress |
| **MiniGrid FourRooms** (19×19 grid) | Pipeline validation / controlled ablation | Complete |

## Project Structure

```
Project/
├── cfgs/                           # Experiment configs (MiniGrid)
│   ├── base.yaml                   #   Shared defaults
│   ├── blind.yaml                  #   Blind agent (GPS+Compass only)
│   ├── foveated.yaml               #   Foveated agent (core experiment)
│   ├── uniform.yaml                #   Uniform-vision baseline
│   └── matched_compute.yaml        #   Matched info budget control
│
├── habitat_configs/                # Experiment configs (Habitat DD-PPO)
│   ├── ddppo_pointnav_blind.yaml           #   Blind agent on test scenes
│   └── ddppo_pointnav_blind_gibson.yaml    #   Blind agent on Gibson
│
├── src/                            # Main package
│   ├── envs/                       #   Environment + foveation
│   │   ├── foveation.py            #     Eccentricity-dependent blur
│   │   ├── nav_env.py              #     MiniGrid navigation wrapper
│   │   └── wrappers.py             #     PointGoal / Blind / Foveated / Uniform / Matched wrappers
│   ├── models/                     #   Agent architecture
│   │   ├── __init__.py             #     NavigationAgent (CNN + Vector encoder, GRU memory)
│   │   ├── encoder.py              #     CNNEncoder + VectorEncoder
│   │   ├── memory.py               #     GRU recurrent memory
│   │   └── policy.py               #     Action + gaze policy head
│   └── training/                   #   RL training
│       ├── ppo.py                  #     PPO algorithm
│       └── rollout.py              #     Rollout buffer
│
├── scripts/
│   ├── train.py                    # MiniGrid training entry point
│   └── probe.py                    # Linear probing analysis
│
├── setup_env.sh                    # Cluster environment setup
├── submit_job.sh                   # SLURM batch — MiniGrid training
├── submit_habitat.sh               # SLURM batch — Habitat DD-PPO training
├── submit_all.sh                   # Submit all 4 MiniGrid conditions
├── submit_probe.sh                 # SLURM batch — probing analysis
├── submit_all_probes.sh            # Submit probing for all agents
├── sync_to_cluster.sh              # Upload code → SCITAS
└── sync_from_cluster.sh            # Download results ← SCITAS
```

## Datasets

| Dataset | Size | Access | Purpose |
|---------|------|--------|---------|
| **Gibson (Habitat trainval)** | 13 GB (492 scenes) | [License form](http://gibsonenv.stanford.edu/database/) | Main training scenes |
| **Matterport3D** | ~15 GB (90 scenes) | [Access agreement](https://niessner.github.io/Matterport/) | Evaluation (pending) |
| **PointNav episodes (Gibson v1)** | 385 MB | [Direct download](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) | Pre-generated start/goal pairs |
| **Habitat test scenes** | 89 MB | Free (via `habitat_sim.utils.datasets_download`) | Pipeline validation |

## Setup

### MiniGrid (local / cluster)

```bash
# Create environment
bash setup_env.sh

# Train blind agent
python scripts/train.py --config cfgs/blind.yaml

# Probe hidden states
python scripts/probe.py \
    --checkpoint outputs/blind_agent/checkpoint_final.pt \
    --config cfgs/blind.yaml \
    --n_episodes 200 \
    --output probing_results/blind/
```

### Habitat (SCITAS cluster)

```bash
# One-time setup
conda create -n habitat python=3.9 cmake=3.22 -y
conda activate habitat
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat
cd ~/habitat-lab && pip install -e habitat-lab && pip install -e habitat-baselines

# Download data
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /scratch/izar/$USER/habitat_data/

# Train blind agent (4 GPUs)
sbatch submit_habitat.sh pointnav/ddppo_pointnav_blind_gibson
```

## Storage (SCITAS Izar)

| What | Where | Notes |
|------|-------|-------|
| Code | `/home/<username>/CS503_Project/` | 100 GB, backed up |
| Data & checkpoints | `/scratch/izar/<username>/` | Large, ephemeral (>2 weeks deleted) |
| Habitat repos | `/home/<username>/habitat-{sim,lab}/` | Cloned from GitHub |

## Current Status

### Phase 1: Pipeline + MiniGrid PoC ✅
- [x] MiniGrid PointNav environment with BFS reward shaping
- [x] PPO training pipeline with SyncVectorEnv (4 conditions)
- [x] NavigationAgent with GRU memory (CNN + Vector encoders)
- [x] Linear probing script (occupancy, position, collision)
- [x] Blind agent: 35% success, spatial maps detectable in hidden states

### Phase 2: Habitat Replication (in progress)
- [x] Habitat-sim + habitat-lab + habitat-baselines installed on SCITAS
- [x] Gibson dataset downloaded (492 scenes, 13 GB)
- [x] DD-PPO blind agent config created (3-layer LSTM-512)
- [ ] Blind agent training on Gibson (500M steps, 4× V100)
- [ ] Sighted agent training on Gibson
- [ ] Probing analysis on Habitat hidden states

### Phase 3: Foveation Extension (planned)
- [ ] Foveation transform on Habitat RGB observations
- [ ] Foveated agent with learned gaze control
- [ ] Matched-compute control condition
- [ ] Comparative probing: what does foveation change in memory?

## Team Workstreams

The project is split into 4 independent workstreams. Each member develops against **shared interfaces** (checkpoint format, hidden state shape, config schema) so work proceeds in parallel without blocking.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Habitat Simulator                             │
│                  (Gibson 492 scenes, PointNav)                       │
└──────────┬───────────────────┬───────────────────┬───────────────────┘
           │                   │                   │
     ┌─────▼─────┐     ┌──────▼──────┐     ┌──────▼──────┐
     │   Blind    │     │   Uniform   │     │  Foveated   │
     │ GPS+Comp.  │     │  RGB-D full │     │ RGB-D blur  │
     │  only      │     │  resolution │     │ + gaze ctrl │
     └─────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   DD-PPO Training   │
                    │  (3-layer LSTM-512) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Linear Probing    │
                    │  on frozen hidden   │
                    │  states             │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
        ┌──────────┐   ┌──────────┐   ┌───────────────┐
        │Occupancy │   │ Position │   │   Collision    │
        │   Map    │   │Prediction│   │  Detection     │
        │  Decoder │   │          │   │                │
        └──────────┘   └──────────┘   └───────────────┘
```

### Workstream Assignments

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐                              │
│  │   MEMBER A       │  │   MEMBER B       │                             │
│  │   Training &     │  │   Visual         │                             │
│  │   Infrastructure │  │   Encoder        │                             │
│  │                  │  │                  │                              │
│  │  • DD-PPO loop   │  │  • ResNet visual │                             │
│  │  • SLURM scripts │  │    encoder       │                             │
│  │  • Blind agent   │  │  • RGB-D obs     │                             │
│  │    training      │  │    pipeline      │                             │
│  │  • Checkpoint    │  │  • Uniform agent │                             │
│  │    management    │  │    on Gibson     │                             │
│  │  • Cluster ops   │  │  • Matched-      │                             │
│  │                  │  │    compute agent │                             │
│  └─────────────────┘  └─────────────────┘                              │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐                              │
│  │   MEMBER C       │  │   MEMBER D       │                             │
│  │   Foveation      │  │   Probing &      │                             │
│  │   Module         │  │   Analysis       │                             │
│  │                  │  │                  │                              │
│  │  • Foveation     │  │  • Hidden state  │                             │
│  │    transform for │  │    extraction    │                             │
│  │    Habitat RGB   │  │  • Occupancy,    │                             │
│  │  • Gaze action   │  │    position,     │                             │
│  │    head + policy │  │    collision     │                             │
│  │  • Gaze-aware    │  │    probes        │                             │
│  │    memory arch.  │  │  • Visualization │                             │
│  │  • Foveated      │  │    + statistics  │                             │
│  │    agent train   │  │  • Paper figures │                             │
│  └─────────────────┘  └─────────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### File Ownership

Each member owns a set of files to avoid merge conflicts:

| Member | Owns | Branch |
|--------|------|--------|
| **A** | `submit_habitat.sh`, `habitat_configs/`, cluster scripts | `feat/training-infra` |
| **B** | Visual encoder integration in `habitat-baselines/` policy | `feat/visual-encoder` |
| **C** | `src/envs/foveation.py`, Habitat foveation wrapper, gaze head | `feat/foveation` |
| **D** | `scripts/probe.py`, `scripts/visualize.py`, `analysis/` | `feat/probing` |

### Shared Interfaces (Contracts)

All members develop against these shared formats — no need to wait for each other:

```
Checkpoint format (produced by A/B/C, consumed by D):
  {
      "state_dict":  model.state_dict(),
      "config":      { ... },       # full training config
      "timesteps":   int,           # total env steps
  }

Hidden state shape (produced during rollout, consumed by probing):
  hidden_states:  (N, num_layers × hidden_size)   # e.g., (50000, 1536) for 3-layer LSTM-512
  agent_pos:      (N, 3)                           # x, y, z world coords
  occupancy:      (N, H, W)                        # top-down binary map
  collision:      (N,)                             # bool per step

Observation pipeline (B and C must produce compatible obs):
  Uniform:   (B, 256, 256, 3) uint8 RGB   → ResNet50 encoder
  Foveated:  (B, 256, 256, 3) uint8 RGB   → foveation_transform() → ResNet50 encoder
  Blind:     (B, pointgoal_dim) float32    → VectorEncoder (no visual)
```

### Development & Testing Independence

Each member can test their component without waiting for others:

| Member | Can test with | Without needing |
|--------|--------------|-----------------|
| **A** | Habitat test scenes (89 MB, free) | B, C, D |
| **B** | Habitat test scenes + pretrained ResNet weights | A, C, D |
| **C** | Any RGB tensor (even random noise for foveation transform) | A, B, D |
| **D** | Synthetic hidden states (`torch.randn(N, 1536)`) | A, B, C |

### Timeline

```
Week 1─2   ████████████████████████████████████████  All 4 develop in parallel
                                                      A: blind training on Gibson
                                                      B: visual encoder integration
                                                      C: foveation for Habitat RGB
                                                      D: probing pipeline + mock tests

Week 3     ████████████████████████████████████████  Integration
                                                      Merge branches → main
                                                      Run sighted + foveated training

Week 4─5   ████████████████████████████████████████  Training runs
                                                      A monitors cluster jobs
                                                      D starts probing blind agent

Week 5─6   ████████████████████████████████████████  Probing & analysis
                                                      D probes all 4 conditions
                                                      All: compare memory content

Week 7─8   ████████████████████████████████████████  Paper writing
                                                      All contribute sections

Week 9─10  ████████████████████████████████████████  Revision + submission
```

---

## References

- Wijmans et al., "Emergence of Maps in the Memories of Blind Navigation Agents," ICLR 2023. [Paper](https://arxiv.org/abs/2301.13261) | [Code](https://github.com/erikwijmans/emergence-of-maps)
- Wijmans et al., "DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames," ICLR 2020. [Paper](https://arxiv.org/abs/1911.00357)
- Savva et al., "Habitat: A Platform for Embodied AI Research," ICCV 2019. [Paper](https://arxiv.org/abs/1904.01201)
