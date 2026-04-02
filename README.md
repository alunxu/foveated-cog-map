# Agentic Cognitive Maps
## How Foveation Shapes Memory Content in Navigation Agents

**CS503 Visual Intelligence — EPFL, Spring 2026**

> When a visual agent navigates through a foveated sensor, what does it choose to remember?

---

## What This Project Is About

Wijmans et al. (ICLR 2023) made a surprising discovery: navigation agents that receive **no visual input at all** (only GPS+Compass) develop internal spatial maps in their recurrent memory. These "blind" agents learn to remember room layouts, their own position, and wall locations — all detectable by probing their LSTM hidden states.

We extend this finding by asking: **what happens when you give the agent biological vision?**

Human eyes are foveated — sharp at the center of gaze, blurry in the periphery. When an agent navigates with such a sensor, it must decide where to look and what to remember. We compare the memory content of four types of agents:

| Agent | What it "sees" | The question |
|-------|---------------|-------------|
| **Blind** | Nothing (GPS+Compass only) | Does it build spatial maps? (replication) |
| **Uniform** | Full-resolution camera | Does vision change what it remembers? |
| **Foveated** | Sharp center + blurry periphery | Does selective attention change memory? |
| **Matched-compute** | Low-res uniform camera | Is it about WHERE info is, or HOW MUCH? |

## Project Structure

```
Project/
├── cfgs/                           # MiniGrid experiment configs
│   ├── base.yaml
│   ├── blind.yaml
│   ├── foveated.yaml
│   ├── uniform.yaml
│   └── matched_compute.yaml
│
├── habitat_configs/                # Habitat DD-PPO configs
│   ├── ddppo_pointnav_blind.yaml
│   └── ddppo_pointnav_blind_gibson.yaml
│
├── src/
│   ├── envs/
│   │   ├── foveation.py            # Eccentricity-dependent blur
│   │   ├── nav_env.py              # MiniGrid navigation wrapper
│   │   └── wrappers.py             # PointGoal / Blind / Foveated / Uniform / Matched
│   ├── models/
│   │   ├── __init__.py             # NavigationAgent (encoder → GRU → policy)
│   │   ├── encoder.py              # CNNEncoder + VectorEncoder
│   │   ├── memory.py               # GRU recurrent memory
│   │   └── policy.py               # Action + gaze policy head
│   └── training/
│       ├── ppo.py                  # PPO algorithm
│       └── rollout.py              # Rollout buffer
│
├── scripts/
│   ├── train.py                    # MiniGrid training
│   ├── probe.py                    # Linear probing analysis
│   └── cluster/                    # SLURM & cluster scripts
│       ├── setup_env.sh
│       ├── submit_job.sh           # MiniGrid training
│       ├── submit_habitat.sh       # Habitat DD-PPO training
│       ├── submit_all.sh           # All 4 MiniGrid conditions
│       ├── submit_probe.sh         # Probing analysis
│       ├── submit_all_probes.sh    # Probe all agents
│       ├── sync_to_cluster.sh      # Upload code → SCITAS
│       └── sync_from_cluster.sh    # Download results ← SCITAS
│
└── Cluster_Tutorial/               # SCITAS + gnoto guides
```

## Environments

| Environment | Purpose | Scenes |
|-------------|---------|--------|
| **Habitat + Gibson** | Main results — photorealistic 3D buildings | 492 real scanned buildings |
| **MiniGrid FourRooms** | Pipeline validation / appendix ablation | 19×19 procedural grid |

## Datasets

| Dataset | Size | Access |
|---------|------|--------|
| **Gibson (Habitat trainval)** | 13 GB | [License form](http://gibsonenv.stanford.edu/database/) |
| **Matterport3D** | ~15 GB | [Access agreement](https://niessner.github.io/Matterport/) |
| **PointNav episodes (Gibson v1)** | 385 MB | [Direct download](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) |

---

## Team & Workstreams

Four members work in parallel. Each builds one piece of the pipeline and can test independently using mock data.

### What each member does (in plain English)

```
 Member A — "Make the agent learn to walk around buildings"
 ──────────────────────────────────────────────────────────
   Trains the blind agent (GPS+Compass only) to navigate in
   hundreds of real 3D buildings using DD-PPO on 4 GPUs.
   Manages cluster jobs, checkpoints, and training infra.

 Member B — "Give the agent eyes"
 ──────────────────────────────────────────────────────────
   Adds a visual system (ResNet CNN) that processes RGB-D camera
   images. Trains the sighted agent on the same buildings so we
   can compare: does vision change what the agent remembers?

 Member C — "Make the eyes biological"
 ──────────────────────────────────────────────────────────
   Replaces the perfect camera with human-like foveated vision
   (sharp center, blurry periphery). The agent must also learn
   WHERE to look. This is the core novelty of the project.

 Member D — "Read the agent's mind"
 ──────────────────────────────────────────────────────────
   Probes all trained agents' memory (LSTM hidden states) to
   ask: does it secretly know the room layout? Its position?
   Where walls are? Compares across all 4 conditions.
```

### Parallel development

All members work independently during Weeks 1-2. The only sync point is Week 3 (C wraps B's encoder with foveation):

```
Week 1─2: Fully parallel (no dependencies)
───────────────────────────────────────────────────────────────

  Member A              Member B              Member C              Member D
  (Training Infra)      (Visual Encoder)      (Foveation)           (Probing)
       │                      │                     │                     │
  DD-PPO scripts,        ResNet50 +            Blur transform +     Probe classifiers
  blind agent on         RGB-D pipeline        gaze action head     + visualization
  Gibson
       │                      │                     │                     │
  Tests with:            Tests with:           Tests with:          Tests with:
  Habitat test scenes    Habitat test scenes   torch.randn(...)     torch.randn(...)
  (free, 89 MB)          + pretrained ResNet   (any RGB tensor)     (fake hidden states)


Week 3: Integration (only sync point)
───────────────────────────────────────────────────────────────

                     B (visual encoder) ──► C (wraps with foveation)
                              │
                              ▼
                     Merge into main
                     Start sighted + foveated training


Week 4+: Training ──► Probing
───────────────────────────────────────────────────────────────

  A monitors cluster ─── checkpoints ──────► D probes all 4 conditions
```

### File ownership

| Member | Owns | Branch |
|--------|------|--------|
| **A** | `scripts/cluster/`, `habitat_configs/` | `feat/training-infra` |
| **B** | Visual encoder in `habitat-baselines/` | `feat/visual-encoder` |
| **C** | `src/envs/foveation.py`, Habitat foveation wrapper | `feat/foveation` |
| **D** | `scripts/probe.py`, `scripts/visualize.py`, `analysis/` | `feat/probing` |

### Shared interface contracts

All members develop against these formats:

```python
# Checkpoint (produced by A/B/C, consumed by D)
checkpoint = {
    "state_dict":  model.state_dict(),
    "config":      {...},
    "timesteps":   int,
}

# Hidden states for probing (produced during rollout, consumed by D)
probe_data = {
    "hidden_states": np.array,  # (N, num_layers * hidden_size)
    "agent_pos":     np.array,  # (N, 3)
    "occupancy":     np.array,  # (N, H, W)
    "collision":     np.array,  # (N,)
}
```

---

## Quick Start

### MiniGrid (local)

```bash
bash scripts/cluster/setup_env.sh
python scripts/train.py --config cfgs/blind.yaml
python scripts/probe.py --checkpoint outputs/blind_agent/checkpoint_final.pt \
    --config cfgs/blind.yaml --n_episodes 200 --output probing_results/blind/
```

### Habitat (SCITAS cluster)

```bash
# One-time setup (on cluster)
conda create -n habitat python=3.9 cmake=3.22 -y && conda activate habitat
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat
cd ~/habitat-lab && pip install -e habitat-lab && pip install -e habitat-baselines

# Train blind agent (4 GPUs)
sbatch scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_blind_gibson
```

### Cluster storage

| What | Where | Notes |
|------|-------|-------|
| Code | `/home/<user>/CS503_Project/` | 100 GB, backed up |
| Data + checkpoints | `/scratch/izar/<user>/` | Large, files >2 weeks deleted |

---

## Current Status

- [x] MiniGrid pipeline complete (training + probing, 4 conditions)
- [x] Habitat installed on SCITAS (habitat-sim 0.3.3 + habitat-baselines)
- [x] Gibson dataset downloaded (492 scenes, 13 GB)
- [ ] **Blind agent training on Gibson** (in progress, 4x V100)
- [ ] Sighted agent training on Gibson
- [ ] Foveation module for Habitat
- [ ] Probing analysis on Habitat hidden states
- [ ] Paper writing

## References

- Wijmans et al., "Emergence of Maps in the Memories of Blind Navigation Agents," ICLR 2023. [Paper](https://arxiv.org/abs/2301.13261) | [Code](https://github.com/erikwijmans/emergence-of-maps)
- Wijmans et al., "DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames," ICLR 2020. [Paper](https://arxiv.org/abs/1911.00357)
- Savva et al., "Habitat: A Platform for Embodied AI Research," ICCV 2019. [Paper](https://arxiv.org/abs/1904.01201)
