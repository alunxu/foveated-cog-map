# What Do Foveated Agents Remember?
## How Perceptual Uncertainty Shapes the Content of Learned Cognitive Maps

**CS503 Visual Intelligence — EPFL, Spring 2026**

> When a visual agent navigates through a foveated sensor, what does it choose to remember?

---

## 1. Project Overview

### The Big Picture

Imagine a robot navigating inside a building. It uses a recurrent neural network (LSTM) as its "brain" — at each step, it reads sensor input, updates its hidden state, and decides what to do next. That hidden state is the agent's **working memory**.

Wijmans et al. (ICLR 2023) discovered something surprising: even a **blind** agent (receiving only GPS+Compass, no camera) develops **spatial maps inside its memory**. By probing the LSTM hidden states with simple linear classifiers, they showed the agent secretly encodes room layouts, its own position, and wall locations — all without ever "seeing" anything.

Our project extends this by asking: **what happens when you give the agent eyes — specifically, human-like foveated eyes?**

### Why Foveation?

Human vision is not uniform. We see sharply only at the center of gaze (the **fovea**, ~2 degrees), while the periphery is progressively blurred. This forces us to actively choose **where to look**, creating a tight loop between attention, perception, and memory.

We hypothesize that an agent with foveated vision will develop qualitatively different spatial representations than one with a perfect camera — because it must selectively attend to parts of the scene and remember what it cannot currently see clearly.

### Four Experimental Conditions

We train four types of navigation agents and compare what their memories encode:

| Agent | Input | What We Test |
|-------|-------|-------------|
| **Blind** | GPS+Compass only (4D vector) | Baseline: do spatial maps emerge without any vision? (replication of Wijmans) |
| **Uniform** | Full-resolution RGB camera (128x128) | Does adding uniform vision change memory content? |
| **Foveated** | RGB with eccentricity-dependent blur + learned gaze | Does selective attention reshape what gets remembered? |
| **Matched-Compute** | Low-resolution RGB (64x64) | Control: is it about WHERE info is (foveation) or HOW MUCH total info? |

The **blind vs. uniform** comparison tells us whether vision changes memory. The **uniform vs. foveated** comparison tells us whether selective attention matters. The **foveated vs. matched-compute** comparison isolates the effect of spatial attention from total information bandwidth.

### Hypotheses

**H1 (Uncertainty-driven memory):** A foveated agent's memory over-represents peripherally-observed (high-uncertainty) regions compared to a uniform-vision agent. Memory content tracks *what was poorly seen* rather than *what was clearly seen*, because uncertain regions are where memory adds the most decision-relevant information.

**H2 (Confidence-annotated maps):** The foveated agent's hidden state encodes not just spatial layout but *location-dependent confidence* — a "belief map" rather than a plain map. Linear probes should be able to decode both "what is at location X" and "how well did I see location X" from the hidden state.

**H3 (Gaze-memory coupling):** The foveated agent's gaze strategy and memory content are jointly adapted — the agent looks at regions its memory is uncertain about, and remembers regions it hasn't yet foveated. This coupling should be absent in the uniform-vision agent.

Each hypothesis is testable within a semester and produces interesting results whether confirmed or refuted.

### Ecological Motivation

This connects to ecological vision and active perception. In ecology, different species have different eye designs adapted to their environmental niche. An animal with a narrow fovea and wide peripheral field faces a fundamentally different memory problem than one with uniform vision — it must remember what it *couldn't see well*, not just what it *did see*. Our project tests whether this principle emerges in learned agents.

---

## 2. Methods

### 2.1 Task: PointGoal Navigation

The agent must navigate from a random start to a goal location specified as a relative vector (distance + direction). The agent receives:
- **Pointgoal**: 4D vector (distance, sin/cos of angle to goal) — all conditions
- **RGB image**: first-person camera view — sighted conditions only

Actions: `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT` (+ `STOP` in Habitat)

### 2.2 Agent Architecture

All four conditions share the same recurrent backbone for fair comparison:

```
Observation → Encoder → [concatenate with pointgoal] → LSTM (3 layers, 512-d) → Policy Head
                                                            ↑
                                                    This hidden state is what we probe
```

| Component | Blind | Uniform | Foveated | Matched |
|-----------|-------|---------|----------|---------|
| Visual encoder | None | ResNet18 | ResNet18 + foveation | ResNet18 |
| Input resolution | N/A | 128x128 | 128x128 (foveated) | 64x64 |
| Pointgoal encoder | Linear(4→128) | Linear(4→128) | Linear(4→128) | Linear(4→128) |
| LSTM | 3 layers, 512-d | 3 layers, 512-d | 3 layers, 512-d | 3 layers, 512-d |
| Policy head | Categorical | Categorical | Categorical + gaze decoder | Categorical |

### 2.3 Foveation Transform

Our foveation simulates biological vision with eccentricity-dependent blur:

```
Given gaze position (gx, gy):
  1. Compute per-pixel distance from gaze center
  2. Map distance → eccentricity ∈ [0, 1] (0 inside fovea, 1 at corners)
  3. Apply falloff: sigma = eccentricity^2 × sigma_max  (quadratic)
  4. Multi-scale Gaussian blur: interpolate between 5 pre-computed blur levels
```

Parameters: `fovea_radius=16px`, `blur_sigma_max=6.0`, `falloff=quadratic`

The foveated agent's **gaze position is learned**, not fixed. A small MLP decodes gaze (x, y) from the previous LSTM hidden state:

```
Previous LSTM hidden → MLP(512→64→2) → Sigmoid → gaze ∈ [0,1]²
```

This means the agent **decides where to look based on what it remembers**, then processes the foveated view through the visual encoder. Gradients flow end-to-end: navigation loss → LSTM → gaze decoder → foveation → visual features.

### 2.4 Training Algorithm

- **MiniGrid**: PPO (Proximal Policy Optimization), 2M environment steps
- **Habitat**: DD-PPO (Decentralized Distributed PPO), 500M environment steps on 2 V100 GPUs
  - DD-PPO runs multiple independent PPO workers that periodically sync gradients
  - Each worker manages multiple parallel environments (8 for blind, 6 for sighted)

### 2.5 Linear Probing (Reading the Agent's Mind)

After training, we **freeze** the agent's weights and collect LSTM hidden states across many episodes. We then train simple linear classifiers/regressors to decode spatial information:

| Probe | Target | Question | Tests |
|-------|--------|----------|-------|
| **Occupancy grid** | Binary grid of walls vs. free space | Does memory encode the room layout? | All conditions |
| **Agent position** | (x, y) coordinates | Does memory encode where the agent is? | All conditions |
| **Collision prediction** | Will the next step hit a wall? | Are there "collision neurons" in memory? | All conditions |
| **Perceptual uncertainty** | Per-location observation quality | Does memory encode *how well* each area was seen? | Foveated only (H2) |
| **Target location** | Goal position relative to agent | Does memory encode where the goal is? | All conditions |

If a linear probe can decode this information, it means the LSTM has learned a **linear representation** of spatial structure — a cognitive map.

### 2.6 Gaze-Memory Coupling Analysis (H3)

Beyond probing *what* the memory encodes, we analyze *how gaze and memory interact*:

1. **Information-gain gaze analysis**: Does the agent's gaze direction correlate with memory uncertainty? We compare the learned gaze policy to an information-gain-maximizing Bayesian ideal observer — does the agent look where its memory is weakest?
2. **Perceptual uncertainty tracking**: Using the `FoveationTransform.get_uncertainty_map()` method, we compute a per-location uncertainty map from the agent's gaze history. We then test whether this uncertainty is decodable from the hidden state (H2).
3. **Gaze fixation patterns**: Visualize where the foveated agent looks over time — does it develop systematic scanning patterns (e.g., looking at doorways, obstacles, the goal direction)?

### 2.7 Ablations

To understand which factors drive our results, we plan the following ablation experiments:

| Ablation | What We Vary | What We Learn |
|----------|-------------|---------------|
| **Foveation strength** | `blur_sigma_max`: 2.0, 4.0, 6.0, 8.0 | How much blur is needed to change memory content? |
| **Memory capacity** | LSTM hidden size: 128, 256, 512 | Do smaller memories show stronger uncertainty-driven effects? |
| **Episode length** | Max steps: 250, 500, 1000 | Does memory pressure change what gets remembered? |
| **Gaze ablation** | Fixed-center vs. random vs. learned gaze | Is learned gaze necessary, or does any foveation suffice? |

---

## 3. Environments & Datasets

### 3.1 MiniGrid FourRooms (Proof-of-Concept)

A simple 19x19 procedural grid world with 4 rooms connected by doorways. Used for rapid prototyping and pipeline validation.

- **Observation**: 7x7 egocentric grid (limited view — vision is nearly useless here)
- **Training**: ~2 hours per condition on 1 GPU
- **Purpose**: Validate the "maps in memory" finding for blind agents; appendix material
- **Limitation**: Egocentric view is too small for meaningful vision comparisons — Habitat is essential for the main claim

### 3.2 Habitat + Gibson (Main Results)

Habitat is a high-performance 3D simulation platform. Gibson provides 492 real-world 3D-scanned buildings for photorealistic navigation.

- **Observation**: First-person RGB camera (128x128 for uniform/foveated, 64x64 for matched)
- **Training**: ~5-7 days per condition on 2 V100 GPUs (500M steps)
- **Purpose**: The real experimental testbed. High-fidelity vision makes the foveation comparison meaningful
- **Task episodes**: Pre-computed PointNav episodes from Facebook/Meta (385 MB)

### 3.3 Dataset Access

| Dataset | Size | How to Get It |
|---------|------|---------------|
| **Gibson scenes (trainval)** | 13 GB | Apply at [gibsonenv.stanford.edu](http://gibsonenv.stanford.edu/database/) — approval usually within 24h |
| **Matterport3D** | ~15 GB | Apply at [niessner.github.io/Matterport](https://niessner.github.io/Matterport/) — may take days |
| **PointNav episodes (Gibson v1)** | 385 MB | [Direct download](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) (no license needed) |
| **Habitat test scenes** | 89 MB | `python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes` (free) |

**Only one team member needs to download Gibson/MP3D.** The data lives on the shared cluster scratch space.

---

## 4. Project Structure

```
Project/
├── cfgs/                               # MiniGrid experiment configs (Phase 1)
│   ├── base.yaml                       #   Shared defaults
│   ├── blind.yaml                      #   Blind: VectorEncoder only
│   ├── uniform.yaml                    #   Uniform: CNN on 64x64
│   ├── foveated.yaml                   #   Foveated: CNN + gaze control
│   └── matched_compute.yaml            #   Matched: CNN on 32x32
│
├── habitat_configs/                    # Habitat DD-PPO configs (Phase 2)
│   ├── ddppo_pointnav_blind_gibson.yaml
│   ├── ddppo_pointnav_uniform_gibson.yaml
│   ├── ddppo_pointnav_foveated_gibson.yaml
│   └── ddppo_pointnav_matched_gibson.yaml
│
├── src/
│   ├── envs/
│   │   ├── foveation.py                # FoveationTransform (numpy, for MiniGrid)
│   │   ├── nav_env.py                  # MiniGrid navigation environment
│   │   └── wrappers.py                 # Blind / Uniform / Foveated / Matched wrappers
│   ├── models/
│   │   ├── __init__.py                 # NavigationAgent (encoder → GRU → policy)
│   │   ├── encoder.py                  # CNNEncoder + VectorEncoder
│   │   ├── memory.py                   # GRU/LSTM recurrent memory
│   │   └── policy.py                   # Action + gaze policy head
│   ├── habitat/
│   │   ├── __init__.py                 # Registers custom policies with habitat-baselines
│   │   ├── torch_foveation.py          # TorchFoveationTransform (GPU, for Habitat)
│   │   └── foveated_policy.py          # FoveatedPointNavResNetPolicy (DD-PPO compatible)
│   └── training/
│       ├── ppo.py                      # PPO algorithm
│       └── rollout.py                  # Rollout buffer
│
├── scripts/
│   ├── train.py                        # MiniGrid training entry point
│   ├── probe.py                        # Linear probing analysis
│   └── cluster/                        # SLURM & cluster management
│       ├── setup_env.sh                # Conda environment setup
│       ├── submit_job.sh               # Submit one MiniGrid condition
│       ├── submit_habitat.sh           # Submit one Habitat DD-PPO job
│       ├── submit_all.sh               # Submit all 4 MiniGrid conditions
│       ├── submit_probe.sh             # Submit probing for one checkpoint
│       ├── submit_all_probes.sh        # Submit probing for all conditions
│       ├── run_habitat.py              # Custom entry point (registers foveated policy)
│       ├── sync_to_cluster.sh          # Upload code → SCITAS
│       └── sync_from_cluster.sh        # Download results ← SCITAS
│
└── Cluster_Tutorial/                   # SCITAS usage guides
```

---

## 5. Possible Methods & Alternatives

Throughout the project, there are several design choices where we picked one approach but could explore alternatives:

### Visual Encoder
- **Current**: ResNet18 (lightweight, fits 2-GPU budget)
- **Alternative**: ResNet50 (Wijmans used this; more capacity but slower/more memory)
- **Alternative**: ViT (Vision Transformer) — would change the nature of visual processing entirely

### Foveation Model
- **Current**: Multi-scale Gaussian blur with quadratic falloff, learned gaze
- **Alternative**: Log-polar transform (warps image like biological retina)
- **Alternative**: Attention-based masking (zero out periphery instead of blurring)
- **Alternative**: Stochastic gaze (sample gaze from a distribution with entropy bonus — we currently use deterministic gaze)

### Gaze Control
- **Current**: Deterministic from LSTM hidden (MLP decoder with sigmoid)
- **Alternative**: Stochastic with separate PPO loss (add gaze log-prob to action log-prob)
- **Alternative**: Saliency-based (pre-computed saliency map, no learned gaze)
- **Alternative**: Random gaze (ablation — does learned gaze matter?)

### Probing Method
- **Current**: Linear probes (logistic regression / ridge regression)
- **Alternative**: MLP probes (more capacity, but harder to interpret as "linear readout")
- **Alternative**: CKA (Centered Kernel Alignment) to compare representation similarity across conditions
- **Alternative**: Representational dissimilarity matrices (RDM) for fine-grained comparison

### Training Algorithm
- **Current**: DD-PPO (distributed PPO)
- **Alternative**: IMPALA, R2D2 (off-policy methods — potentially more sample efficient)

---

## 6. Team & Workstreams

Four members work in parallel on independent modules, meeting at integration points:

```
 Member A — "Make the agent learn to walk around buildings"
   Trains the blind agent (GPS+Compass only) to navigate in 492 real 3D
   buildings using DD-PPO. Manages cluster jobs, checkpoints, and training.

 Member B — "Give the agent eyes"
   Adds a visual system (ResNet18) that processes RGB camera images.
   Trains the sighted (uniform) and matched-compute agents.

 Member C — "Make the eyes biological"
   Implements foveated vision (sharp center, blurry periphery) with a
   learned gaze controller. This is the core novelty of the project.

 Member D — "Read the agent's mind"
   Probes all trained agents' LSTM hidden states to decode spatial
   information (occupancy, position, uncertainty). Tests hypotheses
   H1-H3 with information-theoretic analysis and gaze-memory coupling.
```

### Parallel Development Timeline

```
Phase 1 (Weeks 1-2): Independent development
───────────────────────────────────────────────

  Member A              Member B              Member C              Member D
  Train blind agent     Build visual          Build foveation       Build probing
  on Gibson (DD-PPO)    encoder pipeline      transform + gaze      classifiers +
                        for Habitat           controller            visualization

  Tests with:           Tests with:           Tests with:           Tests with:
  Habitat test scenes   Habitat test scenes   torch.randn(...)      torch.randn(...)


Phase 2 (Week 3): Integration
───────────────────────────────────────────────

  B's visual encoder ──► C wraps with foveation ──► merged into main
                                                    Start training all sighted conditions


Phase 3 (Weeks 4-6): Training → Probing → Paper
───────────────────────────────────────────────

  A monitors cluster ──► checkpoints ──► D probes all 4 conditions
                                         Compare memory representations
                                         Write paper
```

### Shared Interface Contract

All members develop against these data formats:

```python
# Checkpoint format (produced by A/B/C, consumed by D)
checkpoint = {
    "state_dict":  model.state_dict(),
    "config":      {...},
    "timesteps":   int,
}

# Probing data (produced during evaluation rollouts, consumed by D)
probe_data = {
    "hidden_states":    np.array,   # (N, num_layers * hidden_size)
    "agent_pos":        np.array,   # (N, 2) or (N, 3)
    "occupancy":        np.array,   # (N, H, W)
    "collision":        np.array,   # (N,) binary
    "target_pos":       np.array,   # (N, 2) goal location
    "uncertainty_map":  np.array,   # (N, H, W) perceptual uncertainty (foveated only)
    "gaze_history":     np.array,   # (N, 2) gaze positions (foveated only)
}
```

---

## 7. Quick Start

### MiniGrid (local, for development)

```bash
# Setup
bash scripts/cluster/setup_env.sh
conda activate cs503_project

# Train one condition
python scripts/train.py --config cfgs/blind.yaml

# Run probing
python scripts/probe.py --checkpoint outputs/blind_agent/checkpoint_final.pt \
    --config cfgs/blind.yaml --n_episodes 200 --output probing_results/blind/
```

### Habitat (SCITAS cluster)

```bash
# One-time environment setup
conda create -n habitat python=3.9 cmake=3.22 -y && conda activate habitat
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat
cd ~/habitat-lab && pip install -e habitat-lab && pip install -e habitat-baselines

# Train (2 V100 GPUs per job, cs-503 QoS allows max 2 concurrent jobs)
sbatch scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_blind_gibson
sbatch scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_uniform_gibson
# After above finish:
sbatch scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_foveated_gibson
sbatch scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_matched_gibson
```

### Cluster Storage

| What | Path | Notes |
|------|------|-------|
| Code | `/home/<user>/CS503_Project/` | 100 GB, backed up |
| Datasets | `/scratch/izar/<user>/habitat_data/` | Gibson scenes live here |
| Checkpoints | `/scratch/izar/<user>/habitat_checkpoints/` | Training outputs |

**Important**: `/scratch/izar/` files older than 2 weeks are auto-deleted. Copy important checkpoints to `/home/`.

---

## 8. Current Status & Results

### Phase 1: MiniGrid (Complete)

| Agent | Final Success Rate | SPL | Notes |
|-------|-------------------|-----|-------|
| **Blind** | **70%** | 0.194 | Strong replication of Wijmans finding |
| Uniform | 0% | 0.000 | Expected: 7x7 egocentric view too small for useful vision |
| Matched-Compute | 10% | 0.047 | Partial learning with real 32x32 downsampling |
| Foveated | N/A | N/A | Hit wall-time limit at 60%; gaze stayed random (no signal in MiniGrid) |

**Key insight**: MiniGrid validates that blind agents build spatial maps, but its tiny egocentric view makes vision comparisons meaningless. Habitat (with full first-person RGB) is essential for the foveation study.

### Phase 2: Habitat Gibson (In Progress)

| Agent | Status | Notes |
|-------|--------|-------|
| **Blind** | Training (96% success at 215M/500M steps) | On track to replicate Wijmans |
| Uniform | Config ready, queued | Waiting for blind to finish |
| Foveated | Config + custom policy ready | Needs uniform encoder validated first |
| Matched-Compute | Config ready, queued | Waiting for GPU slot |

---

## 9. Detailed TODO

### Phase 2A: Finish Habitat Blind Training
- [x] Set up Habitat + Gibson on SCITAS
- [x] Download Gibson dataset (492 scenes, 13 GB)
- [x] Create DD-PPO config for blind agent
- [x] Submit blind agent training (job 2823083, 2 GPUs)
- [ ] **Wait for blind training to finish (~April 8)**
- [ ] Verify final metrics match Wijmans (target: >95% success, >0.9 SPL)
- [ ] Save final checkpoint to `/home/` for safety

### Phase 2B: Train Sighted Agents
- [x] Create uniform vision config (`ddppo_pointnav_uniform_gibson.yaml`)
- [x] Create matched-compute config (`ddppo_pointnav_matched_gibson.yaml`)
- [ ] Submit uniform agent training
  - [ ] Monitor first 50M steps for non-zero success
  - [ ] If OOM: reduce `num_environments` from 6 to 4
- [ ] Submit matched-compute agent training
  - [ ] Verify faster per-step throughput (smaller images)
- [ ] Wait for both to complete (~5-7 days each)

### Phase 2C: Train Foveated Agent
- [x] Implement GPU foveation transform (`src/habitat/torch_foveation.py`)
- [x] Implement custom policy (`src/habitat/foveated_policy.py`)
- [x] Create foveated config (`ddppo_pointnav_foveated_gibson.yaml`)
- [x] Create custom entry point (`scripts/cluster/run_habitat.py`)
- [ ] **Test with fixed center gaze first** (disable gaze learning)
  - [ ] Verify foveation doesn't crash training
  - [ ] Compare early success rate with uniform agent
- [ ] Enable gaze learning and train to 500M steps
  - [ ] Monitor gaze patterns — does the agent learn to look at useful things?
  - [ ] Log gaze heatmaps to TensorBoard
- [ ] If gaze stays random: try stochastic gaze with entropy bonus

### Phase 3: Probing & Analysis
- [ ] Build Habitat probing pipeline (extract LSTM hidden states during evaluation)
  - [ ] Run each trained agent on 1000+ evaluation episodes
  - [ ] Save hidden states + ground-truth spatial info at each step
  - [ ] For foveated agent: also save gaze history + uncertainty maps
- [ ] **Standard probes** (all 4 conditions):
  - [ ] Occupancy grid decoder (is room layout in memory?)
  - [ ] Agent position decoder (does it know where it is?)
  - [ ] Collision predictor (are there collision neurons?)
  - [ ] Target location decoder (does it remember where the goal is?)
- [ ] **H1 test: Uncertainty-driven memory**
  - [ ] For foveated agent: split probing by eccentricity — is memory more accurate for peripherally-seen vs. foveated regions?
  - [ ] Compare to uniform agent: does foveated agent over-represent uncertain regions?
- [ ] **H2 test: Confidence-annotated maps**
  - [ ] Train probe to decode per-location perceptual uncertainty from hidden state
  - [ ] Does the agent encode not just "what is there" but "how well did I see it"?
- [ ] **H3 test: Gaze-memory coupling**
  - [ ] Compute correlation between gaze direction and memory uncertainty
  - [ ] Compare to Bayesian ideal observer (information-gain-maximizing gaze)
  - [ ] Visualize gaze fixation patterns (doorways? obstacles? goal direction?)
- [ ] **Cross-condition comparison**:
  - [ ] Do sighted agents have MORE spatial info than blind? Or different?
  - [ ] Does foveation create SELECTIVE memory (sharp near gaze, fuzzy elsewhere)?
  - [ ] Matched-compute ablation: is it total info or spatial attention?
- [ ] **Ablations** (if time permits):
  - [ ] Vary foveation strength (blur_sigma_max: 2, 4, 6, 8)
  - [ ] Vary memory capacity (hidden_size: 128, 256, 512)
  - [ ] Gaze ablation: fixed-center vs. random vs. learned
- [ ] **Visualization**:
  - [ ] Decoded occupancy maps overlaid on ground truth
  - [ ] Gaze trajectory plots for foveated agent
  - [ ] Uncertainty maps vs. memory accuracy heatmaps
  - [ ] Comparison bar charts across all 4 conditions

### Phase 4: MiniGrid Probing (Appendix)
- [ ] Run probing on 3 completed MiniGrid checkpoints (blind, uniform, matched)
  - [ ] Submit probing jobs via `submit_all_probes.sh`
- [ ] Include as supplementary material (validates pipeline, shows blind>sighted in limited-view setting)

### Phase 5: Paper Writing
- [ ] Introduction: spatial memory in navigation, foveation hypothesis
- [ ] Related work: Wijmans 2023, foveated vision in RL, cognitive maps
- [ ] Methods: 4 conditions, architecture, foveation transform, probing
- [ ] Results: MiniGrid (appendix), Habitat (main), cross-condition comparison
- [ ] Discussion: what foveation changes about memory, implications for embodied AI
- [ ] Figures: architecture diagram, foveation visualization, probing results, gaze patterns

---

## 10. Related Work & How We Differ

| Area | Key Papers | What They Do | What's Different in Our Work |
|------|-----------|-------------|------------------------------|
| **Cognitive maps in RL** | Wijmans et al. (ICLR 2023); Gornet & Thomson (Nature MI 2024) | Show spatial maps emerge in blind/sighted agent memory | We study how *varying perceptual quality* across the visual field shapes memory content |
| **Foveated RL agents** | Pourrahimi & Bashivan (bioRxiv 2025) | Foveated Transformer on visual search; probe for brain-like features | They study static scene search; we study navigation with spatial memory. They ask "what does it look like?" vs. our "what does it remember?" |
| **Optimal gaze in RL** | Radulescu et al. (2022); Zhou & Eckstein (2022) | Show RL agents converge to Bayesian-optimal fixation strategies | They don't probe internal memory — we study what the memory encodes, not just where the agent looks |
| **Sensor design & task perf.** | Atanov et al. (ECCV 2024, VILAB) | Show photoreceptor design critically shapes task performance | We extend from "sensor shapes performance" to "sensor shapes memory content" |
| **Perception-action-memory** | Beker et al. (NeurIPS 2022, PALMER) | Study perception-action loops with memory for planning | We specifically study how foveation constrains what memory encodes |

**The gap we fill**: No existing work studies how foveation — spatially varying perceptual quality — shapes the *content and structure* of cognitive maps in navigation agents. The components exist separately; we combine them.

---

## 11. Conference Extension Path

The course project delivers a clean empirical study with probing results and hypothesis tests. For a venue like NeurIPS/ECCV/ICLR:

- **Method contribution**: If confidence-annotated memory (H2) proves beneficial, design a training objective that explicitly encourages it, and show it improves navigation under sensor degradation or domain shift.
- **Analysis contribution**: Scale to the full Habitat benchmark, test with Transformer memory vs. LSTM, and submit to an analysis-friendly venue (NeurIPS Datasets & Benchmarks, ECCV analysis track).

---

## 12. References

**Core:**
- Wijmans et al., "Emergence of Maps in the Memories of Blind Navigation Agents," ICLR 2023. [Paper](https://arxiv.org/abs/2301.13261) | [Code](https://github.com/erikwijmans/emergence-of-maps)
- Wijmans et al., "DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames," ICLR 2020. [Paper](https://arxiv.org/abs/1911.00357)

**Environments & Datasets:**
- Savva et al., "Habitat: A Platform for Embodied AI Research," ICCV 2019. [Paper](https://arxiv.org/abs/1904.01201)
- Xia et al., "Gibson Env: Real-World Perception for Embodied Agents," CVPR 2018. [Paper](https://arxiv.org/abs/1808.10654)

**Foveation & Active Vision:**
- Pourrahimi & Bashivan, "Probing brain-like features in foveated visual search agents," bioRxiv 2025
- Radulescu et al., "RL search agents converge to Bayesian-optimal fixation policies," 2022
- Zhou & Eckstein, "Deep Q-learning convergence with ideal observer gaze," 2022
- Lettvin, "What the Frog's Eye Tells the Frog's Brain," 1959

**Probing & Representation Analysis:**
- Alain & Bengio, "Understanding intermediate layers using linear classifier probes," ICLR Workshop 2017
- Gornet & Thomson, "Cognitive maps in navigation agents," Nature Machine Intelligence 2024

**VILAB Connections:**
- Atanov et al., "Photoreceptor design shapes task performance," ECCV 2024
- Beker et al., "PALMER: Perception-Action Loop with Memory for Planning," NeurIPS 2022

**Foundations:**
- Tolman, "Cognitive maps in rats and men," Psychological Review 1948
