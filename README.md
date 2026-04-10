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

We train five navigation agents (varying only visual input) and compare what their memories encode:

| Agent | Input | What We Test |
|-------|-------|-------------|
| **Blind** | Non-visual sensor stack only (goal, GPS, compass, close-to-goal) | Baseline: do spatial maps emerge without any vision? (replication of Wijmans) |
| **Uniform** | 256×256 RGB via ResNet-18 | Does adding uniform vision change memory content? |
| **Foveated (fixed)** | 256×256 RGB with eccentricity-dependent Gaussian blur (σ_max=8), gaze locked at centre | Does spatial non-uniformity reshape what gets remembered? |
| **Foveated (learned)** | Same blur, gaze predicted by MLP (~33K params) from LSTM state | Does active gaze amplify the effect? |
| **Matched-Compute** | 48×48 uniform RGB (2,304 pixels, exceeding foveated budget) | Control: is it about spatial distribution or total info volume? |

The **blind vs. uniform** comparison tells us whether vision changes memory. The **uniform vs. foveated** comparison tells us whether spatial non-uniformity matters. The **foveated vs. matched-compute** comparison isolates spatial distribution from total information volume. The **fixed vs. learned gaze** comparison isolates the contribution of active gaze (H3 test).

### Hypotheses

**H1 (Compensatory memory):** Layout-probe accuracy stratified by observation eccentricity: a passive memory predicts falling accuracy; compensatory memory predicts flat or *rising* accuracy in the periphery. A separate confidence probe (inverse cumulative blur σ) tests whether input quality is recorded at all.

**H2 (Representational divergence):** CKA between foveated and uniform agents: low similarity despite matched task performance indicates qualitatively different spatial codes. Cross-heading position-probe generalization tests whether foveated codes are more appearance-invariant.

**H3 (Epistemic gaze):** Correlation of learned gaze coordinates with position-probe error (memory-uncertainty proxy). If the learned-gaze agent shows stronger H1–H2 effects than the fixed-centre variant, fixation is driven by epistemic demands.

Each hypothesis is testable within a semester and produces interesting results whether confirmed or refuted.

### Motivation

Three implications if confirmed:
- **Belief-state emergence without Bayesian machinery.** If pure RL under foveation induces uncertainty tracking in recurrent memory, task pressure alone can discover belief-state-like representations — without explicit probabilistic objectives or architectural priors.
- **Abstract spatial codes from degraded input.** Foveation degrades visual detail within each frame, preventing reliance on appearance-based position encoding. If this forces more abstract, appearance-invariant spatial representations, it identifies input degradation as a sufficient condition for representational abstraction — an information-bottleneck effect driven by sensor structure.
- **Gaze as epistemic action.** If learned gaze targets memory-weak regions, the navigation objective alone suffices for information-gain-maximizing fixation — without auxiliary exploration rewards or hand-designed gaze shaping.

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

| Component | Blind | Uniform | Foveated (fixed/learned) | Matched |
|-----------|-------|---------|----------|---------|
| Visual encoder | None | ResNet-18 | ResNet-18 + foveation | ResNet-18 |
| Input resolution | N/A | 256×256 | 256×256 (foveated, σ_max=8) | 48×48 |
| Non-visual sensors | goal-in-start-frame, GPS, compass, close-to-goal (each → 32-d) | same | same | same |
| LSTM | 3 layers, 512-d | 3 layers, 512-d | 3 layers, 512-d | 3 layers, 512-d |
| Policy head | Categorical | Categorical | Categorical (+ gaze MLP for learned) | Categorical |

### 2.3 Foveation Transform

Our foveation simulates biological vision with eccentricity-dependent blur:

```
Given gaze position (gx, gy):
  1. Compute per-pixel distance from gaze center
  2. Map distance → eccentricity ∈ [0, 1] (0 inside fovea, 1 at corners)
  3. Apply falloff: sigma = eccentricity^2 × sigma_max  (quadratic)
  4. Multi-scale Gaussian blur: interpolate between 5 pre-computed blur levels
```

Parameters: `fovea_radius=16px`, `blur_sigma_max=8.0`, `falloff=quadratic`

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
| **GPS / Compass** | Episodic displacement + heading | Can memory reconstruct its own sensor input? | All (sanity check) |
| **Absolute position** | World (x,z) per scene | Does memory encode where the agent *is*? (Wijmans replication) | All conditions |
| **Distance-to-goal** | Euclidean dist to goal | Does memory encode task-relevant spatial info? | All conditions |
| **Multi-layer comparison** | Same targets, per LSTM layer (h₁/h₂/h₃, c₁/c₂/c₃) | Where does spatial info live in the network? | All conditions |
| **Control task** | Shuffled labels (Hewitt & Liang 2019) | Is probe R² genuine or artifact of expressivity? | All conditions |
| **Accuracy vs. timestep** | GPS R² stratified by step-in-episode | Does spatial encoding improve over time? | H1 test |
| **Cross-heading generalization** | Train position probe on heading A, test on opposite | Are codes allocentric (appearance-invariant)? | H2 test |
| **Path-history (lag-k)** | GPS at t−k decoded from hidden at t | How much trajectory history does memory retain? | H1 (inspired by SPACE route retracing) |
| **Visited-region** | Binary grid of visited cells | Does memory encode spatial working memory? | All (inspired by SPACE CSWM) |
| **Per-unit rate maps** | Spatial information per neuron (bits) | Are there place-cell-like neurons? | All (Banino et al.) |
| **CKA cross-condition** | Representation geometry similarity | Do foveated and uniform agents develop divergent spatial codes? | H2 test |
| **Probe transfer** | Train probe on condition A, test on B | Do conditions share a common spatial code? | H2 test |

If a linear probe can decode this information, it means the LSTM has learned a **linear representation** of spatial structure — a cognitive map.

> **Note:** Path-history and visited-region probes are inspired by Ramakrishnan, Wijmans et al. (ICLR 2025) "Does Spatial Cognition Emerge in Frontier Models?", which benchmarks large-scale spatial cognition (route retracing, shortcut discovery, map sketching) and visuospatial working memory in frontier models. Their SPACE benchmark finds that disembodied models fail at spatial cognition — motivating our study of *embodied* agents where spatial representations demonstrably emerge.

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

Habitat is a high-performance 3D simulation platform. Gibson provides ~490 real-world 3D-scanned buildings for photorealistic navigation. **Train on a merged Gibson-0+ (411) ∪ MP3D-train (61) = 472-scene pool, evaluate on Matterport3D test (18 scenes, 1008 episodes)** — matching Wijmans et al. 2023 Appendix A.1 ("train on 411 Gibson + 72 MP3D scenes").

- **Observation**: First-person RGB camera (256×256 for uniform/foveated, 48×48 for matched)
- **Training**: ~5-7 days per condition on 2 V100 GPUs (500M steps)
- **Purpose**: The real experimental testbed. High-fidelity vision makes the foveation comparison meaningful
- **Training episodes**: Gibson-0+ (`pointnav_gibson_0_plus_v1.zip`, 320 MB, 411 scenes) **merged with** MP3D train (`pointnav_mp3d_v1.zip` → `train/`, 61 scenes available) via symlinks under `data/datasets/pointnav/mp3d_gibson/v1/train/`. Configs point at `data_path: data/datasets/pointnav/mp3d_gibson/v1/{split}/{split}.json.gz`.
- **Evaluation episodes**: Matterport3D test split (18 scenes, 56 eps each) — held out entirely from training

#### Gibson scene-count naming (easy to get confused)

Every Gibson scene has a quality rating 0–5 assigned by Stanford. "Gibson-N+" means "all scenes with quality ≥ N", nested subsets of the same ~572-scene dataset:

| Filter | Scenes | Pre-built episodes? | Used where |
|---|---|---|---|
| Gibson-4+ | 72 train + 14 val | `pointnav_gibson_v1.zip` | Standard Habitat benchmark; the emergence-of-maps public code release |
| **Gibson-0+** | **411 train** | **`pointnav_gibson_0_plus_v1.zip`** | **Our training set** — largest public option |
| Gibson-2+ | 572 train (per Wijmans 2020 paper text) | ❌ not distributed | Would need to be regenerated from raw `.glb` files |

We use Gibson-0+ because it's the largest publicly pre-built Gibson PointNav episode set — 5.7× more scene diversity than the default, with no episode-generation pipeline required. See `habitat_configs/gibson_0plus_scenes.txt` for the authoritative 411-scene list.

### 3.3 Dataset Access

| Dataset | Size | How to Get It |
|---------|------|---------------|
| **Gibson scenes (trainval)** | 13 GB | Apply at [gibsonenv.stanford.edu](http://gibsonenv.stanford.edu/database/) — approval usually within 24h |
| **Matterport3D** | ~15 GB | Apply at [niessner.github.io/Matterport](https://niessner.github.io/Matterport/) — may take days |
| **PointNav episodes (Gibson-0+)** | 320 MB | `bash scripts/cluster/download_gibson_0plus.sh` (no license needed) — 411 train scenes |
| **PointNav episodes (MP3D train + test)** | 385 MB | [pointnav_mp3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip) — provides 61-scene train split (merged with Gibson) and 18-scene test split (held out for eval) |
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
│   ├── probe.py                        # Linear probing analysis (MiniGrid)
│   ├── habitat_probe_collect.py        # Collect all LSTM layers + pose (Habitat)
│   ├── habitat_probe_analysis.py       # Comprehensive single-condition analysis
│   ├── habitat_probe_cross.py          # Cross-condition CKA + probe transfer
│   ├── habitat_probe_train.py          # Legacy global GPS/compass probe
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

## 5. Methods Landscape: Alternatives & Extensions

Each design choice in our pipeline has alternatives grounded in recent literature. This section maps the space of methods we could trial, organized by component.

### 5.1 Foveation Model

| Method | Description | Reference | Pros | Cons |
|--------|------------|-----------|------|------|
| **Gaussian blur (ours)** | Multi-scale blur, sigma increases with eccentricity | Standard approach | Simple, differentiable, fast | Not biologically faithful |
| **Log-polar transform** | Warp image into log-polar coordinates centered on gaze | "Foveated Retinotopy Improves Classification" (2024, arXiv:2402.15480) | More biological, gives scale/rotation invariance for free | Requires resampling; distorts spatial layout |
| **ECG (Exponential Cartesian Geometry)** | Image pyramid: full-res center crop, progressively coarser surrounding rings | "Seeing More with Less" (2024, seeingmorewithless.github.io) | Outperforms log-polar on classification; simple to implement | Less biologically grounded than log-polar |
| **FoveaTer** | Foveated tokenization in Vision Transformer: high-res patches near gaze, coarse elsewhere | "FoveaTer: Foveated Transformer for Image Classification" (2022) | Native ViT integration; attention-based | Requires switching from ResNet to ViT |
| **Two-stream (ventral+dorsal)** | Separate "what" (foveal) and "where" (peripheral) processing streams | "Towards Two-Stream Foveation-based Active Vision Learning" (arXiv) | Neuroscience-grounded; separates identity from spatial info | More complex architecture; harder to compare fairly |
| **Retino-cortical model** | Midget cells (fovea, high acuity) + parasol cells (periphery, motion) | "Biologically Inspired Deep Learning for Foveal-Peripheral Vision" (2021, Frontiers Comp Neuro) | 10x speedup, 2.5x memory reduction, only 0.39% accuracy drop | Complex to implement; may not add signal for navigation |

**Recommendation**: Start with Gaussian blur (done). If results are promising, try ECG as the easiest upgrade, then log-polar for the paper.

### 5.2 Gaze Control

| Method | Description | Reference | Pros | Cons |
|--------|------------|-----------|------|------|
| **Deterministic MLP (ours)** | Decode gaze from previous LSTM hidden via MLP + sigmoid | Our implementation | Simple, no action space changes, differentiable | No exploration; gaze may get stuck |
| **Stochastic + entropy bonus** | Sample gaze from Normal distribution, add log-prob to PPO loss | Standard RL approach (cf. `src/models/policy.py`) | Encourages gaze exploration | Requires action space extension; more variance |
| **Return-guided contrastive** | Self-supervised contrastive signal from returns guides where to look | "Gaze on the Prize" (2025, arXiv:2510.08442) | Principled task-relevant gaze objective | Adds contrastive loss complexity |
| **Information-gain maximizing** | Saccade targets selected by maximizing expected information gain | "Joint Learning of Saccades by Active Efficient Coding" (2017, Front Neurorobot) | Bayesian-optimal gaze; connects to H3 analysis | Requires maintaining explicit belief state |
| **Privileged-sensor training** | Train gaze with access to full-res images; deploy with foveated only | "Real-World RL of Active Perception" (2025, NeurIPS, arXiv:2512.01188) | Better gaze signal during training | Train/deploy mismatch; may not transfer |
| **Fixed-center (ablation)** | Gaze always at image center | Ablation baseline | Isolates foveation from gaze learning | No active vision |
| **Random gaze (ablation)** | Gaze uniformly random each step | Ablation baseline | Isolates learned gaze from random foveation | No task-relevant attention |

**Recommendation**: Start with deterministic MLP (done). Ablate with fixed-center and random gaze. If deterministic gaze stays centered, switch to stochastic + entropy bonus.

### 5.3 Visual Encoder

| Method | Description | Reference | Pros | Cons |
|--------|------------|-----------|------|------|
| **ResNet18 (ours)** | Standard CNN backbone | Habitat-baselines default | Lightweight, fits 2 GPUs | Less capacity than ResNet50 |
| **ResNet50** | Larger CNN backbone (Wijmans used this) | Wijmans et al. (ICLR 2020) | Matches original DD-PPO paper | Slower, more memory |
| **ViT (Vision Transformer)** | Patch-based self-attention | Dosovitskiy et al. (ICLR 2021) | Attention maps as interpretable gaze proxy | Much larger; different representation structure |
| **Foveated ViT** | ViT with variable patch sizes (larger in periphery) | "Look, Focus, Act" (2025, arXiv:2507.15833) | Natural foveation via patch granularity | Novel architecture, less tested |

### 5.4 Memory Architecture

| Method | Description | Reference | Pros | Cons |
|--------|------------|-----------|------|------|
| **LSTM 3x512 (ours)** | Standard recurrent memory | Wijmans et al. (ICLR 2023) | Matches Wijmans; fair comparison across conditions | No spatial inductive bias |
| **SRU (Spatially-Enhanced RNN)** | RNN with spatial registration of egocentric observations | Yang et al. (2025, IJRR, arXiv:2506.05997) | Explicit spatial structure; designed for long-range nav | Harder to compare with Wijmans baseline |
| **DNC (Differentiable Neural Computer)** | External read/write memory with attention addressing | Graves et al. (2016, Nature) | Explicit memory slots = interpretable cognitive map | Complex; may not scale to DD-PPO |
| **Decision Transformer** | RL as sequence modeling; attention over past states | Chen et al. (2021, NeurIPS) | Attention patterns reveal which past states matter | Offline RL paradigm; different training setup |
| **Topological GNN memory** | Graph-structured memory: nodes = visited locations | "Cognitive Navigation via TMFT" (2024, IEEE/CAA) | Most natural cognitive map substrate | Graph construction requires explicit localization |
| **Multi-timescale memory** | Fast episodic + slow semantic memory (hippocampal model) | "Memory-Augmented Transformers" survey (2025, arXiv:2508.10824) | Parallels hippocampal fast/slow consolidation | Architectural complexity; unclear probing targets |

**Recommendation**: Keep LSTM for the main experiments (fair comparison with Wijmans). If extending for a paper, try SRU as a principled spatial upgrade.

### 5.5 Probing & Representation Analysis

| Method | What It Measures | Reference | When to Use |
|--------|-----------------|-----------|-------------|
| **Linear probes (ours)** | Linearly decodable spatial info | Alain & Bengio (2017, ICLR Workshop) | Core analysis — always do this |
| **MDL probing** | Minimum description length of labels given representations | Voita & Titov (2020, EMNLP) [code](https://github.com/lena-voita/description-length-probing) | Measures *ease of access* to info, not just presence |
| **CKA** | Representation similarity across conditions/layers | Kornblith et al. (2019, ICML, arXiv:1905.00414) | Compare which LSTM layers are similar across blind/uniform/foveated |
| **RSA** | Representational dissimilarity matrices | Kriegeskorte et al. (2008, Front Syst Neuro) | Compare LSTM geometry against theoretical spatial geometry |
| **SVCCA** | Canonical correlation after SVD denoising | Raghu et al. (2017, NeurIPS) | Track how representations converge during training |
| **Nonlinear probes** | Multi-layer MLP probes at varying complexity | "Probing Classifiers: Promises, Shortcomings" (2022, Comp Ling, MIT Press) | Reveals nonlinearly encoded spatial info that linear probes miss |
| **MINE** | Mutual information via neural estimation | Belghazi et al. (2018, ICML, arXiv:1801.04062) | Quantify bits of spatial info in hidden states per condition |
| **IB analysis** | Information plane (I(X;T) vs I(T;Y)) tracking | Shwartz-Ziv & Tishby (2017, arXiv:1503.02406) | Compare compression-prediction trade-offs across conditions |

**Recommendation**: Linear probes (core) + CKA (cross-condition comparison) + MDL probing (ease of access). Add MINE if information-theoretic framing is central to the paper.

### 5.6 Neuroscience-Inspired Probes

Beyond generic spatial probing, we can test for specific neuroscience-predicted cell types:

| Probe Target | What to Look For | Reference | How to Test |
|-------------|-----------------|-----------|------------|
| **Place cells** | Individual LSTM units with localized spatial firing fields | Banino et al. (2018, Nature) "Vector-Based Navigation Using Grid-Like Representations" | Plot each unit's activation as a function of agent position; look for peaked spatial tuning |
| **Grid cells** | Units with periodic hexagonal spatial firing patterns | Banino et al. (2018, Nature); [code](https://github.com/google-deepmind/grid-cells) | Compute gridness score via spatial autocorrelation |
| **Border cells** | Units that fire near walls/boundaries | "RL as a Framework for Insect Navigation" (2024, Frontiers Comp Neuro) | Correlate unit activation with distance-to-nearest-wall |
| **Head direction cells** | Units encoding the agent's heading | "Learning Place Cells and Remapping" (2024, eLife) | Correlate unit activation with agent heading angle |
| **Successor representations** | Units encoding expected future state occupancy | "Neural Network Based Successor Representations" (2022, Scientific Reports) | Probe for SR matrix rather than current position; reveals predictive cognitive map |

**Key prediction**: Foveated agents may develop stronger border/obstacle cells (peripherally detected) and weaker place cells (position uncertainty due to foveation), compared to uniform-vision agents.

### 5.7 Information-Theoretic Extensions

| Analysis | Description | Reference | What It Reveals |
|----------|------------|-----------|-----------------|
| **MI(hidden; position)** | Mutual information between LSTM state and spatial variables | MINE (Belghazi et al., 2018) | Bits of spatial info per condition |
| **Information Bottleneck** | Track compression-prediction trade-off during training | Shwartz-Ziv & Tishby (2017) | Whether foveated agents compress differently |
| **Disentanglement metrics** | DCI, MIG, SAP scores on learned representations | "Disentangled Representation Learning" (2024, IEEE TPAMI) | Whether spatial factors are cleanly separated in memory |
| **Information gain of gaze** | Reduction in spatial uncertainty per fixation | Radulescu et al. (2022); Zhou & Eckstein (2022) | Whether learned gaze approximates Bayesian-optimal |

### 5.8 Key Gap: No Foveated Vision in Habitat/Gibson

Our literature search found **no published papers** combining foveated vision with Habitat or Gibson for PointGoal navigation. This is a genuine novelty of our project. The closest works study foveation for visual search (Pourrahimi & Bashivan, 2025) or robotic manipulation (Look, Focus, Act, 2025), but not embodied spatial navigation with memory probing.

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

## 7. Setup Guide (Start Here If You're New)

### 7.1 SCITAS Cluster Access

1. **Get a SCITAS account**: https://scitas.epfl.ch/ (requires EPFL credentials)
2. **SSH in**: `ssh <username>@izar.epfl.ch`
3. **Key directories**:

| What | Path | Quota | Notes |
|------|------|-------|-------|
| Home (code) | `/home/<user>/` | 100 GB, backed up | Git repo lives here |
| Scratch (data) | `/scratch/izar/<user>/` | Large, shared | **Files >2 weeks auto-deleted** |
| Our code | `/home/<user>/CS503_Project/` | — | Clone from GitHub |
| Datasets | `/scratch/izar/<user>/habitat_data/` | ~30 GB | Gibson + MP3D scenes |
| Checkpoints | `/scratch/izar/<user>/habitat_checkpoints/` | Growing | Training outputs |

4. **Job submission**: Uses SLURM. `cs-503` QoS allows **max 2 concurrent jobs** on GPU partition.

```bash
sbatch scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_blind_gibson  # submit
squeue -u $USER                                                                  # check status
scancel <job_id>                                                                 # cancel
```

### 7.2 Clone the Repo

```bash
# On the cluster
cd ~
git clone https://github.com/alunxu/agentic-cognitive-map.git CS503_Project
cd CS503_Project
```

### 7.3 Environment Setup

**Two conda environments**: one for MiniGrid (local dev), one for Habitat (cluster training).

#### MiniGrid environment (local or cluster, CPU-only OK)
```bash
bash scripts/cluster/setup_env.sh
conda activate cs503_project
```

#### Habitat environment (cluster only, needs GPU nodes)
```bash
conda create -n habitat python=3.9 cmake=3.22 -y
conda activate habitat

# PyTorch (must match cluster CUDA driver — 12.1 on Izar)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Habitat simulator (headless, no display needed)
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat

# Habitat lab + baselines (from source for DD-PPO)
cd ~
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines

# IMPORTANT: Patch for blind policy (RunningMeanAndVar(0) assertion error)
# In habitat-baselines/rl/ddppo/policy/resnet_policy.py, change:
#   if normalize_visual_inputs:
# to:
#   if normalize_visual_inputs and self._n_input_channels > 0:

# Symlink data directory (avoids Hydra {split} parsing issues)
ln -s /scratch/izar/$USER/habitat_data ~/habitat-lab/data
```

### 7.4 Dataset Download

**Only one person needs to do this** — data is shared on scratch.

#### Gibson scenes (~490 .glb files, 13 GB) — already downloaded
```bash
# Gibson .glb files live at:
/scratch/izar/wxu/habitat_data/scene_datasets/gibson/
```

#### Gibson-0+ PointNav episodes (411 train scenes, 320 MB) — required
```bash
# One-shot download + verification + .glb cross-check:
bash scripts/cluster/download_gibson_0plus.sh

# That script:
#   1. downloads pointnav_gibson_0_plus_v1.zip
#   2. extracts to $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/
#   3. cross-checks all 411 scene names against the .glb files on disk
#      (any scene whose .glb is missing will crash training)
```

#### Matterport3D (~15 GB) — access granted
```bash
# 1. Download script (already on cluster at ~/download_mp.py)
wget -O ~/download_mp.py https://kaldir.vc.cit.tum.de/matterport/download_mp.py

# 2. Download Habitat-compatible MP3D scenes
python3 ~/download_mp.py -o /scratch/izar/$USER/mp3d_raw --task_data habitat

# 3. Unzip and move .glb files to Habitat data dir
mkdir -p /scratch/izar/$USER/habitat_data/scene_datasets/mp3d/
unzip /scratch/izar/$USER/mp3d_raw/v1/tasks/mp3d_habitat.zip -d /scratch/izar/$USER/habitat_data/scene_datasets/mp3d/

# 4. Download PointNav episodes for MP3D (train + val + test splits)
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip
unzip pointnav_mp3d_v1.zip -d /scratch/izar/$USER/habitat_data/datasets/
```

#### Merged Gibson + MP3D training pool (required) — Wijmans A.1

Our configs train on `data/datasets/pointnav/mp3d_gibson/v1/train/`, a merged pool
built by symlinking episode files from both datasets into one virtual split.

```bash
# Assumes Gibson-0+ and MP3D PointNav episodes above are already in place.
MERGED=/scratch/izar/$USER/habitat_data/datasets/pointnav/mp3d_gibson/v1/train
mkdir -p "$MERGED/content"

# 1. Gibson-0+ train_extra_large → content/ (one json.gz per scene)
ln -sfn /scratch/izar/$USER/habitat_data/datasets/pointnav/gibson/v1/train_extra_large/content/*.json.gz "$MERGED/content/"

# 2. MP3D train → content/ (scenes whose .glb actually exists on disk)
ln -sfn /scratch/izar/$USER/habitat_data/datasets/pointnav/mp3d/v1/train/content/*.json.gz "$MERGED/content/"

# 3. Top-level split file (any non-empty json.gz works; habitat-lab only
#    scans content/ at load time)
ln -sfn /scratch/izar/$USER/habitat_data/datasets/pointnav/gibson/v1/train_extra_large/train_extra_large.json.gz "$MERGED/train.json.gz"

# 4. Dummy val split (habitat-baselines eval defaults to val and won't
#    accept split=train for the eval path). We symlink val -> train.
VAL=/scratch/izar/$USER/habitat_data/datasets/pointnav/mp3d_gibson/v1/val
mkdir -p "$VAL"
ln -sfn ../train/content "$VAL/content"
ln -sfn ../train/train.json.gz "$VAL/val.json.gz"
```

After this, `ls $MERGED/content/ | wc -l` should show 472 (411 Gibson + 61 MP3D).

#### habitat-lab eval-video patch — required for rendering trajectory MP4s

Two upstream `habitat-lab` bugs prevent `eval.video_option=[disk]` from
producing a file on our scene pool. Both are fixed in
`patches/habitat_lab_eval_video.patch`:

```bash
cd ~/habitat-lab
git apply ~/CS503_Project/patches/habitat_lab_eval_video.patch
```

Fix summary:
1. `overlay_frame` tried to `.2f`-format the `top_down_map` numpy array and crashed — now skips non-scalar metrics.
2. `images_to_video` rejected frames whose width changed as the top-down-map tile grew — now zero-pads every frame to the first frame's shape.

#### If you're using another team member's data
```bash
# Symlink to wxu's data (no need to re-download; merged pool already built)
ln -s /scratch/izar/wxu/habitat_data /scratch/izar/$USER/habitat_data
```

### 7.5 Copy Configs to Habitat

Our custom Habitat configs must be placed where habitat-baselines can find them:

```bash
# Copy all Gibson configs
cp ~/CS503_Project/habitat_configs/ddppo_pointnav_*_gibson.yaml \
   ~/habitat-lab/habitat-baselines/habitat_baselines/config/pointnav/
```

### 7.6 Training Commands

```bash
cd ~/CS503_Project

# Submit training (2 V100 GPUs, up to 72h wall time)
sbatch scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_blind_gibson
sbatch scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_uniform_gibson
# The cs-503 QOS caps you at 2 concurrent jobs. To launch foveated + matched
# in parallel to the first two, switch to the normal QOS (longer queue but
# no concurrent-job cap):
sbatch --qos=normal scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_foveated_gibson
sbatch --qos=normal scripts/cluster/submit_habitat.sh pointnav/ddppo_pointnav_matched_gibson

# Monitor progress
squeue -u $USER                        # job status
tail -f slurm_logs/<job_id>.err        # live training logs
ls /scratch/izar/$USER/habitat_checkpoints/<run_name>/  # checkpoints
```

See [`docs/foveation_design.md`](docs/foveation_design.md) for the frozen
PoC hyperparameters (fovea_radius=16, blur_sigma_max=8.0, fixed center
gaze, matched-compute 48×48) and the rationale behind each choice.

### 7.7 Recording Agent Videos (for Evaluation & Figures)

Habitat can save first-person RGB videos of trained agents navigating, which are invaluable for:
- **Qualitative analysis**: see what the agent actually looks at, how it plans turns, where it gets stuck
- **Figures for the paper**: side-by-side comparison of what each of the 4 agents sees on the same episode is the most compelling visual we can produce
- **Debugging the foveated agent**: visualize gaze position and blur pattern frame by frame

#### How to enable — use the helper script

`scripts/cluster/submit_habitat_eval.sh` wraps all the gotchas
(`load_resume_state_config=False` to respect CLI overrides,
`num_environments=1`, `eval.video_option=[disk]`, the right config key
path, the val→train split handling) into a one-liner. Training state
is never touched.

```bash
cd ~/CS503_Project
# sbatch <config_name> <ckpt_path> [num_episodes]
sbatch scripts/cluster/submit_habitat_eval.sh \
    pointnav/ddppo_pointnav_uniform_gibson \
    /scratch/izar/$USER/habitat_checkpoints/uniform_gibson/ckpt.4.pth \
    5

# Videos end up in /scratch/izar/$USER/eval_videos/<run_name>/
# Filenames embed every metric:
#   episode=57311_1-ckpt=4-distance_to_goal=0.07-success=1.00-spl=0.95-...mp4
```

Prerequisite (one-time per habitat-lab install): apply the eval-video
patch documented in §7.4, otherwise ffmpeg will reject frames and no
MP4 will be written. Every frame is a composite of first-person RGB on
the left and the top-down trajectory map on the right, padded to a
fixed canvas per episode.

#### For the foveated agent

The foveated view (post-blur) is what the policy actually sees, but `habitat_baselines.video_option` records the raw sensor output before our foveation transform. To record the *foveated* view, we'll need a small hook in `src/habitat/foveated_policy.py` that dumps the post-foveation tensor to disk during evaluation. This is a Member D task — straightforward (5-10 lines of code) but worth noting in advance.

#### Side-by-side comparison videos

To render "what all 4 agents see on the same episode," use a fixed random seed so all four agents evaluate on the same PointNav episodes. Then stack the 4 videos horizontally with `ffmpeg -i blind.mp4 -i uniform.mp4 -i foveated.mp4 -i matched.mp4 -filter_complex hstack=inputs=4 comparison.mp4`.

### 7.8 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `RunningMeanAndVar(0)` assertion | Patch `resnet_policy.py` (see 7.3 above) |
| `{split}` in Hydra CLI | Use symlink `~/habitat-lab/data → /scratch/izar/$USER/habitat_data` |
| CUDA driver mismatch | Use PyTorch 2.1.0+cu121 (matches Izar's CUDA 12.1) |
| Job stuck on `(Resources)` | Cluster busy; check `sinfo -p gpu` for free nodes |
| 4-GPU jobs never start | Only 2 nodes have 4 GPUs (often drained). Use 2 GPUs instead |
| `ConnectionResetError` in env init | Reduce `num_environments` in config (6→4) |

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

### Phase 2: Habitat Gibson + MP3D (5-condition run in flight)

Snapshot — 2026-04-10 (all 4 initial conditions training, learned-gaze pending):

| Agent | Job | Steps | Success | SPL | Target | Notes |
|-------|-----|-------|---:|---:|---:|-------|
| **Blind** | 2826964 | ~162M | 78.3% | 0.452 | 500M | 32% done, will need resubmission |
| **Uniform** | 2826965 | ~110M | 92.2% | 0.756 | 250M | 44% done, strong |
| **Foveated (fixed)** | 2827419 | ~63M | 93.0% | 0.691 | 250M | 25% done — remarkably strong, beating uniform at fewer steps |
| **Matched-Compute** | 2827420 | ~90M | 87.4% | 0.527 | 250M | 36% done |
| **Foveated (learned)** | — | — | — | — | 250M | Pending — to be launched after fixed-gaze validates |

**Dataset.** All conditions train on a merged Gibson-0+ (411) ∪ MP3D-train (61) = 472-scene pool and evaluate on Matterport3D test (18 scenes, 1008 episodes), matching Wijmans 2023 Appendix A.1. See §3.2 for the Gibson-0/2/4+ naming clarification and §7.4 for the symlink recipe.

**Probing pipeline v2 deployed (2026-04-10).** Comprehensive probing analysis (`habitat_probe_analysis.py`) with 12 experiments across 5 phases: baseline probes (GPS/compass, per-scene position, distance-to-goal, multi-layer comparison, control tasks), H1 tests (accuracy vs. timestep, cross-heading generalization), SPACE-inspired probes (path-history lag-k decoding, visited-region spatial working memory), and per-unit rate maps. Cross-condition CKA and probe transfer via `habitat_probe_cross.py`. Jobs 2828732 (blind ckpt.16) and 2828733 (uniform ckpt.22) submitted with v2 pipeline.

**Initial probing results (2026-04-10, legacy pipeline, 500 eps).** Blind agent (ckpt.10): GPS R²=0.876, Compass R²=0.604. Uniform agent (ckpt.14): GPS R²=0.453, Compass R²=0.706. Blind agent encodes GPS much more strongly (no vision → heavier reliance on memory for position tracking). Uniform agent encodes heading better (visual landmarks anchor orientation).

**Eval video pipeline shipped (2026-04-09).** `scripts/cluster/submit_habitat_eval.sh` plus the `patches/habitat_lab_eval_video.patch` upstream fixes produce playable MP4s (RGB + top-down trajectory map) with every metric embedded in the filename.

**Foveation PoC design frozen (2026-04-09).** The five conditions share a single set of frozen hyperparameters documented in [`docs/foveation_design.md`](docs/foveation_design.md). No sweeps in the PoC — one seed per condition, one setting per agent.

**Training will require resubmission.** No job will finish within the 72h SLURM wall limit. DD-PPO resumes automatically from the latest checkpoint, so resubmitting the same `sbatch` command continues training.

---

## 9. Detailed TODO

### Phase 2A: Finish Habitat Blind Training
- [x] Set up Habitat + Gibson on SCITAS
- [x] Download Gibson scene .glb files (~490 scenes, 13 GB)
- [x] Create DD-PPO config for blind agent
- [x] ~~Submit blind agent training (job 2823083, 2 GPUs) on Gibson-4+~~ *Discarded 2026-04-08 after Gibson-0+ migration*
- [x] Migrate all 5 Habitat configs to Gibson-0+ train (411 scenes) + MP3D test eval
- [x] Run `scripts/cluster/download_gibson_0plus.sh` on cluster — verified 411/411 Gibson + 18/18 MP3D
- [x] Normalize nested MP3D layout (`mp3d/mp3d/*` → `mp3d/*`)
- [ ] Re-submit blind training on Gibson-0+ (411 scenes)
- [ ] Verify final blind metrics match Wijmans (target: >95% success, >0.9 SPL)
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
- [x] Build Habitat probing pipeline v1 (collect LSTM top-h, GPS/compass probe)
- [x] Run initial probing on blind (ckpt.10) and uniform (ckpt.14), 500 episodes
- [x] Build comprehensive probing pipeline v2 (`habitat_probe_analysis.py`)
  - [x] Collect ALL LSTM layers (h₁/h₂/h₃ + c₁/c₂/c₃) + distance-to-goal + step index
  - [x] 1a: Per-scene absolute position probe (Wijmans replication)
  - [x] 1b: Global GPS/compass probe
  - [x] 1c: Distance-to-goal probe
  - [x] 1d: Multi-layer comparison (which layer encodes spatial info?)
  - [x] 1e-f: Control tasks + Hewitt & Liang selectivity index
  - [x] 2a: Accuracy vs. timestep-in-episode
  - [x] 2b: Cross-heading generalization (allocentric vs. egocentric codes)
  - [x] 2c: Path-history lag-k decoding (inspired by SPACE route retracing)
  - [x] 2d: Visited-region probe (inspired by SPACE CSWM)
  - [x] 5a: Per-unit rate maps + place-cell identification
- [x] Build cross-condition analysis (`habitat_probe_cross.py`)
  - [x] 3a: Pairwise CKA (per-layer, h and c states)
  - [x] 3b: Cross-condition probe transfer
- [ ] Submit v2 probing for blind + uniform (jobs 2828732, 2828733) ← in queue
- [ ] Run v2 probing on foveated + matched (after sufficient training)
- [ ] Run cross-condition CKA analysis (all 4 conditions)
- [ ] **H1 test: Compensatory memory**
  - [ ] Compare accuracy-vs-timestep curves across conditions
  - [ ] Compare path-history decay rates: blind vs. foveated vs. uniform
- [ ] **H2 test: Representational divergence**
  - [ ] Interpret CKA: low foveated↔uniform despite matched task performance?
  - [ ] Cross-heading generalization: foveated more allocentric?
  - [ ] Probe transfer: do probes trained on one condition work on another?
- [ ] **H3 test: Epistemic gaze** (requires learned-gaze agent)
  - [ ] Correlate learned gaze with position-probe error
  - [ ] Compare learned-gaze vs. fixed-centre: does active gaze amplify H1–H2?
- [ ] **Visualization**:
  - [ ] Per-unit rate maps (place-cell-like neurons)
  - [ ] Cross-condition CKA heatmap (layer × layer)
  - [ ] R² bar charts across all conditions
  - [ ] Path-history decay curves

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
| **Spatial cognition benchmark** | Ramakrishnan, Wijmans et al. (ICLR 2025, SPACE) | Show frontier models fail at spatial cognition (route retracing, shortcut discovery, map sketching) | They test disembodied models; we study *embodied* agents where spatial representations emerge. Their tasks inspire our path-history and visited-region probes |
| **Sensor design & task perf.** | Atanov et al. (ECCV 2024, VILAB) | Show photoreceptor design critically shapes task performance | We extend from "sensor shapes performance" to "sensor shapes memory content" |
| **Perception-action-memory** | Beker et al. (NeurIPS 2022, PALMER) | Study perception-action loops with memory for planning | We specifically study how foveation constrains what memory encodes |

**The gap we fill**: No existing work studies how foveation — spatially varying perceptual quality — shapes the *content and structure* of cognitive maps in navigation agents. The components exist separately; we combine them.

---

## 11. References

**Core:**
- Wijmans et al., "Emergence of Maps in the Memories of Blind Navigation Agents," ICLR 2023. [Paper](https://arxiv.org/abs/2301.13261) | [Code](https://github.com/erikwijmans/emergence-of-maps)
- Wijmans et al., "DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames," ICLR 2020. [Paper](https://arxiv.org/abs/1911.00357)
- Ramakrishnan, Wijmans et al., "Does Spatial Cognition Emerge in Frontier Models?" ICLR 2025. [Paper](https://arxiv.org/abs/2410.06468) | [Code](https://github.com/apple/ml-space-benchmark) — SPACE benchmark; inspires our path-history and visited-region probes

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
