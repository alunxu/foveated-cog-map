# How Foveated Vision Shapes Cognitive Maps in Navigation Agents

**CS503 Visual Intelligence — EPFL, Spring 2026**
**Submission target: NeurIPS 2026**

> Holding task and architecture fixed, does the structure of the visual sensor reshape what an agent's memory encodes?

---

## 1. Project Overview

### The Big Picture

Wijmans et al. (ICLR 2023) showed that map-like representations emerge in the recurrent state of navigation agents trained with deep RL — *including agents with no vision at all*. Their result establishes that *some* spatial structure is learned without being designed in. It leaves open the complementary question we take up here: **holding task and architecture fixed, does the structure of the visual sensor reshape what an agent's memory encodes?**

We treat this as a controlled silicon case study using a recurrent (LSTM-3-layer-512) PointNav agent. The architecture, training algorithm (DD-PPO), reward, and dataset are held fixed; **only the visual sensor varies** across five conditions (Figure 1 in the paper).

### Why Foveation? Why Sensor Variety?

Biological evidence is consistent: sensor structure reshapes spatial memory format. Primates with foveated eyes show place-view-action coding in hippocampus; rodents with near-panoramic vision show classic place fields; bats remap their hippocampal populations under vision vs. echolocation; congenitally blind humans recruit occipital-hippocampal circuits for spatial tasks. The pattern across systems is that "the structure of the sensor shapes the structure of spatial memory."

Whether the same holds for *artificial* navigation agents — whose memory also develops cognitive-map-like structure under task pressure alone — has not been tested directly, because prior work has either removed vision entirely or used spatially uniform input. We isolate the principle by varying only the visual sensor.

### Five Experimental Conditions

We train five navigation agents (varying only visual input) and compare what their memories encode:

| Agent | Input | Encoder spatial output | Role |
|-------|-------|------------------------|------|
| **Blind** | Non-visual sensor stack only (goal-in-start-frame, GPS, compass, close-to-goal) | None | Baseline: do spatial maps emerge without any vision? (Wijmans replication) |
| **Coarse (1×1)** | 48×48 uniform RGB | 1×1 (encoder collapsed) | Has visual input but encoder cannot resolve world-frame position. Tests whether the bottleneck is "no vision" or "no spatial features" |
| **Uniform** | 256×256 full-resolution RGB | 8×8 | Does adding spatially rich vision change memory content? |
| **Foveated (fix)** | 256×256 with eccentricity-dependent Gaussian blur (σ_max=8, quadratic falloff), gaze locked at center | 8×8 | Spatially non-uniform input within the rich-encoder regime |
| **Foveated (learned)** | Same blur, gaze (x,y) predicted by lightweight MLP from previous LSTM state | 8×8 | Does active gaze amplify the effect? |

All five share an identical 3-layer LSTM-512 backbone. Sighted conditions train to ~250M environment frames; blind trains to 342M to allow slower convergence from proprioception alone.

### Three Hypotheses

**H1 (Encoder–memory race):** When the visual encoder supplies little spatial-feature variety per step, the recurrent state should compensate by carrying a world-frame spatial code; richer encoder output should let memory rely on visual features instead.

**H2 (Format divergence):** Different sensor conditions should produce hidden-state representations in *distinct linear subspaces*, not scaled versions of a common code.

**H3 (Gaze location):** Within the rich-encoder regime, *where* a foveated sensor is centered should act as a second content axis independent of the sensor transform.

Each H has a direct biological analog (paper §5.2 *Biological precedent*).

### Headline Findings (single-seed; multi-seed replication in flight)

1. **Encoder–memory race (H1)**: visual-encoder capacity tracks top-layer LSTM spatial encoding *inversely*. Bottleneck conditions (blind, coarse) carry a strong linear GPS code at the policy readout (R² 0.78–0.95); rich-encoder conditions (uniform, foveated, foveated-learned) do not. **Cross-training probes show the GPS code emerges *transiently* in rich-encoder agents (R²~0.7–0.8 at ~50M frames) then progressively dissipates** — direct mechanistic evidence for the *substitution mechanism* (LSTM hands off integrated GPS to the visual route as the encoder learns).
2. **Format divergence (H2)**: the five agents at comparable task competence (success 93–99%) develop hidden states in linearly disjoint subspaces. Cross-condition memory transplants are asymmetric: bottleneck-donor states are toxic to rich-encoder recipients while the reverse is benign.
3. **Probe-readable vs. policy-used dissociation**: at the H1×shortcut intersection, two conditions go off-diagonal — coarse has a readable GPS code that its policy under-uses; uniform has no readable GPS but its policy is highly memory-dependent. "Has a cognitive map" and "uses a cognitive map" can dissociate in either direction.

### Motivation

If the principle holds:
- **Sensor structure as a representational-content axis.** Encoder capacity becomes a *training-time lever* for inducing cognitive-map-like memory structure — without changing task, architecture, or reward.
- **The encoder–memory race is interface-level.** The claim is about what an upstream encoder doesn't supply forcing a downstream memory to compensate. It should generalise (with adapted measurements) to transformer-based navigators, attention-over-history architectures, etc.
- **Convergent with biology.** The pattern parallels independent animal-navigation findings on hippocampal compensation under sensory deprivation and cross-modality remapping (paper §5.2).

---

## 2. Methods

### 2.1 Task: PointGoal Navigation

The agent must navigate from a random start to a goal location specified as a relative vector (distance + direction). The agent receives:
- **Pointgoal**: 4D vector (distance, sin/cos of angle to goal) — all conditions
- **RGB image**: first-person camera view — sighted conditions only

Actions: `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT` (+ `STOP` in Habitat)

### 2.2 Agent Architecture

All five conditions share the same recurrent backbone for fair comparison:

```
Observation → Encoder → [concatenate with non-visual sensor stack] → LSTM (3 layers, 512-d) → Policy Head
                                                                          ↑
                                                                  This hidden state is what we probe
```

The non-visual sensor stack — `goal-in-start-frame`, `GPS`, `compass`, `close-to-goal-indicator` (each → 32-d) plus a 32-d previous-action embedding — is identical across all five conditions; only the visual encoder varies.

| Component | Blind | Coarse (1×1) | Uniform | Foveated (fixed/learned) |
|-----------|-------|--------------|---------|--------------------------|
| Visual encoder | None | ResNet-18 | ResNet-18 | ResNet-18 + foveation transform |
| Input resolution | N/A | 48×48 | 256×256 | 256×256 (with σ_max=8 blur) |
| Encoder spatial output | None | 1×1 (collapsed) | 8×8 | 8×8 |
| LSTM | 3 layers, 512-d | (same) | (same) | (same) |
| Policy head | Categorical | (same) | (same) | (same; + gaze MLP for learned) |

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

- **Habitat**: DD-PPO (Decentralized Distributed PPO) on 2× V100 GPUs (Izar SCITAS) or 1× hc GPU (collaborator's H200)
  - Sighted conditions train to **~250M environment frames**; **blind trains to ~342M** to allow slower convergence from proprioception alone
  - 72h SLURM walltime → DD-PPO auto-resumes from `latest.pth` checkpoint
  - Multi-seed replication (seed=2 of all 5 conditions) currently in flight on Izar
  - Each worker manages multiple parallel environments (typically 6 per worker, 2 workers per node)

### 2.4.1 Numerical Stability: NaN Sanitisation (Important for Downstream Users)

**Symptom observed in our foveated run**: training proceeded normally for ~170M frames, then at one update the policy's logit tensor contained NaN, the PPO loss became NaN, backward produced NaN gradients, and `optimizer.step()` wrote NaN into every parameter. From that point on the network produced NaN for every input, but the training loop did not crash — it silently continued for 2.5 more days until the frame budget was hit, leaving a corrupted checkpoint (90 of 97 parameter tensors all-NaN).

**Root cause is upstream, not in our code**:

- `torch.nn.utils.clip_grad_norm_` does not sanitise NaN. If any gradient is NaN, the global norm is NaN → clip coefficient is NaN → `grad *= NaN` leaves the grad NaN. The optimizer then propagates NaN into the weights.
- The originating NaN on a single mini-batch can come from rare float32 edge cases that standard RL training hits only after many millions of updates: softmax under-flow then `log(0)`, value-head overflow, zero-variance advantage normalisation, LSTM cell blow-up combined with gate saturation. Our debug run (see `scripts/cluster_debug/`) pins the exact op.
- DD-PPO's gradient all-reduce averages NaN with finite values → all workers get NaN → the corruption is systemic from the first bad batch.

**Our fix** (`src/habitat/wijmans_policy.py`):

1. At import time, monkey-patches `habitat_baselines.rl.ppo.ppo.PPO.before_step` to replace non-finite gradient elements with zero via `torch.nan_to_num_`. A bad mini-batch degenerates to a no-op update instead of corrupting weights.
2. At policy construction time, wraps the action-head's `nn.Linear.forward` so NaN logits are replaced with 0 and clamped to [-10, 10]. This prevents `torch.multinomial` from raising on NaN probabilities.
3. Exposes a sanitisation counter (`wijmans_policy.NAN_SANITISATION_STATS`) and a per-update `nan_sanitised` metric in `learner_metrics`, so you can see whether the safety net fired during training.

**Scope and guarantees**:

- The fix is a **no-op on clean training paths**: `nan_to_num` of a finite tensor is identity. A training run with zero NaN events produces bitwise-identical weights with or without the patch. Across our five conditions, four of them had zero NaN events throughout full training runs, confirming this.
- The fix is installed automatically when you `import src.habitat`, which is what our training entry point does. Anyone using this codebase as a library for a different task inherits the fix without extra wiring.
- If your `learner_metrics` tensorboard shows `nan_sanitised > 0`, rare numerical instability occurred and was absorbed safely. If it stays at 0, nothing is happening — the patch is inert.

**For downstream users doing unrelated work with habitat-baselines DDPPO**: this failure mode is not specific to foveated agents; it can surface on any long-horizon float32 DDPPO run. If you port this code to a different task and see `nan_sanitised > 0`, your hyperparameters and numerical precision are tight; investigate rather than ignore. If you see the original crash (`torch.multinomial: probability tensor contains inf, nan`) without the fix, our patch is what you want.

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

### 2.7 Ablations & Follow-up Experiments

To understand which factors drive the H1 / H2 / H3 results, we run a structured set of ablation and control experiments. Status as of 2026-04-26:

| Experiment | Configs | Status | Targets |
|---|---|---|---|
| **Foveation strength sweep** | `foveated_sigma{2,4,12}_gibson` (σ=2/4/12) + existing `foveated_gibson` (σ=8) + `foveated_strong_gibson` (σ=20) | σ=20 RUNNING; σ=2/4/12 PENDING in queue | H1 as continuous lever; falsifiable bound on encoder–memory race |
| **Log-polar foveation** (F3) | `foveated_logpolar_gibson` | RUNNING (~22h / 72h walltime) | Test "spatial sampling" as the H1 mechanism (vs. blur strength). Predicted GPS R² ≥ 0.3 (between coarse +0.78 and uniform ~0). If matches uniform, H1 mechanism story needs reframing. |
| **Foveated-shifted (H3 control)** | `foveated_shifted_gibson` | PENDING | Static gaze hardcoded at (0.49, 0.62), matching the position learned-gaze collapsed to. Isolates static-gaze-location effect from learned-gaze dynamics. |
| **Stochastic gaze policy** | `foveated_stochastic_gibson` (FoveatedStochasticGazePolicy) | Queued for collaborator's hc cluster (see `docs/hc_launch_recipe.md`) | Reparameterized bounded-σ Gaussian gaze sampling; designed to fix gaze collapse without aux loss. Re-tests H3 with a working learned gaze. |
| **Encoder-capacity scaling sweep** | `matched{32,64,96,192}_gibson` (Coarse at 4 input resolutions) | Pending hc cluster | Bridges Coarse (1×1 collapse @ 48×48) to Uniform (8×8 @ 256×256). Causal H1 test: GPS R² should monotonically decrease with input resolution. |
| **Multi-seed replication** | `{blind,uniform,foveated,foveated_learned}_gibson_seed2` + matched | RUNNING (5 conditions on Izar) | Gates strength of every quantitative claim |
| **fov_v2 (clean re-run)** | `foveated_v2_gibson` | RUNNING | Re-train fov-fix from scratch on the post-NaN-fix clean ckpt chain (current `foveated_gibson` uses ckpt.36 = ~174M frames, last clean before NaN-corruption window) |
| **Across-checkpoint probing** | All 5 conditions × 4–5 ckpts each | DONE | Substitution mechanism direct evidence (paper §4.2 Figure 3) |

---

## 3. Environments & Datasets

### 3.1 Habitat + Gibson

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
├── habitat_configs/                    # Habitat DD-PPO training configs (Hydra)
│   ├── ddppo_pointnav_blind_gibson.yaml
│   ├── ddppo_pointnav_uniform_gibson.yaml
│   ├── ddppo_pointnav_matched_gibson.yaml             # Coarse 48×48 → 1×1 encoder output
│   ├── ddppo_pointnav_foveated_gibson.yaml            # Foveated (fix), σ=8
│   ├── ddppo_pointnav_foveated_learned_gibson.yaml    # Foveated (learned), deterministic gaze
│   ├── ddppo_pointnav_foveated_v2_gibson.yaml         # Clean re-run of fov-fix
│   ├── ddppo_pointnav_foveated_shifted_gibson.yaml    # H3 control: gaze hardcoded at (0.49, 0.62)
│   ├── ddppo_pointnav_foveated_stochastic_gibson.yaml # Stochastic-gaze variant
│   ├── ddppo_pointnav_foveated_sigma{2,4,12}_gibson.yaml  # σ_max strength sweep
│   ├── ddppo_pointnav_foveated_strong_gibson.yaml     # σ=20 (F4 strength endpoint)
│   ├── ddppo_pointnav_foveated_logpolar_gibson.yaml   # F3 log-polar resampling
│   ├── ddppo_pointnav_foveated_normaliser_gibson.yaml # F2 invariance control
│   ├── ddppo_pointnav_foveated_learned_div_gibson.yaml # learned-gaze + diversity aux loss
│   └── ddppo_pointnav_matched{32,64,96,128,192}_gibson.yaml # Encoder-capacity scaling sweep
│
├── src/
│   ├── habitat/                        # Habitat DD-PPO integration
│   │   ├── __init__.py                 # Registers all custom policies with habitat-baselines
│   │   ├── wijmans_policy.py           # Wijmans et al. PointNav policy + NaN-sanitisation patch
│   │   ├── wijmans_sensors.py          # Custom sensors (GoalInStartFrame, CloseToGoal)
│   │   ├── foveated_policy.py          # Foveated policy (fixed-center gaze)
│   │   ├── foveated_learned_policy.py  # Foveated policy (deterministic learned gaze)
│   │   ├── foveated_shifted_policy.py  # Foveated, gaze hardcoded at (0.49, 0.62) — H3 control
│   │   ├── foveated_stochastic_policy.py # Foveated, bounded-σ Gaussian gaze sampling
│   │   ├── foveated_sigma{2,4,12}_policy.py # σ-strength variants of foveated
│   │   ├── foveated_strong_policy.py   # σ=20 variant (F4)
│   │   ├── foveated_logpolar_policy.py # Log-polar resampling (F3)
│   │   ├── foveated_normalised_policy.py # σ=8 + RunningMeanAndVar (F2)
│   │   ├── gaze_diversity_loss.py      # Anti-collapse aux loss for learned gaze
│   │   └── torch_foveation.py          # GPU-native differentiable foveation transforms
│   └── utils/                          # Shared utilities
│       ├── probing.py                  # Ridge probe fitting, feature prep, episode split
│       └── habitat_env.py              # Config loading, policy loading, geometry helpers
│
├── scripts/
│   ├── probing/                        # Representation-probing pipeline
│   │   ├── collect.py                  # Collect LSTM hidden states + ground-truth pose
│   │   ├── analyze.py                  # Comprehensive single-condition probes (12 experiments)
│   │   ├── analyze_cross.py            # Cross-condition CKA + probe transfer
│   │   ├── analyze_extra_states.py     # Probes h_0/h_1/h_2/c_0/c_1/c_2 × {GPS, compass}
│   │   ├── analyze_encoder_features.py # Probes the post-ResNet-18 feature map directly
│   │   ├── temporal_probe.py           # Step-binned GPS R² across episode
│   │   ├── lagk_all_targets.py         # Lag-k path-history probe
│   │   ├── masked_heading_probe.py     # H3 causal test: encoding vs. passthrough
│   │   ├── goal_vector_probe.py        # Goal-vector probe (ego-frame goal)
│   │   ├── population_coding_analysis.py # Per-unit spatial information + place-cell stats
│   │   ├── unaligned_cka.py            # Cross-condition linear CKA
│   │   ├── compare_det.py              # Deterministic-vs-stochastic comparison
│   │   └── ...                         # ~25 more diagnostic scripts
│   ├── eval/                           # Behavioral evaluation scripts
│   │   └── shortcut.py                 # Shortcut discovery (paired same-scene-different-goal eps)
│   ├── paper_figures/                  # All paper figure generators (PDF-only output)
│   │   ├── make_setup_panels.py        # Fig 1: 5-condition setup panels
│   │   ├── make_setup_pipeline.py      # Fig 1: training+analysis schematic
│   │   ├── make_h1_mega_figure.py      # Fig 2: H1 evidence (3 panels)
│   │   ├── make_substitution_figure.py # Fig 3: substitution mechanism cross-training
│   │   ├── make_h2_probe_transfer.py   # Fig 4: H2 probe-transfer heatmap
│   │   ├── make_5x5_transplant_matrix.py  # Fig 4: H2 5×5 transplant matrix
│   │   ├── make_shortcut_paired_trajectory_figure.py # Fig 5: persistent-memory failures
│   │   ├── make_synthesis_figure.py    # Fig 6: 3-axis synthesis
│   │   ├── make_extra_states_figure.py # Appfig 10: per-(state×layer) probes
│   │   ├── make_*figure*.py            # Other appfigN_ generators
│   │   └── ...
│   └── cluster/                        # SLURM job scripts
│       ├── common.sh                   # Shared env setup (sourced by all submit_*.sh)
│       ├── submit_train.sh             # Submit DD-PPO training (Izar V100)
│       ├── submit_train_seeded.sh      # Multi-seed training with isolated checkpoint dir
│       ├── submit_probe_deterministic.sh  # Deterministic-rollout probing
│       ├── submit_eval.sh              # Submit evaluation + video
│       ├── submit_shortcut.sh          # Submit shortcut eval
│       ├── submit_transplant.sh        # Submit memory transplant experiment
│       ├── run_habitat.py              # Custom entry point (registers policies)
│       └── ...
│
├── docs/
│   ├── NeurIPS_2026/                   # Paper source + final figures
│   │   ├── neurips_2026.tex
│   │   ├── neurips_2026.pdf
│   │   ├── MASTER_TRACK.md             # Single source of truth for project state
│   │   ├── literature.bib
│   │   └── fig/                        # 21 figures, named figN_*/appfigN_*
│   ├── hc_launch_recipe.md             # Friend's high-compute cluster launch recipe
│   ├── foveation_design.md             # Foveation PoC design rationale
│   └── ...
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
   Trains the sighted (uniform) and coarse (1×1) agents.

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
sbatch scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_blind_gibson  # submit
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
sbatch scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_blind_gibson
sbatch scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_uniform_gibson
# The cs-503 QOS caps you at 2 concurrent jobs. To launch foveated + matched
# in parallel to the first two, switch to the normal QOS (longer queue but
# no concurrent-job cap):
sbatch --qos=normal scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_gibson
sbatch --qos=normal scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched_gibson

# Monitor progress
squeue -u $USER                        # job status
tail -f slurm_logs/<job_id>.err        # live training logs
ls /scratch/izar/$USER/habitat_checkpoints/<run_name>/  # checkpoints
```

See [`docs/foveation_design.md`](docs/foveation_design.md) for the frozen
PoC hyperparameters (fovea_radius=16, blur_sigma_max=8.0, fixed center
gaze, coarse 48×48) and the rationale behind each choice.

### 7.7 Recording Agent Videos (for Evaluation & Figures)

Habitat can save first-person RGB videos of trained agents navigating, which are invaluable for:
- **Qualitative analysis**: see what the agent actually looks at, how it plans turns, where it gets stuck
- **Figures for the paper**: side-by-side comparison of what each of the 4 agents sees on the same episode is the most compelling visual we can produce
- **Debugging the foveated agent**: visualize gaze position and blur pattern frame by frame

#### How to enable — use the helper script

`scripts/cluster/submit_eval.sh` wraps all the gotchas
(`load_resume_state_config=False` to respect CLI overrides,
`num_environments=1`, `eval.video_option=[disk]`, the right config key
path, the val→train split handling) into a one-liner. Training state
is never touched.

```bash
cd ~/CS503_Project
# sbatch <config_name> <ckpt_path> [num_episodes]
sbatch scripts/cluster/submit_eval.sh \
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

**Snapshot — 2026-04-26.** Paper at NeurIPS 2026 submission stage (~2 weeks from deadline). All 5 main conditions trained (single seed). Multi-seed replication, σ_max sweep, log-polar control, foveated-shifted, and stochastic gaze in flight.

### Single-seed final metrics (Table 1 of paper)

| Condition | Frames | SPL | Success | GPS R² (Gibson, no-cap) | Compass R² |
|-----------|---:|---:|---:|---:|---:|
| **Blind** | 342M | 0.59 | 0.93 | **+0.95±0.02** | **+0.81±0.08** |
| **Coarse (1×1)** | 250M | 0.67 | 0.98 | **+0.78±0.10** | **+0.64±0.10** |
| **Uniform** | 250M | **0.85** | **0.99** | −0.31±0.86 | +0.36±0.23 |
| **Foveated (fix)** | 174M (ckpt.36) | 0.78 | 0.96 | +0.06±0.88 | +0.07±0.69 |
| **Foveated (learned)** | 250M | 0.82 | 0.98 | −2.43±3.98 | −1.34±3.14 |

5-fold episode-level CV, deterministic-rollout. Bold = H1-relevant ordering (blind > coarse ≫ rich-encoder).

### Headline findings (paper §4)

1. **Encoder–memory race (H1, §4.2)**. Visual-encoder capacity tracks top-layer LSTM spatial encoding *inversely*. Bottleneck conditions hold a strong linear GPS code; rich-encoder conditions do not. Replicates on held-out MP3D (cross-dataset). **Cross-training probes at 4–5 checkpoints/condition** (paper Fig 3, `fig3_substitution_dynamics.pdf`) confirm the substitution mechanism: rich-encoder agents learn an integrated GPS code transiently (R²~0.7–0.8 at ~50M frames) then progressively unlearn its linear top-layer readability as the visual route consolidates. Bottleneck conditions hold tight high R² across the entire training trajectory.

2. **Format divergence (H2, §4.3)**. Pairwise linear CKA <10⁻⁴ between conditions; 1-NN purity = 1.000 vs. chance 0.20 on pooled hidden states; cross-condition probe transfer collapses (off-diagonal R² ≪ −800). Memory transplants (5×5 matrix, midpoint t=30) show *asymmetric* costs: bottleneck-donor states are toxic to rich-encoder recipients (uniform suffers up to 0.21 SPL drop), the reverse is benign.

3. **Probe-readable vs. policy-used dissociation (§4.5)**. Coarse has a readable GPS code (R²=+0.78) but low memory reliance (shortcut SPL drop ~5%); Uniform has no readable GPS but high memory reliance (~41% drop). Trajectory analysis of persistent-memory failures shows uniform "locks onto" the previous episode's goal location (margin +1.83m at n=46), suggesting its persistent code is a non-spatial scene/history anchor.

4. **Foveation: convergence on H1, divergence on H2 + behaviour (§4.4)**. Foveated (fix) and uniform group together as "rich-encoder pass-through" on H1 (both at chance). But MP3D held-out generalisation, shortcut SPL drop, transplant cost, and CKA all read foveation ≠ uniform. The four in-flight foveation experiments (F1–F4 sigma sweep + log-polar) test what the differential is.

### Cluster status (Izar SCITAS, 2026-04-26 16:19 CEST)

**RUNNING (8 jobs)**:
- 5 multi-seed (seed=2): `bld_s2`, `mtc_s2`, `fov_lrn_s2` (~3–4h each, just started); `uni_s2`, `fov_s2` (~70h, hitting 72h walltime today ~17:24)
- 3 foveation experiments: `fov_v2_gibson` (~21h), `fov_strong_gibson` (~20h, σ=20), `fov_logpolar_gibson` (~22h)

**PENDING (5 jobs)**:
- `foveated_sigma{2,4,12}_gibson` (cs-503 / normal QOS)
- `foveated_shifted_gibson` (normal QOS — H3 control)
- Stochastic-gaze smoke test

**Probing**: 23 across-checkpoint analyses landed (substitution mechanism finding integrated). Standard 5-condition probing complete.

### Paper artefacts

- Source: `docs/NeurIPS_2026/neurips_2026.tex`
- Build: 28 pages, 6 main figures + 7 appendix figures (PDF-only, named `figN_*` / `appfigN_*`)
- Master tracker: `docs/NeurIPS_2026/MASTER_TRACK.md`
- Friend's high-compute launch recipe: `docs/hc_launch_recipe.md`

---

## 9. Status TODO

Phases 1–3 below are the original course-era milestones. We are now in **Phase 4 + multi-seed wait + follow-up experiments**.

### Phase 1–3 (Mostly complete)

- [x] **Phase 1: Infrastructure & training setup.** Habitat + Gibson on SCITAS, configs for 5 conditions, foveation transform + foveated policies (fixed + learned + shifted + stochastic), foveation PoC frozen at σ=8, eval video pipeline.
- [x] **Phase 2: Training.** All 5 conditions trained to single-seed convergence (frame budgets in §8 table). DD-PPO NaN-corruption discovered + patched (§2.4.1). Foveated (fix) ckpt.36 used due to silent NaN window in original run.
- [x] **Phase 3: Probing & analysis.**
  - [x] Comprehensive probing pipeline (12 experiments per condition × 5 conditions × multiple checkpoints)
  - [x] Cross-condition: linear CKA, 1-NN purity, probe transfer
  - [x] Behavioral interventions: 5×5 memory transplant matrix, shortcut discovery (paired same-scene-different-goal)
  - [x] Substitution mechanism: across-training probing on all 5 conditions × 4-5 checkpoints
  - [x] H3 instrumentation: gaze-collapse measurement, foveated-shifted control queued
  - [ ] 4 coarse-recipient transplant cells (queued, low priority)

### Phase 4: Paper finalisation (current)

- [x] Bio-first narrative restructure (paper §5.2)
- [x] H1 / H2 / H3 hypothesis framing reconciled with current findings
- [x] Figure pipeline: 13 active figure scripts, PDF-only output, named `figN_*` / `appfigN_*`
- [x] `\uncertain` / `\pendnote` macros for color-coded interpretive claims and pending-data caveats
- [x] Substitution mechanism integrated (§4.2 cross-training paragraph + Fig 3)
- [x] Coarse rename (Matched → Coarse) global with collision fixes
- [x] Tier 1+2 polish on §4.1 / §4.2 / §4.3 / §4.4 / §4.5 / §4.7
- [x] H3 reframed as foveated-shifted control (not 6th agent)
- [x] §3.2 probe grouping by results-section purpose (H1/H2/boundary)
- [x] §5.1 synthesis figure: 5 conditions in 2D (H1×H2) → expanded to 3-axis (with shortcut SPL drop as marker size)
- [x] §5.2–§5.5 + §6 + §1 narrative consistency audit
- [ ] Multi-seed land → update single-seed magnitudes throughout paper
- [ ] σ_max sweep + log-polar + foveated-shifted land → update §4.4 + appendix table
- [ ] Optional: stochastic gaze land → upgrade §4.6 H3 from "control" to "test"

### Phase 5: Follow-up & outstanding

- Items left for follow-up (in paper §5.5 limitations):
  - Encoder-resolution scaling sweep (matched at {32, 64, 96, 128, 192} input pixels)
  - Direct causal test: visual ablation mid-rollout in rich-encoder agents
  - Architecture scope: transformer-based navigators, attention-over-history (paper claims interface-level generalisation but doesn't verify)

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

**The gap we fill**: Prior emergent-cognitive-maps work uses either no vision (blind) or spatially uniform vision; prior foveated-RL work doesn't probe internal memory. No existing work systematically varies *sensor structure* (no vision → resolution-collapsed → uniform → foveated fix → foveated learned) holding task and architecture fixed and asks how the structure of the visual sensor reshapes the *format* of learned spatial memory. Our paper isolates this principle in a controlled silicon case study and links it to convergent biological evidence (paper §5.2).

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
