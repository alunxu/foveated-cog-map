# Sensor Structure Shapes the Format of Cognitive Maps in Navigation Agents

Course project for CS503 Visual Intelligence, EPFL, Spring 2026.

We train PointGoal navigation agents in Habitat while holding the task,
reward, optimizer, recurrent architecture, non-visual sensors, and training
data fixed. The experimental manipulation is the visual sensor bandwidth:

- `blind`: no camera input
- `coarse`: low-bandwidth visual input
- `foveated`: high-acuity center with blurred periphery
- `foveated_logpolar`: log-polar/foveated sampling
- `uniform`: full-resolution visual input

The main question is whether sensor structure changes the format of the
agent's learned spatial memory. The website and final presentation summarize
the empirical findings; this repository contains the code needed to reproduce
the training, evaluation, probing, figure-generation, and website artifacts.

## What Is In This Code Submission

This repository is intended to be a code/reproducibility package, not a full
data dump. It includes:

```text
README.md                  self-contained running instructions
requirements.txt           pip dependencies for analysis + post-Habitat setup
pyproject.toml             editable-install metadata
docker/Dockerfile          reference Habitat/DD-PPO environment
habitat_configs/           Hydra configs for all sensor conditions
src/                       custom Habitat policies, sensors, transforms, utils
scripts/eval/              evaluation, hidden-state collection, interventions
scripts/probing/           probing and representation analyses
scripts/probing/world_model_probe/
                            lightweight encoder-scale x sensor pilot
scripts/paper_figures/     figure-generation scripts
scripts/cluster/           generic SLURM/cluster helpers
scripts/cluster_rcp/       RunAI/RCP submission wrappers used for this project
scripts/website/           RCP script for real rollout + h2-memory video
docs/                      GitHub Pages website and pre-rendered visual assets
presentation/              final presentation PDF
tests/                     small CPU-oriented regression tests
results/                   selected small JSON summaries used by figures/website
data/                      small JSON summaries used by legacy figures
```

The following are intentionally not included because they are large or
license-restricted:

```text
data/datasets/             Habitat datasets and scene meshes
release_ckpts/             trained checkpoints, distributed separately
outputs/                   generated slide/website scratch artifacts
literature/                downloaded papers and reading notes
large result NPZs          rollout dumps, video frames, huge ablation outputs
.git/                      version-control metadata
```

## Website Story To Code Map

The website is the most compact reading order for the project. The code is
organized to mirror that story:

| Website section / claim | Main files to inspect | What they reproduce |
|---|---|---|
| Title and five-agent setup | `habitat_configs/`, `src/habitat/wijmans_policy.py`, `src/habitat/wijmans_sensors.py`, `src/habitat/torch_foveation.py` | The controlled Habitat PointGoal agents and the five visual sensor conditions. |
| Real navigation + memory movie | `scripts/website/capture_rcp_navigation_memory.py` | RCP script for real MP3D-test rollouts with synchronized top-layer LSTM `h2` memory readouts. |
| Navigation performance | `scripts/eval/eval_paper_5cond.py`, `scripts/cluster_rcp/submit_5cond_eval_mp3d_test.sh` | SPL/success/step evaluation on held-out MP3D test episodes. |
| Spatial-memory readability | `scripts/eval/probe_agent.py`, `scripts/probing/analyze.py`, `scripts/probing/probe_cv_summary.py` | Hidden-state collection and linear/MLP position probes. |
| Format shift / scene invariance | `scripts/probing/unaligned_cka.py`, `scripts/probing/analyze_subspace_divergence.py`, `scripts/probing/extra/leave_one_scene_out.py`, `scripts/paper_figures/make_format_2panel.py` | Cross-condition geometry, scene-invariance, and format visualizations. |
| Temporal and policy-reliance results | `scripts/eval/transplant.py`, `scripts/eval/shortcut.py`, `scripts/probing/temporal_probe.py`, `scripts/paper_figures/make_temporal_maps_figure.py`, `scripts/paper_figures/make_consumption_2panel.py` | Memory-transplant, shortcut, and temporal-consumption analyses. |
| Broader encoder/sensor pilot | `scripts/probing/world_model_probe/05_run_scale_sensor.sh`, `scripts/probing/world_model_probe/08_aggregate_scale_sensor.py`, `scripts/cluster_rcp/submit_wm_scale_sensor_pilot.sh` | The lightweight 2x2 encoder-scale x sensor-constraint pilot. |
| Static webpage | `docs/index.html`, `docs/style.css`, `docs/assets/`, `docs/js/` | The final interactive summary page served by GitHub Pages. |
| Final presentation | `presentation/final_presentation.pdf`, `docs/assets/presentation_assets/` | The submitted presentation and generated slide visuals. |

## Tested Environment

The full Habitat training/evaluation pipeline was run in a Linux container on
RCP/RunAI. The tested core versions were:

```text
Python                    3.9
habitat-sim               0.3.3, headless, withbullet
habitat-lab/baselines     stable branch, installed from source
PyTorch                   2.1.0
torchvision               0.16.0
CUDA                      12.1 on cluster GPUs
numpy                     <2
protobuf                  >=4.21,<5
Pillow                    10.4.0
opencv-python             <4.10
scipy                     >=1.10
scikit-learn              >=1.3
hydra-core                >=1.3
omegaconf                 >=2.3
matplotlib                >=3.7
imageio                   >=2.31
```

`requirements.txt` contains the pip-installable dependencies. Habitat itself is
not pip-only: install `habitat-sim` through conda and install
`habitat-lab`/`habitat-baselines` from source.

## Quick Verification Without Habitat

These commands check the analysis-only parts that do not require scene meshes,
GPUs, checkpoints, or Habitat simulation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run lightweight tests:

```bash
python -m pytest tests/test_torch_foveation.py -v
```

The NaN-sanitisation test imports `habitat-baselines`, so it is skipped unless
the full Habitat stack is installed:

```bash
python -m pytest tests/test_nan_sanitisation.py -v
```

Open the website locally:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Then open `http://127.0.0.1:8000/`. The website uses pre-rendered assets in
`docs/assets/`.

## Full Habitat Setup

Use Linux for the full reproduction path. macOS can run some plotting and tests
but cannot reliably run the Habitat simulator.

```bash
# 1. Create the environment.
conda create -n habitat python=3.9 cmake=3.22 -c conda-forge -y
conda activate habitat

# 2. Install habitat-sim.
conda install habitat-sim=0.3.3 withbullet headless \
  -c conda-forge -c aihabitat -y

# 3. Install PyTorch.
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121

# 4. Install Habitat-Lab and Habitat-Baselines from source.
git clone --branch stable --depth 1 https://github.com/facebookresearch/habitat-lab.git

# Required for the blind/no-camera policy path.
sed -i 's|if normalize_visual_inputs:|if normalize_visual_inputs and self._n_input_channels > 0:|' \
  habitat-lab/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py

python -m pip install -e habitat-lab/habitat-lab
python -m pip install -e habitat-lab/habitat-baselines

# 5. Install this project.
python -m pip install 'protobuf>=4.21,<5' 'pillow==10.4.0' \
  'opencv-python<4.10' 'numpy<2'
python -m pip install -r requirements.txt
python -m pip install -e .
```

The reference container is `docker/Dockerfile`. On RCP we used the built image:

```text
registry.rcp.epfl.ch/dhlab-wxu/habitat:v2
```

## Data Layout

Set a data root:

```bash
export HABITAT_DATA_DIR=/path/to/habitat_data
```

Expected layout:

```text
${HABITAT_DATA_DIR}/
  scene_datasets/
    gibson/              Gibson 0+ scene meshes and navmeshes
    mp3d/                Matterport3D scene meshes and metadata
  datasets/pointnav/
    gibson/v1/train/
    gibson/v1/val/
    mp3d/v1/train/
    mp3d/v1/val/
    mp3d/v1/test/test.json.gz
```

The main evaluation uses the Wijmans-style MP3D held-out test split:
18 scenes and 1008 deterministic PointGoal episodes.

## Checkpoints

Training from scratch is possible but expensive. A full condition run is
approximately 250M DD-PPO frames. If using existing checkpoints, arrange them as:

```text
${CHECKPOINT_ROOT}/
  blind/ckpt.49.pth
  coarse/ckpt.49.pth
  foveated/ckpt.49.pth
  foveated_logpolar/ckpt.49.pth
  uniform/ckpt.49.pth
```

The RCP scripts use the project-specific scratch layout under
`/scratch/wxu/habitat_checkpoints_rcp/`; adjust those paths if running on a
different cluster.

## Reproduce Main Navigation Experiments

All commands below assume:

```bash
conda activate habitat
export HABITAT_DATA_DIR=/path/to/habitat_data
export CHECKPOINT_ROOT=/path/to/checkpoints
```

### Train One Agent

```bash
python -m habitat_baselines.run \
  --config-name=pointnav/ddppo_pointnav_foveated_gibson \
  habitat_baselines.total_num_steps=2.5e8 \
  habitat_baselines.num_environments=16 \
  habitat_baselines.rl.ppo.num_steps=256 \
  habitat_baselines.rl.ddppo.train_encoder=True
```

Swap the config name for other conditions:

```text
pointnav/ddppo_pointnav_blind_gibson
pointnav/ddppo_pointnav_coarse_gibson
pointnav/ddppo_pointnav_foveated_gibson
pointnav/ddppo_pointnav_foveated_logpolar_gibson
pointnav/ddppo_pointnav_uniform_gibson
```

### Evaluate SPL / Success

```bash
python scripts/eval/eval_paper_5cond.py \
  --config pointnav/ddppo_pointnav_foveated_gibson \
  --ckpt ${CHECKPOINT_ROOT}/foveated/ckpt.49.pth \
  --data-path ${HABITAT_DATA_DIR}/datasets/pointnav/mp3d/v1/test/test.json.gz \
  --split test \
  --no-sample \
  --out results/eval/foveated_mp3d_test.json
```

### Collect Hidden States And Run Probes

```bash
python scripts/eval/probe_agent.py \
  --config pointnav/ddppo_pointnav_foveated_gibson \
  --ckpt ${CHECKPOINT_ROOT}/foveated/ckpt.49.pth \
  --episodes 500 \
  --out probing_data/foveated_det.npz

python scripts/probing/analyze.py \
  --data probing_data/foveated_det.npz \
  --out results/probing_results/foveated_det_analysis.json
```

The probe target is the top recurrent layer, referred to as `h2` in the paper.
The main probes use episode-level cross-validation to avoid leakage across
steps from the same rollout.

### Memory Consumption Experiments

```bash
python scripts/eval/transplant.py --out results/transplant_results/
python scripts/eval/shortcut.py --out results/shortcut_results/
```

These scripts expect trained policies and Habitat data. They are easier to run
through the cluster wrappers in `scripts/cluster/` or `scripts/cluster_rcp/`.

## Encoder-Scale x Sensor-Constraint Pilot

This is the lightweight broader-impact pilot used to test whether perceptual
constraints can match or beat encoder scale in a small Memory-Maze setting. It
does not replace the main Habitat experiments.

Local/cluster command:

```bash
ROOT=/tmp/wmprobe_scale_sensor \
ENCODERS="dinov2_vits14 dinov2_vitb14" \
CONDITIONS="foveated uniform" \
TRAIN_TRAJ=200 EVAL_TRAJ=50 LSTM_STEPS=3000 PROBE_STEPS=5000 \
bash scripts/probing/world_model_probe/05_run_scale_sensor.sh
```

RCP wrapper:

```bash
bash scripts/cluster_rcp/submit_wm_scale_sensor_pilot.sh pilot
```

The final table is written to:

```text
${ROOT}/results/summary_scale_sensor.md
${ROOT}/results/summary_scale_sensor.json
```

## Regenerate Figures

The website includes pre-rendered figures in `docs/assets/`. To regenerate
manuscript-style figures from summary JSONs:

```bash
python scripts/paper_figures/make_magnitude_3panel.py
python scripts/paper_figures/make_format_2panel.py
python scripts/paper_figures/make_temporal_maps_figure.py
python scripts/paper_figures/make_consumption_2panel.py
python scripts/paper_figures/render_5cond_appendix.py
```

Some figure scripts read analysis files from historical RCP/local paths such as
`/tmp/rcp_analysis` and `/tmp/rcp_analysis_v3`. If rerunning from a fresh
machine, first run the evaluation/probing commands above or edit the path
constants in the figure scripts to point to your regenerated JSONs.

## Recreate Website Rollout Video

The website video was generated on RCP to avoid copying raw rollout frames:

```bash
python scripts/website/capture_rcp_navigation_memory.py \
  --episode-index 0 \
  --dataset-split test \
  --data-path data/datasets/pointnav/mp3d/v1/test/test.json.gz \
  --excursion-dir /scratch/wxu/habitat_checkpoints_rcp/excursion_results \
  --out /scratch/wxu/habitat_checkpoints_rcp/website_media/real_navigation_memory.mp4
```

It renders real simulator rollouts and synchronizes them with a linear readout
from the real top-layer LSTM `h2` state to episode-relative x/z position.

## File Hierarchy

```text
habitat_configs/
  ddppo_pointnav_*_gibson.yaml
    Hydra configs for training/evaluating each visual sensor condition.

src/habitat/
  wijmans_policy.py
    3-layer LSTM PointGoal policy and custom visual encoder wiring.
  wijmans_sensors.py
    Non-visual sensor definitions: GPS, compass, goal vector, action history.
  torch_foveation.py
    Differentiable Gaussian and log-polar foveation transforms.
  foveated_*_policy.py
    Condition-specific policy registrations.

src/utils/
  habitat_env.py
    Shared Habitat config loading, policy loading, rollout helpers.
  probing.py
    Common probing utilities.

scripts/eval/
  eval_paper_5cond.py
    SPL/success evaluation for the five canonical agents.
  probe_agent.py
    Deterministic rollout collection for hidden-state probes.
  transplant.py
    Memory-transplant intervention.
  shortcut.py
    Shortcut/trajectory intervention for policy reliance.

scripts/probing/
  analyze.py
    Main Ridge/MLP probing battery.
  temporal_probe.py, population_coding_analysis.py, unaligned_cka.py
    Additional representational analyses.
  world_model_probe/
    Lightweight Memory-Maze scale x sensor pilot.

scripts/paper_figures/
  make_*figure*.py
    Matplotlib scripts for paper/presentation figures.

scripts/cluster/
  Generic SLURM-style helpers for cluster execution.

scripts/cluster_rcp/
  RCP/RunAI wrappers used in the actual project. These include hard-coded
  project paths and are included for verification; adapt paths before reuse.

docs/
  index.html, figures.html, style.css, js/, assets/
    Static GitHub Pages website, interactive figures, and pre-rendered assets.

presentation/
  final_presentation.pdf
    Final submitted presentation.

tests/
  test_torch_foveation.py
    CPU test for foveation geometry.
  test_nan_sanitisation.py
    Habitat-baselines-dependent regression test for NaN protection.
```

## Notes On Reproducibility Scope

- The code is designed for verification and reproduction, but full training is
  compute-intensive and requires Habitat datasets/checkpoints.
- The submitted zip includes code and small summary artifacts only.
- To exactly reproduce all final figures, use the same checkpoint set and
  MP3D/Gibson data layout described above.
- The RCP-specific scripts document the actual commands used for the final
  cluster runs, but they require EPFL RCP access and path edits outside the
  original project account.
