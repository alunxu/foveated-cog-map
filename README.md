# Sensor Structure Shapes Cognitive Maps in Navigation Agents

LSTM-based PointGoal navigation agents trained in [Habitat](https://aihabitat.org/) where
**only the visual sensor varies** across conditions; everything else (task,
reward, architecture, training algorithm, dataset) is held fixed. Used as a
controlled in-silico testbed for studying how sensor structure shapes the
format of learned spatial memory.

Conditions: `blind`, `coarse` (48×48), `foveated` (256×256 + Gaussian blur),
`foveated_logpolar` (log-polar resampling), `uniform` (256×256). Two
normaliser-ablation variants: `F2` (foveated + RunningMeanAndVar),
`F-LP2` (log-polar + RunningMeanAndVar).

**This README only covers code structure and Izar setup.** The paper
write-up (motivation, methods, results, biology background) is
distributed separately.

---

## Repository layout

```
.
├── habitat_configs/           # 23 .yaml — Hydra training configs (one per condition + sweeps)
├── src/
│   ├── habitat/               # Custom policies
│   │   ├── wijmans_policy.py            # 3-layer LSTM + Wijmans sensor stack
│   │   ├── foveated_policy.py           # Gaussian-blur foveation
│   │   ├── foveated_normalised_policy.py # F2 (normaliser enabled)
│   │   ├── foveated_logpolar_policy.py  # log-polar foveation
│   │   ├── foveated_logpolar_normalised_policy.py  # F-LP2
│   │   └── wijmans_sensors.py           # GPS / compass / goal-in-start-frame
│   └── utils/                 # Probe helpers, env loader, common utilities
├── scripts/
│   ├── eval/                  # SPL/Success evaluators, shortcut, transplant
│   │   ├── eval_paper_5cond.py          # Main eval: SPL, Success, mean steps
│   │   ├── shortcut.py / shortcut_with_trajectories.py
│   │   ├── transplant.py                # Memory-transplant 5×5 matrix
│   │   ├── probe_agent.py               # h₂ collection + Ridge/MLP probes
│   │   └── gps_ablation_analyze.py
│   ├── probing/               # Per-condition representational analyses
│   │   ├── collect.py                   # h₂ collection (PointNav rollout → NPZ)
│   │   ├── analyze.py                   # Full probing battery (Ridge + MLP + control)
│   │   ├── unaligned_cka.py             # Cross-condition CKA
│   │   ├── temporal_probe.py / extended_lag_probe.py
│   │   ├── analyze_subspace_divergence.py / population_coding_analysis.py
│   │   └── ... (~30 probe-family scripts)
│   ├── paper_figures/         # Figure generation (matplotlib); see paper_figures/STATUS.md for the script→figure index
│   │   ├── make_magnitude_3panel.py     # Figure 2 (magnitude)
│   │   ├── make_format_2panel.py        # Figure 3 (format)
│   │   ├── make_temporal_maps_figure.py # Figure 4 (temporal)
│   │   ├── make_consumption_2panel.py   # Figure 5 (consumption)
│   │   ├── make_5x5_transplant_matrix.py # Appendix figa7a (5×5 transplant)
│   │   └── render_5cond_appendix.py     # Appendix figa9a/9b/13 (population-coding battery)
│   ├── cluster/               # Izar SLURM submit scripts + cross-cluster utilities (see scripts/cluster/README.md)
│   └── data/                  # Dataset utilities (download, split)
├── results/                   # Probe + eval JSONs (small text artifacts)
│   ├── probing_results/
│   ├── shortcut_results/
│   └── transplant_results/
├── docker/
│   └── Dockerfile             # Reference container build (Habitat + DD-PPO)
├── data/                      # Habitat-format datasets (gitignored; populated on cluster)
├── requirements.txt           # Pip deps (for analysis-only setups; full env needs conda)
└── pyproject.toml             # Editable-install metadata
```

**Where the heavy data lives** (not in git):

| Path | Contents |
|---|---|
| `/scratch/izar/$USER/habitat_data/` | Canonical Habitat data (~36 GB): Gibson + MP3D scene meshes + PointNav episode JSONs |
| `/scratch/izar/$USER/habitat_checkpoints/` | Trained checkpoints (one dir per condition) + probing NPZs |

---

## Setup (Izar, SLURM + conda)

```bash
# 1. Load CUDA module
module load gcc/12.2.0 cuda/12.1.1

# 2. Create env (Python 3.9; matches Dockerfile)
conda create -n habitat python=3.9 cmake=3.22 -c conda-forge -y
conda activate habitat

# 3. Habitat-sim from aihabitat channel
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y

# 4. Torch matching Izar CUDA
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Habitat-lab + habitat-baselines from source (apply blind-policy patch first)
git clone --branch stable --depth 1 https://github.com/facebookresearch/habitat-lab.git
sed -i 's|if normalize_visual_inputs:|if normalize_visual_inputs and self._n_input_channels > 0:|' \
    habitat-lab/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py
pip install -e habitat-lab/habitat-lab
pip install -e habitat-lab/habitat-baselines

# 6. Pin downgraded deps + the rest of requirements.txt
pip install 'protobuf>=4.21,<5' 'pillow==10.4.0' 'opencv-python<4.10' 'numpy<2'
pip install -r requirements.txt

# 7. This repo (editable)
pip install -e .
```

### Habitat data setup

Configs reference paths relative to a fixed Habitat data root. On Izar:

```bash
export HABITAT_DATA_DIR=/scratch/izar/$USER/habitat_data
```

Layout under `HABITAT_DATA_DIR/`:

```
scene_datasets/
  gibson/              # 411 .glb + .navmesh (Gibson 0+ split)
  mp3d/                # 90 MP3D scenes (.glb + region metadata)
datasets/pointnav/
  gibson/v1/{train,val}/...
  mp3d/v1/{train,val,test}/test.json.gz   # 1008 Wijmans eval episodes
```

For the Wijmans-protocol eval, you also need the MP3D test PointNav
episodes (`mp3d/v1/test/test.json.gz`, ~43 KB) and the 18 MP3D test scene
meshes (~6 GB) under `HABITAT_DATA_DIR/scene_datasets/mp3d/`.

---

## Common workflows

All scripts below assume `conda activate habitat` + the Habitat data root
exported as in **Setup → Habitat data setup**. On Izar wrap each invocation in an
`sbatch` job (1 GPU, `--time=24:00:00` for training, `--time=04:00:00`
for eval / probing).

### Train a single condition

```bash
# DD-PPO entry point — config selects the condition + the sensor stack
python -m habitat_baselines.run \
  --config-name=pointnav/ddppo_pointnav_blind_gibson \
  habitat_baselines.total_num_steps=2.5e8 \
  habitat_baselines.num_environments=16 \
  habitat_baselines.rl.ppo.num_steps=256
```

Swap the config to switch condition: `*_coarse_*`, `*_foveated_*`,
`*_uniform_*`, `*_foveated_logpolar_*` (see `habitat_configs/`).

### Evaluate trained agents (SPL, Success, mean steps)

```bash
# Wijmans-protocol: 1008 episodes on 18 held-out MP3D test scenes (canonical)
python scripts/eval/eval_paper_5cond.py \
  --config pointnav/ddppo_pointnav_foveated_gibson \
  --ckpt   /path/to/checkpoints/foveated/ckpt.49.pth \
  --data-path ${HABITAT_DATA_DIR}/datasets/pointnav/mp3d/v1/test/test.json.gz \
  --split test --no-sample \
  --out    results/eval/foveated_mp3d_test.json
```

### Collect hidden states + run probing

```bash
# 1. Collect deterministic rollouts (writes <out>.npz)
python scripts/eval/probe_agent.py \
  --config pointnav/ddppo_pointnav_foveated_gibson \
  --ckpt   /path/to/checkpoints/foveated/ckpt.49.pth \
  --episodes 500 --out probing_data/foveated_det.npz

# 2. Run probe battery
python scripts/probing/analyze.py \
  --data probing_data/foveated_det.npz \
  --out  results/probing/foveated_probe.json
```

### Memory transplant + shortcut paradigm

```bash
python scripts/eval/transplant.py --out results/transplant_results/
python scripts/eval/shortcut.py   --out results/shortcut_results/
```

### Regenerate paper figures

```bash
python scripts/paper_figures/make_magnitude_3panel.py      # Figure 2 (magnitude)
# Default output is fig_magnitude.pdf; rename to fig2_magnitude.pdf for paper:
mv docs/manuscript/fig/fig_magnitude.pdf docs/manuscript/fig/fig2_magnitude.pdf
python scripts/paper_figures/make_format_2panel.py         # Figure 3 (format)
python scripts/paper_figures/make_temporal_maps_figure.py  # Figure 4 (temporal)
python scripts/paper_figures/make_consumption_2panel.py    # Figure 5 (consumption)
python scripts/paper_figures/render_5cond_appendix.py      # Appendix figures
```

Each figure reads from `/tmp/rcp_analysis/<cond>_det_analysis.json` (probe
summaries) and `results/shortcut_results/<cond>_traj.{json,npz}` (shortcut
trajectories). See `scripts/paper_figures/STATUS.md` for the full
script→figure mapping, including legacy/stale scripts kept for git history.

---

## SLURM cheatsheet (Izar)

| Task | Command |
|---|---|
| Submit training | `sbatch --gres=gpu:1 --time=24:00:00 -p gpu run_training.sh` |
| Watch logs | `tail -f slurm_logs/<jobid>.out` |
| Cancel | `scancel <jobid>` |
| GPU resources | V100 / A100 (`-p gpu`) |
| Storage | `/scratch/izar/$USER/...` (Lustre, persistent) |

Training is GPU-bound — a single A100 reaches 250M frames in ~36h with
`num_environments=16, ppo.num_steps=256`. Eval (1008 episodes) is
single-env and runs ~2.5-4h depending on condition.

---

## Reproducibility notes

- **Unified hyperparams** across the 5 canonical conditions + F2 + F-LP2:
  `seed=0`, `total_num_steps=2.5e8`, `num_environments=16`,
  `ppo.num_steps=256`, `use_linear_lr_decay=False`.
- **Single training seed** per condition (a comparative-cognition
  modelling convention; the paper's findings come from cross-condition
  multi-method convergence rather than seed-replicate variance).
- **Eval protocol**: Wijmans 2023 Appx A.1 — 1008 episodes across 18
  held-out MP3D test scenes (`mp3d/v1/test/`), deterministic policy.
- **Probe protocol**: 500-episode deterministic rollouts; Ridge α=10
  with 5-fold episode-level CV; Hewitt–Liang permutation control.

---

## Pointers

- Hydra training configs: `habitat_configs/ddppo_pointnav_*_gibson.yaml`
- Reference container env: `docker/Dockerfile` (apt + conda pin list — also useful as a setup checklist when conda step ordering is unclear)
