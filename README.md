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

See `docs/manuscript/main.pdf` for the writeup with full motivation and
results. **This README only covers code structure and setup.**

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
│   ├── cluster/               # Job submission scripts (RCP RunAI + Izar SLURM)
│   │   ├── submit_blind_retrain.sh / submit_blind_resume.sh
│   │   ├── submit_F2_normaliser.sh / submit_F_LP2.sh
│   │   ├── submit_5cond_eval.sh / submit_5cond_eval_mp3d_test.sh
│   │   ├── submit_probe_collect_rcp.sh
│   │   ├── submit_transplant_rcp.sh / submit_shortcut_rcp.sh
│   │   └── _*_inner.sh        # PVC-resident scripts called by submitters
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
│   ├── paper_figures/         # Figure generation (matplotlib)
│   │   ├── make_consumption_2panel.py   # Figure 5 (canonical 3-panel)
│   │   ├── make_temporal_maps_figure.py # Figure 4
│   │   ├── make_5x5_transplant_matrix.py
│   │   └── render_5cond_appendix.py     # Appendix battery
│   └── data/                  # Dataset utilities (download, split)
├── results/                   # Probe + eval JSONs (small text artifacts)
│   ├── probing_results/
│   ├── shortcut_results/
│   └── transplant_results/
├── docs/
│   ├── manuscript/            # LaTeX source for the paper (main.tex + figs + literature.bib)
│   └── release/               # Release notes
├── docker/
│   └── Dockerfile             # RCP container build (Habitat + DD-PPO)
├── data/                      # Habitat-format datasets (gitignored; populated on cluster)
├── requirements.txt           # Pip deps (for analysis-only setups; full env needs Docker / conda)
└── pyproject.toml             # Editable-install metadata
```

**Where the heavy data lives** (not in git):

| Cluster | Path | Contents |
|---|---|---|
| RCP PVC `dhlab-scratch` | `/scratch/wxu/habitat-lab-izar/data/` | Canonical Habitat data (~36 GB): Gibson + MP3D scene meshes + PointNav episode JSONs |
| RCP PVC `dhlab-scratch` | `/scratch/wxu/habitat_checkpoints_rcp/` | Trained checkpoints (`dh-blind/`, `dh-probe-1..4/`, `dh-fnorm/`, `dh-flp2/`) + probing NPZs |
| Izar SLURM | `/scratch/izar/wxu/habitat_data/` | Mirror of habitat data (rsync'd from RCP) |

---

## Setup

There are three paths depending on what you want to do.

### Path 1 — RCP cluster (recommended, full training + eval pipeline)

The Dockerfile at `docker/Dockerfile` is the source of truth for the env.
Pre-built image is on the EPFL registry:

```
registry.rcp.epfl.ch/dhlab-wxu/habitat:v2
```

Just use this image in any RunAI submission; no setup required on your
side. The image bundles:

- Python 3.9 + Miniconda
- `habitat-sim 0.3.3` (with bullet, headless)
- `habitat-lab` + `habitat-baselines` (from source, with a one-line patch
  for the blind-policy `normalize_visual_inputs` guard)
- `torch 2.1.0 + CUDA 12.1`, `protobuf<5`, `numpy<2`, `pillow 10.4`

To rebuild the image (rare):

```bash
bash docker/build_with_kaniko.sh
```

### Path 2 — Izar cluster (SLURM, conda-based)

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

### Path 3 — local analysis-only (no GPU, no Habitat sim)

If you only want to re-analyse existing NPZ files (probing, plotting,
paper figure regen) without running new rollouts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

You will **not** be able to run `scripts/eval/eval_paper_5cond.py`,
`scripts/probing/collect.py`, `scripts/eval/shortcut.py`, or anything that
spins up `habitat.Env(...)`. Probing/figure scripts that load existing
NPZs work fine.

### Habitat data setup (cluster only)

Configs reference paths relative to a fixed Habitat data root. On RCP this
is auto-mounted via the `dhlab-scratch` PVC. On Izar, set:

```bash
export HABITAT_DATA_DIR=/scratch/izar/$USER/habitat_data
```

And either rsync from RCP (the canonical copy lives in
`/scratch/wxu/habitat-lab-izar/data/`) or use the dataset downloader:

```bash
bash scripts/cluster/download_gibson_0plus.sh   # 411 Gibson + 61 MP3D-train scenes
```

For the Wijmans-protocol eval, you also need the MP3D test PointNav
episodes (`mp3d/v1/test/test.json.gz`, ~43KB) and the 18 MP3D test scene
meshes — these are already on RCP under
`/scratch/wxu/habitat-lab-izar/data/`.

---

## Common workflows

### Train a single condition (RCP)

```bash
# Blind from scratch (250M frames, ~21h on 4×A100; or resume from existing ckpt)
bash scripts/cluster/submit_blind_retrain.sh        # fresh
bash scripts/cluster/submit_blind_resume.sh         # resume from latest checkpoint

# Sighted conditions are pre-trained as dh-probe-1..4 (coarse / foveated /
# uniform / log-polar). To re-train one, edit the relevant submit script's
# config-name override and run it.

# Normaliser ablations
bash scripts/cluster/submit_F2_normaliser.sh        # foveated + normaliser
bash scripts/cluster/submit_F_LP2.sh                # log-polar + normaliser
```

### Evaluate trained agents (SPL, Success, mean steps)

```bash
# Wijmans-protocol: 1008 episodes on 18 held-out MP3D test scenes (canonical)
bash scripts/cluster/submit_5cond_eval_mp3d_test.sh

# Train-pool sample: 500 random episodes from training scenes (faster, less rigorous)
bash scripts/cluster/submit_5cond_eval.sh
```

### Collect hidden states for probing

```bash
# One condition (writes /scratch/.../probing_data_rcp/<cond>_det.npz)
bash scripts/cluster/submit_probe_collect_rcp.sh coarse 500     # 500 deterministic rollouts
```

### Run downstream probe analyses on collected NPZ

```bash
python scripts/probing/analyze.py \
  --data /scratch/.../coarse_det.npz \
  --out  /tmp/coarse_probe.json
```

### Memory transplant + shortcut paradigm

```bash
bash scripts/cluster/submit_transplant_rcp.sh     # 5×5 cross-condition memory transplant
bash scripts/cluster/submit_shortcut_rcp.sh      # paired-episode Tolman-style test
```

### Regenerate paper figures

```bash
python scripts/paper_figures/make_consumption_2panel.py    # Figure 5 (consumption)
python scripts/paper_figures/make_temporal_maps_figure.py  # Figure 4 (temporal)
python scripts/paper_figures/render_5cond_appendix.py      # Appendix figures
```

---

## Cluster cheatsheet

| Task | RCP (RunAI) | Izar (SLURM) |
|---|---|---|
| Submit training | `runai-rcp-prod submit ... --command -- bash -c "..."` | `sbatch scripts/cluster/submit_*.sh` |
| Watch logs | `kubectl logs -f <pod>` | `tail -f slurm_logs/<jobid>.out` |
| Cancel | `runai-rcp-prod delete job <name>` | `scancel <jobid>` |
| GPU resources | A100 80GB / H100 (4-GPU pods) | V100 / A100 (varies) |
| Storage | `/scratch/wxu/...` (PVC, persistent) | `/scratch/izar/wxu/...` (Lustre, persistent) |

When submitting on RCP, use the **PVC-resident inner-script pattern** to
avoid the runai-cli inline-quoting bug: the inner script lives in
`scripts/cluster/_*_inner.sh` (on the PVC) and the submit script just
invokes it via `bash /scratch/.../scripts/cluster/_*_inner.sh ARGS`.

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

- Paper draft: `docs/manuscript/main.tex` → compile via `cd docs/manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main`
- Old README (with full motivation, biology background, hypotheses): `README.OLD.md`
