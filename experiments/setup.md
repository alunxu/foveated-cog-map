# One-time cluster setup

Start-to-finish setup for running our DD-PPO PointNav experiments on a
fresh cluster: conda environment → Habitat stack → Gibson dataset →
repo install → sanity check.

Total time: ~30–60 min (including env install) plus license wait for
Gibson if the `habitat-sim` automatic download does not cover it
(see §3 fallback).

## Requirements on the target cluster

- SLURM scheduler
- GPU nodes with CUDA 11.8 or 12.1 compatible drivers
- Outbound internet access (conda + pip + dataset download)
- Per-user scratch space ≥ 50 GB (14 GB Gibson + checkpoints + probe data)

## 1. Clone the repo

```bash
cd ~
git clone https://github.com/alunxu/foveated-cog-map.git cs503-project
cd cs503-project
```

The repo includes:
- `habitat_configs/` — all DD-PPO configs, including new `matched{32,64,96,192}_gibson`
- `src/habitat/` — custom Wijmans-faithful policies + foveated variants
- `scripts/cluster/submit_*.sh` — SLURM submission scripts
- `scripts/probing/` — probe data collection + analysis

## 2. Conda environment

```bash
# Fresh env with Python 3.9 (habitat-sim wheel constraint)
conda create -n habitat python=3.9 cmake=3.22 -y
conda activate habitat

# PyTorch — pick the CUDA tag matching the cluster's driver
# H100/H200 clusters typically run CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# If CUDA 11.8, use --index-url https://download.pytorch.org/whl/cu118

# Habitat simulator (headless — no X display needed)
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat

# Habitat lab + baselines (from source, for DD-PPO access)
cd ~
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines

# ─── Critical patch for blind policy ───
# habitat-baselines has an assertion bug when the visual encoder has 0
# input channels (blind agent). Edit
# ~/habitat-lab/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py
# and change:
#     if normalize_visual_inputs:
# to:
#     if normalize_visual_inputs and self._n_input_channels > 0:
# (exactly one site in the file). Without this, `blind_gibson` fails at
# import. Required for the multi-seed experiment's blind runs.

# Our repo's dependencies + editable install
cd ~/cs503-project
pip install -r requirements.txt
pip install -e .
```

Quick sanity:
```bash
python -c "import habitat, habitat_sim; print(habitat.__version__, habitat_sim.__version__)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 3. Gibson dataset

Gibson's `.glb` scene meshes require one of:

### 3a (recommended) — Habitat's automatic download

```bash
# Set the target path
export HABITAT_DATA=/scratch/$USER/habitat_data
mkdir -p $HABITAT_DATA

# Pull Gibson Habitat scenes (~13 GB). This is the Facebook-redistributed
# Habitat version; no separate license form for most installs.
python -m habitat_sim.utils.datasets_download \
    --uids gibson_habitat \
    --data-path $HABITAT_DATA

# Also get the free Habitat test scenes (~90 MB, useful for dry-runs)
python -m habitat_sim.utils.datasets_download \
    --uids habitat_test_scenes \
    --data-path $HABITAT_DATA
```

After download, expected layout:
```
$HABITAT_DATA/scene_datasets/gibson/<Scene>.glb   (411+ files)
$HABITAT_DATA/scene_datasets/habitat-test-scenes/ (small)
```

### 3b (fallback) — Stanford Gibson license

If `habitat_sim.utils.datasets_download --uids gibson_habitat` fails
(possible on some versions), apply at
`http://gibsonenv.stanford.edu/database/` (approval typically within
24h), then follow Stanford's download instructions. The final `.glb`
files go into `$HABITAT_DATA/scene_datasets/gibson/`.

### 3c — PointNav episode files (always required, no license)

These are the per-scene episode specs (start/goal pairs). ~700 MB.

```bash
cd ~/cs503-project
bash scripts/cluster/download_gibson_0plus.sh
```

The script:
1. Downloads `pointnav_gibson_0_plus_v1.zip` (320 MB) from Facebook
2. Extracts to `$HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/`
3. Cross-checks the 411 scene names against the `.glb` files in §3a/b;
   any scene whose `.glb` is missing will crash training on that scene.
4. Also downloads MP3D test episodes (optional — not needed for these
   experiments).

### 3d — merged dataset for config compatibility

Our configs look for a merged `mp3d_gibson/` layout. Since friend is
only training (not eval), a simple symlink suffices:

```bash
mkdir -p $HABITAT_DATA/datasets/pointnav/mp3d_gibson/v1
ln -s $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large \
      $HABITAT_DATA/datasets/pointnav/mp3d_gibson/v1/train
```

(If MP3D train episodes are also present, remove the symlink and merge
Gibson + MP3D into the same `train/content/` directory. For these
experiments, Gibson-only is sufficient.)

### 3e — symlink into habitat-lab

```bash
ln -s $HABITAT_DATA ~/habitat-lab/data
```

Required because the Hydra configs resolve `data/datasets/...` relative
to the cwd, which is `~/habitat-lab` during training.

## 4. Sanity check — dry-run training (1 GPU, 1 hour)

Before launching all 4 resolution-sweep runs, verify everything works
with a time-capped test:

```bash
cd ~/cs503-project
sbatch --gres=gpu:1 --ntasks-per-node=1 --time=01:00:00 \
    scripts/cluster/submit_train.sh \
    pointnav/ddppo_pointnav_matched32_gibson
```

Watch the log (`slurm_logs/<jobid>.out`) for:
- `Obs space: ... rgb: 32x32` (correct input resolution)
- `reward: ...` climbing above 0 within ~100 updates
- `ckpt.0.pth` saved to `data/checkpoints/matched32_gibson/` within 5–15 min

If you see those, kill the dry-run (it's a 1h test) and proceed to the
full runs. If something fails, common issues:
- **Missing `.glb`**: scene in episode file has no mesh — download §3a
  again, or delete the missing scene's episode file.
- **CUDA OOM**: reduce `num_environments` in the config (6 → 4) or run
  at lower spatial resolution.
- **Blind import error**: apply the normalize_visual_inputs patch in §2.

## 5. Run experiments

Now the experiments docs are actionable. See:

- `experiments/encoder_capacity_scaling.md` — train 4 matched variants +
  det-probe matched128 from Izar (scaling-law experiment, primary
  priority)
- `experiments/multiseed_robustness.md` — train 6 additional seeds for
  cross-seed error bars

Each run is ~10–15h on H100 for 250M frames at the default config.
With parallelism the full scaling sweep finishes in one overnight run.

## 6. Sending results back

Assume ssh between clusters is blocked — use shared cloud storage
(Google Drive, Dropbox, S3 bucket, WeTransfer) for transfers.

Per completed experiment, friend sends:

1. **Final checkpoint** (`data/checkpoints/<cfg>/ckpt.49.pth`, ~400 MB)
2. **Probe npz** (`data/probing_data/<cfg>_det.npz`, 1–2 GB — optional
   but speeds up downstream re-analysis on the paper side)
3. **Probe analysis JSON** (`data/probing_results/<cfg>_det_analysis.json`,
   ~60 KB — this is the small summary that actually goes into the paper)

Per-experiment totals:
- encoder_capacity_scaling (4 runs): ~1.6 GB ckpts + 4–8 GB npz + ~300 KB jsons
- multiseed_robustness (6 runs): ~2.4 GB ckpts + 6–12 GB npz + ~400 KB jsons

If cloud-storage quota is tight, prioritise: **JSON first** (tiny,
directly usable), **npz second** (lets us redo analyses without
re-probing), **ckpts last** (reproducibility, not strictly required
for the paper).
