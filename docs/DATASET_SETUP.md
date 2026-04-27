# Dataset setup — friend's hc cluster

This document covers the **data** required to reproduce / extend our
training pipeline (Gibson + MP3D point-navigation).  It assumes you
have already signed the relevant agreements and have access to the
download links.

Software setup (Habitat-sim, habitat-lab, conda env) is covered
separately in `experiments/foveation_transform_fix_retrain.md`.

---

## Final directory structure

The project's hydra configs assume this layout under one root, by
default referenced via the env var `$HABITAT_DATA` or the project
default `<project_root>/data/`:

```
<HABITAT_DATA>/
├── scene_datasets/
│   ├── gibson/                 # Gibson 3D mesh files (.glb + .navmesh)
│   └── mp3d/                   # Matterport3D scene meshes
└── datasets/
    └── pointnav/
        ├── gibson/v1/          # Habitat Gibson PointNav episodes (incl. train_extra_large)
        ├── mp3d/v1/            # Habitat MP3D PointNav episodes
        └── mp3d_gibson/v1/     # OUR merged dataset (recreated via script)
```

Total disk footprint after extraction: **~35 GB** (mostly the scene
meshes; episodes are ~700 MB; merged dataset is symlinks only).

---

## 1. Gibson scene meshes — ~11 GB zipped, ~13 GB extracted

Source (signed-agreement Gibson Database):

```
https://storage.googleapis.com/gibson_scenes/gibson_habitat_trainval.zip
```

Download + extract:

```bash
mkdir -p $HABITAT_DATA/scene_datasets
cd $HABITAT_DATA/scene_datasets
curl -L -o gibson_habitat_trainval.zip \
    https://storage.googleapis.com/gibson_scenes/gibson_habitat_trainval.zip
unzip gibson_habitat_trainval.zip       # extracts top-level dir gibson/
rm gibson_habitat_trainval.zip
```

Verify: `ls scene_datasets/gibson/ | wc -l` should give **~984 files**
(2 per scene: `<scene>.glb` mesh + `<scene>.navmesh`).

---

## 2. Matterport3D scene meshes — ~21 GB extracted

Matterport3D ships a Python download script `download_mp.py` rather
than direct file links.  After signing the MP3D Terms of Use the
access email gives you the URL to the script (not an attachment); you
need to download the script yourself first, then run it.

```bash
mkdir -p $HABITAT_DATA/scene_datasets
cd $HABITAT_DATA/scene_datasets

# 1. Fetch the download script (~10 KB).
wget http://kaldir.vc.cit.tum.de/matterport/download_mp.py

# 2. Use it to pull only the Habitat-sim subset (~21 GB instead of 1.3 TB).
python download_mp.py \
    -o $HABITAT_DATA/scene_datasets/mp3d/ \
    --task habitat \
    --type matterport_mesh
```

`--task habitat` filters to only the files Habitat needs (skips
panoramic images / depth / etc., saves ~1 TB).  When prompted, accept
the Terms of Use to continue.

**Important — flatten the directory structure**: by default `download_mp.py` saves files under
`<out>/v1/tasks/habitat/<scene_id>/...` (with sibling `v1/scans/`).  Habitat-sim
expects `<out>/<scene_id>/<scene_id>.glb` directly.  Flatten with:

```bash
cd $HABITAT_DATA/scene_datasets/mp3d
mv v1/tasks/habitat/* .
rm -rf v1            # optional — removes the now-empty original tree
```

Verify: `ls scene_datasets/mp3d/ | wc -l` should give **~85--90 scenes**
(directories), each containing 4 files: `<id>.glb`, `<id>.house`,
`<id>.navmesh`, `<id>_semantic.ply`.

If you only need the 18 MP3D test scenes we use for held-out
evaluation, you can pass `--id 17DRP5sb8fy` (etc.) for each one to
download a specific scan.  The full id list is at
`http://kaldir.vc.cit.tum.de/matterport/v1/scans.txt`.

---

## 3. PointNav episode datasets — ~700 MB

Two standard Habitat datasets + our merged symlink layer.

### 3a. Gibson-0+ PointNav (411 scenes, ~325 MB extracted)

> **CRITICAL**: download `pointnav_gibson_0_plus_v1.zip` (the **0+**
> variant), NOT `pointnav_gibson_v1.zip`.  These are two different
> files on the same Habitat URL prefix.  The standard `_v1.zip`
> extracts to `gibson/v1/{train, val, val_mini}/` (only ~72 train
> scenes) and **does not contain** the 411-scene `train_extra_large`
> split that our `mp3d_gibson` merge depends on.  If `train_extra_large/`
> is missing after step 3a, you grabbed the wrong zip.

```bash
mkdir -p $HABITAT_DATA/datasets/pointnav
cd $HABITAT_DATA/datasets/pointnav
curl -L -o pointnav_gibson_0_plus_v1.zip \
    https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_0_plus_v1.zip
unzip pointnav_gibson_0_plus_v1.zip     # extracts to gibson/v1/train_extra_large/
rm pointnav_gibson_0_plus_v1.zip
```

After extraction you should have:

```
datasets/pointnav/gibson/v1/train_extra_large/
├── train_extra_large.json.gz
└── content/                    # 411 per-scene .json.gz files
```

A helper script `scripts/cluster/download_gibson_0plus.sh` automates
both this and step 3b together; you can use either approach.

### 3b. MP3D PointNav v1 (~380 MB extracted)

```bash
cd $HABITAT_DATA/datasets/pointnav
curl -L -o pointnav_mp3d_v1.zip \
    https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip
unzip pointnav_mp3d_v1.zip               # extracts to mp3d/v1/{train,val,test}/
rm pointnav_mp3d_v1.zip
```

After extraction:

```
datasets/pointnav/mp3d/v1/
├── train/content/                # 61 scene episode files
├── val/content/                  # 11 scene episode files
└── test/content/                 # 18 scene episode files (for held-out eval)
```

### 3c. Our merged mp3d_gibson dataset (recreate via script)

This is a custom-merged train split combining **Gibson-0+ (411 scenes)
+ MP3D train (61 scenes) = 472 training scenes**, used for the H1/H2
main experiments.  It is implemented as a directory of symlinks
pointing back into the standard datasets you downloaded in 3a / 3b —
no episode data is duplicated, so the directory itself is only ~270 KB.

Recreate the symlink structure with our script (must be run AFTER 3a
and 3b complete, and within the project root):

```bash
HABITAT_DATA=$HABITAT_DATA bash scripts/data/setup_mp3d_gibson_dataset.sh
```

Or if the project-level `data/` symlink is set up (step 5):

```bash
bash scripts/data/setup_mp3d_gibson_dataset.sh
```

Expected output:

```
Done.
  Gibson scenes linked: 411
  MP3D scenes linked:   61
  Total:                472
  Output: <HABITAT_DATA>/datasets/pointnav/mp3d_gibson/v1
```

Verify: the resulting `mp3d_gibson/v1/train/content/` should contain
**472 symlinks** ending in `.json.gz`.

---

## 4. Verify the layout

After all three steps, check:

```bash
find $HABITAT_DATA -maxdepth 3 -type d | sort
```

You should see (paths may differ; tree is what matters):

```
$HABITAT_DATA
$HABITAT_DATA/datasets
$HABITAT_DATA/datasets/pointnav
$HABITAT_DATA/datasets/pointnav/gibson
$HABITAT_DATA/datasets/pointnav/gibson/v1
$HABITAT_DATA/datasets/pointnav/mp3d
$HABITAT_DATA/datasets/pointnav/mp3d/v1
$HABITAT_DATA/datasets/pointnav/mp3d_gibson
$HABITAT_DATA/datasets/pointnav/mp3d_gibson/v1
$HABITAT_DATA/scene_datasets
$HABITAT_DATA/scene_datasets/gibson
$HABITAT_DATA/scene_datasets/mp3d
```

---

## 5. Wire the dataset path into Habitat configs

Two equivalent ways:

**(a) Symlink from project default** (simplest if `$HABITAT_DATA`
matches the project's `data/` dir):

```bash
cd <project_root>
ln -s $HABITAT_DATA data
```

**(b) Set the environment variable** (used by our hydra configs):

```bash
export HABITAT_DATA=/path/to/your/habitat_data
```

The configs in `habitat_configs/ddppo_pointnav_*.yaml` use relative
paths like `data/scene_datasets/gibson/<scene>.glb`, which resolve via
the symlink or via `$HABITAT_DATA` once exported.

---

## 6. Sanity check: load one scene

```bash
cd <project_root>
conda activate habitat
python -c "
import habitat
from src.utils.habitat_env import load_habitat_config
cfg = load_habitat_config(
    'pointnav/ddppo_pointnav_blind_gibson',
    '',
    overrides=['habitat.dataset.split=val',
               'habitat.environment.iterator_options.shuffle=False'],
)
env = habitat.Env(config=cfg.habitat)
obs = env.reset()
print('Scene:', env.current_episode.scene_id)
print('Start:', env.current_episode.start_position)
print('Goal:', env.current_episode.goals[0].position)
env.close()
"
```

If this prints scene + start + goal without errors, the dataset setup
is complete.

---

## Troubleshooting

- **"Scene mesh not found"** during `env.reset()`:
  - Check that `scene_datasets/gibson/<scene>.glb` exists and is readable.
  - Symlink / `$HABITAT_DATA` may be pointing to the wrong directory.
- **"Episode iterator empty"** or **"No such file: train_extra_large"**:
  - You probably downloaded the standard `pointnav_gibson_v1.zip` instead
    of `pointnav_gibson_0_plus_v1.zip`.  Re-download the 0+ variant
    (step 3a).
  - Or the merged `mp3d_gibson` symlinks weren't created.  Re-run
    `bash scripts/data/setup_mp3d_gibson_dataset.sh`.
- **`mp3d_gibson` symlinks point to non-existent files**:
  - Step 3a or 3b incomplete.  Verify
    `gibson/v1/train_extra_large/content/` has 411 files and
    `mp3d/v1/train/content/` has 61 files, then re-run the merge script.
- **Gibson `.navmesh` missing for some scenes**: a few scenes in the
  Habitat-trainval Gibson zip have only `.glb` (no `.navmesh`); habitat
  will regenerate the navmesh on first use, which can take 1-2 min per
  scene.  This is normal.
