# Data setup for friend's cluster

The experiments in this folder (`encoder_capacity_scaling.md`,
`multiseed_robustness.md`) only train on Gibson. Friend needs:

| Item | Size | Path in our codebase expects |
|------|------|------------------------------|
| Gibson scene `.glb` files | ~13 GB | `$HABITAT_DATA/scene_datasets/gibson/*.glb` |
| Gibson PointNav episodes | ~700 MB | `$HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/` |
| Gibson+MP3D merged episode symlinks | tiny | `$HABITAT_DATA/datasets/pointnav/mp3d_gibson/v1/train/` |

MP3D scenes are **not** needed for these experiments (only used for
eval, which happens on Izar). Total: ~14 GB.

## Option A — rsync from Izar (fastest if friend's cluster allows outbound SSH to Izar)

On friend's cluster:

```bash
# Pick a location — typically /scratch or similar
export HABITAT_DATA=/scratch/$USER/habitat_data
mkdir -p $HABITAT_DATA/scene_datasets

# Pull Gibson scenes (13GB, ~5-30 min depending on bandwidth)
rsync -avz --progress \
    wxu@izar.epfl.ch:/scratch/izar/wxu/habitat_data/scene_datasets/gibson/ \
    $HABITAT_DATA/scene_datasets/gibson/

# Pull Gibson PointNav episodes (~700MB)
rsync -avz --progress \
    wxu@izar.epfl.ch:/scratch/izar/wxu/habitat_data/datasets/pointnav/gibson/ \
    $HABITAT_DATA/datasets/pointnav/gibson/

# Pull the mp3d_gibson merged symlinks dir (tiny)
rsync -avz --progress \
    wxu@izar.epfl.ch:/scratch/izar/wxu/habitat_data/datasets/pointnav/mp3d_gibson/ \
    $HABITAT_DATA/datasets/pointnav/mp3d_gibson/
```

Replace `wxu` with wenxuan's SCITAS username. This requires either
(a) friend's cluster's outbound SSH is not firewalled, or (b) an SSH
key on Izar.

Alternatively pull FROM Izar side if friend's cluster accepts inbound:
```bash
# On Izar (wenxuan):
rsync -avz --progress \
    /scratch/izar/wxu/habitat_data/scene_datasets/gibson/ \
    $FRIEND_USER@$FRIEND_HOST:$FRIEND_HABITAT_DATA/scene_datasets/gibson/
```

## Option B — direct download via habitat-sim utility (no license needed)

The Gibson Habitat distribution is freely downloadable with Habitat-Sim's
built-in utility:

```bash
# In the habitat conda env
python -m habitat_sim.utils.datasets_download \
    --uids habitat_example_scenes gibson_habitat \
    --data-path $HABITAT_DATA
```

Check `habitat_sim.utils.datasets_download --help` for the exact uid
names (may be `gibson_tiny`, `gibson_habitat`, or similar depending
on the habitat-sim version).

Then separately get the PointNav episode files via our script:

```bash
# Sets up $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/
bash scripts/cluster/download_gibson_0plus.sh
```

This downloads only the .json.gz episode files (small), and cross-checks
them against the .glb scenes you just downloaded.

## Option C — if friend already has Habitat running on their cluster

Many embodied-AI groups already have Gibson set up. Just point
`$HABITAT_DATA` at the existing install; the config files use relative
paths that resolve from there.

## Sanity check after download

```bash
ls $HABITAT_DATA/scene_datasets/gibson/*.glb | wc -l      # should be 492 (or >= 411)
ls $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/content/*.json.gz | wc -l  # should be 411
```

If both are correct, launch a 1-GPU dry-run:

```bash
sbatch --gres=gpu:1 --time=01:00:00 --ntasks-per-node=1 \
    scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched32_gibson
```

and watch for 2-3 checkpoint saves in the first hour to confirm
everything's wired up. Then kick off the full 4 resolution-sweep jobs.
