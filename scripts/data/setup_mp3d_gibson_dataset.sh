#!/bin/bash
# Recreate the mp3d_gibson merged PointNav dataset (472 training scenes:
# 411 Gibson-0+ + 61 MP3D train) as a symlink-only directory.
#
# This is the dataset our hydra configs point to (e.g.,
#   habitat.dataset.data_path: data/datasets/pointnav/mp3d_gibson/v1/{split}/{split}.json.gz
# ).  Internally it re-uses the standard Habitat PointNav datasets'
# per-scene episode files via symlinks; no episode data is duplicated.
#
# Prereq: the standard datasets must already exist:
#   $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/content/<scene>.json.gz
#   $HABITAT_DATA/datasets/pointnav/mp3d/v1/train/content/<scene>.json.gz
# (See docs/DATASET_SETUP.md for download links.)
#
# Usage:
#   HABITAT_DATA=/path/to/habitat_data bash scripts/data/setup_mp3d_gibson_dataset.sh
#
# Or with the project's default data/ symlink:
#   bash scripts/data/setup_mp3d_gibson_dataset.sh

set -e

HABITAT_DATA="${HABITAT_DATA:-$(pwd)/data}"
echo "Using HABITAT_DATA=$HABITAT_DATA"

GIBSON_SRC="$HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/content"
MP3D_SRC="$HABITAT_DATA/datasets/pointnav/mp3d/v1/train/content"

if [ ! -d "$GIBSON_SRC" ]; then
    echo "ERROR: Gibson PointNav not found at $GIBSON_SRC" >&2
    echo "       (expecting train_extra_large content dir)" >&2
    exit 1
fi
if [ ! -d "$MP3D_SRC" ]; then
    echo "ERROR: MP3D PointNav not found at $MP3D_SRC" >&2
    exit 1
fi

OUT_DIR="$HABITAT_DATA/datasets/pointnav/mp3d_gibson/v1"
mkdir -p "$OUT_DIR/train/content" "$OUT_DIR/val/content"

# ---- Scene list (411 Gibson + 61 MP3D = 472 total) ----
# Gibson-0+ scenes from train_extra_large.
gibson_scenes=$(ls "$GIBSON_SRC" | sed 's/\.json\.gz$//')
# MP3D train scenes (61).
mp3d_scenes=$(ls "$MP3D_SRC" | sed 's/\.json\.gz$//')

n_gibson=0
n_mp3d=0

for scene in $gibson_scenes; do
    src="$GIBSON_SRC/${scene}.json.gz"
    for split in train val; do
        dst="$OUT_DIR/$split/content/${scene}.json.gz"
        if [ ! -e "$dst" ]; then
            ln -s "$src" "$dst"
        fi
    done
    n_gibson=$((n_gibson + 1))
done

for scene in $mp3d_scenes; do
    src="$MP3D_SRC/${scene}.json.gz"
    for split in train val; do
        dst="$OUT_DIR/$split/content/${scene}.json.gz"
        if [ ! -e "$dst" ]; then
            ln -s "$src" "$dst"
        fi
    done
    n_mp3d=$((n_mp3d + 1))
done

# ---- Top-level placeholder JSON (Habitat needs this) ----
# train.json.gz / val.json.gz contain {"episodes": []}; the actual
# episodes come from the per-scene files in content/.
echo '{"episodes": []}' | gzip > "$OUT_DIR/train/train.json.gz"
echo '{"episodes": []}' | gzip > "$OUT_DIR/val/val.json.gz"

echo ""
echo "Done."
echo "  Gibson scenes linked: $n_gibson"
echo "  MP3D scenes linked:   $n_mp3d"
echo "  Total:                $((n_gibson + n_mp3d))"
echo "  Output:               $OUT_DIR"
