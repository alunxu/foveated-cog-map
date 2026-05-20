#!/usr/bin/env bash
#
# Download + verify the Gibson-0+ PointNav episode set (train) and the
# Matterport3D PointNav test set (eval), for Wijmans-faithful replication.
#
# What it does:
#   1. Downloads pointnav_gibson_0_plus_v1.zip (320 MB) into the habitat data
#      dir and unzips into data/datasets/pointnav/gibson/v1/train_extra_large/
#   2. Cross-checks the 411 Gibson-0+ scene names against the .glb files on
#      disk. (Expected: 100% present — these 411 scenes are a strict subset of
#      the 492-scene Gibson Full+ trainval release distributed as
#      gibson_habitat_trainval.zip. Verified off-cluster against Stanford's
#      gibson/data/data.json metadata: 411 ⊂ Full+ train (412) ⊂ Full+ trainval
#      (492). The only Full+ train scene without 0+ episodes is "Tolstoy".)
#   3. Downloads pointnav_mp3d_v1.zip (~380 MB) and extracts the test split,
#      which contains 18 MP3D scenes × 56 episodes = 1008 episodes — matching
#      Wijmans 2023 Appendix A.1 verbatim.
#   4. Cross-checks the 18 required MP3D test scenes against the .glb files
#      under scene_datasets/mp3d/. All 18 are in the standard 90-scan
#      Matterport3D Habitat task release, so they should already be present
#      from the existing download_mp.py --task_data habitat step.
#
# Re-running is safe: existing downloads and extractions are skipped.
#
# Usage on SCITAS:
#     bash scripts/cluster/download_gibson_0plus.sh
#
# Expected outputs:
#     $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/train_extra_large.json.gz
#     $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/content/<Scene>.json.gz  (411 files)
#     $HABITAT_DATA/datasets/pointnav/mp3d/v1/test/test.json.gz
#
set -euo pipefail

HABITAT_DATA="${HABITAT_DATA:-/scratch/izar/$USER/habitat_data}"
GIBSON_POINTNAV_DIR="$HABITAT_DATA/datasets/pointnav/gibson/v1"
GIBSON_SCENES_DIR="$HABITAT_DATA/scene_datasets/gibson"
MP3D_POINTNAV_DIR="$HABITAT_DATA/datasets/pointnav/mp3d/v1"
MP3D_SCENES_DIR="$HABITAT_DATA/scene_datasets/mp3d"

GIBSON_ZIP_URL="https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_0_plus_v1.zip"
GIBSON_ZIP_NAME="pointnav_gibson_0_plus_v1.zip"
GIBSON_SPLIT_DIR="$GIBSON_POINTNAV_DIR/train_extra_large"

MP3D_ZIP_URL="https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip"
MP3D_ZIP_NAME="pointnav_mp3d_v1.zip"

# Resolve this script's directory so we can find the authoritative scene lists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GIBSON_SCENE_LIST="$REPO_ROOT/habitat_configs/gibson_0plus_scenes.txt"
MP3D_SCENE_LIST="$REPO_ROOT/habitat_configs/mp3d_test_scenes.txt"

echo "=== Gibson-0+ + MP3D-test PointNav dataset install ==="
echo "HABITAT_DATA        = $HABITAT_DATA"
echo "Gibson scenes dir   = $GIBSON_SCENES_DIR"
echo "Gibson episodes dir = $GIBSON_POINTNAV_DIR"
echo "MP3D scenes dir     = $MP3D_SCENES_DIR"
echo "MP3D episodes dir   = $MP3D_POINTNAV_DIR"
echo

if [[ ! -d "$GIBSON_SCENES_DIR" ]]; then
    echo "ERROR: Gibson scene dir not found: $GIBSON_SCENES_DIR" >&2
    echo "       Point HABITAT_DATA at the dir that contains scene_datasets/gibson/" >&2
    exit 1
fi

if [[ ! -f "$GIBSON_SCENE_LIST" ]]; then
    echo "ERROR: expected Gibson scene list at $GIBSON_SCENE_LIST" >&2
    exit 1
fi

if [[ ! -f "$MP3D_SCENE_LIST" ]]; then
    echo "ERROR: expected MP3D scene list at $MP3D_SCENE_LIST" >&2
    exit 1
fi

mkdir -p "$GIBSON_POINTNAV_DIR" "$MP3D_POINTNAV_DIR"
cd "$GIBSON_POINTNAV_DIR"

# =============================================================================
# Part A — Gibson-0+ training episodes
# =============================================================================
# -----------------------------------------------------------------------------
# A.1 Download (idempotent)
# -----------------------------------------------------------------------------
if [[ -d "$GIBSON_SPLIT_DIR" && -f "$GIBSON_SPLIT_DIR/train_extra_large.json.gz" ]]; then
    existing=$(find "$GIBSON_SPLIT_DIR/content" -maxdepth 1 -name '*.json.gz' 2>/dev/null | wc -l)
    if [[ "$existing" -eq 411 ]]; then
        echo "[A.1] Gibson episodes already extracted (411 scene files found). Skipping."
    else
        echo "[A.1] Partial Gibson extract found ($existing / 411). Re-extracting."
        rm -rf "$GIBSON_SPLIT_DIR"
    fi
fi

if [[ ! -d "$GIBSON_SPLIT_DIR" ]]; then
    if [[ ! -f "$GIBSON_ZIP_NAME" ]]; then
        echo "[A.1] Downloading $GIBSON_ZIP_NAME (320 MB) ..."
        curl -L -o "$GIBSON_ZIP_NAME" "$GIBSON_ZIP_URL"
    else
        echo "[A.1] $GIBSON_ZIP_NAME already present, reusing."
    fi

    echo "[A.1] Unzipping Gibson-0+ episodes ..."
    unzip -q -o "$GIBSON_ZIP_NAME"
    rm "$GIBSON_ZIP_NAME"
fi

# -----------------------------------------------------------------------------
# A.2 Sanity-check Gibson episode files
# -----------------------------------------------------------------------------
n_gibson_episodes=$(find "$GIBSON_SPLIT_DIR/content" -maxdepth 1 -name '*.json.gz' | wc -l)
echo "[A.2] Gibson-0+ scene episode files: $n_gibson_episodes (expected 411)"

if [[ "$n_gibson_episodes" -ne 411 ]]; then
    echo "ERROR: expected 411 Gibson per-scene episode files, found $n_gibson_episodes" >&2
    exit 2
fi

if [[ ! -f "$GIBSON_SPLIT_DIR/train_extra_large.json.gz" ]]; then
    echo "ERROR: top-level train_extra_large.json.gz missing" >&2
    exit 3
fi

# -----------------------------------------------------------------------------
# A.3 Cross-check: are all 411 Gibson scene .glb files on disk?
# -----------------------------------------------------------------------------
echo "[A.3] Cross-checking 411 Gibson scene names against $GIBSON_SCENES_DIR/*.glb ..."

gibson_missing_count=0
gibson_missing_names=()
while IFS= read -r scene; do
    [[ -z "$scene" ]] && continue
    if [[ ! -f "$GIBSON_SCENES_DIR/$scene.glb" ]]; then
        gibson_missing_names+=("$scene")
        gibson_missing_count=$((gibson_missing_count + 1))
    fi
done < "$GIBSON_SCENE_LIST"

gibson_present_count=$(( 411 - gibson_missing_count ))
echo "      Present: $gibson_present_count / 411"
echo "      Missing: $gibson_missing_count"

if [[ "$gibson_missing_count" -gt 0 ]]; then
    echo
    echo "WARNING: $gibson_missing_count Gibson scenes have episode files but no .glb on disk."
    echo "         Training will crash on these scenes."
    echo "         Remediation: either (a) download gibson_habitat_trainval.zip and"
    echo "         extract the missing scenes, or (b) delete the corresponding episode"
    echo "         file(s) in $GIBSON_SPLIT_DIR/content/ so Habitat skips them."
    echo
    printf '         %s\n' "${gibson_missing_names[@]}" | head -40
    if [[ "$gibson_missing_count" -gt 40 ]]; then
        echo "         ... ($((gibson_missing_count - 40)) more)"
    fi
    echo
    MISSING_LOG="$GIBSON_POINTNAV_DIR/missing_glb_from_gibson_0plus.txt"
    printf '%s\n' "${gibson_missing_names[@]}" > "$MISSING_LOG"
    echo "      Full list written to: $MISSING_LOG"
fi

# =============================================================================
# Part B — Matterport3D test episodes (Wijmans 2023 Appendix A.1 eval set)
# =============================================================================
# -----------------------------------------------------------------------------
# B.1 Download (idempotent)
# -----------------------------------------------------------------------------
cd "$MP3D_POINTNAV_DIR"

if [[ -f "$MP3D_POINTNAV_DIR/test/test.json.gz" ]]; then
    echo "[B.1] MP3D test episodes already extracted. Skipping download."
else
    if [[ ! -f "$MP3D_ZIP_NAME" ]]; then
        echo "[B.1] Downloading $MP3D_ZIP_NAME (~380 MB) ..."
        curl -L -o "$MP3D_ZIP_NAME" "$MP3D_ZIP_URL"
    else
        echo "[B.1] $MP3D_ZIP_NAME already present, reusing."
    fi

    echo "[B.1] Extracting only the test split ..."
    unzip -q -o "$MP3D_ZIP_NAME" 'test/*'
    rm "$MP3D_ZIP_NAME"
fi

# -----------------------------------------------------------------------------
# B.2 Cross-check: are all 18 MP3D test .glb scene files on disk?
# -----------------------------------------------------------------------------
if [[ ! -d "$MP3D_SCENES_DIR" ]]; then
    echo "[B.2] WARNING: MP3D scene dir not found at $MP3D_SCENES_DIR"
    echo "      Skipping MP3D .glb cross-check. Run download_mp.py --task_data habitat"
    echo "      to install MP3D scene files before evaluating."
else
    echo "[B.2] Cross-checking 18 MP3D test scenes against $MP3D_SCENES_DIR/ ..."
    mp3d_missing_count=0
    mp3d_missing_names=()
    while IFS= read -r scene; do
        [[ -z "$scene" ]] && continue
        # Habitat MP3D layout: scene_datasets/mp3d/<hash>/<hash>.glb
        if [[ ! -f "$MP3D_SCENES_DIR/$scene/$scene.glb" ]]; then
            mp3d_missing_names+=("$scene")
            mp3d_missing_count=$((mp3d_missing_count + 1))
        fi
    done < "$MP3D_SCENE_LIST"
    mp3d_present_count=$(( 18 - mp3d_missing_count ))
    echo "      Present: $mp3d_present_count / 18"
    echo "      Missing: $mp3d_missing_count"

    if [[ "$mp3d_missing_count" -gt 0 ]]; then
        echo
        echo "WARNING: $mp3d_missing_count MP3D test scenes are missing their .glb file."
        echo "         Wijmans 2023 A.1 evaluation requires all 18."
        echo "         Remediation: re-run download_mp.py --task_data habitat, or"
        echo "         download just the missing scenes with:"
        echo "           python download_mp.py -o \$HABITAT_DATA --id <SCENE_HASH> --task_data habitat"
        echo
        printf '         %s\n' "${mp3d_missing_names[@]}"
    fi
fi

echo
echo "=== Done ==="
echo "Training: use the existing Gibson configs (split already set to train_extra_large)."
echo "Eval on MP3D test: add  +pointnav/mp3d_test_eval_override  to the hydra CLI."
