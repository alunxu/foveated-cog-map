#!/bin/bash
#SBATCH --job-name=cs503_prb_mp3d
#SBATCH --time=03:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# B2: MP3D generalisation probe.
#
# Usage:
#   sbatch submit_probe_mp3d.sh <config_name> <ckpt_path> [num_episodes]
#
# Runs eval rollouts and probing on MP3D-val (held-out scenes from a dataset
# our agents did see during training). Produces <cond>_mp3d.npz and
# <cond>_mp3d_analysis.json for later cross-dataset comparison.

CONFIG_NAME=${1:?"config name required"}
CKPT_PATH=${2:?"ckpt path required"}
NUM_EPISODES=${3:-300}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}" | sed 's/_gibson/_mp3d/')
ABBREV=$(echo "${RUN_NAME}" | sed 's/blind/bld/;s/uniform/uni/;s/foveated/fov/;s/matched/mtc/;s/learned/lrn/')
scontrol update JobId=${SLURM_JOB_ID} JobName="cs503_mp3d_${ABBREV}" 2>/dev/null || true
NPZ_PATH="${PROBE_DIR}/${RUN_NAME}.npz"
JSON_PATH="${RESULTS_DIR}/${RUN_NAME}_analysis.json"

echo "============================================"
echo "  MP3D Generalisation Probe"
echo "  Config:   ${CONFIG_NAME}"
echo "  Ckpt:     ${CKPT_PATH}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Out NPZ:  ${NPZ_PATH}"
echo "  Out JSON: ${JSON_PATH}"
echo "============================================"

cd /home/${USER}/habitat-lab

# Step 1: Collect on MP3D val split.
# The config's default data_path uses `{split}` placeholders which Hydra's
# override grammar parses as special tokens. Pass the literal expanded path
# (already substituted for split=val) to avoid the parser error.
python -u ${PROJECT_DIR}/scripts/probing/collect.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes=${NUM_EPISODES} \
    --split=val \
    --override 'habitat.dataset.data_path=data/datasets/pointnav/mp3d/v1/val/val.json.gz' \
    --collect-occupancy \
    --out="${NPZ_PATH}"

if [ $? -ne 0 ]; then
    echo "ERROR: MP3D data collection failed"
    exit 1
fi

# Step 2: Probe analysis (reuse same analyzer; no MP3D-specific logic).
python -u ${PROJECT_DIR}/scripts/probing/analyze.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_PATH}" \
    --pca-dim=0 \
    --min-steps-scene=15

echo ""
echo "MP3D probing completed at $(date)"
