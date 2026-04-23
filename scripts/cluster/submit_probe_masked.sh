#!/bin/bash
#SBATCH --job-name=cs503_prb_masked
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

# H3-revised causal test. Collects probing data with either compass or GPS
# sensor masked (zeroed) before the policy's forward pass. Lets us ask:
# does the hidden state still linearly encode the true quantity, or was it
# pass-through from the direct sensor input?
#
# Usage:
#   sbatch submit_probe_masked.sh <config_name> <ckpt_path> <mask> [num_episodes]
#   mask ∈ {compass, gps}

CONFIG_NAME=${1:?"config name required"}
CKPT_PATH=${2:?"ckpt path required"}
MASK=${3:?"mask target required: compass or gps"}
NUM_EPISODES=${4:-300}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}")
OUT_SUFFIX="_mask_${MASK}"
NPZ_PATH="${PROBE_DIR}/${RUN_NAME}${OUT_SUFFIX}.npz"
JSON_PATH="${RESULTS_DIR}/${RUN_NAME}${OUT_SUFFIX}_analysis.json"

echo "============================================"
echo "  Sensor-masked Probing"
echo "  Config:   ${CONFIG_NAME}"
echo "  Ckpt:     ${CKPT_PATH}"
echo "  Mask:     ${MASK}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Out NPZ:  ${NPZ_PATH}"
echo "============================================"

cd /home/${USER}/habitat-lab

MASK_FLAG=""
if [ "${MASK}" = "compass" ]; then
    MASK_FLAG="--mask-compass"
elif [ "${MASK}" = "gps" ]; then
    MASK_FLAG="--mask-gps"
else
    echo "ERROR: unknown mask target '${MASK}'; expected compass or gps"
    exit 1
fi

python -u ${PROJECT_DIR}/scripts/probing/collect.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes=${NUM_EPISODES} \
    ${MASK_FLAG} \
    --collect-occupancy \
    --out="${NPZ_PATH}"

python -u ${PROJECT_DIR}/scripts/probing/analyze.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_PATH}" \
    --pca-dim=0 \
    --min-steps-scene=15

echo "Done at $(date)"
