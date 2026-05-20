#!/bin/bash
#SBATCH --job-name=cs503_analysis
#SBATCH --time=03:00:00
#SBATCH --account=cs-503
#SBATCH --partition=mig24gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Probing pipeline on Kuma MIG (H100 virtual GPU).
# Uses mig24gb by default (safe for all conditions).
# For blind-only, override: sbatch --partition=mig12gb submit_probe_mig.sh ...
#
# Usage:
#   sbatch submit_probe_mig.sh <config_name> <ckpt_path> [num_episodes]
# Example:
#   sbatch scripts/cluster/submit_probe_mig.sh \
#     pointnav/ddppo_pointnav_blind_gibson \
#     /scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.19.pth 500

CONFIG_NAME=${1:?"Error: config name required (e.g. pointnav/ddppo_pointnav_blind_gibson)"}
CKPT_PATH=${2:?"Error: ckpt path required"}
NUM_EPISODES=${3:-500}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}")
ABBREV=$(echo "${RUN_NAME}" | sed 's/blind/bld/;s/uniform/uni/;s/foveated/fov/;s/matched/mtc/;s/learned/lrn/')
scontrol update JobId=${SLURM_JOB_ID} JobName="cs503_prb_${ABBREV}" 2>/dev/null || true
NPZ_PATH="${PROBE_DIR}/${RUN_NAME}.npz"
JSON_PATH="${RESULTS_DIR}/${RUN_NAME}_analysis.json"
JSON_LEGACY="${RESULTS_DIR}/${RUN_NAME}.json"

echo "============================================"
echo "  Habitat Probing Pipeline (MIG)"
echo "  Config:    ${CONFIG_NAME}"
echo "  Ckpt:      ${CKPT_PATH}"
echo "  Episodes:  ${NUM_EPISODES}"
echo "  Partition: ${SLURM_JOB_PARTITION}"
echo "  Node:      $(hostname)"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Date:      $(date)"
echo "============================================"

cd /home/${USER}/habitat-lab

# Step 1: Collect probing data (all LSTM layers)
echo ""
echo "=== Step 1: Collecting probing data (all layers) ==="
python -u ${PROJECT_DIR}/scripts/probing/collect.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes=${NUM_EPISODES} \
    --collect-occupancy \
    --out="${NPZ_PATH}"

if [ $? -ne 0 ]; then
    echo "ERROR: Data collection failed"
    exit 1
fi

# Step 2: Comprehensive analysis
echo ""
echo "=== Step 2: Comprehensive probing analysis ==="
python -u ${PROJECT_DIR}/scripts/probing/analyze.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_PATH}" \
    --pca-dim=0 \
    --min-steps-scene=15

if [ $? -ne 0 ]; then
    echo "WARNING: Comprehensive analysis failed, falling back to legacy"
fi

# Step 3: Legacy probes (backward compat)
echo ""
echo "=== Step 3: Legacy probe (backward compat) ==="
python -u ${PROJECT_DIR}/scripts/probing/analyze_legacy.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_LEGACY}"

echo ""
echo "Probing completed at $(date)"
echo "Data:     ${NPZ_PATH}"
echo "Analysis: ${JSON_PATH}"
echo "Legacy:   ${JSON_LEGACY}"
