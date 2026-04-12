#!/bin/bash
#SBATCH --job-name=cs503_eval
#SBATCH --time=04:00:00
#SBATCH --account=cs-503
#SBATCH --partition=mig24gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Shortcut discovery eval on Kuma MIG (H100 virtual GPU).
#
# Usage:
#   sbatch submit_shortcut_mig.sh <config_name> <ckpt_path> [episodes_per_scene] [max_scenes]
# Example:
#   sbatch scripts/cluster/submit_shortcut_mig.sh \
#     pointnav/ddppo_pointnav_blind_gibson \
#     /scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.19.pth 10 20

CONFIG_NAME=${1:?"Error: config name required (e.g. pointnav/ddppo_pointnav_blind_gibson)"}
CKPT_PATH=${2:?"Error: ckpt path required"}
EPISODES_PER_SCENE=${3:-10}
MAX_SCENES=${4:-20}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}")
ABBREV=$(echo "${RUN_NAME}" | sed 's/blind/bld/;s/uniform/uni/;s/foveated/fov/;s/matched/mtc/;s/learned/lrn/')
scontrol update JobId=${SLURM_JOB_ID} JobName="cs503_evl_${ABBREV}" 2>/dev/null || true
OUT_DIR="/scratch/izar/${USER}/shortcut_results"
OUT_PATH="${OUT_DIR}/${RUN_NAME}.json"

echo "============================================"
echo "  Shortcut Discovery Eval (MIG)"
echo "  Config:    ${CONFIG_NAME}"
echo "  Ckpt:      ${CKPT_PATH}"
echo "  Eps/scene: ${EPISODES_PER_SCENE}"
echo "  Max scenes: ${MAX_SCENES}"
echo "  Partition: ${SLURM_JOB_PARTITION}"
echo "  Node:      $(hostname)"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Date:      $(date)"
echo "============================================"

cd /home/${USER}/habitat-lab

mkdir -p "${OUT_DIR}"

echo ""
echo "=== Running shortcut discovery eval ==="
python -u ${PROJECT_DIR}/scripts/eval/shortcut.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes-per-scene=${EPISODES_PER_SCENE} \
    --max-scenes=${MAX_SCENES} \
    --out="${OUT_PATH}"

echo ""
echo "Shortcut eval completed at $(date)"
echo "Results: ${OUT_PATH}"
