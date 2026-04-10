#!/bin/bash
#SBATCH --job-name=habitat_shortcut
#SBATCH --time=04:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Usage:
#   sbatch submit_habitat_shortcut.sh <config_name> <ckpt_path> [episodes_per_scene] [max_scenes]
# Example:
#   sbatch scripts/cluster/submit_habitat_shortcut.sh \
#     pointnav/ddppo_pointnav_blind_gibson \
#     /scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.16.pth 10 20
#
# Shortcut discovery / cognitive-map behavioral evaluation:
#   Compares navigation with persistent LSTM memory (across episodes in the
#   same scene) vs. reset memory. If persistent > reset, accumulated spatial
#   representations functionally help navigation — the behavioral signature
#   of a cognitive map.

CONFIG_NAME=${1:?"Error: config name required (e.g. pointnav/ddppo_pointnav_blind_gibson)"}
CKPT_PATH=${2:?"Error: ckpt path required"}
EPISODES_PER_SCENE=${3:-10}
MAX_SCENES=${4:-20}

RUN_NAME=$(basename "${CONFIG_NAME}" | sed 's/ddppo_pointnav_//')
OUT_DIR="/scratch/izar/${USER}/shortcut_results"
OUT_PATH="${OUT_DIR}/${RUN_NAME}.json"

echo "============================================"
echo "  Shortcut Discovery / Cognitive-Map Eval"
echo "  Config:   ${CONFIG_NAME}"
echo "  Ckpt:     ${CKPT_PATH}"
echo "  Eps/scene: ${EPISODES_PER_SCENE}"
echo "  Max scenes: ${MAX_SCENES}"
echo "  Node:     $(hostname)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Date:     $(date)"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate habitat

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/${USER}/CS503_Project:${PYTHONPATH}"

cd /home/${USER}/habitat-lab

echo ""
echo "=== Running shortcut discovery eval ==="
python -u /home/${USER}/CS503_Project/scripts/habitat_shortcut_eval.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes-per-scene=${EPISODES_PER_SCENE} \
    --max-scenes=${MAX_SCENES} \
    --out="${OUT_PATH}"

echo ""
echo "Shortcut eval completed at $(date)"
echo "Results: ${OUT_PATH}"
