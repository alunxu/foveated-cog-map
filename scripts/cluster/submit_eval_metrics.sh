#!/bin/bash
#SBATCH --job-name=cs503_evalm
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Metrics-only eval: SPL + Succ aggregate over N episodes, no video.
#
# Usage:
#   sbatch submit_eval_metrics.sh <config_name> <ckpt_path> [num_episodes=500]
#
# Reads:  <ckpt_path>
# Writes: stdout-captured aggregate metrics in the slurm log.
# Does NOT touch training checkpoints.

CONFIG_NAME=${1:?"config name required"}
CKPT_PATH=${2:?"ckpt path required"}
NUM_EPISODES=${3:-500}

echo "============================================"
echo "  Metrics-only eval"
echo "  Config:   ${CONFIG_NAME}"
echo "  Ckpt:     ${CKPT_PATH}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Node:     $(hostname)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Date:     $(date)"
echo "============================================"

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

cd /home/${USER}/habitat-lab

srun python -u ${PROJECT_DIR}/scripts/cluster/run_habitat.py \
    --config-name="${CONFIG_NAME}" \
    habitat_baselines.evaluate=True \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.num_environments=8 \
    habitat_baselines.test_episode_count=${NUM_EPISODES} \
    "habitat_baselines.eval_ckpt_path_dir=${CKPT_PATH}" \
    "habitat_baselines.eval.video_option=[]" \
    habitat.dataset.split=val \
    habitat.dataset.scenes_dir="${DATA_DIR}/scene_datasets"

echo "Metrics-only eval completed at $(date)"
