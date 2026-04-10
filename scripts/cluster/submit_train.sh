#!/bin/bash
#SBATCH --job-name=habitat_ddppo
#SBATCH --time=72:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Usage: sbatch submit_train.sh <config_name>
# Example: sbatch submit_train.sh pointnav/ddppo_pointnav_blind_gibson
#
# For 1-GPU test:
#   sbatch --gres=gpu:1 --ntasks-per-node=1 --time=01:00:00 \
#     submit_train.sh pointnav/ddppo_pointnav_blind

CONFIG_NAME=${1:?"Error: config name required (e.g., pointnav/ddppo_pointnav_blind_gibson)"}

echo "============================================"
echo "  Habitat DD-PPO Training"
echo "  Config:   ${CONFIG_NAME}"
echo "  Node:     $(hostname)"
echo "  Nodes:    ${SLURM_NNODES}"
echo "  GPUs/node: ${SLURM_NTASKS_PER_NODE}"
echo "  Total GPUs: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "  Date:     $(date)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "============================================"

source "$(dirname "$0")/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}")
mkdir -p "${CKPT_DIR}/${RUN_NAME}"

cd /home/${USER}/habitat-lab

# Custom entry point that imports src.habitat first to register custom
# sensors (GoalInStartFrameSensor, CloseToGoalSensor) and policies
# (WijmansPointNavPolicy, FoveatedWijmansPolicy) before invoking
# habitat_baselines.run.main().
srun python -u ${PROJECT_DIR}/scripts/cluster/run_habitat.py \
    --config-name="${CONFIG_NAME}" \
    habitat_baselines.evaluate=False \
    habitat.dataset.scenes_dir="${DATA_DIR}/scene_datasets" \
    "habitat_baselines.checkpoint_folder=${CKPT_DIR}/${RUN_NAME}" \
    "habitat_baselines.eval_ckpt_path_dir=${CKPT_DIR}/${RUN_NAME}" \
    "habitat_baselines.tensorboard_dir=${CKPT_DIR}/${RUN_NAME}/tb"

echo "Training completed at $(date)"
