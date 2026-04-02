#!/bin/bash
#SBATCH --job-name=habitat_ddppo
#SBATCH --time=72:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Usage: sbatch submit_habitat.sh <config_name>
# Example: sbatch submit_habitat.sh pointnav/ddppo_pointnav_blind_gibson
#
# For 1-GPU test:
#   sbatch --gres=gpu:1 --ntasks-per-node=1 --time=01:00:00 \
#     submit_habitat.sh pointnav/ddppo_pointnav_blind

CONFIG_NAME=${1:?"Error: config name required (e.g., pointnav/ddppo_pointnav_blind_gibson)"}

echo "============================================"
echo "  Habitat DD-PPO Training"
echo "  Config:   ${CONFIG_NAME}"
echo "  Node:     $(hostname)"
echo "  GPUs:     ${SLURM_NTASKS_PER_NODE}"
echo "  Date:     $(date)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate habitat

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1

DATA_DIR="/scratch/izar/${USER}/habitat_data"
CKPT_DIR="/scratch/izar/${USER}/habitat_checkpoints"
RUN_NAME=$(basename "${CONFIG_NAME}" | sed 's/ddppo_pointnav_//')
mkdir -p "${CKPT_DIR}/${RUN_NAME}"

cd /home/${USER}/habitat-lab

srun python -u -m habitat_baselines.run \
    --config-name="${CONFIG_NAME}" \
    habitat_baselines.evaluate=False \
    habitat.dataset.scenes_dir="${DATA_DIR}/scene_datasets" \
    "habitat_baselines.checkpoint_folder=${CKPT_DIR}/${RUN_NAME}" \
    "habitat_baselines.eval_ckpt_path_dir=${CKPT_DIR}/${RUN_NAME}" \
    "habitat_baselines.tensorboard_dir=${CKPT_DIR}/${RUN_NAME}/tb"

echo "Training completed at $(date)"
