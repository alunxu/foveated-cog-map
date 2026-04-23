#!/bin/bash
#SBATCH --job-name=cs503_tr_seed
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

# Usage: sbatch submit_train_seeded.sh <config_name> <seed>
# Example:
#   sbatch submit_train_seeded.sh pointnav/ddppo_pointnav_foveated_learned_gibson 1
#
# Saves to ${CKPT_DIR}/<run_name>_seed<N> so seed 0 and this job don't collide.

CONFIG_NAME=${1:?"config name required"}
SEED=${2:?"seed required (integer, 1-999)"}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}")
RUN_NAME_SEEDED="${RUN_NAME}_seed${SEED}"
ABBREV=$(echo "${RUN_NAME_SEEDED}" | sed 's/blind/bld/;s/uniform/uni/;s/foveated/fov/;s/matched/mtc/;s/learned/lrn/;s/_seed/_s/')
scontrol update JobId=${SLURM_JOB_ID} JobName="cs503_tr_${ABBREV}" 2>/dev/null || true
mkdir -p "${CKPT_DIR}/${RUN_NAME_SEEDED}"

echo "============================================"
echo "  Habitat DD-PPO Training (seeded)"
echo "  Config:   ${CONFIG_NAME}"
echo "  Seed:     ${SEED}"
echo "  Run:      ${RUN_NAME_SEEDED}"
echo "  Node:     $(hostname)"
echo "  Date:     $(date)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "============================================"

cd /home/${USER}/habitat-lab

srun python -u ${PROJECT_DIR}/scripts/cluster/run_habitat.py \
    --config-name="${CONFIG_NAME}" \
    habitat_baselines.evaluate=False \
    habitat.seed=${SEED} \
    habitat.dataset.scenes_dir="${DATA_DIR}/scene_datasets" \
    "habitat_baselines.checkpoint_folder=${CKPT_DIR}/${RUN_NAME_SEEDED}" \
    "habitat_baselines.eval_ckpt_path_dir=${CKPT_DIR}/${RUN_NAME_SEEDED}" \
    "habitat_baselines.tensorboard_dir=${CKPT_DIR}/${RUN_NAME_SEEDED}/tb"

echo "Training completed at $(date)"
