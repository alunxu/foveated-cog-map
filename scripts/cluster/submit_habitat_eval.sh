#!/bin/bash
#SBATCH --job-name=habitat_eval
#SBATCH --time=00:45:00
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
#   sbatch submit_habitat_eval.sh <config_name> <ckpt_path> [num_episodes]
# Example:
#   sbatch scripts/cluster/submit_habitat_eval.sh \
#     pointnav/ddppo_pointnav_uniform_gibson \
#     /scratch/izar/wxu/habitat_checkpoints/uniform_gibson/ckpt.1.pth 5
#
# Writes MP4 videos to /scratch/izar/${USER}/eval_videos/<run_name>/
# Does NOT touch training checkpoints.

CONFIG_NAME=${1:?"Error: config name required"}
CKPT_PATH=${2:?"Error: ckpt path required"}
NUM_EPISODES=${3:-5}

echo "============================================"
echo "  Habitat Eval + Video Recording"
echo "  Config:   ${CONFIG_NAME}"
echo "  Ckpt:     ${CKPT_PATH}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Node:     $(hostname)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Date:     $(date)"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate habitat

# llvmlite (pulled by quaternion → numba) needs a newer libstdc++ than
# some cluster nodes provide in /lib64. Conda's own copy fixes this.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/${USER}/CS503_Project:${PYTHONPATH}"

DATA_DIR="/scratch/izar/${USER}/habitat_data"
CKPT_DIR="/scratch/izar/${USER}/habitat_checkpoints"
VIDEO_ROOT="/scratch/izar/${USER}/eval_videos"
RUN_NAME=$(basename "${CONFIG_NAME}" | sed 's/ddppo_pointnav_//')
VIDEO_DIR="${VIDEO_ROOT}/${RUN_NAME}"
mkdir -p "${VIDEO_DIR}"

cd /home/${USER}/habitat-lab

srun python -u /home/${USER}/CS503_Project/scripts/cluster/run_habitat.py \
    --config-name="${CONFIG_NAME}" \
    habitat_baselines.evaluate=True \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.num_environments=1 \
    habitat_baselines.test_episode_count=${NUM_EPISODES} \
    "habitat_baselines.eval_ckpt_path_dir=${CKPT_PATH}" \
    "habitat_baselines.video_dir=${VIDEO_DIR}" \
    "habitat_baselines.eval.video_option=[disk]" \
    habitat.dataset.split=train \
    habitat.dataset.scenes_dir="${DATA_DIR}/scene_datasets"

echo "Eval completed at $(date)"
echo "Videos under: ${VIDEO_DIR}"
ls -lh "${VIDEO_DIR}" 2>/dev/null || true
