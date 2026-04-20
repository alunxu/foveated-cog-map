#!/bin/bash
#SBATCH --job-name=nan_debug
#SBATCH --time=24:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Resumes foveated training from ckpt.36.pth (last clean snapshot before the
# NaN event in job 2836021) with anomaly detection and forward-NaN hooks, to
# pin down the exact operation that first produced NaN. See
# scripts/cluster_debug/run_habitat_nan_debug.py.

set -e

echo "============================================"
echo "  NaN-debug run (foveated, from ckpt.36.pth)"
echo "  Date:   $(date)"
echo "  Job:    ${SLURM_JOB_ID}"
echo "============================================"

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

# Resume state lives in a dedicated dir so we don't touch the real
# foveated_gibson dir (which now holds the good re-run from scratch).
DEBUG_CKPT_DIR="${CKPT_DIR}/foveated_debug_nan"

# Directory for per-rank nan_diagnostic_rank{N}.txt files
export NAN_DEBUG_DIR="${PROJECT_DIR}/nan_debug_out/${SLURM_JOB_ID}"
mkdir -p "${NAN_DEBUG_DIR}"

cd /home/${USER}/habitat-lab

srun python -u "${PROJECT_DIR}/scripts/cluster_debug/run_habitat_nan_debug.py" \
    --config-name="pointnav/ddppo_pointnav_foveated_gibson" \
    habitat_baselines.evaluate=False \
    habitat.dataset.scenes_dir="${DATA_DIR}/scene_datasets" \
    "habitat_baselines.checkpoint_folder=${DEBUG_CKPT_DIR}" \
    "habitat_baselines.eval_ckpt_path_dir=${DEBUG_CKPT_DIR}" \
    "habitat_baselines.tensorboard_dir=${DEBUG_CKPT_DIR}/tb"

echo "Debug run ended at $(date); diagnostics in ${NAN_DEBUG_DIR}"
