#!/bin/bash
#SBATCH --job-name=cs503_bug
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Bug-baseline (Wijmans Table 1 row "Clairvoyant Bug" replication via
# simpler always-right wall-recovery). Provides classical-controller
# context for our 5-condition agent SPLs.
#
# Usage:
#   sbatch submit_bug_baseline.sh [config] [episodes]

CONFIG=${1:-"pointnav/ddppo_pointnav_blind_gibson"}
EPISODES=${2:-200}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

OUT_DIR="/scratch/izar/${USER}/bug_baseline_results"
OUT_PATH="${OUT_DIR}/bug.json"
mkdir -p "${OUT_DIR}"

echo "============================================"
echo "  Bug-baseline classical controller"
echo "  Config:    ${CONFIG}"
echo "  Episodes:  ${EPISODES}"
echo "  Output:    ${OUT_PATH}"
echo "  Date:      $(date)"
echo "============================================"

cd /home/${USER}/habitat-lab

python -u ${PROJECT_DIR}/scripts/eval/bug_baseline.py \
    --config="${CONFIG}" \
    --episodes=${EPISODES} \
    --out="${OUT_PATH}"
