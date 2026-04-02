#!/bin/bash
#SBATCH --job-name=cs503_probe
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Usage: sbatch submit_probe.sh <checkpoint> <config> <output_dir> [n_episodes]
# Example: sbatch submit_probe.sh outputs/blind_agent/checkpoint_final.pt cfgs/blind.yaml probing_results/blind/ 200

CHECKPOINT=${1:?"Error: checkpoint path required"}
CONFIG=${2:?"Error: config file required"}
OUTPUT_DIR=${3:?"Error: output directory required"}
N_EPISODES=${4:-200}

echo "============================================"
echo "  CS503 Project — Probing Analysis"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Config:     ${CONFIG}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Episodes:   ${N_EPISODES}"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate cs503_project

python scripts/probe.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --n_episodes "${N_EPISODES}" \
    --output "${OUTPUT_DIR}"

echo "Probing complete at $(date)"
