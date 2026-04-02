#!/bin/bash
#SBATCH --job-name=cs503_nav
#SBATCH --time=06:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Usage: sbatch submit_job.sh <config_file>
# Example: sbatch submit_job.sh cfgs/foveated.yaml
#
# Optional env vars:
#   WANDB_API_KEY  — set before sbatch if you want W&B logging

CONFIG_FILE=${1:?"Error: config file not provided. Usage: sbatch submit_job.sh cfgs/<condition>.yaml"}

echo "============================================"
echo "  CS503 Project — Navigation Agent Training"
echo "  Config:   ${CONFIG_FILE}"
echo "  Node:     $(hostname)"
echo "  Date:     $(date)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "============================================"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate cs503_project

# Use scratch for outputs (large checkpoints)
SCRATCH_DIR="/scratch/izar/${USER}/CS503_Project"
mkdir -p "${SCRATCH_DIR}/outputs"

# Create slurm log directory if needed
mkdir -p slurm_logs

# Training
python scripts/train.py \
    --config "${CONFIG_FILE}" \
    --overrides "output_dir=${SCRATCH_DIR}/outputs"

echo "Job completed at $(date)"
