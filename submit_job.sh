#!/bin/bash
#SBATCH --job-name=cs503_project
#SBATCH --time=04:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_%j.err

# Usage: sbatch submit_job.sh <config_file> <wandb_key> [num_gpus]
# Example: sbatch submit_job.sh cfgs/example.yaml YOUR_WANDB_KEY 1

CONFIG_FILE=${1:?  "Error: config file not provided"}
WANDB_KEY=${2:?    "Error: wandb key not provided"}
NUM_GPUS=${3:-1}

echo "============================================"
echo "  CS503 Project — SLURM Job"
echo "  Config:   ${CONFIG_FILE}"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Node:     $(hostname)"
echo "  Date:     $(date)"
echo "============================================"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate cs503_project

# Set environment variables
export WANDB_API_KEY="${WANDB_KEY}"
export OMP_NUM_THREADS=1

# Create output directory for this run
mkdir -p outputs

# Launch training
torchrun --nproc_per_node="${NUM_GPUS}" scripts/train.py --config "${CONFIG_FILE}"

echo "Job completed at $(date)"
