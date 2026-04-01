#!/bin/bash
#SBATCH --job-name=cs503_project_multi
#SBATCH --time=06:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/slurm_multi_%j.out
#SBATCH --error=outputs/slurm_multi_%j.err

# Usage: sbatch submit_job_multi_node.sh <config_file> <wandb_key>
# Example: sbatch submit_job_multi_node.sh cfgs/example.yaml YOUR_WANDB_KEY

CONFIG_FILE=${1:?  "Error: config file not provided"}
WANDB_KEY=${2:?    "Error: wandb key not provided"}

echo "============================================"
echo "  CS503 Project — Multi-Node SLURM Job"
echo "  Config:   ${CONFIG_FILE}"
echo "  Nodes:    ${SLURM_JOB_NUM_NODES}"
echo "  GPUs/node: 2"
echo "  Master:   $(hostname)"
echo "  Date:     $(date)"
echo "============================================"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate cs503_project

export WANDB_API_KEY="${WANDB_KEY}"
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

mkdir -p outputs

srun torchrun \
    --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node=2 \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/train.py --config "${CONFIG_FILE}"

echo "Job completed at $(date)"
