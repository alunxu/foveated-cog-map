#!/bin/bash
#SBATCH --job-name=cs503_smoke
#SBATCH --time=00:15:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Smoke-test: instantiate each new foveated policy + one forward pass.
# Run on a compute node (login node has llvmlite/glibc issues).

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

cd "${PROJECT_DIR}"
python3 scripts/cluster/smoke_policy.py
