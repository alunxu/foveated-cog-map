#!/bin/bash
#SBATCH --job-name=cs503_temporal
#SBATCH --time=00:30:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

cd "${PROJECT_DIR}"
python3 scripts/probing/temporal_probe.py \
    --in-dir /scratch/izar/wxu/probing_data \
    --suffix _det \
    --out /scratch/izar/wxu/probing_results/temporal_probe_det.json
