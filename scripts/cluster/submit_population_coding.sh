#!/bin/bash
#SBATCH --job-name=cs503_pop_coding
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Population coding analysis (5 conditions × per-unit spatial info, rate
# maps, sparse decoding, intrinsic dim).  CPU-bound mostly, ~30-60 min.

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

cd "${PROJECT_DIR}"
python3 scripts/probing/population_coding_analysis.py \
    --in-dir /scratch/izar/wxu/probing_data \
    --suffix _det \
    --out-json /scratch/izar/wxu/probing_results/population_coding_det.json \
    --out-fig-dir docs/NeurIPS_2026/fig
