#!/bin/bash
#SBATCH --job-name=cs503_mlp_sanity
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# B3: Non-linear (MLP) probe sanity check.
# Re-runs GPS / compass / path-history lag-5 / goal-vector probes with a
# 2-layer MLP instead of Ridge, to verify the condition ordering is not a
# linear-probe artefact.

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

OUT_PATH="${RESULTS_DIR}/mlp_sanity.json"

echo "============================================"
echo "  MLP Probe Sanity Check"
echo "  Date:   $(date)"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Output: ${OUT_PATH}"
echo "============================================"

python -u ${PROJECT_DIR}/scripts/probing/mlp_probe_sanity.py \
    --in-dir ${PROBE_DIR} \
    --out ${OUT_PATH}
