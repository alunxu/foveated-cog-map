#!/bin/bash
#SBATCH --job-name=habitat_cross
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Usage:
#   sbatch submit_habitat_cross.sh
#
# Runs cross-condition CKA and probe-transfer analysis on all available
# probing data. No GPU required — CPU-only computation.
#
# Expects probing data at:
#   /scratch/izar/$USER/probing_data/{blind,uniform,foveated,matched}_gibson.npz

echo "============================================"
echo "  Cross-Condition Analysis (CKA + Transfer)"
echo "  Node:  $(hostname)"
echo "  Job:   ${SLURM_JOB_ID}"
echo "  Date:  $(date)"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate habitat

export PYTHONPATH="/home/${USER}/CS503_Project:${PYTHONPATH}"

PROBE_DIR="/scratch/izar/${USER}/probing_data"
RESULTS_DIR="/scratch/izar/${USER}/probing_results"
OUT_PATH="${RESULTS_DIR}/cross_analysis.json"

# Build data arguments from available .npz files
DATA_ARGS=""
for cond in blind_gibson uniform_gibson foveated_gibson matched_gibson; do
    npz="${PROBE_DIR}/${cond}.npz"
    if [ -f "${npz}" ]; then
        name=$(echo "${cond}" | sed 's/_gibson//')
        DATA_ARGS="${DATA_ARGS} ${name}=${npz}"
        echo "  Found: ${cond}"
    else
        echo "  Missing: ${cond} (skipping)"
    fi
done

if [ -z "${DATA_ARGS}" ]; then
    echo "ERROR: No probing data found"
    exit 1
fi

echo ""
echo "=== Running cross-condition analysis ==="
python -u /home/${USER}/CS503_Project/scripts/habitat_probe_cross.py \
    --data ${DATA_ARGS} \
    --out "${OUT_PATH}"

echo ""
echo "Cross-analysis completed at $(date)"
echo "Results: ${OUT_PATH}"
