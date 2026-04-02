#!/usr/bin/env bash
# Submit all 4 training conditions to SLURM
# Note: cs-503 QoS allows max 2 jobs in queue at a time.
# This script submits in two batches.
#
# Usage: bash submit_all.sh [batch]
#   batch 1 (default): blind + uniform
#   batch 2: foveated + matched_compute
set -euo pipefail

mkdir -p slurm_logs

BATCH=${1:-1}

if [ "${BATCH}" -eq 1 ]; then
    echo "Batch 1: blind + uniform"
    echo ""
    for cfg in cfgs/blind.yaml cfgs/uniform.yaml; do
        JOB_ID=$(sbatch --parsable submit_job.sh "${cfg}")
        echo "  ${cfg}  ->  Job ${JOB_ID}"
    done
    echo ""
    echo "After these complete, run: bash submit_all.sh 2"
elif [ "${BATCH}" -eq 2 ]; then
    echo "Batch 2: foveated + matched_compute"
    echo ""
    for cfg in cfgs/foveated.yaml cfgs/matched_compute.yaml; do
        JOB_ID=$(sbatch --parsable submit_job.sh "${cfg}")
        echo "  ${cfg}  ->  Job ${JOB_ID}"
    done
else
    echo "Usage: bash submit_all.sh [1|2]"
    exit 1
fi

echo ""
echo "Monitor with: squeue -u ${USER}"
echo "Outputs will be in: /scratch/${USER}/CS503_Project/outputs/"
