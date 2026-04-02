#!/usr/bin/env bash
# Submit probing jobs for all 4 trained agents
# Run AFTER training jobs complete
# Usage: bash submit_all_probes.sh
set -euo pipefail

SCRATCH="/scratch/izar/${USER}/CS503_Project"
PROBE_DIR="${SCRATCH}/probing_results"

mkdir -p slurm_logs

echo "Submitting probing jobs for all 4 conditions..."
echo ""

CONDITIONS=(
    "blind_agent:cfgs/blind.yaml"
    "uniform_agent:cfgs/uniform.yaml"
    "foveated_agent:cfgs/foveated.yaml"
    "matched_compute_agent:cfgs/matched_compute.yaml"
)

for entry in "${CONDITIONS[@]}"; do
    RUN_NAME="${entry%%:*}"
    CONFIG="${entry##*:}"
    CKPT="${SCRATCH}/outputs/${RUN_NAME}/checkpoint_final.pt"

    if [ ! -f "${CKPT}" ]; then
        echo "  SKIP  ${RUN_NAME} — checkpoint not found: ${CKPT}"
        continue
    fi

    JOB_ID=$(sbatch --parsable scripts/cluster/submit_probe.sh "${CKPT}" "${CONFIG}" "${PROBE_DIR}/${RUN_NAME}/" 200)
    echo "  ${RUN_NAME}  ->  Job ${JOB_ID}"
done

echo ""
echo "Monitor with: squeue -u ${USER}"
echo "Results will be in: ${PROBE_DIR}/"
