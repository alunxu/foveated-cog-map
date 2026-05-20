#!/bin/bash
#SBATCH --job-name=cs503_h3
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# H3 gaze-memory coupling analysis.
#
# Usage:
#   sbatch scripts/cluster/submit_h3.sh [fixed_gaze_npz]
#
# Reads the learned-gaze probing data at
#   $PROBE_DIR/foveated_learned_gibson.npz
# and (optionally) the fixed-gaze data at the first positional argument,
# and writes the H3 report to $RESULTS_DIR/h3_analysis.json.

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

LEARNED_NPZ="${PROBE_DIR}/foveated_learned_gibson.npz"
FIXED_NPZ="${1:-${PROBE_DIR}/foveated_gibson.npz}"
OUT_JSON="${RESULTS_DIR}/h3_analysis.json"

echo "============================================"
echo "  H3 Gaze-Memory Coupling Analysis"
echo "  Learned npz: ${LEARNED_NPZ}"
echo "  Fixed npz:   ${FIXED_NPZ}"
echo "  Output:      ${OUT_JSON}"
echo "  Job:         ${SLURM_JOB_ID}"
echo "  Date:        $(date)"
echo "============================================"

mkdir -p "${RESULTS_DIR}"

# Always run intra-condition analysis on learned-gaze. Fall back to
# no-contrast if fixed-gaze npz isn't ready yet (foveated-resume is
# ~40h behind fov-learned).
if [ -f "${FIXED_NPZ}" ]; then
    python -u "${PROJECT_DIR}/scripts/probing/analyze_h3.py" \
        --learned-gaze-npz "${LEARNED_NPZ}" \
        --fixed-gaze-npz "${FIXED_NPZ}" \
        --out "${OUT_JSON}"
else
    echo "[note] fixed-gaze npz missing (${FIXED_NPZ}); running learned-only."
    python -u "${PROJECT_DIR}/scripts/probing/analyze_h3.py" \
        --learned-gaze-npz "${LEARNED_NPZ}" \
        --out "${OUT_JSON}"
fi

echo ""
echo "H3 analysis completed at $(date)"
echo "Report: ${OUT_JSON}"
