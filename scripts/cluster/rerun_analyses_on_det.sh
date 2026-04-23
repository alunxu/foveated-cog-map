#!/bin/bash
# After `resubmit_probes_deterministic.sh` has finished (check with squeue),
# run this to re-generate all downstream analyses using the new _det.npz
# probe files. Produces fresh JSON reports alongside the existing ones,
# suffixed with `_det_analysis.json` so we can diff old vs new.
#
# Usage: bash scripts/cluster/rerun_analyses_on_det.sh

set -e
source "$(dirname "$0")/common.sh"

DET_SUFFIX="_det"
ANALYSIS_SUFFIX="_det_analysis"

# Sanity: which det files exist?
echo "Available det probe files:"
ls -1 "${PROBE_DIR}"/*_det.npz 2>/dev/null || {
    echo "ERROR: no _det.npz files found in ${PROBE_DIR}."
    echo "Run resubmit_probes_deterministic.sh first."
    exit 1
}

echo ""

# Canonical Gibson 5 conditions
CONDITIONS=(blind uniform foveated foveated_learned matched)

# ---- Step 1: Per-condition baseline probes + lag curves ----
echo "=== Step 1: Per-condition baseline probes ==="
for cond in "${CONDITIONS[@]}"; do
    NPZ="${PROBE_DIR}/${cond}_gibson${DET_SUFFIX}.npz"
    JSON="${RESULTS_DIR}/${cond}_gibson${ANALYSIS_SUFFIX}.json"
    if [ ! -f "${NPZ}" ]; then
        echo "SKIP ${cond}: ${NPZ} missing"
        continue
    fi
    echo "→ ${cond}"
    python -u "${PROJECT_DIR}/scripts/probing/analyze.py" \
        --data="${NPZ}" \
        --out="${JSON}" \
        --pca-dim=0 \
        --min-steps-scene=15 || echo "[warn] ${cond} analyze.py failed"
done

# ---- Step 2: Extended lag probe (H2) ----
echo ""
echo "=== Step 2: Extended lag probes (H2 path-history) ==="
python -u "${PROJECT_DIR}/scripts/probing/extended_lag_probe.py" \
    --in-dir="${PROBE_DIR}" \
    --suffix="${DET_SUFFIX}" \
    --out="${RESULTS_DIR}/extended_lag${DET_SUFFIX}.json" \
    --max-lag=10 || true

# ---- Step 3: Cross-condition probe transfer matrix (H2) ----
echo ""
echo "=== Step 3: Cross-condition transfer matrix ==="
CROSS_ARGS="--data"
for cond in "${CONDITIONS[@]}"; do
    NPZ="${PROBE_DIR}/${cond}_gibson${DET_SUFFIX}.npz"
    [ -f "${NPZ}" ] || continue
    CROSS_ARGS="${CROSS_ARGS} ${cond}=${NPZ}"
done
python -u "${PROJECT_DIR}/scripts/probing/analyze_cross.py" \
    ${CROSS_ARGS} \
    --out="${RESULTS_DIR}/cross_transfer${DET_SUFFIX}.json" || true

# ---- Step 4: H3 analyses ----
echo ""
echo "=== Step 4: H3 gaze-memory analysis ==="
python -u "${PROJECT_DIR}/scripts/probing/analyze_h3.py" \
    --learned-gaze-npz "${PROBE_DIR}/foveated_learned_gibson${DET_SUFFIX}.npz" \
    --fixed-gaze-npz "${PROBE_DIR}/foveated_gibson${DET_SUFFIX}.npz" \
    --out "${RESULTS_DIR}/h3${DET_SUFFIX}_analysis.json" || true

# ---- Step 5: Goal-vector probe (H3 A1) ----
echo ""
echo "=== Step 5: Goal-vector probes ==="
python -u "${PROJECT_DIR}/scripts/probing/goal_vector_probe.py" \
    --in-dir="${PROBE_DIR}" \
    --suffix="${DET_SUFFIX}" \
    --out="${RESULTS_DIR}/goal_vector${DET_SUFFIX}.json" || true

# ---- Step 6: Unaligned CKA (H2) ----
echo ""
echo "=== Step 6: Unaligned CKA ==="
CKA_ARGS="--data"
for cond in "${CONDITIONS[@]}"; do
    NPZ="${PROBE_DIR}/${cond}_gibson${DET_SUFFIX}.npz"
    [ -f "${NPZ}" ] || continue
    CKA_ARGS="${CKA_ARGS} ${cond}=${NPZ}"
done
python -u "${PROJECT_DIR}/scripts/probing/unaligned_cka.py" \
    ${CKA_ARGS} \
    --out="${RESULTS_DIR}/cka${DET_SUFFIX}.json" || true

echo ""
echo "=== All re-analyses complete at $(date) ==="
echo "Inspect ${RESULTS_DIR}/*${DET_SUFFIX}*.json and diff against originals."
