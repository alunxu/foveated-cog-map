#!/bin/bash
# Resubmit ALL probe data collection for every condition × variant with
# --deterministic=True, to replace data collected under the earlier
# stochastic-sampling bug (collect.py previously hardcoded
# deterministic=False, causing fov-fix/uniform probes to collapse to
# ~4-step episodes at 0% success while their deterministic eval SPL was
# ~0.83). See docs/NeurIPS_2026/FINDING_DETERMINISTIC_BUG.md for details.
#
# The new files are written to ${PROBE_DIR}/<run_name>_det[_variant].npz
# so we do not overwrite the stochastic collection (kept for diffing).
#
# Usage: bash scripts/cluster/resubmit_probes_deterministic.sh [gibson|all]
#   gibson (default) — just the 5 canonical Gibson conditions
#   all              — also mp3d + sensor-masked variants (many more jobs)

set -e

MODE="${1:-gibson}"
PROJECT_DIR="/home/${USER}/cs503-project"
cd "${PROJECT_DIR}"

CKPT_ROOT="/scratch/izar/${USER}/habitat_checkpoints"

# Helper: submit one deterministic probe job for a (config, ckpt) pair.
submit_det_probe() {
    local cfg="$1"
    local ckpt="$2"
    local eps="${3:-500}"
    if [ ! -f "${ckpt}" ]; then
        echo "SKIP ${cfg}: ckpt missing (${ckpt})"
        return
    fi
    echo "→ ${cfg}  (ckpt: ${ckpt})"
    sbatch scripts/cluster/submit_probe_deterministic.sh "${cfg}" "${ckpt}" "${eps}"
}

echo "============================================"
echo "  Resubmitting probe collections (det=True)"
echo "  Mode: ${MODE}"
echo "============================================"

# --- Gibson 5-condition canonical ---
# Checkpoints match paper: ckpt.49 for fully-converged conditions; fov-fix
# uses the pre-corruption ckpt.36 from foveated_gibson_corrupt_job2836021
# (see §Appendix training-stability — last clean intermediate at ~174M).
echo ""
echo "# Gibson canonical 5 conditions"
submit_det_probe \
    "pointnav/ddppo_pointnav_blind_gibson" \
    "${CKPT_ROOT}/blind_gibson/ckpt.49.pth"
submit_det_probe \
    "pointnav/ddppo_pointnav_uniform_gibson" \
    "${CKPT_ROOT}/uniform_gibson/ckpt.49.pth"
submit_det_probe \
    "pointnav/ddppo_pointnav_foveated_gibson" \
    "${CKPT_ROOT}/foveated_gibson_corrupt_job2836021/ckpt.36.pth"
submit_det_probe \
    "pointnav/ddppo_pointnav_foveated_learned_gibson" \
    "${CKPT_ROOT}/foveated_learned_gibson/ckpt.49.pth"
submit_det_probe \
    "pointnav/ddppo_pointnav_matched_gibson" \
    "${CKPT_ROOT}/matched_gibson/ckpt.49.pth"

if [ "${MODE}" = "all" ]; then
    # --- MP3D (held-out scene dataset) ---
    echo ""
    echo "# MP3D held-out scenes"
    for name in blind uniform foveated foveated_learned matched; do
        cfg="pointnav/ddppo_pointnav_${name}_gibson"
        if [ "${name}" = "foveated" ]; then
            ckpt="${CKPT_ROOT}/${name}_gibson/ckpt.36.pth"
        else
            ckpt="${CKPT_ROOT}/${name}_gibson/ckpt.9.pth"
        fi
        if [ ! -f "${ckpt}" ]; then
            echo "SKIP ${name} MP3D: ckpt missing"
            continue
        fi
        echo "→ ${cfg} (MP3D)"
        # The MP3D-specific script handles dataset overrides.
        sbatch scripts/cluster/submit_probe_mp3d.sh "${cfg}" "${ckpt}" 300
    done

    # --- Sensor-masked (compass + GPS) ---
    echo ""
    echo "# Sensor-masked (compass + GPS) — fov-learned, fov-fix, uniform only"
    for name in foveated foveated_learned uniform; do
        cfg="pointnav/ddppo_pointnav_${name}_gibson"
        if [ "${name}" = "foveated" ]; then
            ckpt="${CKPT_ROOT}/${name}_gibson/ckpt.36.pth"
        else
            ckpt="${CKPT_ROOT}/${name}_gibson/ckpt.9.pth"
        fi
        [ -f "${ckpt}" ] || continue
        for mask in compass gps; do
            echo "→ ${cfg} (mask=${mask})"
            sbatch scripts/cluster/submit_probe_masked.sh "${cfg}" "${ckpt}" "${mask}" 300
        done
    done
fi

echo ""
echo "All jobs submitted. Track with: squeue -u ${USER}"
echo "Output NPZ pattern: /scratch/izar/${USER}/probing_data/*_det*.npz"
