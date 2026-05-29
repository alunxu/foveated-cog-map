#!/bin/bash
#SBATCH --job-name=cs503_prb_seed
#SBATCH --time=03:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Usage: sbatch submit_probe_seeded.sh <config_name> <seed> [num_episodes]
#
# Probes a seed-N training run. Writes to <condition>_gibson_seed<N>.npz
# and <condition>_gibson_seed<N>_analysis.json so it does not clobber
# the seed-0 probe data.

CONFIG_NAME=${1:?"config name required"}
SEED=${2:?"seed required"}
NUM_EPISODES=${3:-500}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}")
RUN_NAME_SEEDED="${RUN_NAME}_seed${SEED}"
ABBREV=$(echo "${RUN_NAME_SEEDED}" | sed 's/blind/bld/;s/uniform/uni/;s/foveated/fov/;s/matched/mtc/;s/learned/lrn/;s/_seed/_s/')
scontrol update JobId=${SLURM_JOB_ID} JobName="cs503_prb_${ABBREV}" 2>/dev/null || true

CKPT_PATH="${CKPT_DIR}/${RUN_NAME_SEEDED}/latest.pth"
NPZ_PATH="${PROBE_DIR}/${RUN_NAME_SEEDED}.npz"
JSON_PATH="${RESULTS_DIR}/${RUN_NAME_SEEDED}_analysis.json"
JSON_LEGACY="${RESULTS_DIR}/${RUN_NAME_SEEDED}.json"

echo "============================================"
echo "  Probing Pipeline (seeded)"
echo "  Config:   ${CONFIG_NAME}"
echo "  Seed:     ${SEED}"
echo "  Ckpt:     ${CKPT_PATH}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Out NPZ:  ${NPZ_PATH}"
echo "  Out JSON: ${JSON_PATH}"
echo "============================================"

cd /home/${USER}/habitat-lab

python -u ${PROJECT_DIR}/scripts/probing/collect.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes=${NUM_EPISODES} \
    --collect-occupancy \
    --out="${NPZ_PATH}"

[ $? -ne 0 ] && echo "ERROR: collect failed" && exit 1

python -u ${PROJECT_DIR}/scripts/probing/analyze.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_PATH}" \
    --pca-dim=0 \
    --min-steps-scene=15

python -u ${PROJECT_DIR}/scripts/probing/analyze_legacy.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_LEGACY}" || true

echo "Done $(date)"
