#!/bin/bash
# Inner script for eval5-<COND> jobs. Takes 4 args: COND CONFIG CKPT N_EPS.
# Avoids runai-cli inline-quoting bug by reading args from argv directly.
set -e

COND="$1"
CFG="$2"
CKPT="$3"
N_EPS="$4"
SPLIT="${5:-train}"

OUT_DIR=/scratch/wxu/habitat_checkpoints_rcp/eval_5cond
OUT_JSON=${OUT_DIR}/${COND}.json
LOG=${OUT_DIR}/${COND}.log

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:$PYTHONPATH
export USER=wxu
export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data

HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config
mkdir -p "$HB_CONFIG/pointnav"
for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do
    ln -sf "$cfg" "$HB_CONFIG/pointnav/$(basename "$cfg")"
done

mkdir -p "$OUT_DIR"

echo "=== ${COND}  cfg=${CFG}  ckpt=${CKPT}  N_EPS=${N_EPS} ==="
if [ ! -f "${CKPT}" ]; then
    echo "  MISSING ckpt; aborting"
    exit 1
fi

python -u /scratch/wxu/dh-spatial/scripts/eval/eval_paper_5cond.py \
    --config "${CFG}" \
    --ckpt "${CKPT}" \
    --episodes "${N_EPS}" \
    --split "${SPLIT}" \
    --out "${OUT_JSON}" 2>&1 | tee "${LOG}"

echo "=== DONE ${COND} ==="
ls -la "${OUT_JSON}"
