#!/bin/bash
# Inner script: collect probe rollout npz for one (cond, ckpt) pair.
# Args: COND CONFIG CKPT OUT_STEM
set -e
COND="$1"
CFG="$2"
CKPT="$3"
OUT_STEM="$4"

OUT_DIR=/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp
OUT_NPZ=${OUT_DIR}/${OUT_STEM}.npz
LOG=${OUT_DIR}/${OUT_STEM}.log

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

echo "=== ${COND} ckpt=${CKPT} → ${OUT_NPZ} ==="
python -u /scratch/wxu/dh-spatial/scripts/probing/collect.py \
    --config-name "${CFG}" \
    --ckpt "${CKPT}" \
    --episodes 500 \
    --deterministic 1 \
    --out "${OUT_NPZ}" 2>&1 | tee "${LOG}"

echo "=== analyze: ${OUT_NPZ} → analysis.json ==="
python -u /scratch/wxu/dh-spatial/scripts/probing/analyze.py \
    --data "${OUT_NPZ}" \
    --out /scratch/wxu/habitat_checkpoints_rcp/analysis_results/${OUT_STEM}_analysis.json 2>&1 | tail -20

echo "=== DONE ${OUT_STEM} ==="
