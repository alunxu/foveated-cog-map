#!/bin/bash
# Inner script for eval5-mp3d-<COND> jobs (Wijmans 2023 protocol).
# Takes 3 args: COND CONFIG CKPT
# Evaluates ALL 1008 episodes of MP3D PointNav test set (held-out 18 scenes).
set -e

COND="$1"
CFG="$2"
CKPT="$3"

OUT_DIR=/scratch/wxu/habitat_checkpoints_rcp/eval_5cond_mp3d_test
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

echo "=== ${COND}  cfg=${CFG}  ckpt=${CKPT}  protocol=MP3D_TEST (Wijmans 2023) ==="
if [ ! -f "${CKPT}" ]; then
    echo "  MISSING ckpt; aborting"
    exit 1
fi

# Wijmans 2023 protocol: 1008 episodes across 18 MP3D test scenes.
# data-path override switches from merged mp3d_gibson pool to pure MP3D dataset.
# --no-sample evaluates ALL 1008 episodes (deterministic full eval).
python -u /scratch/wxu/dh-spatial/scripts/eval/eval_paper_5cond.py \
    --config "${CFG}" \
    --ckpt "${CKPT}" \
    --data-path "data/datasets/pointnav/mp3d/v1/test/test.json.gz" \
    --split test \
    --no-sample \
    --episodes 1008 \
    --out "${OUT_JSON}" 2>&1 | tee "${LOG}"

echo "=== DONE ${COND} ==="
ls -la "${OUT_JSON}"
