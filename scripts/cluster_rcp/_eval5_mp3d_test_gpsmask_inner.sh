#!/bin/bash
# Inner script for eval5-mp3d-gpsmask-<COND> jobs.
# Same Wijmans 2023 protocol as _eval5_mp3d_test_inner.sh, but masks the GPS +
# compass sensor channels at policy time (zero-injected). Tests §5 dissociation:
# if the policy reads from memory's integrated GPS code, success should collapse.
# Output goes to eval_5cond_mp3d_test_gpsmask/ so baseline (no mask) results are
# not overwritten.
set -e

COND="$1"
CFG="$2"
CKPT="$3"

OUT_DIR=/scratch/wxu/habitat_checkpoints_rcp/eval_5cond_mp3d_test_gpsmask
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

echo "=== ${COND}  cfg=${CFG}  ckpt=${CKPT}  protocol=MP3D_TEST  ablation=GPS+COMPASS_MASKED ==="
if [ ! -f "${CKPT}" ]; then
    echo "  MISSING ckpt; aborting"
    exit 1
fi

python -u /scratch/wxu/dh-spatial/scripts/eval/eval_paper_5cond.py \
    --config "${CFG}" \
    --ckpt "${CKPT}" \
    --data-path "data/datasets/pointnav/mp3d/v1/test/test.json.gz" \
    --split test \
    --no-sample \
    --episodes 1008 \
    --mask-gps --mask-compass \
    --out "${OUT_JSON}" 2>&1 | tee "${LOG}"

echo "=== DONE ${COND} (gpsmask) ==="
ls -la "${OUT_JSON}"
