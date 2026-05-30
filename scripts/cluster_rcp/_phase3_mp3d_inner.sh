#!/bin/bash
# Phase 3: MP3D-test probe collection inner script (PVC-resident, avoids quoting bug).
# Usage: bash _phase3_mp3d_inner.sh <cond> <config> <ckpt>
set -e
COND="$1"
CFG="$2"
CKPT="$3"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat
pip install --quiet "scikit-learn" "numpy<2" 2>&1 | tail -2
python -c "import numpy, sklearn; print(f'numpy={numpy.__version__}, sklearn={sklearn.__version__}')"

cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:$PYTHONPATH
export USER=wxu
export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data
HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config
mkdir -p "$HB_CONFIG/pointnav"
for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do
    ln -sf "$cfg" "$HB_CONFIG/pointnav/$(basename $cfg)"
done

NPZ_DIR=/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp
RESULT_DIR=/scratch/wxu/habitat_checkpoints_rcp/analysis_results
mkdir -p "$NPZ_DIR" "$RESULT_DIR"
NPZ="$NPZ_DIR/${COND}_mp3d_test_det.npz"
JSON="$RESULT_DIR/${COND}_mp3d_test_det_analysis.json"

echo "=== Collecting MP3D-test probes: $COND ==="
echo "  ckpt: $CKPT"
echo "  npz : $NPZ"

python -u /scratch/wxu/dh-spatial/scripts/probing/collect.py \
    --config-name="$CFG" \
    --ckpt="$CKPT" \
    --episodes=300 \
    --deterministic=true \
    --split=test \
    --override habitat.dataset.data_path=data/datasets/pointnav/mp3d/v1/test/test.json.gz \
    --out="$NPZ" 2>&1 | tee "$NPZ.log"

echo "=== COLLECT_DONE ==="
ls -la "$NPZ"

python -u /scratch/wxu/dh-spatial/scripts/probing/analyze.py \
    --data="$NPZ" \
    --out="$JSON" \
    --pca-dim=0 \
    --min-steps-scene=15 2>&1 | tail -20

echo "=== ANALYZE_DONE ==="
ls -la "$JSON"
