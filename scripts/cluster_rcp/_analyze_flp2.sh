#!/bin/bash
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat
pip install scikit-learn 2>&1 | tail -2
cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:$PYTHONPATH
for ck in 10 20 30 40 49; do
    NPZ=/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/flp2_det_ckpt${ck}.npz
    OUT=/scratch/wxu/habitat_checkpoints_rcp/analysis_results/flp2_det_ckpt${ck}_analysis.json
    if [ -f "$NPZ" ] && [ ! -f "$OUT" ]; then
        echo "== $NPZ"
        python -u /scratch/wxu/dh-spatial/scripts/probing/analyze.py --data "$NPZ" --out "$OUT" 2>&1 | tail -3
    elif [ ! -f "$NPZ" ]; then
        echo "== $NPZ: still running, skip"
    else
        echo "== $OUT: already exists, skip"
    fi
done
echo "DONE"
