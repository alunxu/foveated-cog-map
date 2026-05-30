#!/bin/bash
# Install sklearn + run analyze.py on all blind / fnorm ckpt NPZ files locally on PVC.
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat

pip install scikit-learn 2>&1 | tail -3

cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:$PYTHONPATH

for cond in blind fnorm; do
    for ck in 10 20 30 40 49; do
        NPZ=/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/${cond}_det_ckpt${ck}.npz
        OUT=/scratch/wxu/habitat_checkpoints_rcp/analysis_results/${cond}_det_ckpt${ck}_analysis.json
        if [ -f "$NPZ" ] && [ ! -f "$OUT" ]; then
            echo "== analyzing $NPZ"
            python -u /scratch/wxu/dh-spatial/scripts/probing/analyze.py --data "$NPZ" --out "$OUT" 2>&1 | tail -3
            echo "== done $OUT"
        fi
    done
done
echo "DONE ALL"
