#!/bin/bash
# Phase 1b v5: use CLEAN rewritten scripts (JSON-first save, try/except savefig).
set -e
cd /scratch/wxu/dh-spatial
source /scratch/wxu/miniconda3/etc/profile.d/conda.sh
conda activate habitat
pip install scikit-learn 'numpy<2' 2>&1 | tail -2
echo "numpy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "sklearn: $(python -c 'import sklearn; print(sklearn.__version__)')"

echo ""; echo "=== running LOSO ==="
python scripts/probing/loop_5cond_loso_clean.py

echo ""; echo "=== running subspace_v2 ==="
python scripts/probing/loop_5cond_subspace_v2_clean.py

echo ""; echo "DONE. Outputs:"
ls -la /scratch/wxu/habitat_checkpoints_rcp/analysis_results/loso_5cond.json /scratch/wxu/habitat_checkpoints_rcp/analysis_results/subspace_divergence_5cond.json
