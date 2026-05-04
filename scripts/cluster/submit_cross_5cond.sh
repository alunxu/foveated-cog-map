#!/bin/bash
# Post-blind comprehensive lens-set:
#   1. analyze.py on blind_izar_det.npz -> blind_izar_det_analysis.json
#      (Table 1 SPL/Success/GPS R^2/Compass R^2; Skaggs raw + per-scene; eigenspectrum)
#   2. unaligned_cka.py 5-cond -> cka_5cond.json
#   3. procrustes_shape_analyze.py 5-cond -> procrustes_5cond.json
#
# Pre-req: blind_izar_det.npz exists (probe-5-blind-izar completed).
# CPU-only. ETA ~30-45 min.
set -e

JOB_NAME="post-blind-lenses"
NPZ_DIR="/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
RESULTS_DIR="/scratch/wxu/habitat_checkpoints_rcp/analysis_results"

# Symlinks: procrustes_shape_analyze.py expects --suffix _gibson_det,
# meaning files like <stem>_gibson_det.npz where stem comes from COND_NPZ_MAP.
# Coarse uses legacy stem "matched" (per COND_NPZ_MAP), the rest use cond name.
SYMLINK_CMDS="cd ${NPZ_DIR} && ln -sf coarse_det.npz matched_gibson_det.npz && ln -sf foveated_det.npz foveated_gibson_det.npz && ln -sf uniform_det.npz uniform_gibson_det.npz && ln -sf foveated_logpolar_det.npz foveated_logpolar_gibson_det.npz && ln -sf blind_izar_det.npz blind_izar_gibson_det.npz"

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; export PATH=/home/wxu/.local/bin:\$PATH; pip install --quiet --user scikit-learn 2>&1 | tail -3; mkdir -p ${RESULTS_DIR}; test -f ${NPZ_DIR}/blind_izar_det.npz || { echo MISSING blind_izar_det.npz; exit 1; }; ${SYMLINK_CMDS}; cd /scratch/wxu/dh-spatial; echo STEP1_BLIND_ANALYZE; python -u scripts/probing/analyze.py --data ${NPZ_DIR}/blind_izar_det.npz --out ${RESULTS_DIR}/blind_izar_det_analysis.json --pca-dim 0 --min-steps-scene 15 2>&1 | tail -10; echo STEP2_CKA; python -u scripts/probing/unaligned_cka.py --data ${NPZ_DIR}/coarse_det.npz ${NPZ_DIR}/foveated_det.npz ${NPZ_DIR}/uniform_det.npz ${NPZ_DIR}/foveated_logpolar_det.npz ${NPZ_DIR}/blind_izar_det.npz --out ${RESULTS_DIR}/cka_5cond.json 2>&1 | tail -10; echo STEP3_PROCRUSTES; python -u scripts/eval/procrustes_shape_analyze.py --in-dir ${NPZ_DIR} --out ${RESULTS_DIR}/procrustes_5cond.json --suffix _gibson_det 2>&1 | tail -10; echo POST_BLIND_DONE; ls -la ${RESULTS_DIR}/blind_izar_det_analysis.json ${RESULTS_DIR}/cka_5cond.json ${RESULTS_DIR}/procrustes_5cond.json"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu --image="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2" \
    --cpu=8 --memory=48G --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu \
    --command -- bash -c "$INNER_CMD"
echo "Submitted $JOB_NAME (CPU-only, ETA ~30-45 min)"
