#!/bin/bash
# Phase B launcher: run all analysis lenses on Phase A NPZs.
#
# Prerequisites: Phase A complete (NPZs at /scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/<cond>_det.npz)
# At minimum: 4 retrain conditions (coarse/foveated/uniform/foveated_logpolar) + blind seed=100 NPZs.
# F2 (foveated_normaliser) is added separately when its ckpt + NPZ ready.
#
# What it runs (sequential, single 1-GPU pod):
#   1. compat symlinks: <cond>_det.npz <-> <legacy>_gibson_det.npz (for old scripts)
#   2. analyze.py per condition: linear Ridge probe + rate maps + per-step stats
#   3. run_mlp_probe_proper.py: 2-layer MLP probe (5-fold CV)
#   4. unaligned_cka.py: cross-condition CKA matrix
#   5. procrustes_shape_analyze.py: Procrustes shape distance matrix
#   6. lagk_all_targets.py: lag-k probe profile per condition
#
# Output: /scratch/wxu/habitat_checkpoints_rcp/analysis_results/
#   - <cond>_det_analysis.json  (linear probe + rate maps)
#   - mlp_probe.json            (cross-condition MLP)
#   - cka.json                  (CKA matrix)
#   - procrustes.json           (shape distance)
#   - lagk_<cond>.json          (lag-k per condition)
#
# ETA: ~75-90 min on 1 A100 GPU.
# Resource: 1 GPU, 16 CPU, 64G mem (analysis is mostly CPU-bound + numpy)
#
# Usage:
#   bash scripts/cluster/submit_phase_b_analysis.sh

set -e

JOB_NAME="probe-analysis-b"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
NPZ_DIR="/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
RESULTS_DIR="/scratch/wxu/habitat_checkpoints_rcp/analysis_results"

echo "=================================================="
echo "  Phase B: cross-condition analysis lenses"
echo "  Job:        $JOB_NAME"
echo "  NPZ dir:    $NPZ_DIR"
echo "  Results:    $RESULTS_DIR"
echo "=================================================="

# Build symlink commands inline. compat: legacy "matched" name aliases coarse,
# legacy "_gibson_det" suffix aliases "_det". foveated_logpolar is new (no
# legacy alias). Blind excluded for now — friend's seed=100 ckpt skipped while
# own dh-blind retrain in flight; rerun Phase B with blind once ckpt.49 ready.
SYMLINK_CMDS="cd ${NPZ_DIR} && ln -sf coarse_det.npz matched_gibson_det.npz && ln -sf uniform_det.npz uniform_gibson_det.npz && ln -sf foveated_det.npz foveated_gibson_det.npz && ln -sf foveated_logpolar_det.npz foveated_logpolar_gibson_det.npz && ln -sf coarse_det_scenes.txt matched_gibson_det_scenes.txt && ln -sf uniform_det_scenes.txt uniform_gibson_det_scenes.txt && ln -sf foveated_det_scenes.txt foveated_gibson_det_scenes.txt && ln -sf foveated_logpolar_det_scenes.txt foveated_logpolar_gibson_det_scenes.txt"

# sklearn is needed by analyze.py + lagk_all_targets.py but not in habitat
# conda env. Pip install --user goes to /home/wxu/.local (PVC mount) and
# persists across pods. unaligned_cka.py uses --data <npz1> <npz2> ... (nargs+),
# not --in-dir.
INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; export PATH=/home/wxu/.local/bin:\$PATH; pip install --quiet --user scikit-learn 2>&1 | tail -3; mkdir -p ${RESULTS_DIR}; ${SYMLINK_CMDS}; cd /scratch/wxu/dh-spatial; echo STEP1_LINEAR_PROBE; for cond in coarse foveated uniform foveated_logpolar; do npz=${NPZ_DIR}/\${cond}_det.npz; [ -f \$npz ] || { echo SKIP \$cond no npz; continue; }; echo running analyze.py on \$cond; python -u scripts/probing/analyze.py --data \$npz --out ${RESULTS_DIR}/\${cond}_det_analysis.json --pca-dim 0 --min-steps-scene 15 2>&1 | tail -10; done; echo STEP2_MLP_PROBE; python -u scripts/probing/run_mlp_probe_proper.py --in-dir ${NPZ_DIR} --out ${RESULTS_DIR}/mlp_probe.json 2>&1 | tail -10; echo STEP3_CKA; python -u scripts/probing/unaligned_cka.py --data ${NPZ_DIR}/coarse_det.npz ${NPZ_DIR}/foveated_det.npz ${NPZ_DIR}/uniform_det.npz ${NPZ_DIR}/foveated_logpolar_det.npz --out ${RESULTS_DIR}/cka.json 2>&1 | tail -10; echo STEP4_PROCRUSTES; python -u scripts/eval/procrustes_shape_analyze.py --in-dir ${NPZ_DIR} --out ${RESULTS_DIR}/procrustes.json --suffix _gibson_det 2>&1 | tail -10; echo STEP5_LAGK; for cond in coarse foveated uniform foveated_logpolar; do npz=${NPZ_DIR}/\${cond}_det.npz; [ -f \$npz ] || continue; python -u scripts/probing/lagk_all_targets.py --data \$npz --out ${RESULTS_DIR}/lagk_\${cond}.json 2>&1 | tail -5; done; echo PHASE_B_DONE; ls -la ${RESULTS_DIR}/"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu \
    --image="$IMAGE" \
    --gpu=1 \
    --cpu=16 \
    --memory=64G \
    --pvc=dhlab-scratch:/scratch \
    --pvc=home:/home/wxu \
    --large-shm \
    --command -- bash -c "$INNER_CMD"

echo ""
echo "Submitted. Monitor with:"
echo "  kubectl logs -n runai-dhlab-wxu \$(kubectl get pods -n runai-dhlab-wxu -l release=$JOB_NAME -o name | head -1 | cut -d/ -f2) -f"
