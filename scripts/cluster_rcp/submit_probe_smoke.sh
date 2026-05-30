#!/bin/bash
# SMOKE TEST: probe-collect on 1 condition with minimal episodes.
# Validates that collect.py runs end-to-end on retrain ckpt before launching
# the full 5-condition analysis pipeline. Mirrors the submit_F2_normaliser.sh
# pattern (single-line INNER_CMD, ln -sf for symlinks).
#
# Usage:
#   bash scripts/cluster/submit_probe_smoke.sh
#
# Output:    /scratch/wxu/probing_data_rcp_smoke/coarse_smoke.npz
# ETA:       ~5-10 min on 1 GPU (5 episodes)
# Resource:  1 GPU, 8 CPU, 32G mem

set -e

JOB_NAME="dh-probe-smoke"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
CONFIG_NAME="pointnav/ddppo_pointnav_coarse_gibson"
CKPT_PATH="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth"
EPISODES=5
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/probing_data_smoke"
OUT_NPZ="${OUT_DIR}/coarse_smoke.npz"

echo "=================================================="
echo "  Probe-collect SMOKE TEST"
echo "  Job:      $JOB_NAME"
echo "  Config:   $CONFIG_NAME"
echo "  Ckpt:     $CKPT_PATH"
echo "  Episodes: $EPISODES"
echo "  Out:      $OUT_NPZ"
echo "=================================================="

# Single-line INNER_CMD (no embedded comments, no nested quotes). Mirrors F2
# launcher pattern. Verification of NPZ contents is done out-of-band via
# kubectl exec after pod completes (avoids quote-escaping hell in -c).
INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB_CONFIG/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB_CONFIG/pointnav/\$n; done; mkdir -p ${OUT_DIR}; nvidia-smi --query-gpu=name --format=csv,noheader; python -u /scratch/wxu/dh-spatial/scripts/probing/collect.py --config-name=${CONFIG_NAME} --ckpt=${CKPT_PATH} --episodes=${EPISODES} --deterministic=True --out=${OUT_NPZ} 2>&1 | tee -a ${OUT_DIR}/coarse_smoke.log; echo SMOKE_DONE; ls -la ${OUT_NPZ}"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu \
    --image="$IMAGE" \
    --gpu=1 \
    --cpu=8 \
    --memory=32G \
    --pvc=dhlab-scratch:/scratch \
    --pvc=home:/home/wxu \
    --large-shm \
    --command -- bash -c "$INNER_CMD"

echo ""
echo "Submitted. Monitor with:"
echo "  kubectl logs -n runai-dhlab-wxu \$(kubectl get pods -n runai-dhlab-wxu -l release=$JOB_NAME -o name | head -1 | cut -d/ -f2) -f"
