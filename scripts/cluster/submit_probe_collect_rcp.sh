#!/bin/bash
# Probe-collect on RCP for one condition (1 GPU runai job).
#
# Phase A of the post-retrain analysis pipeline. Generates the per-condition
# h2 NPZ used by every downstream lens (linear/MLP probes, CKA, Procrustes,
# lag-k, place-cell signature, etc.).
#
# Usage:
#   bash scripts/cluster/submit_probe_collect_rcp.sh <condition> [episodes]
#
#   condition  ∈ {coarse, foveated, uniform, foveated_logpolar, blind, fnorm}
#   episodes   default 500
#
# Output:
#   /scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/<condition>_det.npz
#   /scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/<condition>_det_scenes.txt
#
# ETA: ~1.75h on A100 (5min sim init + ~1.5h rollout for 500 episodes)
# Resource: 1 GPU, 8 CPU, 32G mem

set -e

COND="${1:?'condition required (coarse|foveated|uniform|foveated_logpolar|blind|fnorm)'}"
EPISODES="${2:-500}"

# Map condition → config name + ckpt path
case "$COND" in
  coarse)
    CONFIG_NAME="pointnav/ddppo_pointnav_coarse_gibson"
    CKPT_PATH="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth"
    ;;
  foveated)
    CONFIG_NAME="pointnav/ddppo_pointnav_foveated_gibson"
    CKPT_PATH="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2/ckpt.49.pth"
    ;;
  uniform)
    CONFIG_NAME="pointnav/ddppo_pointnav_uniform_gibson"
    CKPT_PATH="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3/ckpt.49.pth"
    ;;
  foveated_logpolar)
    CONFIG_NAME="pointnav/ddppo_pointnav_foveated_logpolar_gibson"
    CKPT_PATH="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4/ckpt.49.pth"
    ;;
  blind)
    # Friend's blind seed=100 ckpt. Different hyperparams (num_envs=32, seed=100)
    # — see paper §limitations footnote.
    CONFIG_NAME="pointnav/ddppo_pointnav_blind_gibson"
    CKPT_PATH="/scratch/wxu/habitat_checkpoints_rcp/blind_seed_2_friend/ckpt.49.pth"
    ;;
  fnorm)
    CONFIG_NAME="pointnav/ddppo_pointnav_foveated_normaliser_gibson"
    CKPT_PATH="/scratch/wxu/habitat_checkpoints_rcp/dh-fnorm/ckpt.49.pth"
    ;;
  *)
    echo "Unknown condition: $COND" >&2
    exit 1
    ;;
esac

# Job name pattern: probe-<cond>. Distinct from dh-probe-N (training pods,
# misleadingly named) — "probe-" prefix here actually does probing, not training.
JOB_NAME="probe-${COND//_/-}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
OUT_NPZ="${OUT_DIR}/${COND}_det.npz"

echo "=================================================="
echo "  Probe-collect (RCP)"
echo "  Condition:  $COND"
echo "  Job name:   $JOB_NAME"
echo "  Config:     $CONFIG_NAME"
echo "  Ckpt:       $CKPT_PATH"
echo "  Episodes:   $EPISODES"
echo "  Out NPZ:    $OUT_NPZ"
echo "=================================================="

# Single-line INNER_CMD: same pattern as F2 launcher (no embedded comments,
# no nested quotes). Includes USER + HABITAT_DATA_DIR env vars to fix
# habitat_env.py:99 path resolution under nobody/nogroup pod uid.
INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB_CONFIG/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB_CONFIG/pointnav/\$n; done; mkdir -p ${OUT_DIR}; nvidia-smi --query-gpu=name --format=csv,noheader; python -u /scratch/wxu/dh-spatial/scripts/probing/collect.py --config-name=${CONFIG_NAME} --ckpt=${CKPT_PATH} --episodes=${EPISODES} --deterministic=True --out=${OUT_NPZ} 2>&1 | tee -a ${OUT_DIR}/${COND}_det.log; echo COLLECT_DONE; ls -la ${OUT_NPZ}"

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
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job $JOB_NAME --project dhlab-wxu"
