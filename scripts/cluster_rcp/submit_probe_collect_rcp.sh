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
# Optional 3rd arg: ckpt index for cross-checkpoint substitution-dynamics
# probing (used by substitution_dynamics figure / paper §H1 decay-rate claim
# at L260). Default ckpt.49 (final, converged). Common sweep values:
# 10, 20, 30, 40, 49 → 5 points across training.
CKPT_NUM="${3:-49}"

# Map condition → config name + ckpt dir + numbered probe ID.
# Job naming: probe-N (numbered). Mapping:
#   probe-1 = coarse,  probe-2 = foveated,  probe-3 = uniform,
#   probe-4 = foveated_logpolar,  probe-5 = blind,  probe-6 = fnorm (F2)
case "$COND" in
  coarse)
    CONFIG_NAME="pointnav/ddppo_pointnav_coarse_gibson"
    CKPT_DIR="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1"
    PROBE_ID=1
    ;;
  foveated)
    CONFIG_NAME="pointnav/ddppo_pointnav_foveated_gibson"
    CKPT_DIR="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2"
    PROBE_ID=2
    ;;
  uniform)
    CONFIG_NAME="pointnav/ddppo_pointnav_uniform_gibson"
    CKPT_DIR="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3"
    PROBE_ID=3
    ;;
  foveated_logpolar)
    CONFIG_NAME="pointnav/ddppo_pointnav_foveated_logpolar_gibson"
    CKPT_DIR="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4"
    PROBE_ID=4
    ;;
  blind)
    # Friend's blind seed=100 ckpt. Different hyperparams (num_envs=32, seed=100)
    # — see paper §limitations footnote.
    CONFIG_NAME="pointnav/ddppo_pointnav_blind_gibson"
    CKPT_DIR="/scratch/wxu/habitat_checkpoints_rcp/blind_seed_2_friend"
    PROBE_ID=5
    ;;
  fnorm)
    CONFIG_NAME="pointnav/ddppo_pointnav_foveated_normaliser_gibson"
    CKPT_DIR="/scratch/wxu/habitat_checkpoints_rcp/dh-fnorm"
    PROBE_ID=6
    ;;
  *)
    echo "Unknown condition: $COND" >&2
    exit 1
    ;;
esac

CKPT_PATH="${CKPT_DIR}/ckpt.${CKPT_NUM}.pth"

# Numbered job name: probe-N (default ckpt.49) or probe-N-cN (specific ckpt).
if [ "$CKPT_NUM" = "49" ]; then
  JOB_NAME="probe-${PROBE_ID}"
else
  JOB_NAME="probe-${PROBE_ID}-c${CKPT_NUM}"
fi
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
# Output dir + npz name. Default ckpt.49 → <cond>_det.npz (final result).
# Cross-ckpt sweep → <cond>_det_ckpt<N>.npz (preserves ckpt.49 NPZ).
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
if [ "$CKPT_NUM" = "49" ]; then
  OUT_NPZ="${OUT_DIR}/${COND}_det.npz"
else
  OUT_NPZ="${OUT_DIR}/${COND}_det_ckpt${CKPT_NUM}.npz"
fi

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
LOG_FILE="${OUT_NPZ%.npz}.log"
INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB_CONFIG/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB_CONFIG/pointnav/\$n; done; mkdir -p ${OUT_DIR}; nvidia-smi --query-gpu=name --format=csv,noheader; python -u /scratch/wxu/dh-spatial/scripts/probing/collect.py --config-name=${CONFIG_NAME} --ckpt=${CKPT_PATH} --episodes=${EPISODES} --deterministic=True --out=${OUT_NPZ} 2>&1 | tee -a ${LOG_FILE}; echo COLLECT_DONE; ls -la ${OUT_NPZ}"

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
