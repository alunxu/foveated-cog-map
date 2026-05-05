#!/bin/bash
# Excursion-forgetting (WJ-F) on RCP — 1 condition per runai job.
# Updates §"Boundaries: null results" L428 numbers (blind +0.17, uniform +0.31,
# foveated +0.34, coarse +0.43 — currently \toconfirm).
#
# Usage:
#   bash scripts/cluster/submit_excursion_rcp.sh <condition> [episodes]
#   condition ∈ {coarse, foveated, uniform, foveated_logpolar, blind, fnorm}
# ETA: ~1-2h on 1 A100 (100 episodes, mid-rollout perturbation).

set -e

COND="${1:?'condition required'}"
EPISODES="${2:-100}"

case "$COND" in
  coarse)            CFG="pointnav/ddppo_pointnav_coarse_gibson";            CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth"; ID=1 ;;
  foveated)          CFG="pointnav/ddppo_pointnav_foveated_gibson";          CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2/ckpt.49.pth"; ID=2 ;;
  uniform)           CFG="pointnav/ddppo_pointnav_uniform_gibson";           CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3/ckpt.49.pth"; ID=3 ;;
  foveated_logpolar) CFG="pointnav/ddppo_pointnav_foveated_logpolar_gibson"; CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4/ckpt.49.pth"; ID=4 ;;
  blind)             CFG="pointnav/ddppo_pointnav_blind_gibson";             CKPT="/scratch/wxu/habitat_checkpoints_rcp/blind_izar/ckpt.25.pth"; ID=5 ;;
  fnorm)             CFG="pointnav/ddppo_pointnav_foveated_normaliser_gibson"; CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-fnorm/ckpt.49.pth"; ID=6 ;;
  *) echo "Unknown condition: $COND" >&2; exit 1 ;;
esac

JOB_NAME="exc-${ID}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/excursion_results"
OUT_NPZ="${OUT_DIR}/${COND}_excursion.npz"

echo "Excursion ($COND): job=$JOB_NAME, ckpt=$CKPT, episodes=$EPISODES, out=$OUT_NPZ"

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB_CONFIG/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB_CONFIG/pointnav/\$n; done; mkdir -p ${OUT_DIR}; nvidia-smi --query-gpu=name --format=csv,noheader; python -u /scratch/wxu/dh-spatial/scripts/eval/excursion_forgetting.py --config=${CFG} --ckpt=${CKPT} --episodes=${EPISODES} --warmup-steps=50 --detour-steps=25 --recovery-steps=100 --seed=42 --out=${OUT_NPZ} 2>&1 | tee -a ${OUT_DIR}/${COND}_excursion.log; echo EXC_DONE; ls -la ${OUT_NPZ}"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu --image="$IMAGE" --gpu=1 --cpu=8 --memory=32G \
    --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu --large-shm \
    --command -- bash -c "$INNER_CMD"
echo "Submitted."
