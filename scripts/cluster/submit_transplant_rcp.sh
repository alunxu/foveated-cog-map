#!/bin/bash
# Cross-condition memory transplant on RCP — 1 (donor, recipient) pair per job.
# Used to populate §"Format axis (H2)" 5×5 transplant matrix (Figure 4 right).
# Currently has 4×4 = 16 cells (skip self-transplant) when blind included = 5×5 - 5 = 20 pairs.
#
# Usage:
#   bash scripts/cluster/submit_transplant_rcp.sh <donor> <recipient> [episodes] [midpoint]
#   donor, recipient ∈ {coarse, foveated, uniform, foveated_logpolar, blind, fnorm}
#   episodes default 200, midpoint default 30 (matching paper's 5×5 matrix)
# ETA: ~30 min on 1 A100 per pair (200 episodes × 500 max steps).

set -e

DONOR="${1:?'donor condition required'}"
RECIP="${2:?'recipient condition required'}"
EPISODES="${3:-200}"
MIDPOINT="${4:-30}"

if [ "$DONOR" = "$RECIP" ]; then
  echo "donor == recipient ($DONOR); self-transplant skipped." >&2
  exit 1
fi

# Map cond -> (config, ckpt, id)
get_cfg() {
  case "$1" in
    coarse)            echo "pointnav/ddppo_pointnav_coarse_gibson" ;;
    foveated)          echo "pointnav/ddppo_pointnav_foveated_gibson" ;;
    uniform)           echo "pointnav/ddppo_pointnav_uniform_gibson" ;;
    foveated_logpolar) echo "pointnav/ddppo_pointnav_foveated_logpolar_gibson" ;;
    blind)             echo "pointnav/ddppo_pointnav_blind_gibson" ;;
    fnorm)             echo "pointnav/ddppo_pointnav_foveated_normaliser_gibson" ;;
  esac
}
get_ckpt() {
  case "$1" in
    coarse)            echo "/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth" ;;
    foveated)          echo "/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2/ckpt.49.pth" ;;
    uniform)           echo "/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3/ckpt.49.pth" ;;
    foveated_logpolar) echo "/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4/ckpt.49.pth" ;;
    blind)             echo "/scratch/wxu/habitat_checkpoints_rcp/blind_seed_2_friend/ckpt.49.pth" ;;
    fnorm)             echo "/scratch/wxu/habitat_checkpoints_rcp/dh-fnorm/ckpt.49.pth" ;;
  esac
}
get_id() {
  case "$1" in
    coarse) echo 1 ;; foveated) echo 2 ;; uniform) echo 3 ;;
    foveated_logpolar) echo 4 ;; blind) echo 5 ;; fnorm) echo 6 ;;
  esac
}

D_CFG=$(get_cfg "$DONOR")
D_CKPT=$(get_ckpt "$DONOR")
D_ID=$(get_id "$DONOR")
R_CFG=$(get_cfg "$RECIP")
R_CKPT=$(get_ckpt "$RECIP")
R_ID=$(get_id "$RECIP")

[ -z "$D_CFG" ] || [ -z "$R_CFG" ] && { echo "Unknown condition." >&2; exit 1; }

JOB_NAME="tp-${D_ID}-to-${R_ID}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/transplant_results"
OUT_JSON="${OUT_DIR}/${DONOR}_to_${RECIP}_mid${MIDPOINT}.json"

echo "Transplant ${DONOR}->${RECIP} (mid=${MIDPOINT}): job=$JOB_NAME, eps=$EPISODES, out=$OUT_JSON"

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB_CONFIG/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB_CONFIG/pointnav/\$n; done; mkdir -p ${OUT_DIR}; nvidia-smi --query-gpu=name --format=csv,noheader; python -u /scratch/wxu/dh-spatial/scripts/eval/transplant.py --donor-config=${D_CFG} --donor-ckpt=${D_CKPT} --recipient-config=${R_CFG} --recipient-ckpt=${R_CKPT} --episodes=${EPISODES} --midpoint-step=${MIDPOINT} --max-steps=500 --out=${OUT_JSON} 2>&1 | tee -a ${OUT_DIR}/${DONOR}_to_${RECIP}_mid${MIDPOINT}.log; echo TP_DONE; ls -la ${OUT_JSON}"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu --image="$IMAGE" --gpu=1 --cpu=8 --memory=32G \
    --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu --large-shm \
    --command -- bash -c "$INNER_CMD"
echo "Submitted."
