#!/bin/bash
# Shortcut consumption (paired same-scene-different-goal) on RCP — 1 condition per runai job.
# Updates §"Consumption axis" L407 (uniform locks-onto-old margin +1.83m,
# blind margin -0.38m — currently \toconfirm) + Figure shortcut_paired_traj.
#
# Usage:
#   bash scripts/cluster/submit_shortcut_rcp.sh <condition> [eps_per_scene] [max_scenes]
# ETA: ~1-2h on 1 A100 (default 10 eps × 20 scenes = 200 paired episodes).

set -e

COND="${1:?'condition required'}"
EPS_PER_SCENE="${2:-10}"
MAX_SCENES="${3:-20}"

case "$COND" in
  coarse)            CFG="pointnav/ddppo_pointnav_coarse_gibson";            CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth"; ID=1 ;;
  foveated)          CFG="pointnav/ddppo_pointnav_foveated_gibson";          CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2/ckpt.49.pth"; ID=2 ;;
  uniform)           CFG="pointnav/ddppo_pointnav_uniform_gibson";           CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3/ckpt.49.pth"; ID=3 ;;
  foveated_logpolar) CFG="pointnav/ddppo_pointnav_foveated_logpolar_gibson"; CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4/ckpt.49.pth"; ID=4 ;;
  blind)             CFG="pointnav/ddppo_pointnav_blind_gibson";             CKPT="/scratch/wxu/habitat_checkpoints_rcp/blind_seed_2_friend/ckpt.49.pth"; ID=5 ;;
  fnorm)             CFG="pointnav/ddppo_pointnav_foveated_normaliser_gibson"; CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-fnorm/ckpt.49.pth"; ID=6 ;;
  *) echo "Unknown condition: $COND" >&2; exit 1 ;;
esac

JOB_NAME="sc-${ID}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/shortcut_results"
OUT_JSON="${OUT_DIR}/${COND}_traj.json"
OUT_NPZ="${OUT_DIR}/${COND}_traj.npz"

echo "Shortcut ($COND): job=$JOB_NAME, ckpt=$CKPT, eps_per_scene=$EPS_PER_SCENE × max_scenes=$MAX_SCENES, out=$OUT_JSON"

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB_CONFIG/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB_CONFIG/pointnav/\$n; done; mkdir -p ${OUT_DIR}; nvidia-smi --query-gpu=name --format=csv,noheader; python -u /scratch/wxu/dh-spatial/scripts/eval/shortcut_with_trajectories.py --config-name=${CFG} --ckpt=${CKPT} --episodes-per-scene=${EPS_PER_SCENE} --max-scenes=${MAX_SCENES} --out-json=${OUT_JSON} --out-traj-npz=${OUT_NPZ} 2>&1 | tee -a ${OUT_DIR}/${COND}_traj.log; echo SC_DONE; ls -la ${OUT_JSON} ${OUT_NPZ}"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu --image="$IMAGE" --gpu=1 --cpu=8 --memory=32G \
    --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu --large-shm \
    --command -- bash -c "$INNER_CMD"
echo "Submitted."
