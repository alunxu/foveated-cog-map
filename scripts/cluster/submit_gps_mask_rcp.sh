#!/bin/bash
# Probe-collect with --mask-gps --mask-compass for §4.5 GPS-sensor ablation.
# Single condition per submission, mirrors submit_probe_collect_rcp.sh pattern.
# Output: <cond>_mask_gps.npz; 1 GPU, ~30-60 min.
set -e
COND="${1:?'condition required: blind_izar | coarse | foveated | foveated_logpolar | uniform'}"
EPISODES="${2:-300}"

case "$COND" in
  coarse)
    CFG="pointnav/ddppo_pointnav_coarse_gibson"
    CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth"; ID=1 ;;
  foveated)
    CFG="pointnav/ddppo_pointnav_foveated_gibson"
    CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2/ckpt.49.pth"; ID=2 ;;
  uniform)
    CFG="pointnav/ddppo_pointnav_uniform_gibson"
    CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3/ckpt.49.pth"; ID=3 ;;
  foveated_logpolar)
    CFG="pointnav/ddppo_pointnav_foveated_logpolar_gibson"
    CKPT="/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4/ckpt.49.pth"; ID=4 ;;
  blind_izar)
    CFG="pointnav/ddppo_pointnav_blind_gibson"
    CKPT="/scratch/wxu/habitat_checkpoints_rcp/blind_izar/ckpt.25.pth"; ID=5 ;;
  *) echo "Unknown $COND" >&2; exit 1 ;;
esac

JOB="gpsmask-${ID}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"
OUT_NPZ="${OUT_DIR}/${COND}_mask_gps.npz"
LOG="${OUT_NPZ%.npz}.log"

INNER="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB/pointnav/\$n; done; mkdir -p ${OUT_DIR}; nvidia-smi --query-gpu=name --format=csv,noheader; python -u /scratch/wxu/dh-spatial/scripts/probing/collect.py --config-name=${CFG} --ckpt=${CKPT} --episodes=${EPISODES} --deterministic=True --mask-gps --mask-compass --out=${OUT_NPZ} 2>&1 | tee -a ${LOG}; echo COLLECT_DONE; ls -la ${OUT_NPZ}"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB" \
    --project dhlab-wxu --image="$IMAGE" --gpu=1 --cpu=8 --memory=32G \
    --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu --large-shm \
    --command -- bash -c "$INNER" 2>&1 | tail -2
