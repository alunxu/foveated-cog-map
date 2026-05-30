#!/bin/bash
# Submit one downstream large-encoder control on RCP.
#
# Usage:
#   bash scripts/cluster_rcp/submit_resnet50_downstream.sh uniform [job-name]
#   bash scripts/cluster_rcp/submit_resnet50_downstream.sh foveated [job-name]
#
# Env overrides:
#   FRAMES=250000000 NPROC=4 SEED=0 bash ... uniform

set -euo pipefail

COND="${1:?'condition required: uniform or foveated'}"
JOB_NAME="${2:-dh-r50-${COND}}"

case "$COND" in
  uniform)
    CONFIG="pointnav/ddppo_pointnav_uniform_resnet50_gibson"
    ;;
  foveated)
    CONFIG="pointnav/ddppo_pointnav_foveated_resnet50_gibson"
    ;;
  *)
    echo "Unknown condition: $COND" >&2
    exit 1
    ;;
esac

IMAGE="${IMAGE:-registry.rcp.epfl.ch/dhlab-wxu/habitat:v2}"
NPROC="${NPROC:-4}"
FRAMES="${FRAMES:-250000000}"
SEED="${SEED:-0}"
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/${JOB_NAME}"

echo "=================================================="
echo "  ResNet50 downstream control"
echo "  Job name:  $JOB_NAME"
echo "  Config:    $CONFIG"
echo "  Condition: $COND"
echo "  GPU:       $NPROC"
echo "  Frames:    $FRAMES"
echo "  Seed:      $SEED"
echo "  Out:       $OUT_DIR"
echo "=================================================="

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB_CONFIG/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB_CONFIG/pointnav/\$n; done; mkdir -p ${OUT_DIR}; echo CONFIGS_AT_LAUNCH:; ls -la \$HB_CONFIG/pointnav/$(basename ${CONFIG}).yaml; grep -n \"backbone:\" \$HB_CONFIG/pointnav/$(basename ${CONFIG}).yaml; nvidia-smi --query-gpu=name --format=csv,noheader; torchrun --standalone --nproc_per_node=${NPROC} /scratch/wxu/dh-spatial/run_with_custom.py --config-name ${CONFIG} habitat.seed=${SEED} habitat_baselines.checkpoint_folder=${OUT_DIR} habitat_baselines.tensorboard_dir=${OUT_DIR}/tb habitat_baselines.total_num_steps=${FRAMES} habitat_baselines.num_environments=16 habitat_baselines.rl.ppo.num_steps=256 habitat_baselines.rl.ppo.use_linear_lr_decay=False hydra.run.dir=${OUT_DIR}/hydra_runs hydra.output_subdir=null 2>&1 | tee -a ${OUT_DIR}/run.log"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu \
    --image="$IMAGE" \
    --gpu="$NPROC" \
    --cpu=32 \
    --memory=160G \
    --pvc=dhlab-scratch:/scratch \
    --pvc=home:/home/wxu \
    --large-shm \
    --command -- bash -c "$INNER_CMD"

echo ""
echo "Submitted. Monitor with:"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job $JOB_NAME -p dhlab-wxu"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod logs $JOB_NAME --follow -p dhlab-wxu"
