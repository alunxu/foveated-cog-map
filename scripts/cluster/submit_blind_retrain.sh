#!/bin/bash
# Blind retrain on RCP RunAI (4 GPU torchrun).
#
# Why retrain blind:
#   The previous blind ckpt used by paper §results was on izar
#   (/scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.{34,49}.pth).
#   izar is currently down; the ckpt is inaccessible. Friend's seed=100
#   blind ckpt is the only reachable backup, but its hyperparams differ
#   (seed=100, num_environments=32) from our 4-condition retrain set.
#
#   This launcher trains a new blind with hyperparams *identical* to the
#   4 sighted retrain conditions (commit 9674b8c):
#     - seed: 0
#     - total_num_steps: 2.5e8 (250M frames)
#     - num_environments: 16  (8 was the prior blind YAML default; fixed)
#     - ppo.num_steps: 256
#     - use_linear_lr_decay: False
#     - lr: 2.5e-4
#     - hidden_size: 512, num_recurrent_layers: 3
#     - max_episode_steps: 2000
#
#   Architectural difference (intended): force_blind_policy=True; no rgb/depth
#   sim_sensors. All other model + sensor stack identical.
#
# Config: habitat_configs/ddppo_pointnav_blind_gibson.yaml
# Output: /scratch/wxu/habitat_checkpoints_rcp/dh-blind/{ckpt.0..49.pth, latest.pth, run.log}
# ETA:    ~14-17h on 4xH100 (blind has no visual encoder, faster than sighted).
#
# Usage:
#   bash scripts/cluster/submit_blind_retrain.sh [job-name]
# Default job name: dh-blind

set -e

JOB_NAME="${1:-dh-blind}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
NPROC=4
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/${JOB_NAME}"

echo "=================================================="
echo "  Blind retrain (RCP)"
echo "  Job:     $JOB_NAME"
echo "  Config:  pointnav/ddppo_pointnav_blind_gibson"
echo "  Policy:  WijmansPointNavPolicy (force_blind_policy=True)"
echo "  GPU:     ${NPROC} (torchrun --nproc_per_node=$NPROC)"
echo "  Frames:  250M (unified retrain budget, seed=0)"
echo "  Out:     $OUT_DIR"
echo "=================================================="

# Single-line INNER_CMD (no embedded comments, no nested quotes). Mirrors
# F2 launcher fix (commit 71fc60a). Includes USER + HABITAT_DATA_DIR for
# nobody pod uid path resolution. ln -sf for image-baked yaml override.
INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; export USER=wxu; export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data; HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config; mkdir -p \$HB_CONFIG/pointnav; for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do n=\$(basename \$cfg); ln -sf \$cfg \$HB_CONFIG/pointnav/\$n; done; mkdir -p ${OUT_DIR}; nvidia-smi --query-gpu=name --format=csv,noheader; torchrun --standalone --nproc_per_node=${NPROC} /scratch/wxu/dh-spatial/run_with_custom.py --config-name pointnav/ddppo_pointnav_blind_gibson habitat.seed=0 habitat_baselines.checkpoint_folder=${OUT_DIR} habitat_baselines.tensorboard_dir=${OUT_DIR}/tb habitat_baselines.total_num_steps=250000000 habitat_baselines.num_environments=16 habitat_baselines.rl.ppo.num_steps=256 habitat_baselines.rl.ppo.use_linear_lr_decay=False hydra.run.dir=${OUT_DIR}/hydra_runs hydra.output_subdir=null 2>&1 | tee -a ${OUT_DIR}/run.log"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu \
    --image="$IMAGE" \
    --gpu="$NPROC" \
    --cpu=32 \
    --memory=128G \
    --pvc=dhlab-scratch:/scratch \
    --pvc=home:/home/wxu \
    --large-shm \
    --command -- bash -c "$INNER_CMD"

echo ""
echo "Submitted. Monitor with:"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job $JOB_NAME --project dhlab-wxu"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod logs $JOB_NAME --follow"
echo "  kubectl exec -n runai-dhlab-wxu \$(kubectl get pods -n runai-dhlab-wxu -l release=$JOB_NAME -o name | head -1 | cut -d/ -f2) -- tail -f ${OUT_DIR}/run.log"
