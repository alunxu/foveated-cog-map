#!/bin/bash
# Submit F2 (foveated-with-normaliser) training on RCP RunAI.
#
# F2 = test whether RunningMeanAndVar input normaliser confounds the H1
# "rich-encoder pass-through" finding. Trains foveated condition with
# normaliser ENABLED (matching uniform/coarse/blind) on the same 4-GPU
# torchrun pattern as the dh-probe-1..4 retrain pods.
#
# Config: habitat_configs/ddppo_pointnav_foveated_normaliser_gibson.yaml
#         (FoveatedNormalisedWijmansPolicy → _force_enable_normaliser=True)
#
# Usage:
#   bash scripts/cluster/submit_F2_normaliser.sh [job-name]
#
# Default job name: dh-fnorm
#
# Output:    /scratch/wxu/habitat_checkpoints_rcp/<job-name>/
# Log:       /scratch/wxu/habitat_checkpoints_rcp/<job-name>/run.log
# ETA:       ~17h on 4xH100 to 250M frames (vs. dh-probe-2 foveated reference)
#
# Recommended: launch AFTER dh-probe-1 (coarse) finishes (~4h from current
# state) so we don't oversubscribe the namespace's GPU quota.

set -e

JOB_NAME="${1:-dh-fnorm}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
NPROC=4

echo "=================================================="
echo "  F2 / F-norm submit"
echo "  Job name:  $JOB_NAME"
echo "  Config:    pointnav/ddppo_pointnav_foveated_normaliser_gibson"
echo "  Policy:    FoveatedNormalisedWijmansPolicy"
echo "  GPU:       ${NPROC} (torchrun --nproc_per_node=$NPROC)"
echo "  Frames:    250M (unified retrain budget)"
echo "  Image:     $IMAGE"
echo "=================================================="

# Mirror the dh-probe-1..4 pod command exactly. The CLI hyperparam overrides
# are now redundant (config YAML already has unified values) but kept as
# a defensive belt-and-suspenders against config drift.
INNER_CMD="
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH
HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config
mkdir -p \$HB_CONFIG/pointnav
for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do
  n=\$(basename \$cfg)
  [ ! -e \$HB_CONFIG/pointnav/\$n ] && ln -s \$cfg \$HB_CONFIG/pointnav/\$n
done
OUT_DIR=/scratch/wxu/habitat_checkpoints_rcp/${JOB_NAME}
mkdir -p \$OUT_DIR
nvidia-smi --query-gpu=name --format=csv,noheader
torchrun --standalone --nproc_per_node=${NPROC} /scratch/wxu/dh-spatial/run_with_custom.py \\
    --config-name pointnav/ddppo_pointnav_foveated_normaliser_gibson \\
    habitat.seed=0 \\
    habitat_baselines.checkpoint_folder=\$OUT_DIR \\
    habitat_baselines.tensorboard_dir=\$OUT_DIR/tb \\
    habitat_baselines.total_num_steps=250000000 \\
    habitat_baselines.num_environments=16 \\
    habitat_baselines.rl.ppo.num_steps=256 \\
    habitat_baselines.rl.ppo.use_linear_lr_decay=False \\
    hydra.run.dir=\$OUT_DIR/hydra_runs hydra.output_subdir=null 2>&1 | tee -a \$OUT_DIR/run.log
"

# Submit via runai-rcp-prod with 4-GPU node
RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
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
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job $JOB_NAME"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod logs $JOB_NAME --follow"
echo "  kubectl exec -n runai-dhlab-wxu \$(kubectl get pods -n runai-dhlab-wxu -l release=$JOB_NAME -o name | head -1 | cut -d/ -f2) -- tail -f /scratch/wxu/habitat_checkpoints_rcp/${JOB_NAME}/run.log"
