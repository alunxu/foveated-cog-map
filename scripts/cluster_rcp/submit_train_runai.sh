#!/bin/bash
# Submit a training job on RCP RunAI (RCP cluster).
#
# Each invocation:
#   1. Builds a runai submit command for the requested config
#   2. Mounts persistent /scratch/wxu PVC (where conda env + datasets live)
#   3. Activates the pre-built habitat conda env and runs run_habitat.py
#
# Usage:
#   bash scripts/cluster/submit_train_runai.sh <config-name> [--seed N] [--gpu-type h200|a100]
#
# Examples:
#   bash scripts/cluster/submit_train_runai.sh pointnav/ddppo_pointnav_blind_gibson
#   bash scripts/cluster/submit_train_runai.sh pointnav/ddppo_pointnav_coarse_gibson --seed 2
#   bash scripts/cluster/submit_train_runai.sh pointnav/ddppo_pointnav_foveated_stochastic_gibson --gpu-type h200

set -e

CONFIG_NAME="${1:?'config name required (e.g. pointnav/ddppo_pointnav_blind_gibson)'}"
shift || true

SEED=""
GPU_TYPE="h200"  # default
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Derive a job name from the config (RunAI requires lowercase + dash-separated, no underscores)
RUN_NAME=$(echo "$CONFIG_NAME" | sed 's|.*/ddppo_pointnav_||' | sed 's|_gibson||' | tr '_' '-')
if [ -n "$SEED" ]; then
    RUN_NAME="${RUN_NAME}-s${SEED}"
fi
JOB_NAME="dh-spatial-tr-${RUN_NAME}"

echo "=================================================="
echo "  RunAI submit"
echo "  Config:    $CONFIG_NAME"
echo "  Seed:      ${SEED:-default}"
echo "  GPU type:  $GPU_TYPE"
echo "  Job name:  $JOB_NAME"
echo "=================================================="

# Pick image — use minimal pytorch base; conda env on PVC handles the rest
IMAGE="registry.rcp.epfl.ch/ethical-probe/gemma-pipeline:v3"

# Build seed override flag for hydra
SEED_FLAG=""
if [ -n "$SEED" ]; then
    SEED_FLAG="habitat.seed=$SEED"
fi

# Inner command: source conda env + run training
INNER_CMD="
set -e
source /scratch/wxu/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /scratch/wxu/dh-spatial
mkdir -p logs/${RUN_NAME}
echo \"=== starting training: $CONFIG_NAME ===\"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
python -u scripts/cluster/run_habitat.py \\
    --config-name=$CONFIG_NAME \\
    hydra.job.chdir=False \\
    habitat_baselines.evaluate=False \\
    $SEED_FLAG \\
    2>&1 | tee logs/${RUN_NAME}/run.log
"

# Submit via runai
RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --image="$IMAGE" \
    --node-pools="$GPU_TYPE" \
    --gpu=1 \
    --cpu=8 \
    --memory=64G \
    --pvc=dhlab-scratch:/scratch \
    --pvc=home:/home/wxu \
    --large-shm \
    --command -- bash -c "$INNER_CMD"

echo ""
echo "Submitted. Check status with:"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job $JOB_NAME"
echo "Logs:"
echo "  /scratch/wxu/dh-spatial/logs/${RUN_NAME}/run.log  (inside pod)"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod logs $JOB_NAME --follow"
