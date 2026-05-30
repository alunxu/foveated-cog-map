#!/bin/bash
# Submit F-LP2 (log-polar foveated + RunningMeanAndVar normaliser) on RCP.
# Companion to F2: removes the disabled-normaliser confound from the
# log-polar condition. Together with F2 (foveated counterpart) this lets
# us flip foveated and foveated_logpolar in the canonical 5-cond pipeline
# to a fully unified-normaliser variant.
#
# Hyperparams: identical to dh-probe-1..4 (seed=0, 250M frames, 16 env,
# ppo.num_steps=256, no LR decay).
#
# Uses PVC-resident inner script to avoid runai-cli inline-quoting bug.
#
# Usage:
#   bash scripts/cluster/submit_F_LP2.sh [job-name]
# Default job name: dh-flp2
#
# Output:  /scratch/wxu/habitat_checkpoints_rcp/<job-name>/
# ETA:     ~17h on 4xH100 to 250M frames

set -e

JOB_NAME="${1:-dh-flp2}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
NPROC=4

echo "=================================================="
echo "  F-LP2 / Foveated-LogPolar-with-Normaliser submit"
echo "  Job name:  $JOB_NAME"
echo "  Config:    pointnav/ddppo_pointnav_foveated_logpolar_normaliser_gibson"
echo "  Policy:    FoveatedLogPolarNormalisedWijmansPolicy"
echo "  GPU:       ${NPROC} (torchrun --nproc_per_node=$NPROC)"
echo "  Frames:    250M (unified)"
echo "=================================================="

INNER_CMD="bash /scratch/wxu/dh-spatial/scripts/cluster/_F_LP2_inner.sh"

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
