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

# Avoid runai-cli inline-quoting issues (the same bug that bit
# dh-blind-resume on 2026-05-12 — bash -c argument got truncated/reparsed
# silently). Full launch logic lives in a PVC-resident script
# (_F2_normaliser_inner.sh); INNER_CMD just invokes it.
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/${JOB_NAME}"
INNER_CMD="bash /scratch/wxu/dh-spatial/scripts/cluster/_F2_normaliser_inner.sh"

# Submit via runai-rcp-prod with 4-GPU node, dhlab-wxu project
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
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job $JOB_NAME"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod logs $JOB_NAME --follow"
echo "  kubectl exec -n runai-dhlab-wxu \$(kubectl get pods -n runai-dhlab-wxu -l release=$JOB_NAME -o name | head -1 | cut -d/ -f2) -- tail -f /scratch/wxu/habitat_checkpoints_rcp/${JOB_NAME}/run.log"
