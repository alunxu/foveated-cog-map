#!/bin/bash
# Blind RESUME on RCP RunAI (4 GPU torchrun).
#
# Why resume:
#   Previous dh-blind-0-1 run (started May 8 13:15, ended May 9 10:21 UTC)
#   hit Run:AI 24h wall-clock and stopped cleanly at:
#     Num updates: 12800, Num frames: 195,154,064 (~78% of 250M target)
#     exit code: 0 (graceful), .habitat-resume-state.pth saved
#   We want the remaining ~55M frames (~5-7h more on 4xH100) to finish at
#   250M to match the 4 sighted retrain conditions.
#
# How resume works:
#   Habitat PPOTrainer auto-resumes from OUT_DIR/.habitat-resume-state.pth
#   if present. We point this job at the SAME OUT_DIR as dh-blind so the
#   resume state is found; only the JOB_NAME changes (k8s pod uniqueness).
#
# Config: identical to submit_blind_retrain.sh (commit 9674b8c hyperparams).
#
# Usage:
#   bash scripts/cluster/submit_blind_resume.sh [job-name]
# Default job name: dh-blind-resume

set -e

JOB_NAME="${1:-dh-blind-resume}"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
NPROC=4
# CRITICAL: same OUT_DIR as the original dh-blind run, so the resume
# state pth is found and PPOTrainer continues from frame 195M.
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/dh-blind"

echo "=================================================="
echo "  Blind RESUME (RCP) — picking up from ~195M frames"
echo "  Job:     $JOB_NAME"
echo "  Resume:  $OUT_DIR/.habitat-resume-state.pth (PPOTrainer auto-detects)"
echo "  Target:  250M (~5-7h more on 4xH100)"
echo "  Config:  pointnav/ddppo_pointnav_blind_gibson"
echo "  GPU:     ${NPROC} (torchrun --nproc_per_node=$NPROC)"
echo "=================================================="

# Avoid runai-cli inline-quoting issues: full launch logic lives in a
# PVC-resident script (_resume_blind_inner.sh). INNER_CMD just invokes it.
INNER_CMD="bash /scratch/wxu/dh-spatial/scripts/cluster/_resume_blind_inner.sh"

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
echo ""
echo "Expect first log line confirming resume from frame ~195M, like:"
echo "  'Resuming from checkpoint at 195154064 frames'"
echo "Then ~5-7h to reach 250M; ckpts.{36-49} should appear in ${OUT_DIR}."
