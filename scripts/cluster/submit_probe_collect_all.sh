#!/bin/bash
# Phase A launcher: probe-collect on ALL 5 currently-available conditions
# in parallel (5 separate runai jobs, 1 GPU each). F2 (foveated_normaliser)
# launched separately when its ckpt.49 is ready (~05-05).
#
# Usage:
#   bash scripts/cluster/submit_probe_collect_all.sh [episodes]
#
# Total time: ~1.75h wall (parallel) vs 8.75h serial.
# Total GPU-hours: ~9 (5 conditions × 1.75h on A100).
# Output: /scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/<cond>_det.npz

set -e

EPISODES="${1:-500}"

CONDITIONS=(coarse foveated uniform foveated_logpolar blind)

echo "=================================================="
echo "  Phase A: probe-collect ALL conditions"
echo "  Episodes per condition: $EPISODES"
echo "  Conditions: ${CONDITIONS[*]}"
echo "  Note: F2 (fnorm) NOT included; launch separately when ckpt.49 ready"
echo "=================================================="
echo

for cond in "${CONDITIONS[@]}"; do
  echo "--- launching $cond ---"
  bash "$(dirname "$0")/submit_probe_collect_rcp.sh" "$cond" "$EPISODES" 2>&1 | grep -E "Submitted|Job.*submitted|error|fail" | head -3
  sleep 2  # avoid runai rate-limit
  echo
done

echo "=================================================="
echo "  All 5 jobs submitted. Track via:"
echo "    kubectl get pods -n runai-dhlab-wxu | grep dh-pcr-"
echo "    RUNAI_CURRENT_CTX=rcp runai-rcp-prod list jobs --project dhlab-wxu | grep dh-pcr"
echo "=================================================="
