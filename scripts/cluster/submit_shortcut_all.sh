#!/bin/bash
# Shortcut consumption on all 5 conditions in parallel (5 × 1-GPU jobs).
# Updates §"Consumption axis" L407 + Figure shortcut_paired_traj.
set -e
EPS_PER_SCENE="${1:-10}"
MAX_SCENES="${2:-20}"
CONDS=(coarse foveated uniform foveated_logpolar blind)
echo "Shortcut sweep: ${CONDS[*]}, eps_per_scene=$EPS_PER_SCENE × max_scenes=$MAX_SCENES"
for c in "${CONDS[@]}"; do
  echo "--- $c ---"
  bash "$(dirname "$0")/submit_shortcut_rcp.sh" "$c" "$EPS_PER_SCENE" "$MAX_SCENES" 2>&1 | grep -E "Submitted|error" | head -1
  sleep 2
done
echo
echo "Track: kubectl get pods -n runai-dhlab-wxu | grep ^sc-"
