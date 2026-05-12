#!/bin/bash
# Excursion-forgetting on all 5 conditions in parallel (5 × 1-GPU jobs).
# Updates §"Boundaries" L428 numbers. F2 (fnorm) launched separately when ready.
set -e
EPS="${1:-100}"
# Blind excluded for now — using own dh-blind retrain instead of friend's
# inconsistent seed=100 ckpt. Add `blind` to this list after dh-blind ckpt.49 ready.
CONDS=(coarse foveated uniform foveated_logpolar)
echo "Excursion sweep: ${CONDS[*]}, eps=$EPS each"
for c in "${CONDS[@]}"; do
  echo "--- $c ---"
  bash "$(dirname "$0")/submit_excursion_rcp.sh" "$c" "$EPS" 2>&1 | grep -E "Submitted|error" | head -1
  sleep 2
done
echo
echo "Track: kubectl get pods -n runai-dhlab-wxu | grep ^exc-"
