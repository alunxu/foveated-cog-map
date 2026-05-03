#!/bin/bash
# Excursion-forgetting on all 5 conditions in parallel (5 × 1-GPU jobs).
# Updates §"Boundaries" L428 numbers. F2 (fnorm) launched separately when ready.
set -e
EPS="${1:-100}"
CONDS=(coarse foveated uniform foveated_logpolar blind)
echo "Excursion sweep: ${CONDS[*]}, eps=$EPS each"
for c in "${CONDS[@]}"; do
  echo "--- $c ---"
  bash "$(dirname "$0")/submit_excursion_rcp.sh" "$c" "$EPS" 2>&1 | grep -E "Submitted|error" | head -1
  sleep 2
done
echo
echo "Track: kubectl get pods -n runai-dhlab-wxu | grep ^exc-"
