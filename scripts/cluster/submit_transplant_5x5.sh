#!/bin/bash
# 5×5 cross-condition transplant matrix (skip self-transplants = 20 pairs).
# Updates §"Format axis (H2)" Figure 4 right panel.
# 4×4 retrain pairs + blind row/col. F2 (fnorm) excluded for now (added later).
set -e
EPISODES="${1:-200}"
MIDPOINT="${2:-30}"
CONDS=(coarse foveated uniform foveated_logpolar blind)

echo "Transplant 5×5 matrix: ${CONDS[*]} (skip self), eps=$EPISODES, mid=$MIDPOINT"
echo "Total jobs: 20 (5 donors × 5 recipients - 5 self-pairs)"
echo

n=0
for d in "${CONDS[@]}"; do
  for r in "${CONDS[@]}"; do
    [ "$d" = "$r" ] && continue
    n=$((n+1))
    echo "[$n] $d -> $r"
    bash "$(dirname "$0")/submit_transplant_rcp.sh" "$d" "$r" "$EPISODES" "$MIDPOINT" 2>&1 | grep -E "Submitted|error" | head -1
    sleep 2
  done
done
echo
echo "Submitted $n pairs. Track: kubectl get pods -n runai-dhlab-wxu | grep ^tp-"
