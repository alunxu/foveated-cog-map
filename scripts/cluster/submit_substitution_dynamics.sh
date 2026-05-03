#!/bin/bash
# Substitution-dynamics cross-checkpoint probing.
# Updates §H1 L260 decay-rate paragraph + Figure fig:substitution_dynamics.
#
# Probes each condition at 5 training stages (~50M, 100M, 150M, 200M, 250M).
# With num_checkpoints=50 over 250M frames, 1 ckpt per 5M frames.
# Default ckpts: 10, 20, 30, 40, 49 → ~50/100/150/200/250 M frames.
#
# Total jobs: 5 conds × 5 ckpts = 25 (sequential per ckpt-num is OK; parallel
# across conds within each ckpt-num is what we do). With 1 GPU each = 25 GPU-h.
#
# Usage:
#   bash scripts/cluster/submit_substitution_dynamics.sh [episodes_per_ckpt]
# Note: episodes 200 is enough for cross-ckpt probing (less than the canonical 500).
set -e

EPS="${1:-200}"
CONDS=(coarse foveated uniform foveated_logpolar blind)
CKPTS=(10 20 30 40 49)

echo "Substitution dynamics: ${#CONDS[@]} conds × ${#CKPTS[@]} ckpts = $((${#CONDS[@]} * ${#CKPTS[@]})) jobs"
echo "Episodes per ckpt: $EPS"
echo

n=0
for c in "${CONDS[@]}"; do
  for k in "${CKPTS[@]}"; do
    n=$((n+1))
    echo "[$n] $c ckpt.$k"
    bash "$(dirname "$0")/submit_probe_collect_rcp.sh" "$c" "$EPS" "$k" 2>&1 | grep -E "Submitted|error" | head -1
    sleep 2
  done
done
echo
echo "Submitted $n cross-ckpt probes. Track: kubectl get pods -n runai-dhlab-wxu | grep -E '^probe-[1-6]-c'"
