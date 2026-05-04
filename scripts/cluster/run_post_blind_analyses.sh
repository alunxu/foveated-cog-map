#!/bin/bash
# Meta-launcher: run all 4 post-blind analyses once blind_izar_det.npz is ready.
# Idempotent — re-running any of these jobs just overwrites the JSONs.
#   mlp_rcp.sh         → mlp_probe.json       (linear+MLP probe, 5 conds)
#   lagk_rcp.sh        → lagk_summary.json    (lag-k profile, 5 conds)
#   submit_skaggs_recompute.sh → skaggs_rectified.json (Skaggs SI, 5 conds)
#   submit_cross_5cond.sh → cross_5cond.json   (Procrustes+CKA 5×5)
#
# Pre-req: blind_izar_det.npz exists at probing_data_rcp/.
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "=== Re-running 4 analysis jobs (blind_izar will be picked up) ==="

bash "$ROOT/scripts/probing/mlp_rcp.sh"
bash "$ROOT/scripts/probing/lagk_rcp.sh"
bash "$ROOT/scripts/cluster/submit_skaggs_recompute.sh"
bash "$ROOT/scripts/cluster/submit_cross_5cond.sh"

echo ""
echo "All 4 jobs submitted. Track via:"
echo "  RUNAI_CURRENT_CTX=rcp kubectl get pods -n runai-dhlab-wxu | grep -E 'mlp-recompute|lagk-recompute|skaggs-recompute|cross-5cond'"
