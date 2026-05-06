#!/bin/bash
# Probe blind_izar at ckpt 5 (50M-equivalent) and ckpt 25 (250M-equivalent)
# to fill in fig_magnitude.pdf Panel B blind curve at the [50M, 250M] window
# endpoints (currently only ckpt10/20 = 100M/200M are within window).
#
# Each job ~1.75h on A100. Submit both in parallel (different ckpt nums get
# different job names per submit_probe_collect_rcp.sh).
#
# Usage:
#   bash scripts/cluster/submit_blind_50_250.sh
set -e

EPS="${1:-200}"  # 200 ep for ckpt-sweep speed (vs 500 for canonical)

echo "Submitting blind_izar ckpt5 (50M-equivalent)..."
bash "$(dirname "$0")/submit_probe_collect_rcp.sh" blind_izar "$EPS" 5

echo
echo "Submitting blind_izar ckpt25 (250M-equivalent, full 500 ep — this is the canonical converged endpoint)..."
bash "$(dirname "$0")/submit_probe_collect_rcp.sh" blind_izar 500 25

echo
echo "Both submitted. After completion, ANALYSE step also needs to run:"
echo "  python scripts/probing/analyze.py --data <npz> --out <json>"
echo "to produce blind_izar_det_ckpt{5,25}_analysis.json files; the figure"
echo "script (make_magnitude_3panel.py) will pick them up automatically once"
echo "they land in /tmp/rcp_analysis/."
