#!/bin/bash
# ship_to_izar.sh — friend-side helper to rsync a finished hc training
# run's checkpoints + tensorboard data to wxu's Izar account.
#
# Usage:
#   bash scripts/cluster/ship_to_izar.sh <run_name>
#
# Examples:
#   bash scripts/cluster/ship_to_izar.sh foveated_stochastic_gibson
#   bash scripts/cluster/ship_to_izar.sh matched64_gibson
#   bash scripts/cluster/ship_to_izar.sh blind_gibson_seed2
#
# What it sends:
#   - latest.pth                          (final ckpt for main paper Table 1)
#   - ckpt.10/20/30/40/49.pth             (across-training, for §4.2 fig 3)
#   - tb/                                 (training curves, for appfig 7)
#
# Pre-requisites (one-time):
#   1. Generated SSH keypair: ~/.ssh/id_izar_wxu_deploy
#   2. wxu has appended ~/.ssh/id_izar_wxu_deploy.pub to Izar's
#      ~/.ssh/authorized_keys
#   3. Verified `ssh -i ~/.ssh/id_izar_wxu_deploy wxu@izar.epfl.ch 'echo ok'`
#
# After this script succeeds, wxu's probe_hc_arrival.sh on Izar will
# automatically submit the probing pipeline against the new ckpts.

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name>"
    echo ""
    echo "Examples:"
    echo "  $0 foveated_stochastic_gibson"
    echo "  $0 matched64_gibson"
    echo "  $0 blind_gibson_seed2"
    exit 1
fi

RUN_NAME="$1"

# === Edit these two paths to match your cluster's layout ===
HC_CKPT_BASE="${HC_CKPT_BASE:-/path/to/hc/checkpoints}"  # override via env
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_izar_wxu_deploy}"
# ============================================================

HC_CKPT_DIR="$HC_CKPT_BASE/$RUN_NAME"
IZAR_USER="wxu"
IZAR_HOST="izar.epfl.ch"
IZAR_DEST="/scratch/izar/${IZAR_USER}/habitat_checkpoints/${RUN_NAME}"

# Sanity checks
if [ ! -d "$HC_CKPT_DIR" ]; then
    echo "ERROR: $HC_CKPT_DIR does not exist."
    echo "Make sure HC_CKPT_BASE points to your local checkpoints root."
    exit 1
fi

if [ ! -f "$HC_CKPT_DIR/latest.pth" ]; then
    echo "ERROR: $HC_CKPT_DIR/latest.pth missing — has training started?"
    exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
    echo "ERROR: SSH key $SSH_KEY not found."
    echo "See docs/hc_launch_recipe.md § 'How to ship checkpoints' for setup."
    exit 1
fi

# Build file list — only send what exists, don't error on missing intermediate
files=("latest.pth")
for ck in 10 20 30 40 49; do
    if [ -f "$HC_CKPT_DIR/ckpt.${ck}.pth" ]; then
        files+=("ckpt.${ck}.pth")
    fi
done

echo "==================================================="
echo "  ship_to_izar.sh"
echo "  Run name:  $RUN_NAME"
echo "  Source:    $HC_CKPT_DIR"
echo "  Files:     ${files[*]} + tb/"
echo "  Dest:      ${IZAR_USER}@${IZAR_HOST}:${IZAR_DEST}/"
echo "==================================================="

# Ensure dest directory exists on Izar
ssh -i "$SSH_KEY" "${IZAR_USER}@${IZAR_HOST}" "mkdir -p '$IZAR_DEST'"

# rsync the ckpt files + tb dir
rsync -avzP -e "ssh -i $SSH_KEY" \
    "${files[@]/#/${HC_CKPT_DIR}/}" \
    "${HC_CKPT_DIR}/tb/" \
    "${IZAR_USER}@${IZAR_HOST}:${IZAR_DEST}/"

echo ""
echo "✓ Shipped $RUN_NAME to Izar."
echo ""
echo "wxu's probe_hc_arrival.sh on Izar will detect the new files within"
echo "the next cron tick (≤15 min) and submit the probing pipeline."
