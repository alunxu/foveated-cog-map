#!/usr/bin/env bash
# Sync results from SCITAS back to local machine
# Run from local machine
set -euo pipefail

USERNAME="wxu"
REMOTE_HOST="izar.epfl.ch"
REMOTE_DIR="/home/${USERNAME}/CS503_Project"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Syncing results ${USERNAME}@${REMOTE_HOST}:${REMOTE_DIR}/outputs/ → ${LOCAL_DIR}/outputs/"

mkdir -p "${LOCAL_DIR}/outputs"

rsync -avz --progress \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='wandb/' \
    "${USERNAME}@${REMOTE_HOST}:${REMOTE_DIR}/outputs/" "${LOCAL_DIR}/outputs/"

echo ""
echo "✅ Results synced to ${LOCAL_DIR}/outputs/"
echo ""
echo "To also download checkpoints (large), run:"
echo "  rsync -avz ${USERNAME}@${REMOTE_HOST}:/scratch/${USERNAME}/CS503_Project/checkpoints/ ${LOCAL_DIR}/outputs/checkpoints/"
