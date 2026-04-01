#!/usr/bin/env bash
# Sync local Project/ directory to SCITAS Home
# Run from local machine
set -euo pipefail

USERNAME="wxu"
REMOTE_HOST="izar.epfl.ch"
REMOTE_DIR="/home/${USERNAME}/CS503_Project"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Syncing ${LOCAL_DIR} → ${USERNAME}@${REMOTE_HOST}:${REMOTE_DIR}"

rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='outputs/' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' \
    --exclude='wandb/' \
    --exclude='node_modules/' \
    "${LOCAL_DIR}/" "${USERNAME}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
echo "✅ Sync complete: ${REMOTE_DIR}"
echo "   Next: ssh ${USERNAME}@${REMOTE_HOST}"
echo "         cd ${REMOTE_DIR}"
