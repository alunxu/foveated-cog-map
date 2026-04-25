#!/bin/bash
#SBATCH --job-name=cs503_encfeat
#SBATCH --time=03:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Collect encoder feature-map output (post ResNet, pre-LSTM) for one
# condition, then probe it for GPS / compass.

CONFIG_NAME=${1:?config name required}
CKPT_PATH=${2:?ckpt path required}
COND_SHORT=${3:?short cond name required (e.g. matched)}
NUM_EPISODES=${4:-300}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

OUT_NPZ="${PROBE_DIR}/${COND_SHORT}_gibson_encfeat_det.npz"

cd /home/${USER}/habitat-lab

# Step 1: collect encoder features
python -u "${PROJECT_DIR}/scripts/probing/collect_encoder_features.py" \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes=${NUM_EPISODES} \
    --out="${OUT_NPZ}"

# Step 2: probe (single condition; the multi-condition aggregator can
# be re-run on demand)
python -u "${PROJECT_DIR}/scripts/probing/analyze_encoder_features.py" \
    --in-dir "${PROBE_DIR}" \
    --conditions ${COND_SHORT} \
    --suffix _encfeat_det \
    --out "${RESULTS_DIR}/${COND_SHORT}_encoder_features_det.json"
