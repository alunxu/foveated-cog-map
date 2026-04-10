#!/bin/bash
#SBATCH --job-name=habitat_probe
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

# Usage:
#   sbatch submit_habitat_probe.sh <config_name> <ckpt_path> [num_episodes]
# Example:
#   sbatch scripts/cluster/submit_habitat_probe.sh \
#     pointnav/ddppo_pointnav_blind_gibson \
#     /scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.9.pth 500
#
# Pipeline:
#   1. Collect probing data (all LSTM layers h+c, ground-truth pose)
#   2. Run comprehensive single-condition analysis (baseline probes,
#      control tasks, rate maps, cross-heading generalization)
#   3. (Optional) Run legacy probe for backward compatibility

CONFIG_NAME=${1:?"Error: config name required (e.g. pointnav/ddppo_pointnav_blind_gibson)"}
CKPT_PATH=${2:?"Error: ckpt path required"}
NUM_EPISODES=${3:-500}

RUN_NAME=$(basename "${CONFIG_NAME}" | sed 's/ddppo_pointnav_//')
PROBE_DIR="/scratch/izar/${USER}/probing_data"
RESULTS_DIR="/scratch/izar/${USER}/probing_results"
NPZ_PATH="${PROBE_DIR}/${RUN_NAME}.npz"
JSON_PATH="${RESULTS_DIR}/${RUN_NAME}_analysis.json"
JSON_LEGACY="${RESULTS_DIR}/${RUN_NAME}.json"

echo "============================================"
echo "  Habitat Probing Pipeline (v2)"
echo "  Config:   ${CONFIG_NAME}"
echo "  Ckpt:     ${CKPT_PATH}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Node:     $(hostname)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Date:     $(date)"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate habitat

# llvmlite (pulled by quaternion → numba) needs a newer libstdc++ than
# some cluster nodes provide in /lib64. Conda's own copy fixes this.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/${USER}/CS503_Project:${PYTHONPATH}"

DATA_DIR="/scratch/izar/${USER}/habitat_data"

cd /home/${USER}/habitat-lab

# Step 1: Collect probing data (all LSTM layers)
echo ""
echo "=== Step 1: Collecting probing data (all layers) ==="
python -u /home/${USER}/CS503_Project/scripts/habitat_probe_collect.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes=${NUM_EPISODES} \
    --collect-occupancy \
    --out="${NPZ_PATH}"

if [ $? -ne 0 ]; then
    echo "ERROR: Data collection failed"
    exit 1
fi

# Step 2: Comprehensive analysis
echo ""
echo "=== Step 2: Comprehensive probing analysis ==="
python -u /home/${USER}/CS503_Project/scripts/habitat_probe_analysis.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_PATH}" \
    --pca-dim=0 \
    --min-steps-scene=15

if [ $? -ne 0 ]; then
    echo "WARNING: Comprehensive analysis failed, falling back to legacy"
fi

# Step 3: Legacy probes (backward compat)
echo ""
echo "=== Step 3: Legacy probe (backward compat) ==="
python -u /home/${USER}/CS503_Project/scripts/habitat_probe_train.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_LEGACY}"

echo ""
echo "Probing completed at $(date)"
echo "Data:     ${NPZ_PATH}"
echo "Analysis: ${JSON_PATH}"
echo "Legacy:   ${JSON_LEGACY}"
