#!/bin/bash
#SBATCH --job-name=habitat_probe
#SBATCH --time=02:00:00
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
#     /scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.9.pth 100
#
# Collects probing data (LSTM hidden states + ground-truth pose), then
# trains linear probes and prints the result table.

CONFIG_NAME=${1:?"Error: config name required (e.g. pointnav/ddppo_pointnav_blind_gibson)"}
CKPT_PATH=${2:?"Error: ckpt path required"}
NUM_EPISODES=${3:-100}

RUN_NAME=$(basename "${CONFIG_NAME}" | sed 's/ddppo_pointnav_//')
PROBE_DIR="/scratch/izar/${USER}/probing_data"
RESULTS_DIR="/scratch/izar/${USER}/probing_results"
NPZ_PATH="${PROBE_DIR}/${RUN_NAME}.npz"
JSON_PATH="${RESULTS_DIR}/${RUN_NAME}.json"

echo "============================================"
echo "  Habitat Probing Pipeline"
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

# Step 1: Collect probing data
echo ""
echo "=== Step 1: Collecting probing data ==="
python -u /home/${USER}/CS503_Project/scripts/habitat_probe_collect.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes=${NUM_EPISODES} \
    --out="${NPZ_PATH}"

if [ $? -ne 0 ]; then
    echo "ERROR: Data collection failed"
    exit 1
fi

# Step 2: Train probes
echo ""
echo "=== Step 2: Training linear probes ==="
python -u /home/${USER}/CS503_Project/scripts/habitat_probe_train.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_PATH}"

echo ""
echo "Probing completed at $(date)"
echo "Data: ${NPZ_PATH}"
echo "Results: ${JSON_PATH}"
