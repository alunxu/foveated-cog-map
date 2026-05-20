#!/bin/bash
# ─── Shared environment setup for all SLURM jobs ───
# Source this at the top of every submit_*.sh script:
#   source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

eval "$(conda shell.bash hook)"
conda activate habitat

# llvmlite (pulled by quaternion → numba) needs a newer libstdc++
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Silence Habitat/Magnum spam
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1

# Project root for custom policies + shared utilities
export PYTHONPATH="/home/${USER}/cs503-project:${PYTHONPATH}"

# Standard cluster paths
export DATA_DIR="/scratch/izar/${USER}/habitat_data"
export CKPT_DIR="/scratch/izar/${USER}/habitat_checkpoints"
export PROBE_DIR="/scratch/izar/${USER}/probing_data"
export RESULTS_DIR="/scratch/izar/${USER}/probing_results"
export PROJECT_DIR="/home/${USER}/cs503-project"

# Helper: extract short run name from config path
# e.g. "pointnav/ddppo_pointnav_blind_gibson" → "blind_gibson"
run_name_from_config() {
    basename "$1" | sed 's/ddppo_pointnav_//'
}
