#!/bin/bash
#SBATCH --job-name=cs503_probe_agent
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

# Probe-agent (Wijmans Fig 3 replication) for one condition.
#
# Usage:
#   sbatch submit_probe_agent.sh <cond_name> <config> <ckpt> [episodes]

COND_NAME=${1:?cond name required (e.g. blind)}
CONFIG=${2:?config required (e.g. pointnav/ddppo_pointnav_blind_gibson)}
CKPT=${3:?ckpt path required}
EPISODES=${4:-150}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

OUT_DIR="/scratch/izar/${USER}/probe_agent_results"
OUT_PATH="${OUT_DIR}/${COND_NAME}.json"
mkdir -p "${OUT_DIR}"

echo "============================================"
echo "  Probe-Agent (Wijmans Fig 3) — ${COND_NAME}"
echo "  Config:    ${CONFIG}"
echo "  Ckpt:      ${CKPT}"
echo "  Episodes:  ${EPISODES}"
echo "  Output:    ${OUT_PATH}"
echo "  Date:      $(date)"
echo "============================================"

cd /home/${USER}/habitat-lab

python -u ${PROJECT_DIR}/scripts/eval/probe_agent.py \
    --config="${CONFIG}" \
    --ckpt="${CKPT}" \
    --episodes=${EPISODES} \
    --out="${OUT_PATH}"
