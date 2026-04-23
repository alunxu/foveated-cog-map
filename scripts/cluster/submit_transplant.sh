#!/bin/bash
#SBATCH --job-name=cs503_transplant
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
#   sbatch submit_transplant.sh <donor_name> <donor_config> <donor_ckpt> \
#                                <recip_name> <recip_config> <recip_ckpt> \
#                                [episodes]
#
# Runs mid-episode hidden-state transplant between two trained agents.
# Baseline vs self-transplant vs cross-transplant SPL comparison. See
# scripts/eval/transplant.py for the protocol.

DONOR_NAME=${1:?donor name required (e.g. foveated)}
DONOR_CFG=${2:?donor config required}
DONOR_CKPT=${3:?donor ckpt required}
RECIP_NAME=${4:?recipient name required}
RECIP_CFG=${5:?recipient config required}
RECIP_CKPT=${6:?recipient ckpt required}
EPISODES=${7:-150}
MIDPOINT_STEP=${8:-30}  # episodes avg ~100 steps; 30 = first 1/3 captured by donor

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

OUT_DIR="/scratch/izar/${USER}/transplant_results"
OUT_PATH="${OUT_DIR}/${DONOR_NAME}_to_${RECIP_NAME}.json"
mkdir -p "${OUT_DIR}"

echo "============================================"
echo "  Memory Transplant Evaluation"
echo "  Donor:      ${DONOR_NAME} (${DONOR_CFG})"
echo "  Recipient:  ${RECIP_NAME} (${RECIP_CFG})"
echo "  Episodes:   ${EPISODES}"
echo "  Output:     ${OUT_PATH}"
echo "  Date:       $(date)"
echo "============================================"

cd /home/${USER}/habitat-lab

python -u ${PROJECT_DIR}/scripts/eval/transplant.py \
    --donor-config="${DONOR_CFG}" \
    --donor-ckpt="${DONOR_CKPT}" \
    --recipient-config="${RECIP_CFG}" \
    --recipient-ckpt="${RECIP_CKPT}" \
    --episodes=${EPISODES} \
    --midpoint-step=${MIDPOINT_STEP} \
    --out="${OUT_PATH}"
