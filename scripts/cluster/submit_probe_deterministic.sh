#!/bin/bash
#SBATCH --job-name=cs503_prb_det
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

# Re-collect probe data with deterministic=True action selection.
# Writes to <run_name>_det.npz so we don't overwrite the stochastic
# collection (which we may need for diffing).
#
# Usage:
#   sbatch submit_probe_deterministic.sh <config_name> <ckpt_path> [num_episodes]

CONFIG_NAME=${1:?"config name required"}
CKPT_PATH=${2:?"ckpt path required"}
NUM_EPISODES=${3:-500}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}")
ABBREV=$(echo "${RUN_NAME}" | sed 's/blind/bld/;s/uniform/uni/;s/foveated/fov/;s/matched/mtc/;s/learned/lrn/')
scontrol update JobId=${SLURM_JOB_ID} JobName="cs503_prb_det_${ABBREV}" 2>/dev/null || true

NPZ_PATH="${PROBE_DIR}/${RUN_NAME}_det.npz"
JSON_PATH="${RESULTS_DIR}/${RUN_NAME}_det_analysis.json"

echo "============================================"
echo "  Deterministic Probing Pipeline"
echo "  Config:   ${CONFIG_NAME}"
echo "  Ckpt:     ${CKPT_PATH}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Out NPZ:  ${NPZ_PATH}"
echo "  Node:     $(hostname)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "============================================"

cd /home/${USER}/habitat-lab

# Step 1: Collect (deterministic)
python -u ${PROJECT_DIR}/scripts/probing/collect.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes=${NUM_EPISODES} \
    --deterministic=True \
    --collect-occupancy \
    --out="${NPZ_PATH}"

if [ $? -ne 0 ]; then
    echo "ERROR: Deterministic collection failed"
    exit 1
fi

# Step 2: Analysis pipeline
python -u ${PROJECT_DIR}/scripts/probing/analyze.py \
    --data="${NPZ_PATH}" \
    --out="${JSON_PATH}" \
    --pca-dim=0 \
    --min-steps-scene=15

# Step 3: Quick sanity summary: episode length + success rate
python -u - <<PY
import numpy as np
d = np.load("${NPZ_PATH}", allow_pickle=True)
ep = d["episode_ids"]
d2g = d["distance_to_goal"]
pos = d["positions"]
goal = d["goal_positions"]

eps = np.unique(ep)
lens, finals, starts = [], [], []
for e in eps:
    idx = np.where(ep == e)[0]
    lens.append(len(idx))
    finals.append(np.linalg.norm(pos[idx[-1]] - goal[idx[0]]))
    starts.append(d2g[idx[0]])
lens = np.array(lens); finals = np.array(finals); starts = np.array(starts)
print(f"\n== Post-collection sanity ==")
print(f"Episodes: {len(eps)}")
print(f"Ep len: mean {lens.mean():.1f}, median {np.median(lens):.0f}, min {lens.min()}, max {lens.max()}")
print(f"Start dist to goal: {starts.mean():.2f} m")
print(f"Final dist to goal: {finals.mean():.2f} m")
print(f"Success (<0.2m):    {100*(finals<0.2).mean():.1f} %")
print(f"Success (<1.0m):    {100*(finals<1.0).mean():.1f} %")
PY

echo "Done at $(date)"
