#!/bin/bash
#SBATCH --job-name=cs503_evl_traj
#SBATCH --time=04:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Shortcut discovery eval that saves per-episode trajectories.  Used
# for the §4.5 H3 paired-episode visualization (Phase B).
#
# Usage:
#   sbatch submit_shortcut_traj.sh <config> <ckpt> [eps_per_scene] [max_scenes]

CONFIG_NAME=${1:?"config name required"}
CKPT_PATH=${2:?"ckpt path required"}
EPISODES_PER_SCENE=${3:-10}
MAX_SCENES=${4:-20}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

RUN_NAME=$(run_name_from_config "${CONFIG_NAME}")
ABBREV=$(echo "${RUN_NAME}" | sed 's/blind/bld/;s/uniform/uni/;s/foveated/fov/;s/matched/mtc/;s/learned/lrn/')
scontrol update JobId=${SLURM_JOB_ID} JobName="cs503_evl_${ABBREV}_traj" 2>/dev/null || true

OUT_DIR="/scratch/izar/${USER}/shortcut_results"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/${RUN_NAME}_traj.json"
OUT_NPZ="${OUT_DIR}/${RUN_NAME}_traj.npz"

echo "Shortcut + trajectory eval"
echo "  Config:    ${CONFIG_NAME}"
echo "  Ckpt:      ${CKPT_PATH}"
echo "  Eps/scene: ${EPISODES_PER_SCENE}, Max scenes: ${MAX_SCENES}"
echo "  Out:       ${OUT_JSON}, ${OUT_NPZ}"

cd /home/${USER}/habitat-lab

python -u ${PROJECT_DIR}/scripts/eval/shortcut_with_trajectories.py \
    --config-name="${CONFIG_NAME}" \
    --ckpt="${CKPT_PATH}" \
    --episodes-per-scene=${EPISODES_PER_SCENE} \
    --max-scenes=${MAX_SCENES} \
    --out-json="${OUT_JSON}" \
    --out-traj-npz="${OUT_NPZ}"

echo "Completed at $(date)"
