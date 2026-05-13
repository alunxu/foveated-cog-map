#!/bin/bash
# submit_collect_all.sh
#
# Submits 5 parallel collect jobs (one per condition), then submits the
# cogneuro analysis job with a dependency on all 5 completing.
#
# Usage (from repo root):
#   bash scripts/probing/submit_collect_all.sh

set -euo pipefail

REPO_DIR="/home/bruneau/CS503/foveated-cog-map"
CKPT_DIR="${REPO_DIR}/checkpoints"
PROBING_DIR="/scratch/izar/bruneau/probing_data"
RESULTS_DIR="/scratch/izar/bruneau/probing_results"
HABITAT_DATA="/scratch/izar/bruneau/habitat_data"
LOG_DIR="${REPO_DIR}/slurm_logs"

mkdir -p "${PROBING_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

submit_collect() {
    local cond=$1; local config=$2; local ckpt=$3
    sbatch --parsable \
        --job-name="collect_${cond}" \
        --gres=gpu:1 --cpus-per-task=8 --mem=16G \
        --time=04:00:00 --partition=gpu \
        --output="${LOG_DIR}/%j_collect_${cond}.out" \
        --error="${LOG_DIR}/%j_collect_${cond}.err" \
        --wrap="
set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate habitat
export LD_LIBRARY_PATH=/home/bruneau/miniconda3/envs/habitat/lib:\${LD_LIBRARY_PATH:-}
export HABITAT_DATA_DIR=${HABITAT_DATA}
cd ${REPO_DIR}
echo Node: \$(hostname)  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)
python scripts/probing/collect.py \
    --config-name ${config} \
    --ckpt ${ckpt} \
    --episodes 500 \
    --out ${PROBING_DIR}/${cond}.npz \
    --deterministic true \
    --checkpoint-every 50
echo collect done: ${cond}
"
}

echo "Submitting collect jobs..."

JID_BLIND=$(submit_collect \
    "blind_gibson" \
    "ddppo_pointnav_blind_gibson" \
    "${CKPT_DIR}/blind/ckpt.34.pth")
echo "  blind_gibson              → job ${JID_BLIND}"

JID_COARSE=$(submit_collect \
    "coarse_gibson" \
    "ddppo_pointnav_coarse_gibson" \
    "${CKPT_DIR}/coarse/ckpt.49.pth")
echo "  coarse_gibson             → job ${JID_COARSE}"

JID_FOVEATED=$(submit_collect \
    "foveated_gibson" \
    "ddppo_pointnav_foveated_gibson" \
    "${CKPT_DIR}/foveated/ckpt.49.pth")
echo "  foveated_gibson           → job ${JID_FOVEATED}"

JID_LOGPOLAR=$(submit_collect \
    "foveated_logpolar_gibson" \
    "ddppo_pointnav_foveated_logpolar_gibson" \
    "${CKPT_DIR}/foveated_logpolar/ckpt.49.pth")
echo "  foveated_logpolar_gibson  → job ${JID_LOGPOLAR}"

JID_UNIFORM=$(submit_collect \
    "uniform_gibson" \
    "ddppo_pointnav_uniform_gibson" \
    "${CKPT_DIR}/uniform/ckpt.49.pth")
echo "  uniform_gibson            → job ${JID_UNIFORM}"

ALL_COLLECT="${JID_BLIND}:${JID_COARSE}:${JID_FOVEATED}:${JID_LOGPOLAR}:${JID_UNIFORM}"

JID_ANALYSIS=$(sbatch --parsable \
    --job-name="cogneuro_analyses" \
    --cpus-per-task=8 --mem=32G \
    --time=04:00:00 --partition=cpu \
    --dependency="afterok:${ALL_COLLECT}" \
    --output="${LOG_DIR}/%j_cogneuro.out" \
    --error="${LOG_DIR}/%j_cogneuro.err" \
    --wrap="
set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate habitat-analysis
export LD_LIBRARY_PATH=/home/bruneau/miniconda3/envs/habitat-analysis/lib:\${LD_LIBRARY_PATH:-}
cd ${REPO_DIR}
mkdir -p ${RESULTS_DIR} docs/manuscript/fig slurm_logs

echo '── grid-cell signature ──'
python scripts/probing/grid_cell_signature.py \
    --in-dir    ${PROBING_DIR} \
    --conds     blind_gibson coarse_gibson foveated_gibson foveated_logpolar_gibson uniform_gibson \
    --out       ${RESULTS_DIR}/grid_cell_signature.json \
    --n-bins    24 --n-shuffles 200 --n-top-scenes 10 --min-scene-steps 500 \
    --layer -1 --save-maps

echo '── time cells ──'
python scripts/probing/time_cells.py \
    --in-dir    ${PROBING_DIR} \
    --conds     blind_gibson coarse_gibson foveated_gibson foveated_logpolar_gibson uniform_gibson \
    --out       ${RESULTS_DIR}/time_cells.json \
    --out-npz   ${RESULTS_DIR}/time_cells_curves.npz \
    --n-bins    20 --n-shuffles 500 --min-ep-steps 20 --layer -1

echo '── figures ──'
python scripts/paper_figures/make_grid_cell_figure.py \
    --data ${RESULTS_DIR}/grid_cell_signature.json \
    --out  docs/manuscript/fig/grid_cell_signature.pdf

python scripts/paper_figures/make_time_cells_figure.py \
    --data   ${RESULTS_DIR}/time_cells.json \
    --curves ${RESULTS_DIR}/time_cells_curves.npz \
    --out    docs/manuscript/fig/time_cells.pdf

echo All done. Results in ${RESULTS_DIR}
")
echo ""
echo "  cogneuro_analyses         → job ${JID_ANALYSIS}  (waits for ${ALL_COLLECT})"
echo ""
echo "Watch:  squeue -u \$USER"
echo "Logs:   tail -f ${LOG_DIR}/*_collect_blind_gibson.out"
