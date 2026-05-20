#!/bin/bash
#SBATCH --job-name=cs503_topdown
#SBATCH --time=00:30:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Render Habitat top-down occupancy map for one Gibson val episode.
# Used for the Fig 1 setup figure (trajectory_overlay on floor plan).
#
# Usage:
#   sbatch submit_topdown_render.sh <config_name> <episode_id> <out_png> <out_json>

CONFIG_NAME=${1:-"pointnav/ddppo_pointnav_blind_gibson"}
EPISODE_ID=${2:-414}
OUT_PNG=${3:-"/scratch/izar/$USER/topdown/scene_ep${EPISODE_ID}.png"}
OUT_JSON=${4:-"/scratch/izar/$USER/topdown/scene_ep${EPISODE_ID}.json"}

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

mkdir -p "$(dirname "$OUT_PNG")"

cd /home/${USER}/habitat-lab

echo "Render top-down map"
echo "  config:    $CONFIG_NAME"
echo "  ep id:     $EPISODE_ID"
echo "  out png:   $OUT_PNG"
echo "  out json:  $OUT_JSON"

python -u "${PROJECT_DIR}/scripts/utils/render_scene_topdown.py" \
    --config-name="${CONFIG_NAME}" \
    --episode-id=${EPISODE_ID} \
    --out-png="${OUT_PNG}" \
    --out-json="${OUT_JSON}" \
    --map-resolution=1024

echo "Completed at $(date)"
