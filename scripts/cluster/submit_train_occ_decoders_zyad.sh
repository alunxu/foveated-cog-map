#!/bin/bash
#SBATCH --job-name=occ_decoders
#SBATCH --time=08:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/scratch/izar/%u/cs503_foveated/logs/occ_decoders_%j.out
#SBATCH --error=/scratch/izar/%u/cs503_foveated/logs/occ_decoders_%j.err

set -e

export CS503_SCRATCH=/scratch/izar/${USER}/cs503_foveated
export HABITAT_DATA=/scratch/izar/${USER}/habitat_data

cd /home/${USER}/cs503/foveated-cog-map
source scripts/cluster/common.sh

OUT_DIR=/scratch/izar/${USER}/occupancy_decoder_results
SCENE_OCC_DIR=/scratch/izar/${USER}/scene_occupancy
mkdir -p "$OUT_DIR"

run_decoder () {
  COND=$1
  NPZ=$2

  echo "============================================"
  echo "Training occupancy decoder: $COND"
  echo "Hidden NPZ: $NPZ"
  echo "============================================"

  python -u scripts/probing/train_occupancy_decoder.py \
    --hidden-npz "$NPZ" \
    --scene-occ-dir "$SCENE_OCC_DIR" \
    --cond-name "$COND" \
    --out-dir "$OUT_DIR" \
    --grid-size 32 \
    --grid-res 0.5 \
    --epochs 80 \
    --batch 64
}

run_decoder blind      /scratch/izar/${USER}/probing_data/blind_gibson_det.npz
run_decoder matched128 /scratch/izar/${USER}/probing_data/matched_gibson_det.npz
run_decoder uniform    /scratch/izar/${USER}/probing_data/uniform_gibson_det.npz
run_decoder foveated   /scratch/izar/${USER}/probing_data/foveated_gibson_det.npz

echo "All occupancy decoders done."
