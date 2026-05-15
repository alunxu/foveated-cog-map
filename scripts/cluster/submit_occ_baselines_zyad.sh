#!/bin/bash
#SBATCH --job-name=occ_baselines
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/izar/%u/cs503_foveated/logs/occ_baselines_%j.out
#SBATCH --error=/scratch/izar/%u/cs503_foveated/logs/occ_baselines_%j.err

set -e

cd /home/${USER}/cs503/foveated-cog-map
source scripts/cluster/common.sh

python -u scripts/probing/eval_occupancy_baselines.py \
  --scene-occ-dir /scratch/izar/${USER}/scene_occupancy \
  --out-dir /scratch/izar/${USER}/occupancy_decoder_results \
  --grid-size 32 \
  --grid-res 0.5 \
  --dilate-m 2.5
