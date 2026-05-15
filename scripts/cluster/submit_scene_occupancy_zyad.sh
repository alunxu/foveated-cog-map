#!/bin/bash
#SBATCH --job-name=scene_occ
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/izar/%u/cs503_foveated/logs/scene_occ_%j.out
#SBATCH --error=/scratch/izar/%u/cs503_foveated/logs/scene_occ_%j.err

set -e

export CS503_SCRATCH=/scratch/izar/${USER}/cs503_foveated
export HABITAT_DATA=/scratch/izar/${USER}/habitat_data

cd /home/${USER}/cs503/foveated-cog-map
source scripts/cluster/common.sh

python -u scripts/probing/compute_scene_occupancy.py \
  --config-name pointnav/ddppo_pointnav_matched_gibson \
  --scene-ids-file /scratch/izar/${USER}/scene_occupancy_scenes.txt \
  --out-dir /scratch/izar/${USER}/scene_occupancy \
  --grid-res 0.5 \
  --split train_extra_large
