#!/bin/bash
#SBATCH --job-name=blind_wij_dbg
#SBATCH --time=00:30:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/izar/%u/cs503_foveated/logs/blind_wij_dbg_%j.out
#SBATCH --error=/scratch/izar/%u/cs503_foveated/logs/blind_wij_dbg_%j.err

set -e

export CS503_SCRATCH=/scratch/izar/${USER}/cs503_foveated
export HABITAT_DATA=/scratch/izar/${USER}/habitat_data

cd /home/${USER}/cs503/foveated-cog-map
source scripts/cluster/common.sh

python -u scripts/probing/collect.py \
  --config-name pointnav/ddppo_pointnav_blind_gibson \
  --ckpt ${CS503_SCRATCH}/checkpoints/spatial-memory-checkpoints/blind/ckpt.34.pth \
  --episodes 2 \
  --deterministic=True \
  --split train_extra_large \
  --override habitat.dataset.data_path=data/datasets/pointnav/gibson/v1/train_extra_large/train_extra_large.json.gz \
  --override habitat.dataset.scenes_dir=data/scene_datasets \
  --out /scratch/izar/${USER}/probing_data/blind_gibson_det_debug.npz
