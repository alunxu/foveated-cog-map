#!/bin/bash
#SBATCH --job-name=collect_gibson
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/scratch/izar/%u/cs503_foveated/logs/collect_%x_%j.out
#SBATCH --error=/scratch/izar/%u/cs503_foveated/logs/collect_%x_%j.err

set -e

COND=${1:?condition required: blind|matched|uniform|foveated}
EPISODES=${2:-500}

export CS503_SCRATCH=/scratch/izar/${USER}/cs503_foveated
export HABITAT_DATA=/scratch/izar/${USER}/habitat_data

cd /home/${USER}/cs503/foveated-cog-map
source scripts/cluster/common.sh

case "$COND" in
  blind)
    CONFIG="pointnav/ddppo_pointnav_blind_gibson"
    CKPT="${CS503_SCRATCH}/checkpoints/spatial-memory-checkpoints/blind/ckpt.34.pth"
    OUT="/scratch/izar/${USER}/probing_data/blind_gibson_det.npz"
    ;;
  matched|coarse)
    CONFIG="pointnav/ddppo_pointnav_matched_gibson"
    CKPT="${CS503_SCRATCH}/checkpoints/spatial-memory-checkpoints/coarse/ckpt.49.pth"
    OUT="/scratch/izar/${USER}/probing_data/matched_gibson_det.npz"
    ;;
  uniform)
    CONFIG="pointnav/ddppo_pointnav_uniform_gibson"
    CKPT="${CS503_SCRATCH}/checkpoints/spatial-memory-checkpoints/uniform/ckpt.49.pth"
    OUT="/scratch/izar/${USER}/probing_data/uniform_gibson_det.npz"
    ;;
  foveated)
    CONFIG="pointnav/ddppo_pointnav_foveated_gibson"
    CKPT="${CS503_SCRATCH}/checkpoints/spatial-memory-checkpoints/foveated/ckpt.49.pth"
    OUT="/scratch/izar/${USER}/probing_data/foveated_gibson_det.npz"
    ;;
  *)
    echo "Unknown condition: $COND"
    exit 1
    ;;
esac

echo "Condition: $COND"
echo "Config:    $CONFIG"
echo "Ckpt:      $CKPT"
echo "Episodes:  $EPISODES"
echo "Out:       $OUT"

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python -u scripts/probing/collect.py \
  --config-name "$CONFIG" \
  --ckpt "$CKPT" \
  --episodes "$EPISODES" \
  --deterministic=True \
  --split train_extra_large \
  --override habitat.dataset.data_path=data/datasets/pointnav/gibson/v1/train_extra_large/train_extra_large.json.gz \
  --override habitat.dataset.scenes_dir=data/scene_datasets \
  --out "$OUT"
