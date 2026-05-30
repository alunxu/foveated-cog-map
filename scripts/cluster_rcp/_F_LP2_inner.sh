#!/bin/bash
# Inner script for F-LP2 (log-polar foveated + RunningMeanAndVar normaliser ENABLED).
# Companion to F2 for the log-polar condition. Mirrors dh-probe-N hyperparams.
set -ex

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:$PYTHONPATH
export USER=wxu
export HABITAT_DATA_DIR=/scratch/izar/wxu/habitat_data

HB_CONFIG=/opt/habitat-lab/habitat-baselines/habitat_baselines/config
mkdir -p "$HB_CONFIG/pointnav"
for cfg in /scratch/wxu/dh-spatial/habitat_configs/*.yaml; do
    ln -sf "$cfg" "$HB_CONFIG/pointnav/$(basename "$cfg")"
done

OUT_DIR=/scratch/wxu/habitat_checkpoints_rcp/dh-flp2
mkdir -p "$OUT_DIR"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== CONFIGS_AT_LAUNCH ==="
ls -la "$HB_CONFIG/pointnav/ddppo_pointnav_foveated_logpolar_normaliser_gibson.yaml"
head -3 "$HB_CONFIG/pointnav/ddppo_pointnav_foveated_logpolar_normaliser_gibson.yaml"

echo "=== PYTHON_CHECK ==="
python -c "import torch, habitat; print('torch=' + torch.__version__, 'habitat=' + habitat.__version__)"

echo "=== STARTING_TORCHRUN_AT $(date -u +%FT%TZ) ==="
exec torchrun --standalone --nproc_per_node=4 \
    /scratch/wxu/dh-spatial/run_with_custom.py \
    --config-name pointnav/ddppo_pointnav_foveated_logpolar_normaliser_gibson \
    habitat.seed=0 \
    habitat_baselines.checkpoint_folder="$OUT_DIR" \
    habitat_baselines.tensorboard_dir="$OUT_DIR/tb" \
    habitat_baselines.total_num_steps=250000000 \
    habitat_baselines.num_environments=16 \
    habitat_baselines.rl.ppo.num_steps=256 \
    habitat_baselines.rl.ppo.use_linear_lr_decay=False \
    hydra.run.dir="$OUT_DIR/hydra_runs" \
    hydra.output_subdir=null
