#!/bin/bash
mkdir -p /scratch/wxu/dh-spatial/logs/smoke /scratch/wxu/dh-spatial/hydra_runs /scratch/wxu/dh-spatial/checkpoints/smoke_blind/tb
LOGFILE=/scratch/wxu/dh-spatial/logs/smoke/full_$(date +%s).log
exec > $LOGFILE 2>&1
echo "=== start $(date) ==="
export PATH=/scratch/wxu/miniconda3/bin:$PATH
source /scratch/wxu/miniconda3/etc/profile.d/conda.sh
conda activate habitat-izar
nvidia-smi --query-gpu=name --format=csv,noheader
export EGL_DEVICE_ID=0
export EGL_PLATFORM=device
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MAGNUM_LOG=quiet
cd /scratch/wxu/dh-spatial/hydra_runs   
echo "--- Train ---"
PYTHONPATH=/scratch/wxu/dh-spatial python -u /scratch/wxu/dh-spatial/scripts/cluster/run_habitat.py --config-name=pointnav/ddppo_pointnav_blind_gibson hydra.job.chdir=False hydra.run.dir=/scratch/wxu/dh-spatial/hydra_runs/\${now:%Y-%m-%d}/\${now:%H-%M-%S} hydra.output_subdir=null habitat_baselines.evaluate=False habitat_baselines.total_num_steps=5000000 habitat_baselines.checkpoint_folder=/scratch/wxu/dh-spatial/checkpoints/smoke_blind habitat_baselines.tensorboard_dir=/scratch/wxu/dh-spatial/checkpoints/smoke_blind/tb habitat.dataset.scenes_dir=/scratch/wxu/dh-spatial/data/scene_datasets habitat_baselines.num_environments=1
echo "=== end $(date) ==="
