#!/usr/bin/env bash
# Inner script run on RCP after files have been copied to /scratch/wxu/dh-spatial.

set -euo pipefail

JOB_NAME="${1:-wm-ss-manual}"
OUT_BASE="${OUT_BASE:-/scratch/wxu/dh-spatial/outputs/wmprobe_scale_sensor}"
OUT_ROOT="${OUT_BASE}/${JOB_NAME}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:${PYTHONPATH:-}

mkdir -p "$OUT_ROOT"
python -m pip install --quiet --no-cache-dir \
  'numpy<2' 'scikit-learn>=1.3' 'gym==0.26.2' \
  'mujoco==2.3.7' 'dm-control==1.0.14' memory-maze

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_HOME="$OUT_ROOT/cache/torch"
export XDG_CACHE_HOME="$OUT_ROOT/cache/xdg"
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"
export ROOT="$OUT_ROOT"

export TRAIN_TRAJ="${TRAIN_TRAJ:-200}"
export EVAL_TRAJ="${EVAL_TRAJ:-50}"
export TRAJ_LEN="${TRAJ_LEN:-501}"
export LSTM_STEPS="${LSTM_STEPS:-3000}"
export PROBE_STEPS="${PROBE_STEPS:-5000}"
export LSTM_TRAIN_CACHE_N="${LSTM_TRAIN_CACHE_N:-200}"
export ENCODERS="${ENCODERS:-dinov2_vits14 dinov2_vitb14}"
export CONDITIONS="${CONDITIONS:-foveated uniform}"

chmod +x scripts/probing/world_model_probe/05_run_scale_sensor.sh
bash scripts/probing/world_model_probe/05_run_scale_sensor.sh 2>&1 | tee -a "$OUT_ROOT/run.log"
