#!/usr/bin/env bash
# Submit the fast Memory-Maze encoder-scale x sensor-constraint pilot to RCP.
#
# Usage:
#   bash scripts/cluster_rcp/submit_wm_scale_sensor_pilot.sh [job-suffix]
#
# Optional env overrides:
#   TRAIN_TRAJ=200 EVAL_TRAJ=50 TRAJ_LEN=501 LSTM_STEPS=3000 PROBE_STEPS=5000 \
#   bash scripts/cluster_rcp/submit_wm_scale_sensor_pilot.sh

set -euo pipefail

JOB_SUFFIX="${1:-$(date +%m%d-%H%M)}"
JOB_NAME="wm-ss-${JOB_SUFFIX}"
PROJECT="dhlab-wxu"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/dhlab-wxu/habitat:v2}"
GPU="${GPU:-1}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-64G}"

TRAIN_TRAJ="${TRAIN_TRAJ:-200}"
EVAL_TRAJ="${EVAL_TRAJ:-50}"
TRAJ_LEN="${TRAJ_LEN:-501}"
LSTM_STEPS="${LSTM_STEPS:-3000}"
PROBE_STEPS="${PROBE_STEPS:-5000}"
LSTM_TRAIN_CACHE_N="${LSTM_TRAIN_CACHE_N:-200}"
ENCODERS="${ENCODERS:-dinov2_vits14 dinov2_vitb14}"
CONDITIONS="${CONDITIONS:-foveated uniform}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAYLOAD="$(
  tar -cz -C "$REPO_ROOT" \
    scripts/probing/world_model_probe/01_generate_data.py \
    scripts/probing/world_model_probe/02_cache_features.py \
    scripts/probing/world_model_probe/03_train_lstm.py \
    scripts/probing/world_model_probe/04_probe.py \
    scripts/probing/world_model_probe/05_run_scale_sensor.sh \
    scripts/probing/world_model_probe/08_aggregate_scale_sensor.py \
  | base64 | tr -d '\n'
)"

OUT_BASE="${OUT_BASE:-/scratch/wxu/dh-spatial/outputs/wmprobe_scale_sensor}"
OUT_ROOT="${OUT_BASE}/${JOB_NAME}"

echo "=================================================="
echo "  RCP Memory-Maze scale x sensor pilot"
echo "  Job:        ${JOB_NAME}"
echo "  Image:      ${IMAGE}"
echo "  Encoders:   ${ENCODERS}"
echo "  Conditions: ${CONDITIONS}"
echo "  Train/eval: ${TRAIN_TRAJ}/${EVAL_TRAJ}, T=${TRAJ_LEN}"
echo "  Steps:      LSTM=${LSTM_STEPS}, probe=${PROBE_STEPS}"
echo "  Out:        ${OUT_ROOT}"
echo "=================================================="

read -r -d '' INNER_CMD <<EOF || true
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:\${PYTHONPATH:-}
mkdir -p ${OUT_ROOT}
printf '%s' '${PAYLOAD}' | base64 -d | tar -xz -C .
echo BOOTSTRAP_OK

python -m pip install --quiet --no-cache-dir \
  'numpy<2' 'scikit-learn>=1.3' 'gym==0.26.2' \
  'mujoco==2.3.7' 'dm-control==1.0.14' memory-maze

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_HOME=${OUT_ROOT}/cache/torch
export XDG_CACHE_HOME=${OUT_ROOT}/cache/xdg
mkdir -p "\${TORCH_HOME}" "\${XDG_CACHE_HOME}"
export ROOT=${OUT_ROOT}
export TRAIN_TRAJ=${TRAIN_TRAJ}
export EVAL_TRAJ=${EVAL_TRAJ}
export TRAJ_LEN=${TRAJ_LEN}
export LSTM_STEPS=${LSTM_STEPS}
export PROBE_STEPS=${PROBE_STEPS}
export LSTM_TRAIN_CACHE_N=${LSTM_TRAIN_CACHE_N}
export ENCODERS="${ENCODERS}"
export CONDITIONS="${CONDITIONS}"

chmod +x scripts/probing/world_model_probe/05_run_scale_sensor.sh
bash scripts/probing/world_model_probe/05_run_scale_sensor.sh 2>&1 | tee -a ${OUT_ROOT}/run.log
EOF

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
  --project "$PROJECT" \
  --image="$IMAGE" \
  --gpu="$GPU" \
  --cpu="$CPU" \
  --memory="$MEMORY" \
  --pvc=dhlab-scratch:/scratch \
  --pvc=home:/home/wxu \
  --large-shm \
  --command -- bash -lc "$INNER_CMD"

echo ""
echo "Submitted. Monitor with:"
echo "  kubectl logs -n runai-${PROJECT} \$(kubectl get pods -n runai-${PROJECT} -l release=${JOB_NAME} -o name | head -1 | cut -d/ -f2) -f"
echo ""
echo "Summary will be:"
echo "  ${OUT_ROOT}/results/summary_scale_sensor.md"
