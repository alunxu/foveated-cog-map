#!/usr/bin/env bash
# Orchestrator for the 5-condition world-model probe sweep.
#
# Stages (each can be skipped with the env vars below):
#   1. Cache DINOv2-Base features for {train, eval} x {5 conditions}
#   2. Train per-condition LSTM (frozen across-conditions config)
#   3. Run linear + MLP probe per-condition
#   4. Aggregate results into one summary JSON
#
# Env vars:
#   PYTHON              python interpreter (default /tmp/wmprobe_venv/bin/python)
#   DATA_TRAIN          dir of train .npz (default /tmp/memory_maze_data/9x9_train)
#   DATA_EVAL           dir of eval .npz (default /tmp/memory_maze_data/9x9_eval)
#   FEAT_TRAIN          out dir for train features (default /tmp/wmprobe_feats/train)
#   FEAT_EVAL           out dir for eval features (default /tmp/wmprobe_feats/eval)
#   LSTM_OUT            out dir for LSTM ckpts (default /tmp/wmprobe_lstm)
#   PROBE_OUT           out dir for probe results (default /tmp/wmprobe_results)
#   LSTM_STEPS          (default 8000)
#   PROBE_STEPS         (default 10000)
#   CACHE_LIMIT         max trajectories per condition (default unset = all)
#   SKIP_CACHE          set to 1 to skip stage 1
#   SKIP_LSTM           set to 1 to skip stage 2
#   SKIP_PROBE          set to 1 to skip stage 3

set -euo pipefail

PYTHON=${PYTHON:-/tmp/wmprobe_venv/bin/python}
DATA_TRAIN=${DATA_TRAIN:-/tmp/memory_maze_data/9x9_train}
DATA_EVAL=${DATA_EVAL:-/tmp/memory_maze_data/9x9_eval}
FEAT_TRAIN=${FEAT_TRAIN:-/tmp/wmprobe_feats/train}
FEAT_EVAL=${FEAT_EVAL:-/tmp/wmprobe_feats/eval}
LSTM_OUT=${LSTM_OUT:-/tmp/wmprobe_lstm}
PROBE_OUT=${PROBE_OUT:-/tmp/wmprobe_results}
LSTM_STEPS=${LSTM_STEPS:-8000}
PROBE_STEPS=${PROBE_STEPS:-10000}

CONDITIONS=(blind coarse foveated uniform foveated_logpolar)

mkdir -p "$FEAT_TRAIN" "$FEAT_EVAL" "$LSTM_OUT" "$PROBE_OUT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${SKIP_CACHE:-0}" != "1" ]]; then
    LIMIT_FLAGS=""
    if [[ -n "${CACHE_LIMIT:-}" ]]; then
        LIMIT_FLAGS="--limit ${CACHE_LIMIT}"
    fi
    echo "================================================================="
    echo "Stage 1: caching DINOv2 features"
    echo "================================================================="
    "$PYTHON" "$SCRIPT_DIR/02_cache_features.py" \
        --data_dir "$DATA_TRAIN" --out_root "$FEAT_TRAIN" $LIMIT_FLAGS
    "$PYTHON" "$SCRIPT_DIR/02_cache_features.py" \
        --data_dir "$DATA_EVAL" --out_root "$FEAT_EVAL" $LIMIT_FLAGS
fi

if [[ "${SKIP_LSTM:-0}" != "1" ]]; then
    echo "================================================================="
    echo "Stage 2: training per-condition LSTM"
    echo "================================================================="
    for COND in "${CONDITIONS[@]}"; do
        echo "--- condition: $COND ---"
        "$PYTHON" "$SCRIPT_DIR/03_train_lstm.py" \
            --feat_root "$FEAT_TRAIN" \
            --eval_feat_root "$FEAT_EVAL" \
            --raw_train "$DATA_TRAIN" \
            --raw_eval "$DATA_EVAL" \
            --condition "$COND" \
            --out_dir "$LSTM_OUT/$COND" \
            --steps "$LSTM_STEPS"
    done
fi

if [[ "${SKIP_PROBE:-0}" != "1" ]]; then
    echo "================================================================="
    echo "Stage 3: probes"
    echo "================================================================="
    for COND in "${CONDITIONS[@]}"; do
        echo "--- condition: $COND ---"
        "$PYTHON" "$SCRIPT_DIR/04_probe.py" \
            --lstm_dir "$LSTM_OUT/$COND" \
            --condition "$COND" \
            --feat_root "$FEAT_TRAIN" \
            --eval_feat_root "$FEAT_EVAL" \
            --raw_train "$DATA_TRAIN" \
            --raw_eval "$DATA_EVAL" \
            --out_path "$PROBE_OUT/$COND.json" \
            --probe_steps "$PROBE_STEPS"
    done
fi

echo "================================================================="
echo "Stage 4: aggregate"
echo "================================================================="
"$PYTHON" "$SCRIPT_DIR/05_aggregate.py" \
    --probe_dir "$PROBE_OUT" \
    --out_summary "$PROBE_OUT/summary.json" \
    --out_md "$PROBE_OUT/summary.md"

echo "Sweep complete. See $PROBE_OUT/summary.md"
