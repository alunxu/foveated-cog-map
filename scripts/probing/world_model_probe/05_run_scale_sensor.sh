#!/usr/bin/env bash
# Fast 2x2 pilot for the broader-implication claim:
#   encoder scale x sensor constraint
#
# Default crossing:
#   DINOv2-S / DINOv2-B  x  foveated / uniform
#
# The decisive comparison is:
#   small constrained (dinov2_vits14 + foveated)
#   vs.
#   larger unconstrained (dinov2_vitb14 + uniform)
#
# This is a pilot, not the final Habitat experiment. It is designed to quickly
# test whether task-aligned perceptual constraints can match or beat simply
# scaling the frozen encoder on a spatial-memory readout.

set -euo pipefail

PYTHON=${PYTHON:-python}
ROOT=${ROOT:-/tmp/wmprobe_scale_sensor}
DATA_TRAIN=${DATA_TRAIN:-${ROOT}/data/9x9_train}
DATA_EVAL=${DATA_EVAL:-${ROOT}/data/9x9_eval}
FEAT_ROOT=${FEAT_ROOT:-${ROOT}/features}
LSTM_ROOT=${LSTM_ROOT:-${ROOT}/lstm}
PROBE_ROOT=${PROBE_ROOT:-${ROOT}/probes}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results}

TRAIN_TRAJ=${TRAIN_TRAJ:-200}
EVAL_TRAJ=${EVAL_TRAJ:-50}
TRAJ_LEN=${TRAJ_LEN:-501}
TRAIN_SEED_OFFSET=${TRAIN_SEED_OFFSET:-0}
EVAL_SEED_OFFSET=${EVAL_SEED_OFFSET:-10000}

ENCODERS=${ENCODERS:-"dinov2_vits14 dinov2_vitb14"}
CONDITIONS=${CONDITIONS:-"foveated uniform"}
TARGET_RES=${TARGET_RES:-56}
BLUR_SIGMA=${BLUR_SIGMA:-4.0}
CACHE_BATCH=${CACHE_BATCH:-64}
CACHE_LIMIT=${CACHE_LIMIT:-}

LSTM_STEPS=${LSTM_STEPS:-3000}
LSTM_BATCH=${LSTM_BATCH:-16}
LSTM_SEQ_LEN=${LSTM_SEQ_LEN:-100}
LSTM_TRAIN_CACHE_N=${LSTM_TRAIN_CACHE_N:-200}
PROBE_STEPS=${PROBE_STEPS:-5000}
SKIP_DATA=${SKIP_DATA:-0}
SKIP_CACHE=${SKIP_CACHE:-0}
SKIP_LSTM=${SKIP_LSTM:-0}
SKIP_PROBE=${SKIP_PROBE:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DATA_TRAIN" "$DATA_EVAL" "$FEAT_ROOT" "$LSTM_ROOT" "$PROBE_ROOT" "$RESULT_ROOT"

read -r -a ENCODER_ARR <<< "$ENCODERS"
read -r -a CONDITION_ARR <<< "$CONDITIONS"

echo "================================================================="
echo "Scale x sensor pilot"
echo "================================================================="
echo "ROOT        = $ROOT"
echo "PYTHON      = $PYTHON"
echo "ENCODERS    = ${ENCODER_ARR[*]}"
echo "CONDITIONS  = ${CONDITION_ARR[*]}"
echo "TRAIN/EVAL  = ${TRAIN_TRAJ}/${EVAL_TRAJ} trajectories, T=${TRAJ_LEN}"
echo "LSTM/PROBE  = ${LSTM_STEPS}/${PROBE_STEPS} steps"

count_npz() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo 0
        return
    fi
    find "$dir" -maxdepth 1 -type f -name 'traj_*.npz' | wc -l | tr -d '[:space:]'
}

has_expected_npz() {
    local dir="$1"
    local expected="$2"
    local found
    found="$(count_npz "$dir")"
    [[ "$found" -ge "$expected" ]]
}

if [[ "$SKIP_DATA" != "1" ]]; then
    echo "================================================================="
    echo "Stage 0: Memory-Maze trajectories"
    echo "================================================================="
    if has_expected_npz "$DATA_TRAIN" "$TRAIN_TRAJ"; then
        echo "train data complete: $DATA_TRAIN ($(count_npz "$DATA_TRAIN")/$TRAIN_TRAJ)"
    else
        echo "train data incomplete: $DATA_TRAIN ($(count_npz "$DATA_TRAIN")/$TRAIN_TRAJ)"
        "$PYTHON" "$SCRIPT_DIR/01_generate_data.py" \
            --out_dir "$DATA_TRAIN" \
            --num_traj "$TRAIN_TRAJ" \
            --T "$TRAJ_LEN" \
            --seed_offset "$TRAIN_SEED_OFFSET"
    fi
    if has_expected_npz "$DATA_EVAL" "$EVAL_TRAJ"; then
        echo "eval data complete: $DATA_EVAL ($(count_npz "$DATA_EVAL")/$EVAL_TRAJ)"
    else
        echo "eval data incomplete: $DATA_EVAL ($(count_npz "$DATA_EVAL")/$EVAL_TRAJ)"
        "$PYTHON" "$SCRIPT_DIR/01_generate_data.py" \
            --out_dir "$DATA_EVAL" \
            --num_traj "$EVAL_TRAJ" \
            --T "$TRAJ_LEN" \
            --seed_offset "$EVAL_SEED_OFFSET"
    fi
fi

if [[ "$SKIP_CACHE" != "1" ]]; then
    echo "================================================================="
    echo "Stage 1: cache frozen encoder features"
    echo "================================================================="
    LIMIT_FLAGS=()
    if [[ -n "$CACHE_LIMIT" ]]; then
        LIMIT_FLAGS=(--limit "$CACHE_LIMIT")
    fi
    for ENC in "${ENCODER_ARR[@]}"; do
        echo "--- encoder: $ENC / train ---"
        "$PYTHON" "$SCRIPT_DIR/02_cache_features.py" \
            --data_dir "$DATA_TRAIN" \
            --out_root "$FEAT_ROOT/$ENC/train" \
            --conditions "${CONDITION_ARR[@]}" \
            --target_res "$TARGET_RES" \
            --blur_sigma "$BLUR_SIGMA" \
            --batch_size "$CACHE_BATCH" \
            --encoder "$ENC" \
            "${LIMIT_FLAGS[@]}"
        echo "--- encoder: $ENC / eval ---"
        "$PYTHON" "$SCRIPT_DIR/02_cache_features.py" \
            --data_dir "$DATA_EVAL" \
            --out_root "$FEAT_ROOT/$ENC/eval" \
            --conditions "${CONDITION_ARR[@]}" \
            --target_res "$TARGET_RES" \
            --blur_sigma "$BLUR_SIGMA" \
            --batch_size "$CACHE_BATCH" \
            --encoder "$ENC" \
            "${LIMIT_FLAGS[@]}"
    done
fi

if [[ "$SKIP_LSTM" != "1" ]]; then
    echo "================================================================="
    echo "Stage 2: train recurrent memory per cell"
    echo "================================================================="
    for ENC in "${ENCODER_ARR[@]}"; do
        for COND in "${CONDITION_ARR[@]}"; do
            echo "--- encoder: $ENC / condition: $COND ---"
            "$PYTHON" "$SCRIPT_DIR/03_train_lstm.py" \
                --feat_root "$FEAT_ROOT/$ENC/train" \
                --eval_feat_root "$FEAT_ROOT/$ENC/eval" \
                --raw_train "$DATA_TRAIN" \
                --raw_eval "$DATA_EVAL" \
                --condition "$COND" \
                --out_dir "$LSTM_ROOT/$ENC/$COND" \
                --steps "$LSTM_STEPS" \
                --batch_size "$LSTM_BATCH" \
                --seq_len "$LSTM_SEQ_LEN" \
                --train_cache_n "$LSTM_TRAIN_CACHE_N"
        done
    done
fi

if [[ "$SKIP_PROBE" != "1" ]]; then
    echo "================================================================="
    echo "Stage 3: probe recurrent memory"
    echo "================================================================="
    for ENC in "${ENCODER_ARR[@]}"; do
        mkdir -p "$PROBE_ROOT/$ENC"
        for COND in "${CONDITION_ARR[@]}"; do
            echo "--- encoder: $ENC / condition: $COND ---"
            "$PYTHON" "$SCRIPT_DIR/04_probe.py" \
                --lstm_dir "$LSTM_ROOT/$ENC/$COND" \
                --condition "$COND" \
                --feat_root "$FEAT_ROOT/$ENC/train" \
                --eval_feat_root "$FEAT_ROOT/$ENC/eval" \
                --raw_train "$DATA_TRAIN" \
                --raw_eval "$DATA_EVAL" \
                --out_path "$PROBE_ROOT/$ENC/$COND.json" \
                --probe_steps "$PROBE_STEPS"
        done
    done
fi

echo "================================================================="
echo "Stage 4: aggregate"
echo "================================================================="
"$PYTHON" "$SCRIPT_DIR/08_aggregate_scale_sensor.py" \
    --probe_root "$PROBE_ROOT" \
    --out_json "$RESULT_ROOT/summary_scale_sensor.json" \
    --out_md "$RESULT_ROOT/summary_scale_sensor.md"

echo "Done. Summary:"
echo "  $RESULT_ROOT/summary_scale_sensor.md"
