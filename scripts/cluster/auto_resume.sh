#!/bin/bash
# auto_resume.sh — detect TIMEOUT'd training jobs and resubmit them.
#
# Logic:
#   For each known training-job checkpoint folder:
#     - If the folder has latest.pth (resume target exists)
#     - AND no current/pending job with the matching abbreviated name
#     - AND fewer than 49 ckpts (training not yet converged)
#     → submit resume on normal QOS
#
# habitat-baselines' DDPPOTrainer auto-resumes from latest.pth when
# checkpoint_folder is set; submit_train{_seeded}.sh derives the folder
# name from the config name + seed, so re-running the same sbatch
# resumes the same run.
#
# Idempotent: if the same job is already running/pending, no resubmit.

set -e

# Map: checkpoint-folder-name → "submit-script-args"
# These are all training runs we want to keep alive until convergence.
declare -A RESUME_MAP=(
    # Per 2026-04-27 strategy: all foveation variants + bld/mtc/fov_lrn
    # multi-seed moved to hc cluster (4× H100 + 1× H200, ~2-3 days each).
    # Izar only finishes uni-s2 / fov-s2 — the two multi-seed runs that
    # were already >50% complete on Izar; restarting on hc would waste
    # their progress (140M / 130M frames already trained).
    ["uniform_gibson_seed2"]="seeded pointnav/ddppo_pointnav_uniform_gibson 2"
    ["foveated_gibson_seed2"]="seeded pointnav/ddppo_pointnav_foveated_gibson 2"
)

CKPT_BASE="/scratch/izar/wxu/habitat_checkpoints"
LOG_FILE="/home/wxu/cs503-project/auto_resume.log"
PROJECT_DIR="/home/wxu/cs503-project"
RECENT_SUBMITS_FILE="/home/wxu/cs503-project/.auto_resume_recent.tsv"
CONVERGENCE_CKPT_THRESHOLD=49  # configs default num_checkpoints=50
DEDUP_WINDOW_MIN=120           # treat as already-handled if submitted in last N min

# Get all currently-queued job names ONCE (fast vs squeue per-loop).
# We fetch the FULL job name (post-rename inside the running script) via %j.
queued_names=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || echo "")

abbrev_for_folder() {
    # Apply the same sed chain as submit_train_seeded.sh's ABBREV
    echo "$1" | sed 's/blind/bld/;s/uniform/uni/;s/foveated/fov/;s/matched/mtc/;s/learned/lrn/;s/_seed/_s/'
}

now=$(date '+%Y-%m-%d %H:%M:%S')
echo "==================================================" >> "$LOG_FILE"
echo "[$now] auto_resume.sh tick" >> "$LOG_FILE"

actions_taken=0
for folder in "${!RESUME_MAP[@]}"; do
    ckpt_dir="$CKPT_BASE/$folder"

    # 1. Folder must exist + have latest.pth
    if [ ! -f "$ckpt_dir/latest.pth" ]; then
        continue
    fi

    # 2. Convergence check: skip if num_ckpts >= threshold
    n_ckpts=$(find "$ckpt_dir" -name 'ckpt.*.pth' -type f 2>/dev/null | wc -l)
    if [ "$n_ckpts" -ge "$CONVERGENCE_CKPT_THRESHOLD" ]; then
        echo "[$now]   $folder: $n_ckpts ckpts, converged → skip" >> "$LOG_FILE"
        continue
    fi

    abbrev=$(abbrev_for_folder "$folder")
    expected_name="cs503_tr_${abbrev}"

    # 3a. Skip if a RUNNING job (post-rename) matches the expected name
    if echo "$queued_names" | grep -Fxq "$expected_name"; then
        continue
    fi

    # 3b. Skip if we already submitted a resume for this folder recently
    #     (PENDING jobs still carry the generic "cs503_tr_seed" / "cs503_tr"
    #     name until they start running, so rely on a tracking file).
    if [ -f "$RECENT_SUBMITS_FILE" ]; then
        last_submit_epoch=$(awk -v f="$folder" '
            $1 == f { print $2 }
        ' "$RECENT_SUBMITS_FILE" | tail -1)
        if [ -n "$last_submit_epoch" ]; then
            now_epoch=$(date +%s)
            age_min=$(( (now_epoch - last_submit_epoch) / 60 ))
            if [ "$age_min" -lt "$DEDUP_WINDOW_MIN" ]; then
                echo "[$now]   $folder: submitted ${age_min}min ago (< ${DEDUP_WINDOW_MIN}min window) → skip" >> "$LOG_FILE"
                continue
            fi
        fi
    fi

    # 4. Submit resume — use --job-name to set the queue name immediately
    #    so the rename race (PENDING shows generic name) doesn't trip us.
    args="${RESUME_MAP[$folder]}"
    kind=$(echo "$args" | awk '{print $1}')
    config_arg=$(echo "$args" | awk '{print $2}')
    seed_arg=$(echo "$args" | awk '{print $3}')

    cd "$PROJECT_DIR"
    if [ "$kind" = "seeded" ]; then
        out=$(sbatch --qos=normal --time=71:55:00 \
            --job-name="$expected_name" \
            scripts/cluster/submit_train_seeded.sh \
            "$config_arg" "$seed_arg" 2>&1)
    else
        out=$(sbatch --qos=normal --time=71:55:00 \
            --job-name="$expected_name" \
            scripts/cluster/submit_train.sh \
            "$config_arg" 2>&1)
    fi

    # Record submission for dedup
    echo -e "${folder}\t$(date +%s)\t${out}" >> "$RECENT_SUBMITS_FILE"

    echo "[$now]   $folder: RESUMED ($n_ckpts ckpts saved) — $out" >> "$LOG_FILE"
    actions_taken=$((actions_taken + 1))
done

echo "[$now] tick complete; actions_taken=$actions_taken" >> "$LOG_FILE"
echo "actions_taken=$actions_taken"  # stdout — for cron consumer
