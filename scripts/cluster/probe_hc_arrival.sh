#!/bin/bash
# probe_hc_arrival.sh — submit probing jobs for hc-trained checkpoints.
#
# When the friend rsyncs a freshly-trained model to
# /scratch/izar/wxu/habitat_checkpoints/<run_name>/ , we want to run
# the standard probing pipeline (deterministic-rollout collect + Ridge
# probe analyze) on the latest checkpoint and on a handful of
# intermediate ones to populate the across-training plot
# (fig3_substitution_dynamics.pdf).
#
# This script:
#   1. For each known hc-target run (defined in HC_RUNS below),
#   2. checks if the checkpoint folder has new checkpoints,
#   3. submits probing jobs for the across-checkpoint set
#      ({10, 20, 30, 40, 49, latest}) that don't yet have an
#      analysis JSON in /scratch/izar/wxu/probing_results/.
#
# Idempotent: skips checkpoints whose analysis JSON already exists or
# whose probing job is already in queue. Logs to probe_hc.log.

set -e

CKPT_BASE="/scratch/izar/wxu/habitat_checkpoints"
RESULTS_DIR="/scratch/izar/wxu/probing_results"
PROJECT_DIR="/home/wxu/cs503-project"
LOG_FILE="$PROJECT_DIR/probe_hc.log"

# Map: run_name → config_name (used in submit_probe_deterministic.sh)
# These match what the hc launch recipe is launching.
declare -A HC_RUNS=(
    # Tier 1 — Stochastic gaze + scaling sweep + multi-seed
    # NOTE: matched128_gibson is NOT in this map — it was already trained
    # on Izar earlier (50 ckpts converged); its probing was launched
    # manually on 2026-04-27 via 5× submit_probe_deterministic.sh.
    ["foveated_stochastic_gibson"]="pointnav/ddppo_pointnav_foveated_stochastic_gibson"
    ["matched64_gibson"]="pointnav/ddppo_pointnav_matched64_gibson"
    ["blind_gibson_seed2"]="pointnav/ddppo_pointnav_blind_gibson"
    ["matched_gibson_seed2"]="pointnav/ddppo_pointnav_matched_gibson"
    # Tier 2
    ["matched32_gibson"]="pointnav/ddppo_pointnav_matched32_gibson"
    ["matched96_gibson"]="pointnav/ddppo_pointnav_matched96_gibson"
    ["matched192_gibson"]="pointnav/ddppo_pointnav_matched192_gibson"
    ["foveated_learned_gibson_seed2"]="pointnav/ddppo_pointnav_foveated_learned_gibson"
    ["foveated_logpolar_gibson"]="pointnav/ddppo_pointnav_foveated_logpolar_gibson"
    # Tier 3
    ["foveated_v2_gibson"]="pointnav/ddppo_pointnav_foveated_v2_gibson"
    ["foveated_strong_gibson"]="pointnav/ddppo_pointnav_foveated_strong_gibson"
    ["foveated_sigma2_gibson"]="pointnav/ddppo_pointnav_foveated_sigma2_gibson"
    ["foveated_sigma12_gibson"]="pointnav/ddppo_pointnav_foveated_sigma12_gibson"
    ["foveated_shifted_gibson"]="pointnav/ddppo_pointnav_foveated_shifted_gibson"
)

# Probe these specific checkpoint indices, plus 'latest', whenever
# they exist. The four early-training points (10/20/30) populate the
# substitution-dynamics figure; ckpt 49 (or similar) is the converged
# model used in the main paper Table 1 / synthesis figure.
PROBE_CKPTS=(10 20 30 40 49)

now=$(date '+%Y-%m-%d %H:%M:%S')
echo "==================================================" >> "$LOG_FILE"
echo "[$now] probe_hc_arrival.sh tick" >> "$LOG_FILE"

# Cache of all queued probing job names so we don't double-submit.
queued_probe_names=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep '^cs503_prb_' || echo "")

submitted=0
for run in "${!HC_RUNS[@]}"; do
    ckpt_dir="$CKPT_BASE/$run"
    config="${HC_RUNS[$run]}"

    # Skip if folder doesn't exist yet (training hasn't started or
    # rsync hasn't happened).
    if [ ! -d "$ckpt_dir" ]; then
        continue
    fi

    # For each named ckpt index, probe if the file exists and analysis
    # doesn't yet.
    for ck in "${PROBE_CKPTS[@]}"; do
        ckpt_path="$ckpt_dir/ckpt.${ck}.pth"
        json_path="$RESULTS_DIR/${run}_ckpt${ck}_det_analysis.json"

        # Skip if checkpoint not present
        [ -f "$ckpt_path" ] || continue

        # Skip if analysis already done
        [ -f "$json_path" ] && continue

        # Skip if probing job already in queue for this (run, ckpt)
        expected_probe_name="cs503_prb_det_${run%_gibson*}_ckpt${ck}"
        if echo "$queued_probe_names" | grep -Fq "$expected_probe_name"; then
            continue
        fi

        # Submit probing job (uses submit_probe_deterministic.sh which
        # already invokes analyze.py on the collected npz at end).
        cd "$PROJECT_DIR"
        out=$(sbatch --qos=normal \
            scripts/cluster/submit_probe_deterministic.sh \
            "$config" "$ckpt_path" 2>&1) || true
        echo "[$now]   $run ckpt${ck}: $out" >> "$LOG_FILE"
        submitted=$((submitted + 1))
    done

    # Also probe the latest if it exists and we haven't done it yet.
    latest_path="$ckpt_dir/latest.pth"
    if [ -f "$latest_path" ]; then
        latest_json="$RESULTS_DIR/${run}_det_analysis.json"
        if [ ! -f "$latest_json" ]; then
            expected_probe_name="cs503_prb_det_${run%_gibson*}"
            if ! echo "$queued_probe_names" | grep -Fq "$expected_probe_name"; then
                cd "$PROJECT_DIR"
                out=$(sbatch --qos=normal \
                    scripts/cluster/submit_probe_deterministic.sh \
                    "$config" "$latest_path" 2>&1) || true
                echo "[$now]   $run latest: $out" >> "$LOG_FILE"
                submitted=$((submitted + 1))
            fi
        fi
    fi
done

echo "[$now] tick complete; probes submitted=$submitted" >> "$LOG_FILE"
echo "submitted=$submitted"
