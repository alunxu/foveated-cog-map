#!/bin/bash
# Submit training-dynamics probes: probe each condition at multiple
# intermediate training checkpoints to see how the GPS code emerges
# during training.
#
# Output: <probing_data>/<cond>_gibson_ckpt<N>_det.npz +
#         <probing_results>/<cond>_gibson_ckpt<N>_det_analysis.json
#
# Run from project root.

set -e

# Per-condition checkpoints.  Pick a sweep covering early/mid/late
# training to capture emergence dynamics.
NAMES="blind matched uniform foveated foveated_learned"

cfg_for() {
    case "$1" in
        blind) echo "pointnav/ddppo_pointnav_blind_gibson" ;;
        matched) echo "pointnav/ddppo_pointnav_matched_gibson" ;;
        uniform) echo "pointnav/ddppo_pointnav_uniform_gibson" ;;
        foveated) echo "pointnav/ddppo_pointnav_foveated_gibson" ;;
        foveated_learned) echo "pointnav/ddppo_pointnav_foveated_learned_gibson" ;;
    esac
}

ckpt_dir_for() {
    case "$1" in
        blind) echo "/scratch/izar/wxu/habitat_checkpoints/blind_gibson" ;;
        matched) echo "/scratch/izar/wxu/habitat_checkpoints/matched_gibson" ;;
        uniform) echo "/scratch/izar/wxu/habitat_checkpoints/uniform_gibson" ;;
        foveated) echo "/scratch/izar/wxu/habitat_checkpoints/foveated_gibson_corrupt_job2836021" ;;
        foveated_learned) echo "/scratch/izar/wxu/habitat_checkpoints/foveated_learned_gibson_buggy_transform" ;;
    esac
}

# Available checkpoint indices per condition (existing on Izar).
# We pick a sparse sweep — every 10th checkpoint where available — to
# capture emergence without 5×5 = 25 jobs.
ckpt_sweep_for() {
    case "$1" in
        blind)            echo "10 20 30 34" ;;            # final = 34
        matched)          echo "10 20 30 40 49" ;;
        uniform)          echo "10 20 30 40 49" ;;
        foveated)         echo "10 20 30 36" ;;            # final = 36 (NaN window)
        foveated_learned) echo "10 20 30 40 49" ;;
    esac
}

submit() {
    local cond="$1" ckpt_idx="$2"
    local cfg=$(cfg_for "$cond")
    local ckpt_dir=$(ckpt_dir_for "$cond")
    local ckpt_path="${ckpt_dir}/ckpt.${ckpt_idx}.pth"
    local out_npz="/scratch/izar/wxu/probing_data/${cond}_gibson_ckpt${ckpt_idx}_det.npz"

    # Skip if already done.
    if ssh izar "test -f ${out_npz}" 2>/dev/null; then
        echo "  skip ${cond} ckpt.${ckpt_idx} (exists)"
        return
    fi

    # Verify the source ckpt exists.
    if ! ssh izar "test -f ${ckpt_path}" 2>/dev/null; then
        echo "  skip ${cond} ckpt.${ckpt_idx} (no ckpt at ${ckpt_path})"
        return
    fi

    echo "  sbatch ${cond} ckpt.${ckpt_idx}"
    # Override qos/time to normal qos with 6h walltime for parallelism
    # (cs-503 default is MaxJobsPU=1; normal allows up to ~10 in flight).
    ssh izar "cd ~/cs503-project && sbatch --qos=normal --time=6:00:00 scripts/cluster/submit_probe_deterministic.sh \
      ${cfg} ${ckpt_path} 500 ${cond}_gibson_ckpt${ckpt_idx}" 2>/dev/null | tail -1
}

echo "=== J: training-dynamics probe sweep ==="
for cond in $NAMES; do
    sweep=$(ckpt_sweep_for "$cond")
    for idx in $sweep; do
        submit "$cond" "$idx"
    done
done

echo ""
echo "Done. Check 'squeue -u wxu' for status."
