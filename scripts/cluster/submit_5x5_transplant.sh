#!/bin/bash
# Submit the 5x5 cross-condition transplant matrix (existing 3 pairs already done).
# Also extended midpoints (200, 400, 800) for the temporal-stability follow-up.
#
# Run from project root.

# Per-condition (config, ckpt) tuples.
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

ckpt_for() {
    case "$1" in
        blind) echo "/scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.34.pth" ;;
        matched) echo "/scratch/izar/wxu/habitat_checkpoints/matched_gibson/ckpt.49.pth" ;;
        uniform) echo "/scratch/izar/wxu/habitat_checkpoints/uniform_gibson/ckpt.49.pth" ;;
        foveated) echo "/scratch/izar/wxu/habitat_checkpoints/foveated_gibson_corrupt_job2836021/ckpt.36.pth" ;;
        foveated_learned) echo "/scratch/izar/wxu/habitat_checkpoints/foveated_learned_gibson_buggy_transform/ckpt.49.pth" ;;
    esac
}

submit() {
    local donor="$1" recip="$2" midpoint="$3"
    if [ "$donor" = "$recip" ]; then return; fi  # skip self-transplants

    local d_cfg=$(cfg_for "$donor")
    local d_ckpt=$(ckpt_for "$donor")
    local r_cfg=$(cfg_for "$recip")
    local r_ckpt=$(ckpt_for "$recip")

    # Output filename convention.
    local fname
    if [ "$midpoint" = "30" ]; then
        fname="${donor}_to_${recip}.json"
    else
        fname="${donor}_to_${recip}_mid${midpoint}.json"
    fi
    if ssh izar "test -f /scratch/izar/wxu/transplant_results/${fname}" 2>/dev/null; then
        echo "  skip ${fname} (exists)"
        return
    fi

    echo "  sbatch ${donor} -> ${recip} mid=${midpoint}"
    ssh izar "cd ~/cs503-project && sbatch scripts/cluster/submit_transplant.sh \
      ${donor} ${d_cfg} ${d_ckpt} \
      ${recip} ${r_cfg} ${r_ckpt} \
      150 ${midpoint}" 2>/dev/null | tail -1
}

echo "=== A: 5x5 cross-condition transplant matrix (midpoint=30) ==="
for d in $NAMES; do
    for r in $NAMES; do
        submit "$d" "$r" 30
    done
done

echo ""
echo "=== H: extended-midpoint sweep ==="
EXTEND_DONORS="foveated foveated_learned foveated foveated"
EXTEND_RECIPS="foveated_learned foveated uniform blind"
for mid in 200 400 800; do
    set -- $EXTEND_DONORS
    donors=("$@")
    set -- $EXTEND_RECIPS
    recips=("$@")
    for i in 0 1 2 3; do
        submit "${donors[$i]}" "${recips[$i]}" "$mid"
    done
done

echo ""
echo "Done submitting. Check 'squeue -u wxu' for job status."
