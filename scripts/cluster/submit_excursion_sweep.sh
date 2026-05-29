#!/bin/bash
# WJ-F Excursion Forgetting sweep across the 4 main paper conditions.
#
# For each condition, run scripts/eval/excursion_forgetting.py with
# warmup=50 / detour=25 / recovery=100 and 100 target valid episodes.
# Output goes to /scratch/izar/$USER/excursion_results/<cond>_det.npz.
#
# Usage:
#   bash scripts/cluster/submit_excursion_sweep.sh

set -e

NAMES="blind matched uniform foveated"
EPISODES=100

cfg_for() {
    case "$1" in
        blind)    echo "pointnav/ddppo_pointnav_blind_gibson" ;;
        matched)  echo "pointnav/ddppo_pointnav_matched_gibson" ;;
        uniform)  echo "pointnav/ddppo_pointnav_uniform_gibson" ;;
        foveated) echo "pointnav/ddppo_pointnav_foveated_gibson" ;;
    esac
}

ckpt_for() {
    case "$1" in
        blind)    echo "/scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.34.pth" ;;
        matched)  echo "/scratch/izar/wxu/habitat_checkpoints/matched_gibson/ckpt.49.pth" ;;
        uniform)  echo "/scratch/izar/wxu/habitat_checkpoints/uniform_gibson/ckpt.49.pth" ;;
        foveated) echo "/scratch/izar/wxu/habitat_checkpoints/foveated_gibson_corrupt_job2836021/ckpt.36.pth" ;;
    esac
}

OUT_DIR="/scratch/izar/${USER}/excursion_results"
LOG_DIR="/home/${USER}/cs503-project/slurm_logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

for name in $NAMES; do
    cfg=$(cfg_for "$name")
    ckpt=$(ckpt_for "$name")
    out_path="${OUT_DIR}/${name}_det.npz"
    if [ -f "$out_path" ]; then
        echo "  skip ${name} (exists)"
        continue
    fi
    echo "  submit ${name}"
    sbatch \
      --job-name="cs503_wjf_${name}" \
      --account=cs-503 --qos=normal \
      --time=02:00:00 \
      --gres=gpu:1 --cpus-per-task=4 --mem=24G \
      --output="${LOG_DIR}/%j.out" --error="${LOG_DIR}/%j.err" \
      --chdir=/home/${USER}/habitat-lab \
      --wrap="source /home/${USER}/cs503-project/scripts/cluster/common.sh && python -u /home/${USER}/cs503-project/scripts/eval/excursion_forgetting.py --config ${cfg} --ckpt ${ckpt} --episodes ${EPISODES} --out ${out_path}"
done

echo ""
echo "Done. Check 'squeue -u ${USER}' for status."
echo "Once NPZs land, write a small analyzer to probe GPS R^2 per segment."
