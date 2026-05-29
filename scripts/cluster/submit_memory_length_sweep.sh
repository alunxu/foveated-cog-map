#!/bin/bash
# WJ-A memory-length-budget sweep (Wijmans Fig 2 replication).
#
# For each of 4 conditions, run probing-data collection with the LSTM
# hidden state RESET every K steps for K ∈ {1, 4, 16, 64, 256, 1000}.
# Probe GPS R^2 from the resulting hidden states (existing analyze.py
# pipeline).  Plotted as R^2 vs K per condition.
#
# Usage (on izar):
#   bash scripts/cluster/submit_memory_length_sweep.sh

set -e

NAMES="blind matched128 uniform foveated"
KS=(1 4 16 64 256 1000)
EPISODES=100   # smaller than main 500 to keep total compute manageable

cfg_for() {
    case "$1" in
        blind)      echo "pointnav/ddppo_pointnav_blind_gibson" ;;
        matched128) echo "pointnav/ddppo_pointnav_matched128_gibson" ;;
        uniform)    echo "pointnav/ddppo_pointnav_uniform_gibson" ;;
        foveated)   echo "pointnav/ddppo_pointnav_foveated_gibson" ;;
    esac
}

ckpt_for() {
    case "$1" in
        blind)      echo "/scratch/izar/wxu/habitat_checkpoints/blind_gibson/ckpt.34.pth" ;;
        matched128) echo "/scratch/izar/wxu/habitat_checkpoints/matched128_gibson/ckpt.49.pth" ;;
        uniform)    echo "/scratch/izar/wxu/habitat_checkpoints/uniform_gibson/ckpt.49.pth" ;;
        foveated)   echo "/scratch/izar/wxu/habitat_checkpoints/foveated_gibson_corrupt_job2836021/ckpt.36.pth" ;;
    esac
}

OUT_DIR="/scratch/izar/${USER}/probing_data_memlen"
LOG_DIR="/home/${USER}/cs503-project/slurm_logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

for name in $NAMES; do
    cfg=$(cfg_for "$name")
    ckpt=$(ckpt_for "$name")
    for k in "${KS[@]}"; do
        out_path="${OUT_DIR}/${name}_k${k}_det.npz"
        if [ -f "$out_path" ]; then
            echo "  skip ${name} k=${k} (exists)"
            continue
        fi
        echo "  submit ${name} k=${k}"
        sbatch \
          --job-name="cs503_memlen_${name}_k${k}" \
          --account=cs-503 --qos=normal \
          --time=01:30:00 \
          --gres=gpu:1 --cpus-per-task=4 --mem=24G \
          --output="${LOG_DIR}/%j.out" --error="${LOG_DIR}/%j.err" \
          --chdir=/home/${USER}/cs503-project \
          --wrap="source /home/${USER}/cs503-project/scripts/cluster/common.sh && cd /home/${USER}/habitat-lab && python -u /home/${USER}/cs503-project/scripts/probing/collect.py --config-name=${cfg} --ckpt=${ckpt} --episodes=${EPISODES} --out=${out_path} --reset-every=${k} --deterministic=True"
    done
done

echo ""
echo "Done. Check 'squeue -u ${USER}' for status. Once collected,"
echo "run analyze.py on each NPZ to get probed GPS R^2."
