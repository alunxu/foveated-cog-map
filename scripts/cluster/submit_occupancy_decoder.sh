#!/bin/bash
# WJ-C stage 2-3: train per-condition occupancy decoders.
#
# Prerequisite: scene_occupancy/ pre-computed by compute_scene_occupancy.py.
#
# Usage (on izar):
#   bash scripts/cluster/submit_occupancy_decoder.sh

set -e

NAMES="blind matched128 uniform foveated"

OUT_DIR="/scratch/izar/${USER}/occupancy_decoder_results"
SCENE_OCC_DIR="/scratch/izar/${USER}/scene_occupancy"
LOG_DIR="/home/${USER}/cs503-project/slurm_logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

hidden_for() {
    case "$1" in
        blind)      echo "/scratch/izar/wxu/probing_data/blind_gibson_det.npz" ;;
        matched128) echo "/scratch/izar/wxu/probing_data/matched_gibson_det.npz" ;;
        uniform)    echo "/scratch/izar/wxu/probing_data/uniform_gibson_det.npz" ;;
        foveated)   echo "/scratch/izar/wxu/probing_data/foveated_gibson_det.npz" ;;
    esac
}

for name in $NAMES; do
    out_json="${OUT_DIR}/${name}_occupancy.json"
    if [ -f "$out_json" ]; then
        echo "  skip ${name} (exists)"
        continue
    fi
    hidden_npz=$(hidden_for "$name")
    echo "  submit decoder ${name} (using ${hidden_npz})"
    sbatch \
      --job-name="cs503_occdec_${name}" \
      --account=cs-503 --qos=normal \
      --time=01:30:00 \
      --gres=gpu:1 --cpus-per-task=4 --mem=24G \
      --output="${LOG_DIR}/%j.out" --error="${LOG_DIR}/%j.err" \
      --chdir=/home/${USER}/cs503-project \
      --wrap="source /home/${USER}/cs503-project/scripts/cluster/common.sh && python -u /home/${USER}/cs503-project/scripts/probing/train_occupancy_decoder.py --hidden-npz=${hidden_npz} --scene-occ-dir=${SCENE_OCC_DIR} --cond-name=${name} --out-dir=${OUT_DIR} --grid-size=32 --grid-res=0.5 --epochs=80"
done

echo "Done. Once decoders finish, render via make_occupancy_decoder_figure.py."
