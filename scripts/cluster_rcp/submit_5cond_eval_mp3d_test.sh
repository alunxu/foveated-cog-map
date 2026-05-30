#!/bin/bash
# Wijmans-faithful eval: 5 conditions × 1008 MP3D PointNav test episodes
# (18 held-out scenes × 56 episodes/scene), matching Wijmans 2023 Appx A.1.
#
# Why this script exists:
#   Earlier eval (submit_5cond_eval.sh) sampled 500 random episodes from the
#   training scene pool (mp3d_gibson/v1/train, where val is symlinked to train).
#   That setup violates the "Wijmans-faithful" claim in our configs: Wijmans
#   2023 evaluates on the 18 held-out MP3D test scenes (1008 standard episodes
#   from Savva et al. 2019). This script switches to that canonical protocol.
#
# Dataset:
#   /scratch/wxu/dh-spatial/data/datasets/pointnav/mp3d/v1/test/test.json.gz
#   - 1008 episodes, 18 unique MP3D scenes, 56 episodes/scene
#   - Geodesic/Euclidean ratio >= 1.1 (filter built into the dataset)
#   - None of these 18 scenes are in the training set (411 Gibson + 61 MP3D-train)
#
# Hyperparams:
#   - All 5 conditions use the SAME checkpoint as submit_5cond_eval.sh:
#       blind: dh-blind/ckpt.49 (May 12 unified-hp RCP retrain)
#       coarse: dh-probe-1/ckpt.49
#       foveated: dh-probe-2/ckpt.49
#       uniform: dh-probe-3/ckpt.49
#       foveated_logpolar: dh-probe-4/ckpt.49
#   - Evaluates ALL 1008 episodes (no rng sub-sample; --no-sample)
#   - Deterministic policy (argmax)
#
# Output:
#   /scratch/wxu/habitat_checkpoints_rcp/eval_5cond_mp3d_test/<cond>.json
#
# ETA: ~2-4h per condition single-env, parallel wall-clock ~4h.
set -e

IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/eval_5cond_mp3d_test"

declare -a CONDS=(
    "blind|pointnav/ddppo_pointnav_blind_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-blind/ckpt.49.pth"
    "coarse|pointnav/ddppo_pointnav_coarse_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth"
    "foveated|pointnav/ddppo_pointnav_foveated_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2/ckpt.49.pth"
    "uniform|pointnav/ddppo_pointnav_uniform_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3/ckpt.49.pth"
    "foveated_logpolar|pointnav/ddppo_pointnav_foveated_logpolar_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4/ckpt.49.pth"
)

for entry in "${CONDS[@]}"; do
    IFS='|' read -r COND CFG CKPT <<< "$entry"
    JOB_NAME="eval5m-${COND//_/-}"

    # PVC-resident inner script avoids runai-cli inline-quoting bug.
    INNER_CMD="bash /scratch/wxu/dh-spatial/scripts/cluster/_eval5_mp3d_test_inner.sh ${COND} ${CFG} ${CKPT}"

    echo "Submitting ${JOB_NAME} ..."
    RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod -p dhlab-wxu submit "$JOB_NAME" \
        --image="$IMAGE" \
        --gpu=1 --cpu=4 --memory=24G \
        --pvc=dhlab-scratch:/scratch \
        --pvc=home:/home/wxu \
        --large-shm \
        --command -- bash -c "$INNER_CMD"
done

echo
echo "All 5 conditions submitted (Wijmans MP3D-test protocol). Outputs:"
echo "  ${OUT_DIR}/{blind,coarse,foveated,uniform,foveated_logpolar}.json"
echo
echo "Monitor: kubectl get pods | grep eval5m-"
